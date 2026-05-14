#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_simplified_dn_dshape_first.py

Wrapper around fit_simplified_dn_toposafe.py that adds explicit D-shape
penalties to the existing toposafe score.

It does NOT modify fit_simplified_dn_toposafe.py.

Use case:
    py .\fit_simplified_dn_dshape_first.py --target ... --seed ... [same args as toposafe]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import fit_simplified_dn_toposafe as base


# ---------------------------------------------------------------------
# Environment-backed config.
# This matters on Windows multiprocessing: child processes inherit env vars.
# ---------------------------------------------------------------------

ENV_PREFIX = "STAR_DSHAPE_"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(ENV_PREFIX + name, str(default)))
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(ENV_PREFIX + name, None)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _get_cfg() -> Dict[str, float]:
    return {
        # Overall strength.
        "dshape_weight": _env_float("WEIGHT", 1.0),

        # Inboard side should be close to vertical / low curvature.
        "inboard_weight": _env_float("INBOARD_WEIGHT", 45.0),
        "inboard_sigma_m": _env_float("INBOARD_SIGMA_M", 0.20),

        # Explicit curve-to-target match. Keeps the LCFS from becoming a nice
        # but wrong ellipse.
        "target_rms_weight": _env_float("TARGET_RMS_WEIGHT", 35.0),
        "target_rms_sigma_m": _env_float("TARGET_RMS_SIGMA_M", 0.22),

        # Push effective triangularity upward without letting it dominate alone.
        "delta_weight": _env_float("DELTA_WEIGHT", 22.0),
        "delta_target": _env_float("DELTA_TARGET", 0.48),
        "delta_sigma": _env_float("DELTA_SIGMA", 0.10),

        # Maintain elongation.
        "kappa_weight": _env_float("KAPPA_WEIGHT", 10.0),
        "kappa_min": _env_float("KAPPA_MIN", 2.02),
        "kappa_sigma": _env_float("KAPPA_SIGMA", 0.12),

        # Maintain aspect ratio; not too low.
        "A_weight": _env_float("A_WEIGHT", 8.0),
        "A_min": _env_float("A_MIN", 1.62),
        "A_sigma": _env_float("A_SIGMA", 0.12),

        # Keep wall/leak/chamfer sane.
        "chamfer_weight": _env_float("CHAMFER_WEIGHT", 16.0),
        "chamfer_max_m": _env_float("CHAMFER_MAX_M", 0.34),
        "chamfer_sigma_m": _env_float("CHAMFER_SIGMA_M", 0.08),

        # Avoid axis drifting while chasing D-shape.
        "axis_weight": _env_float("AXIS_WEIGHT", 8.0),
        "R0_target": _env_float("R0_TARGET", 4.0),
        "Raxis_sigma_m": _env_float("RAXIS_SIGMA_M", 0.25),
    }


_TARGET_CACHE: Dict[str, Any] = {}


def _load_target_from_env() -> Optional[Dict[str, Any]]:
    path = os.environ.get(ENV_PREFIX + "TARGET_PATH", "").strip()
    if not path:
        return None

    p = str(Path(path))
    if p in _TARGET_CACHE:
        return _TARGET_CACHE[p]

    try:
        obj = json.loads(Path(p).read_text(encoding="utf-8"))
        _TARGET_CACHE[p] = obj
        return obj
    except Exception:
        return None


# ---------------------------------------------------------------------
# Robust extraction helpers
# ---------------------------------------------------------------------

def _as_xy_array(obj: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(obj, dtype=float)
    except Exception:
        return None

    if arr.ndim != 2 or arr.shape[0] < 8 or arr.shape[1] < 2:
        return None

    arr = arr[:, :2]
    m = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    arr = arr[m]

    if arr.shape[0] < 8:
        return None

    return arr


def _iter_dicts(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_dicts(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_dicts(v)


def _find_named_xy(obj: Any, preferred_names: Sequence[str]) -> Optional[np.ndarray]:
    preferred = [s.lower() for s in preferred_names]

    def rec(o: Any, key_path: str = "") -> Optional[np.ndarray]:
        if isinstance(o, dict):
            # First: preferred named keys.
            for k, v in o.items():
                lk = str(k).lower()
                if any(name in lk for name in preferred):
                    arr = _as_xy_array(v)
                    if arr is not None:
                        return arr

            # Then recurse.
            for k, v in o.items():
                found = rec(v, key_path + "." + str(k))
                if found is not None:
                    return found

        elif isinstance(o, (list, tuple)):
            arr = _as_xy_array(o)
            if arr is not None:
                return arr
            for i, v in enumerate(o):
                found = rec(v, key_path + f"[{i}]")
                if found is not None:
                    return found

        return None

    return rec(obj)


def _extract_target_boundary() -> Optional[np.ndarray]:
    target = _load_target_from_env()
    if not isinstance(target, dict):
        return None

    # Common names used in the target files / plotting logic.
    arr = _find_named_xy(
        target,
        preferred_names=(
            "boundary",
            "plasma_target",
            "target_boundary",
            "simplified",
            "lcfs",
            "xy",
        ),
    )

    if arr is None:
        return None

    return _close_curve(arr)


def _close_curve(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    if xy.shape[0] < 2:
        return xy
    if np.linalg.norm(xy[0] - xy[-1]) > 1.0e-9:
        xy = np.vstack([xy, xy[0]])
    return xy


def _extract_lcfs_from_score_args(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[np.ndarray]:
    # First use base extractor if possible. Try all dict pairs.
    dicts: List[Dict[str, Any]] = []
    for a in args:
        if isinstance(a, dict):
            dicts.append(a)
    for v in kwargs.values():
        if isinstance(v, dict):
            dicts.append(v)

    if hasattr(base, "_extract_lcfs_curve"):
        for shape in dicts:
            for diag in dicts:
                if shape is diag:
                    continue
                try:
                    arr = base._extract_lcfs_curve(shape, diag)
                    arr = _as_xy_array(arr)
                    if arr is not None:
                        return _close_curve(arr)
                except Exception:
                    pass

    # Fallback: search common names in args/kwargs.
    for obj in list(args) + list(kwargs.values()):
        arr = _find_named_xy(
            obj,
            preferred_names=(
                "separatrix_xy",
                "rebuilt_sep",
                "lcfs",
                "boundary",
                "plasma_boundary",
            ),
        )
        if arr is not None:
            return _close_curve(arr)

    return None


def _score_info_from_args(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    # The original score returns score_info. This helper is used only when
    # searching supplemental values inside call args, if needed.
    for obj in list(args) + list(kwargs.values()):
        if isinstance(obj, dict):
            if "score_info" in obj and isinstance(obj["score_info"], dict):
                return obj["score_info"]
            for d in _iter_dicts(obj):
                if "A" in d and ("kappa" in d or "delta_bar" in d):
                    return d
    return {}


# ---------------------------------------------------------------------
# Geometry metrics
# ---------------------------------------------------------------------

def _polar_resample(
    xy: np.ndarray,
    n: int = 181,
    center: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return theta grid and radius r(theta), using max radius per angular bin.

    This is robust for almost star-shaped LCFS curves around magnetic axis/R0.
    """
    xy = np.asarray(xy, dtype=float)
    if center is None:
        center = (float(np.nanmean(xy[:, 0])), float(np.nanmean(xy[:, 1])))

    cx, cz = center
    x = xy[:, 0] - cx
    z = xy[:, 1] - cz
    theta = np.arctan2(z, x)
    r = np.sqrt(x * x + z * z)

    # Sort by theta and interpolate periodically.
    order = np.argsort(theta)
    th = theta[order]
    rr = r[order]

    # Remove duplicate-ish theta by binning.
    grid = np.linspace(-np.pi, np.pi, n)
    rr_grid = np.full_like(grid, np.nan, dtype=float)

    # Nearest-bin max radius; avoids inner duplicated points.
    idx = np.searchsorted(grid, th)
    idx = np.clip(idx, 0, n - 1)
    for i, rad in zip(idx, rr):
        if not np.isfinite(rad):
            continue
        if not np.isfinite(rr_grid[i]) or rad > rr_grid[i]:
            rr_grid[i] = rad

    good = np.isfinite(rr_grid)
    if np.count_nonzero(good) < max(10, n // 8):
        return grid, rr_grid

    # Fill missing by periodic interpolation.
    g_ext = np.r_[grid[good] - 2 * np.pi, grid[good], grid[good] + 2 * np.pi]
    r_ext = np.r_[rr_grid[good], rr_grid[good], rr_grid[good]]
    rr_fill = np.interp(grid, g_ext, r_ext)

    return grid, rr_fill


def _target_rms_term(lcfs: np.ndarray, target: Optional[np.ndarray], cfg: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    if target is None:
        return 0.0, {"dshape_target_rms_m": float("nan"), "dshape_target_term": 0.0}

    lcfs = _close_curve(lcfs)
    target = _close_curve(target)

    R0 = cfg["R0_target"]
    center = (R0, 0.0)

    th_l, r_l = _polar_resample(lcfs, n=241, center=center)
    th_t, r_t = _polar_resample(target, n=241, center=center)

    good = np.isfinite(r_l) & np.isfinite(r_t)
    if np.count_nonzero(good) < 30:
        return 0.0, {"dshape_target_rms_m": float("nan"), "dshape_target_term": 0.0}

    rms = float(np.sqrt(np.mean((r_l[good] - r_t[good]) ** 2)))
    term = cfg["target_rms_weight"] * (rms / max(cfg["target_rms_sigma_m"], 1e-9)) ** 2

    return float(term), {
        "dshape_target_rms_m": rms,
        "dshape_target_term": float(term),
    }


def _inboard_straightness_term(lcfs: np.ndarray, cfg: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """
    Penalize curvature/waviness of the inboard side.

    For D-shape we want the inboard LCFS to be relatively vertical:
    R_in(z) should vary slowly over central |Z| range.
    """
    xy = np.asarray(lcfs, dtype=float)
    R = xy[:, 0]
    Z = xy[:, 1]

    if xy.shape[0] < 20:
        return 0.0, {"dshape_inboard_std_m": float("nan"), "dshape_inboard_term": 0.0}

    R0 = cfg["R0_target"]

    # Inboard half. Use a broad condition so it works even if R0 shifts slightly.
    mask = R < R0
    if np.count_nonzero(mask) < 20:
        return 0.0, {"dshape_inboard_std_m": float("nan"), "dshape_inboard_term": 0.0}

    Rin = R[mask]
    Zin = Z[mask]

    # Exclude X-point nose/ends. Keep central 70% by |Z|.
    zabs = np.abs(Zin)
    zlim = np.nanpercentile(zabs, 75.0)
    central = zabs <= zlim
    Rin = Rin[central]
    Zin = Zin[central]

    if Rin.size < 15:
        return 0.0, {"dshape_inboard_std_m": float("nan"), "dshape_inboard_term": 0.0}

    # Bin in Z and take the minimum R in each bin as the actual inboard edge.
    nb = 18
    bins = np.linspace(float(np.nanmin(Zin)), float(np.nanmax(Zin)), nb + 1)
    vals = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (Zin >= lo) & (Zin < hi)
        if np.count_nonzero(m) >= 2:
            vals.append(float(np.nanmin(Rin[m])))

    if len(vals) < 6:
        return 0.0, {"dshape_inboard_std_m": float("nan"), "dshape_inboard_term": 0.0}

    vals_arr = np.asarray(vals, dtype=float)
    std = float(np.nanstd(vals_arr))

    term = cfg["inboard_weight"] * (std / max(cfg["inboard_sigma_m"], 1e-9)) ** 2

    return float(term), {
        "dshape_inboard_std_m": std,
        "dshape_inboard_term": float(term),
    }


def _scalar_soft_terms(score_info: Dict[str, Any], cfg: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    out: Dict[str, float] = {}
    total = 0.0

    def get_num(*names: str, default: float = float("nan")) -> float:
        for name in names:
            if name in score_info:
                try:
                    v = float(score_info[name])
                    if math.isfinite(v):
                        return v
                except Exception:
                    pass
        return default

    A = get_num("A", "aspect", "aspect_ratio")
    kappa = get_num("kappa", "k")
    delta = get_num("delta_bar", "delta", "triangularity")
    chamfer = get_num("boundary_chamfer_m", "chamfer_m")
    Rax = get_num("Rax", "R_axis", "Rmag", "R_axis_m")

    if math.isfinite(A) and A < cfg["A_min"]:
        t = cfg["A_weight"] * ((cfg["A_min"] - A) / max(cfg["A_sigma"], 1e-9)) ** 2
        total += t
        out["dshape_A_soft_term"] = float(t)
    else:
        out["dshape_A_soft_term"] = 0.0

    if math.isfinite(kappa) and kappa < cfg["kappa_min"]:
        t = cfg["kappa_weight"] * ((cfg["kappa_min"] - kappa) / max(cfg["kappa_sigma"], 1e-9)) ** 2
        total += t
        out["dshape_kappa_soft_term"] = float(t)
    else:
        out["dshape_kappa_soft_term"] = 0.0

    if math.isfinite(delta) and delta < cfg["delta_target"]:
        t = cfg["delta_weight"] * ((cfg["delta_target"] - delta) / max(cfg["delta_sigma"], 1e-9)) ** 2
        total += t
        out["dshape_delta_soft_term"] = float(t)
    else:
        out["dshape_delta_soft_term"] = 0.0

    if math.isfinite(chamfer) and chamfer > cfg["chamfer_max_m"]:
        t = cfg["chamfer_weight"] * ((chamfer - cfg["chamfer_max_m"]) / max(cfg["chamfer_sigma_m"], 1e-9)) ** 2
        total += t
        out["dshape_chamfer_soft_term"] = float(t)
    else:
        out["dshape_chamfer_soft_term"] = 0.0

    if math.isfinite(Rax):
        t = cfg["axis_weight"] * ((Rax - cfg["R0_target"]) / max(cfg["Raxis_sigma_m"], 1e-9)) ** 2
        total += t
        out["dshape_axis_term"] = float(t)
    else:
        out["dshape_axis_term"] = 0.0

    out["dshape_scalar_soft_total"] = float(total)
    return float(total), out


# ---------------------------------------------------------------------
# Monkey patch score function
# ---------------------------------------------------------------------

_ORIGINAL_COMPUTE_TOPOSAFE_SCORE = base.compute_toposafe_score


def compute_toposafe_score_dshape(*args: Any, **kwargs: Any) -> Any:
    """
    Calls original compute_toposafe_score and adds D-shape terms.

    The original function is expected to return either:
        score
    or:
        (score, score_info)

    This wrapper preserves the same return structure.
    """
    original = _ORIGINAL_COMPUTE_TOPOSAFE_SCORE(*args, **kwargs)

    if isinstance(original, tuple) and len(original) >= 2:
        score = original[0]
        score_info = original[1] if isinstance(original[1], dict) else {}
        tail = original[2:]
        tuple_mode = True
    else:
        score = original
        score_info = {}
        tail = ()
        tuple_mode = False

    try:
        score_f = float(score)
    except Exception:
        return original

    if not math.isfinite(score_f):
        return original

    cfg = _get_cfg()
    if cfg["dshape_weight"] <= 0.0:
        return original

    lcfs = _extract_lcfs_from_score_args(args, kwargs)
    target_boundary = _extract_target_boundary()

    dshape_total = 0.0
    dshape_info: Dict[str, float] = {}

    # Existing scalar soft gates.
    scalar_term, scalar_info = _scalar_soft_terms(score_info, cfg)
    dshape_total += scalar_term
    dshape_info.update(scalar_info)

    # Explicit LCFS D-shape terms.
    if lcfs is not None:
        in_term, in_info = _inboard_straightness_term(lcfs, cfg)
        rms_term, rms_info = _target_rms_term(lcfs, target_boundary, cfg)

        dshape_total += in_term + rms_term
        dshape_info.update(in_info)
        dshape_info.update(rms_info)
        dshape_info["dshape_has_lcfs"] = 1.0
    else:
        dshape_info["dshape_has_lcfs"] = 0.0
        dshape_info["dshape_inboard_term"] = 0.0
        dshape_info["dshape_target_term"] = 0.0
        dshape_info["dshape_inboard_std_m"] = float("nan")
        dshape_info["dshape_target_rms_m"] = float("nan")

    dshape_total *= cfg["dshape_weight"]

    new_score = score_f + dshape_total

    if isinstance(score_info, dict):
        score_info.update(dshape_info)
        score_info["dshape_weight"] = float(cfg["dshape_weight"])
        score_info["dshape_extra_term"] = float(dshape_total)
        score_info["score_total_with_dshape"] = float(new_score)

    if tuple_mode:
        return (float(new_score), score_info, *tail)

    return float(new_score)


# Patch at import time so Windows multiprocessing workers also see it when
# they import this script as __mp_main__.
base.compute_toposafe_score = compute_toposafe_score_dshape


# ---------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------

def _extract_target_path_from_argv(argv: Sequence[str]) -> Optional[str]:
    for i, a in enumerate(argv):
        if a == "--target" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--target="):
            return a.split("=", 1)[1]
    return None


def _parse_dshape_args(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)

    ap.add_argument("--dshape-weight", type=float, default=1.0)

    ap.add_argument("--inboard-weight", type=float, default=45.0)
    ap.add_argument("--inboard-sigma-m", type=float, default=0.20)

    ap.add_argument("--target-rms-weight", type=float, default=35.0)
    ap.add_argument("--target-rms-sigma-m", type=float, default=0.22)

    ap.add_argument("--delta-weight", type=float, default=22.0)
    ap.add_argument("--delta-target", type=float, default=0.48)
    ap.add_argument("--delta-sigma", type=float, default=0.10)

    ap.add_argument("--kappa-weight", type=float, default=10.0)
    ap.add_argument("--kappa-min", type=float, default=2.02)
    ap.add_argument("--kappa-sigma", type=float, default=0.12)

    ap.add_argument("--A-weight", type=float, default=8.0)
    ap.add_argument("--A-min", type=float, default=1.62)
    ap.add_argument("--A-sigma", type=float, default=0.12)

    ap.add_argument("--chamfer-weight", type=float, default=16.0)
    ap.add_argument("--chamfer-max-m", type=float, default=0.34)
    ap.add_argument("--chamfer-sigma-m", type=float, default=0.08)

    ap.add_argument("--axis-weight", type=float, default=8.0)
    ap.add_argument("--R0-target", type=float, default=4.0)
    ap.add_argument("--Raxis-sigma-m", type=float, default=0.25)

    ns, remaining = ap.parse_known_args(list(argv))
    return ns, remaining


def _store_dshape_env(ns: argparse.Namespace, target_path: Optional[str]) -> None:
    pairs = {
        "WEIGHT": ns.dshape_weight,

        "INBOARD_WEIGHT": ns.inboard_weight,
        "INBOARD_SIGMA_M": ns.inboard_sigma_m,

        "TARGET_RMS_WEIGHT": ns.target_rms_weight,
        "TARGET_RMS_SIGMA_M": ns.target_rms_sigma_m,

        "DELTA_WEIGHT": ns.delta_weight,
        "DELTA_TARGET": ns.delta_target,
        "DELTA_SIGMA": ns.delta_sigma,

        "KAPPA_WEIGHT": ns.kappa_weight,
        "KAPPA_MIN": ns.kappa_min,
        "KAPPA_SIGMA": ns.kappa_sigma,

        "A_WEIGHT": ns.A_weight,
        "A_MIN": ns.A_min,
        "A_SIGMA": ns.A_sigma,

        "CHAMFER_WEIGHT": ns.chamfer_weight,
        "CHAMFER_MAX_M": ns.chamfer_max_m,
        "CHAMFER_SIGMA_M": ns.chamfer_sigma_m,

        "AXIS_WEIGHT": ns.axis_weight,
        "R0_TARGET": ns.R0_target,
        "RAXIS_SIGMA_M": ns.Raxis_sigma_m,
    }

    for k, v in pairs.items():
        os.environ[ENV_PREFIX + k] = str(v)

    if target_path:
        os.environ[ENV_PREFIX + "TARGET_PATH"] = str(target_path)


def main() -> None:
    original_argv = sys.argv[1:]

    dshape_ns, remaining = _parse_dshape_args(original_argv)
    target_path = _extract_target_path_from_argv(remaining)
    _store_dshape_env(dshape_ns, target_path)

    print("[INFO] fit_simplified_dn_dshape_first wrapper active")
    print("[INFO] Added D-shape terms:")
    print(f"       dshape_weight       = {dshape_ns.dshape_weight}")
    print(f"       inboard_weight      = {dshape_ns.inboard_weight}")
    print(f"       target_rms_weight   = {dshape_ns.target_rms_weight}")
    print(f"       delta_target        = {dshape_ns.delta_target}")
    print(f"       kappa_min           = {dshape_ns.kappa_min}")
    print(f"       A_min               = {dshape_ns.A_min}")
    print(f"       chamfer_max_m       = {dshape_ns.chamfer_max_m}")
    print(f"       target              = {target_path}")

    # Pass only original toposafe-compatible args to base.main().
    sys.argv = [sys.argv[0]] + remaining
    base.main()


if __name__ == "__main__":
    main()
