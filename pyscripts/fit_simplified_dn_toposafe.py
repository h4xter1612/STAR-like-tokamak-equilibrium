#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fit_simplified_dn_toposafe.py

Topology-safe fit against a simplified STAR-like DN target.

This script is intentionally NOT a final CAD/divertor optimizer.

Goal
----
Find whether the CURRENT CAD coil set can move toward a clean STAR-like DN target
without buying fake improvement through grotesque CS current.

Inputs
------
1) Simplified target:
   results/targets/star_simplified_dn_target.json

2) Seed currents:
   results/seeds/baseline_visual_sano_4MA.json

Outputs
-------
results/fit_simplified_dn_toposafe_results.jsonl
results/fit_simplified_dn_toposafe_best.json

Recommended first run
---------------------
py .\\fit_simplified_dn_toposafe.py ^
  --target .\\results\\targets\\star_simplified_dn_target.json ^
  --seed .\\results\\seeds\\baseline_visual_sano_4MA.json ^
  --stage pf-only ^
  --iters 40 ^
  --pop 24 ^
  --workers 4 ^
  --timeout 180 ^
  --dxf .\\cad\\star_baseline.dxf

Notes
-----
- stage=pf-only keeps CS and PF1 fixed, scans PF2..PF6.
- stage=pf1-pf keeps CS fixed, scans PF1..PF6.
- stage=release-cs scans all families but penalizes CS strongly.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


FAMILIES_ALL = ("CS", "CS_MID", "CS_END", "PF1", "PF2", "PF3", "PF4", "PF5", "PF6")


# ---------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------
def _to_builtin(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]
    return obj


def _json_dumps_safe(obj: Any, **kwargs) -> str:
    return json.dumps(_to_builtin(obj), default=str, **kwargs)


def _load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_json_dumps_safe(obj, indent=2))


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(_json_dumps_safe(obj) + "\n")


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
def _here() -> Path:
    return Path(__file__).resolve().parent


def _results_dir() -> Path:
    d = _here() / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_path_maybe(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None

    p = Path(str(path)).expanduser()

    if p.is_absolute():
        return str(p)

    p_cwd = (Path.cwd() / p).resolve()
    if p_cwd.exists():
        return str(p_cwd)

    p_script = (_here() / p).resolve()
    if p_script.exists():
        return str(p_script)

    return str(path)


# ---------------------------------------------------------------------
# Small utils
# ---------------------------------------------------------------------
def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _env_float(name: str, default: float) -> float:
    """Read a float from environment without breaking old workflows."""
    try:
        val = os.environ.get(name, None)
        if val is None or str(val).strip() == "":
            return float(default)
        return float(val)
    except Exception:
        return float(default)


def _currents_A_to_MA(curr_A: Dict[str, float]) -> Dict[str, float]:
    return {k: float(curr_A.get(k, 0.0)) / 1e6 for k in FAMILIES_ALL}


def _currents_MA_to_A(curr_MA: Dict[str, float]) -> Dict[str, float]:
    return {k: float(curr_MA.get(k, 0.0)) * 1e6 for k in FAMILIES_ALL}


def _fmt_currents_MA(curr_A: Dict[str, float], keys: Optional[List[str]] = None) -> str:
    cm = _currents_A_to_MA(curr_A)
    use = keys if keys else list(FAMILIES_ALL)
    return ", ".join(f"{k}={cm.get(k, 0.0):+.3f}" for k in use)


def _parse_keys_csv(s: str) -> List[str]:
    out: List[str] = []
    for tok in str(s).split(","):
        k = tok.strip().upper()
        if k in FAMILIES_ALL and k not in out:
            out.append(k)
    return out


# ---------------------------------------------------------------------
# Seed / target loading
# ---------------------------------------------------------------------
def _load_seed_currents(seed_path: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    j = _load_json(seed_path)

    if isinstance(j.get("currents_A", None), dict):
        currents_A = {k: float(j["currents_A"].get(k, 0.0)) for k in FAMILIES_ALL}
    elif isinstance(j.get("best_currents_A", None), dict):
        currents_A = {k: float(j["best_currents_A"].get(k, 0.0)) for k in FAMILIES_ALL}
    elif isinstance(j.get("currents_MA", None), dict):
        currents_A = {k: float(j["currents_MA"].get(k, 0.0)) * 1e6 for k in FAMILIES_ALL}
    elif isinstance(j.get("best_currents_MA", None), dict):
        currents_A = {k: float(j["best_currents_MA"].get(k, 0.0)) * 1e6 for k in FAMILIES_ALL}
    else:
        raise ValueError(f"Could not find currents_A/currents_MA in seed file: {seed_path}")

    # Seed provides defaults, but ramp/continuation wrappers can override physics
    # through environment variables. This is critical for STAR_IP_A ramp-up.
    Ip_default = _safe_float(j.get("Ip_A", 4.0e6), 4.0e6)
    paxis_default = _safe_float(j.get("paxis_Pa", 2.0e3), 2.0e3)
    fvac_default = _safe_float(j.get("fvac", 20.8), 20.8)
    alpha_m_default = _safe_float(j.get("alpha_m", 1.5), 1.5)
    alpha_n_default = _safe_float(j.get("alpha_n", 1.1), 1.1)

    physics = {
        "Ip_A": _env_float("STAR_IP_A", Ip_default),
        "paxis_Pa": _env_float("STAR_PAXIS_PA", paxis_default),
        "fvac": _env_float("STAR_FVAC", fvac_default),
        "alpha_m": _env_float("STAR_ALPHA_M", alpha_m_default),
        "alpha_n": _env_float("STAR_ALPHA_N", alpha_n_default),
    }

    return currents_A, physics


def _target_boundary(target: Dict[str, Any]) -> np.ndarray:
    b = target.get("boundary", {})
    R = np.asarray(b.get("R", []), float)
    Z = np.asarray(b.get("Z", []), float)
    if R.size < 20 or Z.size < 20 or R.size != Z.size:
        raise ValueError("Target boundary is invalid.")
    P = np.column_stack([R, Z])
    if np.linalg.norm(P[0] - P[-1]) < 1e-12:
        P = P[:-1]
    return P


def _target_xpoints(target: Dict[str, Any]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    xp = target.get("xpoints_target", {})
    upper = xp.get("upper", None)
    lower = xp.get("lower", None)

    up = None
    lo = None

    if isinstance(upper, dict):
        up = (_safe_float(upper.get("R", np.nan)), _safe_float(upper.get("Z", np.nan)))
        if not (np.isfinite(up[0]) and np.isfinite(up[1])):
            up = None

    if isinstance(lower, dict):
        lo = (_safe_float(lower.get("R", np.nan)), _safe_float(lower.get("Z", np.nan)))
        if not (np.isfinite(lo[0]) and np.isfinite(lo[1])):
            lo = None

    return lo, up


def _target_scalars(target: Dict[str, Any]) -> Dict[str, float]:
    s = target.get("scalars_target", {})
    return {
        "R0": _safe_float(s.get("R0", 4.0), 4.0),
        "Z0": _safe_float(s.get("Z0", 0.0), 0.0),
        "A": _safe_float(s.get("A", 2.0), 2.0),
        "kappa": _safe_float(s.get("kappa", 2.23), 2.23),
        "delta_u": _safe_float(s.get("delta_u", 0.62), 0.62),
        "delta_l": _safe_float(s.get("delta_l", 0.62), 0.62),
        "delta_bar": _safe_float(s.get("delta_bar", 0.62), 0.62),
    }


def _target_tolerances(target: Dict[str, Any]) -> Dict[str, float]:
    t = target.get("shape_tolerances", {})
    return {
        "sigma_R0_m": _safe_float(t.get("sigma_R0_m", 0.20), 0.20),
        "sigma_Z0_m": _safe_float(t.get("sigma_Z0_m", 0.15), 0.15),
        "sigma_A": _safe_float(t.get("sigma_A", 0.20), 0.20),
        "sigma_kappa": _safe_float(t.get("sigma_kappa", 0.18), 0.18),
        "sigma_delta": _safe_float(t.get("sigma_delta", 0.12), 0.12),
        "sigma_boundary_chamfer_m": _safe_float(t.get("sigma_boundary_chamfer_m", 0.18), 0.18),
        "sigma_xpoint_m": _safe_float(t.get("sigma_xpoint_m", 0.35), 0.35),
    }


def _target_weights(target: Dict[str, Any]) -> Dict[str, float]:
    w = target.get("objective_weights", {})
    return {
        "topology_fail": _safe_float(w.get("topology_fail", 1.0e8), 1.0e8),
        "not_double_null": _safe_float(w.get("not_double_null", 2.0e5), 2.0e5),
        "axis": _safe_float(w.get("axis", 30.0), 30.0),
        "shape_scalars": _safe_float(w.get("shape_scalars", 20.0), 20.0),
        "boundary_chamfer": _safe_float(w.get("boundary_chamfer", 15.0), 15.0),
        "xpoints": _safe_float(w.get("xpoints", 10.0), 10.0),
        "vertical_symmetry": _safe_float(w.get("vertical_symmetry", 8.0), 8.0),
        "current_regularization": _safe_float(w.get("current_regularization", 3.0), 3.0),
        "cs_regularization_extra": _safe_float(w.get("cs_regularization_extra", 12.0), 12.0),
        "current_step_regularization": _safe_float(w.get("current_step_regularization", 1.5), 1.5),
    }


def _target_current_limits(target: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    cr = target.get("coil_regularization", {})
    imax = cr.get("imax_recommended_A", {})
    operating = cr.get("operating_limit_A", {})

    default_imax = {
        "CS": 73.036e6,
        "CS_MID": 73.036e6,
        "CS_END": 73.036e6,
        "PF1": 15.000e6,
        "PF2": 15.000e6,
        "PF3": 7.500e6,
        "PF4": 9.437e6,
        "PF5": 11.850e6,
        "PF6": 19.752e6,
    }

    imax_A = {k: float(imax.get(k, default_imax[k])) for k in FAMILIES_ALL}

    # STAR segmented-CS convention:
    # CS_MID and CS_END are control subfamilies of the central OH/CS winding pack.
    # If the target JSON contains small local values for CS_MID/CS_END from an
    # earlier discretization, do not let those artificially clip ramp-up scans.
    # Use at least the parent CS recommended value for the segmented CS controls.
    if "CS" in imax_A:
        imax_A["CS_MID"] = max(float(imax_A.get("CS_MID", 0.0)), float(imax_A["CS"]))
        imax_A["CS_END"] = max(float(imax_A.get("CS_END", 0.0)), float(imax_A["CS"]))

    operating_A = {k: float(operating.get(k, 0.35 * imax_A[k])) for k in FAMILIES_ALL}

    # Keep operating limits consistent with the same segmented-CS convention.
    if "CS" in operating_A:
        operating_A["CS_MID"] = max(float(operating_A.get("CS_MID", 0.0)), float(operating_A["CS"]))
        operating_A["CS_END"] = max(float(operating_A.get("CS_END", 0.0)), float(operating_A["CS"]))

    return imax_A, operating_A


# ---------------------------------------------------------------------
# Curve / X-point extraction
# ---------------------------------------------------------------------
def _open_curve_from_closed(R: Any, Z: Any) -> Optional[np.ndarray]:
    try:
        R = np.asarray(R, float)
        Z = np.asarray(Z, float)
        if R.size < 20 or Z.size < 20 or R.size != Z.size:
            return None
        P = np.column_stack([R, Z])
        if np.linalg.norm(P[0] - P[-1]) < 1e-12:
            P = P[:-1]
        if P.shape[0] < 20:
            return None
        return P
    except Exception:
        return None


def _extract_lcfs_curve(shape: Dict[str, Any], diag: Dict[str, Any]) -> Optional[np.ndarray]:
    key_pairs = [
        ("R_sep", "Z_sep"),
        ("R_separatrix", "Z_separatrix"),
        ("R_lcfs", "Z_lcfs"),
        ("lcfs_R", "lcfs_Z"),
        ("R_LCFS", "Z_LCFS"),
    ]

    for rk, zk in key_pairs:
        if rk in shape and zk in shape:
            P = _open_curve_from_closed(shape.get(rk), shape.get(zk))
            if P is not None:
                return P

    if isinstance(diag, dict):
        P = _open_curve_from_closed(diag.get("R_lcfs", None), diag.get("Z_lcfs", None))
        if P is not None:
            return P

    fb = shape.get("fallback_lcfs", None)
    if isinstance(fb, dict):
        P = _open_curve_from_closed(fb.get("R", None), fb.get("Z", None))
        if P is not None:
            return P

    return None


def _extract_xpoints(shape: Dict[str, Any]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []

    xpv = shape.get("xpoints_valid", None)
    if isinstance(xpv, list) and xpv:
        src = xpv
    else:
        src = shape.get("xpoints", None) or shape.get("xpoint", None) or []

    if isinstance(src, dict):
        src = list(src.values())

    if not isinstance(src, list):
        return out

    for item in src:
        if isinstance(item, dict):
            R = _safe_float(item.get("R", item.get("r", np.nan)))
            Z = _safe_float(item.get("Z", item.get("z", np.nan)))
            if np.isfinite(R) and np.isfinite(Z):
                out.append((R, Z))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            R = _safe_float(item[0], np.nan)
            Z = _safe_float(item[1], np.nan)
            if np.isfinite(R) and np.isfinite(Z):
                out.append((R, Z))

    return out


def _pick_lower_upper(points: List[Tuple[float, float]]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    if not points:
        return None, None
    lower = min(points, key=lambda p: p[1])
    upper = max(points, key=lambda p: p[1])
    return lower, upper


def _rms_chamfer_sym(P: np.ndarray, Q: np.ndarray) -> float:
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)

    if P.ndim != 2 or Q.ndim != 2 or P.shape[1] != 2 or Q.shape[1] != 2:
        return float("inf")
    if P.shape[0] < 20 or Q.shape[0] < 20:
        return float("inf")

    # Downsample to keep O(N*M) cheap.
    maxn = 240
    if P.shape[0] > maxn:
        P = P[np.linspace(0, P.shape[0] - 1, maxn).astype(int)]
    if Q.shape[0] > maxn:
        Q = Q[np.linspace(0, Q.shape[0] - 1, maxn).astype(int)]

    d2_pq = []
    for p in P:
        d2_pq.append(float(np.min(np.sum((Q - p) ** 2, axis=1))))

    d2_qp = []
    for q in Q:
        d2_qp.append(float(np.min(np.sum((P - q) ** 2, axis=1))))

    return float(math.sqrt(0.5 * (float(np.mean(d2_pq)) + float(np.mean(d2_qp)))))


def _point_dist(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]) -> float:
    if a is None or b is None:
        return float("inf")
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


# ---------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------
def compute_toposafe_score(
    *,
    target: Dict[str, Any],
    shape: Dict[str, Any],
    diag: Dict[str, Any],
    currents_A: Dict[str, float],
    ref_currents_A: Dict[str, float],
    seed_currents_A: Dict[str, float],
    free_keys: List[str],
    stage: str,
) -> Tuple[float, Dict[str, Any]]:
    scal_t = _target_scalars(target)
    tol = _target_tolerances(target)
    w = _target_weights(target)
    imax_A, operating_A = _target_current_limits(target)

    target_curve = _target_boundary(target)
    lower_t, upper_t = _target_xpoints(target)

    reason = str(shape.get("reason", "")).strip().lower()
    has_true_sep = bool(shape.get("ok_sep", False)) and (reason == "ok")

    ok_diag = bool(isinstance(diag, dict) and diag.get("ok", False))

    Rax = _safe_float(diag.get("R_ax", shape.get("R_ax", np.nan)))
    Zax = _safe_float(diag.get("Z_ax", shape.get("Z_ax", np.nan)))

    R0 = _safe_float(diag.get("R0", np.nan))
    A = _safe_float(diag.get("A", np.nan))
    kap = _safe_float(diag.get("kappa", np.nan))
    du = _safe_float(diag.get("delta_u", np.nan))
    dl = _safe_float(diag.get("delta_l", np.nan))
    area = _safe_float(diag.get("area_m2", diag.get("area", np.nan)), np.nan)

    inside_wall = bool(diag.get("inside_WALL_INNER", False))
    outside_frac = _safe_float(diag.get("outside_frac", 1.0), 1.0)
    signed_gap = _safe_float(diag.get("signed_gap_to_WALL_INNER_m", np.nan), np.nan)
    wall_gap_target_m = _env_float("STAR_WALL_GAP_TARGET_M", 0.02)

    wall_term = 0.0

    if not inside_wall:
        wall_term += 5.0e5

    if np.isfinite(outside_frac) and outside_frac > 0.0:
        wall_term += 1.0e7 * outside_frac
    else:
        if not np.isfinite(outside_frac):
            wall_term += 5.0e5

    if not np.isfinite(signed_gap):
        wall_term += 1.0e6
    elif signed_gap < 0.0:
        wall_term += 2.0e7 * abs(signed_gap)
    elif signed_gap < wall_gap_target_m:
        wall_term += 5.0e4 * ((wall_gap_target_m - signed_gap) / max(wall_gap_target_m, 1e-9)) ** 2
    dbar = 0.5 * (du + dl) if np.isfinite(du) and np.isfinite(dl) else float("nan")


    branch_Rax_min = _env_float("STAR_BRANCH_RAX_MIN", 3.5)
    branch_Rax_max = _env_float("STAR_BRANCH_RAX_MAX", 5.0)
    branch_Zax_max = _env_float("STAR_BRANCH_ZAX_MAX", 0.8)
    branch_area_min = _env_float("STAR_BRANCH_AREA_MIN", 18.0)
    branch_area_max = _env_float("STAR_BRANCH_AREA_MAX", 45.0)
    branch_weight = _env_float("STAR_BRANCH_TERM_WEIGHT", 1.0e6)

    branch_term = 0.0

    if not np.isfinite(Rax) or not (branch_Rax_min <= Rax <= branch_Rax_max):
        branch_term += branch_weight

    if not np.isfinite(Zax) or abs(Zax) > branch_Zax_max:
        branch_term += branch_weight

    if not np.isfinite(area) or not (branch_area_min <= area <= branch_area_max):
        branch_term += branch_weight

    for name, val in [
        ("R0", R0),
        ("A", A),
        ("kappa", kap),
        ("delta_bar", dbar),
    ]:
        if not np.isfinite(val):
            branch_term += branch_weight


    xps = _extract_xpoints(shape)
    lower_xp, upper_xp = _pick_lower_upper(xps)
    n_xp = int(len(xps))

    lcfs = _extract_lcfs_curve(shape, diag)
    chamfer = _rms_chamfer_sym(lcfs, target_curve) if lcfs is not None else float("inf")

    score = 0.0
    fail_reasons: List[str] = []

    # Topology gate.
    if not has_true_sep:
        score += float(w["topology_fail"])
        fail_reasons.append("no_true_separatrix")

    # Diagnostic gate.
    if not ok_diag:
        score += 0.35 * float(w["topology_fail"])
        fail_reasons.append("no_valid_plasma_diag")

    score += wall_term

    score += branch_term

    if branch_term > 0.0:
        fail_reasons.append("branch_gate_failed")

    if not inside_wall:
        fail_reasons.append("outside_WALL_INNER")
    if np.isfinite(outside_frac) and outside_frac > 0.0:
        fail_reasons.append("outside_frac_nonzero")
    if np.isfinite(signed_gap) and signed_gap < wall_gap_target_m:
        fail_reasons.append("wall_gap_below_target")

    # Double-null preference.
    if n_xp < 2:
        score += float(w["not_double_null"])
        fail_reasons.append("less_than_two_xpoints")

    # Axis term.
    axis_term = 0.0
    if np.isfinite(Rax):
        axis_term += ((Rax - scal_t["R0"]) / max(tol["sigma_R0_m"], 1e-12)) ** 2
    else:
        axis_term += 1e4

    if np.isfinite(Zax):
        axis_term += ((Zax - scal_t["Z0"]) / max(tol["sigma_Z0_m"], 1e-12)) ** 2
    else:
        axis_term += 1e4

    score += float(w["axis"]) * axis_term

    # Shape scalar term.
    shape_term = 0.0

    if np.isfinite(R0):
        shape_term += ((R0 - scal_t["R0"]) / max(tol["sigma_R0_m"], 1e-12)) ** 2
    else:
        shape_term += 1e4

    if np.isfinite(A):
        shape_term += ((A - scal_t["A"]) / max(tol["sigma_A"], 1e-12)) ** 2
    else:
        shape_term += 1e4

    if np.isfinite(kap):
        shape_term += ((kap - scal_t["kappa"]) / max(tol["sigma_kappa"], 1e-12)) ** 2
    else:
        shape_term += 1e4

    if np.isfinite(dbar):
        shape_term += ((dbar - scal_t["delta_bar"]) / max(tol["sigma_delta"], 1e-12)) ** 2
    else:
        shape_term += 1e4

    shape_weight_scale = _env_float("STAR_SHAPE_WEIGHT_SCALE", 0.35)
    score += shape_weight_scale * float(w["shape_scalars"]) * shape_term

    # Boundary term.
    if np.isfinite(chamfer):
        boundary_term = (chamfer / max(tol["sigma_boundary_chamfer_m"], 1e-12)) ** 2
    else:
        boundary_term = 1e4
        fail_reasons.append("no_lcfs_curve_for_boundary_fit")

    boundary_weight_scale = _env_float("STAR_BOUNDARY_WEIGHT_SCALE", 0.25)
    score += boundary_weight_scale * float(w["boundary_chamfer"]) * boundary_term

    # X-point term.
    dx_lower = _point_dist(lower_xp, lower_t)
    dx_upper = _point_dist(upper_xp, upper_t)

    x_term = 0.0
    if np.isfinite(dx_lower):
        x_term += (dx_lower / max(tol["sigma_xpoint_m"], 1e-12)) ** 2
    else:
        x_term += 1e4

    if np.isfinite(dx_upper):
        x_term += (dx_upper / max(tol["sigma_xpoint_m"], 1e-12)) ** 2
    else:
        x_term += 1e4

    xpoint_weight_scale = _env_float("STAR_XPOINT_WEIGHT_SCALE", 0.15)
    score += xpoint_weight_scale * float(w["xpoints"]) * x_term

    # Vertical symmetry term.
    sym_term = 0.0
    if np.isfinite(du) and np.isfinite(dl):
        sym_term += ((du - dl) / max(0.10, 1e-12)) ** 2
    else:
        sym_term += 1e3

    if lower_xp is not None and upper_xp is not None:
        sym_term += ((lower_xp[0] - upper_xp[0]) / 0.35) ** 2
        sym_term += ((lower_xp[1] + upper_xp[1]) / 0.50) ** 2
    else:
        sym_term += 1e3

    sym_weight_scale = _env_float("STAR_SYM_WEIGHT_SCALE", 0.25)
    score += sym_weight_scale * float(w["vertical_symmetry"]) * sym_term

    # Current regularization.
    reg_term = 0.0
    cs_reg_term = 0.0
    step_reg_term = 0.0
    hard_violation = False

    for k in FAMILIES_ALL:
        I = abs(float(currents_A.get(k, 0.0)))
        imax = max(float(imax_A.get(k, 1.0)), 1.0)
        op = max(float(operating_A.get(k, 0.35 * imax)), 1.0)

        # Hard fail if above 80% of recommended Imax.
        if I > 0.80 * imax:
            hard_violation = True

        # Soft operating regularization.
        reg_term += (I / op) ** 2

        # Step regularization relative to previous best/ref.
        # Let segmented CS move more freely during ramp-up.
        if k in free_keys:
            dI = float(currents_A.get(k, 0.0)) - float(ref_currents_A.get(k, 0.0))

            if k in ("CS_MID", "CS_END"):
                denom = max(0.60 * op, 1.0)
            else:
                denom = max(0.25 * op, 1.0)

            step_reg_term += (dI / denom) ** 2

    # Extra CS-specific regularization.
    Ics = abs(float(currents_A.get("CS", 0.0)))
    cs_soft = 0.25 * max(float(imax_A.get("CS", 73.036e6)), 1.0)
    cs_reg_term = (Ics / max(cs_soft, 1.0)) ** 2

    # In pf-only/pf1-pf, CS should remain exactly fixed. Add huge penalty if changed.
    if stage in ("pf-only", "pf1-pf"):
        dcs_seed = abs(float(currents_A.get("CS", 0.0)) - float(seed_currents_A.get("CS", 0.0)))
        if dcs_seed > 1e-6:
            score += 1e7 * (dcs_seed / 1e6) ** 2
            fail_reasons.append("cs_changed_in_fixed_cs_stage")

    if hard_violation:
        score += 5.0e7
        fail_reasons.append("current_above_0p8_imax")

    current_reg_scale = _env_float("STAR_CURRENT_REG_SCALE", 0.20)
    cs_reg_scale = _env_float("STAR_CS_REG_SCALE", 0.10)
    step_reg_scale = _env_float("STAR_STEP_REG_SCALE", 0.15)

    score += current_reg_scale * float(w["current_regularization"]) * reg_term
    score += cs_reg_scale * float(w["cs_regularization_extra"]) * cs_reg_term
    score += step_reg_scale * float(w["current_step_regularization"]) * step_reg_term

    info = {
        "ok": bool(
            has_true_sep
            and ok_diag
            and inside_wall
            and np.isfinite(signed_gap)
            and signed_gap > 0.0
            and np.isfinite(outside_frac)
            and outside_frac == 0.0
            and branch_term == 0.0
        ),
        "has_true_sep": bool(has_true_sep),
        "shape_reason": reason,
        "n_xpoints": n_xp,
        "fail_reasons": fail_reasons,

        "score_total": float(score),
        "axis_term": float(axis_term),
        "shape_term": float(shape_term),
        "boundary_term": float(boundary_term),
        "x_term": float(x_term),
        "sym_term": float(sym_term),
        "reg_term": float(reg_term),
        "cs_reg_term": float(cs_reg_term),
        "step_reg_term": float(step_reg_term),

        "Rax": Rax,
        "Zax": Zax,
        "R0": R0,
        "A": A,
        "kappa": kap,
        "delta_u": du,
        "delta_l": dl,
        "delta_bar": dbar,
        "boundary_chamfer_m": chamfer,

        "lower_xpoint": lower_xp,
        "upper_xpoint": upper_xp,
        "dx_lower_m": dx_lower,
        "dx_upper_m": dx_upper,

        "inside_WALL_INNER": bool(inside_wall),
        "outside_frac": float(outside_frac),
        "signed_gap_to_WALL_INNER_m": float(signed_gap),
        "wall_gap_target_m": float(wall_gap_target_m),
        "wall_term": float(wall_term),

        "area": area,
        "branch_term": float(branch_term),
        "branch_Rax_min": float(branch_Rax_min),
        "branch_Rax_max": float(branch_Rax_max),
        "branch_Zax_max": float(branch_Zax_max),
        "branch_area_min": float(branch_area_min),
        "branch_area_max": float(branch_area_max),
    }

    return float(score), info


# ---------------------------------------------------------------------
# Subprocess worker
# ---------------------------------------------------------------------
def _worker_eval(
    currents_A: Dict[str, float],
    target: Dict[str, Any],
    dxf_path: Optional[str],
    cfg_overrides: Dict[str, Any],
    physics: Dict[str, float],
    ref_currents_A: Dict[str, float],
    seed_currents_A: Dict[str, float],
    free_keys: List[str],
    stage: str,
    redirect_solver_noise: bool,
    conn,
) -> None:
    score = 1e18
    try:
        import config_star_bean as cfg
        import star_equilibrium as se

        # Apply config overrides.
        for k, v in (cfg_overrides or {}).items():
            try:
                setattr(cfg, k, v)
            except Exception:
                pass

        # Apply physics.
        cfg.Ip = float(physics.get("Ip_A", 4.0e6))
        cfg.paxis = float(physics.get("paxis_Pa", 2.0e3))
        cfg.fvac = float(physics.get("fvac", 20.8))
        cfg.alpha_m = float(physics.get("alpha_m", 1.5))
        cfg.alpha_n = float(physics.get("alpha_n", 1.1))
        cfg.vacuum_only = False

        # Apply currents into cfg.
        for k, v in currents_A.items():
            try:
                setattr(cfg, f"{k}_current", float(v))
            except Exception:
                pass

        eq, tokamak, geom, shape = se.build_equilibrium(
            verbose=False,
            redirect_solver_noise=bool(redirect_solver_noise),
            dxf_path=dxf_path,
        )

        diag = shape.get("plasma_diag", None) or {}
        if (not diag) or (not diag.get("ok", False)):
            try:
                diag = se.plasma_diagnostics(eq, geom, shape)
            except Exception:
                diag = {"ok": False, "reason": "plasma_diagnostics_failed"}

        score, score_info = compute_toposafe_score(
            target=target,
            shape=shape,
            diag=diag,
            currents_A=currents_A,
            ref_currents_A=ref_currents_A,
            seed_currents_A=seed_currents_A,
            free_keys=free_keys,
            stage=stage,
        )

        conn.send({
            "ok": True,
            "ok_solve": True,
            "score": float(score),
            "score_info": score_info,
            "currents_A": dict(currents_A),
            "currents_MA": _currents_A_to_MA(currents_A),
            "physics": dict(physics),
            "diag": diag,
            "shape": {
                "ok_sep": bool(shape.get("ok_sep", False)),
                "reason": shape.get("reason", None),
                "xpoints": shape.get("xpoints", []),
                "xpoints_valid": shape.get("xpoints_valid", []),
                "psi_sep": shape.get("psi_sep", None),
                "psi_eval": shape.get("psi_eval", None),
            },
        })
        conn.close()

    except Exception as e:
        try:
            conn.send({
                "ok": False,
                "ok_solve": False,
                "score": float(score),
                "error": repr(e),
                "currents_A": dict(currents_A),
                "currents_MA": _currents_A_to_MA(currents_A),
            })
            conn.close()
        except Exception:
            pass


def eval_case(
    currents_A: Dict[str, float],
    *,
    target: Dict[str, Any],
    dxf_path: Optional[str],
    cfg_overrides: Dict[str, Any],
    physics: Dict[str, float],
    ref_currents_A: Dict[str, float],
    seed_currents_A: Dict[str, float],
    free_keys: List[str],
    stage: str,
    timeout_s: float,
    redirect_solver_noise: bool,
) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)

    p = ctx.Process(
        target=_worker_eval,
        args=(
            dict(currents_A),
            dict(target),
            dxf_path,
            dict(cfg_overrides or {}),
            dict(physics),
            dict(ref_currents_A),
            dict(seed_currents_A),
            list(free_keys),
            str(stage),
            bool(redirect_solver_noise),
            send_conn,
        ),
    )

    p.start()
    send_conn.close()

    if recv_conn.poll(float(timeout_s)):
        try:
            res = recv_conn.recv()
        except EOFError:
            res = {
                "ok": False,
                "ok_solve": False,
                "score": 1e18,
                "error": "EOFError receiving worker result",
            }
    else:
        try:
            p.terminate()
        except Exception:
            pass
        res = {
            "ok": False,
            "ok_solve": False,
            "score": 1e18,
            "error": f"timeout>{timeout_s:.1f}s",
            "currents_A": dict(currents_A),
            "currents_MA": _currents_A_to_MA(currents_A),
        }

    p.join(timeout=3.0)
    if p.is_alive():
        try:
            p.kill()
        except Exception:
            pass

    try:
        recv_conn.close()
    except Exception:
        pass

    return res


# ---------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------
def _stage_free_fixed(stage: str) -> Tuple[List[str], List[str]]:
    stage = str(stage).strip().lower()

    if stage == "pf-only":
        free = ["PF2", "PF3", "PF4", "PF5", "PF6"]
        fixed = ["CS", "CS_MID", "CS_END", "PF1"]

    elif stage == "pf1-pf":
        free = ["PF1", "PF2", "PF3", "PF4", "PF5", "PF6"]
        fixed = ["CS", "CS_MID", "CS_END"]

    elif stage == "release-cs":
        # Legacy mode: release parent CS.
        # If you pass --free-keys manually, this will be overridden.
        free = ["CS", "PF1", "PF2", "PF3", "PF4", "PF5", "PF6"]
        fixed = ["CS_MID", "CS_END"]

    else:
        raise ValueError(f"Unknown stage: {stage}")

    return free, fixed


def _make_bounds_MA(
    center_A: Dict[str, float],
    target: Dict[str, Any],
    free_keys: List[str],
    *,
    stage: str,
    widen: float,
) -> Dict[str, Tuple[float, float]]:
    imax_A, operating_A = _target_current_limits(target)
    center_MA = _currents_A_to_MA(center_A)

    # Conservative local spans. This is not brute force.
    base_span = {
        "CS": 2.5,
        "CS_MID": 6.0,
        "CS_END": 5.0,
        "PF1": 1.2,
        "PF2": 2.5,
        "PF3": 1.5,
        "PF4": 2.0,
        "PF5": 2.5,
        "PF6": 3.0,
    }

    bounds: Dict[str, Tuple[float, float]] = {}

    for k in free_keys:
        c = float(center_MA.get(k, 0.0))
        span = float(base_span.get(k, 1.0)) + float(widen) * max(abs(c), 0.50)

        # Hard limit: 80% of recommended.
        hard_MA = 0.80 * float(imax_A[k]) / 1e6

        # Extra cautious for CS in release stage.
        if k == "CS":
            if stage == "release-cs":
                hard_MA = min(hard_MA, 0.80 * float(imax_A[k]) / 1e6)
                span = min(span, 4.0)
            else:
                hard_MA = abs(c)

        lo = max(-hard_MA, c - span)
        hi = min(+hard_MA, c + span)

        if hi < lo:
            lo, hi = hi, lo

        bounds[k] = (float(lo), float(hi))

    return bounds


def _sigma_from_bounds_MA(bounds_MA: Dict[str, Tuple[float, float]], frac: float) -> Dict[str, float]:
    return {k: float(frac) * (float(hi) - float(lo)) for k, (lo, hi) in bounds_MA.items()}


def _clip_to_bounds_A(curr_A: Dict[str, float], bounds_MA: Dict[str, Tuple[float, float]], free_keys: List[str]) -> Dict[str, float]:
    out = dict(curr_A)
    for k in free_keys:
        lo, hi = bounds_MA[k]
        out[k] = float(np.clip(out.get(k, 0.0) / 1e6, lo, hi)) * 1e6
    return out


def _sample_uniform_A(
    base_A: Dict[str, float],
    bounds_MA: Dict[str, Tuple[float, float]],
    free_keys: List[str],
) -> Dict[str, float]:
    out = dict(base_A)
    for k in free_keys:
        lo, hi = bounds_MA[k]
        out[k] = random.uniform(lo, hi) * 1e6
    return out


def _sample_gauss_A(
    center_A: Dict[str, float],
    sigma_MA: Dict[str, float],
    bounds_MA: Dict[str, Tuple[float, float]],
    free_keys: List[str],
) -> Dict[str, float]:
    out = dict(center_A)
    center_MA = _currents_A_to_MA(center_A)
    for k in free_keys:
        lo, hi = bounds_MA[k]
        s = float(sigma_MA.get(k, 0.5))
        val = random.gauss(float(center_MA.get(k, 0.0)), s)
        out[k] = float(np.clip(val, lo, hi)) * 1e6
    return out


# ---------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------
def fit_toposafe(
    *,
    target_path: str,
    seed_path: str,
    dxf_path: Optional[str],
    stage: str,
    workers: int,
    timeout_s: float,
    iters: int,
    pop: int,
    seed: int,
    widen: float,
    sigma_frac: float,
    no_coarse: bool,
    show_solver: bool,
    free_keys_override: Optional[List[str]],
    no_early_stop: bool,
) -> Dict[str, Any]:
    random.seed(int(seed))
    np.random.seed(int(seed))

    target = _load_json(target_path)
    seed_currents_A, physics = _load_seed_currents(seed_path)

    # Final physics override from environment. This protects ramp-up wrappers even
    # when the seed JSON still contains the baseline 4.0 MA value.
    physics["Ip_A"] = _env_float("STAR_IP_A", physics.get("Ip_A", 4.0e6))
    physics["paxis_Pa"] = _env_float("STAR_PAXIS_PA", physics.get("paxis_Pa", 2.0e3))
    physics["fvac"] = _env_float("STAR_FVAC", physics.get("fvac", 20.8))
    physics["alpha_m"] = _env_float("STAR_ALPHA_M", physics.get("alpha_m", 1.5))
    physics["alpha_n"] = _env_float("STAR_ALPHA_N", physics.get("alpha_n", 1.1))

    stage = str(stage).strip().lower()
    free_keys, fixed_keys = _stage_free_fixed(stage)

    if free_keys_override:
        free_keys = [k for k in free_keys_override if k in FAMILIES_ALL]
        fixed_keys = [k for k in FAMILIES_ALL if k not in free_keys]

    if not free_keys:
        raise ValueError("free_keys is empty.")

    cfg_overrides: Dict[str, Any] = {}
    if not no_coarse:
        cfg_overrides.update({
            "nx_eq": 65,
            "ny_eq": 129,
            "blanket_enabled": False,
            "blanket_n_filaments": 0,
            "target_rel_tol_ramp": 1.0e-4,
            "target_rel_tol": 5.0e-5,
            "f_list_equilibrium": (0.10, 0.25, 0.45, 0.70, 1.00),
        })

    dxf_path = _resolve_path_maybe(dxf_path)

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    log_jsonl = _results_dir() / "fit_simplified_dn_toposafe_results.jsonl"
    best_json = _results_dir() / "fit_simplified_dn_toposafe_best.json"
    run_json = _results_dir() / f"fit_simplified_dn_toposafe_run_{run_tag}.json"

    best: Optional[Dict[str, Any]] = None
    best_score = float("inf")
    best_currents_A = dict(seed_currents_A)
    ref_currents_A = dict(seed_currents_A)

    print("[INFO] fit_simplified_dn_toposafe started")
    print(f"[INFO] target={Path(target_path).resolve()}")
    print(f"[INFO] seed={Path(seed_path).resolve()}")
    print(f"[INFO] dxf={dxf_path or 'cfg/auto'}")
    print(f"[INFO] stage={stage}")
    print(f"[INFO] free_keys={free_keys}")
    print(f"[INFO] fixed_keys={fixed_keys}")
    print(f"[INFO] workers={workers} iters={iters} pop={pop} timeout={timeout_s:.1f}s seed={seed}")
    print(f"[INFO] physics={physics}")
    print(f"[INFO] initial currents [MA]: {_fmt_currents_MA(seed_currents_A)}")

    # Evaluate seed first.
    print("\n[SEED] evaluating baseline...")
    seed_res = eval_case(
        seed_currents_A,
        target=target,
        dxf_path=dxf_path,
        cfg_overrides=cfg_overrides,
        physics=physics,
        ref_currents_A=ref_currents_A,
        seed_currents_A=seed_currents_A,
        free_keys=free_keys,
        stage=stage,
        timeout_s=timeout_s,
        redirect_solver_noise=(not show_solver),
    )

    _append_jsonl(log_jsonl, {
        "kind": "seed",
        "stage": stage,
        "result": seed_res,
    })

    if float(seed_res.get("score", 1e18)) < best_score:
        best = seed_res
        best_score = float(seed_res.get("score", 1e18))
        best_currents_A = dict(seed_res.get("currents_A", seed_currents_A))
        ref_currents_A = dict(best_currents_A)

    si = seed_res.get("score_info", {}) if isinstance(seed_res.get("score_info", {}), dict) else {}
    print(
        f"[SEED] score={float(seed_res.get('score', 1e18)):.6g} "
        f"sep={si.get('has_true_sep', False)} "
        f"nxp={si.get('n_xpoints', 0)} "
        f"A={_safe_float(si.get('A', np.nan)):.3f} "
        f"k={_safe_float(si.get('kappa', np.nan)):.3f} "
        f"d={_safe_float(si.get('delta_bar', np.nan)):.3f} "
        f"CS={_currents_A_to_MA(best_currents_A).get('CS', 0.0):+.3f} MA"
    )

    for it in range(1, int(iters) + 1):
        bounds_MA = _make_bounds_MA(
            best_currents_A,
            target,
            free_keys,
            stage=stage,
            widen=widen,
        )
        sigma_MA = _sigma_from_bounds_MA(bounds_MA, sigma_frac)

        # Always include the current best.
        batch: List[Dict[str, float]] = []

        # Always include the current best.
        batch.append(dict(best_currents_A))

        # Directed segmented-CS pushes for ramp-up.
        # Sign convention in current STAR runs: useful CS_MID/CS_END are negative.
        for dmid_MA, dend_MA in [
            (-0.5,  0.0),
            (-1.0,  0.0),
            (-1.5,  0.0),
            ( 0.0, -0.5),
            ( 0.0, -1.0),
            (-0.5, -0.5),
            (-1.0, -0.5),
            (-1.0, -1.0),
            (-1.5, -0.5),
            (-1.5, -1.0),
        ]:
            cand = dict(best_currents_A)

            if "CS_MID" in free_keys:
                cand["CS_MID"] = float(cand.get("CS_MID", 0.0)) + dmid_MA * 1e6

            if "CS_END" in free_keys:
                cand["CS_END"] = float(cand.get("CS_END", 0.0)) + dend_MA * 1e6

            batch.append(cand)

        n_global = max(2, int(pop // 5))
        n_local = max(0, int(pop) - n_global - len(batch))

        for _ in range(n_global):
            batch.append(_sample_uniform_A(best_currents_A, bounds_MA, free_keys))

        for _ in range(n_local):
            batch.append(_sample_gauss_A(best_currents_A, sigma_MA, bounds_MA, free_keys))

        batch = batch[: int(pop)]

        batch_best: Optional[Dict[str, Any]] = None
        batch_best_score = float("inf")
        ok_solve = 0
        ok_sep = 0

        t0 = time.time()

        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
            futs = {}
            for curr in batch:
                curr2 = _clip_to_bounds_A(curr, bounds_MA, free_keys)
                fut = ex.submit(
                    eval_case,
                    curr2,
                    target=target,
                    dxf_path=dxf_path,
                    cfg_overrides=cfg_overrides,
                    physics=physics,
                    ref_currents_A=ref_currents_A,
                    seed_currents_A=seed_currents_A,
                    free_keys=free_keys,
                    stage=stage,
                    timeout_s=timeout_s,
                    redirect_solver_noise=(not show_solver),
                )
                futs[fut] = curr2

            for fut in as_completed(futs):
                res = fut.result()
                sc = float(res.get("score", 1e18))

                if bool(res.get("ok_solve", False)):
                    ok_solve += 1

                si = res.get("score_info", {})
                if isinstance(si, dict) and bool(si.get("has_true_sep", False)):
                    ok_sep += 1

                if sc < batch_best_score:
                    batch_best_score = sc
                    batch_best = res

                _append_jsonl(log_jsonl, {
                    "kind": "eval",
                    "iteration": int(it),
                    "stage": stage,
                    "result": res,
                })

        improved = False
        if batch_best is not None and batch_best_score < best_score:
            improved = True
            best = batch_best
            best_score = float(batch_best_score)
            best_currents_A = dict(batch_best.get("currents_A", best_currents_A))
            ref_currents_A = dict(best_currents_A)

            _save_json(best_json, {
                "schema": "fit_simplified_dn_toposafe_best.v1",
                "stage": stage,
                "target_path": str(Path(target_path).resolve()),
                "seed_path": str(Path(seed_path).resolve()),
                "dxf_path": dxf_path,
                "best_score": float(best_score),
                "best_currents_A": best_currents_A,
                "best_currents_MA": _currents_A_to_MA(best_currents_A),
                "best_result": best,
                "free_keys": free_keys,
                "fixed_keys": fixed_keys,
                "physics": physics,
                "log_jsonl": str(log_jsonl),
            })

        si_best = {}
        if best is not None and isinstance(best.get("score_info", None), dict):
            si_best = best["score_info"]

        dt = time.time() - t0
        print(
            f"[ITER {it:03d}] "
            f"batch_best={batch_best_score:.6g} global_best={best_score:.6g} "
            f"{'IMPROVED' if improved else '':>8s} "
            f"| ok={ok_solve}/{len(batch)} sep={ok_sep}/{len(batch)} "
            f"| A={_safe_float(si_best.get('A', np.nan)):.3f} "
            f"k={_safe_float(si_best.get('kappa', np.nan)):.3f} "
            f"d={_safe_float(si_best.get('delta_bar', np.nan)):.3f} "
            f"Rax={_safe_float(si_best.get('Rax', np.nan)):.3f} "
            # f"CS={_currents_A_to_MA(best_currents_A).get('CS', 0.0):+.3f} MA "
            f"Zax={_safe_float(si_best.get('Zax', np.nan)):.3f} "
            f"area={_safe_float(si_best.get('area', np.nan)):.2f} "
            f"gap={_safe_float(si_best.get('signed_gap_to_WALL_INNER_m', np.nan)):.4f} "
            f"out={_safe_float(si_best.get('outside_frac', np.nan)):.4f} "
            f"wall={_safe_float(si_best.get('wall_term', np.nan)):.1f} "
            f"branch={_safe_float(si_best.get('branch_term', np.nan)):.1f} "
            f"| {dt:.1f}s"
        )

        # Early stop if quite decent.
        # Disabled by --no-early-stop. For release-cs exploration this is usually needed,
        # because the previous thresholds are too permissive for STAR-like targets.
        if (not no_early_stop) and bool(si_best.get("has_true_sep", False)):
            if (
                float(si_best.get("shape_term", 1e9)) < 15.0
                and float(si_best.get("boundary_term", 1e9)) < 10.0
                and best_score < 1500.0
            ):
                print("[STOP] early criterion reached.")
                break

    out = {
        "schema": "fit_simplified_dn_toposafe_run.v1",
        "stage": stage,
        "target_path": str(Path(target_path).resolve()),
        "seed_path": str(Path(seed_path).resolve()),
        "dxf_path": dxf_path,
        "best_score": float(best_score),
        "best_currents_A": best_currents_A,
        "best_currents_MA": _currents_A_to_MA(best_currents_A),
        "best_result": best,
        "free_keys": free_keys,
        "fixed_keys": fixed_keys,
        "physics": physics,
        "log_jsonl": str(log_jsonl),
        "best_json": str(best_json),
    }

    _save_json(run_json, out)
    _save_json(best_json, out)

    print("\n[OK] finished")
    print(f"[OK] best score: {best_score:.6g}")
    print(f"[OK] best currents [MA]: {_fmt_currents_MA(best_currents_A)}")
    print(f"[OK] best json: {best_json}")
    print(f"[OK] run json : {run_json}")
    print(f"[OK] log jsonl: {log_jsonl}")

    return out


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--target",
        type=str,
        default=str(_here() / "results" / "targets" / "star_simplified_dn_target.json"),
        help="Path to star_simplified_dn_target.json",
    )
    ap.add_argument(
        "--seed",
        type=str,
        default=str(_here() / "results" / "seeds" / "baseline_visual_sano_4MA.json"),
        help="Path to seed currents JSON",
    )
    ap.add_argument(
        "--dxf",
        type=str,
        default=None,
        help="DXF path. If omitted, star_equilibrium uses cfg.dxf_path/default.",
    )

    ap.add_argument(
        "--stage",
        type=str,
        default="pf-only",
        choices=["pf-only", "pf1-pf", "release-cs"],
        help="Fit stage.",
    )

    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--seed-rng", type=int, default=11)

    ap.add_argument(
        "--widen",
        type=float,
        default=0.18,
        help="Local bound widening factor around current best.",
    )
    ap.add_argument(
        "--sigma-frac",
        type=float,
        default=0.22,
        help="Gaussian sigma as fraction of current local bound width.",
    )

    ap.add_argument(
        "--free-keys",
        type=str,
        default=None,
        help="Optional override, e.g. PF2,PF3,PF4,PF5,PF6",
    )

    ap.add_argument(
        "--no-coarse",
        action="store_true",
        help="Disable coarse solver overrides.",
    )
    ap.add_argument(
        "--show-solver",
        action="store_true",
        help="Do not redirect solver noise.",
    )
    ap.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping criterion.",
    )

    return ap.parse_args()


def main() -> None:
    mp.freeze_support()
    args = parse_args()

    free_override = _parse_keys_csv(args.free_keys) if args.free_keys else None

    fit_toposafe(
        target_path=str(args.target),
        seed_path=str(args.seed),
        dxf_path=args.dxf,
        stage=str(args.stage),
        workers=int(args.workers),
        timeout_s=float(args.timeout),
        iters=int(args.iters),
        pop=int(args.pop),
        seed=int(args.seed_rng),
        widen=float(args.widen),
        sigma_frac=float(args.sigma_frac),
        no_coarse=bool(args.no_coarse),
        show_solver=bool(args.show_solver),
        free_keys_override=free_override,
        no_early_stop=bool(args.no_early_stop),
    )


if __name__ == "__main__":
    main()
