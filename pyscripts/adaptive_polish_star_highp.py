#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
adaptive_polish_star_highp.py

Adaptive current optimizer for the STAR-like high-pressure branch.

Purpose
-------
Polish a difficult high-p / high-Ip equilibrium by varying active coil-family
currents directly, without engineering current limits. It is intended for
exploration, not as a final engineering-constrained optimizer.

Default target:
    Ip      = 13.2 MA
    paxis   = 1.2 MPa
    R0      ~ 4.208 m
    A       ~ 2.03
    kappa   ~ 2.16
    delta   ~ 0.47
    LCFS fully inside WALL_INNER
    then soft x-point / DN plausibility

Recommended use:
    1) Put this file in pyscripts/
    2) Copy the seed JSON into results/seeds/
    3) Run from pyscripts with passives OFF for speed.

Example:
    $env:STAR_PASSIVE_STRUCTURES="0"
    $env:STAR_PASSIVE_USE_STAR_VESSEL="0"
    $env:STAR_PLOT_PASSIVES="0"
    $env:STAR_EQ_DOMAIN_SOURCE="outer"

    py .\adaptive_polish_star_highp.py `
      --seed .\results\seeds\star_13p2MA_1p2MPa_ugly_seed.json `
      --dxf .\cad\star_baseline.dxf `
      --workers 8 --pop 16 --max-total-iters 90
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import datetime as _dt
import importlib
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


DEFAULT_FREE_KEYS = ["CS_MID", "CS_END", "PF1", "PF2", "PF3", "PF4", "PF5", "PF6"]
DEFAULT_FIXED_KEYS = ["CS"]

TARGET_DEFAULT = {
    "R0": 4.2083,
    "A": 2.0275,
    "kappa": 2.16,
    "delta": 0.4695,
    "area": 27.47,
    "min_gap_m": 0.060,
    "preferred_gap_m": 0.090,
}

SIGMA_DEFAULT = {
    "R0": 0.18,
    "A": 0.12,
    "kappa": 0.13,
    "delta": 0.055,
    "area": 4.0,
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _as_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _parse_keys_csv(s: str) -> List[str]:
    return [x.strip().upper() for x in str(s).split(",") if x.strip()]


def _load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_currents(d: Dict[str, Any]) -> Dict[str, float]:
    """
    Accept several seed formats.
    """
    for key in ("currents_A", "family_currents_A", "BASE_FAMILY_CURRENTS_A", "best_currents_A"):
        v = d.get(key)
        if isinstance(v, dict) and v:
            return {str(k).upper(): float(vv) for k, vv in v.items()}

    # Some fit outputs keep currents at top level.
    out = {}
    for k in ["CS", "CS_MID", "CS_END", "PF1", "PF2", "PF3", "PF4", "PF5", "PF6"]:
        if k in d:
            out[k] = float(d[k])
        kk = f"{k}_current"
        if kk in d:
            out[k] = float(d[kk])
    if out:
        return out

    raise ValueError(f"Could not find currents in seed JSON. Keys={list(d.keys())}")


def _load_seed(path: str | Path) -> Tuple[Dict[str, float], Dict[str, Any]]:
    d = _load_json(path)
    currents = _find_currents(d)
    physics = dict(d.get("physics", {}) or {})
    return currents, physics


def _write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _append_jsonl(path: str | Path, obj: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Geometry / containment metrics
# ---------------------------------------------------------------------------

def _drop_duplicate_endpoint(xy: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    xy = np.asarray(xy, float)
    if xy.shape[0] >= 2 and np.linalg.norm(xy[0] - xy[-1]) < tol:
        return xy[:-1]
    return xy


def _point_segment_min_dist(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Vectorized point-to-segment distances.
    P: (N,2), A/B: (M,2). Returns min distance for each P.
    """
    P = np.asarray(P, float)
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    AB = B - A
    denom = np.sum(AB * AB, axis=1)
    denom = np.where(denom <= 1e-30, 1e-30, denom)

    best = np.full(P.shape[0], np.inf, dtype=float)
    # Chunk to avoid huge memory.
    for i0 in range(0, P.shape[0], 512):
        PP = P[i0:i0 + 512]
        AP = PP[:, None, :] - A[None, :, :]
        t = np.sum(AP * AB[None, :, :], axis=2) / denom[None, :]
        t = np.clip(t, 0.0, 1.0)
        Q = A[None, :, :] + t[:, :, None] * AB[None, :, :]
        D = np.sqrt(np.sum((PP[:, None, :] - Q) ** 2, axis=2))
        best[i0:i0 + 512] = np.min(D, axis=1)
    return best


def _inside_wall_metrics(R_sep: np.ndarray, Z_sep: np.ndarray, geom: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from matplotlib.path import Path as MplPath
    except Exception:
        return {"inside": False, "outside_frac": 1.0, "min_gap_m": -1.0, "reason": "matplotlib_path_failed"}

    if R_sep.size < 20 or Z_sep.size != R_sep.size:
        return {"inside": False, "outside_frac": 1.0, "min_gap_m": -1.0, "reason": "no_lcfs_points"}

    if "R_inner" not in geom or "Z_inner" not in geom:
        return {"inside": True, "outside_frac": 0.0, "min_gap_m": float("nan"), "reason": "no_inner_wall"}

    wall = np.column_stack([
        np.asarray(geom["R_inner"], float).ravel(),
        np.asarray(geom["Z_inner"], float).ravel(),
    ])
    wall = _drop_duplicate_endpoint(wall)
    pts = np.column_stack([R_sep, Z_sep])

    path = MplPath(wall, closed=True)
    inside_each = path.contains_points(pts, radius=-1.0e-9)
    outside_frac = float(1.0 - np.mean(inside_each))

    A = wall
    B = np.roll(wall, -1, axis=0)
    d = _point_segment_min_dist(pts, A, B)
    min_dist = float(np.nanmin(d)) if d.size else float("nan")

    # Signed-ish gap: negative if any point outside.
    min_gap = min_dist if outside_frac <= 0.0 else -min_dist

    return {
        "inside": bool(outside_frac <= 0.0),
        "outside_frac": outside_frac,
        "min_gap_m": float(min_gap),
    }


def _get_lcfs_points(shape: Dict[str, Any], diag: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    for src in (shape, diag, shape.get("plasma_diag", {}) or {}):
        R = src.get("R_sep", None) or src.get("R_lcfs", None)
        Z = src.get("Z_sep", None) or src.get("Z_lcfs", None)
        if R is not None and Z is not None:
            try:
                R = np.asarray(R, float).ravel()
                Z = np.asarray(Z, float).ravel()
                if R.size > 20 and Z.size == R.size:
                    return R, Z
            except Exception:
                pass
    return None, None


def _extract_xpoints(shape: Dict[str, Any]) -> List[Tuple[float, float]]:
    """
    Best-effort extraction from heterogeneous analyze_star outputs.
    """
    out: List[Tuple[float, float]] = []

    def maybe_add(obj: Any):
        if isinstance(obj, dict):
            # Common forms: {"R":..., "Z":...}, {"R_x":..., "Z_x":...}
            pairs = [
                ("R", "Z"),
                ("Rx", "Zx"),
                ("R_x", "Z_x"),
                ("R_xpt", "Z_xpt"),
            ]
            for rk, zk in pairs:
                if rk in obj and zk in obj:
                    R = _as_float(obj.get(rk))
                    Z = _as_float(obj.get(zk))
                    if np.isfinite(R) and np.isfinite(Z):
                        out.append((R, Z))
            for v in obj.values():
                maybe_add(v)
        elif isinstance(obj, (list, tuple)):
            # Pair-like
            if len(obj) >= 2 and all(isinstance(x, (int, float, np.floating)) for x in obj[:2]):
                R = _as_float(obj[0])
                Z = _as_float(obj[1])
                if np.isfinite(R) and np.isfinite(Z):
                    out.append((R, Z))
            else:
                for v in obj:
                    maybe_add(v)

    # Only inspect keys that likely refer to nulls/xpoints to avoid collecting arbitrary R/Z.
    for k, v in shape.items():
        lk = str(k).lower()
        if "xpt" in lk or "xpoint" in lk or "null" in lk:
            maybe_add(v)

    # Unique by rounding.
    uniq = []
    seen = set()
    for R, Z in out:
        key = (round(float(R), 3), round(float(Z), 3))
        if key not in seen:
            seen.add(key)
            uniq.append((float(R), float(Z)))
    return uniq


# ---------------------------------------------------------------------------
# Worker evaluation
# ---------------------------------------------------------------------------

def _evaluate_candidate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker-safe candidate evaluator.

    Important: set environment variables before importing config/star_equilibrium.
    """
    currents = {str(k).upper(): float(v) for k, v in payload["currents"].items()}
    dxf = payload["dxf"]
    ip_A = float(payload["ip_A"])
    paxis_Pa = float(payload["paxis_Pa"])
    domain_source = str(payload.get("domain_source", "outer"))
    passives = bool(payload.get("passives", False))
    target = dict(payload["target"])
    sigma = dict(payload["sigma"])
    phase = str(payload.get("phase", "containment"))
    cs_segmented = bool(payload.get("cs_segmented", True))
    cs_mid_fraction = float(payload.get("cs_mid_fraction", 0.45))

    try:
        os.environ["STAR_IP_A"] = str(ip_A)
        os.environ["STAR_PAXIS_PA"] = str(paxis_Pa)
        os.environ["STAR_EQ_DOMAIN_SOURCE"] = domain_source
        os.environ["STAR_PASSIVE_STRUCTURES"] = "1" if passives else "0"
        os.environ["STAR_PASSIVE_USE_STAR_VESSEL"] = "1" if passives else "0"
        os.environ["STAR_PLOT_PASSIVES"] = "0"

        import config_star_bean as cfg
        # Force current controls in this worker.
        cfg.cs_segmented = cs_segmented
        cfg.cs_mid_fraction = cs_mid_fraction
        cfg.cs_segment_zcut_m = None
        cfg.cs_segment_keep_parent = True
        cfg.Ip = ip_A
        cfg.paxis = paxis_Pa
        cfg.passive_structures_enabled = passives
        cfg.passive_use_star_vessel = passives
        cfg.plot_passive_filaments = False
        cfg.eq_domain_source = domain_source

        for k in ["CS", "CS_MID", "CS_END", "PF1", "PF2", "PF3", "PF4", "PF5", "PF6"]:
            setattr(cfg, f"{k}_current", float(currents.get(k, 0.0)))

        cfg.BASE_FAMILY_CURRENTS_A = {k: float(currents.get(k, 0.0)) for k in currents}

        import star_equilibrium as se
        # make sure module sees updated cfg object
        se.cfg = cfg

        eq, tokamak, geom, shape = se.build_equilibrium(
            verbose=False,
            redirect_solver_noise=True,
            dxf_path=dxf,
        )

        diag = shape.get("plasma_diag", None)
        if not isinstance(diag, dict) or not diag.get("ok", False):
            try:
                diag = se.plasma_diagnostics(eq, geom, shape)
            except Exception:
                diag = {}

        ok_sep = bool(diag.get("ok", False))
        R_sep, Z_sep = _get_lcfs_points(shape, diag)

        containment = {"inside": False, "outside_frac": 1.0, "min_gap_m": -1.0}
        if R_sep is not None and Z_sep is not None:
            containment = _inside_wall_metrics(R_sep, Z_sep, geom)

        R0 = _as_float(diag.get("R0", diag.get("R0_plasma", np.nan)))
        A = _as_float(diag.get("A", diag.get("A_plasma", np.nan)))
        kappa = _as_float(diag.get("kappa", diag.get("kappa_plasma", np.nan)))
        du = _as_float(diag.get("delta_u", np.nan))
        dl = _as_float(diag.get("delta_l", np.nan))
        delta = 0.5 * (du + dl) if np.isfinite(du + dl) else np.nan
        area = _as_float(diag.get("area_m2", diag.get("area", np.nan)))

        xpts = _extract_xpoints(shape)

        score, components = _score_metrics(
            ok_sep=ok_sep,
            containment=containment,
            R0=R0,
            A=A,
            kappa=kappa,
            delta=delta,
            area=area,
            xpts=xpts,
            target=target,
            sigma=sigma,
            phase=phase,
        )

        return {
            "ok": True,
            "score": float(score),
            "components": components,
            "currents_A": currents,
            "metrics": {
                "ok_sep": ok_sep,
                "method": diag.get("method", None),
                "R0": R0,
                "A": A,
                "kappa": kappa,
                "delta": delta,
                "delta_u": du,
                "delta_l": dl,
                "area_m2": area,
                "inside_wall": bool(containment.get("inside", False)),
                "outside_frac": float(containment.get("outside_frac", 1.0)),
                "min_gap_m": float(containment.get("min_gap_m", -1.0)),
                "n_xpoints_detected": len(xpts),
                "xpoints": xpts,
            },
        }

    except Exception as e:
        return {
            "ok": False,
            "score": 1.0e30,
            "error": repr(e),
            "traceback": traceback.format_exc(limit=8),
            "currents_A": currents,
            "metrics": {
                "ok_sep": False,
                "inside_wall": False,
                "outside_frac": 1.0,
                "min_gap_m": -1.0,
            },
        }


def _score_metrics(
    *,
    ok_sep: bool,
    containment: Dict[str, Any],
    R0: float,
    A: float,
    kappa: float,
    delta: float,
    area: float,
    xpts: List[Tuple[float, float]],
    target: Dict[str, float],
    sigma: Dict[str, float],
    phase: str,
) -> Tuple[float, Dict[str, float]]:
    """
    Lower is better.

    phase:
      containment -> hard search for valid LCFS inside WALL_INNER
      shape       -> preserve containment while forcing STAR-like shape
      xpoints     -> preserve shape while softly preferring upper/lower X-points
    """
    score = 0.0
    comp: Dict[str, float] = {}

    if not ok_sep:
        comp["no_sep"] = 1.0e8
        return 1.0e8, comp

    outside_frac = float(containment.get("outside_frac", 1.0))
    min_gap = float(containment.get("min_gap_m", -1.0))
    min_gap_target = float(target.get("min_gap_m", 0.06))
    pref_gap = float(target.get("preferred_gap_m", 0.09))

    # Containment dominates all phases.
    p_out = 5.0e6 * outside_frac**2
    p_gap = 0.0
    if min_gap < min_gap_target:
        p_gap = 2.0e5 * ((min_gap_target - min_gap) / max(0.02, min_gap_target)) ** 2
    comp["containment"] = p_out + p_gap
    score += comp["containment"]

    # Soft preference for not sitting exactly on wall.
    p_prefgap = 0.0
    if np.isfinite(min_gap) and min_gap < pref_gap:
        p_prefgap = 150.0 * ((pref_gap - min_gap) / 0.05) ** 2
    comp["preferred_gap"] = p_prefgap
    score += p_prefgap

    # Shape score.
    def norm(name: str, val: float) -> float:
        tv = float(target[name])
        sg = float(sigma[name])
        if not np.isfinite(val):
            return 1.0e6
        return ((val - tv) / sg) ** 2

    shape_raw = (
        2.0 * norm("R0", R0)
        + 5.0 * norm("A", A)
        + 4.0 * norm("kappa", kappa)
        + 2.0 * norm("delta", delta)
    )

    # Area can be useful, but do not make it mandatory at high beta.
    if np.isfinite(area) and "area" in target:
        shape_raw += 0.35 * ((area - float(target["area"])) / float(sigma.get("area", 4.0))) ** 2

    if phase == "containment":
        comp["shape"] = 0.10 * shape_raw
    elif phase == "shape":
        comp["shape"] = 1.0 * shape_raw
    else:
        comp["shape"] = 1.25 * shape_raw
    score += comp["shape"]

    # X-point / double-null plausibility: soft and only strong in xpoints phase.
    xp = 0.0
    if phase == "xpoints":
        upper = [(R, Z) for R, Z in xpts if Z > 0.5]
        lower = [(R, Z) for R, Z in xpts if Z < -0.5]
        if not upper:
            xp += 400.0
        if not lower:
            xp += 400.0
        # Prefer x-points near the vertical ends and inboard-ish, but soft.
        for group, zt in [(upper, 4.5), (lower, -4.5)]:
            if group:
                best = min(((R - 3.0) / 0.8) ** 2 + ((Z - zt) / 0.8) ** 2 for R, Z in group)
                xp += 35.0 * best
    else:
        # mild reward/penalty for detected upper/lower nulls without forcing
        upper = any(Z > 0.5 for _, Z in xpts)
        lower = any(Z < -0.5 for _, Z in xpts)
        if not (upper and lower):
            xp += 25.0
    comp["xpoints"] = xp
    score += xp

    return float(score), {k: float(v) for k, v in comp.items()}


# ---------------------------------------------------------------------------
# Sampling / adaptive optimizer
# ---------------------------------------------------------------------------

def _sample_candidate(
    base: Dict[str, float],
    *,
    free_keys: List[str],
    fixed_keys: List[str],
    sigma_frac: float,
    abs_floor_A: float,
    rng: random.Random,
) -> Dict[str, float]:
    cand = dict(base)
    for k in free_keys:
        k = k.upper()
        if k in fixed_keys:
            continue
        v = float(base.get(k, 0.0))
        scale = max(abs(v) * sigma_frac, abs_floor_A)
        cand[k] = v + rng.gauss(0.0, scale)
    for k in fixed_keys:
        cand[k.upper()] = float(base.get(k.upper(), 0.0))
    return cand


def _stage_from_best(best_result: Dict[str, Any], current_stage: str, it_in_stage: int, args: argparse.Namespace) -> str:
    m = best_result.get("metrics", {}) or {}
    inside = bool(m.get("inside_wall", False))
    gap = float(m.get("min_gap_m", -1.0))
    A = float(m.get("A", float("nan")))
    k = float(m.get("kappa", float("nan")))
    R0 = float(m.get("R0", float("nan")))
    d = float(m.get("delta", float("nan")))

    if current_stage == "containment":
        if inside and gap >= float(args.min_gap_m):
            return "shape"
        if it_in_stage >= int(args.max_containment_iters):
            return "shape"

    if current_stage == "shape":
        good_shape = (
            inside
            and gap >= 0.8 * float(args.min_gap_m)
            and np.isfinite(A) and abs(A - float(args.target_A)) <= float(args.shape_A_tol)
            and np.isfinite(k) and abs(k - float(args.target_kappa)) <= float(args.shape_kappa_tol)
            and np.isfinite(R0) and abs(R0 - float(args.target_R0)) <= float(args.shape_R0_tol)
            and np.isfinite(d) and abs(d - float(args.target_delta)) <= float(args.shape_delta_tol)
        )
        if good_shape:
            return "xpoints"
        if it_in_stage >= int(args.max_shape_iters):
            return "xpoints"

    return current_stage


def run_optimizer(args: argparse.Namespace) -> None:
    seed_currents, seed_physics = _load_seed(args.seed)

    # Explicit physics wins over seed.
    ip_A = float(args.Ip_A if args.Ip_A is not None else seed_physics.get("Ip_A", 13.2e6))
    paxis_Pa = float(args.paxis_Pa if args.paxis_Pa is not None else seed_physics.get("paxis_Pa", 1.2e6))

    free_keys = _parse_keys_csv(args.free_keys)
    fixed_keys = _parse_keys_csv(args.fixed_keys)

    target = {
        "R0": float(args.target_R0),
        "A": float(args.target_A),
        "kappa": float(args.target_kappa),
        "delta": float(args.target_delta),
        "area": float(args.target_area),
        "min_gap_m": float(args.min_gap_m),
        "preferred_gap_m": float(args.preferred_gap_m),
    }
    sigma = {
        "R0": float(args.sigma_R0),
        "A": float(args.sigma_A),
        "kappa": float(args.sigma_kappa),
        "delta": float(args.sigma_delta),
        "area": float(args.sigma_area),
    }

    if args.outdir is None:
        outdir = Path("results") / f"adaptive_highp_polish_{_now_tag()}"
    else:
        outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "seed": str(args.seed),
        "dxf": str(args.dxf),
        "Ip_A": ip_A,
        "paxis_Pa": paxis_Pa,
        "free_keys": free_keys,
        "fixed_keys": fixed_keys,
        "target": target,
        "sigma": sigma,
        "args": vars(args),
    }
    _write_json(outdir / "run_config.json", run_config)

    rng = random.Random(int(args.seed_rng))

    best_currents = {k: float(seed_currents.get(k, 0.0)) for k in ["CS", "CS_MID", "CS_END", "PF1", "PF2", "PF3", "PF4", "PF5", "PF6"]}
    current_best_result: Optional[Dict[str, Any]] = None

    stage = "containment"
    stage_iter = 0
    sigma_frac = float(args.sigma_frac)
    no_improve = 0

    print(f"[OUTDIR] {outdir.resolve()}")
    print(f"[PHYSICS] Ip={ip_A/1e6:.3f} MA paxis={paxis_Pa:.3e} Pa")
    print(f"[FREE] {free_keys}")
    print(f"[FIXED] {fixed_keys}")
    print(f"[TARGET] {target}")

    with cf.ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
        for it in range(1, int(args.max_total_iters) + 1):
            stage_iter += 1

            candidates = [dict(best_currents)]
            for _ in range(max(0, int(args.pop) - 1)):
                candidates.append(_sample_candidate(
                    best_currents,
                    free_keys=free_keys,
                    fixed_keys=fixed_keys,
                    sigma_frac=sigma_frac,
                    abs_floor_A=float(args.abs_step_floor_A),
                    rng=rng,
                ))

            payloads = []
            for cand in candidates:
                payloads.append({
                    "currents": cand,
                    "dxf": str(args.dxf),
                    "ip_A": ip_A,
                    "paxis_Pa": paxis_Pa,
                    "domain_source": str(args.domain_source),
                    "passives": bool(args.passives),
                    "target": target,
                    "sigma": sigma,
                    "phase": stage,
                    "cs_segmented": bool(args.cs_segmented),
                    "cs_mid_fraction": float(args.cs_mid_fraction),
                })

            futs = [ex.submit(_evaluate_candidate, p) for p in payloads]
            results = []
            for fut in cf.as_completed(futs, timeout=float(args.batch_timeout_s)):
                try:
                    results.append(fut.result(timeout=1.0))
                except Exception as e:
                    results.append({"ok": False, "score": 1e30, "error": repr(e)})

            if not results:
                print(f"[ITER {it:03d}] no results returned")
                continue

            results.sort(key=lambda r: float(r.get("score", 1e30)))
            batch_best = results[0]

            improved = (
                current_best_result is None
                or float(batch_best.get("score", 1e30)) < float(current_best_result.get("score", 1e30))
            )

            if improved:
                current_best_result = batch_best
                best_currents = dict(batch_best["currents_A"])
                no_improve = 0
            else:
                no_improve += 1

            # Adaptive sigma.
            if no_improve >= int(args.shrink_after):
                sigma_frac *= float(args.shrink_factor)
                sigma_frac = max(sigma_frac, float(args.sigma_frac_min))
                no_improve = 0
            elif improved and it % int(args.expand_every) == 0:
                sigma_frac = min(float(args.sigma_frac_max), sigma_frac * float(args.expand_factor))

            m = (current_best_result or {}).get("metrics", {}) or {}
            comp = (current_best_result or {}).get("components", {}) or {}

            status = (
                f"[ITER {it:03d} {stage:11s}] "
                f"batch={float(batch_best.get('score',1e30)):.3g} "
                f"best={float((current_best_result or {}).get('score',1e30)):.3g} "
                f"{'IMPROVED' if improved else '        '} "
                f"sigma={sigma_frac:.4f} "
                f"sep={m.get('ok_sep')} in={m.get('inside_wall')} gap={_as_float(m.get('min_gap_m')):.3f} "
                f"R0={_as_float(m.get('R0')):.3f} A={_as_float(m.get('A')):.3f} "
                f"k={_as_float(m.get('kappa')):.3f} d={_as_float(m.get('delta')):.3f} "
                f"xp={m.get('n_xpoints_detected')}"
            )
            print(status, flush=True)

            # Save everything important.
            record = {
                "iter": it,
                "stage": stage,
                "stage_iter": stage_iter,
                "sigma_frac": sigma_frac,
                "improved": improved,
                "batch_best": batch_best,
                "global_best": current_best_result,
            }
            _append_jsonl(outdir / "manifest.jsonl", record)

            best_obj = {
                "schema": "star_adaptive_polish_best_v1",
                "iter": it,
                "stage": stage,
                "score": float((current_best_result or {}).get("score", 1e30)),
                "components": (current_best_result or {}).get("components", {}),
                "metrics": (current_best_result or {}).get("metrics", {}),
                "physics": {
                    "Ip_A": ip_A,
                    "paxis_Pa": paxis_Pa,
                },
                "cs_segmented": bool(args.cs_segmented),
                "cs_mid_fraction": float(args.cs_mid_fraction),
                "currents_A": best_currents,
                "family_currents_A": best_currents,
                "BASE_FAMILY_CURRENTS_A": best_currents,
                "target": target,
            }
            _write_json(outdir / "best.json", best_obj)
            _write_json(outdir / f"best_iter_{it:03d}_{stage}.json", best_obj)

            # Write a config snippet for quick manual copy-paste.
            snippet = _config_snippet(best_currents, ip_A, paxis_Pa, args)
            (outdir / "best_config_snippet.py").write_text(snippet, encoding="utf-8")

            # Stage transition if criteria met.
            new_stage = _stage_from_best(current_best_result or {}, stage, stage_iter, args)
            if new_stage != stage:
                print(f"[STAGE] {stage} -> {new_stage}", flush=True)
                stage = new_stage
                stage_iter = 0
                sigma_frac = max(float(args.sigma_frac_stage_reset), sigma_frac)

    print("[DONE]")
    print(f"[BEST] {outdir / 'best.json'}")
    print(f"[SNIPPET] {outdir / 'best_config_snippet.py'}")


def _config_snippet(currents: Dict[str, float], ip_A: float, paxis_Pa: float, args: argparse.Namespace) -> str:
    keys = ["CS", "CS_MID", "CS_END", "PF1", "PF2", "PF3", "PF4", "PF5", "PF6"]
    lines = []
    lines.append("# Best current snippet from adaptive_polish_star_highp.py")
    lines.append("cs_segmented = True")
    lines.append(f"cs_mid_fraction = {float(args.cs_mid_fraction):.8g}")
    lines.append("cs_segment_zcut_m = None")
    lines.append("cs_segment_keep_parent = True")
    lines.append("")
    for k in keys:
        lines.append(f"{k}_current = {float(currents.get(k,0.0)):.12g}")
    lines.append("")
    lines.append(f'Ip = _env_float("STAR_IP_A", {float(ip_A):.12g})')
    lines.append(f'paxis = _env_float("STAR_PAXIS_PA", {float(paxis_Pa):.12g})')
    lines.append("")
    lines.append("BASE_FAMILY_CURRENTS_A = {")
    for k in keys:
        lines.append(f'    "{k}": {k}_current,')
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--seed", required=True, help="Seed JSON with currents_A/family_currents_A.")
    ap.add_argument("--dxf", default=r".\cad\star_baseline.dxf")
    ap.add_argument("--outdir", default=None)

    ap.add_argument("--Ip-A", dest="Ip_A", type=float, default=13.2e6)
    ap.add_argument("--paxis-Pa", dest="paxis_Pa", type=float, default=1.2e6)

    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pop", type=int, default=14)
    ap.add_argument("--max-total-iters", type=int, default=90)
    ap.add_argument("--batch-timeout-s", type=float, default=2400.0)

    ap.add_argument("--free-keys", type=str, default=",".join(DEFAULT_FREE_KEYS))
    ap.add_argument("--fixed-keys", type=str, default=",".join(DEFAULT_FIXED_KEYS))

    ap.add_argument("--sigma-frac", type=float, default=0.055)
    ap.add_argument("--sigma-frac-min", type=float, default=0.008)
    ap.add_argument("--sigma-frac-max", type=float, default=0.14)
    ap.add_argument("--sigma-frac-stage-reset", type=float, default=0.035)
    ap.add_argument("--abs-step-floor-A", type=float, default=0.18e6)
    ap.add_argument("--shrink-after", type=int, default=3)
    ap.add_argument("--shrink-factor", type=float, default=0.65)
    ap.add_argument("--expand-every", type=int, default=7)
    ap.add_argument("--expand-factor", type=float, default=1.10)
    ap.add_argument("--seed-rng", type=int, default=23)

    ap.add_argument("--domain-source", type=str, default="outer", choices=["outer", "machine", "inner"])
    ap.add_argument("--passives", action="store_true", help="Use STAR_VESSEL passives during optimization. Usually leave OFF.")

    ap.add_argument("--cs-segmented", action="store_true", default=True)
    ap.add_argument("--cs-mid-fraction", type=float, default=0.45)

    # Target geometry.
    ap.add_argument("--target-R0", type=float, default=TARGET_DEFAULT["R0"])
    ap.add_argument("--target-A", type=float, default=TARGET_DEFAULT["A"])
    ap.add_argument("--target-kappa", type=float, default=TARGET_DEFAULT["kappa"])
    ap.add_argument("--target-delta", type=float, default=TARGET_DEFAULT["delta"])
    ap.add_argument("--target-area", type=float, default=TARGET_DEFAULT["area"])
    ap.add_argument("--min-gap-m", type=float, default=TARGET_DEFAULT["min_gap_m"])
    ap.add_argument("--preferred-gap-m", type=float, default=TARGET_DEFAULT["preferred_gap_m"])

    # Metric sigmas.
    ap.add_argument("--sigma-R0", type=float, default=SIGMA_DEFAULT["R0"])
    ap.add_argument("--sigma-A", type=float, default=SIGMA_DEFAULT["A"])
    ap.add_argument("--sigma-kappa", type=float, default=SIGMA_DEFAULT["kappa"])
    ap.add_argument("--sigma-delta", type=float, default=SIGMA_DEFAULT["delta"])
    ap.add_argument("--sigma-area", type=float, default=SIGMA_DEFAULT["area"])

    # Stage transition tolerances.
    ap.add_argument("--max-containment-iters", type=int, default=18)
    ap.add_argument("--max-shape-iters", type=int, default=55)
    ap.add_argument("--shape-R0-tol", type=float, default=0.20)
    ap.add_argument("--shape-A-tol", type=float, default=0.17)
    ap.add_argument("--shape-kappa-tol", type=float, default=0.18)
    ap.add_argument("--shape-delta-tol", type=float, default=0.09)

    args = ap.parse_args()
    run_optimizer(args)


if __name__ == "__main__":
    main()
