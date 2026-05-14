#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fit_simplified_dn_corridor.py

Topology-safe + divertor-leg corridor fit for STAR-like equilibrium.

This script extends fit_simplified_dn_toposafe.py by adding an explicit
leg-corridor objective using CAD marker windows:

    LEG_UPPER_WIN
    LEG_LOWER_WIN

Purpose
-------
The previous optimizer could find a clean separatrix, but it often accepted
solutions where the divertor leg exited through the wrong corridor, e.g.
PF5-PF6 instead of PF4-PF5.

This script keeps the same target/shape/topology objective and adds:

    leg_corridor_penalty

which penalizes distance between the obtained separatrix/leg and the desired
CAD leg windows.

Requirements
------------
- fit_simplified_dn_toposafe.py must exist in the same folder.
- star_machine_cad.py must already import:
    geom["marker_windows"]["leg_upper"]
    geom["marker_windows"]["leg_lower"]

Recommended first run
---------------------
py .\\fit_simplified_dn_corridor.py ^
  --target .\\results\\targets\\star_simplified_dn_target.json ^
  --seed .\\results\\fit_simplified_dn_toposafe_best.json ^
  --stage pf1-pf ^
  --iters 30 ^
  --pop 20 ^
  --workers 6 ^
  --timeout 180 ^
  --widen 0.18 ^
  --sigma-frac 0.20 ^
  --corridor-weight 180 ^
  --corridor-sigma 0.18 ^
  --no-early-stop ^
  --dxf .\\cad\\star_baseline.dxf
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Reuse stable helpers from the previous optimizer.
import fit_simplified_dn_toposafe as base


FAMILIES_ALL = base.FAMILIES_ALL


# ---------------------------------------------------------------------
# Geometry / distance helpers
# ---------------------------------------------------------------------
def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    p = np.asarray(p, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    ab = b - a
    den = float(np.dot(ab, ab))

    if den <= 1e-30:
        return float(np.linalg.norm(p - a))

    t = float(np.dot(p - a, ab) / den)
    t = max(0.0, min(1.0, t))
    q = a + t * ab
    return float(np.linalg.norm(p - q))


def _point_to_polyline_distance(point: np.ndarray, poly: np.ndarray, *, closed: bool = False) -> float:
    P = np.asarray(poly, float)
    if P.ndim != 2 or P.shape[0] < 2 or P.shape[1] != 2:
        return float("inf")

    p = np.asarray(point, float)

    n = P.shape[0]
    nseg = n if closed else n - 1

    dmin = float("inf")
    for i in range(nseg):
        a = P[i]
        b = P[(i + 1) % n]
        d = _point_to_segment_distance(p, a, b)
        if d < dmin:
            dmin = d

    return float(dmin)


def _rms_points_to_polyline(points: np.ndarray, poly: np.ndarray, *, closed: bool = False) -> float:
    P = np.asarray(points, float)
    Q = np.asarray(poly, float)

    if P.ndim != 2 or P.shape[0] < 1 or P.shape[1] != 2:
        return float("inf")
    if Q.ndim != 2 or Q.shape[0] < 2 or Q.shape[1] != 2:
        return float("inf")

    vals = []
    for p in P:
        vals.append(_point_to_polyline_distance(p, Q, closed=closed))

    if not vals:
        return float("inf")

    return float(math.sqrt(float(np.mean(np.asarray(vals, float) ** 2))))


def _resample_polyline(poly: np.ndarray, n: int = 80, *, closed: bool = False) -> np.ndarray:
    P = np.asarray(poly, float)
    if P.ndim != 2 or P.shape[0] < 2 or P.shape[1] != 2:
        return P

    if closed:
        if np.linalg.norm(P[0] - P[-1]) > 1e-12:
            P = np.vstack([P, P[0]])
    else:
        if P.shape[0] >= 2 and np.linalg.norm(P[0] - P[-1]) < 1e-12:
            P = P[:-1]

    if P.shape[0] < 2:
        return P

    d = np.diff(P, axis=0)
    ds = np.hypot(d[:, 0], d[:, 1])
    s = np.r_[0.0, np.cumsum(ds)]
    total = float(s[-1])

    if total <= 1e-12:
        return P

    s_new = np.linspace(0.0, total, int(n), endpoint=not closed)
    R_new = np.interp(s_new, s, P[:, 0])
    Z_new = np.interp(s_new, s, P[:, 1])

    return np.column_stack([R_new, Z_new])


def _bbox_mask(points: np.ndarray, win: np.ndarray, *, pad: float) -> np.ndarray:
    P = np.asarray(points, float)
    W = np.asarray(win, float)

    if P.ndim != 2 or W.ndim != 2 or P.shape[1] != 2 or W.shape[1] != 2:
        return np.zeros(P.shape[0], dtype=bool)

    rmin = float(np.min(W[:, 0]) - pad)
    rmax = float(np.max(W[:, 0]) + pad)
    zmin = float(np.min(W[:, 1]) - pad)
    zmax = float(np.max(W[:, 1]) + pad)

    return (
        (P[:, 0] >= rmin)
        & (P[:, 0] <= rmax)
        & (P[:, 1] >= zmin)
        & (P[:, 1] <= zmax)
    )


def _half_mask(points: np.ndarray, *, upper: bool) -> np.ndarray:
    P = np.asarray(points, float)
    if P.ndim != 2 or P.shape[1] != 2:
        return np.zeros(P.shape[0], dtype=bool)

    if upper:
        return P[:, 1] >= 0.0
    return P[:, 1] <= 0.0


def _get_marker_window(geom: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    mw = geom.get("marker_windows", None)
    if not isinstance(mw, dict):
        return None

    pack = mw.get(key, None)
    if not isinstance(pack, dict):
        return None

    xy = pack.get("xy", None)
    if xy is None:
        return None

    try:
        xy = np.asarray(xy, float)
    except Exception:
        return None

    if xy.ndim != 2 or xy.shape[0] < 2 or xy.shape[1] != 2:
        return None

    return {
        "xy": xy,
        "closed": bool(pack.get("closed", False)),
        "length_m": float(pack.get("length_m", np.nan)),
    }


def _corridor_distance_for_leg(
    *,
    lcfs: np.ndarray,
    win_pack: Dict[str, Any],
    upper: bool,
    roi_pad_m: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    Distance between a desired CAD corridor window and the obtained separatrix.

    Uses two components:
      1) window -> LCFS distance:
         are the desired corridor points close to the separatrix?

      2) LCFS-in-window-ROI -> window distance:
         does the separatrix actually pass through the local region around
         the corridor, instead of being far away globally?

    This is intentionally geometric and conservative.
    """
    lcfs = np.asarray(lcfs, float)
    win = np.asarray(win_pack["xy"], float)
    closed_win = bool(win_pack.get("closed", False))

    if lcfs.ndim != 2 or lcfs.shape[0] < 20 or lcfs.shape[1] != 2:
        return float("inf"), {"ok": False, "reason": "invalid_lcfs"}

    if win.ndim != 2 or win.shape[0] < 2 or win.shape[1] != 2:
        return float("inf"), {"ok": False, "reason": "invalid_window"}

    # Use only the corresponding half of the LCFS.
    m_half = _half_mask(lcfs, upper=upper)
    lcfs_half = lcfs[m_half]

    if lcfs_half.shape[0] < 10:
        return float("inf"), {"ok": False, "reason": "not_enough_lcfs_half_points"}

    # Resample window for stable distance estimate.
    win_s = _resample_polyline(win, n=100, closed=closed_win)

    # Component 1: desired window to LCFS.
    d_win_to_lcfs = _rms_points_to_polyline(win_s, lcfs_half, closed=False)

    # Component 2: local LCFS points near the window back to the window.
    m_roi = _bbox_mask(lcfs_half, win, pad=float(roi_pad_m))
    lcfs_roi = lcfs_half[m_roi]

    if lcfs_roi.shape[0] >= 5:
        d_lcfs_to_win = _rms_points_to_polyline(lcfs_roi, win_s, closed=closed_win)
        roi_hit = True
    else:
        # No separatrix points in the desired corridor region.
        # Use a strong geometric miss value.
        d_lcfs_to_win = 2.0 * float(roi_pad_m)
        roi_hit = False

    # Weighted RMS. The first term is the primary one; the ROI term prevents
    # false positives where the window is close to an unrelated LCFS segment.
    d_total = math.sqrt(0.70 * d_win_to_lcfs**2 + 0.30 * d_lcfs_to_win**2)

    info = {
        "ok": bool(np.isfinite(d_total)),
        "d_total_m": float(d_total),
        "d_win_to_lcfs_m": float(d_win_to_lcfs),
        "d_lcfs_to_win_m": float(d_lcfs_to_win),
        "roi_hit": bool(roi_hit),
        "n_lcfs_half": int(lcfs_half.shape[0]),
        "n_lcfs_roi": int(lcfs_roi.shape[0]),
        "window_length_m": float(win_pack.get("length_m", np.nan)),
    }

    return float(d_total), info


def compute_leg_corridor_penalty(
    *,
    shape: Dict[str, Any],
    diag: Dict[str, Any],
    geom: Dict[str, Any],
    corridor_weight: float,
    corridor_sigma_m: float,
    corridor_roi_pad_m: float,
    missing_corridor_penalty: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    Add penalty for upper/lower divertor legs missing CAD corridor windows.
    """
    lcfs = base._extract_lcfs_curve(shape, diag)

    out: Dict[str, Any] = {
        "corridor_available": False,
        "leg_upper_distance_m": float("inf"),
        "leg_lower_distance_m": float("inf"),
        "leg_corridor_term": float("inf"),
        "leg_corridor_penalty": float(missing_corridor_penalty),
        "leg_upper_info": {},
        "leg_lower_info": {},
    }

    if lcfs is None:
        out["reason"] = "no_lcfs_curve"
        return float(missing_corridor_penalty), out

    leg_upper = _get_marker_window(geom, "leg_upper")
    leg_lower = _get_marker_window(geom, "leg_lower")

    if leg_upper is None or leg_lower is None:
        out["reason"] = "missing_leg_windows"
        out["has_leg_upper"] = leg_upper is not None
        out["has_leg_lower"] = leg_lower is not None
        return float(missing_corridor_penalty), out

    d_up, info_up = _corridor_distance_for_leg(
        lcfs=lcfs,
        win_pack=leg_upper,
        upper=True,
        roi_pad_m=float(corridor_roi_pad_m),
    )

    d_lo, info_lo = _corridor_distance_for_leg(
        lcfs=lcfs,
        win_pack=leg_lower,
        upper=False,
        roi_pad_m=float(corridor_roi_pad_m),
    )

    sigma = max(float(corridor_sigma_m), 1e-6)

    term = (d_up / sigma) ** 2 + (d_lo / sigma) ** 2
    penalty = float(corridor_weight) * float(term)

    # Extra penalty if the separatrix does not enter the corridor ROI.
    if not bool(info_up.get("roi_hit", False)):
        penalty += 0.25 * float(missing_corridor_penalty)
    if not bool(info_lo.get("roi_hit", False)):
        penalty += 0.25 * float(missing_corridor_penalty)

    out.update({
        "corridor_available": True,
        "leg_upper_distance_m": float(d_up),
        "leg_lower_distance_m": float(d_lo),
        "leg_corridor_term": float(term),
        "leg_corridor_penalty": float(penalty),
        "leg_upper_info": info_up,
        "leg_lower_info": info_lo,
    })

    return float(penalty), out


# ---------------------------------------------------------------------
# Subprocess worker
# ---------------------------------------------------------------------
def _worker_eval_corridor(
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
    corridor_weight: float,
    corridor_sigma_m: float,
    corridor_roi_pad_m: float,
    missing_corridor_penalty: float,
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

        # Base topology-safe score.
        score, score_info = base.compute_toposafe_score(
            target=target,
            shape=shape,
            diag=diag,
            currents_A=currents_A,
            ref_currents_A=ref_currents_A,
            seed_currents_A=seed_currents_A,
            free_keys=free_keys,
            stage=stage,
        )

        # New corridor penalty.
        corridor_penalty, corridor_info = compute_leg_corridor_penalty(
            shape=shape,
            diag=diag,
            geom=geom,
            corridor_weight=float(corridor_weight),
            corridor_sigma_m=float(corridor_sigma_m),
            corridor_roi_pad_m=float(corridor_roi_pad_m),
            missing_corridor_penalty=float(missing_corridor_penalty),
        )

        score = float(score) + float(corridor_penalty)

        score_info["leg_corridor_penalty"] = float(corridor_penalty)
        score_info["leg_corridor_term"] = float(corridor_info.get("leg_corridor_term", np.inf))
        score_info["leg_upper_distance_m"] = float(corridor_info.get("leg_upper_distance_m", np.inf))
        score_info["leg_lower_distance_m"] = float(corridor_info.get("leg_lower_distance_m", np.inf))
        score_info["leg_corridor_available"] = bool(corridor_info.get("corridor_available", False))
        score_info["leg_corridor_info"] = corridor_info
        score_info["score_total_with_corridor"] = float(score)

        marker_keys = []
        if isinstance(geom.get("marker_windows", None), dict):
            marker_keys = sorted(list(geom["marker_windows"].keys()))

        # Export LCFS / separatrix curve for divertor-first strike scoring.
        # This is required by fit_simplified_dn_divertor_first.py.
        separatrix_xy = None
        try:
            lcfs_export = base._extract_lcfs_curve(shape, diag)
            if lcfs_export is not None:
                arr = np.asarray(lcfs_export, dtype=float)

                if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 4:
                    arr = arr[:, :2]
                    mask = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
                    arr = arr[mask]

                    if arr.shape[0] >= 4:
                        separatrix_xy = arr.tolist()
        except Exception as _e:
            separatrix_xy = None
            score_info["separatrix_export_error"] = repr(_e)

        conn.send({
            "ok": True,
            "ok_solve": True,
            "score": float(score),
            "score_info": score_info,
            "separatrix_xy": separatrix_xy,
            "has_separatrix_xy": separatrix_xy is not None,
            "currents_A": dict(currents_A),
            "currents_MA": base._currents_A_to_MA(currents_A),
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
            "geom_summary": {
                "marker_windows": marker_keys,
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
                "currents_MA": base._currents_A_to_MA(currents_A),
            })
            conn.close()
        except Exception:
            pass


def eval_case_corridor(
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
    corridor_weight: float,
    corridor_sigma_m: float,
    corridor_roi_pad_m: float,
    missing_corridor_penalty: float,
) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)

    p = ctx.Process(
        target=_worker_eval_corridor,
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
            float(corridor_weight),
            float(corridor_sigma_m),
            float(corridor_roi_pad_m),
            float(missing_corridor_penalty),
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
            "currents_MA": base._currents_A_to_MA(currents_A),
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
# Main optimization loop
# ---------------------------------------------------------------------
def fit_corridor(
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
    corridor_weight: float,
    corridor_sigma_m: float,
    corridor_roi_pad_m: float,
    missing_corridor_penalty: float,
) -> Dict[str, Any]:
    random.seed(int(seed))
    np.random.seed(int(seed))

    target = base._load_json(target_path)
    seed_currents_A, physics = base._load_seed_currents(seed_path)

    stage = str(stage).strip().lower()
    free_keys, fixed_keys = base._stage_free_fixed(stage)

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

    dxf_path = base._resolve_path_maybe(dxf_path)

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    log_jsonl = base._results_dir() / "fit_simplified_dn_corridor_results.jsonl"
    best_json = base._results_dir() / "fit_simplified_dn_corridor_best.json"
    run_json = base._results_dir() / f"fit_simplified_dn_corridor_run_{run_tag}.json"

    best: Optional[Dict[str, Any]] = None
    best_score = float("inf")
    best_currents_A = dict(seed_currents_A)
    ref_currents_A = dict(seed_currents_A)

    print("[INFO] fit_simplified_dn_corridor started")
    print(f"[INFO] target={Path(target_path).resolve()}")
    print(f"[INFO] seed={Path(seed_path).resolve()}")
    print(f"[INFO] dxf={dxf_path or 'cfg/auto'}")
    print(f"[INFO] stage={stage}")
    print(f"[INFO] free_keys={free_keys}")
    print(f"[INFO] fixed_keys={fixed_keys}")
    print(f"[INFO] workers={workers} iters={iters} pop={pop} timeout={timeout_s:.1f}s seed={seed}")
    print(f"[INFO] corridor_weight={corridor_weight}")
    print(f"[INFO] corridor_sigma_m={corridor_sigma_m}")
    print(f"[INFO] corridor_roi_pad_m={corridor_roi_pad_m}")
    print(f"[INFO] physics={physics}")
    print(f"[INFO] initial currents [MA]: {base._fmt_currents_MA(seed_currents_A)}")

    # Evaluate seed.
    print("\n[SEED] evaluating baseline...")
    seed_res = eval_case_corridor(
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
        corridor_weight=corridor_weight,
        corridor_sigma_m=corridor_sigma_m,
        corridor_roi_pad_m=corridor_roi_pad_m,
        missing_corridor_penalty=missing_corridor_penalty,
    )

    base._append_jsonl(log_jsonl, {
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
        f"A={base._safe_float(si.get('A', np.nan)):.3f} "
        f"k={base._safe_float(si.get('kappa', np.nan)):.3f} "
        f"d={base._safe_float(si.get('delta_bar', np.nan)):.3f} "
        f"legU={base._safe_float(si.get('leg_upper_distance_m', np.nan)):.3f} "
        f"legL={base._safe_float(si.get('leg_lower_distance_m', np.nan)):.3f} "
        f"CS={base._currents_A_to_MA(best_currents_A).get('CS', 0.0):+.3f} MA"
    )

    for it in range(1, int(iters) + 1):
        bounds_MA = base._make_bounds_MA(
            best_currents_A,
            target,
            free_keys,
            stage=stage,
            widen=widen,
        )
        sigma_MA = base._sigma_from_bounds_MA(bounds_MA, sigma_frac)

        batch: List[Dict[str, float]] = []

        # Always include current best.
        batch.append(dict(best_currents_A))

        n_global = max(2, int(pop // 5))
        n_local = max(0, int(pop) - n_global - 1)

        for _ in range(n_global):
            batch.append(base._sample_uniform_A(best_currents_A, bounds_MA, free_keys))

        for _ in range(n_local):
            batch.append(base._sample_gauss_A(best_currents_A, sigma_MA, bounds_MA, free_keys))

        batch = batch[: int(pop)]

        batch_best: Optional[Dict[str, Any]] = None
        batch_best_score = float("inf")
        ok_solve = 0
        ok_sep = 0

        t0 = time.time()

        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
            futs = {}

            for curr in batch:
                curr2 = base._clip_to_bounds_A(curr, bounds_MA, free_keys)

                fut = ex.submit(
                    eval_case_corridor,
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
                    corridor_weight=corridor_weight,
                    corridor_sigma_m=corridor_sigma_m,
                    corridor_roi_pad_m=corridor_roi_pad_m,
                    missing_corridor_penalty=missing_corridor_penalty,
                )

                futs[fut] = curr2

            for fut in as_completed(futs):
                res = fut.result()
                sc = float(res.get("score", 1e18))

                if bool(res.get("ok_solve", False)):
                    ok_solve += 1

                si_eval = res.get("score_info", {})
                if isinstance(si_eval, dict) and bool(si_eval.get("has_true_sep", False)):
                    ok_sep += 1

                if sc < batch_best_score:
                    batch_best_score = sc
                    batch_best = res

                base._append_jsonl(log_jsonl, {
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

            base._save_json(best_json, {
                "schema": "fit_simplified_dn_corridor_best.v1",
                "stage": stage,
                "target_path": str(Path(target_path).resolve()),
                "seed_path": str(Path(seed_path).resolve()),
                "dxf_path": dxf_path,
                "best_score": float(best_score),
                "best_currents_A": best_currents_A,
                "best_currents_MA": base._currents_A_to_MA(best_currents_A),
                "best_result": best,
                "free_keys": free_keys,
                "fixed_keys": fixed_keys,
                "physics": physics,
                "corridor_settings": {
                    "corridor_weight": float(corridor_weight),
                    "corridor_sigma_m": float(corridor_sigma_m),
                    "corridor_roi_pad_m": float(corridor_roi_pad_m),
                    "missing_corridor_penalty": float(missing_corridor_penalty),
                },
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
            f"| A={base._safe_float(si_best.get('A', np.nan)):.3f} "
            f"k={base._safe_float(si_best.get('kappa', np.nan)):.3f} "
            f"d={base._safe_float(si_best.get('delta_bar', np.nan)):.3f} "
            f"legU={base._safe_float(si_best.get('leg_upper_distance_m', np.nan)):.3f} "
            f"legL={base._safe_float(si_best.get('leg_lower_distance_m', np.nan)):.3f} "
            f"Rax={base._safe_float(si_best.get('Rax', np.nan)):.3f} "
            f"CS={base._currents_A_to_MA(best_currents_A).get('CS', 0.0):+.3f} MA "
            f"| {dt:.1f}s"
        )

        if (not no_early_stop) and bool(si_best.get("has_true_sep", False)):
            # Stricter than the old topological early stop.
            if (
                float(si_best.get("leg_upper_distance_m", 1e9)) < 0.18
                and float(si_best.get("leg_lower_distance_m", 1e9)) < 0.18
                and float(si_best.get("shape_term", 1e9)) < 20.0
                and best_score < 500.0
            ):
                print("[STOP] corridor early criterion reached.")
                break

    out = {
        "schema": "fit_simplified_dn_corridor_run.v1",
        "stage": stage,
        "target_path": str(Path(target_path).resolve()),
        "seed_path": str(Path(seed_path).resolve()),
        "dxf_path": dxf_path,
        "best_score": float(best_score),
        "best_currents_A": best_currents_A,
        "best_currents_MA": base._currents_A_to_MA(best_currents_A),
        "best_result": best,
        "free_keys": free_keys,
        "fixed_keys": fixed_keys,
        "physics": physics,
        "corridor_settings": {
            "corridor_weight": float(corridor_weight),
            "corridor_sigma_m": float(corridor_sigma_m),
            "corridor_roi_pad_m": float(corridor_roi_pad_m),
            "missing_corridor_penalty": float(missing_corridor_penalty),
        },
        "log_jsonl": str(log_jsonl),
        "best_json": str(best_json),
    }

    base._save_json(run_json, out)
    base._save_json(best_json, out)

    print("\n[OK] finished")
    print(f"[OK] best score: {best_score:.6g}")
    print(f"[OK] best currents [MA]: {base._fmt_currents_MA(best_currents_A)}")
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
        default=str(base._here() / "results" / "targets" / "star_simplified_dn_target.json"),
        help="Path to star_simplified_dn_target.json",
    )
    ap.add_argument(
        "--seed",
        type=str,
        default=str(base._here() / "results" / "fit_simplified_dn_toposafe_best.json"),
        help="Path to seed currents JSON",
    )
    ap.add_argument(
        "--dxf",
        type=str,
        default=None,
        help="DXF path. If omitted, star_equilibrium uses cfg/default.",
    )

    ap.add_argument(
        "--stage",
        type=str,
        default="pf1-pf",
        choices=["pf-only", "pf1-pf", "release-cs"],
        help="Fit stage.",
    )

    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--seed-rng", type=int, default=11)

    ap.add_argument("--widen", type=float, default=0.18)
    ap.add_argument("--sigma-frac", type=float, default=0.20)

    ap.add_argument(
        "--free-keys",
        type=str,
        default=None,
        help="Optional override, e.g. PF2,PF3,PF4,PF5,PF6",
    )

    ap.add_argument("--no-coarse", action="store_true")
    ap.add_argument("--show-solver", action="store_true")
    ap.add_argument("--no-early-stop", action="store_true")

    # New corridor knobs.
    ap.add_argument(
        "--corridor-weight",
        type=float,
        default=180.0,
        help="Weight multiplying leg-corridor distance term.",
    )
    ap.add_argument(
        "--corridor-sigma",
        type=float,
        default=0.18,
        help="Distance scale [m] for corridor penalty.",
    )
    ap.add_argument(
        "--corridor-roi-pad",
        type=float,
        default=0.70,
        help="ROI padding [m] around corridor windows.",
    )
    ap.add_argument(
        "--missing-corridor-penalty",
        type=float,
        default=5.0e5,
        help="Penalty if LEG_UPPER_WIN/LEG_LOWER_WIN are missing or unusable.",
    )

    return ap.parse_args()


def main() -> None:
    mp.freeze_support()

    args = parse_args()

    free_override = base._parse_keys_csv(args.free_keys) if args.free_keys else None

    fit_corridor(
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
        corridor_weight=float(args.corridor_weight),
        corridor_sigma_m=float(args.corridor_sigma),
        corridor_roi_pad_m=float(args.corridor_roi_pad),
        missing_corridor_penalty=float(args.missing_corridor_penalty),
    )


if __name__ == "__main__":
    main()
