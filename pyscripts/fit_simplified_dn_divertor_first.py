#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fit_simplified_dn_divertor_first.py

Divertor-first / topology-first optimizer for STAR-like DN equilibria.

Purpose
-------
This optimizer is meant to escape the local branch where the shape objective is
mostly satisfied, but the inboard divertor strikes do not land where desired.

It prioritizes:

    1. true separatrix
    2. strike-point proximity:
         STRIKE_U_IN, STRIKE_U_OUT, STRIKE_L_IN, STRIKE_L_OUT
    3. optional X-point windows:
         XPT_UPPER_WIN, XPT_LOWER_WIN
    4. optional leg corridor windows:
         LEG_UPPER_WIN, LEG_LOWER_WIN
    5. soft confinement / bad-wall penalties
    6. weak shape regularization

Important
---------
By default, wall/leak/chamfer is a SOFT penalty, not a hard rejection. This is
intentional because the current STAR-like branch has not achieved perfect
inner-wall containment yet.

Expected dependencies
---------------------
This script assumes your existing scripts expose the same helpers previously used:

    fit_simplified_dn_toposafe.py
    fit_simplified_dn_corridor.py

Specifically, it uses:
    - fit_simplified_dn_toposafe.FAMILIES_ALL
    - fit_simplified_dn_toposafe._load_json
    - fit_simplified_dn_toposafe._load_seed_currents
    - fit_simplified_dn_toposafe._resolve_path_maybe
    - fit_simplified_dn_corridor.eval_case_corridor

If your eval result does not contain a separatrix/LCFS curve array, this script
will still run, but it will not be able to compute strike distances and will
assign a large missing-strike penalty. In that case, add sep_xy/separatrix_xy
to the returned result inside your existing evaluator/analyzer.

Recommended first run
---------------------
py .\\fit_simplified_dn_divertor_first.py --target .\\results\\targets\\star_simplified_dn_target.json --seed .\\results\\gap_v2_pf6_6p5_pf5_soft_release_best.json --stage release-cs --iters 24 --pop 22 --workers 6 --timeout 180 --widen 0.08 --sigma-frac 0.10 --free-keys CS,PF1,PF2,PF3,PF4,PF5 --dxf .\\cad\\star_baseline.dxf
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

import fit_simplified_dn_toposafe as base
import fit_simplified_dn_corridor as corridor


FAMILIES_ALL = list(base.FAMILIES_ALL)


# -------------------------------------------------------------------------
# JSON / path helpers
# -------------------------------------------------------------------------
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


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dumps_safe(obj, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(_json_dumps_safe(obj) + "\n")


def _results_dir() -> Path:
    d = Path(__file__).resolve().parent / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


# -------------------------------------------------------------------------
# Geometry/window extraction
# -------------------------------------------------------------------------
def _as_xy_array(obj: Any) -> Optional[np.ndarray]:
    """
    Convert common window formats to Nx2 float array.

    Supported examples:
        [[R,Z], [R,Z], ...]
        {"xy": [[R,Z], ...]}
        {"points": [[R,Z], ...]}
        {"polyline": [[R,Z], ...]}
        {"segments": ...} is not expanded here.
    """
    if obj is None:
        return None

    if isinstance(obj, dict):
        for k in ("xy", "points", "polyline", "coords", "vertices"):
            if k in obj:
                arr = _as_xy_array(obj[k])
                if arr is not None:
                    return arr
        return None

    try:
        arr = np.asarray(obj, dtype=float)
    except Exception:
        return None

    if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 1:
        return arr[:, :2].astype(float)

    return None


def _recursive_find_windows(obj: Any, out: Dict[str, np.ndarray]) -> None:
    """
    Recursively find dict entries whose names look like window/layer names and
    values look like XY arrays.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            ku = str(k).upper()

            arr = _as_xy_array(v)
            if arr is not None and arr.shape[0] >= 1:
                if (
                    "STRIKE" in ku
                    or "XPT" in ku
                    or "LEG" in ku
                    or ku.startswith("WIN:")
                    or ku.endswith("_WIN")
                ):
                    clean = ku.replace("WIN:", "").replace("WINDOW:", "")
                    out[clean] = arr

            _recursive_find_windows(v, out)

    elif isinstance(obj, list):
        for v in obj:
            _recursive_find_windows(v, out)


def _load_windows(target: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Load target/CAD marker windows already stored in the target JSON.

    This is intentionally JSON-first. If make_simplified_dn_target.py stores CAD
    marker windows in results/targets/star_simplified_dn_target.json, this will
    pick them up.
    """
    windows: Dict[str, np.ndarray] = {}
    _recursive_find_windows(target, windows)

    # Normalize key variants.
    aliases = {}
    for k in list(windows.keys()):
        kk = k.upper()
        aliases[kk] = windows[k]

    return aliases


def _dist_point_to_polyline(p: np.ndarray, poly: np.ndarray) -> float:
    """
    Minimum Euclidean distance from point p to a polyline.
    """
    p = np.asarray(p, dtype=float)[:2]
    poly = np.asarray(poly, dtype=float)[:, :2]

    if poly.shape[0] == 1:
        return float(np.linalg.norm(p - poly[0]))

    best = float("inf")
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a
        den = float(np.dot(ab, ab))
        if den <= 1e-18:
            d = float(np.linalg.norm(p - a))
        else:
            t = float(np.dot(p - a, ab) / den)
            t = max(0.0, min(1.0, t))
            q = a + t * ab
            d = float(np.linalg.norm(p - q))
        if d < best:
            best = d
    return best


def _min_dist_curve_to_window(curve: np.ndarray, win: np.ndarray) -> Tuple[float, Optional[List[float]]]:
    """
    Minimum distance between a curve point cloud and a window polyline.
    Returns distance and the nearest curve point.
    """
    if curve is None or win is None:
        return float("inf"), None

    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 2 or curve.shape[1] < 2 or curve.shape[0] < 1:
        return float("inf"), None

    best = float("inf")
    best_pt = None

    for p in curve[:, :2]:
        d = _dist_point_to_polyline(p, win)
        if d < best:
            best = d
            best_pt = [float(p[0]), float(p[1])]

    return float(best), best_pt


# -------------------------------------------------------------------------
# Result extraction
# -------------------------------------------------------------------------
def _recursive_find_xy_arrays(obj: Any, names: Iterable[str], out: List[np.ndarray]) -> None:
    names_u = [n.upper() for n in names]

    if isinstance(obj, dict):
        for k, v in obj.items():
            ku = str(k).upper()
            if any(n in ku for n in names_u):
                arr = _as_xy_array(v)
                if arr is not None and arr.shape[0] >= 4:
                    out.append(arr)

            _recursive_find_xy_arrays(v, names, out)

    elif isinstance(obj, list):
        for v in obj:
            _recursive_find_xy_arrays(v, names, out)


def _extract_separatrix_xy(result: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Try to find a separatrix/LCFS curve stored inside the evaluator result.

    If this returns None, you likely need to modify your existing analyzer/eval
    to include something like:
        result["separatrix_xy"] = sep_xy.tolist()
    """
    candidates: List[np.ndarray] = []
    _recursive_find_xy_arrays(
        result,
        names=[
            "separatrix",
            "sep_xy",
            "sep_curve",
            "lcfs",
            "boundary_curve",
            "plasma_boundary",
            "rebuilt_sep",
        ],
        out=candidates,
    )

    if not candidates:
        return None

    # Prefer longer curves.
    candidates.sort(key=lambda a: a.shape[0], reverse=True)
    arr = candidates[0][:, :2].astype(float)

    # Remove bad rows.
    mask = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    arr = arr[mask]

    if arr.shape[0] < 4:
        return None

    return arr


def _extract_xpoints(result: Dict[str, Any]) -> List[List[float]]:
    """
    Try to find X-point coordinates in common result/score_info formats.
    """
    xpts: List[List[float]] = []

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                ku = str(k).upper()
                if "XPOINT" in ku or "X_POINT" in ku or ku == "XPOINTS" or ku == "X_POINTS":
                    arr = _as_xy_array(v)
                    if arr is not None:
                        for p in arr:
                            xpts.append([float(p[0]), float(p[1])])
                visit(v)
        elif isinstance(obj, list):
            for v in obj:
                visit(v)

    visit(result)

    # Dedupe approximately.
    clean: List[List[float]] = []
    for p in xpts:
        if not np.isfinite(p[0]) or not np.isfinite(p[1]):
            continue
        if all(np.linalg.norm(np.asarray(p) - np.asarray(q)) > 1e-3 for q in clean):
            clean.append(p)

    return clean


# -------------------------------------------------------------------------
# Divertor-first scoring
# -------------------------------------------------------------------------
def _score_strikes(
    result: Dict[str, Any],
    windows: Dict[str, np.ndarray],
    *,
    strike_sigma_m: float,
    missing_penalty: float,
    w_in: float,
    w_out: float,
    max_weight: float,
) -> Dict[str, Any]:
    """
    Strike score based on nearest distance from separatrix to each strike window.

    This is a robust first proxy. Later, you can replace it with true branch-wall
    intersection detection.
    """
    sep = _extract_separatrix_xy(result)

    names = ["STRIKE_U_IN", "STRIKE_U_OUT", "STRIKE_L_IN", "STRIKE_L_OUT"]
    weights = {
        "STRIKE_U_IN": float(w_in),
        "STRIKE_L_IN": float(w_in),
        "STRIKE_U_OUT": float(w_out),
        "STRIKE_L_OUT": float(w_out),
    }

    distances: Dict[str, float] = {}
    nearest: Dict[str, Optional[List[float]]] = {}
    terms: List[float] = []
    missing: List[str] = []

    if sep is None:
        return {
            "strike_score": float(missing_penalty),
            "strike_distances_m": {n: float("inf") for n in names},
            "strike_nearest_points": {n: None for n in names},
            "strike_missing": names,
            "has_separatrix_curve_for_strikes": False,
        }

    for n in names:
        if n not in windows:
            missing.append(n)
            distances[n] = float("inf")
            nearest[n] = None
            terms.append(float(missing_penalty))
            continue

        d, pt = _min_dist_curve_to_window(sep, windows[n])
        distances[n] = float(d)
        nearest[n] = pt

        # Dimensionless weighted distance.
        wd = weights[n] * (d / max(strike_sigma_m, 1e-6)) ** 2
        terms.append(float(wd))

    finite_terms = [t for t in terms if np.isfinite(t)]
    finite_dist = [d for d in distances.values() if np.isfinite(d)]

    if not finite_terms:
        strike_score = float(missing_penalty)
    else:
        strike_score = float(sum(finite_terms) + max_weight * max(finite_terms))

    if missing:
        strike_score += float(missing_penalty) * len(missing)

    return {
        "strike_score": float(strike_score),
        "strike_distances_m": distances,
        "strike_nearest_points": nearest,
        "strike_missing": missing,
        "strike_max_distance_m": float(max(finite_dist)) if finite_dist else float("inf"),
        "has_separatrix_curve_for_strikes": True,
    }


def _score_xpoints(
    result: Dict[str, Any],
    windows: Dict[str, np.ndarray],
    *,
    xpt_sigma_m: float,
    weight: float,
) -> Dict[str, Any]:
    """
    Soft score for X-points near broad XPT windows.
    """
    xpts = _extract_xpoints(result)

    required = [
        ("XPT_UPPER_WIN", +1),
        ("XPT_LOWER_WIN", -1),
    ]

    out: Dict[str, Any] = {
        "xpoint_score": 0.0,
        "xpoint_distances_m": {},
        "xpoints_found": xpts,
    }

    if not xpts:
        out["xpoint_score"] = 1.0e5 * float(weight)
        return out

    score = 0.0

    for name, sign in required:
        if name not in windows:
            continue

        candidates = []
        for p in xpts:
            if sign > 0 and p[1] <= 0:
                continue
            if sign < 0 and p[1] >= 0:
                continue
            d = _dist_point_to_polyline(np.asarray(p), windows[name])
            candidates.append(d)

        if candidates:
            dmin = float(min(candidates))
            score += float(weight) * (dmin / max(xpt_sigma_m, 1e-6)) ** 2
            out["xpoint_distances_m"][name] = dmin
        else:
            score += 1.0e4 * float(weight)
            out["xpoint_distances_m"][name] = float("inf")

    out["xpoint_score"] = float(score)
    return out


def _score_legs(
    result: Dict[str, Any],
    windows: Dict[str, np.ndarray],
    *,
    leg_sigma_m: float,
    weight: float,
) -> Dict[str, Any]:
    """
    Optional secondary score for LEG_UPPER_WIN / LEG_LOWER_WIN.
    """
    sep = _extract_separatrix_xy(result)

    out: Dict[str, Any] = {
        "leg_score": 0.0,
        "leg_distances_m": {},
    }

    if sep is None:
        out["leg_score"] = 1.0e4 * float(weight)
        return out

    score = 0.0
    for name in ["LEG_UPPER_WIN", "LEG_LOWER_WIN"]:
        if name not in windows:
            continue
        d, _ = _min_dist_curve_to_window(sep, windows[name])
        out["leg_distances_m"][name] = float(d)
        score += float(weight) * (d / max(leg_sigma_m, 1e-6)) ** 2

    out["leg_score"] = float(score)
    return out


def _score_shape_soft(
    result: Dict[str, Any],
    *,
    weight: float,
    min_A_soft: float,
    min_kappa_soft: float,
    max_chamfer_soft: float,
) -> Dict[str, Any]:
    """
    Soft shape/confinement score. This intentionally does not hard-reject bad wall
    unless --hard-wall-gate is used outside.
    """
    si = result.get("score_info", {})
    if not isinstance(si, dict):
        si = {}

    A = _safe_float(si.get("A"))
    kappa = _safe_float(si.get("kappa"))
    delta = _safe_float(si.get("delta_bar", si.get("delta")))
    chamfer = _safe_float(si.get("boundary_chamfer_m", si.get("boundary_chamfer")))

    score = 0.0

    if not np.isfinite(A):
        score += 1.0e4
    elif A < min_A_soft:
        score += ((min_A_soft - A) / 0.15) ** 2 * 200.0

    if not np.isfinite(kappa):
        score += 1.0e4
    elif kappa < min_kappa_soft:
        score += ((min_kappa_soft - kappa) / 0.20) ** 2 * 200.0

    if not np.isfinite(chamfer):
        score += 1.0e4
    elif chamfer > max_chamfer_soft:
        score += ((chamfer - max_chamfer_soft) / 0.15) ** 2 * 250.0

    # Weak preference for not collapsing triangularity, but do not overconstrain.
    if np.isfinite(delta) and delta < 0.20:
        score += ((0.20 - delta) / 0.10) ** 2 * 50.0

    return {
        "shape_soft_score": float(weight) * float(score),
        "A": A,
        "kappa": kappa,
        "delta_bar": delta,
        "boundary_chamfer_m": chamfer,
    }


def _score_current_regularization(
    curr_A: Dict[str, float],
    ref_A: Dict[str, float],
    *,
    weight: float,
) -> Dict[str, Any]:
    """
    Weak current regularization to avoid extreme jumps.
    """
    terms = []
    for k in FAMILIES_ALL:
        c = _safe_float(curr_A.get(k, 0.0)) / 1e6
        r = _safe_float(ref_A.get(k, 0.0)) / 1e6
        scale = max(1.0, abs(r), abs(c))
        terms.append(((c - r) / scale) ** 2)

    val = float(weight) * float(sum(terms))
    return {"current_reg_score": val}


def compute_divertor_first_score(
    result: Dict[str, Any],
    curr_A: Dict[str, float],
    ref_A: Dict[str, float],
    windows: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Tuple[float, Dict[str, Any]]:
    """
    Main divertor-first score.
    """
    si = result.get("score_info", {})
    if not isinstance(si, dict):
        si = {}

    has_true_sep = bool(si.get("has_true_sep", si.get("ok_sep", False)))
    shape_reason = str(si.get("shape_reason", "")).lower()

    # Hard no-separatrix penalty. This remains hard because without a true
    # separatrix the strike-point objective is not meaningful.
    no_sep_penalty = 0.0
    if not has_true_sep or shape_reason not in ("", "ok"):
        no_sep_penalty = float(args.penalty_no_sep)

    strike = _score_strikes(
        result,
        windows,
        strike_sigma_m=float(args.strike_sigma),
        missing_penalty=float(args.missing_strike_penalty),
        w_in=float(args.strike_in_weight),
        w_out=float(args.strike_out_weight),
        max_weight=float(args.strike_max_weight),
    )

    xpt = _score_xpoints(
        result,
        windows,
        xpt_sigma_m=float(args.xpt_sigma),
        weight=float(args.xpt_weight),
    )

    leg = _score_legs(
        result,
        windows,
        leg_sigma_m=float(args.leg_sigma),
        weight=float(args.leg_weight),
    )

    shape = _score_shape_soft(
        result,
        weight=float(args.shape_weight),
        min_A_soft=float(args.min_A_soft),
        min_kappa_soft=float(args.min_kappa_soft),
        max_chamfer_soft=float(args.max_chamfer_soft),
    )

    curr_reg = _score_current_regularization(
        curr_A,
        ref_A,
        weight=float(args.current_reg_weight),
    )

    hard_wall_penalty = 0.0
    if bool(args.hard_wall_gate):
        ch = shape.get("boundary_chamfer_m", float("nan"))
        if not np.isfinite(ch) or ch > float(args.hard_wall_chamfer):
            hard_wall_penalty = float(args.penalty_hard_wall)

    total = (
        float(no_sep_penalty)
        + float(args.strike_weight) * float(strike["strike_score"])
        + float(xpt["xpoint_score"])
        + float(leg["leg_score"])
        + float(shape["shape_soft_score"])
        + float(curr_reg["current_reg_score"])
        + float(hard_wall_penalty)
    )

    info = {
        "divertor_first_score": float(total),
        "no_sep_penalty": float(no_sep_penalty),
        "hard_wall_penalty": float(hard_wall_penalty),
        "has_true_sep": bool(has_true_sep),
        "shape_reason": shape_reason,
        **strike,
        **xpt,
        **leg,
        **shape,
        **curr_reg,
    }

    return float(total), info


# -------------------------------------------------------------------------
# Candidate generation
# -------------------------------------------------------------------------
def _currents_A_to_MA(curr_A: Dict[str, float]) -> Dict[str, float]:
    return {k: _safe_float(curr_A.get(k, 0.0)) / 1e6 for k in FAMILIES_ALL}


def _currents_MA_to_A(curr_MA: Dict[str, float]) -> Dict[str, float]:
    return {k: _safe_float(curr_MA.get(k, 0.0)) * 1e6 for k in FAMILIES_ALL}


def _parse_free_keys(s: str) -> List[str]:
    if not s:
        return list(FAMILIES_ALL)
    keys = [x.strip().upper() for x in s.split(",") if x.strip()]
    bad = [k for k in keys if k not in FAMILIES_ALL]
    if bad:
        raise ValueError(f"Unknown free keys: {bad}. Valid: {FAMILIES_ALL}")
    return keys


def _clip_currents_MA(curr: Dict[str, float], args: argparse.Namespace) -> Dict[str, float]:
    bounds = {
        "CS": (args.cs_min, args.cs_max),
        "PF1": (args.pf1_min, args.pf1_max),
        "PF2": (args.pf2_min, args.pf2_max),
        "PF3": (args.pf3_min, args.pf3_max),
        "PF4": (args.pf4_min, args.pf4_max),
        "PF5": (args.pf5_min, args.pf5_max),
        "PF6": (args.pf6_min, args.pf6_max),
    }
    out = {}
    for k in FAMILIES_ALL:
        lo, hi = bounds[k]
        out[k] = float(np.clip(_safe_float(curr.get(k, 0.0)), lo, hi))
    return out


def _sample_candidate_A(
    center_A: Dict[str, float],
    seed_A: Dict[str, float],
    free_keys: List[str],
    args: argparse.Namespace,
) -> Dict[str, float]:
    center_MA = _currents_A_to_MA(center_A)
    cand_MA = dict(center_MA)

    for k in free_keys:
        c = center_MA.get(k, 0.0)
        s = max(float(args.widen), abs(c) * float(args.sigma_frac))
        cand_MA[k] = random.gauss(c, s)

    cand_MA = _clip_currents_MA(cand_MA, args)

    # Preserve fixed keys exactly from center.
    for k in FAMILIES_ALL:
        if k not in free_keys:
            cand_MA[k] = center_MA.get(k, _safe_float(seed_A.get(k, 0.0)) / 1e6)

    return _currents_MA_to_A(cand_MA)


# -------------------------------------------------------------------------
# Evaluation wrapper
# -------------------------------------------------------------------------
def _evaluate_one(
    curr_A: Dict[str, float],
    *,
    target: Dict[str, Any],
    windows: Dict[str, np.ndarray],
    dxf_path: str,
    cfg_overrides: Dict[str, Any],
    physics: Dict[str, Any],
    seed_A: Dict[str, float],
    ref_A: Dict[str, float],
    free_keys: List[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:

    try:
        res = corridor.eval_case_corridor(
            curr_A,
            target=target,
            dxf_path=dxf_path,
            cfg_overrides=cfg_overrides,
            physics=physics,
            ref_currents_A=ref_A,
            seed_currents_A=seed_A,
            free_keys=free_keys,
            stage=str(args.stage),
            timeout_s=float(args.timeout),
            redirect_solver_noise=True,
            corridor_weight=0.0,
            corridor_sigma_m=float(args.leg_sigma),
            corridor_roi_pad_m=1.0,
            missing_corridor_penalty=0.0,
        )
    except Exception as e:
        curr_MA = _currents_A_to_MA(curr_A)
        return {
            "ok": False,
            "ok_solve": False,
            "score": 1.0e18,
            "error": repr(e),
            "currents_A": curr_A,
            "currents_MA": curr_MA,
            "score_info": {
                "has_true_sep": False,
                "shape_reason": "exception",
            },
            "divertor_first_info": {
                "divertor_first_score": 1.0e18,
                "exception": repr(e),
            },
        }

    if not isinstance(res, dict):
        res = {"raw_result": repr(res)}

    res.setdefault("currents_A", curr_A)
    res.setdefault("currents_MA", _currents_A_to_MA(curr_A))

    score, info = compute_divertor_first_score(
        res,
        curr_A=curr_A,
        ref_A=ref_A,
        windows=windows,
        args=args,
    )

    res["divertor_first_score"] = float(score)
    res["divertor_first_info"] = info
    res["score"] = float(score)

    if "score_info" not in res or not isinstance(res["score_info"], dict):
        res["score_info"] = {}

    res["score_info"]["divertor_first_score"] = float(score)
    res["score_info"]["strike_distances_m"] = info.get("strike_distances_m", {})
    res["score_info"]["strike_max_distance_m"] = info.get("strike_max_distance_m")
    res["score_info"]["xpoint_distances_m"] = info.get("xpoint_distances_m", {})
    res["score_info"]["leg_distances_m"] = info.get("leg_distances_m", {})

    return res


# -------------------------------------------------------------------------
# Main optimization loop
# -------------------------------------------------------------------------
def run(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(int(args.rng_seed))
    np.random.seed(int(args.rng_seed))

    target_path = str(args.target)
    seed_path = str(args.seed)
    dxf_path = base._resolve_path_maybe(args.dxf)

    target = base._load_json(target_path)
    seed_A, physics = base._load_seed_currents(seed_path)
    ref_A = dict(seed_A)

    windows = _load_windows(target)

    required = ["STRIKE_U_IN", "STRIKE_U_OUT", "STRIKE_L_IN", "STRIKE_L_OUT"]
    missing_required = [k for k in required if k not in windows]

    print("[INFO] fit_simplified_dn_divertor_first started")
    print(f"[INFO] target={Path(target_path).resolve()}")
    print(f"[INFO] seed={Path(seed_path).resolve()}")
    print(f"[INFO] dxf={dxf_path}")
    print(f"[INFO] free_keys={_parse_free_keys(args.free_keys)}")
    print(f"[INFO] physics={physics}")
    print(f"[INFO] initial currents [MA]={_currents_A_to_MA(seed_A)}")
    print(f"[INFO] detected windows={sorted(windows.keys())}")

    if missing_required:
        print(f"[WARN] missing required strike windows in target JSON: {missing_required}")
        print("[WARN] If these are in DXF but not target JSON, regenerate target JSON or update star_machine_cad.py/make_simplified_dn_target.py to export marker_windows.")

    cfg_overrides: Dict[str, Any] = {}
    if not bool(args.no_coarse):
        cfg_overrides.update({
            "nx_eq": int(args.nx),
            "ny_eq": int(args.ny),
            "blanket_enabled": False,
            "blanket_n_filaments": 0,
            "target_rel_tol_ramp": float(args.target_rel_tol_ramp),
            "target_rel_tol": float(args.target_rel_tol),
            "f_list_equilibrium": (0.10, 0.25, 0.45, 0.70, 1.00),
        })

    free_keys = _parse_free_keys(args.free_keys)

    out_dir = _results_dir()
    run_tag = time.strftime("%Y%m%d_%H%M%S")

    best_json = out_dir / "fit_simplified_dn_divertor_first_best.json"
    run_json = out_dir / f"fit_simplified_dn_divertor_first_run_{run_tag}.json"
    log_jsonl = out_dir / "fit_simplified_dn_divertor_first_results.jsonl"

    # Seed evaluation.
    print("\n[SEED] evaluating baseline...")
    seed_res = _evaluate_one(
        seed_A,
        target=target,
        windows=windows,
        dxf_path=dxf_path,
        cfg_overrides=cfg_overrides,
        physics=physics,
        seed_A=seed_A,
        ref_A=ref_A,
        free_keys=free_keys,
        args=args,
    )

    best_res = seed_res
    best_score = float(seed_res.get("divertor_first_score", seed_res.get("score", 1.0e18)))
    best_A = _safe_float(seed_res.get("divertor_first_info", {}).get("A"))
    best_k = _safe_float(seed_res.get("divertor_first_info", {}).get("kappa"))
    best_d = _safe_float(seed_res.get("divertor_first_info", {}).get("delta_bar"))
    best_ch = _safe_float(seed_res.get("divertor_first_info", {}).get("boundary_chamfer_m"))

    print(
        f"[SEED] score={best_score:.6g} "
        f"A={best_A:.3f} k={best_k:.3f} d={best_d:.3f} ch={best_ch:.3f} "
        f"sep={seed_res.get('divertor_first_info', {}).get('has_true_sep')}"
    )

    _append_jsonl(log_jsonl, {"kind": "seed", "result": seed_res})

    current_center_A = dict(seed_A)
    t0 = time.time()

    for it in range(1, int(args.iters) + 1):
        batch: List[Dict[str, float]] = []

        # Include current best as a stabilizing candidate every iteration.
        batch.append(dict(best_res.get("currents_A", current_center_A)))

        while len(batch) < int(args.pop):
            batch.append(_sample_candidate_A(current_center_A, seed_A, free_keys, args))

        iter_results: List[Dict[str, Any]] = []
        iter_t0 = time.time()

        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            futs = [
                ex.submit(
                    _evaluate_one,
                    cand,
                    target=target,
                    windows=windows,
                    dxf_path=dxf_path,
                    cfg_overrides=cfg_overrides,
                    physics=physics,
                    seed_A=seed_A,
                    ref_A=ref_A,
                    free_keys=free_keys,
                    args=args,
                )
                for cand in batch
            ]

            for fut in as_completed(futs):
                res = fut.result()
                iter_results.append(res)
                _append_jsonl(log_jsonl, {"kind": "eval", "iteration": it, "result": res})

        iter_results.sort(key=lambda r: float(r.get("divertor_first_score", r.get("score", 1.0e18))))
        batch_best = iter_results[0]
        batch_score = float(batch_best.get("divertor_first_score", batch_best.get("score", 1.0e18)))

        improved = batch_score < best_score

        if improved:
            best_res = batch_best
            best_score = batch_score
            current_center_A = dict(batch_best.get("currents_A", current_center_A))

            _save_json(best_json, {
                "schema": "fit_simplified_dn_divertor_first_best.v1",
                "best_score": best_score,
                "best_currents_A": best_res.get("currents_A", {}),
                "best_currents_MA": best_res.get("currents_MA", {}),
                "best_result": best_res,
                "physics": physics,
                "target_path": str(Path(target_path).resolve()),
                "seed_path": str(Path(seed_path).resolve()),
                "dxf_path": dxf_path,
                "free_keys": free_keys,
                "windows_available": sorted(windows.keys()),
                "log_jsonl": str(log_jsonl),
            })

        info = best_res.get("divertor_first_info", {})
        strike_dist = info.get("strike_distances_m", {})
        max_strike = info.get("strike_max_distance_m", float("nan"))

        ok_solve = sum(bool(r.get("ok_solve", r.get("ok", False))) for r in iter_results)
        sep_count = sum(bool(r.get("divertor_first_info", {}).get("has_true_sep", False)) for r in iter_results)

        print(
            f"[ITER {it:03d}] "
            f"batch_best={batch_score:.6g} global_best={best_score:.6g} "
            f"{'IMPROVED' if improved else '        '} | "
            f"ok={ok_solve}/{len(iter_results)} sep={sep_count}/{len(iter_results)} | "
            f"A={_safe_float(info.get('A')):.3f} "
            f"k={_safe_float(info.get('kappa')):.3f} "
            f"d={_safe_float(info.get('delta_bar')):.3f} "
            f"ch={_safe_float(info.get('boundary_chamfer_m')):.3f} "
            f"maxStrike={_safe_float(max_strike):.3f} "
            f"Uin={_safe_float(strike_dist.get('STRIKE_U_IN')):.3f} "
            f"Lin={_safe_float(strike_dist.get('STRIKE_L_IN')):.3f} "
            f"| {time.time() - iter_t0:.1f}s"
        )

    # Ensure final best is saved.
    _save_json(best_json, {
        "schema": "fit_simplified_dn_divertor_first_best.v1",
        "best_score": best_score,
        "best_currents_A": best_res.get("currents_A", {}),
        "best_currents_MA": best_res.get("currents_MA", {}),
        "best_result": best_res,
        "physics": physics,
        "target_path": str(Path(target_path).resolve()),
        "seed_path": str(Path(seed_path).resolve()),
        "dxf_path": dxf_path,
        "free_keys": free_keys,
        "windows_available": sorted(windows.keys()),
        "log_jsonl": str(log_jsonl),
    })

    run_out = {
        "schema": "fit_simplified_dn_divertor_first_run.v1",
        "best_score": best_score,
        "best_currents_A": best_res.get("currents_A", {}),
        "best_currents_MA": best_res.get("currents_MA", {}),
        "best_result": best_res,
        "physics": physics,
        "target_path": str(Path(target_path).resolve()),
        "seed_path": str(Path(seed_path).resolve()),
        "dxf_path": dxf_path,
        "free_keys": free_keys,
        "windows_available": sorted(windows.keys()),
        "best_json": str(best_json),
        "log_jsonl": str(log_jsonl),
        "runtime_s": float(time.time() - t0),
    }

    _save_json(run_json, run_out)

    print("\n[OK] finished")
    print(f"[OK] best score: {best_score:.6g}")
    print(f"[OK] best currents [MA]: {best_res.get('currents_MA', {})}")
    print(f"[OK] best json: {best_json}")
    print(f"[OK] run json : {run_json}")
    print(f"[OK] log jsonl: {log_jsonl}")

    return run_out


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--target", type=str, default=str(_results_dir() / "targets" / "star_simplified_dn_target.json"))
    ap.add_argument("--seed", type=str, required=True)
    ap.add_argument("--dxf", type=str, default=str(Path(__file__).resolve().parent / "cad" / "star_baseline.dxf"))

    ap.add_argument("--stage", type=str, default="release-cs")
    ap.add_argument("--iters", type=int, default=24)
    ap.add_argument("--pop", type=int, default=22)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--rng-seed", type=int, default=31)

    ap.add_argument("--free-keys", type=str, default="CS,PF1,PF2,PF3,PF4,PF5")
    ap.add_argument("--widen", type=float, default=0.08)
    ap.add_argument("--sigma-frac", type=float, default=0.10)

    # Solver/coarse controls.
    ap.add_argument("--no-coarse", action="store_true")
    ap.add_argument("--nx", type=int, default=65)
    ap.add_argument("--ny", type=int, default=129)
    ap.add_argument("--target-rel-tol-ramp", type=float, default=1.0e-4)
    ap.add_argument("--target-rel-tol", type=float, default=5.0e-5)

    # Current bounds [MA].
    ap.add_argument("--cs-min", type=float, default=-5.0)
    ap.add_argument("--cs-max", type=float, default=5.0)

    ap.add_argument("--pf1-min", type=float, default=-8.0)
    ap.add_argument("--pf1-max", type=float, default=8.0)

    ap.add_argument("--pf2-min", type=float, default=-10.0)
    ap.add_argument("--pf2-max", type=float, default=8.0)

    ap.add_argument("--pf3-min", type=float, default=-4.0)
    ap.add_argument("--pf3-max", type=float, default=8.0)

    ap.add_argument("--pf4-min", type=float, default=-2.0)
    ap.add_argument("--pf4-max", type=float, default=10.0)

    ap.add_argument("--pf5-min", type=float, default=-2.0)
    ap.add_argument("--pf5-max", type=float, default=11.0)

    ap.add_argument("--pf6-min", type=float, default=0.0)
    ap.add_argument("--pf6-max", type=float, default=8.0)

    # Divertor-first scoring.
    ap.add_argument("--strike-weight", type=float, default=1.0)
    ap.add_argument("--strike-sigma", type=float, default=0.30)
    ap.add_argument("--strike-in-weight", type=float, default=4.0)
    ap.add_argument("--strike-out-weight", type=float, default=1.0)
    ap.add_argument("--strike-max-weight", type=float, default=4.0)
    ap.add_argument("--missing-strike-penalty", type=float, default=5.0e5)

    # X-point soft guidance.
    ap.add_argument("--xpt-weight", type=float, default=0.20)
    ap.add_argument("--xpt-sigma", type=float, default=0.45)

    # Leg corridor as secondary guidance.
    ap.add_argument("--leg-weight", type=float, default=0.15)
    ap.add_argument("--leg-sigma", type=float, default=0.45)

    # Soft shape/confinement gates.
    ap.add_argument("--shape-weight", type=float, default=0.08)
    ap.add_argument("--min-A-soft", type=float, default=1.35)
    ap.add_argument("--min-kappa-soft", type=float, default=1.70)
    ap.add_argument("--max-chamfer-soft", type=float, default=0.60)

    # Hard wall gate is OFF by default.
    ap.add_argument("--hard-wall-gate", action="store_true")
    ap.add_argument("--hard-wall-chamfer", type=float, default=0.70)
    ap.add_argument("--penalty-hard-wall", type=float, default=5.0e6)

    ap.add_argument("--penalty-no-sep", type=float, default=1.0e7)
    ap.add_argument("--current-reg-weight", type=float, default=0.02)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
