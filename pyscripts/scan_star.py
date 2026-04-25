# scan_star.py
"""
Stochastic multi-goal scan for STAR-like tokamak equilibrium (FreeGSNKE)
-----------------------------------------------------------------------

This is the "leave-it-running-for-days" version: robust, reproducible,
and with the TWO constraints you asked for:

  (1) separatrix must exist (diverted)           -> --require-sep
  (2) LCFS must be fully inside WALL_INNER       -> --enforce-inner (soft/hard)

It DOES NOT use --infer_x anymore. If your CAD has xpoints_target, this script
uses it directly (lower/upper/double via --null-mode).

Search method:
- Two-stage stochastic scan (stage1 coarse global, stage2 narrower refine)
- PRIOR-centered bounds with HARD limits
- Batch evaluation with hard per-case timeout
- Robust JSONL logging + best snapshot files

Outputs (in ./results):
- scan_multigoal_results.jsonl
- scan_multigoal_best.txt
- scan_multigoal_best_batch.txt

Usage example (recommended):
  py .\scan_star.py --workers 6 --timeout 300 --n1 2000 --n2 2000 --pop 64 --prefer_inner --null-mode double --require-sep --enforce-inner --inner-hard

Notes:
- On Windows we use spawn-per-eval hard timeout (slower but stable).
- For long runs, prefer larger --timeout (e.g. 300-450s) and fewer workers if RAM is limited.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from matplotlib.path import Path as MplPath


# ----------------------------
# JSON-safe helpers
# ----------------------------
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
    if isinstance(obj, (set, frozenset)):
        return [_to_builtin(x) for x in obj]
    return obj


def _json_dumps_safe(obj: Any, **kwargs) -> str:
    return json.dumps(_to_builtin(obj), default=str, **kwargs)


# ----------------------------
# Paths
# ----------------------------
def _here() -> Path:
    return Path(__file__).resolve().parent


def _results_dir() -> Path:
    d = _here() / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(_json_dumps_safe(obj) + "\n")


def _save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _ts_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# ----------------------------
# Small utils
# ----------------------------
def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _open_curve_from_closed(R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    P = np.column_stack([np.asarray(R, float), np.asarray(Z, float)])
    if P.shape[0] < 3:
        return P
    if np.linalg.norm(P[0] - P[-1]) < 1e-12:
        P = P[:-1]
    return P


def _extract_lcfs_curve(shape: Dict[str, Any], diag: Dict[str, Any]) -> Optional[np.ndarray]:
    # 1) true separatrix / lcfs in shape
    key_pairs = [
        ("R_sep", "Z_sep"),
        ("R_separatrix", "Z_separatrix"),
        ("R_lcfs", "Z_lcfs"),
        ("lcfs_R", "lcfs_Z"),
        ("R_LCFS", "Z_LCFS"),
    ]
    for rkey, zkey in key_pairs:
        if rkey in shape and zkey in shape:
            try:
                P = _open_curve_from_closed(np.asarray(shape[rkey], float), np.asarray(shape[zkey], float))
                if P.shape[0] >= 20:
                    return P
            except Exception:
                pass

    # 2) fallback stored in diag (our star_equilibrium puts R_lcfs/Z_lcfs in plasma_diag if available)
    if isinstance(diag, dict):
        Rf = diag.get("R_lcfs", None)
        Zf = diag.get("Z_lcfs", None)
        if Rf is not None and Zf is not None:
            try:
                P = _open_curve_from_closed(np.asarray(Rf, float), np.asarray(Zf, float))
                if P.shape[0] >= 20:
                    return P
            except Exception:
                pass

    # 3) analyze_star fallback_lcfs shape slot
    fb = shape.get("fallback_lcfs", None)
    if isinstance(fb, dict) and ("R" in fb) and ("Z" in fb):
        try:
            P = _open_curve_from_closed(np.asarray(fb["R"], float), np.asarray(fb["Z"], float))
            if P.shape[0] >= 20:
                return P
        except Exception:
            pass

    return None


def _targets_from_geom(geom: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Dict[str, Tuple[float, float]], Dict[str, float]]:
    target_curve = None
    if "R_plasma" in geom and "Z_plasma" in geom:
        try:
            target_curve = _open_curve_from_closed(np.asarray(geom["R_plasma"], float), np.asarray(geom["Z_plasma"], float))
        except Exception:
            target_curve = None

    xt: Dict[str, Tuple[float, float]] = {}
    xpt = geom.get("xpoints_target", None)
    if isinstance(xpt, (list, tuple)):
        for item in xpt:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                R, Z, kind = item[0], item[1], str(item[2]).lower()
                if "lower" in kind:
                    xt["lower"] = (float(R), float(Z))
                elif "upper" in kind:
                    xt["upper"] = (float(R), float(Z))

    meta = geom.get("plasma_auto_meta", None)
    scal: Dict[str, float] = {}
    if isinstance(meta, dict) and meta:
        for k in (
            "geom_R0_mid", "geom_A_mid", "geom_kappa_mid",
            "geom_delta_u", "geom_delta_l",
            "R0_target", "A_target", "kappa_target", "Z0_target",
        ):
            if k in meta and meta[k] is not None:
                scal[k] = _safe_float(meta[k], float("nan"))

    return target_curve, xt, scal


def _rms_chamfer_sym(P: np.ndarray, Q: np.ndarray) -> float:
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    if P.ndim != 2 or Q.ndim != 2 or P.shape[1] != 2 or Q.shape[1] != 2:
        return float("inf")
    if P.shape[0] < 20 or Q.shape[0] < 20:
        return float("inf")

    d2_pq = []
    for p in P:
        d2_pq.append(float(np.min(np.sum((Q - p) ** 2, axis=1))))
    d2_qp = []
    for q in Q:
        d2_qp.append(float(np.min(np.sum((P - q) ** 2, axis=1))))

    return float(math.sqrt(0.5 * (float(np.mean(d2_pq)) + float(np.mean(d2_qp)))))


def _extract_xpoints(shape: Dict[str, Any], diag: Dict[str, Any]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []

    xp = shape.get("xpoints", None)
    if xp is None:
        xp = shape.get("xpoint", None) or shape.get("Xpoints", None)

    if isinstance(xp, (list, tuple)):
        for item in xp:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    out.append((float(item[0]), float(item[1])))
                except Exception:
                    pass
            elif isinstance(item, dict):
                for rk, zk in (("R", "Z"), ("r", "z"), ("Rx", "Zx")):
                    if rk in item and zk in item:
                        try:
                            out.append((float(item[rk]), float(item[zk])))
                        except Exception:
                            pass
    elif isinstance(xp, dict):
        for kk in ("lower", "upper", "x1", "x2"):
            if kk in xp:
                item = xp[kk]
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    out.append((float(item[0]), float(item[1])))
                elif isinstance(item, dict) and "R" in item and "Z" in item:
                    out.append((float(item["R"]), float(item["Z"])))

    # fallback keys in diag (optional)
    if not out and isinstance(diag, dict):
        candidates = [
            ("Rx_lower", "Zx_lower"),
            ("Rx_upper", "Zx_upper"),
            ("R_x_lower", "Z_x_lower"),
            ("R_x_upper", "Z_x_upper"),
        ]
        for rk, zk in candidates:
            if rk in diag and zk in diag:
                try:
                    out.append((float(diag[rk]), float(diag[zk])))
                except Exception:
                    pass

    return out


def _pick_lower_upper_xp(xps: List[Tuple[float, float]]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    if not xps:
        return None, None
    lower = min(xps, key=lambda p: p[1])
    upper = max(xps, key=lambda p: p[1])
    return lower, upper


def _frac_outside(points: np.ndarray, wall_open: np.ndarray, radius: float = -1e-9) -> float:
    P = np.asarray(points, float)
    W = np.asarray(wall_open, float)
    if P.ndim != 2 or W.ndim != 2 or P.shape[1] != 2 or W.shape[1] != 2:
        return float("nan")
    if P.shape[0] < 20 or W.shape[0] < 3:
        return float("nan")
    path = MplPath(W, closed=True)
    inside = path.contains_points(P, radius=float(radius))
    return float(1.0 - np.mean(inside))


# ----------------------------
# Misfit (CAD objective)
# ----------------------------
def _misfit_cad(
    geom: Dict[str, Any],
    shape: Dict[str, Any],
    diag: Dict[str, Any],
    cfg: Any,
    *,
    null_mode: str,
    prefer_inner: bool,
    enforce_inner: bool,
    inner_tol: float,
    inner_hard: bool,
) -> Tuple[float, Dict[str, Any]]:
    # scales / sigmas
    sig_R0 = float(getattr(cfg, "sig_R0_m", 0.25))
    sig_A  = float(getattr(cfg, "sig_A",    0.25))
    sig_k  = float(getattr(cfg, "sig_kappa",0.25))
    sig_d  = float(getattr(cfg, "sig_delta",0.20))

    sig_x      = float(getattr(cfg, "sig_x_m", 0.20))
    sig_shape  = float(getattr(cfg, "sig_shape_m", 0.05))
    sig_strike = float(getattr(cfg, "sig_x_m", 0.20))  # reuse x sigma unless you add separate one

    w_scalar = float(getattr(cfg, "w_scalar", 1.0))
    w_x      = float(getattr(cfg, "w_x", 1.0))
    w_shape  = float(getattr(cfg, "w_shape", 1.0))
    w_strike = float(getattr(cfg, "w_x", 1.0))  # reuse w_x unless you add w_strike

    # penalties
    penalty_no_sep    = float(getattr(cfg, "penalty_no_separatrix", 1e6))
    penalty_no_xp     = float(getattr(cfg, "penalty_no_xpoints", 1e6))
    penalty_no_strike = float(getattr(cfg, "penalty_no_xpoints", 1e6))
    penalty_fallback  = float(getattr(cfg, "penalty_fallback_lcfs", 5e4))
    penalty_neg_delta = float(getattr(cfg, "penalty_neg_delta", 10.0))

    # inner wall penalties
    pen_out_soft = float(getattr(cfg, "penalty_outside_inner", 5e5))
    pen_out_hard = float(getattr(cfg, "penalty_outside_inner_hard", 1e9))
    inner_radius = float(getattr(cfg, "inner_containment_radius", -1e-9))

    target_curve, xt, scal = _targets_from_geom(geom)

    if (not diag) or (not diag.get("ok", False)):
        return 1e9, {"mode": "cad", "ok": False, "reason": "diag_not_ok"}

    reason = str(shape.get("reason", "")).strip().lower()
    has_true_sep = bool(shape.get("ok_sep", False)) and (reason == "ok")

    # LCFS curve (true or fallback)
    lcfs = _extract_lcfs_curve(shape, diag)

    # targets from CAD windows if available
    win_xL = _extract_marker_window(geom, "xpt_lower")
    win_xU = _extract_marker_window(geom, "xpt_upper")
    win_sL = _extract_marker_window(geom, "strike_lower")
    win_sU = _extract_marker_window(geom, "strike_upper")

    # scalar targets from plasma_auto_meta or config
    R0_t = scal.get("geom_R0_mid", float("nan"))
    A_t  = scal.get("geom_A_mid",  float("nan"))
    k_t  = scal.get("geom_kappa_mid", float("nan"))
    du_t = scal.get("geom_delta_u", float("nan"))
    dl_t = scal.get("geom_delta_l", float("nan"))

    if not np.isfinite(R0_t):
        R0_t = float(getattr(cfg, "R0_geom", 4.0))
    if not np.isfinite(A_t):
        A_t = float(getattr(cfg, "A_geom", 2.0))
    if not np.isfinite(k_t):
        k_t = float(getattr(cfg, "kappa_geom", 2.0))
    if not (np.isfinite(du_t) and np.isfinite(dl_t)):
        d_t = float(getattr(cfg, "delta_geom", 0.3))
        du_t, dl_t = d_t, d_t

    # scalars from diag
    R0 = _safe_float(diag.get("R0", float("nan")))
    A  = _safe_float(diag.get("A", float("nan")))
    k  = _safe_float(diag.get("kappa", float("nan")))
    du = _safe_float(diag.get("delta_u", float("nan")))
    dl = _safe_float(diag.get("delta_l", float("nan")))

    if not np.isfinite(R0 + A + k + du + dl):
        return 1e9, {"mode": "cad", "ok": False, "reason": "bad_scalars"}

    dbar_t = 0.5 * (du_t + dl_t)
    dbar   = 0.5 * (du + dl)

    term_scalar = 0.0
    term_scalar += ((R0 - R0_t) / max(sig_R0, 1e-12)) ** 2
    term_scalar += ((A  - A_t)  / max(sig_A,  1e-12)) ** 2
    term_scalar += ((k  - k_t)  / max(sig_k,  1e-12)) ** 2
    term_scalar += ((dbar - dbar_t) / max(sig_d, 1e-12)) ** 2
    term_scalar *= (w_scalar ** 2)

    # X-points
    xps = _extract_xpoints(shape, diag)
    lower_xp, upper_xp = _pick_lower_upper_xp(xps)

    null_mode = str(null_mode).strip().lower()
    if null_mode not in ("lower", "upper", "double"):
        null_mode = "lower"

    dx_lower = None
    dx_upper = None
    term_x = 0.0
    pen_x = 0.0

    if null_mode in ("lower", "double"):
        if win_xL is not None:
            dx_lower = _distance_point_to_window(lower_xp, win_xL)
        elif "lower" in xt and lower_xp is not None:
            dx_lower = float(math.hypot(lower_xp[0] - xt["lower"][0], lower_xp[1] - xt["lower"][1]))
        if dx_lower is None:
            pen_x += penalty_no_xp
        else:
            term_x += (float(dx_lower) / max(sig_x, 1e-12)) ** 2

    if null_mode in ("upper", "double"):
        if win_xU is not None:
            dx_upper = _distance_point_to_window(upper_xp, win_xU)
        elif "upper" in xt and upper_xp is not None:
            dx_upper = float(math.hypot(upper_xp[0] - xt["upper"][0], upper_xp[1] - xt["upper"][1]))
        if dx_upper is None:
            pen_x += penalty_no_xp
        else:
            term_x += (float(dx_upper) / max(sig_x, 1e-12)) ** 2

    term_x *= (w_x ** 2)

    # Strike points (only meaningful if lcfs and inner wall exist)
    lower_sp = None
    upper_sp = None
    ds_lower = None
    ds_upper = None
    term_strike = 0.0
    pen_strike = 0.0

    if (lcfs is not None) and ("R_inner" in geom) and ("Z_inner" in geom):
        wall_inner = _open_curve_from_closed(np.asarray(geom["R_inner"], float), np.asarray(geom["Z_inner"], float))
        strikes = _compute_strike_points(lcfs, wall_inner)
        lower_sp, upper_sp = _pick_lower_upper_points(strikes)

        if null_mode in ("lower", "double") and (win_sL is not None):
            ds_lower = _distance_point_to_window(lower_sp, win_sL)
            if ds_lower is None:
                pen_strike += penalty_no_strike
            else:
                term_strike += (float(ds_lower) / max(sig_strike, 1e-12)) ** 2

        if null_mode in ("upper", "double") and (win_sU is not None):
            ds_upper = _distance_point_to_window(upper_sp, win_sU)
            if ds_upper is None:
                pen_strike += penalty_no_strike
            else:
                term_strike += (float(ds_upper) / max(sig_strike, 1e-12)) ** 2

    term_strike *= (w_strike ** 2)

    # Shape RMS
    term_shape = 0.0
    shape_rms = None
    if (target_curve is not None) and (lcfs is not None) and (target_curve.shape[0] >= 20) and (lcfs.shape[0] >= 20):
        sr = _rms_chamfer_sym(lcfs, target_curve)
        if np.isfinite(sr):
            shape_rms = float(sr)
            term_shape = (w_shape * (shape_rms / max(sig_shape, 1e-12))) ** 2

    misfit = float(math.sqrt(term_scalar + term_x + term_strike + term_shape))
    misfit += float(pen_x + pen_strike)

    # If there is no true separatrix, penalize strongly but do not flatten objective
    if not has_true_sep:
        misfit += penalty_no_sep

    # If only fallback lcfs exists, add extra penalty
    fallback_flag = False
    try:
        fb = shape.get("fallback_lcfs", None)
        if fb is True:
            fallback_flag = True
        elif isinstance(fb, dict):
            if fb.get("ok", False):
                fallback_flag = True
    except Exception:
        pass
    if fallback_flag and (not has_true_sep):
        misfit += penalty_fallback

    # negative delta penalty
    if du < 0.0 or dl < 0.0:
        misfit += penalty_neg_delta * (abs(min(du, 0.0)) + abs(min(dl, 0.0)))

    # containment
    frac_out_inner = None
    inner_enforced = bool(enforce_inner and ("R_inner" in geom) and ("Z_inner" in geom) and (lcfs is not None))
    if inner_enforced and lcfs is not None:
        wall_inner = _open_curve_from_closed(np.asarray(geom["R_inner"], float), np.asarray(geom["Z_inner"], float))
        frac = _frac_outside(lcfs, wall_inner, radius=inner_radius)
        frac_out_inner = float(frac) if np.isfinite(frac) else None

        if frac_out_inner is not None and frac_out_inner > float(inner_tol):
            if inner_hard:
                return float(pen_out_hard), {
                    "mode": "cad",
                    "ok": False,
                    "reason": "lcfs_outside_inner_hard",
                    "frac_out_inner": float(frac_out_inner),
                    "inner_tol": float(inner_tol),
                    "has_true_separatrix": bool(has_true_sep),
                }
            misfit += float(pen_out_soft) * float(frac_out_inner)

    info = {
        "mode": "cad",
        "ok": True,
        "null_mode": null_mode,
        "has_true_separatrix": bool(has_true_sep),
        "ok_sep": bool(has_true_sep),
        "n_xpoints": int(len(xps)),

        "dx_lower_m": dx_lower,
        "dx_upper_m": dx_upper,
        "ds_lower_m": ds_lower,
        "ds_upper_m": ds_upper,

        "lower_xpoint": lower_xp,
        "upper_xpoint": upper_xp,
        "lower_strike": lower_sp,
        "upper_strike": upper_sp,

        "shape_rms_m": shape_rms,
        "term_scalar": float(term_scalar),
        "term_x": float(term_x),
        "term_strike": float(term_strike),
        "term_shape": float(term_shape),

        "frac_out_inner": frac_out_inner,
        "inner_enforced": bool(inner_enforced),
        "prefer_inner": bool(prefer_inner),

        "used_windows": {
            "xpt_lower": bool(win_xL is not None),
            "xpt_upper": bool(win_xU is not None),
            "strike_lower": bool(win_sL is not None),
            "strike_upper": bool(win_sU is not None),
        },
    }
    return float(misfit), info

# ----------------------------
# Worker: evaluate one case in subprocess (hard timeout)
# ----------------------------
def _worker_eval(
    currents_A: Dict[str, float],
    dxf_path: Optional[str],
    cfg_overrides: Dict[str, Any],
    require_sep: bool,
    null_mode: str,
    prefer_inner: bool,
    enforce_inner: bool,
    inner_tol: float,
    inner_hard: bool,
    redirect_solver_noise: bool,
    conn,
) -> None:
    try:
        import config_star_bean as cfg
        import star_equilibrium as se

        # apply overrides
        for k, v in (cfg_overrides or {}).items():
            try:
                setattr(cfg, k, v)
            except Exception:
                pass

        # apply currents into cfg
        for k, v in (currents_A or {}).items():
            try:
                setattr(cfg, f"{str(k).strip()}_current", float(v))
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
                diag = {"ok": False}

        reason = str(shape.get("reason", "")).strip().lower()
        has_true_sep = bool(shape.get("ok_sep", False)) and (reason == "ok")

        xps = shape.get("xpoints", []) or []
        n_xp = 0
        try:
            n_xp = int(len(xps))
        except Exception:
            n_xp = 0

        # Always compute misfit (penalty+proxy if needed)
        misfit, objinfo = _misfit_cad(
            geom=geom,
            shape=shape,
            diag=diag,
            cfg=cfg,
            null_mode=str(null_mode),
            prefer_inner=bool(prefer_inner),
            enforce_inner=bool(enforce_inner),
            inner_tol=float(inner_tol),
            inner_hard=bool(inner_hard),
        )

        # Mark requirement failures (do not collapse to a constant cost)
        req_fail = False
        req_reason = None
        if require_sep and (not has_true_sep):
            req_fail = True
            req_reason = "require_sep_failed"
        if require_sep and str(null_mode).lower().strip() == "double" and n_xp < 2:
            req_fail = True
            req_reason = "require_double_x_failed"

        res = {
            "ok": True,
            "ok_solve": True,
            "misfit": float(misfit),
            "currents_A": dict(currents_A),
            "diag": diag,
            "shape_ok_sep": bool(has_true_sep),
            "require_sep_failed": bool(req_fail),
            "require_reason": req_reason,
            "obj": objinfo,
        }
        conn.send(res)
        conn.close()

    except Exception as e:
        try:
            conn.send({
                "ok": False,
                "ok_solve": False,
                "error": repr(e),
                "currents_A": dict(currents_A),
            })
            conn.close()
        except Exception:
            pass

def eval_case(
    currents_A: Dict[str, float],
    *,
    dxf_path: Optional[str],
    timeout_s: float,
    cfg_overrides: Dict[str, Any],
    require_sep: bool,
    null_mode: str,
    prefer_inner: bool,
    enforce_inner: bool,
    inner_tol: float,
    inner_hard: bool,
    redirect_solver_noise: bool,
) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(
        target=_worker_eval,
        args=(
            dict(currents_A),
            dxf_path,
            dict(cfg_overrides or {}),
            bool(require_sep),
            str(null_mode),
            bool(prefer_inner),
            bool(enforce_inner),
            float(inner_tol),
            bool(inner_hard),
            bool(redirect_solver_noise),
            send_conn,
        ),
    )

    t0 = time.time()
    p.start()

    # close send end in parent for stability
    try:
        send_conn.close()
    except Exception:
        pass

    p.join(timeout_s)
    elapsed = time.time() - t0

    if p.is_alive():
        p.terminate()
        p.join()
        try:
            recv_conn.close()
        except Exception:
            pass
        return {
            "ok": False,
            "ok_solve": False,
            "error": f"timeout>{timeout_s:.1f}s",
            "elapsed_s": float(elapsed),
            "currents_A": dict(currents_A),
        }

    if recv_conn.poll(0.05):
        try:
            res = recv_conn.recv()
        except Exception as e:
            res = {
                "ok": False,
                "ok_solve": False,
                "error": f"recv_failed:{repr(e)}",
                "elapsed_s": float(elapsed),
                "currents_A": dict(currents_A),
                "exitcode": p.exitcode,
            }
    else:
        res = {
            "ok": False,
            "ok_solve": False,
            "error": "no_result_from_worker",
            "elapsed_s": float(elapsed),
            "currents_A": dict(currents_A),
            "exitcode": p.exitcode,
        }

    try:
        recv_conn.close()
    except Exception:
        pass

    res["elapsed_s"] = float(elapsed)
    return res


# ----------------------------
# Sampling / bounds
# ----------------------------
FAMILIES = ("CS", "PF1", "PF2", "PF3", "PF4", "PF5", "PF6")


def _get_prior_from_cfg(cfg: Any) -> Dict[str, float]:
    prior: Dict[str, float] = {}
    for k in FAMILIES:
        attr = f"{k}_current"
        try:
            prior[k] = float(getattr(cfg, attr))
        except Exception:
            prior[k] = 0.0
    return prior


def _default_hard_limits_MA() -> Dict[str, float]:
    # conservative defaults
    return {"CS": 80.0, "PF1": 20.0, "PF2": 20.0, "PF3": 12.0, "PF4": 12.0, "PF5": 15.0, "PF6": 25.0}


def _clip_hard(curr_MA: Dict[str, float], hard_MA: Dict[str, float]) -> Dict[str, float]:
    out = dict(curr_MA)
    for k, lim in hard_MA.items():
        lim = float(abs(lim))
        out[k] = float(np.clip(out.get(k, 0.0), -lim, +lim))
    return out


def _make_stage_bounds(
    prior_MA: Dict[str, float],
    hard_MA: Dict[str, float],
    *,
    widen: float,
) -> Dict[str, Tuple[float, float]]:
    # bounds centered at prior: [prior - widen*|prior| - base, prior + widen*|prior| + base]
    # base ensures exploration if prior is near zero
    base = {"CS": 2.0, "PF1": 2.0, "PF2": 2.0, "PF3": 1.0, "PF4": 1.0, "PF5": 1.5, "PF6": 2.5}
    bounds: Dict[str, Tuple[float, float]] = {}
    for k in FAMILIES:
        p = float(prior_MA.get(k, 0.0))
        span = widen * max(abs(p), 0.5) + float(base.get(k, 1.0))
        lo = p - span
        hi = p + span
        lim = float(abs(hard_MA.get(k, 1e9)))
        lo = max(lo, -lim)
        hi = min(hi, +lim)
        bounds[k] = (float(lo), float(hi))
    return bounds


def _sample_uniform(bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    return {k: random.uniform(bounds[k][0], bounds[k][1]) for k in FAMILIES}


def _sample_gauss(center: Dict[str, float], sigma: Dict[str, float], bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    out = {}
    for k in FAMILIES:
        s = float(sigma.get(k, 1.0))
        x = random.gauss(float(center.get(k, 0.0)), s)
        lo, hi = bounds[k]
        out[k] = float(np.clip(x, lo, hi))
    return out


def _sigma_from_bounds(bounds: Dict[str, Tuple[float, float]], frac: float) -> Dict[str, float]:
    out = {}
    for k in FAMILIES:
        lo, hi = bounds[k]
        out[k] = frac * (hi - lo)
    return out


def _currents_MA_to_A(curr_MA: Dict[str, float]) -> Dict[str, float]:
    return {k: float(curr_MA.get(k, 0.0) * 1e6) for k in FAMILIES}


def _currents_A_to_MA(curr_A: Dict[str, float]) -> Dict[str, float]:
    return {k: float(curr_A.get(k, 0.0) / 1e6) for k in FAMILIES}


# ----------------------------
# Main scan
# ----------------------------
def scan_star(
    *,
    dxf_path: Optional[str],
    workers: int,
    timeout_s: float,
    n1: int,
    n2: int,
    pop: int,
    prefer_inner: bool,
    null_mode: str,
    require_sep: bool,
    enforce_inner: bool,
    inner_tol: float,
    inner_hard: bool,
    show_solver: bool,
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    import config_star_bean as cfg

    # prior + bounds
    prior_A = _get_prior_from_cfg(cfg)
    prior_MA = _currents_A_to_MA(prior_A)

    hard = getattr(cfg, "hard_limits_MA", None)
    if isinstance(hard, dict) and hard:
        hard_MA = {str(k): float(v) for k, v in hard.items()}
    else:
        hard_MA = _default_hard_limits_MA()

    prior_MA = _clip_hard(prior_MA, hard_MA)

    bounds1 = _make_stage_bounds(prior_MA, hard_MA, widen=1.5)
    bounds2 = _make_stage_bounds(prior_MA, hard_MA, widen=0.9)

    # logs
    tag = _ts_tag()
    log_jsonl = _results_dir() / "scan_multigoal_results.jsonl"
    best_path = _results_dir() / "scan_multigoal_best.txt"
    best_batch_path = _results_dir() / "scan_multigoal_best_batch.txt"

    # headline
    print("[INFO] scan_star_multigoal started (PRIOR-centered bounds + HARD limits)")
    print(f"[INFO] dxf={dxf_path or 'cfg/auto'}")
    print(f"[INFO] workers={workers} n1={n1} n2={n2} pop={pop} timeout={timeout_s:.1f}")
    print(f"[INFO] prior [MA]: {prior_MA}")
    print(f"[INFO] hard limits [MA]: {hard_MA}")
    print("[INFO] search bounds stage1 [MA]:")
    for k in FAMILIES:
        lo, hi = bounds1[k]
        print(f"  {k:>3s}: {lo:+8.3f} .. {hi:+8.3f}")
    print("[INFO] search bounds stage2 [MA]:")
    for k in FAMILIES:
        lo, hi = bounds2[k]
        print(f"  {k:>3s}: {lo:+8.3f} .. {hi:+8.3f}")

    print(f"[INFO] null_mode={null_mode} require_sep={require_sep} prefer_inner={prefer_inner}")
    print(f"[INFO] inner-wall: enforce={enforce_inner} tol={inner_tol} hard={inner_hard}")

    # init best = prior
    best_curr_MA = dict(prior_MA)
    best_misfit = float("inf")
    best_res: Optional[Dict[str, Any]] = None

    # helper: evaluate one candidate MA dict
    def eval_one(curr_MA: Dict[str, float]) -> Dict[str, Any]:
        curr_MA2 = _clip_hard(curr_MA, hard_MA)
        curr_A = _currents_MA_to_A(curr_MA2)
        res = eval_case(
            curr_A,
            dxf_path=dxf_path,
            timeout_s=timeout_s,
            cfg_overrides={},
            require_sep=require_sep,
            null_mode=null_mode,
            prefer_inner=prefer_inner,
            enforce_inner=enforce_inner,
            inner_tol=inner_tol,
            inner_hard=inner_hard,
            redirect_solver_noise=(not show_solver),
        )
        res["currents_MA"] = curr_MA2
        return res

    # stage runner
    def run_stage(name: str, n_iter: int, bounds: Dict[str, Tuple[float, float]], sigma_frac0: float) -> None:
        nonlocal best_curr_MA, best_misfit, best_res

        sigma = _sigma_from_bounds(bounds, sigma_frac0)
        t_start = time.time()

        for it in range(1, n_iter + 1):
            # build batch: mix of global uniform + local gaussian around best
            batch: List[Dict[str, float]] = []
            n_global = max(2, pop // 6)  #  ~16% global exploration
            n_local = pop - n_global

            # elite injection: include current best at least once
            batch.append(dict(best_curr_MA))

            for _ in range(n_global - 1):
                batch.append(_sample_uniform(bounds))

            for _ in range(n_local):
                batch.append(_sample_gauss(best_curr_MA, sigma, bounds))

            # ensure size
            batch = batch[:pop]

            # evaluate in parallel (thread pool submits spawn-evals)
            solve_ok = 0
            shape_ok = 0

            batch_best_m = float("inf")
            batch_best_res: Optional[Dict[str, Any]] = None

            with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
                futs = {ex.submit(eval_one, cand): cand for cand in batch}
                for fut in as_completed(futs):
                    cand = futs[fut]
                    try:
                        res = fut.result()
                    except Exception as e:
                        res = {
                            "ok": False, "ok_solve": False, "error": f"future_failed:{repr(e)}",
                            "currents_A": _currents_MA_to_A(cand),
                            "currents_MA": cand,
                        }

                    ok_solve = bool(res.get("ok_solve", False))
                    if ok_solve:
                        solve_ok += 1
                    obj = res.get("obj", {}) if isinstance(res.get("obj", {}), dict) else {}
                    # "shape_ok" is a proxy: CAD objective ok + separatrix ok
                    this_shape_ok = bool(
                        ok_solve
                        and isinstance(obj, dict)
                        and obj.get("ok", False)
                        and obj.get("has_true_separatrix", obj.get("ok_sep", False))
                    )
                    if str(null_mode).lower().strip() == "double":
                        this_shape_ok = bool(this_shape_ok and int(obj.get("n_xpoints", 0)) >= 2)
                    if this_shape_ok:
                        shape_ok += 1

                    m = float(res.get("misfit", 1e9)) if ok_solve else 1e9

                    # log each eval
                    _append_jsonl(log_jsonl, {
                        "event": "eval",
                        "stage": name,
                        "iter": it,
                        "misfit": float(m),
                        "res": res,
                    })

                    if m < batch_best_m:
                        batch_best_m = m
                        batch_best_res = res

            # accept global best if improved
            if batch_best_res is not None and batch_best_m < best_misfit:
                best_misfit = float(batch_best_m)
                best_res = batch_best_res
                best_curr_MA = dict(best_res.get("currents_MA", best_curr_MA))

            # anneal sigma slowly
            for k in sigma:
                sigma[k] = max(0.02, sigma[k] * 0.999)  # floor in MA

            # progress line (similar vibe to your old output)
            elapsed = time.time() - t_start
            rate = (it * pop) / max(1e-9, elapsed)
            last_m = float(batch_best_m) if np.isfinite(batch_best_m) else 1e9
            print(f"{name:<6s} [{'#'*28}] {pop:4d}/{pop:<4d}  solve_ok={solve_ok:4d} shape_ok={shape_ok:4d} best={best_misfit:7.2f} last={last_m:7.2f}  {rate:4.2f}/s")

            # write best-per-batch snapshot (human readable)
            if batch_best_res is not None:
                txt = _json_dumps_safe({
                    "stage": name,
                    "iter": it,
                    "best_batch_misfit": float(batch_best_m),
                    "currents_MA": batch_best_res.get("currents_MA", {}),
                    "obj": batch_best_res.get("obj", {}),
                }, indent=2)
                _save_text(best_batch_path, txt)

            # write global best snapshot (human readable + reuse by refine)
            if best_res is not None:
                txt = _json_dumps_safe({
                    "best_misfit": float(best_misfit),
                    "currents_MA": best_curr_MA,
                    "currents_A": _currents_MA_to_A(best_curr_MA),
                    "obj": best_res.get("obj", {}),
                    "diag": best_res.get("diag", {}),
                }, indent=2)
                _save_text(best_path, txt)

    # stage1 / stage2
    run_stage("coarse", n1, bounds1, sigma_frac0=0.20)
    run_stage("refine", n2, bounds2, sigma_frac0=0.10)

    print("[DONE] Finished.")
    print(f"[DONE] Best GLOBAL written to: {best_path}")
    print(f"[DONE] Best-per-batch written to: {best_batch_path}")
    print(f"[DONE] Log: {log_jsonl}")
    print(f"[DONE] global best misfit = {best_misfit:.6g}")


# ----------------------------
# CLI
# ----------------------------
def main():
    mp.freeze_support()

    ap = argparse.ArgumentParser()
    ap.add_argument("--dxf", type=str, default=None, help="DXF path (defaults to cfg.dxf_path inside star_equilibrium)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--n1", type=int, default=1000, help="Stage1 iterations")
    ap.add_argument("--n2", type=int, default=1000, help="Stage2 iterations")
    ap.add_argument("--pop", type=int, default=64, help="Population per iteration")

    ap.add_argument("--prefer_inner", action="store_true", help="Prefer inner LCFS selection (passed into objective info; your analyze_star may use it).")
    ap.add_argument("--null-mode", type=str, default="lower", choices=["lower", "upper", "double"], help="Which X-point(s) to match (CAD xpoints_target).")
    ap.add_argument("--require-sep", action="store_true", help="Require separatrix (diverted).")

    ap.add_argument("--enforce-inner", action="store_true", help="Enforce LCFS inside WALL_INNER (if exists).")
    ap.add_argument("--inner-tol", type=float, default=0.0, help="Tolerance: allowed fraction of LCFS points outside inner wall.")
    ap.add_argument("--inner-hard", action="store_true", help="Hard fail if outside inner wall beyond tol.")

    ap.add_argument("--show-solver", action="store_true", help="Do NOT redirect solver noise (more verbose).")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    scan_star(
        dxf_path=args.dxf,
        workers=int(args.workers),
        timeout_s=float(args.timeout),
        n1=int(args.n1),
        n2=int(args.n2),
        pop=int(args.pop),
        prefer_inner=bool(args.prefer_inner),
        null_mode=str(args.null_mode),
        require_sep=bool(args.require_sep),
        enforce_inner=bool(args.enforce_inner),
        inner_tol=float(args.inner_tol),
        inner_hard=bool(args.inner_hard),
        show_solver=bool(args.show_solver),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
