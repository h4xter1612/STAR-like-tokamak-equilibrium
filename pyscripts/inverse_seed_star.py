# inverse_seed_star.py
"""
Inverse seed (bounded Gauss-Newton / LM) for STAR-like FreeGSNKE equilibria
-------------------------------------------------------------------------

Goal:
- Produce a good initial set of coil FAMILY currents (CS, PF1..PF6)
  that places:
    - X-points near CAD windows XPT_*_WIN (double-null symmetric target)
    - LCFS close to CAD windows STRIKE_*_WIN (strike regions)
    - LCFS shape close to plasma_target (AUTO Miller)
    - LCFS inside WALL_INNER (soft/hard)
  and prefers true separatrix (reason=="ok") but keeps a smooth-ish proxy
  so optimization has signal even before true separatrix appears.

Method:
- Finite-difference Jacobian
- Levenberg–Marquardt damping
- Hard bounds per family
- Hard timeout per evaluation via spawn subprocess (Windows-safe)

Outputs (./results):
- inverse_seed_log.jsonl
- inverse_seed_best.json

Example:
  py .\inverse_seed_star.py --timeout 420 --max-iters 18 --fd-step-ma 0.12 --max-step-ma 0.60 ^
      --scan-keys PF2,PF3,PF4,PF5,PF6 --fixed-keys CS,PF1 --null-mode double ^
      --enforce-inner --inner-tol 0.005 --inner-hard --no-blanket --fast
"""

from __future__ import annotations

import argparse
import json
import math
import time
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_json_dumps_safe(obj, indent=2))


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


def _bbox(xy: np.ndarray) -> Tuple[float, float, float, float]:
    P = np.asarray(xy, float)
    return float(P[:, 0].min()), float(P[:, 0].max()), float(P[:, 1].min()), float(P[:, 1].max())


# ----------------------------
# Geometry distances (windows)
# ----------------------------
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


def _point_to_polyline_distance(point: Tuple[float, float], poly: np.ndarray, closed: bool = False) -> float:
    P = np.asarray(poly, float)
    if P.ndim != 2 or P.shape[0] < 2:
        return float("inf")
    p = np.asarray(point, float)

    if bool(closed) and P.shape[0] >= 3:
        try:
            if MplPath(P, closed=True).contains_point((float(p[0]), float(p[1]))):
                return 0.0
        except Exception:
            pass

    dmin = float("inf")
    n = P.shape[0]
    m = n if closed else (n - 1)
    for i in range(m):
        a = P[i]
        b = P[(i + 1) % n]
        d = _point_to_segment_distance(p, a, b)
        if d < dmin:
            dmin = d
    return float(dmin)


def _polyline_to_polyline_min_distance(P: np.ndarray, Q: np.ndarray, closedQ: bool = False) -> float:
    """
    min_{p in P vertices} dist(p, polyline Q)
    (cheap + robust; Q is usually small window polyline)
    """
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    if P.ndim != 2 or Q.ndim != 2 or P.shape[0] < 2 or Q.shape[0] < 2:
        return float("inf")
    dmin = float("inf")
    for p in P:
        d = _point_to_polyline_distance((float(p[0]), float(p[1])), Q, closed=bool(closedQ))
        if d < dmin:
            dmin = d
    return float(dmin)


def _extract_marker_window(geom: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
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
    return {"xy": xy, "closed": bool(pack.get("closed", False))}


def _frac_outside(points: np.ndarray, wall_open: np.ndarray, radius: float = -1e-9) -> float:
    P = np.asarray(points, float)
    W = np.asarray(wall_open, float)
    if P.ndim != 2 or W.ndim != 2 or P.shape[1] != 2 or W.shape[1] != 2:
        return float("nan")
    if P.shape[0] < 20 or W.shape[0] < 3:
        return float("nan")
    inside = MplPath(W, closed=True).contains_points(P, radius=float(radius))
    return float(1.0 - np.mean(inside))


# ----------------------------
# X-point extraction
# ----------------------------
def _extract_xpoints(shape: Dict[str, Any]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    xp = shape.get("xpoints", None)
    if isinstance(xp, (list, tuple)):
        for item in xp:
            if isinstance(item, dict) and ("R" in item) and ("Z" in item):
                try:
                    out.append((float(item["R"]), float(item["Z"])))
                except Exception:
                    pass
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    out.append((float(item[0]), float(item[1])))
                except Exception:
                    pass
    return out


def _pick_lower_upper(points: List[Tuple[float, float]]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    if not points:
        return None, None
    lower = min(points, key=lambda p: p[1])
    upper = max(points, key=lambda p: p[1])
    return lower, upper


# ----------------------------
# Chamfer RMS
# ----------------------------
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


# ----------------------------
# Residual construction (worker-side)
# ----------------------------
def _build_residual_vector(
    geom: Dict[str, Any],
    shape: Dict[str, Any],
    diag: Dict[str, Any],
    cfg: Any,
    *,
    scan_keys: List[str],
    I_A: Dict[str, float],
    Iref_A: Dict[str, float],
    bounds_A: Dict[str, Tuple[float, float]],
    null_mode: str,
    enforce_inner: bool,
    inner_tol: float,
    inner_hard: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Returns residual vector r (1D) and a metrics dict.
    Residuals are normalized by sigmas so GN/LM behaves.
    """
    # sigmas / weights
    sig_R0 = float(getattr(cfg, "sig_R0_m", 0.25))
    sig_A  = float(getattr(cfg, "sig_A",    0.25))
    sig_k  = float(getattr(cfg, "sig_kappa",0.25))
    sig_d  = float(getattr(cfg, "sig_delta",0.20))
    sig_x  = float(getattr(cfg, "sig_x_m",  0.20))
    sig_shape = float(getattr(cfg, "sig_shape_m", 0.05))
    sig_strike = float(getattr(cfg, "sig_x_m", 0.20))

    # regularization scale (MA)
    sig_I_MA = float(getattr(cfg, "sig_I_reg_MA", 2.0))
    sig_I_A = sig_I_MA * 1e6

    inner_radius = float(getattr(cfg, "inner_containment_radius", -1e-9))

    # targets
    R0_t = float(getattr(cfg, "R0_geom", 4.0))
    A_t  = float(getattr(cfg, "A_geom", 2.0))
    k_t  = float(getattr(cfg, "kappa_geom", 2.2))
    d_t  = float(getattr(cfg, "delta_geom", 0.6))

    # diag scalars
    R0 = _safe_float(diag.get("R0", float("nan")))
    A  = _safe_float(diag.get("A",  float("nan")))
    k  = _safe_float(diag.get("kappa", float("nan")))
    du = _safe_float(diag.get("delta_u", float("nan")))
    dl = _safe_float(diag.get("delta_l", float("nan")))
    dbar = 0.5 * (du + dl) if np.isfinite(du + dl) else float("nan")

    # separatrix truth (after your analyze_star patch)
    reason = str(shape.get("reason", "")).strip().lower()
    has_true_sep = bool(shape.get("ok_sep", False)) and (reason == "ok")

    # LCFS curve (prefer true separatrix; else allow fallback for shape/strike proximity)
    lcfs = None
    if "R_sep" in shape and "Z_sep" in shape:
        try:
            lcfs = _open_curve_from_closed(np.asarray(shape["R_sep"], float), np.asarray(shape["Z_sep"], float))
        except Exception:
            lcfs = None
    if lcfs is None and isinstance(diag, dict):
        Rf = diag.get("R_lcfs", None)
        Zf = diag.get("Z_lcfs", None)
        if Rf is not None and Zf is not None:
            try:
                lcfs = _open_curve_from_closed(np.asarray(Rf, float), np.asarray(Zf, float))
            except Exception:
                lcfs = None

    # target curve from geom (AUTO plasma target)
    target_curve = None
    if "R_plasma" in geom and "Z_plasma" in geom:
        try:
            target_curve = _open_curve_from_closed(np.asarray(geom["R_plasma"], float), np.asarray(geom["Z_plasma"], float))
        except Exception:
            target_curve = None

    # marker windows
    win_xL = _extract_marker_window(geom, "xpt_lower")
    win_xU = _extract_marker_window(geom, "xpt_upper")
    win_sL = _extract_marker_window(geom, "strike_lower")
    win_sU = _extract_marker_window(geom, "strike_upper")

    # xpoints from shape saddle detector
    xps = _extract_xpoints(shape)
    lower_xp, upper_xp = _pick_lower_upper(xps)

    null_mode = str(null_mode).strip().lower()
    if null_mode not in ("lower", "upper", "double"):
        null_mode = "double"

    # build residual list
    r: List[float] = []
    info: Dict[str, Any] = {
        "has_true_separatrix": bool(has_true_sep),
        "reason": str(shape.get("reason", "")),
        "n_xpoints": int(len(xps)),
        "lower_xpoint": lower_xp,
        "upper_xpoint": upper_xp,
    }

    # scalar residuals (always)
    if np.isfinite(R0):
        r.append((R0 - R0_t) / max(sig_R0, 1e-12))
    else:
        r.append(50.0)
    if np.isfinite(A):
        r.append((A - A_t) / max(sig_A, 1e-12))
    else:
        r.append(50.0)
    if np.isfinite(k):
        r.append((k - k_t) / max(sig_k, 1e-12))
    else:
        r.append(50.0)
    if np.isfinite(dbar):
        r.append((dbar - d_t) / max(sig_d, 1e-12))
    else:
        r.append(50.0)

    # X-point window residuals
    dxL = None
    dxU = None
    if null_mode in ("lower", "double"):
        if lower_xp is not None and win_xL is not None:
            dxL = _point_to_polyline_distance(lower_xp, win_xL["xy"], closed=bool(win_xL.get("closed", False)))
            r.append(dxL / max(sig_x, 1e-12))
        else:
            r.append(25.0)
    if null_mode in ("upper", "double"):
        if upper_xp is not None and win_xU is not None:
            dxU = _point_to_polyline_distance(upper_xp, win_xU["xy"], closed=bool(win_xU.get("closed", False)))
            r.append(dxU / max(sig_x, 1e-12))
        else:
            r.append(25.0)

    info["dx_lower_m"] = dxL
    info["dx_upper_m"] = dxU

    # Strike window residuals: distance of LCFS polyline to strike window polyline
    dsL = None
    dsU = None
    if lcfs is not None and lcfs.shape[0] >= 20:
        if null_mode in ("lower", "double") and win_sL is not None:
            dsL = _polyline_to_polyline_min_distance(lcfs, win_sL["xy"], closedQ=bool(win_sL.get("closed", False)))
            r.append(dsL / max(sig_strike, 1e-12))
        elif null_mode in ("lower", "double"):
            r.append(25.0)

        if null_mode in ("upper", "double") and win_sU is not None:
            dsU = _polyline_to_polyline_min_distance(lcfs, win_sU["xy"], closedQ=bool(win_sU.get("closed", False)))
            r.append(dsU / max(sig_strike, 1e-12))
        elif null_mode in ("upper", "double"):
            r.append(25.0)
    else:
        # no lcfs -> can't do strike distance
        if null_mode in ("lower", "double"):
            r.append(35.0)
        if null_mode in ("upper", "double"):
            r.append(35.0)

    info["ds_lower_m"] = dsL
    info["ds_upper_m"] = dsU

    # Shape (Chamfer) residual
    shape_rms = None
    if lcfs is not None and target_curve is not None and lcfs.shape[0] >= 20 and target_curve.shape[0] >= 20:
        shape_rms = _rms_chamfer_sym(lcfs, target_curve)
        if np.isfinite(shape_rms):
            r.append(shape_rms / max(sig_shape, 1e-12))
        else:
            r.append(25.0)
    else:
        r.append(25.0)
    info["shape_rms_m"] = shape_rms

    # Inner-wall containment residual (soft)
    frac_out = None
    if enforce_inner and lcfs is not None and ("R_inner" in geom) and ("Z_inner" in geom):
        wall_inner = _open_curve_from_closed(np.asarray(geom["R_inner"], float), np.asarray(geom["Z_inner"], float))
        frac_out = _frac_outside(lcfs, wall_inner, radius=inner_radius)
        if np.isfinite(frac_out):
            excess = max(0.0, float(frac_out) - float(inner_tol))
            # scale by tol (so 1x tol ~ 1)
            scale = max(float(inner_tol), 1e-3)
            r.append(excess / scale)
        else:
            r.append(10.0)
    else:
        r.append(0.0)
    info["frac_out_inner"] = frac_out

    # True separatrix preference: small residual when true sep, big when not
    # (keeps some pressure to become diverted, but not a flat cliff)
    r.append(0.0 if has_true_sep else 10.0)

    # Regularization (only scan_keys)
    for k in scan_keys:
        if k not in I_A or k not in Iref_A:
            continue
        r.append((float(I_A[k]) - float(Iref_A[k])) / max(sig_I_A, 1e-12))

    # Hard bound violation residual (should be 0 if we clamp; kept as safety)
    for k in scan_keys:
        lo, hi = bounds_A[k]
        x = float(I_A.get(k, 0.0))
        viol = 0.0
        if x < lo:
            viol = (lo - x) / 1e6
        elif x > hi:
            viol = (x - hi) / 1e6
        r.append(viol)

    rvec = np.asarray(r, float)
    info["r_norm2"] = float(np.dot(rvec, rvec))
    info["R0"] = R0; info["A"] = A; info["kappa"] = k; info["dbar"] = dbar
    return rvec, info


# ----------------------------
# Spawn worker eval (timeout)
# ----------------------------
def _worker_eval_residuals(
    currents_A: Dict[str, float],
    dxf_path: Optional[str],
    cfg_overrides: Dict[str, Any],
    scan_keys: List[str],
    fixed_keys: List[str],
    Iref_A: Dict[str, float],
    bounds_A: Dict[str, Tuple[float, float]],
    null_mode: str,
    enforce_inner: bool,
    inner_tol: float,
    inner_hard: bool,
    conn,
) -> None:
    try:
        import config_star_bean as cfg
        import star_equilibrium as se

        # apply overrides (e.g., no blanket, fast continuation)
        for k, v in (cfg_overrides or {}).items():
            try:
                setattr(cfg, k, v)
            except Exception:
                pass

        # set currents (both fixed and scan keys)
        for k, v in (currents_A or {}).items():
            try:
                setattr(cfg, f"{str(k).strip()}_current", float(v))
            except Exception:
                pass

        eq, tokamak, geom, shape = se.build_equilibrium(
            verbose=False,
            redirect_solver_noise=True,
            dxf_path=dxf_path,
        )

        diag = shape.get("plasma_diag", None) or {}
        if (not diag) or (not diag.get("ok", False)):
            try:
                diag = se.plasma_diagnostics(eq, geom, shape)
            except Exception:
                diag = {"ok": False}

        # Build residuals
        rvec, info = _build_residual_vector(
            geom, shape, diag, cfg,
            scan_keys=scan_keys,
            I_A=currents_A,
            Iref_A=Iref_A,
            bounds_A=bounds_A,
            null_mode=null_mode,
            enforce_inner=enforce_inner,
            inner_tol=inner_tol,
            inner_hard=inner_hard,
        )

        conn.send({
            "ok": True,
            "r": rvec,
            "info": info,
            "shape_reason": str(shape.get("reason", "")),
            "ok_sep": bool(shape.get("ok_sep", False)),
            "n_xpoints": int(len(shape.get("xpoints", []) or [])),
        })
        conn.close()
    except Exception as e:
        try:
            conn.send({"ok": False, "error": repr(e)})
            conn.close()
        except Exception:
            pass


def eval_residuals_timeout(
    currents_A: Dict[str, float],
    *,
    dxf_path: Optional[str],
    timeout_s: float,
    cfg_overrides: Dict[str, Any],
    scan_keys: List[str],
    fixed_keys: List[str],
    Iref_A: Dict[str, float],
    bounds_A: Dict[str, Tuple[float, float]],
    null_mode: str,
    enforce_inner: bool,
    inner_tol: float,
    inner_hard: bool,
) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(
        target=_worker_eval_residuals,
        args=(
            dict(currents_A),
            dxf_path,
            dict(cfg_overrides or {}),
            list(scan_keys),
            list(fixed_keys),
            dict(Iref_A),
            dict(bounds_A),
            str(null_mode),
            bool(enforce_inner),
            float(inner_tol),
            bool(inner_hard),
            send_conn,
        ),
    )
    t0 = time.time()
    p.start()
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
        return {"ok": False, "error": f"timeout>{timeout_s:.1f}s", "elapsed_s": float(elapsed)}

    if recv_conn.poll(0.05):
        try:
            res = recv_conn.recv()
        except Exception as e:
            res = {"ok": False, "error": f"recv_failed:{repr(e)}"}
    else:
        res = {"ok": False, "error": "no_result_from_worker"}

    try:
        recv_conn.close()
    except Exception:
        pass

    res["elapsed_s"] = float(elapsed)
    return res


# ----------------------------
# Bounds
# ----------------------------
FAMILIES = ("CS", "PF1", "PF2", "PF3", "PF4", "PF5", "PF6")


def _get_currents_from_cfg(cfg: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in FAMILIES:
        try:
            out[k] = float(getattr(cfg, f"{k}_current"))
        except Exception:
            out[k] = 0.0
    return out


def _make_bounds_A(cfg: Any, scan_keys: List[str], fixed_keys: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Use (in order):
      1) cfg.OPERATING_FAMILY_CURRENT_LIMIT_A if exists
      2) cfg.MAX_RECOMMENDED_FAMILY_CURRENTS_A * cfg.OPERATING_I_SAFETY_FACTOR
      3) fallback hard limits (MA)
    """
    bounds: Dict[str, Tuple[float, float]] = {}

    op = getattr(cfg, "OPERATING_FAMILY_CURRENT_LIMIT_A", None)
    if isinstance(op, dict) and op:
        for k in scan_keys + fixed_keys:
            lim = float(abs(op.get(k, 0.0)))
            if lim > 0:
                bounds[k] = (-lim, +lim)

    if len(bounds) < len(set(scan_keys + fixed_keys)):
        maxrec = getattr(cfg, "MAX_RECOMMENDED_FAMILY_CURRENTS_A", None)
        sf = float(getattr(cfg, "OPERATING_I_SAFETY_FACTOR", 0.35))
        if isinstance(maxrec, dict) and maxrec:
            for k in scan_keys + fixed_keys:
                if k in bounds:
                    continue
                lim = float(abs(maxrec.get(k, 0.0))) * sf
                if lim > 0:
                    bounds[k] = (-lim, +lim)

    # fallback MA
    fallback_MA = {"CS": 80.0, "PF1": 20.0, "PF2": 20.0, "PF3": 12.0, "PF4": 12.0, "PF5": 15.0, "PF6": 25.0}
    for k in scan_keys + fixed_keys:
        if k in bounds:
            continue
        lim = float(fallback_MA.get(k, 10.0)) * 1e6
        bounds[k] = (-lim, +lim)

    return bounds


def _clamp_A(x: Dict[str, float], bounds_A: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    y = dict(x)
    for k, (lo, hi) in bounds_A.items():
        y[k] = float(np.clip(float(y.get(k, 0.0)), float(lo), float(hi)))
    return y


# ----------------------------
# LM / GN loop
# ----------------------------
def main():
    mp.freeze_support()

    ap = argparse.ArgumentParser()
    ap.add_argument("--dxf", type=str, default=None)
    ap.add_argument("--timeout", type=float, default=420.0)
    ap.add_argument("--max-iters", type=int, default=18)

    ap.add_argument("--scan-keys", type=str, default="PF2,PF3,PF4,PF5,PF6")
    ap.add_argument("--fixed-keys", type=str, default="CS,PF1")

    ap.add_argument("--null-mode", type=str, default="double", choices=["lower", "upper", "double"])
    ap.add_argument("--enforce-inner", action="store_true")
    ap.add_argument("--inner-tol", type=float, default=0.005)
    ap.add_argument("--inner-hard", action="store_true")

    ap.add_argument("--fd-step-ma", type=float, default=0.50, help="Finite-diff step [MA]")
    ap.add_argument("--max-step-ma", type=float, default=0.30, help="Per-iter step clip [MA]")

    ap.add_argument("--lambda0", type=float, default=1e-1)
    ap.add_argument("--lambda-up", type=float, default=10.0)
    ap.add_argument("--lambda-down", type=float, default=0.3)

    ap.add_argument("--no-blanket", action="store_true")
    ap.add_argument("--fast", action="store_true")

    args = ap.parse_args()

    import config_star_bean as cfg

    scan_keys = [s.strip() for s in str(args.scan_keys).split(",") if s.strip()]
    fixed_keys = [s.strip() for s in str(args.fixed_keys).split(",") if s.strip()]
    all_keys = []
    for k in fixed_keys + scan_keys:
        if k not in all_keys:
            all_keys.append(k)

    # initial currents from cfg
    x0 = _get_currents_from_cfg(cfg)
    xA = {k: float(x0.get(k, 0.0)) for k in all_keys}

    # bounds
    bounds_A = _make_bounds_A(cfg, scan_keys, fixed_keys)
    xA = _clamp_A(xA, bounds_A)

    # reference for regularization = initial
    Iref_A = dict(xA)

    # overrides for speed/robustness
    cfg_overrides: Dict[str, Any] = {}
    if args.no_blanket:
        cfg_overrides["blanket_enabled"] = False
        cfg_overrides["blanket_n_filaments"] = 0
    if args.fast:
        cfg_overrides["f_list_equilibrium"] = (0.25, 0.55, 0.80, 1.00)
        cfg_overrides["target_rel_tol_ramp"] = 5e-5
        cfg_overrides["target_rel_tol"] = 2e-5

    # logging
    log_jsonl = _results_dir() / "inverse_seed_log.jsonl"
    best_json = _results_dir() / "inverse_seed_best.json"

    lam = float(args.lambda0)
    fd_hA = float(args.fd_step_ma) * 1e6
    max_step_A = float(args.max_step_ma) * 1e6

    best = {"f": float("inf"), "xA": dict(xA), "info": None}

    def eval_r(xA_try: Dict[str, float]) -> Dict[str, Any]:
        # clamp
        xA_try = _clamp_A(xA_try, bounds_A)
        # ensure fixed keys are present
        for k in fixed_keys:
            if k not in xA_try:
                xA_try[k] = float(xA.get(k, 0.0))
        return eval_residuals_timeout(
            xA_try,
            dxf_path=args.dxf,
            timeout_s=float(args.timeout),
            cfg_overrides=cfg_overrides,
            scan_keys=scan_keys,
            fixed_keys=fixed_keys,
            Iref_A=Iref_A,
            bounds_A=bounds_A,
            null_mode=str(args.null_mode),
            enforce_inner=bool(args.enforce_inner),
            inner_tol=float(args.inner_tol),
            inner_hard=bool(args.inner_hard),
        )

    # initial eval
    r0res = eval_r(xA)
    if not r0res.get("ok", False):
        raise RuntimeError(f"Initial eval failed: {r0res.get('error')}")

    r = np.asarray(r0res["r"], float)
    f = float(np.dot(r, r))

    _append_jsonl(log_jsonl, {"event": "init", "xA": xA, "f": f, "info": r0res.get("info", {})})
    best = {"f": f, "xA": dict(xA), "info": r0res.get("info", {})}

    print("\n[INIT] f =", f)
    print("[INIT] currents [MA]:", {k: xA[k] / 1e6 for k in all_keys})

    for it in range(1, int(args.max_iters) + 1):
        t_it0 = time.time()

        # Build Jacobian wrt scan_keys only
        nvar = len(scan_keys)
        rdim = r.size
        J = np.zeros((rdim, nvar), float)

        # baseline
        x_base = dict(xA)
        r_base = r.copy()

        # FD columns
        for j, key in enumerate(scan_keys):
            xp = dict(x_base)
            # forward step with clamp; if clamp kills perturbation, try negative
            xp[key] = float(xp.get(key, 0.0)) + fd_hA
            xp = _clamp_A(xp, bounds_A)
            if abs(xp[key] - x_base[key]) < 1e-9:
                xp[key] = float(x_base[key]) - fd_hA
                xp = _clamp_A(xp, bounds_A)

            res_p = eval_r(xp)
            if not res_p.get("ok", False):
                # if eval fails, set column ~0 (LM will rely on other directions)
                J[:, j] = 0.0
                _append_jsonl(log_jsonl, {"event": "fd_fail", "iter": it, "key": key, "error": res_p.get("error")})
                continue

            rp = np.asarray(res_p["r"], float)
            dh = float(xp[key] - x_base[key])
            if abs(dh) < 1e-12:
                J[:, j] = 0.0
            else:
                J[:, j] = (rp - r_base) / dh

        # Solve LM step: (J^T J + lam I) dx = -J^T r
        A = J.T @ J
        g = J.T @ r_base
        A_lm = A + lam * np.eye(nvar)

        try:
            dx = -np.linalg.solve(A_lm, g)
        except Exception:
            dx = -np.linalg.lstsq(A_lm, g, rcond=None)[0]

        # clip per-variable step
        for j in range(nvar):
            dx[j] = float(np.clip(dx[j], -max_step_A, +max_step_A))

        # propose update
        x_try = dict(xA)
        for j, key in enumerate(scan_keys):
            x_try[key] = float(x_try.get(key, 0.0)) + float(dx[j])
        x_try = _clamp_A(x_try, bounds_A)

        # accept/reject with lambda adaptation
        accepted = False
        res_try = None
        f_try = float("inf")

        for attempt in range(4):
            res_try = eval_r(x_try)
            if res_try.get("ok", False):
                r_try = np.asarray(res_try["r"], float)
                f_try = float(np.dot(r_try, r_try))
            else:
                f_try = float("inf")

            if f_try < f:
                accepted = True
                break

            # reject -> increase lambda and shrink step
            lam *= float(args.lambda_up)
            dx *= 0.5
            x_try = dict(xA)
            for j, key in enumerate(scan_keys):
                x_try[key] = float(x_try.get(key, 0.0)) + float(dx[j])
            x_try = _clamp_A(x_try, bounds_A)

        if accepted and res_try is not None:
            xA = dict(x_try)
            r = np.asarray(res_try["r"], float)
            f = float(f_try)
            lam = max(1e-12, lam * float(args.lambda_down))

            # update best
            if f < float(best["f"]):
                best = {"f": f, "xA": dict(xA), "info": res_try.get("info", {})}
                _save_json(best_json, {
                    "best_f": float(best["f"]),
                    "currents_MA": {k: float(best["xA"][k] / 1e6) for k in all_keys},
                    "currents_A": {k: float(best["xA"][k]) for k in all_keys},
                    "info": best["info"],
                    "bounds_MA": {k: [bounds_A[k][0] / 1e6, bounds_A[k][1] / 1e6] for k in all_keys if k in bounds_A},
                    "scan_keys": list(scan_keys),
                    "fixed_keys": list(fixed_keys),
                    "cfg_overrides": cfg_overrides,
                })

        t_it = time.time() - t_it0

        _append_jsonl(log_jsonl, {
            "event": "iter",
            "iter": it,
            "accepted": bool(accepted),
            "lambda": float(lam),
            "f": float(f),
            "f_try": float(f_try),
            "dx_MA": {scan_keys[j]: float(dx[j] / 1e6) for j in range(nvar)},
            "x_MA": {k: float(xA[k] / 1e6) for k in all_keys},
            "info": (res_try.get("info", {}) if isinstance(res_try, dict) else {}),
            "iter_time_s": float(t_it),
        })

        print(f"\n[ITER {it}] accepted={accepted}  f={f:.6g}  lambda={lam:.3g}  t={t_it:.1f}s")
        print("  dx [MA]:", {scan_keys[j]: float(dx[j] / 1e6) for j in range(nvar)})
        print("  x  [MA]:", {k: float(xA[k] / 1e6) for k in all_keys})

    print("\n[DONE] Best f =", float(best["f"]))
    print("[DONE] Best currents [MA]:", {k: float(best["xA"][k] / 1e6) for k in all_keys})
    print("[SAVED]", best_json)
    print("[LOG]  ", log_jsonl)

    print("\n--- Paste into config_star_bean.py ---")
    for k in all_keys:
        print(f"{k}_current = {best['xA'][k]:.12g}")


if __name__ == "__main__":
    main()
