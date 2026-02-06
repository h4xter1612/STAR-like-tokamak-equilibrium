"""
scan_star_refine.py

Deterministic + adaptive local refinement of STAR coil FAMILY currents (CS, PF1, PF2, PF3)
using a pattern-search / coordinate-descent scheme with step-size shrink.

NEW in this version:
- Per-iteration parallel evaluation of neighbors (deterministic decision after all complete).
- Robust IPC using multiprocessing.Pipe (reduces "no_result_from_worker" issues).
- CAD-aware objective:
    * matches equilibrium plasma to CAD plasma_target (geom["R_plasma","Z_plasma"] if available)
    * matches magnetic X-point(s) to CAD xpoints_target (geom["xpoints_target"])
    * uses scalar targets from geom["plasma_auto_meta"] (R0/A/kappa/delta) as fallback
    * optionally includes a curve-shape term when separatrix polyline exists in `shape`

Fixes in this patch:
- fallback_lcfs penalty is applied ONLY when explicitly flagged True (default False),
  and the flag is logged in obj["fallback_lcfs"] for grepping.
- dx_upper_m / dx_lower_m are None when not applicable (instead of NaN), printed as "n/a".

Usage examples (Windows):
  py .\scan_star_refine.py --init .\results\scan_multigoal_best_global.json --objective cad --null-mode lower --iter-workers 4 --max-iters 25 --timeout 140 --step0 0.3,0.3,0.3,0.3 --min-step 0.02 --verbose-evals

  py .\scan_star_refine.py --objective diag --iter-workers 4 --max-iters 25

Notes:
- Currents from JSON are read as MA and converted to A internally.
- Currents written to config snippet at end are in e6 (A).
"""

from __future__ import annotations

import argparse
import json
import time
import math
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np


# ----------------------------
# Paths / IO helpers
# ----------------------------
def _here() -> Path:
    return Path(__file__).resolve().parent


def _results_dir() -> Path:
    d = _here() / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


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


def _norm2(v: np.ndarray) -> float:
    return float(np.sqrt(float(np.sum(v * v))))


def _fmt_num(x: Any) -> str:
    """Pretty formatter for logs; handles None/NaN safely."""
    if x is None:
        return "n/a"
    try:
        xf = float(x)
        if not np.isfinite(xf):
            return "nan"
        return f"{xf:.3g}"
    except Exception:
        return "n/a"


def _boolish(x: Any) -> Optional[bool]:
    """Try to interpret x as boolean; returns None if unknown."""
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(float(x)):
            return None
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            return True
        if s in ("0", "false", "f", "no", "n", "off"):
            return False
        return None
    return None


def _extract_fallback_lcfs_flag(shape: Dict[str, Any], diag: Dict[str, Any]) -> bool:
    """
    Conservative fallback_lcfs detection.
    - Default False if missing/unknown.
    - True only if explicitly indicated.
    Supports:
      shape["fallback_lcfs"] = True/False
      shape["fallback_lcfs"] = {"fallback": True} / {"is_fallback": True} / {"used_limiter": True} / {"limiter": True}
      diag may also contain fallback_lcfs similarly.
    """
    cand = None
    if isinstance(shape, dict):
        cand = shape.get("fallback_lcfs", None)

    if cand is None and isinstance(diag, dict):
        cand = diag.get("fallback_lcfs", None)

    b = _boolish(cand)
    if b is not None:
        return bool(b)

    if isinstance(cand, dict):
        for kk in ("fallback", "is_fallback", "used_limiter", "limiter", "is_limiter"):
            bb = _boolish(cand.get(kk, None))
            if bb is not None:
                return bool(bb)

        # If there is a textual reason that clearly indicates fallback/limiter usage.
        reason = cand.get("reason", None)
        if isinstance(reason, str):
            r = reason.lower()
            if ("fallback" in r) or ("limiter" in r):
                return True

    # Unknown format -> do NOT penalize by default
    return False


# ----------------------------
# Geometry / curve metrics
# ----------------------------
def _open_curve_from_closed(R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    P = np.column_stack([np.asarray(R, float), np.asarray(Z, float)])
    if P.shape[0] < 3:
        return P
    if np.linalg.norm(P[0] - P[-1]) < 1e-12:
        P = P[:-1]
    return P


def _rms_chamfer_sym(P: np.ndarray, Q: np.ndarray) -> float:
    """Symmetric chamfer RMS distance between point clouds P and Q."""
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    if P.ndim != 2 or Q.ndim != 2 or P.shape[1] != 2 or Q.shape[1] != 2:
        return float("inf")
    if P.shape[0] < 10 or Q.shape[0] < 10:
        return float("inf")

    d2_pq = []
    for p in P:
        d2_pq.append(float(np.min(np.sum((Q - p) ** 2, axis=1))))
    d2_qp = []
    for q in Q:
        d2_qp.append(float(np.min(np.sum((P - q) ** 2, axis=1))))

    return float(math.sqrt(0.5 * (float(np.mean(d2_pq)) + float(np.mean(d2_qp)))))


def _extract_lcfs_curve_from_shape(shape: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Try to pull a separatrix/LCFS polyline from shape.
    If your analyze module uses different keys, add them here.
    """
    key_pairs = [
        ("R_sep", "Z_sep"),
        ("R_separatrix", "Z_separatrix"),
        ("R_lcfs", "Z_lcfs"),
        ("lcfs_R", "lcfs_Z"),
        ("R_LCFS", "Z_LCFS"),
    ]
    for rkey, zkey in key_pairs:
        if rkey in shape and zkey in shape:
            P = _open_curve_from_closed(np.asarray(shape[rkey], float), np.asarray(shape[zkey], float))
            if P.shape[0] >= 10:
                return P

    for k in ("lcfs", "separatrix", "sep"):
        v = shape.get(k, None)
        if isinstance(v, dict):
            for rkey, zkey in key_pairs:
                if rkey in v and zkey in v:
                    P = _open_curve_from_closed(np.asarray(v[rkey], float), np.asarray(v[zkey], float))
                    if P.shape[0] >= 10:
                        return P
    return None


def _extract_xpoints(shape: Dict[str, Any], diag: Dict[str, Any]) -> List[Tuple[float, float]]:
    """
    Returns list of magnetic X-points (R,Z). Tries several formats.
    """
    out: List[Tuple[float, float]] = []

    # 1) shape-based
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

    if out:
        return out

    # 2) diag-based fallback (if your diagnostics store them)
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


def _targets_from_geom(geom: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Dict[str, Tuple[float, float]], Dict[str, float]]:
    """
    Returns:
      - target curve points (N,2) from geom["R_plasma"], geom["Z_plasma"] if present
      - xpoint targets dict {"lower":(R,Z), "upper":(R,Z)} from geom["xpoints_target"] if present
      - scalar target dict from geom["plasma_auto_meta"] if present
    """
    target_curve = None
    if "R_plasma" in geom and "Z_plasma" in geom:
        target_curve = _open_curve_from_closed(np.asarray(geom["R_plasma"], float), np.asarray(geom["Z_plasma"], float))

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


# ----------------------------
# Objectives
# ----------------------------
def _misfit_diag(diag: Dict[str, Any], cfg: Any) -> Tuple[float, Dict[str, Any]]:
    """
    Legacy/diag objective.
    Uses scalar parameters from diag and targets from cfg (R0_geom, A_geom, kappa_geom, delta_geom).
    """
    if not diag or not diag.get("ok", False):
        return 1e9, {"mode": "diag", "ok": False}

    R0_t = _safe_float(getattr(cfg, "R0_geom", 4.0), 4.0)
    A_t  = _safe_float(getattr(cfg, "A_geom",  2.0), 2.0)
    k_t  = _safe_float(getattr(cfg, "kappa_geom", 2.0), 2.0)
    d_t  = _safe_float(getattr(cfg, "delta_geom", 0.3), 0.3)

    R0 = _safe_float(diag.get("R0", diag.get("R0_plasma", float("nan"))))
    A  = _safe_float(diag.get("A",  diag.get("A_plasma",  float("nan"))))
    k  = _safe_float(diag.get("kappa", diag.get("kappa_plasma", float("nan"))))
    du = _safe_float(diag.get("delta_u", float("nan")))
    dl = _safe_float(diag.get("delta_l", float("nan")))
    a  = _safe_float(diag.get("a", diag.get("a_plasma", float("nan"))))

    if not np.isfinite(R0 + A + k + du + dl + a):
        return 1e9, {"mode": "diag", "ok": False}

    dbar = 0.5 * (du + dl)

    sig_R0 = float(getattr(cfg, "sig_R0_m", 0.25))
    sig_A  = float(getattr(cfg, "sig_A",    0.25))
    sig_k  = float(getattr(cfg, "sig_kappa",0.25))
    sig_d  = float(getattr(cfg, "sig_delta",0.20))

    term = 0.0
    term += ((R0 - R0_t) / sig_R0) ** 2
    term += ((A  - A_t)  / sig_A)  ** 2
    term += ((k  - k_t)  / sig_k)  ** 2
    term += ((dbar - d_t)/ sig_d)  ** 2

    misfit = float(math.sqrt(term))

    # penalties
    pen_neg = float(getattr(cfg, "penalty_neg_delta", 10.0))
    if du < 0.0 or dl < 0.0:
        misfit += pen_neg * (abs(min(du, 0.0)) + abs(min(dl, 0.0)))

    a_min = float(getattr(cfg, "a_min_m", 1.0))
    pen_thin = float(getattr(cfg, "penalty_thin", 5.0))
    if a < a_min:
        misfit += pen_thin * (a_min - a)

    Rax = _safe_float(diag.get("R_ax", float("nan")))
    if np.isfinite(Rax):
        sig_Rax = float(getattr(cfg, "sig_Rax_m", 0.30))
        misfit += abs((Rax - R0_t) / sig_Rax) * float(getattr(cfg, "w_Rax", 0.5))

    info = {
        "mode": "diag",
        "ok": True,
        "R0": R0, "A": A, "kappa": k, "delta_u": du, "delta_l": dl,
        "target_R0": R0_t, "target_A": A_t, "target_kappa": k_t, "target_delta": d_t,
        "dbar": dbar,
    }
    return float(misfit), info


def _misfit_cad(geom: Dict[str, Any], shape: Dict[str, Any], diag: Dict[str, Any], cfg: Any, null_mode: str) -> Tuple[float, Dict[str, Any]]:
    """
    CAD objective:
      - scalar mismatch vs CAD plasma_auto_meta targets (midplane R0/A/kappa/delta) if present
      - xpoint mismatch vs geom["xpoints_target"]
      - optional curve mismatch if separatrix curve is available in shape
    """
    # knobs
    sig_R0 = float(getattr(cfg, "sig_R0_m", 0.25))
    sig_A  = float(getattr(cfg, "sig_A",    0.25))
    sig_k  = float(getattr(cfg, "sig_kappa",0.25))
    sig_d  = float(getattr(cfg, "sig_delta",0.20))

    sig_x     = float(getattr(cfg, "sig_x_m", 0.20))
    sig_shape = float(getattr(cfg, "sig_shape_m", 0.05))

    w_scalar = float(getattr(cfg, "w_scalar", 1.0))
    w_x      = float(getattr(cfg, "w_x", 1.0))
    w_shape  = float(getattr(cfg, "w_shape", 1.0))

    penalty_no_sep   = float(getattr(cfg, "penalty_no_separatrix", 1e6))
    penalty_no_xp    = float(getattr(cfg, "penalty_no_xpoints", 1e6))
    penalty_fallback = float(getattr(cfg, "penalty_fallback_lcfs", 5e4))

    # targets from geom
    target_curve, xt, scal = _targets_from_geom(geom)
    if (not diag) or (not diag.get("ok", False)):
        return 1e9, {"mode": "cad", "ok": False, "reason": "diag_not_ok"}

    # require separatrix?
    ok_sep = bool(shape.get("ok_sep", False))
    if not ok_sep:
        # diverted requirement: hard fail
        return penalty_no_sep, {"mode": "cad", "ok": False, "reason": "no_separatrix"}

    # scalar term: use CAD midplane targets if present, else fallback to cfg targets
    # CAD
    R0_t = scal.get("geom_R0_mid", float("nan"))
    A_t  = scal.get("geom_A_mid",  float("nan"))
    k_t  = scal.get("geom_kappa_mid", float("nan"))
    du_t = scal.get("geom_delta_u", float("nan"))
    dl_t = scal.get("geom_delta_l", float("nan"))

    # fallback
    if not np.isfinite(R0_t):
        R0_t = _safe_float(getattr(cfg, "R0_geom", 4.0), 4.0)
    if not np.isfinite(A_t):
        A_t = _safe_float(getattr(cfg, "A_geom", 2.0), 2.0)
    if not np.isfinite(k_t):
        k_t = _safe_float(getattr(cfg, "kappa_geom", 2.0), 2.0)
    if not np.isfinite(du_t) or not np.isfinite(dl_t):
        d_t = _safe_float(getattr(cfg, "delta_geom", 0.3), 0.3)
        du_t, dl_t = d_t, d_t

    # equilibrium scalars
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

    # xpoint term
    xps = _extract_xpoints(shape, diag)
    if not xps:
        return penalty_no_xp, {"mode": "cad", "ok": False, "reason": "no_xpoints"}

    lower_xp, upper_xp = _pick_lower_upper_xp(xps)

    null_mode = str(null_mode).strip().lower()
    if null_mode not in ("lower", "upper", "double"):
        null_mode = "lower"

    dx_lower: Optional[float] = None
    dx_upper: Optional[float] = None
    term_x = 0.0

    if null_mode in ("lower", "double"):
        if "lower" not in xt or lower_xp is None:
            return penalty_no_xp, {"mode": "cad", "ok": False, "reason": "missing_lower_target_or_xp"}
        dx_lower = float(math.hypot(lower_xp[0] - xt["lower"][0], lower_xp[1] - xt["lower"][1]))
        term_x += (dx_lower / max(sig_x, 1e-12)) ** 2

    if null_mode in ("upper", "double"):
        if "upper" not in xt or upper_xp is None:
            return penalty_no_xp, {"mode": "cad", "ok": False, "reason": "missing_upper_target_or_xp"}
        dx_upper = float(math.hypot(upper_xp[0] - xt["upper"][0], upper_xp[1] - xt["upper"][1]))
        term_x += (dx_upper / max(sig_x, 1e-12)) ** 2

    term_x *= (w_x ** 2)

    # optional curve term (only if both are available)
    term_shape = 0.0
    shape_rms: Optional[float] = None
    lcfs = _extract_lcfs_curve_from_shape(shape)
    if (lcfs is not None) and (target_curve is not None) and (lcfs.shape[0] >= 10) and (target_curve.shape[0] >= 10):
        sr = _rms_chamfer_sym(lcfs, target_curve)
        if np.isfinite(sr):
            shape_rms = float(sr)
            term_shape = (w_shape * (shape_rms / max(sig_shape, 1e-12))) ** 2

    misfit = float(math.sqrt(term_scalar + term_x + term_shape))

    # penalty if fallback_lcfs indicates limiter-based LCFS
    fallback_lcfs = _extract_fallback_lcfs_flag(shape=shape, diag=diag)
    if fallback_lcfs:
        misfit += penalty_fallback

    # optional: negative triangularity penalty
    pen_neg = float(getattr(cfg, "penalty_neg_delta", 10.0))
    if du < 0.0 or dl < 0.0:
        misfit += pen_neg * (abs(min(du, 0.0)) + abs(min(dl, 0.0)))

    info = {
        "mode": "cad",
        "ok": True,
        "null_mode": null_mode,
        "scalar_targets": {"R0": R0_t, "A": A_t, "kappa": k_t, "dbar": dbar_t},
        "scalar_vals": {"R0": R0, "A": A, "kappa": k, "dbar": dbar},
        "dx_lower_m": dx_lower,
        "dx_upper_m": dx_upper,
        "shape_rms_m": shape_rms,
        "term_scalar": float(term_scalar),
        "term_x": float(term_x),
        "term_shape": float(term_shape),
        "ok_sep": bool(ok_sep),
        "n_xpoints": int(len(xps)),
        "fallback_lcfs": bool(fallback_lcfs),
    }
    return float(misfit), info


# ----------------------------
# Subprocess worker (single eval)
# ----------------------------
def _worker_eval(
    currents_A: Dict[str, float],
    dxf_path: Optional[str],
    cfg_overrides: Dict[str, Any],
    require_sep: bool,
    objective: str,
    null_mode: str,
    redirect_solver_noise: bool,
    conn,
) -> None:
    """
    Runs in a child process:
      - patches cfg currents
      - calls star_equilibrium.build_equilibrium()
      - computes misfit
      - sends result via pipe
    """
    try:
        import config_star_bean as cfg
        import star_equilibrium as se

        # apply overrides
        for k, v in (cfg_overrides or {}).items():
            try:
                setattr(cfg, k, v)
            except Exception:
                pass

        # patch currents (A)
        cfg.CS_current  = float(currents_A.get("CS", 0.0))
        cfg.PF1_current = float(currents_A.get("PF1", 0.0))
        cfg.PF2_current = float(currents_A.get("PF2", 0.0))
        cfg.PF3_current = float(currents_A.get("PF3", 0.0))

        # solve
        eq, tokamak, geom, shape = se.build_equilibrium(
            verbose=False,
            redirect_solver_noise=bool(redirect_solver_noise),
            dxf_path=dxf_path,
        )

        # diagnostics dict (prefer stored one)
        diag = shape.get("plasma_diag", None) or {}
        if (not diag) or (not diag.get("ok", False)):
            try:
                diag = se.plasma_diagnostics(eq, geom, shape)
            except Exception:
                diag = {"ok": False}

        # separatrix requirement
        ok_sep = bool(shape.get("ok_sep", False))
        if require_sep and (not ok_sep):
            res = {
                "ok": True,
                "ok_solve": True,
                "require_sep_failed": True,
                "misfit": 1e8,
                "currents_A": dict(currents_A),
                "diag": diag,
                "shape_ok_sep": ok_sep,
                "objective": objective,
            }
            conn.send(res)
            conn.close()
            return

        objective = str(objective).strip().lower()
        if objective == "cad":
            misfit, objinfo = _misfit_cad(geom=geom, shape=shape, diag=diag, cfg=cfg, null_mode=null_mode)
        else:
            misfit, objinfo = _misfit_diag(diag=diag, cfg=cfg)

        res = {
            "ok": True,
            "ok_solve": True,
            "misfit": float(misfit),
            "currents_A": dict(currents_A),
            "diag": diag,
            "shape_ok_sep": ok_sep,
            "objective": objective,
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
                "objective": objective,
            })
            conn.close()
        except Exception:
            pass


def eval_case(
    currents_A: Dict[str, float],
    *,
    dxf_path: Optional[str],
    timeout_s: float,
    require_sep: bool,
    cfg_overrides: Dict[str, Any],
    objective: str,
    null_mode: str,
    redirect_solver_noise: bool,
) -> Dict[str, Any]:
    """
    Parent wrapper: spawn subprocess + hard timeout + robust recv.
    """
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(
        target=_worker_eval,
        args=(
            dict(currents_A),
            dxf_path,
            dict(cfg_overrides or {}),
            bool(require_sep),
            str(objective),
            str(null_mode),
            bool(redirect_solver_noise),
            send_conn,
        ),
    )

    t0 = time.time()
    p.start()
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

    # process exited
    res: Dict[str, Any]
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
# Deterministic adaptive refine (parallel neighbor eval)
# ----------------------------
def refine(
    x0_A: Dict[str, float],
    *,
    dxf_path: Optional[str],
    timeout_s: float,
    require_sep: bool,
    cfg_overrides: Dict[str, Any],
    objective: str,
    null_mode: str,
    redirect_solver_noise: bool,
    step0_MA: Dict[str, float],
    shrink: float,
    min_step_MA: float,
    max_iters: int,
    bounds_MA: Dict[str, Tuple[float, float]],
    iter_workers: int,
    verbose_evals: bool,
    log_jsonl: Path,
) -> Dict[str, Any]:

    keys = ["CS", "PF1", "PF2", "PF3"]

    def clamp(curr: Dict[str, float]) -> Dict[str, float]:
        out = dict(curr)
        for k in keys:
            lo, hi = bounds_MA[k]
            out[k] = float(np.clip(out[k] / 1e6, lo, hi)) * 1e6
        return out

    def to_MA(currA: Dict[str, float]) -> Dict[str, float]:
        return {k: float(currA[k] / 1e6) for k in keys}

    x = clamp(dict(x0_A))
    step_MA = {k: float(step0_MA[k]) for k in keys}

    # Evaluate start
    best = eval_case(
        x,
        dxf_path=dxf_path,
        timeout_s=timeout_s,
        require_sep=require_sep,
        cfg_overrides=cfg_overrides,
        objective=objective,
        null_mode=null_mode,
        redirect_solver_noise=redirect_solver_noise,
    )
    best_m = float(best.get("misfit", 1e9)) if best.get("ok_solve", False) else 1e9

    _append_jsonl(log_jsonl, {"event": "init", "x_A": x, "step_MA": step_MA, "res": best})

    print("\n[INIT]")
    print(to_MA(x), "MA")
    print("misfit =", best_m)

    if not best.get("ok_solve", False):
        print("[WARN] init solve failed; refine may just shrink steps. Check config/DXF/targets.")

    improve_eps = float(getattr(__import__("config_star_bean"), "improve_eps", 0.0)) if _here().joinpath("config_star_bean.py").exists() else 0.0
    if not np.isfinite(improve_eps) or improve_eps <= 0:
        improve_eps = 1e-9

    for it in range(1, max_iters + 1):
        # Build deterministic move list
        moves: List[Tuple[int, str, Dict[str, float]]] = []
        order = 0
        for k in keys:
            hA = step_MA[k] * 1e6
            if hA <= min_step_MA * 1e6:
                continue
            for sgn in (+1.0, -1.0):
                xt = dict(x)
                xt[k] = xt[k] + sgn * hA
                xt = clamp(xt)
                tag = f"{k}{'+' if sgn > 0 else '-'}"
                moves.append((order, tag, xt))
                order += 1

        if not moves:
            print(f"[STOP] no active moves above min-step={min_step_MA} MA")
            break

        # Evaluate all neighbors in parallel (bounded by iter_workers)
        results: List[Tuple[int, str, Dict[str, Any], float]] = []

        t_iter0 = time.time()
        if verbose_evals:
            print(f"\n[ITER {it}] evaluating {len(moves)} neighbors with iter_workers={iter_workers} ...")

        with ThreadPoolExecutor(max_workers=max(1, int(iter_workers))) as ex:
            futs = {}
            for idx, tag, xt in moves:
                fut = ex.submit(
                    eval_case,
                    xt,
                    dxf_path=dxf_path,
                    timeout_s=timeout_s,
                    require_sep=require_sep,
                    cfg_overrides=cfg_overrides,
                    objective=objective,
                    null_mode=null_mode,
                    redirect_solver_noise=redirect_solver_noise,
                )
                futs[fut] = (idx, tag, xt)

            for fut in as_completed(futs):
                idx, tag, xt = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"ok": False, "ok_solve": False, "error": f"future_failed:{repr(e)}", "currents_A": xt}

                m = float(res.get("misfit", 1e9)) if res.get("ok_solve", False) else 1e9
                results.append((idx, tag, res, m))

                _append_jsonl(
                    log_jsonl,
                    {"event": "eval", "iter": it, "order": idx, "move": tag, "x_A": xt, "step_MA": dict(step_MA), "res": res},
                )

                if verbose_evals:
                    ok = res.get("ok_solve", False)
                    el = float(res.get("elapsed_s", float("nan")))
                    extra = ""
                    obj = res.get("obj", {}) if isinstance(res.get("obj", {}), dict) else {}
                    if isinstance(obj, dict) and obj.get("mode") == "cad" and obj.get("ok", False):
                        dxl = obj.get("dx_lower_m", None)
                        dxu = obj.get("dx_upper_m", None)
                        sh  = obj.get("shape_rms_m", None)
                        fb  = obj.get("fallback_lcfs", None)
                        extra = f" | dxL={_fmt_num(dxl)} dxU={_fmt_num(dxu)} shapeRMS={_fmt_num(sh)} fb={fb}"
                    elif isinstance(obj, dict) and obj.get("mode") == "diag" and obj.get("ok", False):
                        extra = f" | R0={_fmt_num(obj.get('R0', None))} A={_fmt_num(obj.get('A', None))} k={_fmt_num(obj.get('kappa', None))}"
                    print(f"  - {tag:4s} ok={ok} misfit={m:.6g} t={el:5.1f}s{extra}")

        # Choose best candidate deterministically:
        #  - minimal misfit
        #  - tie-break by lowest order index
        results.sort(key=lambda t: (t[3], t[0]))
        idx0, tag0, res0, m0 = results[0]

        improved = (m0 + improve_eps) < best_m
        t_iter = time.time() - t_iter0

        if improved:
            x = clamp(dict(res0.get("currents_A", x)))
            best = res0
            best_m = float(m0)
            print(f"\n[ITER {it}] improved ({tag0}) -> misfit={best_m:.6g} @ {to_MA(x)} MA | iter_time={t_iter:.1f}s")
            _append_jsonl(log_jsonl, {"event": "accept", "iter": it, "move": tag0, "x_A": x, "best_misfit": best_m})
        else:
            # shrink steps
            for k in keys:
                step_MA[k] *= float(shrink)
            print(f"\n[ITER {it}] no improvement -> shrink steps: {step_MA} | iter_time={t_iter:.1f}s")
            _append_jsonl(log_jsonl, {"event": "shrink", "iter": it, "step_MA": dict(step_MA), "best_misfit": best_m})

            if max(step_MA.values()) < float(min_step_MA):
                print(f"[STOP] max(step) < {min_step_MA} MA")
                break

    return {
        "objective": str(objective),
        "null_mode": str(null_mode),
        "best_currents_MA": to_MA(x),
        "best_currents_A": x,
        "best_misfit": float(best_m),
        "best_result": best,
        "final_steps_MA": step_MA,
        "log_jsonl": str(log_jsonl),
    }


# ----------------------------
# CLI
# ----------------------------
def main():
    mp.freeze_support()

    ap = argparse.ArgumentParser()
    ap.add_argument("--dxf", type=str, default=None, help="DXF path (defaults to cfg.dxf_path inside star_equilibrium)")
    ap.add_argument("--init", type=str, default=None, help="Init JSON (e.g. scan_multigoal_best_global.json). Reads currents_MA if present.")
    ap.add_argument("--timeout", type=float, default=140.0, help="Hard timeout per eval [s]")
    ap.add_argument("--max-iters", type=int, default=30)
    ap.add_argument("--shrink", type=float, default=0.5)
    ap.add_argument("--min-step", type=float, default=0.02, help="Min step per-family [MA]")
    ap.add_argument("--step0", type=str, default="0.30,0.30,0.30,0.30", help="Initial steps MA: CS,PF1,PF2,PF3")
    ap.add_argument("--bounds", type=str, default="-5,5,-5,5,-5,5,-5,5",
                    help="Bounds MA: CS_lo,CS_hi, PF1_lo,PF1_hi, PF2_lo,PF2_hi, PF3_lo,PF3_hi")

    ap.add_argument("--objective", type=str, default="cad", choices=["cad", "diag"],
                    help="Objective: 'cad' uses plasma_target + xpoints_target from CAD; 'diag' uses cfg scalar targets.")
    ap.add_argument("--null-mode", type=str, default="lower", choices=["lower", "upper", "double"],
                    help="Which X-point(s) to match for CAD objective.")
    ap.add_argument("--require-sep", action="store_true", help="Penalize cases without separatrix (diverted requirement).")

    ap.add_argument("--iter-workers", type=int, default=4, help="Max concurrent evals per iteration.")
    ap.add_argument("--verbose-evals", action="store_true", help="Print each neighbor eval summary.")

    ap.add_argument("--show-solver", action="store_true", help="Do NOT redirect solver noise (more verbose).")
    ap.add_argument("--no-blanket", action="store_true", help="Temporarily disable blanket during refine (speed; changes physics).")
    ap.add_argument("--fast", action="store_true", help="Temporarily use a shorter continuation list (speed; may reduce robustness).")
    ap.add_argument("--tag", type=str, default=None, help="Optional run tag for filenames.")

    args = ap.parse_args()

    import config_star_bean as cfg

    # Load x0 from config (A)
    x0_A = {
        "CS":  float(getattr(cfg, "CS_current", 0.0)),
        "PF1": float(getattr(cfg, "PF1_current", 0.0)),
        "PF2": float(getattr(cfg, "PF2_current", 0.0)),
        "PF3": float(getattr(cfg, "PF3_current", 0.0)),
    }

    # If init JSON is provided, override x0 if it has currents_MA
    if args.init:
        j = _load_json(args.init)
        cm = j.get("currents_MA", None)
        if isinstance(cm, dict) and all(k in cm for k in ("CS", "PF1", "PF2", "PF3")):
            x0_A = {k: float(cm[k]) * 1e6 for k in ("CS", "PF1", "PF2", "PF3")}
            print("[INIT] using currents_MA from JSON:", cm)

    # Parse step0 and bounds
    svals = [float(x.strip()) for x in args.step0.split(",")]
    if len(svals) != 4:
        raise ValueError("--step0 must have 4 values: CS,PF1,PF2,PF3")
    step0_MA = {"CS": svals[0], "PF1": svals[1], "PF2": svals[2], "PF3": svals[3]}

    bvals = [float(x.strip()) for x in args.bounds.split(",")]
    if len(bvals) != 8:
        raise ValueError("--bounds must have 8 values")
    bounds_MA = {
        "CS":  (bvals[0], bvals[1]),
        "PF1": (bvals[2], bvals[3]),
        "PF2": (bvals[4], bvals[5]),
        "PF3": (bvals[6], bvals[7]),
    }

    # Optional overrides inside each subprocess
    cfg_overrides: Dict[str, Any] = {}
    if args.no_blanket:
        cfg_overrides["blanket_enabled"] = False
        cfg_overrides["blanket_n_filaments"] = 0
    if args.fast:
        cfg_overrides["f_list_equilibrium"] = (0.35, 0.70, 1.00)
        cfg_overrides["target_rel_tol_ramp"] = 2e-8

    # Filenames
    tag = args.tag.strip() if args.tag else _ts_tag()
    log_jsonl = _results_dir() / f"scan_refine_{tag}.jsonl"
    best_json = _results_dir() / f"scan_refine_best_{tag}.json"

    # Run refine
    out = refine(
        x0_A,
        dxf_path=args.dxf,
        timeout_s=float(args.timeout),
        require_sep=bool(args.require_sep),
        cfg_overrides=cfg_overrides,
        objective=str(args.objective),
        null_mode=str(args.null_mode),
        redirect_solver_noise=(not bool(args.show_solver)),
        step0_MA=step0_MA,
        shrink=float(args.shrink),
        min_step_MA=float(args.min_step),
        max_iters=int(args.max_iters),
        bounds_MA=bounds_MA,
        iter_workers=int(max(1, args.iter_workers)),
        verbose_evals=bool(args.verbose_evals),
        log_jsonl=log_jsonl,
    )

    _save_json(best_json, out)

    print("\n[DONE] Best refine:")
    print("objective =", out["objective"], "| null_mode =", out["null_mode"])
    print("misfit    =", out["best_misfit"])
    print("currents_MA =", out["best_currents_MA"])
    print("[SAVED]", best_json)
    print("[LOG]  ", log_jsonl)

    # Helpful snippet to paste into config
    cm = out["best_currents_MA"]
    print("\n--- Paste into config_star_bean.py ---")
    print(f"CS_current  = {cm['CS']:.12g}e6")
    print(f"PF1_current = {cm['PF1']:.12g}e6")
    print(f"PF2_current = {cm['PF2']:.12g}e6")
    print(f"PF3_current = {cm['PF3']:.12g}e6")


if __name__ == "__main__":
    main()

