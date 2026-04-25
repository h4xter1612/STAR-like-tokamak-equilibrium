"""
scan_star_refine.py

Deterministic + adaptive local refinement of STAR coil FAMILY currents
using a pattern-search / coordinate-descent scheme with step-size shrink.

Key features:
- Per-iteration parallel evaluation of neighbors (deterministic accept after all complete).
- Robust subprocess eval with hard timeout per case.
- Objectives:
    * "cad": matches equilibrium plasma to CAD plasma_target + xpoints_target
    * "diag": matches scalar targets from config_star_bean.py

NEW (important):
- LCFS-inside-inner-wall constraint (if WALL_INNER exists in geom):
    * Computes frac_out_inner = fraction of LCFS points outside the inner wall polygon
    * If enforce_inner_wall=True:
        - soft: misfit += penalty_outside_inner * frac_out_inner
        - hard: returns penalty_outside_inner_hard if frac_out_inner > tol
    * Logs frac_out_inner inside obj dict, and prints it in verbose-evals.

UPDATED (PF4-6 refine):
- Default scan keys: PF4, PF5, PF6
- Default fixed keys: CS, PF1, PF2, PF3
- The refine varies ONLY scan keys; fixed keys are always passed through unchanged.
- Currents from JSON are read as MA and converted to A internally.
- Currents written to config snippet at end are in e6 (A).

IMPORTANT FIXES:
- JSON-safe writes for json/jsonl (numpy types inside diag/shape no longer crash logging).
- Close send_conn in parent after spawning worker (stability in Windows spawn + Pipe usage).
- FIX: --init no longer overwrites fixed_keys by default (only scan_keys).
  Use --init-use-fixed if you *really* want fixed keys overwritten too.
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
from matplotlib.path import Path as MplPath


DEFAULT_SCAN_KEYS: Tuple[str, ...] = ("PF4", "PF5", "PF6")
DEFAULT_FIXED_KEYS: Tuple[str, ...] = ("CS", "PF1", "PF2", "PF3")


# ----------------------------
# JSON-safe helpers
# ----------------------------
def _to_builtin(obj: Any) -> Any:
    if obj is None:
        return None

    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
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
        f.write(_json_dumps_safe(obj, indent=2))


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(_json_dumps_safe(obj) + "\n")


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


def _fmt_num(x: Any) -> str:
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

        reason = cand.get("reason", None)
        if isinstance(reason, str):
            r = reason.lower()
            if ("fallback" in r) or ("limiter" in r):
                return True

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

    if out:
        return out

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
# Inner wall constraint helpers
# ----------------------------
def _frac_outside(points: np.ndarray, wall_open: np.ndarray, radius: float = -1e-9) -> float:
    P = np.asarray(points, float)
    W = np.asarray(wall_open, float)
    if P.ndim != 2 or W.ndim != 2 or P.shape[1] != 2 or W.shape[1] != 2:
        return float("nan")
    if P.shape[0] < 10 or W.shape[0] < 3:
        return float("nan")
    path = MplPath(W, closed=True)
    inside = path.contains_points(P, radius=float(radius))
    return float(1.0 - np.mean(inside))


# ----------------------------
# Objectives
# ----------------------------
def _misfit_diag(diag: Dict[str, Any], cfg: Any) -> Tuple[float, Dict[str, Any]]:
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

    enforce_inner = bool(getattr(cfg, "enforce_inner_wall", True))
    inner_tol = float(getattr(cfg, "inner_wall_frac_tol", 0.0))
    inner_hard = bool(getattr(cfg, "inner_wall_hard_fail", False))
    pen_out = float(getattr(cfg, "penalty_outside_inner", 5e5))
    pen_out_hard = float(getattr(cfg, "penalty_outside_inner_hard", 1e9))
    inner_radius = float(getattr(cfg, "inner_containment_radius", -1e-9))

    target_curve, xt, scal = _targets_from_geom(geom)
    if (not diag) or (not diag.get("ok", False)):
        return 1e9, {"mode": "cad", "ok": False, "reason": "diag_not_ok"}

    ok_sep = bool(shape.get("ok_sep", False))
    if not ok_sep:
        return penalty_no_sep, {"mode": "cad", "ok": False, "reason": "no_separatrix"}

    lcfs = _extract_lcfs_curve_from_shape(shape)

    R0_t = scal.get("geom_R0_mid", float("nan"))
    A_t  = scal.get("geom_A_mid",  float("nan"))
    k_t  = scal.get("geom_kappa_mid", float("nan"))
    du_t = scal.get("geom_delta_u", float("nan"))
    dl_t = scal.get("geom_delta_l", float("nan"))

    if not np.isfinite(R0_t):
        R0_t = _safe_float(getattr(cfg, "R0_geom", 4.0), 4.0)
    if not np.isfinite(A_t):
        A_t = _safe_float(getattr(cfg, "A_geom", 2.0), 2.0)
    if not np.isfinite(k_t):
        k_t = _safe_float(getattr(cfg, "kappa_geom", 2.0), 2.0)
    if not np.isfinite(du_t) or not np.isfinite(dl_t):
        d_t = _safe_float(getattr(cfg, "delta_geom", 0.3), 0.3)
        du_t, dl_t = d_t, d_t

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

    term_shape = 0.0
    shape_rms: Optional[float] = None
    if (lcfs is not None) and (target_curve is not None) and (lcfs.shape[0] >= 10) and (target_curve.shape[0] >= 10):
        sr = _rms_chamfer_sym(lcfs, target_curve)
        if np.isfinite(sr):
            shape_rms = float(sr)
            term_shape = (w_shape * (shape_rms / max(sig_shape, 1e-12))) ** 2

    misfit = float(math.sqrt(term_scalar + term_x + term_shape))

    fallback_lcfs = _extract_fallback_lcfs_flag(shape=shape, diag=diag)
    if fallback_lcfs:
        misfit += penalty_fallback

    pen_neg = float(getattr(cfg, "penalty_neg_delta", 10.0))
    if du < 0.0 or dl < 0.0:
        misfit += pen_neg * (abs(min(du, 0.0)) + abs(min(dl, 0.0)))

    frac_out_inner: Optional[float] = None
    if enforce_inner and ("R_inner" in geom) and ("Z_inner" in geom):
        wall_inner = _open_curve_from_closed(np.asarray(geom["R_inner"], float), np.asarray(geom["Z_inner"], float))
        if lcfs is None or lcfs.shape[0] < 10 or wall_inner.shape[0] < 3:
            frac_out_inner = None
        else:
            frac = _frac_outside(lcfs, wall_inner, radius=inner_radius)
            frac_out_inner = float(frac) if np.isfinite(frac) else None

        if frac_out_inner is not None and frac_out_inner > float(inner_tol):
            if inner_hard:
                info = {
                    "mode": "cad",
                    "ok": False,
                    "reason": "lcfs_outside_inner",
                    "frac_out_inner": float(frac_out_inner),
                    "inner_tol": float(inner_tol),
                }
                return float(pen_out_hard), info
            misfit += float(pen_out) * float(frac_out_inner)

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
        "frac_out_inner": frac_out_inner,
        "inner_enforced": bool(enforce_inner and ("R_inner" in geom) and ("Z_inner" in geom)),
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
    try:
        import config_star_bean as cfg
        import star_equilibrium as se

        for k, v in (cfg_overrides or {}).items():
            try:
                setattr(cfg, k, v)
            except Exception:
                pass

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
                "obj": {"mode": str(objective), "ok": False, "reason": "require_sep_failed"},
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

    # IMPORTANT: close send end in parent (Windows/spawn stability)
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
    scan_keys: List[str],
    fixed_keys: List[str],
) -> Dict[str, Any]:

    scan_keys = [str(k).strip() for k in (scan_keys or []) if str(k).strip()]
    fixed_keys = [str(k).strip() for k in (fixed_keys or []) if str(k).strip()]

    all_keys: List[str] = []
    for k in fixed_keys + scan_keys:
        if k not in all_keys:
            all_keys.append(k)

    def clamp(curr: Dict[str, float]) -> Dict[str, float]:
        out = dict(curr)
        for k in scan_keys:
            if k in bounds_MA:
                lo, hi = bounds_MA[k]
                out[k] = float(np.clip(out[k] / 1e6, lo, hi)) * 1e6
        return out

    def to_MA(currA: Dict[str, float]) -> Dict[str, float]:
        return {k: float(currA.get(k, 0.0) / 1e6) for k in all_keys}

    x = clamp(dict(x0_A))
    step_MA = {k: float(step0_MA.get(k, 0.0)) for k in scan_keys}

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

    try:
        import config_star_bean as cfg
        improve_eps = float(getattr(cfg, "improve_eps", 1e-9))
    except Exception:
        improve_eps = 1e-9
    if not np.isfinite(improve_eps) or improve_eps <= 0:
        improve_eps = 1e-9

    for it in range(1, max_iters + 1):
        moves: List[Tuple[int, str, Dict[str, float]]] = []
        order = 0
        for k in scan_keys:
            hA = step_MA[k] * 1e6
            if hA <= min_step_MA * 1e6:
                continue
            for sgn in (+1.0, -1.0):
                xt = dict(x)
                xt[k] = xt.get(k, 0.0) + sgn * hA
                xt = clamp(xt)
                tag = f"{k}{'+' if sgn > 0 else '-'}"
                moves.append((order, tag, xt))
                order += 1

        if not moves:
            print(f"[STOP] no active moves above min-step={min_step_MA} MA")
            break

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
                    if isinstance(obj, dict) and obj.get("mode") == "cad":
                        dxl = obj.get("dx_lower_m", None)
                        dxu = obj.get("dx_upper_m", None)
                        sh  = obj.get("shape_rms_m", None)
                        fb  = obj.get("fallback_lcfs", None)
                        out_in = obj.get("frac_out_inner", None)
                        extra = f" | dxL={_fmt_num(dxl)} dxU={_fmt_num(dxu)} shapeRMS={_fmt_num(sh)} outIn={_fmt_num(out_in)} fb={fb}"
                    elif isinstance(obj, dict) and obj.get("mode") == "diag" and obj.get("ok", False):
                        extra = f" | R0={_fmt_num(obj.get('R0', None))} A={_fmt_num(obj.get('A', None))} k={_fmt_num(obj.get('kappa', None))}"
                    print(f"  - {tag:6s} ok={ok} misfit={m:.6g} t={el:5.1f}s{extra}")

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
            for k in scan_keys:
                step_MA[k] *= float(shrink)
            print(f"\n[ITER {it}] no improvement -> shrink steps: {step_MA} | iter_time={t_iter:.1f}s")
            _append_jsonl(log_jsonl, {"event": "shrink", "iter": it, "step_MA": dict(step_MA), "best_misfit": best_m})

            if (len(scan_keys) > 0) and (max(step_MA.values()) < float(min_step_MA)):
                print(f"[STOP] max(step) < {min_step_MA} MA")
                break

    return {
        "objective": str(objective),
        "null_mode": str(null_mode),
        "scan_keys": list(scan_keys),
        "fixed_keys": list(fixed_keys),
        "best_currents_MA": to_MA(x),
        "best_currents_A": x,
        "best_misfit": float(best_m),
        "best_result": best,
        "final_steps_MA": step_MA,
        "log_jsonl": str(log_jsonl),
    }


# ----------------------------
# Init JSON parsing helpers (robust)
# ----------------------------
def _extract_currents_MA_any(j: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Accepts:
      - scan_multigoal outputs: j["currents_MA"]
      - refine outputs: j["best_currents_MA"]
      - legacy: j["best_result"]["currents_A"] (converted)
    """
    if not isinstance(j, dict):
        return None

    cm = j.get("currents_MA", None)
    if isinstance(cm, dict) and cm:
        out = {}
        for k, v in cm.items():
            try:
                out[str(k).strip()] = float(v)
            except Exception:
                pass
        return out if out else None

    cm = j.get("best_currents_MA", None)
    if isinstance(cm, dict) and cm:
        out = {}
        for k, v in cm.items():
            try:
                out[str(k).strip()] = float(v)
            except Exception:
                pass
        return out if out else None

    br = j.get("best_result", None)
    if isinstance(br, dict):
        ca = br.get("currents_A", None)
        if isinstance(ca, dict) and ca:
            out = {}
            for k, v in ca.items():
                try:
                    out[str(k).strip()] = float(v) / 1e6
                except Exception:
                    pass
            return out if out else None

    return None


# ----------------------------
# CLI
# ----------------------------
def main():
    mp.freeze_support()

    ap = argparse.ArgumentParser()
    ap.add_argument("--dxf", type=str, default=None, help="DXF path (defaults to cfg.dxf_path inside star_equilibrium)")
    ap.add_argument("--init", type=str, default=None, help="Init JSON (e.g. scan_multigoal_best_global.json). Reads currents_MA if present.")
    ap.add_argument("--init-use-fixed", action="store_true",
                    help="If set: allow --init to overwrite fixed_keys too. Default: only scan_keys are overwritten.")
    ap.add_argument("--timeout", type=float, default=140.0, help="Hard timeout per eval [s]")
    ap.add_argument("--max-iters", type=int, default=30)
    ap.add_argument("--shrink", type=float, default=0.5)
    ap.add_argument("--min-step", type=float, default=0.02, help="Min step per scanned-family [MA]")

    ap.add_argument("--scan-keys", type=str, default=",".join(DEFAULT_SCAN_KEYS),
                    help="Comma-separated families to scan (default PF4,PF5,PF6)")
    ap.add_argument("--fixed-keys", type=str, default=",".join(DEFAULT_FIXED_KEYS),
                    help="Comma-separated families to keep fixed (default CS,PF1,PF2,PF3)")

    ap.add_argument("--step0", type=str, default="0.30,0.30,0.30",
                    help="Initial steps MA for scan keys, in same order as --scan-keys")
    ap.add_argument("--bounds", type=str, default="-5,5,-5,5,-5,5",
                    help="Bounds MA pairs for scan keys, in same order as --scan-keys: lo1,hi1, lo2,hi2, ...")

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

    scan_keys = [s.strip() for s in str(args.scan_keys).split(",") if s.strip()]
    fixed_keys = [s.strip() for s in str(args.fixed_keys).split(",") if s.strip()]

    all_keys: List[str] = []
    for k in fixed_keys + scan_keys:
        if k not in all_keys:
            all_keys.append(k)

    # base currents from cfg (A)
    x0_A: Dict[str, float] = {}
    for k in all_keys:
        attr = f"{k}_current"
        if hasattr(cfg, attr):
            try:
                x0_A[k] = float(getattr(cfg, attr))
            except Exception:
                x0_A[k] = 0.0
        else:
            x0_A[k] = 0.0

    # init overrides (MA -> A)
    if args.init:
        j = _load_json(args.init)
        cm = _extract_currents_MA_any(j)
        if isinstance(cm, dict) and cm:
            override_keys = list(scan_keys)
            if bool(args.init_use_fixed):
                override_keys = [k for k in (fixed_keys + scan_keys) if k]

            any_hit = False
            used = {}
            for k in override_keys:
                if k in cm:
                    x0_A[k] = float(cm[k]) * 1e6
                    used[k] = float(cm[k])
                    any_hit = True

            if any_hit:
                print("[INIT] using currents (MA) from JSON for keys:", used)
            else:
                print("[INIT] JSON provided but no matching currents found for override keys.")

    svals = [float(x.strip()) for x in args.step0.split(",") if x.strip()]
    if len(svals) != len(scan_keys):
        raise ValueError(f"--step0 must have {len(scan_keys)} values (scan keys order: {scan_keys})")
    step0_MA = {scan_keys[i]: svals[i] for i in range(len(scan_keys))}

    bvals = [float(x.strip()) for x in args.bounds.split(",") if x.strip()]
    if len(bvals) != 2 * len(scan_keys):
        raise ValueError(f"--bounds must have {2*len(scan_keys)} values (pairs for scan keys order: {scan_keys})")
    bounds_MA: Dict[str, Tuple[float, float]] = {}
    for i, k in enumerate(scan_keys):
        bounds_MA[k] = (bvals[2*i], bvals[2*i + 1])

    cfg_overrides: Dict[str, Any] = {}
    if args.no_blanket:
        cfg_overrides["blanket_enabled"] = False
        cfg_overrides["blanket_n_filaments"] = 0
    if args.fast:
        cfg_overrides["f_list_equilibrium"] = (0.35, 0.70, 1.00)
        cfg_overrides["target_rel_tol_ramp"] = 2e-8

    tag = args.tag.strip() if args.tag else _ts_tag()
    log_jsonl = _results_dir() / f"scan_refine_{tag}.jsonl"
    best_json = _results_dir() / f"scan_refine_best_{tag}.json"

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
        scan_keys=list(scan_keys),
        fixed_keys=list(fixed_keys),
    )

    _save_json(best_json, out)

    print("\n[DONE] Best refine:")
    print("objective =", out["objective"], "| null_mode =", out["null_mode"])
    print("scan_keys =", out["scan_keys"], "| fixed_keys =", out["fixed_keys"])
    print("misfit    =", out["best_misfit"])
    print("currents_MA =", out["best_currents_MA"])
    print("[SAVED]", best_json)
    print("[LOG]  ", log_jsonl)

    cm = out["best_currents_MA"]
    print("\n--- Paste into config_star_bean.py ---")
    for k in all_keys:
        print(f"{k}_current = {cm.get(k, 0.0):.12g}e6")


if __name__ == "__main__":
    main()

