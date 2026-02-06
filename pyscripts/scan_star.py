# scan_star_multigoal.py
# Scan equilibria such that:
#   (1) LCFS/separatrix matches CAD plasma target curve (geom["R_plasma"], geom["Z_plasma"])
#   (2) Magnetic X-points lie near CAD geometric xpoint targets (geom["xpoints_target"])
#
# Robust on Windows (spawn), persistent workers + timeout restart, JSONL logging.

from __future__ import annotations

import os
import time
import json
import math
import argparse
from pathlib import Path
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from freegsnke import equilibrium_update, GSstaticsolver
from freegsnke.jtor_update import ConstrainPaxisIp

import config_star_bean as cfg


# -------------------------
# Bounds in MA (family totals)
# -------------------------
BOUNDS_DEFAULT_MA = {
    "CS":  (0.2,  2.5),
    "PF1": (-2.2, 0.8),
    "PF2": (-2.2, 1.4),
    "PF3": (-0.8, 2.8),
}

# -------------------------
# Misfit defaults (tunable)
# -------------------------
PEN_BAD = 200.0
PEN_NO_SEP = 180.0

# Shape (Chamfer) term
SIG_SHAPE_M = 0.05      # 5 cm characteristic scale
W_SHAPE     = 1.00

# X-point term
SIG_X_M     = 0.20      # 20 cm characteristic scale
W_X         = 1.30
PEN_NO_X    = 60.0      # if we want X but none found
PEN_ONLY1X  = 35.0      # if require two but only one found

# Optional current prior regularizer (weak)
W_PRIOR = 0.10  # 0 disables


# -------------------------
# Small helpers
# -------------------------
def _default_dxf() -> str:
    here = Path(__file__).resolve().parent
    return str((here / "cad" / "star_baseline.dxf").resolve())

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _bar(frac: float, width: int = 28) -> str:
    frac = max(0.0, min(1.0, float(frac)))
    k = int(round(frac * width))
    return "[" + "#" * k + "-" * (width - k) + "]"

def _status_line(tag: str, done: int, total: int,
                 solve_ok: int, sep_ok: int,
                 best: float, rate: float, last: Optional[float]) -> str:
    frac = 0.0 if total <= 0 else done / total
    last_s = "" if last is None else f" last={last:8.3f}"
    b = best if np.isfinite(best) else 9999.0
    return (f"{tag:>6s} {_bar(frac)} {done:4d}/{total:<4d} "
            f"solve_ok={solve_ok:4d} sep_ok={sep_ok:4d} "
            f"best={b:8.3f}{last_s} {rate:5.2f}/s")

def _sanitize_mode(x: Any) -> str:
    s = str(x).strip().lower()
    return s if s in ("area", "equal", "same") else "area"

def _get_group_labels(tokamak: Any, fam: str) -> List[str]:
    fam = str(fam).strip().upper()
    groups = getattr(tokamak, "coil_groups", None) or {}
    labs = (
        groups.get(fam, None)
        or groups.get(fam.lower(), None)
        or groups.get(fam.upper(), None)
        or []
    )
    return [str(x).strip().upper() for x in (labs or []) if str(x).strip()]


def print_group_currents_sanity(tokamak: Any, totals_A: Dict[str, float], mode: str, header: str = "") -> Dict[str, Any]:
    """
    Same check as equilibrium, but returns a dict you can store in the JSONL if you want.
    """
    mode = _sanitize_mode(mode)

    # build a coil map compatible with your _iter_coils(tokamak)
    coil_map = {lab: coil for lab, coil in _iter_coils(tokamak)}

    out: Dict[str, Any] = {"mode": mode, "families": {}}

    if header:
        # Optional: only if you want worker prints (usually you don't)
        print(header)

    if not getattr(tokamak, "coil_groups", None):
        out["error"] = "tokamak.coil_groups not present"
        return out

    for fam in ("CS", "PF1", "PF2", "PF3"):
        target = float(totals_A.get(fam, 0.0))
        labs = _get_group_labels(tokamak, fam)
        s = 0.0
        n_ok = 0
        for lab in labs:
            c = coil_map.get(lab)
            if c is None:
                continue
            try:
                s += float(getattr(c, "current"))
                n_ok += 1
            except Exception:
                pass

        denom = max(1.0, abs(target))
        rel = abs(s - target) / denom

        out["families"][fam] = {
            "n_labels": int(len(labs)),
            "n_found": int(n_ok),
            "sum_A": float(s),
            "target_A": float(target),
            "rel_err": float(rel),
        }

    return out

# -------------------------
# FreeGSNKE robustness patches (copy/mask issues)
# -------------------------
def _patch_profiles_copy_once():
    from freegsnke.jtor_update import ConstrainPaxisIp as _C
    if getattr(_C, "_safe_copy_patched", False):
        return
    _orig_copy = _C.copy

    def _safe_copy(self, *args, **kwargs):
        eq = getattr(self, "eq", None) or getattr(self, "_eq", None)
        if eq is not None:
            try:
                mask = np.ones(eq.R.shape, dtype=bool)
            except Exception:
                mask = None
            if mask is not None:
                for name in ("diverted_core_mask", "limiter_core_mask", "core_mask"):
                    try:
                        if getattr(self, name, None) is None:
                            setattr(self, name, mask)
                    except Exception:
                        pass
        return _orig_copy(self, *args, **kwargs)

    _C.copy = _safe_copy
    _C._safe_copy_patched = True

def _patch_copy_into_allow_none_once():
    import freegsnke.copying as _copying
    if getattr(_copying, "_allow_none_patched", False):
        return

    _orig = _copying.copy_into

    def _copy_into_patched(src, dst, name, *args, **kwargs):
        strict = kwargs.get("strict", True)
        if len(args) >= 2:
            strict = args[1]
        try:
            val = getattr(src, name)
        except Exception:
            val = None
        if (not strict) and (val is None):
            try:
                setattr(dst, name, None)
            except Exception:
                pass
            return
        return _orig(src, dst, name, *args, **kwargs)

    _copying.copy_into = _copy_into_patched
    _copying._allow_none_patched = True

    try:
        import freegsnke.jtor_update as _jtor
        _jtor.copy_into = _copy_into_patched
    except Exception:
        pass

def _ensure_profile_masks(profiles: Any, eq: Any) -> None:
    try:
        mask = np.ones(eq.R.shape, dtype=bool)
    except Exception:
        return
    for name in ("diverted_core_mask", "limiter_core_mask", "core_mask"):
        try:
            v = getattr(profiles, name, None)
            if (v is None) or (np.asarray(v).shape != mask.shape):
                setattr(profiles, name, mask.copy())
        except Exception:
            pass


# -------------------------
# Coil family currents application
# -------------------------
def _iter_coils(tokamak: Any):
    for item in getattr(tokamak, "coils", []):
        if isinstance(item, (tuple, list)) and len(item) == 2:
            label, coil = item
        else:
            coil = item
            label = getattr(coil, "label", getattr(coil, "name", ""))
        yield str(label).strip().upper(), coil

def apply_star_family_currents(
    tokamak: Any,
    CS_A: float, PF1_A: float, PF2_A: float, PF3_A: float,
    *, mode: str = "area"
) -> None:
    mode = _sanitize_mode(mode)
    totals = {"CS": float(CS_A), "PF1": float(PF1_A), "PF2": float(PF2_A), "PF3": float(PF3_A)}

    # Best: grouping exists (from your star_machine_cad)
    if hasattr(tokamak, "coil_groups") and getattr(tokamak, "coil_groups", None):
        try:
            from star_machine_cad import apply_group_currents
            apply_group_currents(tokamak, totals, mode=mode)
            return
        except Exception:
            pass

    # Legacy fallback
    mapping = {
        "CS":  float(CS_A),
        "PF1U": float(PF1_A), "PF1L": float(PF1_A),
        "PF2U": float(PF2_A), "PF2L": float(PF2_A),
        "PF3U": float(PF3_A), "PF3L": float(PF3_A),
    }
    for lab, coil in _iter_coils(tokamak):
        if lab in mapping:
            try:
                coil.current = float(mapping[lab])
            except Exception:
                pass


# -------------------------
# Solve with continuation
# -------------------------
def solve_equilibrium(tokamak: Any, geom: Dict[str, Any], *,
                      CS_MA: float, PF1_MA: float, PF2_MA: float, PF3_MA: float,
                      nx: int, ny: int,
                      Ip: float, paxis: float, fvac: float,
                      alpha_m: float, alpha_n: float,
                      target_rel_tol: float,
                      margin_RZ: float,
                      f_list: Tuple[float, ...],
                      coil_group_mode: str,
                      silence_solver: bool) -> Any:

    R_outer = np.asarray(geom["R_outer"], dtype=float)
    Z_outer = np.asarray(geom["Z_outer"], dtype=float)
    Rmin = float(R_outer.min() - margin_RZ)
    Rmax = float(R_outer.max() + margin_RZ)
    Zmin = float(Z_outer.min() - margin_RZ)
    Zmax = float(Z_outer.max() + margin_RZ)

    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=Rmin, Rmax=Rmax,
        Zmin=Zmin, Zmax=Zmax,
        nx=int(nx), ny=int(ny),
    )
    solver = GSstaticsolver.NKGSsolver(eq)

    ctx = None
    if silence_solver:
        import contextlib
        ctx = contextlib.ExitStack()
        devnull = ctx.enter_context(open(os.devnull, "w"))
        ctx.enter_context(contextlib.redirect_stdout(devnull))
        ctx.enter_context(contextlib.redirect_stderr(devnull))

    try:
        for f in f_list:
            f = float(f)

            CS_A  = 1e6 * f * float(CS_MA)
            PF1_A = 1e6 * f * float(PF1_MA)
            PF2_A = 1e6 * f * float(PF2_MA)
            PF3_A = 1e6 * f * float(PF3_MA)

            apply_star_family_currents(tokamak, CS_A, PF1_A, PF2_A, PF3_A, mode=str(coil_group_mode))

            # totals_A = {"CS": CS_A, "PF1": PF1_A, "PF2": PF2_A, "PF3": PF3_A}
            # san = print_group_currents_sanity(tokamak, totals_A, mode=_sanitize_mode(payload.get("coil_group_mode","area")))


            profiles = ConstrainPaxisIp(
                eq=eq,
                paxis=f * float(paxis),
                Ip=f * float(Ip),
                fvac=float(fvac),
                alpha_m=float(alpha_m),
                alpha_n=float(alpha_n),
            )
            _ensure_profile_masks(profiles, eq)

            solver.solve(
                eq=eq,
                profiles=profiles,
                constrain=None,
                target_relative_tolerance=float(target_rel_tol),
                verbose=False,
            )
    finally:
        if ctx is not None:
            ctx.close()

    return eq


# -------------------------
# Shape extraction (your analyze_star.py)
# -------------------------
def extract_shape(eq: Any, geom: Dict[str, Any], *,
                  require_two_x: bool,
                  prefer_inner: bool,
                  psi_percentile: float,
                  null_prefer: str) -> Dict[str, Any]:

    from analyze_star import analyze_star

    p = float(psi_percentile)
    p_use = None if (p <= 0.0) else p

    shp = analyze_star(
        eq, geom,
        require_two_x=bool(require_two_x),
        null_prefer=str(null_prefer),
        prefer_inner_lcfs=bool(prefer_inner),
        psi_percentile_lcfs=p_use,
        edge_pad_cells=2,
    )
    if "fallback_lcfs" not in shp:
        shp["fallback_lcfs"] = None
    return shp


# -------------------------
# Geometry: curve resample + Chamfer distance
# -------------------------
def _drop_dup_endpoint(xy: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    P = np.asarray(xy, float)
    if len(P) < 2:
        return P
    if np.linalg.norm(P[0] - P[-1]) <= tol:
        return P[:-1]
    return P

def _resample_closed_curve_xy(xy_closed: np.ndarray, n: int = 240) -> np.ndarray:
    P = _drop_dup_endpoint(np.asarray(xy_closed, float))
    if len(P) < 3:
        return np.asarray(xy_closed, float)

    # close
    Pc = np.vstack([P, P[0]])
    seg = Pc[1:] - Pc[:-1]
    ds = np.sqrt(np.sum(seg**2, axis=1))
    s = np.hstack([[0.0], np.cumsum(ds)])
    L = float(s[-1])
    if L <= 0:
        return np.vstack([P, P[0]])

    s_new = np.linspace(0.0, L, int(max(40, n)), endpoint=False)
    x_new = np.interp(s_new, s, Pc[:, 0])
    y_new = np.interp(s_new, s, Pc[:, 1])
    Q = np.column_stack([x_new, y_new])
    Q = np.vstack([Q, Q[0]])
    return Q

def _chamfer(A: np.ndarray, B: np.ndarray) -> float:
    """
    Symmetric Chamfer distance between two point sets (closed curves).
    Returns meters (same unit as coordinates).
    """
    A = _drop_dup_endpoint(np.asarray(A, float))
    B = _drop_dup_endpoint(np.asarray(B, float))
    if len(A) < 5 or len(B) < 5:
        return float("inf")

    # (Na,1,2) - (1,Nb,2) -> (Na,Nb,2)
    d2 = np.sum((A[:, None, :] - B[None, :, :])**2, axis=2)
    da = np.sqrt(np.min(d2, axis=1)).mean()
    db = np.sqrt(np.min(d2, axis=0)).mean()
    return float(0.5 * (da + db))

def _get_lcfs_curve_from_shape(shape: Dict[str, Any]) -> Optional[np.ndarray]:
    R = shape.get("R_sep", None)
    Z = shape.get("Z_sep", None)
    if R is not None and Z is not None:
        try:
            R = np.asarray(R, float)
            Z = np.asarray(Z, float)
            if R.size >= 20 and Z.size == R.size:
                return np.column_stack([R, Z])
        except Exception:
            pass

    fb = shape.get("fallback_lcfs", None)
    # accept dict-like fallback
    if isinstance(fb, dict):
        try:
            R = np.asarray(fb.get("R", []), float)
            Z = np.asarray(fb.get("Z", []), float)
            if R.size >= 20 and Z.size == R.size:
                return np.column_stack([R, Z])
        except Exception:
            pass

    return None

def _parse_xpoints(shape: Dict[str, Any]) -> List[Tuple[float, float]]:
    xps = shape.get("xpoints", []) or []
    out: List[Tuple[float, float]] = []
    for x in xps:
        try:
            if isinstance(x, dict):
                R = float(x.get("R", x.get("Rc", np.nan)))
                Z = float(x.get("Z", x.get("Zc", np.nan)))
            else:
                # could be tuple/list
                R = float(x[0])
                Z = float(x[1])
            if np.isfinite(R) and np.isfinite(Z):
                out.append((R, Z))
        except Exception:
            continue
    return out

def _geom_x_targets(geom: Dict[str, Any]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """
    Returns (lower_target, upper_target) from geom["xpoints_target"] if present.
    """
    xs = geom.get("xpoints_target", None)
    if not xs:
        return (None, None)
    pts = []
    for it in xs:
        try:
            # expected: (R, Z, kind)
            R = float(it[0]); Z = float(it[1])
            if np.isfinite(R) and np.isfinite(Z):
                pts.append((R, Z))
        except Exception:
            continue
    if len(pts) < 1:
        return (None, None)
    lower = min(pts, key=lambda p: p[1])
    upper = max(pts, key=lambda p: p[1])
    # if only one, both point to same
    if len(pts) == 1:
        return (lower, None)
    return (lower, upper)

def _xpoint_assignment_cost(found: List[Tuple[float, float]],
                            targets: List[Tuple[float, float]]) -> float:
    """
    Minimal matching cost for up to 2 targets (brute force).
    Costs are Euclidean distances (meters).
    """
    if not targets:
        return 0.0
    if not found:
        return float("inf")

    # If one target, take closest found
    if len(targets) == 1:
        tx, tz = targets[0]
        return float(min(math.hypot(fx - tx, fz - tz) for fx, fz in found))

    # Two targets: try all pairs / permutations
    t0, t1 = targets[0], targets[1]
    best = float("inf")
    for i in range(len(found)):
        for j in range(len(found)):
            if i == j:
                continue
            f0 = found[i]; f1 = found[j]
            c1 = math.hypot(f0[0] - t0[0], f0[1] - t0[1]) + math.hypot(f1[0] - t1[0], f1[1] - t1[1])
            c2 = math.hypot(f0[0] - t1[0], f0[1] - t1[1]) + math.hypot(f1[0] - t0[0], f1[1] - t0[1])
            best = min(best, c1, c2)
    # If found has only 1 point, we can’t do 2-match
    if not np.isfinite(best):
        return float("inf")
    return float(best)


# -------------------------
# Misfit: (shape match) + (Xpoints near geom targets) + optional current prior
# -------------------------
def compute_misfit(shape: Dict[str, Any], *,
                   target_curve_xy: np.ndarray,
                   x_lower: Optional[Tuple[float, float]],
                   x_upper: Optional[Tuple[float, float]],
                   require_two_x: bool,
                   null_prefer: str,
                   currents_MA: Dict[str, float],
                   prior_MA: Optional[Dict[str, float]],
                   w_prior: float) -> float:

    # 1) LCFS curve
    lcfs = _get_lcfs_curve_from_shape(shape)
    if lcfs is None:
        return float(PEN_NO_SEP)

    # Resample both for stable chamfer
    A = _resample_closed_curve_xy(lcfs, n=220)
    B = _resample_closed_curve_xy(target_curve_xy, n=220)

    cham = _chamfer(_drop_dup_endpoint(A), _drop_dup_endpoint(B))
    if not np.isfinite(cham):
        return float(PEN_BAD)

    mis = float(W_SHAPE * (cham / max(1e-9, float(getattr(cfg, "sig_shape_m", SIG_SHAPE_M)))))

    # 2) X-point term
    found = _parse_xpoints(shape)
    prefer = str(null_prefer).strip().lower()

    targets: List[Tuple[float, float]] = []
    if require_two_x:
        # want both (if available)
        if x_lower is not None:
            targets.append(x_lower)
        if x_upper is not None:
            targets.append(x_upper)
        # If you only have one target point (rare), treat as single-target matching
        if len(targets) == 0:
            # no targets -> no penalty
            pass
        else:
            if len(found) == 0:
                mis += float(PEN_NO_X)
            elif len(found) == 1 and len(targets) >= 2:
                mis += float(PEN_ONLY1X)
                # also include distance to the closest target (so it still guides)
                d = _xpoint_assignment_cost(found, [targets[0]])
                mis += float(W_X * (d / max(1e-9, float(getattr(cfg, "sig_x_m", SIG_X_M)))))
            else:
                dsum = _xpoint_assignment_cost(found, targets)
                if not np.isfinite(dsum):
                    mis += float(PEN_NO_X)
                else:
                    mis += float(W_X * ((dsum / max(1, len(targets))) / max(1e-9, float(getattr(cfg, "sig_x_m", SIG_X_M)))))
    else:
        # single-null: match one target depending on null_prefer
        xt = None
        if prefer == "upper":
            xt = x_upper if x_upper is not None else x_lower
        else:
            xt = x_lower if x_lower is not None else x_upper

        if xt is not None:
            if len(found) == 0:
                mis += float(PEN_NO_X)
            else:
                d = _xpoint_assignment_cost(found, [xt])
                mis += float(W_X * (d / max(1e-9, float(getattr(cfg, "sig_x_m", SIG_X_M)))))

    # 3) weak prior on currents
    if prior_MA is not None and w_prior > 0:
        sigs = dict(CS=0.6, PF1=0.9, PF2=0.9, PF3=0.9)
        reg = 0.0
        for kname in ("CS", "PF1", "PF2", "PF3"):
            i = float(currents_MA.get(kname, np.nan))
            i0 = float(prior_MA.get(kname, i))
            if np.isfinite(i) and np.isfinite(i0):
                reg += ((i - i0) / sigs[kname]) ** 2
        mis += float(w_prior) * math.sqrt(max(0.0, reg))

    return float(max(0.0, mis))


# -------------------------
# CEM sampler
# -------------------------
@dataclass
class CEMState:
    mu: np.ndarray
    sigma: np.ndarray

def _reflect_to_bounds(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    span = hi - lo
    span = np.where(span <= 0, 1.0, span)
    T = np.mod(X - lo, 2.0 * span)
    Y = lo + np.where(T <= span, T, 2.0 * span - T)
    return np.minimum(np.maximum(Y, lo), hi)

def _bounds_arrays(bounds: Dict[str, Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.array([bounds["CS"][0], bounds["PF1"][0], bounds["PF2"][0], bounds["PF3"][0]], dtype=float)
    hi = np.array([bounds["CS"][1], bounds["PF1"][1], bounds["PF2"][1], bounds["PF3"][1]], dtype=float)
    return lo, hi

def _sample_uniform_ma(rng: np.random.Generator, n: int, bounds: Dict[str, Tuple[float, float]]) -> np.ndarray:
    lo, hi = _bounds_arrays(bounds)
    return lo + (hi - lo) * rng.random((n, 4))

def _sample_cem_ma(rng: np.random.Generator, state: CEMState, n: int,
                   bounds: Dict[str, Tuple[float, float]], mix_uniform: float) -> np.ndarray:
    n_u = int(round(mix_uniform * n))
    n_g = n - n_u
    X = []
    if n_g > 0:
        Z = rng.standard_normal((n_g, 4))
        Xg = state.mu[None, :] + Z * state.sigma[None, :]
        lo, hi = _bounds_arrays(bounds)
        X.append(_reflect_to_bounds(Xg, lo, hi))
    if n_u > 0:
        X.append(_sample_uniform_ma(rng, n_u, bounds))
    return np.vstack(X)

def _update_cem(state: CEMState, elite: np.ndarray, *, damp: float = 0.50, min_sigma: float = 0.10) -> CEMState:
    mu_new = elite.mean(axis=0)
    sig_new = elite.std(axis=0)
    sig_new = np.maximum(sig_new, float(min_sigma))
    mu = damp * state.mu + (1 - damp) * mu_new
    sigma = damp * state.sigma + (1 - damp) * sig_new
    sigma = np.maximum(sigma, float(min_sigma))
    return CEMState(mu=mu, sigma=sigma)


# -------------------------
# Persistent workers
# -------------------------
@dataclass
class Worker:
    proc: mp.Process
    in_q: mp.Queue
    busy: bool = False
    case_id: Optional[int] = None
    t_start: float = 0.0
    last_payload: Optional[Dict[str, Any]] = None
    last_meta: Optional[Dict[str, Any]] = None

def _worker_main(in_q: mp.Queue, out_q: mp.Queue, init_payload: Dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    from star_machine_cad import make_star_machine_from_cad, CADImportOptions, CADLayers

    unit_scale = init_payload.get("unit_scale", None)
    if unit_scale is not None:
        unit_scale = float(unit_scale)

    opts = CADImportOptions(
        unit_scale=unit_scale,
        resample_walls=str(init_payload["resample_walls"]),
        n_wall=int(init_payload["n_wall"]),
        n_inner=int(init_payload["n_inner"]),
        n_plasma=int(init_payload["n_plasma"]),
        min_wall_pts=int(init_payload["min_wall_pts"]),
        enforce_ccw=bool(init_payload["enforce_ccw"]),
        canonical_start=bool(init_payload["canonical_start"]),
        flatten_distance=float(init_payload["flatten_distance"]),
        label_match_factor=float(init_payload["label_match_factor"]),

        # plasma target must match your CAD auto settings
        plasma_target_mode=str(init_payload["plasma_target_mode"]),
        plasma_fit_to_inner_if_available=bool(init_payload["plasma_fit_to_inner_if_available"]),
        plasma_R0=float(init_payload["plasma_R0"]),
        plasma_A=float(init_payload["plasma_A"]),
        plasma_kappa=float(init_payload["plasma_kappa"]),
        plasma_Z0=float(init_payload["plasma_Z0"]),
        plasma_delta_max=float(init_payload["plasma_delta_max"]),
        plasma_delta_grid=int(init_payload["plasma_delta_grid"]),
        plasma_delta_symmetric=bool(init_payload["plasma_delta_symmetric"]),
        plasma_shrink_iters=int(init_payload["plasma_shrink_iters"]),
        plasma_scale_safety=float(init_payload["plasma_scale_safety"]),
        containment_radius=float(init_payload["containment_radius"]),
        fix_center_if_outside=bool(init_payload["fix_center_if_outside"]),
        center_search_samples=int(init_payload["center_search_samples"]),
        center_search_seed=int(init_payload["center_search_seed"]),
        strike_ray_fallback_len=float(init_payload["strike_ray_fallback_len"]),
    )

    tokamak, geom = make_star_machine_from_cad(
        dxf_path=str(init_payload["dxf_path"]),
        layers=CADLayers(),
        opts=opts,
        strict_expected=True,
    )

    _patch_profiles_copy_once()
    _patch_copy_into_allow_none_once()

    while True:
        msg = in_q.get()
        if msg is None:
            break

        case_id, payload, meta = msg
        t0 = time.perf_counter()

        CS_MA  = float(payload["CS"])
        PF1_MA = float(payload["PF1"])
        PF2_MA = float(payload["PF2"])
        PF3_MA = float(payload["PF3"])

        try:
            eq = solve_equilibrium(
                tokamak, geom,
                CS_MA=CS_MA, PF1_MA=PF1_MA, PF2_MA=PF2_MA, PF3_MA=PF3_MA,
                nx=int(payload["nx"]), ny=int(payload["ny"]),
                Ip=float(payload["Ip"]), paxis=float(payload["paxis"]), fvac=float(payload["fvac"]),
                alpha_m=float(payload["alpha_m"]), alpha_n=float(payload["alpha_n"]),
                target_rel_tol=float(payload["target_rel_tol"]),
                margin_RZ=float(payload["margin_RZ"]),
                f_list=tuple(payload["f_list"]),
                coil_group_mode=_sanitize_mode(payload.get("coil_group_mode", "area")),
                silence_solver=bool(payload.get("silence_solver", True)),
            )

            shape = extract_shape(
                eq, geom,
                require_two_x=bool(meta.get("require_two_x", False)),
                prefer_inner=bool(meta.get("prefer_inner", True)),
                psi_percentile=float(meta.get("psi_percentile", 0.5)),
                null_prefer=str(meta.get("null_prefer", "lower")),
            )

            elapsed = time.perf_counter() - t0
            out_q.put((case_id, {
                "ok_solve": True,
                "elapsed_s": float(elapsed),
                "tag": str(meta.get("tag", "batch")),
                "currents_MA": {"CS": CS_MA, "PF1": PF1_MA, "PF2": PF2_MA, "PF3": PF3_MA},
                "shape": shape,
                # "sanity": san,
            }))

        except Exception as e:
            elapsed = time.perf_counter() - t0
            out_q.put((case_id, {
                "ok_solve": False,
                "elapsed_s": float(elapsed),
                "tag": str(meta.get("tag", "batch")),
                "error": repr(e),
                "currents_MA": {"CS": CS_MA, "PF1": PF1_MA, "PF2": PF2_MA, "PF3": PF3_MA},
                "shape": {
                    "ok_sep": False, "reason": f"solve_failed:{repr(e)}",
                    "xpoints": [], "R_sep": [], "Z_sep": [], "fallback_lcfs": None,
                },
            }))

def _start_worker(ctx: mp.context.BaseContext, init_payload: Dict[str, Any], out_q: mp.Queue) -> Worker:
    in_q = ctx.Queue(maxsize=2)
    p = ctx.Process(target=_worker_main, args=(in_q, out_q, init_payload), daemon=True)
    p.start()
    return Worker(proc=p, in_q=in_q)

def _kill_worker(w: Worker) -> None:
    try:
        if w.proc.is_alive():
            w.proc.terminate()
    except Exception:
        pass
    try:
        w.proc.join(timeout=1.0)
    except Exception:
        pass


# -------------------------
# Dispatcher with timeouts + JSONL + status line
# -------------------------
def run_batch(workers: List[Worker],
              ctx: mp.context.BaseContext,
              init_payload: Dict[str, Any],
              out_q: mp.Queue,
              tasks: List[Tuple[int, Dict[str, Any], Dict[str, Any]]],
              *,
              timeout_s: float,
              jsonl_path: Path,
              best_path: Path,
              target_curve_xy: np.ndarray,
              x_lower: Optional[Tuple[float, float]],
              x_upper: Optional[Tuple[float, float]],
              require_two_x: bool,
              null_prefer: str,
              prior_MA: Optional[Dict[str, float]],
              w_prior: float) -> List[Dict[str, Any]]:

    tag = str(tasks[0][2].get("tag", "batch")) if tasks else "batch"

    done = 0
    total = len(tasks)
    solve_ok = 0
    sep_ok = 0
    best = float("inf")
    last_mis: Optional[float] = None
    t0 = time.time()

    pending = tasks.copy()
    owner: Dict[int, int] = {}
    results: List[Dict[str, Any]] = []

    with open(jsonl_path, "a", encoding="utf-8") as f:
        while pending or any(w.busy for w in workers):
            # assign
            for wi, w in enumerate(workers):
                if not pending:
                    break
                if not w.busy:
                    case_id, payload, meta = pending.pop(0)
                    owner[case_id] = wi
                    w.busy = True
                    w.case_id = case_id
                    w.t_start = time.time()
                    w.last_payload = payload
                    w.last_meta = meta
                    w.in_q.put((case_id, payload, meta))

            # collect
            got_items = []
            try:
                while True:
                    got_items.append(out_q.get_nowait())
            except Exception:
                pass

            # timeouts
            now = time.time()
            for wi, w in enumerate(workers):
                if w.busy and (now - w.t_start) > timeout_s:
                    cid = w.case_id
                    _kill_worker(w)
                    workers[wi] = _start_worker(ctx, init_payload, out_q)

                    payload = w.last_payload or {}
                    cm_timeout = {
                        "CS": float(payload.get("CS", float("nan"))),
                        "PF1": float(payload.get("PF1", float("nan"))),
                        "PF2": float(payload.get("PF2", float("nan"))),
                        "PF3": float(payload.get("PF3", float("nan"))),
                    }

                    rec = {
                        "ok_solve": False,
                        "tag": tag,
                        "elapsed_s": float(timeout_s),
                        "error": f"TimeoutError({timeout_s}s)",
                        "case_id": int(cid) if cid is not None else None,
                        "currents_MA": cm_timeout,
                        "shape": {"ok_sep": False, "reason": "timeout", "xpoints": [], "R_sep": [], "Z_sep": [], "fallback_lcfs": None},
                    }
                    rec["misfit"] = float(PEN_BAD)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()

                    done += 1
                    last_mis = float(rec["misfit"])
                    owner.pop(cid, None)
                    workers[wi].busy = False
                    workers[wi].case_id = None
                    workers[wi].last_payload = None
                    workers[wi].last_meta = None

            # If nothing ready, short blocking wait
            if not got_items:
                try:
                    got_items.append(out_q.get(timeout=0.15))
                except Exception:
                    got_items = []

            for case_id, rec in got_items:
                wi = owner.pop(case_id, None)
                if wi is not None:
                    workers[wi].busy = False
                    workers[wi].case_id = None
                    workers[wi].last_payload = None
                    workers[wi].last_meta = None

                cm = rec.get("currents_MA") or {"CS": float("nan"), "PF1": float("nan"), "PF2": float("nan"), "PF3": float("nan")}

                shp = rec.get("shape", {}) or {}
                mis = compute_misfit(
                    shp,
                    target_curve_xy=target_curve_xy,
                    x_lower=x_lower,
                    x_upper=x_upper,
                    require_two_x=bool(require_two_x),
                    null_prefer=str(null_prefer),
                    currents_MA=cm,
                    prior_MA=prior_MA,
                    w_prior=float(w_prior),
                )
                rec["misfit"] = float(mis)
                rec["case_id"] = int(case_id)

                done += 1
                if rec.get("ok_solve", False):
                    solve_ok += 1
                if _get_lcfs_curve_from_shape(shp) is not None:
                    sep_ok += 1

                last_mis = float(mis)
                if float(mis) < best:
                    best = float(mis)
                    best_path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")

                f.write(json.dumps(rec) + "\n")
                f.flush()
                results.append(rec)

            dt = max(1e-6, time.time() - t0)
            rate = done / dt
            print("\r" + _status_line(tag, done, total, solve_ok, sep_ok, best, rate, last_mis),
                  end="", flush=True)

        print()
    return results


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dxf", type=str, default=_default_dxf())
    ap.add_argument("--outdir", type=str, default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=110.0)

    ap.add_argument("--n1", type=int, default=600)
    ap.add_argument("--n2", type=int, default=1000)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--elite_frac", type=float, default=0.12)

    # X behavior
    ap.add_argument("--null", type=str, default="lower", choices=["lower", "upper", "any"])
    ap.add_argument("--require_two_x", action="store_true")

    # LCFS controls (passed to analyze_star)
    ap.add_argument("--prefer_inner", action="store_true")
    ap.add_argument("--psi_percentile", type=float, default=0.5)  # <=0 => extremes first

    # Prior regularizer
    ap.add_argument("--w_prior", type=float, default=W_PRIOR)

    args = ap.parse_args()
    mp.freeze_support()

    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    jsonl_path = outdir / "scan_multigoal_results.jsonl"
    best_path_global = outdir / "scan_multigoal_best_global.json"
    best_path_batch  = outdir / "scan_multigoal_best_batch.json"

    if not args.append:
        jsonl_path.unlink(missing_ok=True)
        best_path_global.unlink(missing_ok=True)
        best_path_batch.unlink(missing_ok=True)

    rng = np.random.default_rng(int(args.seed))

    # Prior currents (optional): cfg currents are in A -> convert to MA
    prior_MA = None
    if all(hasattr(cfg, k) for k in ("CS_current", "PF1_current", "PF2_current", "PF3_current")):
        try:
            prior_MA = {
                "CS":  float(getattr(cfg, "CS_current"))  / 1e6,
                "PF1": float(getattr(cfg, "PF1_current")) / 1e6,
                "PF2": float(getattr(cfg, "PF2_current")) / 1e6,
                "PF3": float(getattr(cfg, "PF3_current")) / 1e6,
            }
        except Exception:
            prior_MA = None

    bounds = dict(BOUNDS_DEFAULT_MA)

    # CEM init around mid-bounds
    lo, hi = _bounds_arrays(bounds)
    mid = 0.5 * (lo + hi)
    sig = 0.35 * (hi - lo)
    state = CEMState(mu=mid, sigma=sig)

    # -------------------------
    # Build CAD geometry ONCE to get:
    #   - plasma target curve (R_plasma/Z_plasma)
    #   - xpoints_target (lower/upper)
    # -------------------------
    from star_machine_cad import make_star_machine_from_cad, CADImportOptions, CADLayers

    init_payload = {
        "dxf_path": args.dxf,

        # CAD read + walls
        "unit_scale": getattr(cfg, "unit_scale", None),
        "resample_walls": str(getattr(cfg, "resample_walls", "auto")),
        "n_wall": int(getattr(cfg, "n_wall", 801)),
        "n_inner": int(getattr(cfg, "n_inner", 801)),
        "n_plasma": int(getattr(cfg, "n_plasma", 320)),
        "min_wall_pts": int(getattr(cfg, "min_wall_pts", 200)),
        "enforce_ccw": bool(getattr(cfg, "enforce_ccw", True)),
        "canonical_start": bool(getattr(cfg, "canonical_start", True)),
        "flatten_distance": float(getattr(cfg, "flatten_distance", 0.01)),
        "label_match_factor": float(getattr(cfg, "label_match_factor", 2.0)),

        # plasma target (MUST match your machine-cad setup)
        "plasma_target_mode": str(getattr(cfg, "plasma_target_mode", "auto")),
        "plasma_fit_to_inner_if_available": bool(getattr(cfg, "plasma_fit_to_inner_if_available", True)),
        "plasma_R0": float(getattr(cfg, "plasma_R0", 4.0)),
        "plasma_A": float(getattr(cfg, "plasma_A", 2.0)),
        "plasma_kappa": float(getattr(cfg, "plasma_kappa", 2.5)),
        "plasma_Z0": float(getattr(cfg, "plasma_Z0", 0.0)),
        "plasma_delta_max": float(getattr(cfg, "plasma_delta_max", 0.70)),
        "plasma_delta_grid": int(getattr(cfg, "plasma_delta_grid", 17)),
        "plasma_delta_symmetric": bool(getattr(cfg, "plasma_delta_symmetric", True)),
        "plasma_shrink_iters": int(getattr(cfg, "plasma_shrink_iters", 20)),
        "plasma_scale_safety": float(getattr(cfg, "plasma_scale_safety", 0.999)),
        "containment_radius": float(getattr(cfg, "containment_radius", -1e-9)),
        "fix_center_if_outside": bool(getattr(cfg, "fix_center_if_outside", True)),
        "center_search_samples": int(getattr(cfg, "center_search_samples", 800)),
        "center_search_seed": int(getattr(cfg, "center_search_seed", 0)),
        "strike_ray_fallback_len": float(getattr(cfg, "strike_ray_fallback_len", 3.0)),
    }

    # Load geom once in main (same opts)
    unit_scale = init_payload["unit_scale"]
    if unit_scale is not None:
        unit_scale = float(unit_scale)
    opts_main = CADImportOptions(
        unit_scale=unit_scale,
        resample_walls=init_payload["resample_walls"],
        n_wall=init_payload["n_wall"],
        n_inner=init_payload["n_inner"],
        n_plasma=init_payload["n_plasma"],
        min_wall_pts=init_payload["min_wall_pts"],
        enforce_ccw=init_payload["enforce_ccw"],
        canonical_start=init_payload["canonical_start"],
        flatten_distance=init_payload["flatten_distance"],
        label_match_factor=init_payload["label_match_factor"],

        plasma_target_mode=init_payload["plasma_target_mode"],
        plasma_fit_to_inner_if_available=init_payload["plasma_fit_to_inner_if_available"],
        plasma_R0=init_payload["plasma_R0"],
        plasma_A=init_payload["plasma_A"],
        plasma_kappa=init_payload["plasma_kappa"],
        plasma_Z0=init_payload["plasma_Z0"],
        plasma_delta_max=init_payload["plasma_delta_max"],
        plasma_delta_grid=init_payload["plasma_delta_grid"],
        plasma_delta_symmetric=init_payload["plasma_delta_symmetric"],
        plasma_shrink_iters=init_payload["plasma_shrink_iters"],
        plasma_scale_safety=init_payload["plasma_scale_safety"],
        containment_radius=init_payload["containment_radius"],
        fix_center_if_outside=init_payload["fix_center_if_outside"],
        center_search_samples=init_payload["center_search_samples"],
        center_search_seed=init_payload["center_search_seed"],
        strike_ray_fallback_len=init_payload["strike_ray_fallback_len"],
    )
    _, geom_main = make_star_machine_from_cad(dxf_path=args.dxf, layers=CADLayers(), opts=opts_main, strict_expected=True)

    if "R_plasma" not in geom_main or "Z_plasma" not in geom_main:
        raise RuntimeError("geom_main has no plasma target (R_plasma/Z_plasma). Check plasma_target_mode/opts.")

    target_curve_xy = np.column_stack([np.asarray(geom_main["R_plasma"], float), np.asarray(geom_main["Z_plasma"], float)])
    x_lower, x_upper = _geom_x_targets(geom_main)

    print("[INFO] scan_star_multigoal started")
    print(f"[INFO] dxf={args.dxf}")
    print(f"[INFO] workers={args.workers} n1={args.n1} n2={args.n2} pop={args.pop} timeout={args.timeout}")
    print(f"[INFO] target curve points: {len(target_curve_xy)}")
    print(f"[INFO] xpoints_target lower={x_lower} upper={x_upper} require_two_x={args.require_two_x}")
    print(f"[INFO] chamfer: sig_shape_m={float(getattr(cfg,'sig_shape_m',SIG_SHAPE_M))}  x: sig_x_m={float(getattr(cfg,'sig_x_m',SIG_X_M))}")
    print(f"[INFO] lcfs: prefer_inner={args.prefer_inner} psi_percentile={args.psi_percentile} (<=0 => extremes-first)")

    # Solver constants from cfg
    base_solver = dict(
        Ip=float(getattr(cfg, "Ip", 8.0e5)),
        paxis=float(getattr(cfg, "paxis", 2.0e3)),
        fvac=float(getattr(cfg, "fvac", 0.5)),
        alpha_m=float(getattr(cfg, "alpha_m", 1.8)),
        alpha_n=float(getattr(cfg, "alpha_n", 1.2)),
        margin_RZ=float(getattr(cfg, "margin_RZ", 0.5)),
        coil_group_mode=_sanitize_mode(getattr(cfg, "coil_group_mode", "area")),
        silence_solver=bool(getattr(cfg, "silence_solver", True)),
    )

    # Continuation schedules
    f_list_equil = tuple(getattr(cfg, "f_list_equilibrium", (0.10, 0.25, 0.55, 1.0)))
    f_list_coarse = tuple(getattr(cfg, "f_list_coarse", (0.20, 0.55, 1.0))) if hasattr(cfg, "f_list_coarse") else f_list_equil
    f_list_refine = tuple(getattr(cfg, "f_list_refine", (0.15, 0.35, 0.65, 1.0))) if hasattr(cfg, "f_list_refine") else f_list_equil

    # Grids
    nx1 = int(getattr(cfg, "nx_coarse", getattr(cfg, "nx_eq", 65)))
    ny1 = int(getattr(cfg, "ny_coarse", getattr(cfg, "ny_eq", 129)))
    nx2 = int(getattr(cfg, "nx_refine", 129))
    ny2 = int(getattr(cfg, "ny_refine", 257))

    # Tolerances
    tol_coarse = float(getattr(cfg, "target_rel_tol_coarse", 3e-3))
    tol_refine = float(getattr(cfg, "target_rel_tol_refine", 1e-3))

    # MP context
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    workers: List[Worker] = []
    for _ in range(max(1, int(args.workers))):
        workers.append(_start_worker(ctx, init_payload, out_q))

    def build_tasks(X: np.ndarray, *, tag: str, nx: int, ny: int, tol: float,
                    f_list: Tuple[float, ...], start_id: int) -> List[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        tasks = []
        for i in range(X.shape[0]):
            CS, PF1, PF2, PF3 = map(float, X[i, :])
            payload = dict(
                CS=CS, PF1=PF1, PF2=PF2, PF3=PF3,
                nx=int(nx), ny=int(ny),
                Ip=base_solver["Ip"],
                paxis=base_solver["paxis"],
                fvac=base_solver["fvac"],
                alpha_m=base_solver["alpha_m"],
                alpha_n=base_solver["alpha_n"],
                target_rel_tol=float(tol),
                margin_RZ=base_solver["margin_RZ"],
                f_list=tuple(float(x) for x in f_list),
                silence_solver=bool(base_solver["silence_solver"]),
                coil_group_mode=str(base_solver["coil_group_mode"]),
            )
            meta = dict(
                tag=str(tag),
                require_two_x=bool(args.require_two_x),
                prefer_inner=bool(args.prefer_inner),
                psi_percentile=float(args.psi_percentile),
                null_prefer=str(args.null),
            )
            tasks.append((start_id + i, payload, meta))
        return tasks

    def _load_best_global() -> Tuple[float, Optional[Dict[str, Any]]]:
        if not best_path_global.exists():
            return float("inf"), None
        try:
            rec = json.loads(best_path_global.read_text(encoding="utf-8"))
            return float(rec.get("misfit", float("inf"))), rec
        except Exception:
            return float("inf"), None

    best_global_mis, _ = _load_best_global()

    def _maybe_update_best_global(res: List[Dict[str, Any]]) -> None:
        nonlocal best_global_mis
        if not res:
            return
        best_r = None
        best_m = float("inf")
        for r in res:
            try:
                m = float(r.get("misfit", float("inf")))
            except Exception:
                continue
            if m < best_m:
                best_m = m
                best_r = r
        if best_r is None:
            return
        if best_m < best_global_mis:
            best_global_mis = best_m
            best_path_global.write_text(json.dumps(best_r, indent=2) + "\n", encoding="utf-8")

    try:
        case_id0 = 0

        # Anti-stall constants
        STALL_PATIENCE = 4
        IMPROVE_EPS = 1e-3
        MAX_SIGMA = np.array([1.2, 1.8, 1.8, 1.8], float)

        # -------- Stage 1 --------
        remaining = int(args.n1)
        all1: List[Dict[str, Any]] = []
        mix_u = 0.85
        BURNIN_BATCHES = 3
        batch_idx = 0

        best_stage = float("inf")
        stall = 0

        while remaining > 0:
            n = min(int(args.pop), remaining)
            if batch_idx < BURNIN_BATCHES:
                X = _sample_uniform_ma(rng, n, bounds)
            else:
                X = _sample_cem_ma(rng, state, n, bounds, mix_uniform=mix_u)
            batch_idx += 1

            tasks = build_tasks(X, tag="coarse", nx=nx1, ny=ny1, tol=tol_coarse, f_list=f_list_coarse, start_id=case_id0)
            case_id0 += len(tasks)
            remaining -= len(tasks)

            res = run_batch(
                workers, ctx, init_payload, out_q, tasks,
                timeout_s=float(args.timeout),
                jsonl_path=jsonl_path,
                best_path=best_path_batch,
                target_curve_xy=target_curve_xy,
                x_lower=x_lower,
                x_upper=x_upper,
                require_two_x=bool(args.require_two_x),
                null_prefer=str(args.null),
                prior_MA=prior_MA,
                w_prior=float(args.w_prior),
            )
            all1.extend(res)
            _maybe_update_best_global(res)

            # stagnation check
            try:
                best_batch = float(min(r.get("misfit", float("inf")) for r in res))
            except Exception:
                best_batch = float("inf")

            if best_batch < best_stage - IMPROVE_EPS:
                best_stage = best_batch
                stall = 0
            else:
                stall += 1

            if stall >= STALL_PATIENCE:
                state.sigma = np.minimum(state.sigma * 1.35, MAX_SIGMA)
                mix_u = min(0.90, mix_u + 0.10)
                jitter = rng.standard_normal(4) * (0.25 * state.sigma)
                state.mu = _reflect_to_bounds(state.mu + jitter, lo, hi)
                stall = 0

            # Elite update
            ok_res = []
            for r in res:
                if not r.get("ok_solve", False):
                    continue
                shp = r.get("shape") or {}
                if _get_lcfs_curve_from_shape(shp) is not None and np.isfinite(float(r.get("misfit", float("inf")))):
                    ok_res.append(r)

            if len(ok_res) >= max(6, int(0.20 * len(res))):
                ok_res.sort(key=lambda rr: float(rr["misfit"]))
                elite_n = max(6, int(math.ceil(float(args.elite_frac) * len(ok_res))))
                elite = ok_res[:elite_n]
                E = np.vstack([
                    np.array([e["currents_MA"]["CS"], e["currents_MA"]["PF1"], e["currents_MA"]["PF2"], e["currents_MA"]["PF3"]], float)
                    for e in elite
                ])
                state = _update_cem(state, E, damp=0.35, min_sigma=0.18)
                mix_u = max(0.35, mix_u * 0.95)
            else:
                state.sigma = np.minimum(state.sigma * 1.28, MAX_SIGMA)
                mix_u = min(0.90, mix_u * 1.10)
                mix_u = max(0.35, mix_u)

        # Recenter refine around best stage1
        ok1 = [r for r in all1 if r.get("ok_solve", False) and np.isfinite(r.get("misfit", np.inf))]
        if ok1:
            ok1.sort(key=lambda r: float(r["misfit"]))
            b1 = ok1[0]["currents_MA"]
            state.mu = np.array([b1["CS"], b1["PF1"], b1["PF2"], b1["PF3"]], float)
            state.sigma = np.maximum(state.sigma * 0.70, np.array([0.12, 0.16, 0.16, 0.16], float))

        # -------- Stage 2 --------
        remaining = int(args.n2)
        mix_u = 0.20
        best_stage = float("inf")
        stall = 0

        REFINE_WARM_BATCHES = 2
        refine_batch_idx = 0
        SIGMA_CAP_WARM = np.array([0.06, 0.08, 0.08, 0.08], float)

        while remaining > 0:
            n = min(int(args.pop), remaining)

            mix_eff = mix_u
            sigma_saved = state.sigma.copy()
            tol_eff = tol_refine
            if refine_batch_idx < REFINE_WARM_BATCHES:
                mix_eff = 0.0
                state.sigma = np.minimum(state.sigma, SIGMA_CAP_WARM)
                tol_eff = max(tol_refine, 2e-3)

            X = _sample_cem_ma(rng, state, n, bounds, mix_uniform=mix_eff)
            state.sigma = sigma_saved
            refine_batch_idx += 1

            tasks = build_tasks(X, tag="refine", nx=nx2, ny=ny2, tol=tol_eff, f_list=f_list_refine, start_id=case_id0)
            case_id0 += len(tasks)
            remaining -= len(tasks)

            res = run_batch(
                workers, ctx, init_payload, out_q, tasks,
                timeout_s=float(args.timeout),
                jsonl_path=jsonl_path,
                best_path=best_path_batch,
                target_curve_xy=target_curve_xy,
                x_lower=x_lower,
                x_upper=x_upper,
                require_two_x=bool(args.require_two_x),
                null_prefer=str(args.null),
                prior_MA=prior_MA,
                w_prior=float(args.w_prior),
            )
            _maybe_update_best_global(res)

            # stagnation check
            try:
                best_batch = float(min(r.get("misfit", float("inf")) for r in res))
            except Exception:
                best_batch = float("inf")

            if best_batch < best_stage - IMPROVE_EPS:
                best_stage = best_batch
                stall = 0
            else:
                stall += 1

            if stall >= STALL_PATIENCE:
                state.sigma = np.minimum(state.sigma * 1.25, MAX_SIGMA)
                mix_u = min(0.55, mix_u + 0.15)
                jitter = rng.standard_normal(4) * (0.20 * state.sigma)
                state.mu = _reflect_to_bounds(state.mu + jitter, lo, hi)
                stall = 0

            ok_res = []
            for r in res:
                if not r.get("ok_solve", False):
                    continue
                shp = r.get("shape") or {}
                if _get_lcfs_curve_from_shape(shp) is not None and np.isfinite(float(r.get("misfit", float("inf")))):
                    ok_res.append(r)

            if len(ok_res) >= max(6, int(0.20 * len(res))):
                ok_res.sort(key=lambda rr: float(rr["misfit"]))
                elite_n = max(6, int(math.ceil(float(args.elite_frac) * len(ok_res))))
                elite = ok_res[:elite_n]
                E = np.vstack([
                    np.array([e["currents_MA"]["CS"], e["currents_MA"]["PF1"], e["currents_MA"]["PF2"], e["currents_MA"]["PF3"]], float)
                    for e in elite
                ])
                state = _update_cem(state, E, damp=0.50, min_sigma=0.10)
                mix_u = max(0.10, mix_u * 0.95)
            else:
                state.sigma = np.minimum(state.sigma * 1.12, MAX_SIGMA)
                mix_u = min(0.55, mix_u * 1.08)

        print("[DONE] Finished.")
        print("[DONE] Best GLOBAL:", str(best_path_global))
        print("[DONE] Best batch :", str(best_path_batch))
        print("[DONE] Log        :", str(jsonl_path))
        if np.isfinite(best_global_mis):
            print(f"[DONE] global best misfit = {best_global_mis:.6g}")

    finally:
        for w in workers:
            try:
                w.in_q.put(None)
            except Exception:
                pass
        for w in workers:
            _kill_worker(w)

if __name__ == "__main__":
    main()

