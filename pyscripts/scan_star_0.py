# scan_star_multigoal.py
# Multi-objective scan for STAR-like equilibria:
#   - Primary: match target shape (R0, A, kappa, delta)
#   - Secondary: X-point presence / optional X-target (soft)
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
SIG_R0   = 0.25
SIG_A    = 0.20
SIG_KAP  = 0.25
SIG_DEL  = 0.15

PEN_BAD_SHAPE = 120.0
PEN_THIN      = 6.0
PEN_AXIS_OUT  = 8.0

# Divertor/X-point terms (SOFT by default)
PEN_NO_X    = 12.0   # allow limiter solutions (not infinite)
W_XDIST     = 25.0   # if x_target is provided
BONUS_HAS_X = -3.0   # small reward for having >=1 X-point (optional)

# Optional "paper currents" regularizer (weak)
W_PRIOR = 0.15  # set 0 to disable


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
                 solve_ok: int, shape_ok: int,
                 best: float, rate: float, last: Optional[float]) -> str:
    frac = 0.0 if total <= 0 else done / total
    last_s = "" if last is None else f" last={last:7.2f}"
    b = best if np.isfinite(best) else 9999.0
    return (f"{tag:>6s} {_bar(frac)} {done:4d}/{total:<4d} "
            f"solve_ok={solve_ok:4d} shape_ok={shape_ok:4d} "
            f"best={b:7.2f}{last_s} {rate:5.2f}/s")

def _sanitize_mode(x: Any) -> str:
    if callable(x):
        return "area"
    s = str(x).strip().lower()
    return s if s in ("area", "equal", "same") else "area"


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
                for attr in dir(self):
                    if attr.endswith("_core_mask") or attr.endswith("_mask"):
                        try:
                            if getattr(self, attr, None) is None:
                                setattr(self, attr, mask)
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
    candidates = ("diverted_core_mask", "limiter_core_mask", "core_mask",
                  "diverted_mask", "limiter_mask")
    for name in candidates:
        try:
            v = getattr(profiles, name, None)
            if (v is None) or (np.asarray(v).shape != mask.shape):
                setattr(profiles, name, mask.copy())
        except Exception:
            pass
    for attr in dir(profiles):
        if not (attr.endswith("_core_mask") or attr.endswith("_mask")):
            continue
        try:
            v = getattr(profiles, attr, None)
            if v is None:
                setattr(profiles, attr, mask.copy())
        except Exception:
            pass


# -------------------------
# Coil family currents application
# -------------------------
def _iter_coils(tokamak: Any):
    """
    Yield (LABEL_UPPER, coil_obj) supporting both:
      - [(label, coil), ...]
      - [coil, coil, ...] where coil has .label/.name
    """
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
    """
    Apply family TOTAL currents. Prefer tokamak.apply_group_currents if available.
    Fallback supports both coil tuple format and label-based coils.
    """
    mode = _sanitize_mode(mode)

    totals = {"CS": float(CS_A), "PF1": float(PF1_A), "PF2": float(PF2_A), "PF3": float(PF3_A)}

    # 1) Best: machine provides grouping method
    if hasattr(tokamak, "apply_group_currents") and hasattr(tokamak, "coil_groups"):
        try:
            tokamak.apply_group_currents(totals, mode=mode)
            return
        except Exception:
            # If the method exists but fails, fall through.
            pass

    # 2) Next best: if coil_groups exists, distribute ourselves (equal/area)
    groups = getattr(tokamak, "coil_groups", None) or {}
    weights = getattr(tokamak, "coil_group_weights", None) or {}

    coil_map = {lab: coil for lab, coil in _iter_coils(tokamak)}

    if groups:
        for fam, Itot in totals.items():
            labs = groups.get(fam, None) or groups.get(fam.upper(), None) or groups.get(fam.lower(), None) or []
            labs_u = [str(x).strip().upper() for x in (labs or []) if str(x).strip()]
            if not labs_u:
                continue

            if mode == "same":
                # NOT recommended for segmented coils, but honor the option
                for lab in labs_u:
                    c = coil_map.get(lab)
                    if c is not None:
                        try:
                            c.current = float(Itot)
                        except Exception:
                            pass
                continue

            if mode == "area":
                # weights[fam] may be dict(label->w)
                wdict = weights.get(fam, None) or weights.get(fam.upper(), None) or {}
                ws = []
                for lab in labs_u:
                    w = float(wdict.get(lab, 1.0))
                    ws.append(max(0.0, w))
                s = sum(ws) if ws else 0.0
                if s <= 0:
                    ws = [1.0] * len(labs_u)
                    s = float(len(labs_u))
                for lab, w in zip(labs_u, ws):
                    c = coil_map.get(lab)
                    if c is not None:
                        try:
                            c.current = float(Itot) * (float(w) / s)
                        except Exception:
                            pass
                continue

            # equal
            n = float(len(labs_u))
            for lab in labs_u:
                c = coil_map.get(lab)
                if c is not None:
                    try:
                        c.current = float(Itot) / n
                    except Exception:
                        pass
        return

    # 3) Legacy fallback: label names
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

            profiles = ConstrainPaxisIp(
                eq=eq,
                paxis=f * float(paxis),
                Ip=f * float(Ip),
                fvac=float(fvac),
                alpha_m=float(alpha_m),
                alpha_n=float(alpha_n),
            )
            _ensure_profile_masks(profiles, eq)

            for attempt in range(2):
                try:
                    solver.solve(
                        eq=eq,
                        profiles=profiles,
                        constrain=None,
                        target_relative_tolerance=float(target_rel_tol),
                        verbose=False,
                    )
                    break
                except TypeError as e:
                    msg = str(e)
                    if ("Cannot copy" in msg and "NoneType" in msg) or ("without deepcopying" in msg):
                        _ensure_profile_masks(profiles, eq)
                        if attempt == 1:
                            raise
                        continue
                    raise
    finally:
        if ctx is not None:
            ctx.close()

    return eq


# -------------------------
# Shape extraction (unified analyze_star.py)
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
# Misfit: paper-like + soft divertor preference + optional x_target + optional current prior
# -------------------------
def _choose_xpoint(shape: Dict[str, Any], prefer: str) -> Optional[Tuple[float, float]]:
    xps = shape.get("xpoints", []) or []
    if not xps:
        return None
    pts = []
    for x in xps:
        try:
            pts.append((float(x["R"]), float(x["Z"])))
        except Exception:
            continue
    if not pts:
        return None
    prefer = str(prefer).strip().lower()
    if prefer == "lower":
        return min(pts, key=lambda p: p[1])
    if prefer == "upper":
        return max(pts, key=lambda p: p[1])
    return pts[0]

def compute_misfit(shape: Dict[str, Any], *,
                   targets: Dict[str, float],
                   x_target: Optional[Tuple[float, float]],
                   null_prefer: str,
                   pen_no_x: float,
                   w_xdist: float,
                   bonus_has_x: float,
                   currents_MA: Dict[str, float],
                   prior_MA: Optional[Dict[str, float]],
                   w_prior: float) -> float:

    try:
        R0 = float(shape.get("R0_plasma", float("nan")))
        A  = float(shape.get("A_plasma",  float("nan")))
        k  = float(shape.get("kappa_plasma", float("nan")))
        du = float(shape.get("delta_u", float("nan")))
        dl = float(shape.get("delta_l", float("nan")))
        a  = float(shape.get("a_plasma", float("nan")))
        Rax = float(shape.get("R_ax", float("nan")))
    except Exception:
        return float(PEN_BAD_SHAPE)

    if not np.isfinite(R0 + A + k + du + dl + a):
        return float(PEN_BAD_SHAPE)

    dbar = 0.5 * (du + dl)

    # normalized L2-ish
    mis2 = 0.0
    mis2 += ((R0 - targets["R0_target"]) / max(1e-6, targets.get("sig_R0", SIG_R0))) ** 2
    mis2 += ((A  - targets["A_target"])  / max(1e-6, targets.get("sig_A",  SIG_A))) ** 2
    mis2 += ((k  - targets["kappa_target"]) / max(1e-6, targets.get("sig_k", SIG_KAP))) ** 2
    mis2 += 0.25 * ((dbar - targets.get("delta_target", 0.25)) / max(1e-6, targets.get("sig_del", SIG_DEL))) ** 2
    mis = float(math.sqrt(max(0.0, mis2)))

    # thin plasma penalty
    a_min = float(targets.get("a_min", 0.7))
    if a < a_min:
        mis += PEN_THIN * (a_min - a) / 0.2

    # axis sanity
    if np.isfinite(Rax) and abs(Rax - R0) > max(0.4, 0.8 * a):
        mis += PEN_AXIS_OUT * (abs(Rax - R0) / max(0.5, a))

    # X-point soft
    has_x = bool(shape.get("xpoints", []))
    if has_x:
        mis += float(bonus_has_x)
    else:
        mis += float(pen_no_x)

    if x_target is not None:
        xp = _choose_xpoint(shape, null_prefer)
        if xp is None:
            mis += float(pen_no_x)
        else:
            dx = float(math.hypot(xp[0] - x_target[0], xp[1] - x_target[1]))
            mis += float(w_xdist) * dx

    # weak prior on currents
    if prior_MA is not None and w_prior > 0:
        sigs = dict(CS=0.6, PF1=0.9, PF2=0.9, PF3=0.9)
        reg = 0.0
        for kname in ("CS", "PF1", "PF2", "PF3"):
            i = float(currents_MA[kname])
            i0 = float(prior_MA.get(kname, i))
            reg += ((i - i0) / sigs[kname]) ** 2
        mis += float(w_prior) * math.sqrt(reg)

    return float(max(0.0, mis))


# -------------------------
# CEM sampler (with reflection bounds + anti-stall)
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

def _clip_to_bounds_ma(x: np.ndarray, bounds: Dict[str, Tuple[float, float]]) -> np.ndarray:
    lo, hi = _bounds_arrays(bounds)
    return _reflect_to_bounds(x, lo, hi)

def _sample_uniform_ma(rng: np.random.Generator, n: int, bounds: Dict[str, Tuple[float, float]]) -> np.ndarray:
    lo, hi = _bounds_arrays(bounds)
    return lo + (hi - lo) * rng.random((n, 4))

def _sample_cem_ma(rng: np.random.Generator, state: CEMState, n: int, bounds: Dict[str, Tuple[float, float]], mix_uniform: float) -> np.ndarray:
    n_u = int(round(mix_uniform * n))
    n_g = n - n_u
    X = []
    if n_g > 0:
        Z = rng.standard_normal((n_g, 4))
        Xg = state.mu[None, :] + Z * state.sigma[None, :]
        X.append(_clip_to_bounds_ma(Xg, bounds))
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

    opts = CADImportOptions(
        unit_scale=float(init_payload["unit_scale"]),
        resample_walls=str(init_payload["resample_walls"]),
        n_wall=int(init_payload["n_wall"]),
        n_inner=int(init_payload["n_inner"]),
        n_plasma=int(init_payload["n_plasma"]),
        min_wall_pts=int(init_payload["min_wall_pts"]),
        enforce_ccw=bool(init_payload["enforce_ccw"]),
        canonical_start=bool(init_payload["canonical_start"]),
        flatten_distance=float(init_payload["flatten_distance"]),
        label_match_factor=float(init_payload["label_match_factor"]),
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
                "currents_A":  {"CS": 1e6*CS_MA, "PF1": 1e6*PF1_MA, "PF2": 1e6*PF2_MA, "PF3": 1e6*PF3_MA},
                "shape": shape,
            }))

        except Exception as e:
            elapsed = time.perf_counter() - t0
            out_q.put((case_id, {
                "ok_solve": False,
                "elapsed_s": float(elapsed),
                "tag": str(meta.get("tag", "batch")),
                "error": repr(e),
                "currents_MA": {"CS": CS_MA, "PF1": PF1_MA, "PF2": PF2_MA, "PF3": PF3_MA},
                "currents_A":  {"CS": 1e6*CS_MA, "PF1": 1e6*PF1_MA, "PF2": 1e6*PF2_MA, "PF3": 1e6*PF3_MA},
                "shape": {
                    "ok_sep": False, "reason": f"solve_failed:{repr(e)}",
                    "xpoints": [], "R_sep": [], "Z_sep": [],
                    "R0_plasma": float("nan"), "A_plasma": float("nan"), "kappa_plasma": float("nan"),
                    "delta_u": float("nan"), "delta_l": float("nan"), "a_plasma": float("nan"),
                    "R_ax": float("nan"), "Z_ax": float("nan"),
                    "psi_ax": float("nan"), "psi_sep": float("nan"),
                    "fallback_lcfs": None,
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
              targets: Dict[str, float],
              x_target: Optional[Tuple[float, float]],
              null_prefer: str,
              pen_no_x: float,
              w_xdist: float,
              bonus_has_x: float,
              prior_MA: Optional[Dict[str, float]],
              w_prior: float) -> List[Dict[str, Any]]:

    tag = str(tasks[0][2].get("tag", "batch")) if tasks else "batch"

    done = 0
    total = len(tasks)
    solve_ok = 0
    shape_ok = 0
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
                    case_id, payload, meta = pending.pop(0)  # FIFO: más “justo” para debug/estadística
                    owner[case_id] = wi
                    w.busy = True
                    w.case_id = case_id
                    w.t_start = time.time()
                    w.last_payload = payload
                    w.last_meta = meta
                    w.in_q.put((case_id, payload, meta))

            # collect (DRAIN): process ALL completed results immediately
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
                    cm_timeout = None
                    try:
                        cm_timeout = {
                            "CS": float(payload.get("CS", float("nan"))),
                            "PF1": float(payload.get("PF1", float("nan"))),
                            "PF2": float(payload.get("PF2", float("nan"))),
                            "PF3": float(payload.get("PF3", float("nan"))),
                        }
                    except Exception:
                        cm_timeout = None

                    rec = {
                        "ok_solve": False,
                        "tag": tag,
                        "elapsed_s": float(timeout_s),
                        "error": f"TimeoutError({timeout_s}s)",
                        "case_id": int(cid) if cid is not None else None,
                        "currents_MA": cm_timeout,
                        "currents_A": ({k: (1e6*v if (cm_timeout and np.isfinite(v)) else None) for k, v in (cm_timeout or {}).items()} if cm_timeout else None),
                        "shape": {"ok_sep": False, "reason": "timeout", "xpoints": [], "R_sep": [], "Z_sep": [], "fallback_lcfs": None},
                    }
                    rec["misfit"] = float(PEN_BAD_SHAPE)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()

                    done += 1
                    last_mis = float(rec["misfit"])
                    owner.pop(cid, None)
                    workers[wi].busy = False
                    workers[wi].case_id = None
                    workers[wi].last_payload = None
                    workers[wi].last_meta = None

                    # IMPORTANT: old worker is dead; do not keep it "busy"
                    # (we already replaced workers[wi], but ensure bookkeeping doesn't stick)
                    continue




            # If nothing ready, do a short blocking wait for a single item
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
                mis = compute_misfit(
                    rec.get("shape", {}) or {},
                    targets=targets,
                    x_target=x_target,
                    null_prefer=str(null_prefer),
                    pen_no_x=float(pen_no_x),
                    w_xdist=float(w_xdist),
                    bonus_has_x=float(bonus_has_x),
                    currents_MA=cm,
                    prior_MA=prior_MA,
                    w_prior=float(w_prior),
                )
                rec["misfit"] = float(mis)
                rec["case_id"] = int(case_id)

                done += 1
                if rec.get("ok_solve", False):
                    solve_ok += 1
                shp = rec.get("shape") or {}
                if bool(shp.get("ok_sep", False)) or np.isfinite(float(shp.get("R0_plasma", float("nan")))):
                    shape_ok += 1

                last_mis = float(mis)
                if float(mis) < best:
                    best = float(mis)
                    with open(best_path, "w", encoding="utf-8") as bf:
                        bf.write(json.dumps(rec, indent=2) + "\n")

                f.write(json.dumps(rec) + "\n")
                f.flush()
                results.append(rec)

            dt = max(1e-6, time.time() - t0)
            rate = done / dt
            print("\r" + _status_line(tag, done, total, solve_ok, shape_ok, best, rate, last_mis),
                  end="", flush=True)

        print()
    return results


# -------------------------
# X-target inference (optional heuristic)
# -------------------------
def infer_x_target_from_geom(geom: Dict[str, Any], *, prefer: str) -> Optional[Tuple[float, float]]:
    Rin = geom.get("R_inner", None)
    Zin = geom.get("Z_inner", None)
    if Rin is None or Zin is None:
        return None
    R = np.asarray(Rin, float)
    Z = np.asarray(Zin, float)
    if R.size < 10:
        return None
    prefer = str(prefer).strip().lower()
    if prefer == "lower":
        k = int(np.argmin(Z))
        return (float(R[k] + 0.10), float(Z[k] - 0.05))
    if prefer == "upper":
        k = int(np.argmax(Z))
        return (float(R[k] + 0.10), float(Z[k] + 0.05))
    k = int(np.argmin(np.abs(Z)))
    return (float(R[k] + 0.10), float(Z[k]))


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

    # X-point options
    ap.add_argument("--infer_x", action="store_true")
    ap.add_argument("--x_target", type=str, default="")
    ap.add_argument("--null", type=str, default="lower", choices=["lower", "upper", "any"])
    ap.add_argument("--require_two_x", action="store_true")

    # weights
    ap.add_argument("--pen_no_x", type=float, default=PEN_NO_X)
    ap.add_argument("--w_xdist", type=float, default=W_XDIST)
    ap.add_argument("--bonus_has_x", type=float, default=BONUS_HAS_X)
    ap.add_argument("--w_prior", type=float, default=W_PRIOR)

    # LCFS fallback controls
    ap.add_argument("--prefer_inner", action="store_true")
    ap.add_argument("--psi_percentile", type=float, default=0.5)  # <=0 => extremes first

    args = ap.parse_args()
    mp.freeze_support()

    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    jsonl_path = outdir / "scan_multigoal_results.jsonl"
    best_path_global = outdir / "scan_multigoal_best.txt"
    best_path_batch  = outdir / "scan_multigoal_best_batch.txt"  # <- best “local” del batch

    # Reset outputs when NOT appending
    if not args.append:
        if jsonl_path.exists():
            jsonl_path.unlink(missing_ok=True)
        if best_path_global.exists():
            best_path_global.unlink(missing_ok=True)
        if best_path_batch.exists():
            best_path_batch.unlink(missing_ok=True)

    rng = np.random.default_rng(int(args.seed))

    # -------------------------
    # Helpers: global best tracker
    # -------------------------
    def _load_best_global() -> Tuple[float, Optional[Dict[str, Any]]]:
        if not best_path_global.exists():
            return float("inf"), None
        try:
            rec = json.loads(best_path_global.read_text(encoding="utf-8"))
            return float(rec.get("misfit", float("inf"))), rec
        except Exception:
            return float("inf"), None

    best_global_mis, best_global_rec = _load_best_global()

    def _maybe_update_best_global(res: List[Dict[str, Any]]) -> None:
        nonlocal best_global_mis, best_global_rec
        if not res:
            return
        # toma el mejor del batch (ya viene con misfit)
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
            best_global_rec = best_r
            best_path_global.write_text(json.dumps(best_global_rec, indent=2) + "\n", encoding="utf-8")

    # -------------------------
    # Targets from cfg
    # -------------------------
    targets = dict(
        R0_target=float(getattr(cfg, "R0_geom", 4.0)),
        A_target=float(getattr(cfg, "A_geom", 2.0)),
        kappa_target=float(getattr(cfg, "kappa_geom", 2.5)),
        delta_target=float(getattr(cfg, "delta_geom", 0.30)),
        a_min=float(getattr(cfg, "a_min", 0.7)),

        sig_R0=float(getattr(cfg, "sig_R0", SIG_R0)),
        sig_A=float(getattr(cfg, "sig_A", SIG_A)),
        sig_k=float(getattr(cfg, "sig_k", SIG_KAP)),
        sig_del=float(getattr(cfg, "sig_del", SIG_DEL)),
    )

    bounds = dict(BOUNDS_DEFAULT_MA)

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

    # CEM init around mid-bounds
    lo, hi = _bounds_arrays(bounds)
    mid = 0.5 * (lo + hi)
    sig = 0.35 * (hi - lo)
    state = CEMState(mu=mid, sigma=sig)

    # CAD init payload
    init_payload = {
        "dxf_path": args.dxf,
        "unit_scale": float(getattr(cfg, "unit_scale", 1.0)),
        "resample_walls": str(getattr(cfg, "resample_walls", "auto")),
        "n_wall": int(getattr(cfg, "n_wall", 801)),
        "n_inner": int(getattr(cfg, "n_inner", 801)),
        "n_plasma": int(getattr(cfg, "n_plasma", 801)),
        "min_wall_pts": int(getattr(cfg, "min_wall_pts", 200)),
        "enforce_ccw": bool(getattr(cfg, "enforce_ccw", True)),
        "canonical_start": bool(getattr(cfg, "canonical_start", True)),
        "flatten_distance": float(getattr(cfg, "flatten_distance", 0.0)),
        "label_match_factor": float(getattr(cfg, "label_match_factor", 0.92)),
    }

    # build geom once (for x_target inference)
    from star_machine_cad import make_star_machine_from_cad, CADImportOptions, CADLayers
    opts = CADImportOptions(
        unit_scale=float(init_payload["unit_scale"]),
        resample_walls=str(init_payload["resample_walls"]),
        n_wall=int(init_payload["n_wall"]),
        n_inner=int(init_payload["n_inner"]),
        n_plasma=int(init_payload["n_plasma"]),
        min_wall_pts=int(init_payload["min_wall_pts"]),
        enforce_ccw=bool(init_payload["enforce_ccw"]),
        canonical_start=bool(init_payload["canonical_start"]),
        flatten_distance=float(init_payload["flatten_distance"]),
        label_match_factor=float(init_payload["label_match_factor"]),
    )
    _, geom_main = make_star_machine_from_cad(args.dxf, CADLayers(), opts, strict_expected=True)

    # X target
    x_target = None
    if args.x_target.strip():
        parts = args.x_target.replace(" ", "").split(",")
        if len(parts) == 2:
            x_target = (float(parts[0]), float(parts[1]))
    elif args.infer_x:
        x_target = infer_x_target_from_geom(geom_main, prefer=str(args.null))

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
    f_list_coarse = tuple(getattr(cfg, "f_list_coarse", (0.20, 0.55, 1.0)))
    f_list_refine = tuple(getattr(cfg, "f_list_refine", (0.15, 0.35, 0.65, 1.0)))
    if not hasattr(cfg, "f_list_coarse"):
        f_list_coarse = f_list_equil
    if not hasattr(cfg, "f_list_refine"):
        f_list_refine = f_list_equil

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

    try:
        print("[INFO] scan_star_multigoal started")
        print(f"[INFO] dxf={args.dxf}")
        print(f"[INFO] workers={args.workers} n1={args.n1} n2={args.n2} pop={args.pop} timeout={args.timeout}")
        print(f"[INFO] targets: R0={targets['R0_target']} A={targets['A_target']} kappa={targets['kappa_target']} delta={targets['delta_target']}")
        print(f"[INFO] x_target={x_target} null_prefer={args.null} require_two_x={args.require_two_x}")
        print(f"[INFO] pen_no_x={args.pen_no_x} w_xdist={args.w_xdist} bonus_has_x={args.bonus_has_x} w_prior={args.w_prior}")
        print(f"[INFO] lcfs: prefer_inner={args.prefer_inner} psi_percentile={args.psi_percentile} (<=0 => extremes-first)")
        if np.isfinite(best_global_mis):
            print(f"[INFO] resuming global best: {best_global_mis:.6g}")

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
                best_path=best_path_batch,          # <- best del batch aquí
                targets=targets,
                x_target=x_target,
                null_prefer=str(args.null),
                pen_no_x=float(args.pen_no_x),
                w_xdist=float(args.w_xdist),
                bonus_has_x=float(args.bonus_has_x),
                prior_MA=prior_MA,
                w_prior=float(args.w_prior),
            )
            all1.extend(res)
            _maybe_update_best_global(res)          # <- best GLOBAL se actualiza aquí

            # stagnation check (stage1)
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
                try:
                    if np.isfinite(float(shp.get("R0_plasma", float("nan")))) and np.isfinite(float(r.get("misfit", float("inf")))):
                        ok_res.append(r)
                except Exception:
                    continue

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
                best_path=best_path_batch,          # <- best del batch aquí
                targets=targets,
                x_target=x_target,
                null_prefer=str(args.null),
                pen_no_x=float(args.pen_no_x),
                w_xdist=float(args.w_xdist),
                bonus_has_x=float(args.bonus_has_x),
                prior_MA=prior_MA,
                w_prior=float(args.w_prior),
            )
            _maybe_update_best_global(res)          # <- best GLOBAL se actualiza aquí

            # stagnation check (stage2)
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
                try:
                    if np.isfinite(float(shp.get("R0_plasma", float("nan")))) and np.isfinite(float(r.get("misfit", float("inf")))):
                        ok_res.append(r)
                except Exception:
                    continue

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
        print("[DONE] Best GLOBAL written to:", str(best_path_global))
        print("[DONE] Best-per-batch written to:", str(best_path_batch))
        print("[DONE] Log:", str(jsonl_path))
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

