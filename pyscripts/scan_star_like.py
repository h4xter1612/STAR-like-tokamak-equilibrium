"""
scan_star_like.py

Robust scan for convergent STAR-like equilibria using CAD/DXF geometry.
Optimizes ONLY coil-family currents (CS_total, PF1_total, PF2_total, PF3_total).
Coil positions are fixed by DXF.

UPDATED:
  - Uses LCFS-limiter extraction (analyze_star_lcfs.shape_from_lcfs_limiter)
  - STAR-like targets by default: R0~4m, A~2, kappa~2.5 (delta soft)
  - Mixes GLOBAL + LOCAL sampling in stage 1 to avoid "getting stuck"
  - Optional boundary-to-CAD target misfit if geom provides R_plasma/Z_plasma
  - Family current application supports segmented coils via apply_group_currents()

Outputs:
  - results/scan_opt_results.jsonl
  - results/scan_opt_best.txt
"""

from __future__ import annotations

import os
import time
import json
import argparse
from pathlib import Path
import multiprocessing as mp
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from freegsnke import equilibrium_update, GSstaticsolver
from freegsnke.jtor_update import ConstrainPaxisIp

# IMPORTANT: use LCFS-limiter extractor
from analyze_star_lcfs import shape_from_lcfs_limiter

# You can keep your cfg module name; just ensure targets match STAR-like.
import config_star_bean as cfg


# -------------------------
# Penalties / knobs
# -------------------------

PENALTY_NEG_DELTA = 2.0     # keep delta soft; raise only if you really want apple/bean
PENALTY_THIN      = 3.0     # small plasma penalty
PENALTY_AXIS_OUT  = 5.0     # axis far from LCFS center penalty
PENALTY_BAD_LCFS  = 30.0    # if LCFS extraction is weird, penalize heavily


# -------------------------
# Small geometry helpers
# -------------------------

def _default_dxf() -> str:
    here = Path(__file__).resolve().parent
    return str((here / "cad" / "star_baseline.dxf").resolve())

def _results_dir() -> Path:
    here = Path(__file__).resolve().parent
    return (here.parent / "results")

def _poly_arclen(R, Z):
    R = np.asarray(R, float)
    Z = np.asarray(Z, float)
    dR = np.diff(R)
    dZ = np.diff(Z)
    s = np.r_[0.0, np.cumsum(np.sqrt(dR*dR + dZ*dZ))]
    return s

def _resample_polyline_by_s(R, Z, n=400, closed=True):
    R = np.asarray(R, float)
    Z = np.asarray(Z, float)
    if closed:
        if abs(R[0] - R[-1]) + abs(Z[0] - Z[-1]) > 1e-12:
            R = np.r_[R, R[0]]
            Z = np.r_[Z, Z[0]]
    s = _poly_arclen(R, Z)
    if s[-1] <= 0:
        return R, Z
    su = np.linspace(0.0, s[-1], int(n), endpoint=True)
    Ru = np.interp(su, s, R)
    Zu = np.interp(su, s, Z)
    return Ru, Zu

def _rms_boundary_distance(Ra, Za, Rb, Zb, n=400):
    """
    Symmetric-ish RMS distance between two closed curves after arclength resampling.
    This is not a perfect Hausdorff distance, but is robust and cheap.
    """
    Ra, Za = _resample_polyline_by_s(Ra, Za, n=n, closed=True)
    Rb, Zb = _resample_polyline_by_s(Rb, Zb, n=n, closed=True)

    A = np.c_[Ra, Za]
    B = np.c_[Rb, Zb]

    # nearest neighbor approx (O(N^2) but N=400 OK)
    dAB = np.min(((A[:, None, :] - B[None, :, :])**2).sum(axis=2), axis=1)
    dBA = np.min(((B[:, None, :] - A[None, :, :])**2).sum(axis=2), axis=1)
    rms = np.sqrt(0.5*(np.mean(dAB) + np.mean(dBA)))
    return float(rms)


# -------------------------
# Masks robustness (FreeGSNKE)
# -------------------------

def _full_core_mask(eq) -> np.ndarray:
    return np.ones(eq.R.shape, dtype=bool)

def _ensure_profile_masks(profiles, eq, *, force: bool = False):
    mask = _full_core_mask(eq)

    candidates = (
        "diverted_core_mask",
        "limiter_core_mask",
        "diverted_mask",
        "limiter_mask",
        "core_mask",
    )

    for name in candidates:
        try:
            v = getattr(profiles, name, None)
        except Exception:
            continue
        try:
            if force or (v is None) or (np.asarray(v).shape != mask.shape):
                setattr(profiles, name, mask.copy())
        except Exception:
            pass

    for attr in dir(profiles):
        if not attr.endswith("_core_mask"):
            continue
        try:
            v = getattr(profiles, attr)
        except Exception:
            continue
        if v is None:
            try:
                setattr(profiles, attr, mask.copy())
            except Exception:
                pass

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
                for name in ("diverted_core_mask", "limiter_core_mask"):
                    try:
                        if getattr(self, name, None) is None:
                            setattr(self, name, mask)
                    except Exception:
                        pass
                for attr in dir(self):
                    if attr.endswith("_core_mask"):
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
    import freegsnke.jtor_update as _jtor
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
        _jtor.copy_into = _copy_into_patched
    except Exception:
        pass


# -------------------------
# Family current application (segmented coils)
# -------------------------

def apply_star_family_currents(tokamak, CS, PF1, PF2, PF3, *, mode: str = "area"):
    try:
        from star_machine_cad import apply_group_currents
    except Exception:
        apply_group_currents = None

    family = {"CS": float(CS), "PF1": float(PF1), "PF2": float(PF2), "PF3": float(PF3)}
    mode = str(mode).lower().strip()

    if apply_group_currents is not None and hasattr(tokamak, "coil_groups"):
        apply_group_currents(tokamak, family, mode=mode)

        fam_set = {"CS", "PF1", "PF2", "PF3"}
        grouped = set()
        try:
            for fam, labs in (tokamak.coil_groups or {}).items():
                if str(fam).upper() in fam_set:
                    grouped |= set(labs)
        except Exception:
            grouped = set()

        for label, coil in tokamak.coils:
            lab = str(label).strip().upper()
            if lab not in grouped:
                try:
                    coil.current = 0.0
                except Exception:
                    pass
        return

    # Fallback legacy labels
    for label, coil in tokamak.coils:
        lab = str(label).strip().upper()
        if lab == "CS":
            coil.current = float(CS)
        elif lab in ("PF1U", "PF1L"):
            coil.current = float(PF1)
        elif lab in ("PF2U", "PF2L"):
            coil.current = float(PF2)
        elif lab in ("PF3U", "PF3L"):
            coil.current = float(PF3)
        else:
            coil.current = 0.0


# -------------------------
# Misfit objective (STAR-like)
# -------------------------

def compute_misfit(eq, geom, shape, *, stage: str, targets: dict) -> float:
    """
    Lower is better.
    STAR-like objective: match (R0, A, kappa) strongly; delta is soft.
    If CAD provides a plasma target curve (R_plasma/Z_plasma), also match boundary.
    """

    # Extract shape
    R0_pl    = float(shape["R0_plasma"])
    A_pl     = float(shape["A_plasma"])
    kappa_pl = float(shape["kappa_plasma"])
    du       = float(shape["delta_u"])
    dl       = float(shape["delta_l"])
    a_pl     = float(shape["a_plasma"])
    delta_bar = 0.5 * (du + dl)

    R_ax, Z_ax = eq.magneticAxis()[:2]
    R_ax = float(R_ax)

    # Targets (STAR defaults)
    R0_t = float(targets["R0"])
    A_t  = float(targets["A"])
    k_t  = float(targets["kappa"])
    d_t  = float(targets["delta"])

    # Stage weights/tolerances
    stage = str(stage).lower().strip()
    if stage == "refine":
        sig_R0, sig_A, sig_k = 0.25, 0.35, 0.40
        sig_d = 0.30
        w_delta = 0.35   # still soft even in refine
        w_bnd   = 1.2
    else:
        sig_R0, sig_A, sig_k = 0.45, 0.60, 0.70
        sig_d = 0.45
        w_delta = 0.10
        w_bnd   = 0.35

    # Core Miller-like terms
    term_R0 = ((R0_pl - R0_t) / sig_R0) ** 2
    term_A  = ((A_pl  - A_t)  / sig_A) ** 2
    term_k  = ((kappa_pl - k_t) / sig_k) ** 2
    term_d  = ((delta_bar - d_t) / sig_d) ** 2

    mis = float(np.sqrt(term_R0 + term_A + term_k + w_delta * term_d))

    # Penalize nonsense LCFS
    if not np.isfinite(mis) or (a_pl <= 0.05) or (A_pl > 12.0) or (R0_pl < 0.5):
        return float(PENALTY_BAD_LCFS + 100.0)

    # Avoid too thin plasma
    a_min = float(getattr(cfg, "a_min_scan", 0.7))
    if a_pl < a_min:
        mis += PENALTY_THIN * (a_min - a_pl)

    # Axis should not be wildly far from the LCFS "center"
    # (this helps when solver locks into a weird outer region)
    if abs(R_ax - R0_pl) > max(0.4, 0.8 * a_pl):
        mis += PENALTY_AXIS_OUT * (abs(R_ax - R0_pl) / max(0.5, a_pl))

    # Soft triangularity positivity (keep it mild; STAR literature doesn't fix a single delta)
    if delta_bar < 0.0:
        mis += PENALTY_NEG_DELTA * abs(delta_bar)

    # Optional: match CAD plasma target curve if present
    if ("R_plasma" in geom) and ("Z_plasma" in geom) and ("R_sep" in shape) and ("Z_sep" in shape):
        try:
            rms = _rms_boundary_distance(shape["R_sep"], shape["Z_sep"], geom["R_plasma"], geom["Z_plasma"], n=300)
            # normalize by meters; ~0.2-0.4 m typical reasonable mismatch scale
            mis += w_bnd * (rms / 0.30)
        except Exception:
            # don't kill; just ignore
            pass

    return float(mis)


# -------------------------
# Solve with continuation (returns LCFS-limiter shape)
# -------------------------

def solve_with_continuation(tokamak, geom, CS, PF1, PF2, PF3, *,
                            nx: int, ny: int,
                            Ip: float, paxis: float, fvac: float,
                            alpha_m: float, alpha_n: float,
                            target_rel_tol: float,
                            margin_RZ: float,
                            f_list: tuple[float, ...],
                            silence_solver: bool,
                            coil_group_mode: str,
                            prefer_inner: bool,
                            psi_percentile: float):
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
            apply_star_family_currents(
                tokamak,
                f * CS, f * PF1, f * PF2, f * PF3,
                mode=str(coil_group_mode),
            )

            profiles = ConstrainPaxisIp(
                eq=eq,
                paxis=float(f * paxis),
                Ip=float(f * Ip),
                fvac=float(fvac),
                alpha_m=float(alpha_m),
                alpha_n=float(alpha_n),
            )

            _ensure_profile_masks(profiles, eq, force=True)

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
                        _ensure_profile_masks(profiles, eq, force=True)
                        if attempt == 1:
                            raise
                        continue
                    raise

    finally:
        if ctx is not None:
            ctx.close()

    # Extract LCFS from limiter/wall
    shape = shape_from_lcfs_limiter(eq, geom, prefer_inner=bool(prefer_inner), psi_percentile=float(psi_percentile))

    # Add axis
    R_ax, Z_ax = eq.magneticAxis()[:2]
    shape["R_ax"] = float(R_ax)
    shape["Z_ax"] = float(Z_ax)

    return eq, shape


# -------------------------
# Persistent worker process
# -------------------------

def _worker_main(in_q, out_q, init_payload):
    try:
        # Ensure headless matplotlib in spawned workers (Windows)
        import matplotlib
        matplotlib.use("Agg", force=True)

        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered*")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered*")

        from star_machine_cad import make_star_machine_from_cad, CADImportOptions, CADLayers

        opts = CADImportOptions(
            unit_scale=init_payload["unit_scale"],
            resample_walls=init_payload["resample_walls"],
            n_wall=init_payload["n_wall"],
            n_inner=init_payload["n_inner"],
            n_plasma=init_payload["n_plasma"],
            min_wall_pts=init_payload["min_wall_pts"],
            enforce_ccw=init_payload["enforce_ccw"],
            canonical_start=init_payload["canonical_start"],
            flatten_distance=init_payload["flatten_distance"],
            label_match_factor=init_payload["label_match_factor"],
        )

        tokamak, geom = make_star_machine_from_cad(
            dxf_path=init_payload["dxf_path"],
            layers=CADLayers(),
            opts=opts,
            strict_expected=True,
        )

        _patch_profiles_copy_once()
        _patch_copy_into_allow_none_once()

        required = ("R0_plasma", "A_plasma", "kappa_plasma", "delta_u", "delta_l", "a_plasma")

        while True:
            msg = in_q.get()
            if msg is None:
                break

            case_id, payload = msg
            t0 = time.perf_counter()

            try:
                eq, shape = solve_with_continuation(
                    tokamak, geom,
                    payload["CS"], payload["PF1"], payload["PF2"], payload["PF3"],
                    nx=payload["nx"],
                    ny=payload["ny"],
                    Ip=payload["Ip"],
                    paxis=payload["paxis"],
                    fvac=payload["fvac"],
                    alpha_m=payload["alpha_m"],
                    alpha_n=payload["alpha_n"],
                    target_rel_tol=payload["target_rel_tol"],
                    margin_RZ=payload["margin_RZ"],
                    f_list=payload["f_list"],
                    silence_solver=payload["silence_solver"],
                    coil_group_mode=payload["coil_group_mode"],
                    prefer_inner=payload["prefer_inner"],
                    psi_percentile=payload["psi_percentile"],
                )

                # Validate shape keys
                if not all(k in shape for k in required):
                    raise RuntimeError(f"shape missing keys; got={list(shape.keys())}")

                stage = str(payload.get("stage", "coarse"))
                misfit = compute_misfit(eq, geom, shape, stage=stage, targets=payload["targets"])

                R_ax, Z_ax = eq.magneticAxis()[:2]
                elapsed = time.perf_counter() - t0

                out_q.put((case_id, {
                    "ok": True,
                    "stage": stage,
                    "misfit": float(misfit),
                    "CS": float(payload["CS"]),
                    "PF1": float(payload["PF1"]),
                    "PF2": float(payload["PF2"]),
                    "PF3": float(payload["PF3"]),
                    "R_ax": float(R_ax),
                    "Z_ax": float(Z_ax),
                    "shape": {k: float(shape[k]) for k in required},
                    "elapsed_s": float(elapsed),
                    "coil_group_mode": str(payload["coil_group_mode"]),
                }))

            except Exception as e:
                elapsed = time.perf_counter() - t0
                out_q.put((case_id, {
                    "ok": False,
                    "stage": str(payload.get("stage", "unknown")),
                    "error": repr(e),
                    "elapsed_s": float(elapsed),
                    "CS": float(payload.get("CS", np.nan)),
                    "PF1": float(payload.get("PF1", np.nan)),
                    "PF2": float(payload.get("PF2", np.nan)),
                    "PF3": float(payload.get("PF3", np.nan)),
                    "coil_group_mode": str(payload.get("coil_group_mode", "area")),
                }))

    except Exception as e:
        out_q.put((-1, {"ok": False, "error": f"worker_init_failed: {repr(e)}", "elapsed_s": -1.0}))


class CaseRunner:
    def __init__(self, init_payload: dict):
        self.init_payload = dict(init_payload)
        self.ctx = mp.get_context("spawn")
        self.in_q = None
        self.out_q = None
        self.proc = None
        self._case_id = 0
        self._start()

    def _start(self):
        self.in_q = self.ctx.Queue()
        self.out_q = self.ctx.Queue()
        self.proc = self.ctx.Process(
            target=_worker_main,
            args=(self.in_q, self.out_q, self.init_payload),
            daemon=True,
        )
        self.proc.start()

    def stop(self):
        try:
            if self.in_q is not None:
                try:
                    self.in_q.put(None)
                except Exception:
                    pass
            if self.proc is not None and self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=2.0)
        finally:
            self.proc = None

    def restart(self):
        self.stop()
        self._start()

    def run_case(self, payload: dict, timeout_s: float):
        self._case_id += 1
        cid = self._case_id

        if self.proc is None or (not self.proc.is_alive()):
            self.restart()

        t0 = time.perf_counter()
        deadline = t0 + float(timeout_s)

        try:
            self.in_q.put((cid, payload))
        except Exception:
            self.restart()
            return None, "send_failed"

        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                self.restart()
                return None, "timeout"

            try:
                got_cid, res = self.out_q.get(timeout=min(0.25, remaining))
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                self.stop()
                raise

            if got_cid != cid:
                continue

            if "elapsed_s" not in res or res["elapsed_s"] is None:
                res["elapsed_s"] = float(time.perf_counter() - t0)

            if not res.get("ok", False):
                return None, res.get("error", "unknown")

            return res, None


# -------------------------
# Sampling
# -------------------------

def sample_box(center: np.ndarray, halfspan: np.ndarray, n: int, rng: np.random.Generator):
    u = rng.uniform(-1.0, 1.0, size=(int(n), 4))
    return center[None, :] + u * halfspan[None, :]

def sample_uniform(bounds_lo: np.ndarray, bounds_hi: np.ndarray, n: int, rng: np.random.Generator):
    u = rng.uniform(0.0, 1.0, size=(int(n), 4))
    return bounds_lo[None, :] + u * (bounds_hi - bounds_lo)[None, :]


# -------------------------
# Main
# -------------------------

def main():
    # Bounds in A (family totals)
    bounds_lo = 1e6 * np.array([0.2,  -2.0,  -2.0,  -0.5], dtype=float)
    bounds_hi = 1e6 * np.array([2.5,   0.6,   1.2,   2.5], dtype=float)

    def clip_samples(x):
        return np.minimum(np.maximum(x, bounds_lo[None, :]), bounds_hi[None, :])

    ap = argparse.ArgumentParser(add_help=True)

    ap.add_argument("--dxf", default=None)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--n1", type=int, default=500)
    ap.add_argument("--n2", type=int, default=900)
    ap.add_argument("--topk", type=int, default=6)

    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--workers", type=int, default=2)

    ap.add_argument("--out", default=None)
    ap.add_argument("--truncate", action="store_true")

    # CAD options
    ap.add_argument("--unit-scale", type=float, default=None)
    ap.add_argument("--resample-walls", default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--n-wall", type=int, default=801)
    ap.add_argument("--n-inner", type=int, default=801)
    ap.add_argument("--min-wall-pts", type=int, default=200)

    # Family distribution mode
    ap.add_argument("--coil-group-mode", default=None, choices=["area", "equal", "same"])

    # LCFS-limiter extraction settings
    ap.add_argument("--prefer-inner", action="store_true", help="Prefer inner wall (limiter) if available")
    ap.add_argument("--psi-percentile", type=float, default=5.0, help="Percentile for wall-psi boundary estimate")

    # STAR-like targets (override-able)
    ap.add_argument("--R0-target", type=float, default=4.0)
    ap.add_argument("--A-target", type=float, default=2.0)
    ap.add_argument("--kappa-target", type=float, default=2.5)
    ap.add_argument("--delta-target", type=float, default=0.20)

    # Stage 1 sampling controls
    ap.add_argument("--global-frac", type=float, default=0.65, help="fraction of n1 taken as global uniform samples")

    # Local spans (MA)
    ap.add_argument("--span-CS",  type=float, default=0.70)
    ap.add_argument("--span-PF1", type=float, default=1.20)
    ap.add_argument("--span-PF2", type=float, default=1.60)
    ap.add_argument("--span-PF3", type=float, default=1.60)

    # Solver settings per stage
    ap.add_argument("--nx1", type=int, default=29)
    ap.add_argument("--ny1", type=int, default=57)
    ap.add_argument("--tol1", type=float, default=5e-4)
    ap.add_argument("--f1", type=float, nargs="*", default=[0.20, 0.55, 1.0])

    ap.add_argument("--nx2", type=int, default=65)
    ap.add_argument("--ny2", type=int, default=129)
    ap.add_argument("--tol2", type=float, default=2e-5)
    ap.add_argument("--f2", type=float, nargs="*", default=[0.10, 0.20, 0.35, 0.50, 0.65, 0.78, 0.90, 1.0])

    ap.add_argument("--silence-solver", action="store_true")

    args = ap.parse_args()
    mp.freeze_support()

    dxf_path = args.dxf or _default_dxf()
    out_dir = _results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = Path(args.out) if args.out else (out_dir / "scan_opt_results.jsonl")
    if args.truncate:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write("")

    best_txt = out_dir / "scan_opt_best.txt"
    rng = np.random.default_rng(int(args.seed))

    # Center: use cfg currents if present, but stage1 also samples globally anyway.
    CS0  = float(getattr(cfg, "CS_current", 1.2e6))
    PF10 = float(getattr(cfg, "PF1_current", -8.0e5))
    PF20 = float(getattr(cfg, "PF2_current", -3.0e5))
    PF30 = float(getattr(cfg, "PF3_current",  8.0e5))
    center = np.array([CS0, PF10, PF20, PF30], dtype=float)

    halfspan = 1e6 * np.array([args.span_CS, args.span_PF1, args.span_PF2, args.span_PF3], dtype=float)

    coil_group_mode = args.coil_group_mode
    if coil_group_mode is None:
        coil_group_mode = str(getattr(cfg, "coil_group_mode", "area")).lower().strip()

    init_payload = dict(
        dxf_path=str(dxf_path),
        unit_scale=args.unit_scale,
        resample_walls=str(args.resample_walls),
        n_wall=int(args.n_wall),
        n_inner=int(args.n_inner),
        n_plasma=400,
        min_wall_pts=int(args.min_wall_pts),
        enforce_ccw=True,
        canonical_start=True,
        flatten_distance=0.01,
        label_match_factor=2.0,
    )

    targets = dict(R0=float(args.R0_target), A=float(args.A_target), kappa=float(args.kappa_target), delta=float(args.delta_target))

    base_params = dict(
        margin_RZ=float(getattr(cfg, "margin_RZ", 0.5)),
        Ip=float(getattr(cfg, "Ip", 8.0e5)),
        paxis=float(getattr(cfg, "paxis", 2.0e3)),
        fvac=float(getattr(cfg, "fvac", 0.5)),
        alpha_m=float(getattr(cfg, "alpha_m", 1.8)),
        alpha_n=float(getattr(cfg, "alpha_n", 1.2)),
        silence_solver=bool(args.silence_solver),
        coil_group_mode=str(coil_group_mode),
        prefer_inner=bool(args.prefer_inner),
        psi_percentile=float(args.psi_percentile),
        targets=targets,
    )

    n_workers = max(1, int(args.workers))
    runners = [CaseRunner(init_payload) for _ in range(n_workers)]

    file_lock = threading.Lock()
    print_lock = threading.Lock()

    def log_jsonl(rec: dict):
        with file_lock:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    def run_one(runner: CaseRunner, payload: dict, tag: str):
        t0 = time.perf_counter()
        res, err = runner.run_case(payload, timeout_s=float(args.timeout))
        elapsed = time.perf_counter() - t0

        if res is None:
            return {
                "ok": False,
                "tag": tag,
                "stage": str(payload.get("stage", "unknown")),
                "CS": float(payload["CS"]),
                "PF1": float(payload["PF1"]),
                "PF2": float(payload["PF2"]),
                "PF3": float(payload["PF3"]),
                "elapsed_s": float(elapsed),
                "error": str(err),
                "coil_group_mode": str(payload.get("coil_group_mode", "area")),
            }

        rec = dict(res)
        rec["tag"] = tag
        rec["elapsed_s"] = float(rec.get("elapsed_s", elapsed))
        return rec

    def eval_samples(samples: np.ndarray, *, tag: str, stage: str, nx: int, ny: int, tol: float, f_list: tuple[float, ...]):
        best = None
        ok_count = 0
        tried = int(samples.shape[0])
        top_ok = []

        payloads = []
        for i in range(tried):
            CS, PF1, PF2, PF3 = map(float, samples[i, :])
            payload = dict(base_params)
            payload.update(dict(
                CS=CS, PF1=PF1, PF2=PF2, PF3=PF3,
                nx=int(nx), ny=int(ny),
                target_rel_tol=float(tol),
                f_list=tuple(float(x) for x in f_list),
                stage=str(stage),
            ))
            payloads.append(payload)

        done = 0
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = []
            for i, payload in enumerate(payloads):
                runner = runners[i % n_workers]
                futs.append(ex.submit(run_one, runner, payload, tag))

            for fut in as_completed(futs):
                rec = fut.result()
                log_jsonl(rec)
                done += 1

                with print_lock:
                    if rec["ok"]:
                        ok_count += 1
                        if best is None or rec["misfit"] < best["misfit"]:
                            best = rec
                        top_ok.append(rec)
                        print(f"[{tag}] {done:4d}/{tried} OK   misfit={rec['misfit']:.3e}  elapsed={rec['elapsed_s']:.2f}s", flush=True)
                    else:
                        print(f"[{tag}] {done:4d}/{tried} FAIL {rec.get('error','?')}  elapsed={rec['elapsed_s']:.2f}s", flush=True)

        top_ok.sort(key=lambda r: r["misfit"])
        top_ok = top_ok[:max(1, int(args.topk))]
        return best, ok_count, tried, top_ok

    try:
        # -------------------------
        # Stage 1 (coarse) - GLOBAL + LOCAL mix
        # -------------------------
        n1 = int(args.n1)
        frac = float(args.global_frac)
        n1g = max(1, int(round(frac * n1)))
        n1l = max(1, n1 - n1g)

        samples_global = sample_uniform(bounds_lo, bounds_hi, n1g, rng)
        samples_local  = sample_box(center, halfspan, n1l, rng)
        samples1 = clip_samples(np.vstack([samples_global, samples_local]))

        best1, ok1, tried1, top1 = eval_samples(
            samples1,
            tag="coarse",
            stage="coarse",
            nx=int(args.nx1), ny=int(args.ny1),
            tol=float(args.tol1),
            f_list=tuple(args.f1),
        )

        print("\n=== Stage 1 (coarse) ===")
        print(f"Tried: {tried1} | OK: {ok1}")
        if best1 is None:
            print("No convergent equilibria found in coarse stage.")
            print("Try: increase timeout, widen bounds/spans, or loosen tol1, or use fewer nx/ny.")
            return

        print(f"Best misfit (coarse): {best1['misfit']:.4e}")
        print(f"CS={best1['CS']/1e6:.3f} MA | PF1={best1['PF1']/1e6:.3f} MA | PF2={best1['PF2']/1e6:.3f} MA | PF3={best1['PF3']/1e6:.3f} MA")
        print("Shape:", best1["shape"])

        # -------------------------
        # Stage 2 (refine) around Top-K coarse
        # -------------------------
        halfspan2 = 0.25 * halfspan
        centers = [np.array([r["CS"], r["PF1"], r["PF2"], r["PF3"]], dtype=float) for r in top1]

        n_cent = max(1, len(centers))
        n2_per = max(80, int(int(args.n2) // n_cent))

        best2 = None
        ok2_total = 0
        tried2_total = 0

        for j, c in enumerate(centers, start=1):
            samples2 = clip_samples(sample_box(c, halfspan2, n2_per, rng))

            b, ok2, tried2, _ = eval_samples(
                samples2,
                tag=f"refine{j}",
                stage="refine",
                nx=int(args.nx2), ny=int(args.ny2),
                tol=float(args.tol2),
                f_list=tuple(args.f2),
            )

            ok2_total += ok2
            tried2_total += tried2
            if b is not None and (best2 is None or b["misfit"] < best2["misfit"]):
                best2 = b

        print("\n=== Stage 2 (refine) ===")
        print(f"Tried: {tried2_total} | OK: {ok2_total}")

        best = best1 if (best2 is None or best1["misfit"] <= best2["misfit"]) else best2

        print("\n=== BEST OVERALL ===")
        print(f"targets: R0={targets['R0']:.3f} A={targets['A']:.3f} kappa={targets['kappa']:.3f} delta={targets['delta']:.3f}")
        print(f"misfit = {best['misfit']:.6e}")
        print(f"CS  = {best['CS']/1e6:.3f} MA (family total)")
        print(f"PF1 = {best['PF1']/1e6:.3f} MA (family total)")
        print(f"PF2 = {best['PF2']/1e6:.3f} MA (family total)")
        print(f"PF3 = {best['PF3']/1e6:.3f} MA (family total)")
        print(f"coil_group_mode = {coil_group_mode}")
        print(f"R_ax = {best['R_ax']:.3f} m | Z_ax = {best['Z_ax']:.3f} m")
        print("shape =", best["shape"])

        with open(best_txt, "w", encoding="utf-8") as g:
            g.write("BEST EQUILIBRIUM FOUND (CAD OPT SCAN - STAR-LIKE)\n")
            g.write(f"DXF: {dxf_path}\n")
            g.write(f"coil_group_mode: {coil_group_mode}\n")
            g.write(f"targets: R0={targets['R0']} A={targets['A']} kappa={targets['kappa']} delta={targets['delta']}\n")
            g.write(f"misfit: {best['misfit']:.6e}\n")
            g.write(f"CS_total:  {best['CS']:.6e} A\n")
            g.write(f"PF1_total: {best['PF1']:.6e} A\n")
            g.write(f"PF2_total: {best['PF2']:.6e} A\n")
            g.write(f"PF3_total: {best['PF3']:.6e} A\n")
            g.write(f"R_ax: {best['R_ax']:.6f} m\n")
            g.write(f"Z_ax: {best['Z_ax']:.6f} m\n")
            g.write("shape:\n")
            for k, v in best["shape"].items():
                g.write(f"  {k}: {v}\n")

        print(f"\nSaved results log: {jsonl_path}")
        print(f"Saved best summary: {best_txt}")

    finally:
        for r in runners:
            r.stop()


if __name__ == "__main__":
    main()

