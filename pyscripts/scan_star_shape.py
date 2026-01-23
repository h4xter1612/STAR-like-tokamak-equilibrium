"""
scan_star_shape.py

Robust scan for convergent STAR-like equilibria using CAD/DXF geometry.
Optimizes ONLY coil-family currents (CS_total, PF1_total, PF2_total, PF3_total).
Coil positions are fixed by DXF.

This version is UPDATED for segmented CS in CAD, e.g.:
  COIL_CS1M, COIL_CS2U, COIL_CS2L, COIL_CS3U, COIL_CS3L, COIL_CS4U, COIL_CS4L
and any similarly segmented PF families.

Core idea:
  - We treat CS/PF1/PF2/PF3 as FAMILY totals.
  - star_machine_cad.make_star_machine_from_cad() builds tokamak.coil_groups and
    tokamak.coil_group_weights (recommended).
  - We apply currents via apply_group_currents(tokamak, family_currents, mode=...),
    where mode is "area" / "equal" / "same".

Key improvements:
  - Persistent worker processes (CAD import happens ONCE per worker)
  - Hard timeout per case: if a worker hangs, it's killed and restarted
  - Short polling wait: responsive timeouts + Ctrl+C clean
  - Always logs elapsed_s (no more null)
  - Optional parallelism: --workers N

Outputs:
  - results/scan_opt_results.jsonl  (log of all attempts)
  - results/scan_opt_best.txt       (best found summary)
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

from analyze_star_shape import shape_from_separatrix
import config_star_bean as cfg


# -------------------------
# Misfit penalties (tune if needed)
# -------------------------

PENALTY_NEG_DELTA = 10.0
PENALTY_THIN      = 5.0
PENALTY_RSHIFT    = 3.0


# -------------------------
# Utilities
# -------------------------

def _full_core_mask(eq) -> np.ndarray:
    """
    Robust core mask matching current grid.
    Avoid relying on eq.psi() shape in transient phases.
    """
    return np.ones(eq.R.shape, dtype=bool)


def _ensure_profile_masks(profiles, eq, *, force: bool = False):
    """
    Ensure masks exist and match grid shape to avoid FreeGSNKE copy failures when
    some mask attributes are None.
    """
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

    # Future-proof: fill any *_core_mask that is present and None
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
    """
    Patch ConstrainPaxisIp.copy() so that right before copying, it fills None masks.
    This addresses:
      TypeError("Cannot copy <class 'NoneType'> without deepcopying")
    """
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

                # same idea for any *_core_mask present
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
    """
    Patch freegsnke.copying.copy_into so it does NOT raise when copying None
    under non-strict mode. Your build may pass allow_deepcopy=... so the wrapper
    must accept *args/**kwargs and forward them.

    Goal: eliminate TypeError("Cannot copy <class 'NoneType'> without deepcopying")
    without guessing which field is None.
    """
    import freegsnke.copying as _copying
    import freegsnke.jtor_update as _jtor

    if getattr(_copying, "_allow_none_patched", False):
        return

    _orig = _copying.copy_into

    def _copy_into_patched(src, dst, name, *args, **kwargs):
        # Parse (mutable, strict, allow_deepcopy, ...) from args/kwargs safely
        mutable = kwargs.get("mutable", False)
        strict  = kwargs.get("strict", True)

        if len(args) >= 1:
            mutable = args[0]
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

    # jtor_update often imported copy_into directly; patch that reference too
    try:
        _jtor.copy_into = _copy_into_patched
    except Exception:
        pass


def _default_dxf() -> str:
    here = Path(__file__).resolve().parent
    return str((here / "cad" / "star_baseline.dxf").resolve())


def _results_dir() -> Path:
    here = Path(__file__).resolve().parent
    return (here.parent / "results")


# -------------------------
# NEW: Family current application (segmented coils)
# -------------------------

def apply_star_family_currents(tokamak, CS, PF1, PF2, PF3, *, mode: str = "area"):
    """
    Apply FAMILY total currents to a tokamak that may contain segmented coils in CAD.

    Requires star_machine_cad.apply_group_currents() (provided by the machine cad module).
    If grouping is not present, falls back to label-based assignment for legacy CADs.

    mode:
      - "area"  : distribute by segment area (dR*dZ) weights
      - "equal" : equal split among segments in each family
      - "same"  : each segment gets the full family current (NOT recommended)
    """
    # Try to use grouping API
    try:
        from star_machine_cad import apply_group_currents
    except Exception:
        apply_group_currents = None

    family = {"CS": float(CS), "PF1": float(PF1), "PF2": float(PF2), "PF3": float(PF3)}
    mode = str(mode).lower().strip()

    if apply_group_currents is not None and hasattr(tokamak, "coil_groups"):
        apply_group_currents(tokamak, family, mode=mode)

        # Zero any other coils not in those four families
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

    # Fallback (legacy): single CS, PF1U/L, PF2U/L, PF3U/L
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
# Misfit objective
# -------------------------

def compute_misfit(eq, geom, shape, *, stage: str = "coarse") -> float:
    """
    Lower is better.
    Two-stage objective:
      - coarse: prioritize convergence + roughly correct R/A/kappa, delta soft
      - refine: tighten and start enforcing delta>0 and closer to target
    """

    # Targets
    R0_target = float(getattr(cfg, "R0_geom", geom.get("R0", 4.0)))
    A_target  = float(getattr(cfg, "A_geom", 1.7))
    k_target  = float(getattr(cfg, "kappa_geom", 1.8))
    d_target  = float(getattr(cfg, "delta_geom", 0.30))
    a_min     = float(getattr(cfg, "a_min_scan", 0.9))

    # Shape
    R0_pl    = float(shape["R0_plasma"])
    A_pl     = float(shape["A_plasma"])
    kappa_pl = float(shape["kappa_plasma"])
    du       = float(shape["delta_u"])
    dl       = float(shape["delta_l"])
    a_pl     = float(shape["a_plasma"])
    delta_bar = 0.5 * (du + dl)

    # Axis (use axis for major-radius control)
    R_ax = float(eq.magneticAxis()[0])

    # Stage-dependent tolerances/weights
    stage = str(stage).lower().strip()
    if stage == "refine":
        sig_R, sig_A, sig_k, sig_d = 0.25, 0.30, 0.30, 0.15
        delta_min = 0.08
        w_delta_min = 2.0
        w_neg_delta = 10.0
    else:
        sig_R, sig_A, sig_k, sig_d = 0.35, 0.45, 0.40, 0.25
        delta_min = 0.00
        w_delta_min = 0.0
        w_neg_delta = 4.0

    term_R = ((R_ax - R0_target) / sig_R) ** 2
    term_A = ((A_pl - A_target) / sig_A) ** 2
    term_k = ((kappa_pl - k_target) / sig_k) ** 2
    term_d = ((delta_bar - d_target) / sig_d) ** 2

    misfit = float(np.sqrt(term_R + term_A + term_k + term_d))

    if delta_bar < delta_min:
        misfit += w_delta_min * ((delta_min - delta_bar) / 0.05)

    if delta_bar < 0.0:
        misfit += w_neg_delta * abs(delta_bar)

    if a_pl < a_min:
        misfit += PENALTY_THIN * (a_min - a_pl)

    if abs(R0_pl - R0_target) > 1.0:
        misfit += PENALTY_RSHIFT * (abs(R0_pl - R0_target) - 1.0)

    return float(misfit)


# -------------------------
# Solver with continuation
# -------------------------

def solve_with_continuation(tokamak, geom, CS, PF1, PF2, PF3, *,
                            nx: int, ny: int,
                            Ip: float, paxis: float, fvac: float,
                            alpha_m: float, alpha_n: float,
                            target_rel_tol: float,
                            margin_RZ: float,
                            f_list: tuple[float, ...],
                            silence_solver: bool = True,
                            coil_group_mode: str = "area"):
    """
    Build eq and solve with a current+profile ramp in f_list.
    Returns (eq, shape) on success.

    IMPORTANT: CS/PF* are FAMILY totals; distribution to segmented coils is handled here.
    """

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

    # Silence solver prints
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

            # Controlled retries for copy(None) errors
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

    shape = shape_from_separatrix(eq, geom)
    return eq, shape


# -------------------------
# Persistent worker process
# -------------------------

def _worker_main(in_q, out_q, init_payload):
    """
    Worker loop. Loads CAD machine ONCE, then solves many jobs.
    """
    try:
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

        # FreeGSNKE defensive patches
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
                    coil_group_mode=payload.get("coil_group_mode", "area"),
                )

                if not all(k in shape for k in required):
                    raise RuntimeError(f"shape missing keys; got={list(shape.keys())}")

                stage = str(payload.get("stage", "coarse"))
                misfit = compute_misfit(eq, geom, shape, stage=stage)

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
                }))

    except Exception as e:
        out_q.put((-1, {"ok": False, "error": f"worker_init_failed: {repr(e)}", "elapsed_s": -1.0}))


class CaseRunner:
    """
    Manages one persistent worker process with hard per-case timeout.
    If a case times out, the worker is killed and restarted.
    """
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
        """
        Returns (res_dict, err_str_or_None).
        Enforces timeout with short polling.
        """
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
            now = time.perf_counter()
            remaining = deadline - now
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


# -------------------------
# Main
# -------------------------

def main():
    bounds_lo = 1e6 * np.array([0.2,  -1.6,  -1.4,  0.0])
    bounds_hi = 1e6 * np.array([2.0,   0.2,   0.8,  2.0])

    def clip_samples(x):
        return np.minimum(np.maximum(x, bounds_lo[None, :]), bounds_hi[None, :])

    ap = argparse.ArgumentParser(add_help=True)

    ap.add_argument("--dxf", default=None)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--n1", type=int, default=350)
    ap.add_argument("--n2", type=int, default=700)
    ap.add_argument("--topk", type=int, default=6, help="How many coarse OK cases to use as refine centers")

    ap.add_argument("--timeout", type=float, default=35.0)
    ap.add_argument("--workers", type=int, default=2)

    ap.add_argument("--out", default=None, help="JSONL output path (default: results/scan_opt_results.jsonl)")
    ap.add_argument("--truncate", action="store_true", help="Truncate output file at start (recommended)")

    # CAD import options
    ap.add_argument("--unit-scale", type=float, default=None)
    ap.add_argument("--resample-walls", default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--n-wall", type=int, default=801)
    ap.add_argument("--n-inner", type=int, default=801)
    ap.add_argument("--min-wall-pts", type=int, default=200)

    # NEW: how to distribute family totals over segmented coils
    ap.add_argument("--coil-group-mode", default=None, choices=["area", "equal", "same"],
                    help="How to distribute CS/PF family totals across segmented coils")

    # Scan spans (MA)
    ap.add_argument("--span-CS",  type=float, default=0.35)
    ap.add_argument("--span-PF1", type=float, default=0.75)
    ap.add_argument("--span-PF2", type=float, default=1.20)
    ap.add_argument("--span-PF3", type=float, default=1.20)

    # Coarse vs refine solver settings
    ap.add_argument("--nx1", type=int, default=33)
    ap.add_argument("--ny1", type=int, default=65)
    ap.add_argument("--tol1", type=float, default=3e-4)

    ap.add_argument("--nx2", type=int, default=65)
    ap.add_argument("--ny2", type=int, default=129)
    ap.add_argument("--tol2", type=float, default=1e-5)

    # Continuation schedules
    ap.add_argument("--f1", type=float, nargs="*", default=[0.25, 0.60, 1.0])
    ap.add_argument("--f2", type=float, nargs="*", default=[0.10, 0.20, 0.35, 0.50, 0.65, 0.78, 0.88, 1.0])

    ap.add_argument("--silence-solver", action="store_true", help="Silence solver prints (recommended)")

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

    # Center from config (recommended default)
    CS0  = float(getattr(cfg, "CS_current", 1.3e6))
    PF10 = float(getattr(cfg, "PF1_current", -7.0e5))
    PF20 = float(getattr(cfg, "PF2_current", -2.0e5))
    PF30 = float(getattr(cfg, "PF3_current",  7.0e5))
    center = np.array([CS0, PF10, PF20, PF30], dtype=float)

    halfspan = 1e6 * np.array([args.span_CS, args.span_PF1, args.span_PF2, args.span_PF3], dtype=float)

    # coil-group-mode default from cfg if not provided
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

    base_params = dict(
        margin_RZ=float(getattr(cfg, "margin_RZ", 0.5)),
        Ip=float(getattr(cfg, "Ip", 8.0e5)),
        paxis=float(getattr(cfg, "paxis", 2.0e3)),
        fvac=float(getattr(cfg, "fvac", 0.5)),
        alpha_m=float(getattr(cfg, "alpha_m", 1.8)),
        alpha_n=float(getattr(cfg, "alpha_n", 1.2)),
        silence_solver=bool(args.silence_solver),
        coil_group_mode=str(coil_group_mode),
    )

    n_workers = max(1, int(args.workers))
    runners = [CaseRunner(init_payload) for _ in range(n_workers)]

    file_lock = threading.Lock()
    print_lock = threading.Lock()

    def log_jsonl(rec: dict):
        with file_lock:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    def run_one(runner: CaseRunner, payload: dict, tag: str, idx: int, n_total: int):
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
            }

        rec = dict(res)
        rec["tag"] = tag
        rec["elapsed_s"] = float(rec.get("elapsed_s", elapsed))
        # record mode for debugging/repro
        rec["coil_group_mode"] = str(payload.get("coil_group_mode", "area"))
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
                futs.append(ex.submit(run_one, runner, payload, tag, i + 1, tried))

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
                        print(f"[{tag}] {done:4d}/{tried}  OK   misfit={rec['misfit']:.3e}  elapsed={rec['elapsed_s']:.2f}s",
                              flush=True)
                    else:
                        print(f"[{tag}] {done:4d}/{tried}  FAIL {rec.get('error','?')}  elapsed={rec['elapsed_s']:.2f}s",
                              flush=True)

        top_ok.sort(key=lambda r: r["misfit"])
        top_ok = top_ok[:max(1, int(args.topk))]
        return best, ok_count, tried, top_ok

    try:
        # -------------------------
        # Stage 1 (coarse)
        # -------------------------
        samples1 = clip_samples(sample_box(center, halfspan, int(args.n1), rng))

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
            print("Suggested next actions:")
            print("  - Increase spans (especially PF2 and PF3), or")
            print("  - Increase timeout, or")
            print("  - Try --resample-walls never temporarily, or")
            print("  - Reduce nx/ny further for coarse scan, then rerun best at full resolution.")
            return

        print(f"Best misfit (coarse): {best1['misfit']:.4e}")
        print(f"CS={best1['CS']/1e6:.3f} MA | PF1={best1['PF1']/1e6:.3f} MA | PF2={best1['PF2']/1e6:.3f} MA | PF3={best1['PF3']/1e6:.3f} MA")
        print("Shape:", best1["shape"])

        # -------------------------
        # Stage 2 (multi-center refine)
        # -------------------------
        halfspan2 = 0.25 * halfspan

        centers = []
        for r in top1:
            centers.append(np.array([r["CS"], r["PF1"], r["PF2"], r["PF3"]], dtype=float))

        n_cent = max(1, len(centers))
        n2_per = max(60, int(int(args.n2) // n_cent))

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

        print("\n=== Stage 2 (multi-center refine) ===")
        print(f"Tried: {tried2_total} | OK: {ok2_total}")

        best = best1 if (best2 is None or best1["misfit"] <= best2["misfit"]) else best2

        print("\n=== BEST OVERALL ===")
        print(f"misfit = {best['misfit']:.4e}")
        print(f"CS  = {best['CS']/1e6:.3f} MA (family total)")
        print(f"PF1 = {best['PF1']/1e6:.3f} MA (family total)")
        print(f"PF2 = {best['PF2']/1e6:.3f} MA (family total)")
        print(f"PF3 = {best['PF3']/1e6:.3f} MA (family total)")
        print(f"coil_group_mode = {coil_group_mode}")
        print(f"R_ax = {best['R_ax']:.3f} m | Z_ax = {best['Z_ax']:.3f} m")
        print("shape =", best["shape"])

        with open(best_txt, "w", encoding="utf-8") as g:
            g.write("BEST EQUILIBRIUM FOUND (CAD OPT SCAN)\n")
            g.write(f"DXF: {dxf_path}\n")
            g.write(f"coil_group_mode: {coil_group_mode}\n")
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

