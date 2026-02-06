"""
scan_star_like_lcfs.py

Robust scan for convergent STAR-like equilibria using CAD/DXF geometry.
Optimizes ONLY coil-family currents (CS_total, PF1_total, PF2_total, PF3_total).

UPDATED:
- Uses LCFS limiter-based geometry extraction (analyze_star_lcfs.shape_from_lcfs_limiter)
  which is appropriate for LIMITED plasmas (no X-point / no separatrix).
- Rejects bogus LCFS that touches the numerical-domain border (common failure mode).
- Family currents applied via star_machine_cad.apply_group_currents() for segmented coils.

Outputs:
  - results/scan_star_like_results.jsonl
  - results/scan_star_like_best.txt
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

from analyze_star_lcfs import shape_from_lcfs_limiter
import config_star_bean as cfg  # puedes renombrar a config_star_like si lo prefieres


# -------------------------
# Utilities: mask/copy robustness
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
                for attr in dir(self):
                    if attr.endswith("_core_mask") and getattr(self, attr, None) is None:
                        try:
                            setattr(self, attr, mask)
                        except Exception:
                            pass
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
# CAD helpers
# -------------------------

def _default_dxf() -> str:
    here = Path(__file__).resolve().parent
    return str((here / "cad" / "star_baseline.dxf").resolve())

def _results_dir() -> Path:
    here = Path(__file__).resolve().parent
    return (here.parent / "results")


def apply_star_family_currents(tokamak, CS, PF1, PF2, PF3, *, mode: str = "area"):
    """
    Apply FAMILY total currents to possibly segmented coils in CAD.
    """
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

    # fallback legacy labels
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
# Misfit (STAR-like targets)
# -------------------------

def compute_misfit(eq, shape, *, stage: str,
                   R0_target: float, A_target: float, kappa_target: float, delta_target: float) -> float:
    """
    Lower is better. Designed to be robust given experimental/parametric STAR design space.
    - Coarse: focus on R0/A/kappa; delta only forced to be not-crazy.
    - Refine: tighten + prefer positive delta.
    """
    stage = str(stage).lower().strip()

    R0 = float(shape["R0_plasma"])
    A  = float(shape["A_plasma"])
    k  = float(shape["kappa_plasma"])
    du = float(shape["delta_u"])
    dl = float(shape["delta_l"])
    d  = 0.5 * (du + dl)
    a  = float(shape["a_plasma"])
    R_ax = float(shape["R_ax"])

    # Stage tolerances
    if stage == "refine":
        sig_R, sig_A, sig_k, sig_d = 0.35, 0.35, 0.45, 0.25
        w_delta = 1.0
        delta_min = 0.05
        w_neg = 8.0
    else:
        sig_R, sig_A, sig_k, sig_d = 0.60, 0.60, 0.80, 0.40
        w_delta = 0.2
        delta_min = -0.10
        w_neg = 3.0

    # Core terms (use axis for major-radius control; LCFS for A,kappa)
    term_R = ((R_ax - R0_target) / sig_R) ** 2
    term_A = ((A - A_target) / sig_A) ** 2
    term_k = ((k - kappa_target) / sig_k) ** 2
    term_d = ((d - delta_target) / sig_d) ** 2

    mis = float(np.sqrt(term_R + term_A + term_k + w_delta * term_d))

    # Guard rails (evitan "best" absurdos si algo se fue mal)
    # STAR: A~2 => a~2 m; aceptamos amplio pero penalizamos extremos
    if a < 0.6:
        mis += 8.0 * (0.6 - a)
    if a > 3.5:
        mis += 2.0 * (a - 3.5)

    # Penaliza delta muy negativo
    if d < 0.0:
        mis += w_neg * abs(d)

    # Empuja a delta mínimo (suave)
    if d < delta_min:
        mis += 2.0 * ((delta_min - d) / 0.10)

    # Penaliza R0 LCFS muy lejos del target (suave)
    if abs(R0 - R0_target) > 1.5:
        mis += 2.0 * (abs(R0 - R0_target) - 1.5)

    return float(mis)


# -------------------------
# Solve with continuation + LCFS extraction
# -------------------------

def solve_with_continuation(tokamak, geom, CS, PF1, PF2, PF3, *,
                            nx: int, ny: int,
                            Ip: float, paxis: float, fvac: float,
                            alpha_m: float, alpha_n: float,
                            target_rel_tol: float,
                            margin_RZ: float,
                            f_list: tuple[float, ...],
                            coil_group_mode: str,
                            prefer_inner_limiter: bool,
                            silence_solver: bool = True):
    """
    Build eq and solve with current/profile ramp in f_list.
    Returns (eq, shape) on success.
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

    ctx = None
    if silence_solver:
        import contextlib
        ctx = contextlib.ExitStack()
        devnull = ctx.enter_context(open(os.devnull, "w"))
        ctx.enter_context(contextlib.redirect_stdout(devnull))
        ctx.enter_context(contextlib.redirect_stderr(devnull))

    try:
        for f in f_list:
            apply_star_family_currents(tokamak, f*CS, f*PF1, f*PF2, f*PF3, mode=coil_group_mode)

            profiles = ConstrainPaxisIp(
                eq=eq,
                paxis=float(f * paxis),
                Ip=float(f * Ip),
                fvac=float(fvac),
                alpha_m=float(alpha_m),
                alpha_n=float(alpha_n),
            )
            _ensure_profile_masks(profiles, eq, force=True)

            # retry on copy(None)
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
                    if ("NoneType" in msg and "Cannot copy" in msg) or ("without deepcopying" in msg):
                        _ensure_profile_masks(profiles, eq, force=True)
                        if attempt == 1:
                            raise
                        continue
                    raise
    finally:
        if ctx is not None:
            ctx.close()

    shape = shape_from_lcfs_limiter(eq, geom, prefer_inner=prefer_inner_limiter)
    return eq, shape


# -------------------------
# Persistent worker
# -------------------------

def _worker_main(in_q, out_q, init_payload):
    try:
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        from star_machine_cad import make_star_machine_from_cad, CADImportOptions

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
            opts=opts,
            strict_expected=True,
        )

        _patch_profiles_copy_once()
        _patch_copy_into_allow_none_once()

        required = ("R0_plasma","A_plasma","kappa_plasma","delta_u","delta_l","a_plasma","R_ax","Z_ax","psi_axis","psi_lcfs","axis_is_min")

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
                    nx=payload["nx"], ny=payload["ny"],
                    Ip=payload["Ip"], paxis=payload["paxis"], fvac=payload["fvac"],
                    alpha_m=payload["alpha_m"], alpha_n=payload["alpha_n"],
                    target_rel_tol=payload["target_rel_tol"],
                    margin_RZ=payload["margin_RZ"],
                    f_list=payload["f_list"],
                    coil_group_mode=payload["coil_group_mode"],
                    prefer_inner_limiter=payload["prefer_inner_limiter"],
                    silence_solver=payload["silence_solver"],
                )

                if not all(k in shape for k in required):
                    raise RuntimeError(f"shape missing keys; got={list(shape.keys())}")

                stage = str(payload.get("stage","coarse"))
                misfit = compute_misfit(
                    eq, shape, stage=stage,
                    R0_target=payload["R0_target"],
                    A_target=payload["A_target"],
                    kappa_target=payload["kappa_target"],
                    delta_target=payload["delta_target"],
                )

                elapsed = time.perf_counter() - t0
                out_q.put((case_id, {
                    "ok": True,
                    "stage": stage,
                    "misfit": float(misfit),
                    "CS": float(payload["CS"]),
                    "PF1": float(payload["PF1"]),
                    "PF2": float(payload["PF2"]),
                    "PF3": float(payload["PF3"]),
                    "shape": {k: (bool(shape[k]) if isinstance(shape[k], bool) else float(shape[k])) for k in required},
                    "elapsed_s": float(elapsed),
                    "coil_group_mode": str(payload["coil_group_mode"]),
                }))

            except Exception as e:
                elapsed = time.perf_counter() - t0
                out_q.put((case_id, {
                    "ok": False,
                    "stage": str(payload.get("stage","unknown")),
                    "error": repr(e),
                    "elapsed_s": float(elapsed),
                    "CS": float(payload.get("CS", np.nan)),
                    "PF1": float(payload.get("PF1", np.nan)),
                    "PF2": float(payload.get("PF2", np.nan)),
                    "PF3": float(payload.get("PF3", np.nan)),
                    "coil_group_mode": str(payload.get("coil_group_mode","?")),
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
        self.proc = self.ctx.Process(target=_worker_main, args=(self.in_q, self.out_q, self.init_payload), daemon=True)
        self.proc.start()

    def stop(self):
        try:
            if self.in_q is not None:
                try: self.in_q.put(None)
                except Exception: pass
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
            if got_cid != cid:
                continue
            if "elapsed_s" not in res or res["elapsed_s"] is None:
                res["elapsed_s"] = float(time.perf_counter() - t0)
            if not res.get("ok", False):
                return None, res.get("error", "unknown")
            return res, None


def sample_box(center: np.ndarray, halfspan: np.ndarray, n: int, rng: np.random.Generator):
    u = rng.uniform(-1.0, 1.0, size=(int(n), 4))
    return center[None, :] + u * halfspan[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dxf", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=60.0)

    ap.add_argument("--n1", type=int, default=600)
    ap.add_argument("--n2", type=int, default=1200)
    ap.add_argument("--topk", type=int, default=8)

    ap.add_argument("--out", default=None)
    ap.add_argument("--truncate", action="store_true")

    # CAD import
    ap.add_argument("--unit-scale", type=float, default=None)
    ap.add_argument("--resample-walls", default="auto", choices=["auto","always","never"])
    ap.add_argument("--n-wall", type=int, default=801)
    ap.add_argument("--n-inner", type=int, default=801)
    ap.add_argument("--min-wall-pts", type=int, default=200)

    ap.add_argument("--coil-group-mode", default=None, choices=["area","equal","same"])
    ap.add_argument("--prefer-inner-limiter", action="store_true", help="Use inner wall as limiter if available")
    ap.add_argument("--silence-solver", action="store_true")

    # Targets (STAR-like)
    ap.add_argument("--R0-target", type=float, default=4.0)
    ap.add_argument("--A-target", type=float, default=2.0)
    ap.add_argument("--kappa-target", type=float, default=2.5)
    ap.add_argument("--delta-target", type=float, default=0.25)

    # Bounds (A)
    ap.add_argument("--CS-lo", type=float, default=0.0)
    ap.add_argument("--CS-hi", type=float, default=2.0)
    ap.add_argument("--PF1-lo", type=float, default=-2.0)
    ap.add_argument("--PF1-hi", type=float, default=1.0)
    ap.add_argument("--PF2-lo", type=float, default=-2.0)
    ap.add_argument("--PF2-hi", type=float, default=1.5)
    ap.add_argument("--PF3-lo", type=float, default=-1.0)
    ap.add_argument("--PF3-hi", type=float, default=2.0)

    # Spans for sampling around center (MA)
    ap.add_argument("--span-CS",  type=float, default=0.8)
    ap.add_argument("--span-PF1", type=float, default=0.9)
    ap.add_argument("--span-PF2", type=float, default=0.9)
    ap.add_argument("--span-PF3", type=float, default=0.9)

    # Coarse/refine solver settings
    ap.add_argument("--nx1", type=int, default=33)
    ap.add_argument("--ny1", type=int, default=65)
    ap.add_argument("--tol1", type=float, default=3e-4)
    ap.add_argument("--nx2", type=int, default=65)
    ap.add_argument("--ny2", type=int, default=129)
    ap.add_argument("--tol2", type=float, default=1e-5)
    ap.add_argument("--f1", type=float, nargs="*", default=[0.20, 0.55, 1.00])
    ap.add_argument("--f2", type=float, nargs="*", default=[0.10, 0.20, 0.35, 0.50, 0.65, 0.78, 0.90, 1.00])

    args = ap.parse_args()
    mp.freeze_support()

    dxf_path = args.dxf or _default_dxf()
    out_dir = _results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = Path(args.out) if args.out else (out_dir / "scan_star_like_results.jsonl")
    if args.truncate:
        jsonl_path.write_text("", encoding="utf-8")
    best_txt = out_dir / "scan_star_like_best.txt"

    rng = np.random.default_rng(int(args.seed))

    # center: usa cfg si existe, pero no depende de geometría fija
    CS0  = float(getattr(cfg, "CS_current", 0.8e6))
    PF10 = float(getattr(cfg, "PF1_current", -0.8e6))
    PF20 = float(getattr(cfg, "PF2_current", -0.3e6))
    PF30 = float(getattr(cfg, "PF3_current",  0.6e6))
    center = np.array([CS0, PF10, PF20, PF30], dtype=float)

    halfspan = 1e6 * np.array([args.span_CS, args.span_PF1, args.span_PF2, args.span_PF3], dtype=float)

    bounds_lo = 1e6*np.array([args.CS_lo,  args.PF1_lo, args.PF2_lo, args.PF3_lo], float)
    bounds_hi = 1e6*np.array([args.CS_hi,  args.PF1_hi, args.PF2_hi, args.PF3_hi], float)

    def clip_samples(x):
        return np.minimum(np.maximum(x, bounds_lo[None,:]), bounds_hi[None,:])

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
        prefer_inner_limiter=bool(args.prefer_inner_limiter),
        R0_target=float(args.R0_target),
        A_target=float(args.A_target),
        kappa_target=float(args.kappa_target),
        delta_target=float(args.delta_target),
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
            return dict(ok=False, tag=tag, error=str(err), elapsed_s=float(elapsed), **{k: float(payload[k]) for k in ["CS","PF1","PF2","PF3"]})
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
            futs = [ex.submit(run_one, runners[i % n_workers], payloads[i], tag) for i in range(tried)]
            for fut in as_completed(futs):
                rec = fut.result()
                log_jsonl(rec)
                done += 1
                with print_lock:
                    if rec.get("ok", False):
                        ok_count += 1
                        if best is None or rec["misfit"] < best["misfit"]:
                            best = rec
                        top_ok.append(rec)
                        print(f"[{tag}] {done:4d}/{tried} OK   misfit={rec['misfit']:.3e}  t={rec['elapsed_s']:.1f}s", flush=True)
                    else:
                        print(f"[{tag}] {done:4d}/{tried} FAIL {rec.get('error','?')}  t={rec.get('elapsed_s',0):.1f}s", flush=True)

        top_ok.sort(key=lambda r: r["misfit"])
        return best, ok_count, tried, top_ok[:max(1, int(args.topk))]

    try:
        # Stage 1: coarse
        samples1 = clip_samples(sample_box(center, halfspan, int(args.n1), rng))
        best1, ok1, tried1, top1 = eval_samples(samples1, tag="coarse", stage="coarse",
                                                nx=int(args.nx1), ny=int(args.ny1), tol=float(args.tol1),
                                                f_list=tuple(args.f1))

        print("\n=== Stage 1 (coarse) ===")
        print(f"Tried: {tried1} | OK: {ok1}")
        if best1 is None:
            print("No valid equilibria found (likely LCFS extraction rejected them).")
            return

        # Stage 2: refine around top-k
        halfspan2 = 0.25 * halfspan
        centers = [np.array([r["CS"], r["PF1"], r["PF2"], r["PF3"]], float) for r in top1]
        n_cent = max(1, len(centers))
        n2_per = max(80, int(int(args.n2) // n_cent))

        best2 = None
        ok2_total = 0
        tried2_total = 0
        for j, c in enumerate(centers, start=1):
            samples2 = clip_samples(sample_box(c, halfspan2, n2_per, rng))
            b, ok2, tried2, _ = eval_samples(samples2, tag=f"refine{j}", stage="refine",
                                             nx=int(args.nx2), ny=int(args.ny2), tol=float(args.tol2),
                                             f_list=tuple(args.f2))
            ok2_total += ok2
            tried2_total += tried2
            if b is not None and (best2 is None or b["misfit"] < best2["misfit"]):
                best2 = b

        best = best2 if (best2 is not None and best2["misfit"] < best1["misfit"]) else best1

        print("\n=== BEST OVERALL ===")
        print(f"misfit = {best['misfit']:.6e}")
        print(f"CS_total  = {best['CS']/1e6:.3f} MA")
        print(f"PF1_total = {best['PF1']/1e6:.3f} MA")
        print(f"PF2_total = {best['PF2']/1e6:.3f} MA")
        print(f"PF3_total = {best['PF3']/1e6:.3f} MA")
        print(f"coil_group_mode = {best.get('coil_group_mode','?')}")
        print("shape =", best["shape"])

        with open(best_txt, "w", encoding="utf-8") as g:
            g.write("BEST EQUILIBRIUM FOUND (STAR-LIKE LCFS SCAN)\n")
            g.write(f"DXF: {dxf_path}\n")
            g.write(f"seed: {int(args.seed)}\n")
            g.write(f"workers: {int(args.workers)}\n")
            g.write(f"coil_group_mode: {coil_group_mode}\n")
            g.write(f"targets: R0={args.R0_target} A={args.A_target} kappa={args.kappa_target} delta={args.delta_target}\n")
            g.write(f"misfit: {best['misfit']:.6e}\n")
            g.write(f"CS_total:  {best['CS']:.6e} A\n")
            g.write(f"PF1_total: {best['PF1']:.6e} A\n")
            g.write(f"PF2_total: {best['PF2']:.6e} A\n")
            g.write(f"PF3_total: {best['PF3']:.6e} A\n")
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

