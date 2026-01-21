"""
scan_star_shape.py

Robust scan for convergent STAR-like equilibria using CAD/DXF geometry.
Optimizes ONLY coil currents (CS, PF1, PF2, PF3). Coil positions are fixed by DXF.

Key improvements vs. naive subprocess-per-case:
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

def _default_dxf() -> str:
    here = Path(__file__).resolve().parent
    return str((here / "cad" / "star_baseline.dxf").resolve())

def _results_dir() -> Path:
    here = Path(__file__).resolve().parent
    return (here.parent / "results")

def set_star_currents(tokamak, CS, PF1, PF2, PF3):
    """
    Assign coil currents (A) to labels in the STAR CAD machine.
    """
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

def compute_misfit(eq, geom, shape) -> float:
    """
    Lower is better. Targets from config_star_bean if available, else fallback.
    """
    R0_target = float(getattr(cfg, "R0_geom", geom.get("R0", 4.0)))
    A_target  = float(getattr(cfg, "A_geom", 1.7))
    k_target  = float(getattr(cfg, "kappa_target", getattr(cfg, "kappa_geom", 2.2)))
    d_target  = float(getattr(cfg, "delta_target", getattr(cfg, "delta_geom", 0.35)))
    a_min     = float(getattr(cfg, "a_min_scan", 0.9))

    R0_pl    = float(shape["R0_plasma"])
    A_pl     = float(shape["A_plasma"])
    kappa_pl = float(shape["kappa_plasma"])
    du       = float(shape["delta_u"])
    dl       = float(shape["delta_l"])
    a_pl     = float(shape["a_plasma"])

    R_ax, Z_ax = eq.magneticAxis()[:2]
    R_ax = float(R_ax)

    term_Rax   = ((R_ax - R0_target) / 0.25) ** 2
    term_kappa = ((kappa_pl - k_target) / 0.30) ** 2
    delta_bar  = 0.5 * (du + dl)
    term_delta = ((delta_bar - d_target) / 0.20) ** 2
    term_A     = ((A_pl - A_target) / 0.30) ** 2

    misfit = float(np.sqrt(term_Rax + term_kappa + term_delta + term_A))

    if du < 0.0 or dl < 0.0:
        misfit += PENALTY_NEG_DELTA * (abs(min(du, 0.0)) + abs(min(dl, 0.0)))

    if abs(R0_pl - R0_target) > 1.0:
        misfit += PENALTY_RSHIFT * (abs(R0_pl - R0_target) - 1.0)

    if a_pl < a_min:
        misfit += PENALTY_THIN * (a_min - a_pl)

    return float(misfit)


def solve_with_continuation(tokamak, geom, CS, PF1, PF2, PF3, *,
                            nx: int, ny: int,
                            Ip: float, paxis: float, fvac: float,
                            alpha_m: float, alpha_n: float,
                            target_rel_tol: float,
                            margin_RZ: float,
                            f_list: tuple[float, ...],
                            silence_solver: bool = True):
    """
    Build eq and solve with a current+profile ramp in f_list.
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

    # Optionally silence the solver prints (especially "DID NOT CONVERGE")
    if silence_solver:
        import contextlib
        import io
        devnull = open(os.devnull, "w")
        redir_out = contextlib.redirect_stdout(devnull)
        redir_err = contextlib.redirect_stderr(devnull)
        ctx = contextlib.ExitStack()
        ctx.enter_context(redir_out)
        ctx.enter_context(redir_err)
    else:
        ctx = None

    try:
        for f in f_list:
            set_star_currents(tokamak, f * CS, f * PF1, f * PF2, f * PF3)

            profiles = ConstrainPaxisIp(
                eq=eq,
                paxis=float(f * paxis),
                Ip=float(f * Ip),
                fvac=float(fvac),
                alpha_m=float(alpha_m),
                alpha_n=float(alpha_n),
            )
            profiles.diverted_core_mask = np.ones_like(eq.psi(), dtype=bool)

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
                )

                if not all(k in shape for k in required):
                    raise RuntimeError(f"shape missing keys; got={list(shape.keys())}")

                misfit = compute_misfit(eq, geom, shape)
                R_ax, Z_ax = eq.magneticAxis()[:2]

                elapsed = time.perf_counter() - t0
                out_q.put((case_id, {
                    "ok": True,
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
                    "error": repr(e),
                    "elapsed_s": float(elapsed),
                    "CS": float(payload.get("CS", np.nan)),
                    "PF1": float(payload.get("PF1", np.nan)),
                    "PF2": float(payload.get("PF2", np.nan)),
                    "PF3": float(payload.get("PF3", np.nan)),
                }))

    except Exception as e:
        # Catastrophic init failure
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

        # If worker died, restart
        if self.proc is None or (not self.proc.is_alive()):
            self.restart()

        t0 = time.perf_counter()
        deadline = t0 + float(timeout_s)

        # Send job
        try:
            self.in_q.put((cid, payload))
        except Exception:
            self.restart()
            return None, "send_failed"

        # Wait with short polls (responsive + robust)
        while True:
            now = time.perf_counter()
            remaining = deadline - now
            if remaining <= 0:
                # Hard timeout: kill worker and restart
                self.restart()
                return None, "timeout"

            try:
                got_cid, res = self.out_q.get(timeout=min(0.25, remaining))
            except queue.Empty:
                # Keep waiting
                continue
            except KeyboardInterrupt:
                # Clean Ctrl+C
                self.stop()
                raise

            if got_cid != cid:
                # Stale result from old job; ignore
                continue

            # Ensure elapsed exists
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
    ap = argparse.ArgumentParser(add_help=True)

    ap.add_argument("--dxf", default=None)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--n1", type=int, default=80)
    ap.add_argument("--n2", type=int, default=120)

    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--workers", type=int, default=2)

    ap.add_argument("--out", default=None, help="JSONL output path (default: results/scan_opt_results.jsonl)")
    ap.add_argument("--truncate", action="store_true", help="Truncate output file at start (recommended)")

    # CAD import options
    ap.add_argument("--unit-scale", type=float, default=None)
    ap.add_argument("--resample-walls", default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--n-wall", type=int, default=801)
    ap.add_argument("--n-inner", type=int, default=801)
    ap.add_argument("--min-wall-pts", type=int, default=200)

    # Scan spans (MA)
    ap.add_argument("--span-CS",  type=float, default=0.8)
    ap.add_argument("--span-PF1", type=float, default=0.5)
    ap.add_argument("--span-PF2", type=float, default=0.7)
    ap.add_argument("--span-PF3", type=float, default=1.0)

    # Coarse vs refine solver settings
    ap.add_argument("--nx1", type=int, default=33)
    ap.add_argument("--ny1", type=int, default=65)
    ap.add_argument("--tol1", type=float, default=1e-4)

    ap.add_argument("--nx2", type=int, default=65)
    ap.add_argument("--ny2", type=int, default=129)
    ap.add_argument("--tol2", type=float, default=1e-5)

    # Continuation schedule (coarse faster, refine gentler)
    ap.add_argument("--f1", type=float, nargs="*", default=[0.25, 0.60, 1.0])
    ap.add_argument("--f2", type=float, nargs="*", default=[0.15, 0.30, 0.50, 0.70, 0.85, 1.0])

    ap.add_argument("--silence-solver", action="store_true", help="Silence solver prints (recommended)")

    args = ap.parse_args()
    mp.freeze_support()

    dxf_path = args.dxf or _default_dxf()
    out_dir = _results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = Path(args.out) if args.out else (out_dir / "scan_opt_results.jsonl")
    if args.truncate:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write("")  # truncate

    best_txt = out_dir / "scan_opt_best.txt"

    rng = np.random.default_rng(int(args.seed))

    # Centers from cfg
    CS0  = float(getattr(cfg, "CS_current", 0.8e6))
    PF10 = float(getattr(cfg, "PF1_current", -0.2e6))
    PF20 = float(getattr(cfg, "PF2_current", 0.0))
    PF30 = float(getattr(cfg, "PF3_current", 1.0e6))

    center = np.array([CS0, PF10, PF20, PF30], dtype=float)
    halfspan = 1e6 * np.array([args.span_CS, args.span_PF1, args.span_PF2, args.span_PF3], dtype=float)

    # Shared init payload for workers (CAD import happens once per worker)
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

    # Base physical/solver params from cfg
    base_params = dict(
        margin_RZ=float(getattr(cfg, "margin_RZ", 0.5)),
        Ip=float(getattr(cfg, "Ip", 8.0e5)),
        paxis=float(getattr(cfg, "paxis", 2.0e3)),
        fvac=float(getattr(cfg, "fvac", 0.5)),
        alpha_m=float(getattr(cfg, "alpha_m", 1.8)),
        alpha_n=float(getattr(cfg, "alpha_n", 1.2)),
        silence_solver=bool(args.silence_solver),
    )

    # Create runners
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
            rec = {
                "ok": False,
                "tag": tag,
                "CS": float(payload["CS"]),
                "PF1": float(payload["PF1"]),
                "PF2": float(payload["PF2"]),
                "PF3": float(payload["PF3"]),
                "elapsed_s": float(elapsed),
                "error": str(err),
            }
            return rec

        # trust worker elapsed if present; else use measured
        rec = dict(res)
        rec["tag"] = tag
        rec["elapsed_s"] = float(rec.get("elapsed_s", elapsed))
        return rec

    def eval_samples(samples: np.ndarray, tag: str, nx: int, ny: int, tol: float, f_list: tuple[float, ...]):
        best = None
        ok_count = 0
        tried = int(samples.shape[0])

        # Build payloads
        payloads = []
        for i in range(tried):
            CS, PF1, PF2, PF3 = map(float, samples[i, :])
            payload = dict(base_params)
            payload.update(dict(
                CS=CS, PF1=PF1, PF2=PF2, PF3=PF3,
                nx=int(nx), ny=int(ny),
                target_rel_tol=float(tol),
                f_list=tuple(float(x) for x in f_list),
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
                        print(f"[{tag}] {done:4d}/{tried}  OK   misfit={rec['misfit']:.3e}  elapsed={rec['elapsed_s']:.2f}s",
                              flush=True)
                    else:
                        print(f"[{tag}] {done:4d}/{tried}  FAIL {rec.get('error','?')}  elapsed={rec['elapsed_s']:.2f}s",
                              flush=True)

        return best, ok_count, tried

    try:
        # -------------------------
        # Stage 1 (coarse)
        # -------------------------
        samples1 = sample_box(center, halfspan, int(args.n1), rng)
        best1, ok1, tried1 = eval_samples(
            samples1, tag="coarse",
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
            print("  - Try --resample-walls never (strict retro mode) temporarily, or")
            print("  - Reduce nx/ny further for coarse scan, then rerun best at full resolution.")
            return

        print(f"Best misfit: {best1['misfit']:.4e}")
        print(f"CS={best1['CS']/1e6:.3f} MA | PF1={best1['PF1']/1e6:.3f} MA | PF2={best1['PF2']/1e6:.3f} MA | PF3={best1['PF3']/1e6:.3f} MA")
        print("Shape:", best1["shape"])

        # -------------------------
        # Stage 2 (refine)
        # -------------------------
        center2 = np.array([best1["CS"], best1["PF1"], best1["PF2"], best1["PF3"]], dtype=float)
        halfspan2 = 0.35 * halfspan
        samples2 = sample_box(center2, halfspan2, int(args.n2), rng)

        best2, ok2, tried2 = eval_samples(
            samples2, tag="refine",
            nx=int(args.nx2), ny=int(args.ny2),
            tol=float(args.tol2),
            f_list=tuple(args.f2),
        )

        print("\n=== Stage 2 (refine) ===")
        print(f"Tried: {tried2} | OK: {ok2}")

        best = best1 if (best2 is None or best1["misfit"] <= best2["misfit"]) else best2

        print("\n=== BEST OVERALL ===")
        print(f"misfit = {best['misfit']:.4e}")
        print(f"CS  = {best['CS']/1e6:.3f} MA")
        print(f"PF1 = {best['PF1']/1e6:.3f} MA")
        print(f"PF2 = {best['PF2']/1e6:.3f} MA")
        print(f"PF3 = {best['PF3']/1e6:.3f} MA")
        print(f"R_ax = {best['R_ax']:.3f} m | Z_ax = {best['Z_ax']:.3f} m")
        print("shape =", best["shape"])

        with open(best_txt, "w", encoding="utf-8") as g:
            g.write("BEST EQUILIBRIUM FOUND (CAD OPT SCAN)\n")
            g.write(f"DXF: {dxf_path}\n")
            g.write(f"misfit: {best['misfit']:.6e}\n")
            g.write(f"CS:  {best['CS']:.6e} A\n")
            g.write(f"PF1: {best['PF1']:.6e} A\n")
            g.write(f"PF2: {best['PF2']:.6e} A\n")
            g.write(f"PF3: {best['PF3']:.6e} A\n")
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

