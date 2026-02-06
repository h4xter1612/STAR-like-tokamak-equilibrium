"""
scan_star_refine.py

Deterministic + adaptive local refinement of STAR coil FAMILY currents
(CS, PF1, PF2, PF3) using a pattern-search / coordinate-descent scheme
with step-size shrink.

FIXES / FEATURES:
- True parallel neighbor evaluation per iteration (batch parallel).
- Progress printing from parent (--progress) so you can SEE it moving.
- Optional per-worker logs (--worker-logs) capturing solver stdout/stderr
  without interleaving / slowing console output.
- Robust IPC via Pipe (reduces "no_result_from_worker").
- Optional single-thread env inside workers (prevents oversubscription).

Typical:
  py .\scan_star_refine.py --init .\results\scan_multigoal_best_global.json ^
      --max-iters 25 --timeout 140 --step0 0.3,0.3,0.3,0.3 --min-step 0.02 --iter-workers 4 --progress --worker-logs
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# ----------------------------
# Misfit (simple + robust)
# ----------------------------
def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _compute_misfit_from_diag(diag: Dict[str, Any], cfg: Any, shape: Dict[str, Any]) -> float:
    if not diag or not diag.get("ok", False):
        return 1e9

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
        return 1e9

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

    misfit = math.sqrt(term)

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
        w_Rax = float(getattr(cfg, "w_Rax", 0.5))
        misfit += abs((Rax - R0_t) / sig_Rax) * w_Rax

    return float(misfit)


# ----------------------------
# Worker helpers
# ----------------------------
def _set_thread_env_single() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _worker_eval_pipe(
    idx: int,
    move: str,
    iter_tag: str,
    currents_A: Dict[str, float],
    dxf_path: Optional[str],
    cfg_overrides: Dict[str, Any],
    require_sep: bool,
    conn,
    *,
    silence: bool,
    set_single_threads: bool,
    worker_log_path: Optional[str],
) -> None:
    try:
        if set_single_threads:
            _set_thread_env_single()

        import config_star_bean as cfg
        import star_equilibrium as se

        for k, v in (cfg_overrides or {}).items():
            try:
                setattr(cfg, k, v)
            except Exception:
                pass

        cfg.CS_current  = float(currents_A.get("CS", 0.0))
        cfg.PF1_current = float(currents_A.get("PF1", 0.0))
        cfg.PF2_current = float(currents_A.get("PF2", 0.0))
        cfg.PF3_current = float(currents_A.get("PF3", 0.0))

        t0 = time.time()

        # Output routing policy:
        # - if worker_log_path: write stdout+stderr there
        # - elif silence: devnull
        # - else: inherit console (will interleave if parallel)
        if worker_log_path:
            Path(worker_log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(worker_log_path, "w", encoding="utf-8") as lf, \
                 contextlib.redirect_stdout(lf), \
                 contextlib.redirect_stderr(lf):
                eq, tokamak, geom, shape = se.build_equilibrium(
                    verbose=False,
                    redirect_solver_noise=False,
                    dxf_path=dxf_path,
                )
        elif silence:
            with open(os.devnull, "w") as dn, \
                 contextlib.redirect_stdout(dn), \
                 contextlib.redirect_stderr(dn):
                eq, tokamak, geom, shape = se.build_equilibrium(
                    verbose=False,
                    redirect_solver_noise=False,
                    dxf_path=dxf_path,
                )
        else:
            eq, tokamak, geom, shape = se.build_equilibrium(
                verbose=False,
                redirect_solver_noise=False,
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
                "idx": idx,
                "iter_tag": iter_tag,
                "move": move,
                "ok": True,
                "ok_solve": True,
                "require_sep_failed": True,
                "misfit": 1e8,
                "currents_A": currents_A,
                "diag": diag,
                "shape_ok_sep": ok_sep,
                "elapsed_s_inner": float(time.time() - t0),
                "worker_log": worker_log_path,
            }
            conn.send(res)
            return

        misfit = _compute_misfit_from_diag(diag, cfg, shape)

        res = {
            "idx": idx,
            "iter_tag": iter_tag,
            "move": move,
            "ok": True,
            "ok_solve": True,
            "misfit": float(misfit),
            "currents_A": currents_A,
            "diag": diag,
            "shape_ok_sep": ok_sep,
            "elapsed_s_inner": float(time.time() - t0),
            "worker_log": worker_log_path,
        }
        conn.send(res)

    except Exception as e:
        try:
            conn.send({
                "idx": idx,
                "iter_tag": iter_tag,
                "move": move,
                "ok": False,
                "ok_solve": False,
                "error": repr(e),
                "currents_A": currents_A,
                "worker_log": worker_log_path,
            })
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ----------------------------
# Parallel evaluation in batches
# ----------------------------
def _diag_short(diag: Dict[str, Any]) -> str:
    if not isinstance(diag, dict) or not diag.get("ok", False):
        return "diag=NA"
    R0 = _safe_float(diag.get("R0", float("nan")))
    A  = _safe_float(diag.get("A", float("nan")))
    k  = _safe_float(diag.get("kappa", float("nan")))
    du = _safe_float(diag.get("delta_u", float("nan")))
    dl = _safe_float(diag.get("delta_l", float("nan")))
    a  = _safe_float(diag.get("a", float("nan")))
    dbar = 0.5 * (du + dl) if np.isfinite(du + dl) else float("nan")
    return f"R0={R0:.3f} A={A:.3f} k={k:.3f} dbar={dbar:.3f} a={a:.3f}"


def _eval_cases_parallel(
    cases: List[Tuple[str, Dict[str, float]]],
    *,
    dxf_path: Optional[str],
    timeout_s: float,
    require_sep: bool,
    cfg_overrides: Dict[str, Any],
    iter_workers: int,
    silence_workers: bool,
    set_single_threads: bool,
    progress: bool,
    iter_tag: str,
    worker_logs: bool,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Evaluate many cases (move_name, currents_A) in parallel in batches.
    Returns results in the SAME order as `cases`.
    """
    iter_workers = max(1, int(iter_workers))
    ctx = mp.get_context("spawn")
    out: List[Tuple[str, Dict[str, Any]]] = []

    logdir = _results_dir() / "worker_logs"

    for base in range(0, len(cases), iter_workers):
        batch = cases[base:base + iter_workers]

        if progress:
            print(f"\n[{iter_tag}] launching batch {base//iter_workers + 1} ({len(batch)} jobs) ...", flush=True)

        states = []
        for j, (move, xA) in enumerate(batch):
            idx = base + j
            parent_conn, child_conn = ctx.Pipe(duplex=False)

            wlog = None
            if worker_logs:
                safe_move = move.replace("+", "p").replace("-", "m")
                wlog = str(logdir / f"worker_{iter_tag}_{safe_move}_{idx}.log")

            p = ctx.Process(
                target=_worker_eval_pipe,
                args=(idx, move, iter_tag, xA, dxf_path, cfg_overrides, require_sep, child_conn),
                kwargs={
                    "silence": silence_workers and (not worker_logs),  # if logging, don't devnull
                    "set_single_threads": set_single_threads,
                    "worker_log_path": wlog,
                },
            )
            t0 = time.time()
            p.start()
            try:
                child_conn.close()
            except Exception:
                pass

            states.append({"move": move, "idx": idx, "proc": p, "conn": parent_conn, "t0": t0, "xA": xA})

        pending = set(range(len(states)))
        while pending:
            now = time.time()
            done_now = []

            for si in list(pending):
                st = states[si]
                p = st["proc"]
                conn = st["conn"]
                elapsed = now - st["t0"]

                # Timeout
                if p.is_alive() and elapsed > timeout_s:
                    p.terminate()
                    p.join()
                    try:
                        conn.close()
                    except Exception:
                        pass
                    res = {
                        "idx": st["idx"],
                        "iter_tag": iter_tag,
                        "move": st["move"],
                        "ok": False,
                        "ok_solve": False,
                        "error": f"timeout>{timeout_s:.1f}s",
                        "elapsed_s": float(elapsed),
                        "currents_A": st["xA"],
                    }
                    st["_res"] = res
                    done_now.append(si)

                    if progress:
                        print(f"[{iter_tag}] done {st['move']:>4} | TIMEOUT | elapsed={elapsed:6.1f}s", flush=True)
                    continue

                # Finished
                if (not p.is_alive()) and (p.exitcode is not None):
                    p.join()
                    if conn.poll(0.2):
                        try:
                            res = conn.recv()
                        except Exception as e:
                            res = {
                                "idx": st["idx"],
                                "iter_tag": iter_tag,
                                "move": st["move"],
                                "ok": False,
                                "ok_solve": False,
                                "error": f"recv_failed:{repr(e)}",
                                "elapsed_s": float(elapsed),
                                "currents_A": st["xA"],
                            }
                    else:
                        res = {
                            "idx": st["idx"],
                            "iter_tag": iter_tag,
                            "move": st["move"],
                            "ok": False,
                            "ok_solve": False,
                            "error": "no_result_from_worker",
                            "exitcode": p.exitcode,
                            "elapsed_s": float(elapsed),
                            "currents_A": st["xA"],
                        }
                    try:
                        conn.close()
                    except Exception:
                        pass

                    res["elapsed_s"] = float(elapsed)
                    st["_res"] = res
                    done_now.append(si)

                    if progress:
                        if res.get("ok_solve", False):
                            m = float(res.get("misfit", 1e9))
                            ds = _diag_short(res.get("diag", {}))
                            print(f"[{iter_tag}] done {st['move']:>4} | ok | misfit={m:8.4f} | elapsed={elapsed:6.1f}s | {ds}", flush=True)
                        else:
                            print(f"[{iter_tag}] done {st['move']:>4} | FAIL ({res.get('error')}) | elapsed={elapsed:6.1f}s", flush=True)

            for si in done_now:
                pending.discard(si)

            if pending:
                time.sleep(0.05)

        # keep output order exactly as cases in this batch
        for st in states:
            out.append((st["move"], st["_res"]))

    return out


# ----------------------------
# Deterministic adaptive refine
# ----------------------------
def refine(
    x0_A: Dict[str, float],
    *,
    dxf_path: Optional[str],
    timeout_s: float,
    require_sep: bool,
    cfg_overrides: Dict[str, Any],
    step0_MA: Dict[str, float],
    shrink: float,
    min_step_MA: float,
    max_iters: int,
    bounds_MA: Dict[str, Tuple[float, float]],
    log_jsonl: Path,
    iter_workers: int,
    silence_workers: bool,
    set_single_threads: bool,
    progress: bool,
    worker_logs: bool,
) -> Dict[str, Any]:

    keys = ["CS", "PF1", "PF2", "PF3"]

    def clamp(curr: Dict[str, float]) -> Dict[str, float]:
        out = dict(curr)
        for k in keys:
            lo, hi = bounds_MA[k]
            out[k] = float(np.clip(out[k] / 1e6, lo, hi)) * 1e6
        return out

    x = clamp(dict(x0_A))
    step_MA = {k: float(step0_MA[k]) for k in keys}

    # INIT (single evaluation)
    init_cases = [("INIT", x)]
    init_res = _eval_cases_parallel(
        init_cases,
        dxf_path=dxf_path,
        timeout_s=timeout_s,
        require_sep=require_sep,
        cfg_overrides=cfg_overrides,
        iter_workers=1,
        silence_workers=silence_workers,
        set_single_threads=set_single_threads,
        progress=progress,
        iter_tag="INIT",
        worker_logs=worker_logs,
    )[0][1]

    best = init_res
    best_m = float(best.get("misfit", 1e9)) if best.get("ok_solve", False) else 1e9

    _append_jsonl(log_jsonl, {"event": "init", "x_A": x, "step_MA": step_MA, "res": best})

    print("\n[INIT]")
    print({k: x[k]/1e6 for k in keys}, "MA")
    print("misfit =", best_m, flush=True)

    for it in range(1, max_iters + 1):
        neighbor_cases: List[Tuple[str, Dict[str, float]]] = []

        for k in keys:
            hA = step_MA[k] * 1e6
            if hA <= min_step_MA * 1e6:
                continue
            for sgn in (+1.0, -1.0):
                xt = dict(x)
                xt[k] = xt[k] + sgn * hA
                xt = clamp(xt)
                move = f"{k}{'+' if sgn > 0 else '-'}"
                neighbor_cases.append((move, xt))

        if not neighbor_cases:
            print(f"\n[STOP] all steps <= {min_step_MA} MA", flush=True)
            break

        tag = f"ITER{it}"

        # Evaluate all neighbors in parallel
        evals = _eval_cases_parallel(
            neighbor_cases,
            dxf_path=dxf_path,
            timeout_s=timeout_s,
            require_sep=require_sep,
            cfg_overrides=cfg_overrides,
            iter_workers=iter_workers,
            silence_workers=silence_workers,
            set_single_threads=set_single_threads,
            progress=progress,
            iter_tag=tag,
            worker_logs=worker_logs,
        )

        cand_best_m = best_m
        cand_best_x: Optional[Dict[str, float]] = None
        cand_best_res: Optional[Dict[str, Any]] = None
        cand_best_move: Optional[str] = None

        # Log + pick best
        for (move, res) in evals:
            _append_jsonl(
                log_jsonl,
                {"event": "eval", "iter": it, "move": move, "x_A": res.get("currents_A", None), "step_MA": dict(step_MA), "res": res},
            )

            m = float(res.get("misfit", 1e9)) if res.get("ok_solve", False) else 1e9
            if m < cand_best_m:
                cand_best_m = m
                cand_best_res = res
                cand_best_move = move
                cand_best_x = res.get("currents_A", None)

        if cand_best_x is not None and cand_best_res is not None:
            x = clamp(dict(cand_best_x))
            best = cand_best_res
            best_m = float(cand_best_m)
            print(f"\n[ITER {it}] improved ({cand_best_move}) -> misfit={best_m:.6g} @ { {k: x[k]/1e6 for k in keys} } MA", flush=True)
        else:
            for k in keys:
                step_MA[k] *= float(shrink)
            print(f"\n[ITER {it}] no improvement -> shrink steps: {step_MA}", flush=True)

            if max(step_MA.values()) < float(min_step_MA):
                print(f"[STOP] step < {min_step_MA} MA", flush=True)
                break

    return {
        "best_currents_MA": {k: x[k] / 1e6 for k in keys},
        "best_currents_A": x,
        "best_misfit": float(best_m),
        "best_result": best,
        "final_steps_MA": step_MA,
        "log_jsonl": str(log_jsonl),
        "iter_workers": int(iter_workers),
        "timeout_s": float(timeout_s),
        "worker_logs": bool(worker_logs),
    }


# ----------------------------
# CLI
# ----------------------------
def main():
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
    ap.add_argument("--require-sep", action="store_true", help="Penalize cases without separatrix (diverted requirement)")
    ap.add_argument("--no-blanket", action="store_true", help="Temporarily disable blanket during refine (speed; changes physics)")
    ap.add_argument("--fast", action="store_true", help="Temporarily use a shorter continuation list (speed; may reduce robustness)")
    ap.add_argument("--iter-workers", type=int, default=4, help="Parallel workers per iteration")
    ap.add_argument("--no-silence", action="store_true", help="Do NOT silence workers (console will interleave in parallel)")
    ap.add_argument("--no-single-threads", action="store_true", help="Do NOT force BLAS/OMP threads=1 inside workers")
    ap.add_argument("--progress", action="store_true", help="Print per-eval completion lines (recommended)")
    ap.add_argument("--worker-logs", action="store_true", help="Write each worker's stdout/stderr to results/worker_logs/*.log")

    args = ap.parse_args()

    import config_star_bean as cfg

    x0_A = {
        "CS":  float(getattr(cfg, "CS_current", 0.0)),
        "PF1": float(getattr(cfg, "PF1_current", 0.0)),
        "PF2": float(getattr(cfg, "PF2_current", 0.0)),
        "PF3": float(getattr(cfg, "PF3_current", 0.0)),
    }

    if args.init:
        j = _load_json(args.init)
        cm = j.get("currents_MA", None)
        if isinstance(cm, dict) and all(k in cm for k in ("CS", "PF1", "PF2", "PF3")):
            # JSON currents are in MA -> convert to A
            x0_A = {k: float(cm[k]) * 1e6 for k in ("CS", "PF1", "PF2", "PF3")}
            print("[INIT] using currents_MA from JSON:", cm, flush=True)

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

    cfg_overrides: Dict[str, Any] = {}
    if args.no_blanket:
        cfg_overrides["blanket_enabled"] = False
        cfg_overrides["blanket_n_filaments"] = 0
    if args.fast:
        cfg_overrides["f_list_equilibrium"] = (0.35, 0.70, 1.00)
        cfg_overrides["target_rel_tol_ramp"] = 2e-8

    log_jsonl = _results_dir() / "scan_refine_results.jsonl"
    best_json = _results_dir() / "scan_refine_best.json"

    out = refine(
        x0_A,
        dxf_path=args.dxf,
        timeout_s=float(args.timeout),
        require_sep=bool(args.require_sep),
        cfg_overrides=cfg_overrides,
        step0_MA=step0_MA,
        shrink=float(args.shrink),
        min_step_MA=float(args.min_step),
        max_iters=int(args.max_iters),
        bounds_MA=bounds_MA,
        log_jsonl=log_jsonl,
        iter_workers=int(args.iter_workers),
        silence_workers=(not args.no_silence),
        set_single_threads=(not args.no_single_threads),
        progress=bool(args.progress),
        worker_logs=bool(args.worker_logs),
    )

    _save_json(best_json, out)

    print("\n[DONE] Best refine:", flush=True)
    print("misfit =", out["best_misfit"], flush=True)
    print("currents_MA =", out["best_currents_MA"], flush=True)
    print("[SAVED]", best_json, flush=True)
    print("[LOG]  ", log_jsonl, flush=True)

    if out.get("worker_logs", False):
        print("[WORKER LOGS] results/worker_logs/ (one file per eval)", flush=True)

    cm = out["best_currents_MA"]
    print("\n--- Paste into config_star_bean.py ---", flush=True)
    print(f"CS_current  = {cm['CS']:.12g}e6", flush=True)
    print(f"PF1_current = {cm['PF1']:.12g}e6", flush=True)
    print(f"PF2_current = {cm['PF2']:.12g}e6", flush=True)
    print(f"PF3_current = {cm['PF3']:.12g}e6", flush=True)


if __name__ == "__main__":
    main()

