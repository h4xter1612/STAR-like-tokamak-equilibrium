"""
scan_star_refine.py

Deterministic + adaptive local refinement of STAR coil family currents using
a pattern-search / coordinate-descent scheme with step-size shrink.

Key properties:
- Deterministic: no randomness unless YOU add it.
- Adaptive: step sizes shrink when no improvement (bisection-like).
- Robust: each eval runs in a subprocess with a hard timeout.
- Reuses your CAD + equilibrium pipeline via star_equilibrium.build_equilibrium().

Typical usage:
  py .\scan_star_refine.py --init best_global.json --max-iters 30 --timeout 140

Or start from config currents:
  py .\scan_star_refine.py --max-iters 25

Notes:
- Currents are treated as FAMILY totals (CS, PF1, PF2, PF3).
- Distribution across segments uses your apply_group_currents inside star_equilibrium pipeline.
"""

from __future__ import annotations

import argparse
import json
import time
import math
import multiprocessing as mp
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


# ----------------------------
# Misfit (simple + robust)
# ----------------------------
def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return default

def _compute_misfit_from_diag(diag: Dict[str, Any], cfg: Any, shape: Dict[str, Any]) -> float:
    """
    Uses LCFS-based diagnostics (diag) to build a dimensionless misfit.
    Targets taken from config_star_bean:
      R0_geom, A_geom, kappa_geom, delta_geom  (if present)
    Also penalizes:
      - negative triangularity
      - too small a
      - (optional) missing separatrix if --require-sep is used (handled outside)
    """
    if not diag or not diag.get("ok", False):
        return 1e9

    # Targets (fallbacks are sane-ish; adjust in your config)
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

    # Typical tolerances (tune if needed)
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

    # Penalties
    pen_neg = float(getattr(cfg, "penalty_neg_delta", 10.0))
    if du < 0.0 or dl < 0.0:
        misfit += pen_neg * (abs(min(du, 0.0)) + abs(min(dl, 0.0)))

    a_min = float(getattr(cfg, "a_min_m", 1.0))
    pen_thin = float(getattr(cfg, "penalty_thin", 5.0))
    if a < a_min:
        misfit += pen_thin * (a_min - a)

    # Optional: keep axis near target R0 (use if available)
    Rax = _safe_float(diag.get("R_ax", float("nan")))
    if np.isfinite(Rax):
        sig_Rax = float(getattr(cfg, "sig_Rax_m", 0.30))
        misfit += abs((Rax - R0_t) / sig_Rax) * float(getattr(cfg, "w_Rax", 0.5))

    return float(misfit)


# ----------------------------
# Subprocess worker
# ----------------------------
def _worker_eval(
    currents_A: Dict[str, float],
    dxf_path: Optional[str],
    timeout_cfg: Dict[str, Any],
    require_sep: bool,
    out_q,
) -> None:
    """
    Runs in subprocess:
      - patches cfg currents
      - calls star_equilibrium.build_equilibrium(verbose=False,...)
      - returns misfit + diagnostics
    """
    try:
        import config_star_bean as cfg
        import star_equilibrium as se

        # Apply optional speed knobs (only inside this subprocess)
        # Example: disable blanket during refine (if you want) by passing --no-blanket
        for k, v in (timeout_cfg or {}).items():
            try:
                setattr(cfg, k, v)
            except Exception:
                pass

        # Patch currents in config (FAMILY totals)
        cfg.CS_current  = float(currents_A.get("CS", 0.0))
        cfg.PF1_current = float(currents_A.get("PF1", 0.0))
        cfg.PF2_current = float(currents_A.get("PF2", 0.0))
        cfg.PF3_current = float(currents_A.get("PF3", 0.0))

        # Solve equilibrium
        eq, tokamak, geom, shape = se.build_equilibrium(
            verbose=False,
            redirect_solver_noise=False,
            dxf_path=dxf_path,
        )

        # Prefer plasma_diag stored by your build_equilibrium (your version already does that)
        diag = shape.get("plasma_diag", None) or {}
        if (not diag) or (not diag.get("ok", False)):
            # fallback if your build_equilibrium didn't attach it for some reason
            try:
                diag = se.plasma_diagnostics(eq, geom, shape)
            except Exception:
                diag = {"ok": False}

        # Optional: require separatrix solution (diverted)
        if require_sep:
            # your analyze_star returns shape["ok_sep"] when separatrix exists
            ok_sep = bool(shape.get("ok_sep", False))
            if not ok_sep:
                out_q.put({
                    "ok": True,
                    "ok_solve": True,
                    "require_sep_failed": True,
                    "misfit": 1e8,
                    "currents_A": currents_A,
                    "diag": diag,
                    "shape_ok_sep": ok_sep,
                })
                return

        misfit = _compute_misfit_from_diag(diag, cfg, shape)

        out_q.put({
            "ok": True,
            "ok_solve": True,
            "misfit": float(misfit),
            "currents_A": currents_A,
            "diag": diag,
            "shape_ok_sep": bool(shape.get("ok_sep", False)),
        })

    except Exception as e:
        out_q.put({"ok": False, "ok_solve": False, "error": repr(e), "currents_A": currents_A})


def eval_case(
    currents_A: Dict[str, float],
    *,
    dxf_path: Optional[str],
    timeout_s: float,
    require_sep: bool,
    cfg_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Parent process wrapper: subprocess + hard timeout.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_worker_eval,
        args=(currents_A, dxf_path, cfg_overrides, require_sep, q),
    )

    t0 = time.time()
    p.start()
    p.join(timeout_s)
    elapsed = time.time() - t0

    if p.is_alive():
        p.terminate()
        p.join()
        return {
            "ok": False,
            "ok_solve": False,
            "error": f"timeout>{timeout_s:.1f}s",
            "elapsed_s": float(elapsed),
            "currents_A": currents_A,
        }

    if q.empty():
        return {
            "ok": False,
            "ok_solve": False,
            "error": "no_result_from_worker",
            "elapsed_s": float(elapsed),
            "currents_A": currents_A,
        }

    res = q.get()
    res["elapsed_s"] = float(elapsed)
    return res


# ----------------------------
# Deterministic adaptive refine (pattern search)
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

    # Evaluate start
    best = eval_case(x, dxf_path=dxf_path, timeout_s=timeout_s, require_sep=require_sep, cfg_overrides=cfg_overrides)
    if not best.get("ok_solve", False):
        best["misfit"] = 1e9
    best_m = float(best.get("misfit", 1e9))

    _append_jsonl(log_jsonl, {"event": "init", "x_A": x, "step_MA": step_MA, "res": best})

    print("\n[INIT]")
    print({k: x[k]/1e6 for k in keys}, "MA")
    print("misfit =", best_m)

    for it in range(1, max_iters + 1):
        improved = False
        cand_best = None
        cand_best_m = best_m

        # Deterministic neighbor order (important for determinism)
        for k in keys:
            hA = step_MA[k] * 1e6
            if hA <= min_step_MA * 1e6:
                continue

            for sgn in (+1.0, -1.0):
                xt = dict(x)
                xt[k] = xt[k] + sgn * hA
                xt = clamp(xt)

                res = eval_case(xt, dxf_path=dxf_path, timeout_s=timeout_s, require_sep=require_sep, cfg_overrides=cfg_overrides)
                m = float(res.get("misfit", 1e9)) if res.get("ok_solve", False) else 1e9

                _append_jsonl(log_jsonl, {"event": "eval", "iter": it, "move": f"{k}{'+' if sgn>0 else '-'}", "x_A": xt, "step_MA": dict(step_MA), "res": res})

                if m < cand_best_m:
                    cand_best_m = m
                    cand_best = (xt, res)

        if cand_best is not None:
            x, best = cand_best
            best_m = cand_best_m
            improved = True

            print(f"\n[ITER {it}] improved -> misfit={best_m:.6g} @ { {k: x[k]/1e6 for k in keys} } MA")

        if not improved:
            # shrink all steps (bisection-like)
            for k in keys:
                step_MA[k] *= shrink

            print(f"\n[ITER {it}] no improvement -> shrink steps: {step_MA}")

            if max(step_MA.values()) < float(min_step_MA):
                print(f"[STOP] step < {min_step_MA} MA")
                break

    return {
        "best_currents_MA": {k: x[k] / 1e6 for k in keys},
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
    args = ap.parse_args()

    # Load x0 from config by default
    import config_star_bean as cfg
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

    # Optional: overrides inside each subprocess
    cfg_overrides: Dict[str, Any] = {}
    if args.no_blanket:
        cfg_overrides["blanket_enabled"] = False
        cfg_overrides["blanket_n_filaments"] = 0
    if args.fast:
        # fewer continuation steps (speed). tune as you like.
        cfg_overrides["f_list_equilibrium"] = (0.35, 0.70, 1.00)
        # slightly looser ramp tolerance for speed; adjust if needed
        cfg_overrides["target_rel_tol_ramp"] = 2e-8

    log_jsonl = _results_dir() / "scan_refine_results.jsonl"
    best_json = _results_dir() / "scan_refine_best.json"

    # Run refine
    out = refine(
        x0_A,
        dxf_path=args.dxf,
        timeout_s=args.timeout,
        require_sep=args.require_sep,
        cfg_overrides=cfg_overrides,
        step0_MA=step0_MA,
        shrink=float(args.shrink),
        min_step_MA=float(args.min_step),
        max_iters=int(args.max_iters),
        bounds_MA=bounds_MA,
        log_jsonl=log_jsonl,
    )

    _save_json(best_json, out)

    print("\n[DONE] Best refine:")
    print("misfit =", out["best_misfit"])
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

