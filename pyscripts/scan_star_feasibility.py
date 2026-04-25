#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scan_star_feasibility.py

Feasibility-first stochastic scan for STAR-like tokamak equilibrium (FreeGSNKE).

Goal:
- Rebuild a convergent branch after changes in Ip / fvac / paxis.
- Use coarse mesh + physics continuation (ramp factors) to find ANY convergent equilibria.
- Only later, switch to CAD objective / double-null / strike windows refinement.

Key ideas:
- Outer loop: ramp physics (Ip, paxis, fvac) by factor f in a list (default: 0.2..1.0)
- Inner loop: stochastic scan over selected coil FAMILY currents (subset scan_keys)
- Score: prioritize "solver converged" + (optional) true separatrix + (optional) soft inner-wall containment

Outputs (./results):
- scan_feasibility_results.jsonl
- scan_feasibility_best.json

Recommended first run (PowerShell):
  py .\scan_star_feasibility.py --workers 4 --timeout 240 --pop 24 --iters 40 --ramp "0.2,0.35,0.5,0.7,0.85,1.0" --scan-keys "CS,PF1,PF2,PF3" --require-sep

Then:
  take best currents, update config, run refine (CAD objective).

Notes:
- This script uses a subprocess per eval (spawn) for hard timeout stability on Windows.
- It applies temporary config overrides (coarse mesh, tolerances, blanket off) unless you disable with --no-coarse.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from matplotlib.path import Path as MplPath


# ----------------------------
# JSON-safe helpers
# ----------------------------
def _to_builtin(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
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
# Paths
# ----------------------------
def _here() -> Path:
    return Path(__file__).resolve().parent


def _results_dir() -> Path:
    d = _here() / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(_json_dumps_safe(obj) + "\n")


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_json_dumps_safe(obj, indent=2))


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


def _open_curve_from_closed(R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    P = np.column_stack([np.asarray(R, float), np.asarray(Z, float)])
    if P.shape[0] < 3:
        return P
    if np.linalg.norm(P[0] - P[-1]) < 1e-12:
        P = P[:-1]
    return P


def _extract_lcfs_curve(shape: Dict[str, Any], diag: Dict[str, Any]) -> Optional[np.ndarray]:
    # 1) separatrix/LCFS in shape
    key_pairs = [
        ("R_sep", "Z_sep"),
        ("R_separatrix", "Z_separatrix"),
        ("R_lcfs", "Z_lcfs"),
        ("lcfs_R", "lcfs_Z"),
        ("R_LCFS", "Z_LCFS"),
    ]
    for rkey, zkey in key_pairs:
        if rkey in shape and zkey in shape:
            try:
                P = _open_curve_from_closed(np.asarray(shape[rkey], float), np.asarray(shape[zkey], float))
                if P.shape[0] >= 20:
                    return P
            except Exception:
                pass

    # 2) diag fallback LCFS (your star_equilibrium stores R_lcfs/Z_lcfs here)
    if isinstance(diag, dict):
        Rf = diag.get("R_lcfs", None)
        Zf = diag.get("Z_lcfs", None)
        if Rf is not None and Zf is not None:
            try:
                P = _open_curve_from_closed(np.asarray(Rf, float), np.asarray(Zf, float))
                if P.shape[0] >= 20:
                    return P
            except Exception:
                pass

    # 3) analyze_star fallback_lcfs dict
    fb = shape.get("fallback_lcfs", None)
    if isinstance(fb, dict) and ("R" in fb) and ("Z" in fb):
        try:
            P = _open_curve_from_closed(np.asarray(fb["R"], float), np.asarray(fb["Z"], float))
            if P.shape[0] >= 20:
                return P
        except Exception:
            pass

    return None


def _frac_outside(points: np.ndarray, wall_open: np.ndarray, radius: float = -1e-9) -> float:
    P = np.asarray(points, float)
    W = np.asarray(wall_open, float)
    if P.ndim != 2 or W.ndim != 2 or P.shape[1] != 2 or W.shape[1] != 2:
        return float("nan")
    if P.shape[0] < 20 or W.shape[0] < 3:
        return float("nan")
    path = MplPath(W, closed=True)
    inside = path.contains_points(P, radius=float(radius))
    return float(1.0 - np.mean(inside))


# ----------------------------
# Families / sampling
# ----------------------------
FAMILIES_ALL = ("CS", "PF1", "PF2", "PF3", "PF4", "PF5", "PF6")


def _currents_A_to_MA(curr_A: Dict[str, float]) -> Dict[str, float]:
    return {k: float(curr_A.get(k, 0.0) / 1e6) for k in FAMILIES_ALL}


def _currents_MA_to_A(curr_MA: Dict[str, float]) -> Dict[str, float]:
    return {k: float(curr_MA.get(k, 0.0) * 1e6) for k in FAMILIES_ALL}


def _get_prior_from_cfg(cfg: Any) -> Dict[str, float]:
    prior: Dict[str, float] = {}
    for k in FAMILIES_ALL:
        try:
            prior[k] = float(getattr(cfg, f"{k}_current"))
        except Exception:
            prior[k] = 0.0
    return prior


def _default_hard_limits_MA() -> Dict[str, float]:
    # Conservative exploration bounds; you can override with cfg.hard_limits_MA
    return {"CS": 80.0, "PF1": 20.0, "PF2": 20.0, "PF3": 12.0, "PF4": 12.0, "PF5": 15.0, "PF6": 25.0}


def _clip_hard(curr_MA: Dict[str, float], hard_MA: Dict[str, float]) -> Dict[str, float]:
    out = dict(curr_MA)
    for k in FAMILIES_ALL:
        lim = float(abs(hard_MA.get(k, 1e9)))
        out[k] = float(np.clip(out.get(k, 0.0), -lim, +lim))
    return out


def _make_bounds_centered(
    prior_MA: Dict[str, float],
    hard_MA: Dict[str, float],
    scan_keys: List[str],
    *,
    widen: float,
) -> Dict[str, Tuple[float, float]]:
    """
    Bounds only matter for scan_keys. Non-scan keys will be held fixed at the prior.
    """
    base = {"CS": 1.5, "PF1": 1.5, "PF2": 1.5, "PF3": 1.0, "PF4": 1.0, "PF5": 1.2, "PF6": 1.8}
    bounds: Dict[str, Tuple[float, float]] = {}
    for k in scan_keys:
        p = float(prior_MA.get(k, 0.0))
        span = float(widen) * max(abs(p), 0.35) + float(base.get(k, 1.0))
        lo = p - span
        hi = p + span
        lim = float(abs(hard_MA.get(k, 1e9)))
        lo = max(lo, -lim)
        hi = min(hi, +lim)
        bounds[k] = (float(lo), float(hi))
    return bounds


def _sigma_from_bounds(bounds: Dict[str, Tuple[float, float]], frac: float) -> Dict[str, float]:
    sig = {}
    for k, (lo, hi) in bounds.items():
        sig[k] = float(frac) * (float(hi) - float(lo))
    return sig


def _sample_uniform(prior_MA: Dict[str, float], bounds: Dict[str, Tuple[float, float]], scan_keys: List[str]) -> Dict[str, float]:
    x = dict(prior_MA)
    for k in scan_keys:
        lo, hi = bounds[k]
        x[k] = random.uniform(lo, hi)
    return x


def _sample_gauss(center: Dict[str, float], sigma: Dict[str, float], bounds: Dict[str, Tuple[float, float]], scan_keys: List[str]) -> Dict[str, float]:
    x = dict(center)
    for k in scan_keys:
        s = float(sigma.get(k, 0.5))
        lo, hi = bounds[k]
        val = random.gauss(float(center.get(k, 0.0)), s)
        x[k] = float(np.clip(val, lo, hi))
    return x


# ----------------------------
# Worker: evaluate one case in subprocess (hard timeout)
# ----------------------------
def _worker_eval(
    currents_A: Dict[str, float],
    dxf_path: Optional[str],
    cfg_overrides: Dict[str, Any],
    require_sep: bool,
    enforce_inner: bool,
    inner_tol: float,
    redirect_solver_noise: bool,
    conn,
) -> None:
    score = 1e18  # default fail score, ALWAYS defined
    try:
        import config_star_bean as cfg
        import star_equilibrium as se
        import numpy as np

        # apply overrides
        for k, v in (cfg_overrides or {}).items():
            try:
                setattr(cfg, k, v)
            except Exception:
                pass

        # apply currents into cfg
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

        reason = str(shape.get("reason", "")).strip().lower()
        has_true_sep = bool(shape.get("ok_sep", False)) and (reason == "ok")

        # Start scoring (now safe)
        score = 0.0

        if require_sep and (not has_true_sep):
            score += 1e6

        # Optional soft inner-wall penalty
        frac_out_inner = None
        if bool(enforce_inner) and ("R_inner" in geom) and ("Z_inner" in geom):
            # Extract LCFS curve if you use it here; if not, skip penalty safely
            # (If you have _extract_lcfs_curve in this file, use it; else set frac_out_inner=None)
            try:
                lcfs = _extract_lcfs_curve(shape, diag)
            except Exception:
                lcfs = None

            if lcfs is not None and lcfs.shape[0] >= 20:
                try:
                    wall_inner = _open_curve_from_closed(
                        np.asarray(geom["R_inner"], float),
                        np.asarray(geom["Z_inner"], float),
                    )
                    frac = _frac_outside(
                        lcfs, wall_inner,
                        radius=float(getattr(cfg, "inner_containment_radius", -1e-9)),
                    )
                    frac_out_inner = float(frac) if np.isfinite(frac) else None
                except Exception:
                    frac_out_inner = None

            if frac_out_inner is not None:
                score += 200.0 * max(0.0, float(frac_out_inner) - float(inner_tol))

        # Optional: scalar-shape penalty (guarded)
        try:
            meta = geom.get("plasma_auto_meta", {}) or {}
            R0_t = float(meta.get("geom_R0_mid", getattr(cfg, "R0_geom", 4.0)))
            A_t  = float(meta.get("geom_A_mid",  getattr(cfg, "A_geom",  2.0)))
            k_t  = float(meta.get("geom_kappa_mid", getattr(cfg, "kappa_geom", 2.5)))
            du_t = float(meta.get("geom_delta_u", getattr(cfg, "delta_geom", 0.6)))
            dl_t = float(meta.get("geom_delta_l", getattr(cfg, "delta_geom", 0.6)))
            a_t  = float(meta.get("geom_a_mid", R0_t / max(A_t, 1e-9)))

            R0 = float(diag.get("R0", np.nan))
            A  = float(diag.get("A", np.nan))
            k  = float(diag.get("kappa", np.nan))
            du = float(diag.get("delta_u", np.nan))
            dl = float(diag.get("delta_l", np.nan))
            a  = float(diag.get("a", np.nan))
            Rax = float(diag.get("R_ax", np.nan))

            dbar_t = 0.5 * (du_t + dl_t)
            dbar   = 0.5 * (du + dl)

            sR0, sA, sk, sd, sa, sRax = 0.40, 0.35, 0.35, 0.25, 0.60, 0.80
            term = 0.0
            if np.isfinite(R0 + A + k + dbar + a):
                term += ((R0 - R0_t)/sR0)**2
                term += ((A  - A_t )/sA )**2
                term += ((k  - k_t )/sk )**2
                term += ((dbar - dbar_t)/sd)**2
                term += ((a  - a_t )/sa )**2
            if np.isfinite(Rax):
                term += ((Rax - R0_t)/sRax)**2

            score += 50.0 * term
        except Exception:
            # do not kill the worker if scalar penalty has an issue
            pass

        res = {
            "ok": True,
            "ok_solve": True,
            "score": float(score),
            "currents_A": dict(currents_A),
            "currents_MA": _currents_A_to_MA(currents_A),
            "has_true_sep": bool(has_true_sep),
            "frac_out_inner": frac_out_inner,
            "diag": diag,
            "shape_reason": shape.get("reason", None),
        }
        conn.send(res)
        conn.close()

    except Exception as e:
        # IMPORTANT: score always exists here
        try:
            conn.send({
                "ok": False,
                "ok_solve": False,
                "score": float(score),
                "error": repr(e),
                "currents_A": dict(currents_A),
            })
            conn.close()
        except Exception:
            pass

def eval_case(
    currents_A: Dict[str, float],
    *,
    dxf_path: Optional[str],
    timeout_s: float,
    cfg_overrides: Dict[str, Any],
    require_sep: bool,
    enforce_inner: bool,
    inner_tol: float,
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
            bool(enforce_inner),
            float(inner_tol),
            bool(redirect_solver_noise),
            send_conn,
        ),
    )

    t0 = time.time()
    p.start()

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
# Main scan
# ----------------------------
def scan_feasibility(
    *,
    dxf_path: Optional[str],
    workers: int,
    timeout_s: float,
    iters: int,
    pop: int,
    seed: int,
    ramp: List[float],
    scan_keys: List[str],
    require_sep: bool,
    enforce_inner: bool,
    inner_tol: float,
    no_coarse: bool,
    show_solver: bool,
    widen: float,
) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))

    import config_star_bean as cfg

    # Prior and hard limits
    prior_A = _get_prior_from_cfg(cfg)
    prior_MA = _currents_A_to_MA(prior_A)

    hard = getattr(cfg, "hard_limits_MA", None)
    hard_MA = {str(k).upper(): float(v) for k, v in hard.items()} if isinstance(hard, dict) and hard else _default_hard_limits_MA()
    prior_MA = _clip_hard(prior_MA, hard_MA)

    scan_keys = [str(k).strip().upper() for k in scan_keys if str(k).strip()]
    scan_keys = [k for k in scan_keys if k in FAMILIES_ALL]
    if not scan_keys:
        raise ValueError("scan_keys empty. Example: --scan-keys 'CS,PF1,PF2,PF3'")

    bounds = _make_bounds_centered(prior_MA, hard_MA, scan_keys, widen=float(widen))
    sigma = _sigma_from_bounds(bounds, frac=0.18)

    # Coarse overrides (feasibility mode)
    # You can disable with --no-coarse if you want to use cfg as-is.
    base_overrides: Dict[str, Any] = {}
    if not no_coarse:
        base_overrides.update({
            "nx_eq": 65,
            "ny_eq": 129,
            "blanket_enabled": False,
            "blanket_n_filaments": 0,
            # softer + more continuation steps
            "f_list_equilibrium": (0.10, 0.25, 0.45, 0.65, 0.85, 1.00),
            "target_rel_tol_ramp": 1e-4,
            "target_rel_tol": 5e-5,
        })

    # Targets (final physics values from cfg); we ramp them by factor f
    Ip_target = float(getattr(cfg, "Ip", 0.0))
    paxis_target = float(getattr(cfg, "paxis", 0.0))
    fvac_target = float(getattr(cfg, "fvac", 0.0))

    tag = _ts_tag()
    log_jsonl = _results_dir() / "scan_feasibility_results.jsonl"
    best_json = _results_dir() / "scan_feasibility_best.json"

    print("[INFO] scan_star_feasibility started")
    print(f"[INFO] dxf={dxf_path or 'cfg/auto'}")
    print(f"[INFO] workers={workers} iters={iters} pop={pop} timeout={timeout_s:.1f}s seed={seed}")
    print(f"[INFO] scan_keys={scan_keys}")
    print(f"[INFO] require_sep={require_sep} enforce_inner={enforce_inner} inner_tol={inner_tol}")
    print(f"[INFO] physics targets: Ip={Ip_target:.4g} A  paxis={paxis_target:.4g} Pa  fvac={fvac_target:.4g}")
    print(f"[INFO] ramp factors: {ramp}")
    if not no_coarse:
        print("[INFO] coarse overrides enabled:", base_overrides)
    print("[INFO] prior [MA]:", prior_MA)
    print("[INFO] bounds [MA]:")
    for k in scan_keys:
        lo, hi = bounds[k]
        print(f"  {k:>3s}: {lo:+8.3f} .. {hi:+8.3f}")

    best_global: Optional[Dict[str, Any]] = None
    best_score = float("inf")
    best_curr_MA = dict(prior_MA)

    def eval_one(curr_MA: Dict[str, float], cfg_overrides: Dict[str, Any]) -> Dict[str, Any]:
        curr_MA2 = _clip_hard(curr_MA, hard_MA)
        curr_A = _currents_MA_to_A(curr_MA2)
        res = eval_case(
            curr_A,
            dxf_path=dxf_path,
            timeout_s=timeout_s,
            cfg_overrides=cfg_overrides,
            require_sep=require_sep,
            enforce_inner=enforce_inner,
            inner_tol=inner_tol,
            redirect_solver_noise=(not show_solver),
        )
        res["currents_MA"] = curr_MA2
        return res

    # Outer loop: physics ramp
    for f in ramp:
        f = float(f)
        cfg_overrides = dict(base_overrides)

        cfg_overrides["Ip"] = float(f) * float(Ip_target)
        cfg_overrides["paxis"] = float(f) * float(paxis_target)
        cfg_overrides["fvac"] = float(fvac_target)

        print("\n" + "-" * 80)
        print(f"[RAMP] f={f:.3f} -> Ip={cfg_overrides['Ip']:.4g}  paxis={cfg_overrides['paxis']:.4g}  fvac={cfg_overrides['fvac']:.4g}")

        stage_best: Optional[Dict[str, Any]] = None
        stage_best_score = float("inf")
        stage_best_curr = dict(best_curr_MA)

        t0 = time.time()
        for it in range(1, int(iters) + 1):
            batch: List[Dict[str, float]] = []
            n_global = max(2, int(pop // 5))  # ~20% global
            n_local = int(pop) - n_global

            # inject current best
            batch.append(dict(stage_best_curr))

            for _ in range(n_global - 1):
                batch.append(_sample_uniform(stage_best_curr, bounds, scan_keys))

            for _ in range(n_local):
                batch.append(_sample_gauss(stage_best_curr, sigma, bounds, scan_keys))

            batch = batch[: int(pop)]

            ok_solve = 0
            ok_true_sep = 0

            batch_best = None
            batch_best_score = float("inf")

            with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
                futs = {ex.submit(eval_one, cand, cfg_overrides): cand for cand in batch}
                for fut in as_completed(futs):
                    cand = futs[fut]
                    try:
                        res = fut.result()
                    except Exception as e:
                        res = {"ok": False, "ok_solve": False, "error": f"future_failed:{repr(e)}", "currents_MA": cand}

                    if bool(res.get("ok_solve", False)):
                        ok_solve += 1
                        if bool(res.get("has_true_sep", False)):
                            ok_true_sep += 1

                    score = float(res.get("score", 1e18)) if bool(res.get("ok_solve", False)) else 1e18

                    _append_jsonl(log_jsonl, {
                        "event": "eval",
                        "ramp_f": f,
                        "iter": it,
                        "score": score,
                        "res": res,
                        "cfg_overrides": cfg_overrides,
                    })

                    if score < batch_best_score:
                        batch_best_score = score
                        batch_best = res

            # accept stage best
            if batch_best is not None and batch_best_score < stage_best_score:
                stage_best_score = float(batch_best_score)
                stage_best = batch_best
                stage_best_curr = dict(stage_best.get("currents_MA", stage_best_curr))

            # accept global best
            if stage_best is not None and stage_best_score < best_score:
                best_score = float(stage_best_score)
                best_global = stage_best
                best_curr_MA = dict(best_global.get("currents_MA", best_curr_MA))

            # anneal sigma mildly
            for k in sigma:
                sigma[k] = max(0.03, sigma[k] * 0.995)

            elapsed = time.time() - t0
            rate = (it * pop) / max(1e-9, elapsed)

            print(
                f"ramp f={f:>5.2f} | iter {it:>4d}/{iters:<4d} "
                f"| solve_ok={ok_solve:>3d} true_sep={ok_true_sep:>3d} "
                f"| best_stage={stage_best_score:>10.3g} best_global={best_score:>10.3g} "
                f"| {rate:>4.2f}/s"
            )

            # write best snapshot frequently
            if best_global is not None:
                snap = {
                    "best_score": float(best_score),
                    "best_currents_MA": dict(best_curr_MA),
                    "best_currents_A": _currents_MA_to_A(best_curr_MA),
                    "best_res": best_global,
                }
                _save_json(best_json, snap)

            # early stop if we have a very feasible solution at this ramp
            # (score=0 means: solved + (if require_sep) true separatrix + (if enforce_inner) within tolerance)
            if stage_best is not None and float(stage_best_score) <= 1e-9:
                print(f"[RAMP] early stop at f={f:.2f}: found score≈0 feasible case.")
                break

    print("\n[DONE] scan_star_feasibility finished.")
    print(f"[DONE] best_score = {best_score:.6g}")
    print(f"[DONE] best file: {best_json}")
    if best_global is not None:
        print("[DONE] best currents [MA]:", best_curr_MA)


# ----------------------------
# CLI
# ----------------------------
def _parse_float_list(s: str) -> List[float]:
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out


def main():
    mp.freeze_support()

    ap = argparse.ArgumentParser()
    ap.add_argument("--dxf", type=str, default=None, help="DXF path (defaults to cfg.dxf_path inside star_equilibrium)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--iters", type=int, default=50, help="iterations per ramp factor")
    ap.add_argument("--pop", type=int, default=24, help="population per iteration")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--ramp", type=str, default="0.2,0.35,0.5,0.7,0.85,1.0",
                    help="comma-separated physics ramp factors f applied to (Ip,paxis,fvac)")

    ap.add_argument("--scan-keys", type=str, default="CS,PF1,PF2,PF3",
                    help="comma-separated coil family keys to scan (others fixed at prior)")
    ap.add_argument("--widen", type=float, default=0.8, help="bounds widen factor around prior (MA)")

    ap.add_argument("--require-sep", action="store_true", help="Prefer solutions with TRUE separatrix (shape.reason=='ok')")
    ap.add_argument("--enforce-inner", action="store_true", help="Add soft penalty if LCFS is outside WALL_INNER")
    ap.add_argument("--inner-tol", type=float, default=0.10, help="allowed fraction outside inner wall before penalty")

    ap.add_argument("--no-coarse", action="store_true", help="Do NOT apply coarse overrides (mesh/tols/blanket)")
    ap.add_argument("--show-solver", action="store_true", help="Do NOT redirect solver noise (more verbose)")

    args = ap.parse_args()

    ramp = _parse_float_list(args.ramp)
    scan_keys = [s.strip().upper() for s in str(args.scan_keys).split(",") if s.strip()]

    scan_feasibility(
        dxf_path=args.dxf,
        workers=int(args.workers),
        timeout_s=float(args.timeout),
        iters=int(args.iters),
        pop=int(args.pop),
        seed=int(args.seed),
        ramp=ramp,
        scan_keys=scan_keys,
        require_sep=bool(args.require_sep),
        enforce_inner=bool(args.enforce_inner),
        inner_tol=float(args.inner_tol),
        no_coarse=bool(args.no_coarse),
        show_solver=bool(args.show_solver),
        widen=float(args.widen),
    )


if __name__ == "__main__":
    main()
