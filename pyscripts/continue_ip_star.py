#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
continue_ip_star.py

Continuation in plasma current Ip for STAR-like tokamak equilibrium.

Philosophy:
- Start from a LOW-physics but stable diverted equilibrium seed.
- Do NOT assume seed currents are correct.
- At each Ip step:
    1) re-center / re-optimize a subset of PF family currents
    2) prioritize:
         - true separatrix
         - axis near target center
         - weak global shape consistency
    3) use the best stable equilibrium as seed for the next Ip

This script is intentionally NOT a full CAD-final optimizer.
It is a branch-tracking tool.

Recommended first run:
  py .\continue_ip_star.py --workers 4 --timeout 240 --iters 30 --pop 20 ^
      --ip-list "0.8,1.0,1.2,1.5,1.8,2.2,2.6,3.0,3.5,4.0" ^
      --free-keys "PF2,PF3,PF4,PF5,PF6" --fixed-keys "CS,PF1"

Outputs:
- ./results/continue_ip_results.jsonl
- ./results/continue_ip_branch.json
- ./results/continue_ip_last_stable.json
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


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
FAMILIES_ALL = ("CS", "PF1", "PF2", "PF3", "PF4", "PF5", "PF6")
DEFAULT_FREE_KEYS = ("PF2", "PF3", "PF4", "PF5", "PF6")
DEFAULT_FIXED_KEYS = ("CS", "PF1")


# ---------------------------------------------------------------------
# JSON-safe helpers
# ---------------------------------------------------------------------
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
    return obj


def _json_dumps_safe(obj: Any, **kwargs) -> str:
    return json.dumps(_to_builtin(obj), default=str, **kwargs)


# ---------------------------------------------------------------------
# Paths / IO
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Small utils
# ---------------------------------------------------------------------
def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _parse_float_list_csv(s: str) -> List[float]:
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out


def _parse_keys_csv(s: str) -> List[str]:
    out = []
    for tok in str(s).split(","):
        tok = tok.strip().upper()
        if tok and tok in FAMILIES_ALL:
            out.append(tok)
    return out


def _currents_A_to_MA(curr_A: Dict[str, float]) -> Dict[str, float]:
    return {k: float(curr_A.get(k, 0.0) / 1e6) for k in FAMILIES_ALL}


def _currents_MA_to_A(curr_MA: Dict[str, float]) -> Dict[str, float]:
    return {k: float(curr_MA.get(k, 0.0) * 1e6) for k in FAMILIES_ALL}


def _fmt_currents_MA(curr_A: Dict[str, float], keys: Optional[List[str]] = None) -> str:
    cm = _currents_A_to_MA(curr_A)
    use = keys if keys else list(FAMILIES_ALL)
    return ", ".join([f"{k}={cm[k]:+.3f} MA" for k in use])


# ---------------------------------------------------------------------
# Limits / seed
# ---------------------------------------------------------------------
def _get_seed_currents_from_cfg(cfg: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in FAMILIES_ALL:
        try:
            out[k] = float(getattr(cfg, f"{k}_current"))
        except Exception:
            out[k] = 0.0
    return out


def _default_limits_MA() -> Dict[str, float]:
    # Conservative but not too restrictive
    return {
        "CS": 80.0,
        "PF1": 20.0,
        "PF2": 20.0,
        "PF3": 12.0,
        "PF4": 12.0,
        "PF5": 15.0,
        "PF6": 25.0,
    }


def _get_limits_MA_from_cfg(cfg: Any, mode: str = "recommended") -> Dict[str, float]:
    mode = str(mode).strip().lower()

    # Preferred for exploration
    if mode == "recommended":
        d = getattr(cfg, "Imax_recommended_MA", None)
        if isinstance(d, dict) and d:
            out = {}
            for k in FAMILIES_ALL:
                if k in d:
                    out[k] = float(d[k])
            if out:
                for k in FAMILIES_ALL:
                    out.setdefault(k, _default_limits_MA()[k])
                return out

    # More conservative
    if mode == "operating":
        d = getattr(cfg, "OPERATING_FAMILY_CURRENT_LIMIT_A", None)
        if isinstance(d, dict) and d:
            out = {}
            for k in FAMILIES_ALL:
                if k in d:
                    out[k] = float(d[k]) / 1e6
            if out:
                for k in FAMILIES_ALL:
                    out.setdefault(k, _default_limits_MA()[k])
                return out

    return _default_limits_MA()


def _clip_currents_A(curr_A: Dict[str, float], limits_MA: Dict[str, float], free_keys: List[str]) -> Dict[str, float]:
    out = dict(curr_A)
    for k in free_keys:
        limA = float(abs(limits_MA.get(k, 1e9))) * 1e6
        out[k] = float(np.clip(out.get(k, 0.0), -limA, +limA))
    return out


# ---------------------------------------------------------------------
# Sampling / local exploration
# ---------------------------------------------------------------------
def _make_bounds_MA(
    center_A: Dict[str, float],
    limits_MA: Dict[str, float],
    free_keys: List[str],
    widen: float,
) -> Dict[str, Tuple[float, float]]:
    """
    Bounds centered on current seed, but wide enough to move meaningfully.
    """
    center_MA = _currents_A_to_MA(center_A)

    base_span = {
        "CS": 3.0,
        "PF1": 2.0,
        "PF2": 2.2,
        "PF3": 1.2,
        "PF4": 2.2,
        "PF5": 2.2,
        "PF6": 3.0,
    }

    bounds: Dict[str, Tuple[float, float]] = {}
    for k in free_keys:
        c = float(center_MA.get(k, 0.0))
        span = float(widen) * max(abs(c), 0.5) + float(base_span.get(k, 1.5))
        lim = float(abs(limits_MA.get(k, 1e9)))
        lo = max(-lim, c - span)
        hi = min(+lim, c + span)
        bounds[k] = (float(lo), float(hi))
    return bounds


def _sigma_from_bounds_MA(bounds_MA: Dict[str, Tuple[float, float]], frac: float) -> Dict[str, float]:
    out = {}
    for k, (lo, hi) in bounds_MA.items():
        out[k] = float(frac) * (float(hi) - float(lo))
    return out


def _sample_uniform_A(
    seed_A: Dict[str, float],
    bounds_MA: Dict[str, Tuple[float, float]],
    free_keys: List[str],
) -> Dict[str, float]:
    out = dict(seed_A)
    for k in free_keys:
        lo, hi = bounds_MA[k]
        out[k] = random.uniform(lo, hi) * 1e6
    return out


def _sample_gauss_A(
    center_A: Dict[str, float],
    sigma_MA: Dict[str, float],
    bounds_MA: Dict[str, Tuple[float, float]],
    free_keys: List[str],
) -> Dict[str, float]:
    out = dict(center_A)
    center_MA = _currents_A_to_MA(center_A)
    for k in free_keys:
        s = float(sigma_MA.get(k, 0.5))
        lo, hi = bounds_MA[k]
        val = random.gauss(float(center_MA.get(k, 0.0)), s)
        val = float(np.clip(val, lo, hi))
        out[k] = val * 1e6
    return out


# ---------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------
def _pick_lower_upper_points(points: List[Tuple[float, float]]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    if not points:
        return None, None
    lower = min(points, key=lambda p: p[1])
    upper = max(points, key=lambda p: p[1])
    return lower, upper


def _compute_branch_score(
    *,
    shape: Dict[str, Any],
    diag: Dict[str, Any],
    currents_A: Dict[str, float],
    ref_currents_A: Dict[str, float],
    free_keys: List[str],
    limits_MA: Dict[str, float],
    target_R0: float,
    target_Z0: float,
    target_A: float,
    target_kappa: float,
    target_delta: float,
    prefer_double: bool,
) -> Tuple[float, Dict[str, Any]]:
    """
    Stability-first score:
    - enormous penalty if no true separatrix
    - moderate penalty if axis is displaced
    - weak penalty for rough shape mismatch
    - mild regularization to previous stable currents
    - optional preference for 2 X-points (but not fatal if absent)
    """
    reason = str(shape.get("reason", "")).strip().lower()
    has_true_sep = bool(shape.get("ok_sep", False)) and (reason == "ok")

    xpv = shape.get("xpoints_valid", None)
    if isinstance(xpv, list) and xpv:
        xps = []
        for d in xpv:
            try:
                xps.append((float(d["R"]), float(d["Z"])))
            except Exception:
                pass
    else:
        xp0 = shape.get("xpoints", []) or []
        xps = []
        for d in xp0:
            if isinstance(d, dict) and ("R" in d) and ("Z" in d):
                try:
                    xps.append((float(d["R"]), float(d["Z"])))
                except Exception:
                    pass

    n_xp = int(len(xps))
    lower_xp, upper_xp = _pick_lower_upper_points(xps)

    # Diagnostics
    ok_diag = bool(isinstance(diag, dict) and diag.get("ok", False))
    Rax = _safe_float(diag.get("R_ax", shape.get("R_ax", np.nan)))
    Zax = _safe_float(diag.get("Z_ax", shape.get("Z_ax", np.nan)))
    R0  = _safe_float(diag.get("R0", np.nan))
    A   = _safe_float(diag.get("A", np.nan))
    kap = _safe_float(diag.get("kappa", np.nan))
    du  = _safe_float(diag.get("delta_u", np.nan))
    dl  = _safe_float(diag.get("delta_l", np.nan))
    dbar = 0.5 * (du + dl) if np.isfinite(du + dl) else np.nan

    # Hard penalties
    score = 0.0
    if not ok_diag:
        score += 1e9
    if not has_true_sep:
        score += 1e6

    if prefer_double:
        if n_xp < 2:
            score += 2e5

    # Axis centering (important)
    axis_term = 0.0
    if np.isfinite(Rax):
        axis_term += ((Rax - target_R0) / 0.15) ** 2
    else:
        axis_term += 25.0

    if np.isfinite(Zax):
        axis_term += ((Zax - target_Z0) / 0.12) ** 2
    else:
        axis_term += 25.0

    score += 50.0 * axis_term

    # Weak shape consistency (not final CAD fit)
    shape_term = 0.0
    if np.isfinite(A):
        shape_term += ((A - target_A) / 0.50) ** 2
    else:
        shape_term += 4.0

    if np.isfinite(kap):
        shape_term += ((kap - target_kappa) / 0.50) ** 2
    else:
        shape_term += 4.0

    if np.isfinite(dbar):
        shape_term += ((dbar - target_delta) / 0.35) ** 2
    else:
        shape_term += 4.0

    if np.isfinite(R0):
        shape_term += ((R0 - target_R0) / 0.35) ** 2
    else:
        shape_term += 4.0

    score += 10.0 * shape_term

    # Optional preference for upper/lower symmetry when two X-points exist
    sym_term = 0.0
    if prefer_double and lower_xp is not None and upper_xp is not None:
        sym_term += ((upper_xp[0] - lower_xp[0]) / 0.18) ** 2
        sym_term += ((upper_xp[1] + lower_xp[1]) / 0.18) ** 2
        score += 5.0 * sym_term

    # Mild regularization to previous stable currents
    reg_term = 0.0
    curr_MA = _currents_A_to_MA(currents_A)
    ref_MA = _currents_A_to_MA(ref_currents_A)
    for k in free_keys:
        lim = float(abs(limits_MA.get(k, 10.0)))
        sig = max(1.2, 0.25 * lim)
        reg_term += ((curr_MA[k] - ref_MA[k]) / sig) ** 2

    if len(free_keys) > 0:
        reg_term /= float(len(free_keys))
    score += 2.0 * reg_term

    info = {
        "has_true_sep": bool(has_true_sep),
        "n_xpoints": n_xp,
        "axis_term": float(axis_term),
        "shape_term": float(shape_term),
        "sym_term": float(sym_term),
        "reg_term": float(reg_term),
        "Rax": Rax,
        "Zax": Zax,
        "R0": R0,
        "A": A,
        "kappa": kap,
        "delta_u": du,
        "delta_l": dl,
        "dbar": dbar,
        "lower_xpoint": lower_xp,
        "upper_xpoint": upper_xp,
        "shape_reason": reason,
    }
    return float(score), info


# ---------------------------------------------------------------------
# Worker in subprocess
# ---------------------------------------------------------------------
def _worker_eval(
    currents_A: Dict[str, float],
    dxf_path: Optional[str],
    cfg_overrides: Dict[str, Any],
    Ip_A: float,
    paxis_Pa: float,
    fvac: float,
    alpha_m: float,
    alpha_n: float,
    ref_currents_A: Dict[str, float],
    free_keys: List[str],
    limits_MA: Dict[str, float],
    prefer_double: bool,
    conn,
) -> None:
    score = 1e18
    try:
        import config_star_bean as cfg
        import star_equilibrium as se

        # Apply overrides
        for k, v in (cfg_overrides or {}).items():
            try:
                setattr(cfg, k, v)
            except Exception:
                pass

        # Plasma parameters for THIS Ip step
        cfg.Ip = float(Ip_A)
        cfg.paxis = float(paxis_Pa)
        cfg.fvac = float(fvac)
        cfg.alpha_m = float(alpha_m)
        cfg.alpha_n = float(alpha_n)
        cfg.vacuum_only = False

        # Apply currents
        for k, v in currents_A.items():
            try:
                setattr(cfg, f"{k}_current", float(v))
            except Exception:
                pass

        eq, tokamak, geom, shape = se.build_equilibrium(
            verbose=False,
            redirect_solver_noise=True,
            dxf_path=dxf_path,
        )

        diag = shape.get("plasma_diag", None) or {}
        if (not diag) or (not diag.get("ok", False)):
            try:
                diag = se.plasma_diagnostics(eq, geom, shape)
            except Exception:
                diag = {"ok": False}

        score, score_info = _compute_branch_score(
            shape=shape,
            diag=diag,
            currents_A=currents_A,
            ref_currents_A=ref_currents_A,
            free_keys=free_keys,
            limits_MA=limits_MA,
            target_R0=float(getattr(cfg, "R0_geom", 4.0)),
            target_Z0=0.0,
            target_A=float(getattr(cfg, "A_geom", 2.0)),
            target_kappa=float(getattr(cfg, "kappa_geom", 2.23)),
            target_delta=float(getattr(cfg, "delta_geom", 0.62)),
            prefer_double=bool(prefer_double),
        )

        conn.send({
            "ok": True,
            "ok_solve": True,
            "score": float(score),
            "score_info": score_info,
            "currents_A": dict(currents_A),
            "currents_MA": _currents_A_to_MA(currents_A),
            "Ip_A": float(Ip_A),
            "paxis_Pa": float(paxis_Pa),
            "diag": diag,
            "shape": {
                "ok_sep": bool(shape.get("ok_sep", False)),
                "reason": shape.get("reason", None),
                "xpoints": shape.get("xpoints", []),
                "xpoints_valid": shape.get("xpoints_valid", []),
                "psi_sep": shape.get("psi_sep", None),
                "psi_eval": shape.get("psi_eval", None),
            },
        })
        conn.close()

    except Exception as e:
        try:
            conn.send({
                "ok": False,
                "ok_solve": False,
                "score": float(score),
                "error": repr(e),
                "currents_A": dict(currents_A),
                "Ip_A": float(Ip_A),
            })
            conn.close()
        except Exception:
            pass


def eval_case(
    currents_A: Dict[str, float],
    *,
    dxf_path: Optional[str],
    cfg_overrides: Dict[str, Any],
    timeout_s: float,
    Ip_A: float,
    paxis_Pa: float,
    fvac: float,
    alpha_m: float,
    alpha_n: float,
    ref_currents_A: Dict[str, float],
    free_keys: List[str],
    limits_MA: Dict[str, float],
    prefer_double: bool,
) -> Dict[str, Any]:
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)

    p = ctx.Process(
        target=_worker_eval,
        args=(
            dict(currents_A),
            dxf_path,
            dict(cfg_overrides or {}),
            float(Ip_A),
            float(paxis_Pa),
            float(fvac),
            float(alpha_m),
            float(alpha_n),
            dict(ref_currents_A),
            list(free_keys),
            dict(limits_MA),
            bool(prefer_double),
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
            "score": 1e18,
            "error": f"timeout>{timeout_s:.1f}s",
            "elapsed_s": float(elapsed),
            "currents_A": dict(currents_A),
            "Ip_A": float(Ip_A),
        }

    if recv_conn.poll(0.05):
        try:
            res = recv_conn.recv()
        except Exception as e:
            res = {
                "ok": False,
                "ok_solve": False,
                "score": 1e18,
                "error": f"recv_failed:{repr(e)}",
                "elapsed_s": float(elapsed),
                "currents_A": dict(currents_A),
                "exitcode": p.exitcode,
                "Ip_A": float(Ip_A),
            }
    else:
        res = {
            "ok": False,
            "ok_solve": False,
            "score": 1e18,
            "error": "no_result_from_worker",
            "elapsed_s": float(elapsed),
            "currents_A": dict(currents_A),
            "exitcode": p.exitcode,
            "Ip_A": float(Ip_A),
        }

    try:
        recv_conn.close()
    except Exception:
        pass

    res["elapsed_s"] = float(elapsed)
    return res


# ---------------------------------------------------------------------
# Per-step optimization
# ---------------------------------------------------------------------
def optimize_ip_step(
    *,
    Ip_A: float,
    seed_currents_A: Dict[str, float],
    ref_currents_A: Dict[str, float],
    dxf_path: Optional[str],
    timeout_s: float,
    workers: int,
    iters: int,
    pop: int,
    paxis_Pa: float,
    fvac: float,
    alpha_m: float,
    alpha_n: float,
    free_keys: List[str],
    fixed_keys: List[str],
    limits_MA: Dict[str, float],
    prefer_double: bool,
    cfg_overrides: Dict[str, Any],
    widen: float,
) -> Dict[str, Any]:
    """
    Stochastic local exploration around a seed for one fixed Ip.
    """
    seed_currents_A = _clip_currents_A(seed_currents_A, limits_MA, free_keys)
    ref_currents_A = dict(ref_currents_A)

    bounds_MA = _make_bounds_MA(
        center_A=seed_currents_A,
        limits_MA=limits_MA,
        free_keys=free_keys,
        widen=float(widen),
    )
    sigma_MA = _sigma_from_bounds_MA(bounds_MA, frac=0.18)

    def eval_one(curr_A: Dict[str, float]) -> Dict[str, Any]:
        curr2 = _clip_currents_A(curr_A, limits_MA, free_keys)
        return eval_case(
            curr2,
            dxf_path=dxf_path,
            cfg_overrides=cfg_overrides,
            timeout_s=timeout_s,
            Ip_A=Ip_A,
            paxis_Pa=paxis_Pa,
            fvac=fvac,
            alpha_m=alpha_m,
            alpha_n=alpha_n,
            ref_currents_A=ref_currents_A,
            free_keys=free_keys,
            limits_MA=limits_MA,
            prefer_double=prefer_double,
        )

    best = eval_one(seed_currents_A)
    best_score = float(best.get("score", 1e18)) if bool(best.get("ok_solve", False)) else 1e18
    best_currents_A = dict(best.get("currents_A", seed_currents_A))

    stall_count = 0

    for it in range(1, int(iters) + 1):
        batch: List[Dict[str, float]] = []

        n_global = max(2, int(pop // 4))
        n_local = int(pop) - n_global

        # elite
        batch.append(dict(best_currents_A))

        # global exploration
        for _ in range(n_global - 1):
            batch.append(_sample_uniform_A(best_currents_A, bounds_MA, free_keys))

        # local exploration
        for _ in range(n_local):
            batch.append(_sample_gauss_A(best_currents_A, sigma_MA, bounds_MA, free_keys))

        batch_best = None
        batch_best_score = 1e18
        solve_ok = 0
        sep_ok = 0

        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
            futs = {ex.submit(eval_one, cand): cand for cand in batch}
            for fut in as_completed(futs):
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"ok": False, "ok_solve": False, "score": 1e18, "error": f"future_failed:{repr(e)}"}

                if bool(res.get("ok_solve", False)):
                    solve_ok += 1
                    si = res.get("score_info", {}) if isinstance(res.get("score_info", {}), dict) else {}
                    if bool(si.get("has_true_sep", False)):
                        sep_ok += 1

                sc = float(res.get("score", 1e18))
                if sc < batch_best_score:
                    batch_best_score = sc
                    batch_best = res

        improved = (batch_best is not None) and (batch_best_score + 1e-9 < best_score)

        if improved:
            best = batch_best
            best_score = float(batch_best_score)
            best_currents_A = dict(best.get("currents_A", best_currents_A))
            stall_count = 0
        else:
            stall_count += 1
            for k in sigma_MA:
                sigma_MA[k] = max(0.04, 0.85 * sigma_MA[k])

        # If completely stuck, widen once
        if stall_count == 4:
            bounds_MA = _make_bounds_MA(
                center_A=best_currents_A,
                limits_MA=limits_MA,
                free_keys=free_keys,
                widen=float(widen) * 1.25,
            )
            sigma_MA = _sigma_from_bounds_MA(bounds_MA, frac=0.20)

        si_best = best.get("score_info", {}) if isinstance(best.get("score_info", {}), dict) else {}
        print(
            f"Ip={Ip_A/1e6:>5.2f} MA | iter {it:>3d}/{iters:<3d} "
            f"| solve_ok={solve_ok:>2d} sep_ok={sep_ok:>2d} "
            f"| best_score={best_score:>10.4g} "
            f"| Rax={_safe_float(si_best.get('Rax', np.nan)):.3f} "
            f"| Zax={_safe_float(si_best.get('Zax', np.nan)):.3f}"
        )

        # Early stop if already clearly good
        if bool(si_best.get("has_true_sep", False)):
            axis_term = float(si_best.get("axis_term", 1e9))
            shape_term = float(si_best.get("shape_term", 1e9))
            if axis_term < 1.0 and shape_term < 6.0 and best_score < 200.0:
                break

    return {
        "Ip_A": float(Ip_A),
        "Ip_MA": float(Ip_A / 1e6),
        "best_score": float(best_score),
        "best_result": best,
        "best_currents_A": best_currents_A,
        "best_currents_MA": _currents_A_to_MA(best_currents_A),
    }


# ---------------------------------------------------------------------
# Main continuation
# ---------------------------------------------------------------------
def continue_ip(
    *,
    dxf_path: Optional[str],
    timeout_s: float,
    workers: int,
    iters: int,
    pop: int,
    seed: int,
    ip_list_MA: List[float],
    paxis_Pa: float,
    fvac: float,
    alpha_m: float,
    alpha_n: float,
    free_keys: List[str],
    fixed_keys: List[str],
    prefer_double: bool,
    widen: float,
    limit_mode: str,
    no_coarse: bool,
) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))

    import config_star_bean as cfg

    seed_currents_A = _get_seed_currents_from_cfg(cfg)
    limits_MA = _get_limits_MA_from_cfg(cfg, mode=limit_mode)

    free_keys = [k for k in free_keys if k in FAMILIES_ALL]
    fixed_keys = [k for k in fixed_keys if k in FAMILIES_ALL and k not in free_keys]

    if not free_keys:
        raise ValueError("free_keys vacío.")

    cfg_overrides: Dict[str, Any] = {}
    if not no_coarse:
        cfg_overrides.update({
            "nx_eq": 65,
            "ny_eq": 129,
            "blanket_enabled": False,
            "blanket_n_filaments": 0,
            "target_rel_tol_ramp": 1e-4,
            "target_rel_tol": 5e-5,
            "f_list_equilibrium": (0.10, 0.25, 0.45, 0.70, 1.00),
        })

    log_jsonl = _results_dir() / "continue_ip_results.jsonl"
    branch_json = _results_dir() / "continue_ip_branch.json"
    last_stable_json = _results_dir() / "continue_ip_last_stable.json"

    print("[INFO] continue_ip_star started")
    print(f"[INFO] dxf={dxf_path or 'cfg/auto'}")
    print(f"[INFO] workers={workers} timeout={timeout_s:.1f}s iters={iters} pop={pop} seed={seed}")
    print(f"[INFO] free_keys={free_keys}")
    print(f"[INFO] fixed_keys={fixed_keys}")
    print(f"[INFO] prefer_double={prefer_double}")
    print(f"[INFO] paxis={paxis_Pa:.4g} Pa  fvac={fvac:.4g}  alpha_m={alpha_m:.3g}  alpha_n={alpha_n:.3g}")
    print(f"[INFO] Ip list [MA] = {ip_list_MA}")
    print(f"[INFO] limit_mode={limit_mode}  limits_MA={limits_MA}")
    print(f"[INFO] initial seed: {_fmt_currents_MA(seed_currents_A, list(FAMILIES_ALL))}")

    branch: List[Dict[str, Any]] = []
    last_stable: Optional[Dict[str, Any]] = None

    current_seed_A = dict(seed_currents_A)
    current_ref_A = dict(seed_currents_A)

    for ip_MA in ip_list_MA:
        Ip_A = float(ip_MA) * 1e6
        print("\n" + "-" * 90)
        print(f"[STEP] Target Ip = {ip_MA:.3f} MA")
        print(f"[STEP] Seed currents: {_fmt_currents_MA(current_seed_A, free_keys + fixed_keys)}")

        step_out = optimize_ip_step(
            Ip_A=Ip_A,
            seed_currents_A=current_seed_A,
            ref_currents_A=current_ref_A,
            dxf_path=dxf_path,
            timeout_s=timeout_s,
            workers=workers,
            iters=iters,
            pop=pop,
            paxis_Pa=paxis_Pa,
            fvac=fvac,
            alpha_m=alpha_m,
            alpha_n=alpha_n,
            free_keys=free_keys,
            fixed_keys=fixed_keys,
            limits_MA=limits_MA,
            prefer_double=prefer_double,
            cfg_overrides=cfg_overrides,
            widen=widen,
        )

        best = step_out["best_result"]
        score_info = best.get("score_info", {}) if isinstance(best.get("score_info", {}), dict) else {}
        has_true_sep = bool(score_info.get("has_true_sep", False))

        step_rec = {
            "Ip_A": float(Ip_A),
            "Ip_MA": float(ip_MA),
            "best_score": float(step_out["best_score"]),
            "best_currents_A": step_out["best_currents_A"],
            "best_currents_MA": step_out["best_currents_MA"],
            "score_info": score_info,
            "diag": best.get("diag", {}),
            "shape": best.get("shape", {}),
            "has_true_sep": bool(has_true_sep),
        }
        branch.append(step_rec)

        _append_jsonl(log_jsonl, {
            "event": "step_done",
            "step": step_rec,
        })
        _save_json(branch_json, {"branch": branch})

        if has_true_sep:
            last_stable = step_rec
            _save_json(last_stable_json, last_stable)

            # next seed = current best
            current_seed_A = dict(step_out["best_currents_A"])
            current_ref_A = dict(step_out["best_currents_A"])

            print(f"[OK] Stable step at Ip={ip_MA:.3f} MA")
            print(f"[OK] best_score={step_out['best_score']:.6g}")
            print(f"[OK] currents: {_fmt_currents_MA(current_seed_A, free_keys + fixed_keys)}")
        else:
            print(f"[STOP] Lost true separatrix at Ip={ip_MA:.3f} MA")
            print(f"[STOP] Last stable remains: {None if last_stable is None else last_stable['Ip_MA']}")
            break

    print("\n[DONE] continue_ip_star finished.")
    print(f"[DONE] Branch file: {branch_json}")
    if last_stable is not None:
        print(f"[DONE] Last stable file: {last_stable_json}")
        print(f"[DONE] Last stable Ip = {last_stable['Ip_MA']:.3f} MA")
        print(f"[DONE] Last stable currents [MA] = {last_stable['best_currents_MA']}")
    else:
        print("[DONE] No stable step found.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    mp.freeze_support()

    import config_star_bean as cfg

    ap = argparse.ArgumentParser()

    ap.add_argument("--dxf", type=str, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--ip-list",
        type=str,
        default="0.8,1.0,1.2,1.5,1.8,2.2,2.6,3.0,3.5,4.0",
        help="Ip list in MA",
    )

    ap.add_argument("--paxis", type=float, default=float(getattr(cfg, "paxis", 2.0e3)))
    ap.add_argument("--fvac", type=float, default=float(getattr(cfg, "fvac", 20.8)))
    ap.add_argument("--alpha-m", type=float, default=float(getattr(cfg, "alpha_m", 1.5)))
    ap.add_argument("--alpha-n", type=float, default=float(getattr(cfg, "alpha_n", 1.1)))

    ap.add_argument(
        "--free-keys",
        type=str,
        default=",".join(DEFAULT_FREE_KEYS),
        help="Families allowed to move",
    )
    ap.add_argument(
        "--fixed-keys",
        type=str,
        default=",".join(DEFAULT_FIXED_KEYS),
        help="Families kept fixed",
    )

    ap.add_argument("--prefer-double", action="store_true", help="Prefer 2 X-points if possible")
    ap.add_argument("--widen", type=float, default=1.10, help="Exploration widen factor around current seed")
    ap.add_argument(
        "--limit-mode",
        type=str,
        default="recommended",
        choices=["recommended", "operating", "default"],
    )
    ap.add_argument("--no-coarse", action="store_true", help="Use cfg as-is, without coarse overrides")

    args = ap.parse_args()

    ip_list_MA = _parse_float_list_csv(args.ip_list)
    free_keys = _parse_keys_csv(args.free_keys)
    fixed_keys = _parse_keys_csv(args.fixed_keys)

    continue_ip(
        dxf_path=args.dxf,
        timeout_s=float(args.timeout),
        workers=int(args.workers),
        iters=int(args.iters),
        pop=int(args.pop),
        seed=int(args.seed),
        ip_list_MA=ip_list_MA,
        paxis_Pa=float(args.paxis),
        fvac=float(args.fvac),
        alpha_m=float(args.alpha_m),
        alpha_n=float(args.alpha_n),
        free_keys=free_keys,
        fixed_keys=fixed_keys,
        prefer_double=bool(args.prefer_double),
        widen=float(args.widen),
        limit_mode=str(args.limit_mode),
        no_coarse=bool(args.no_coarse),
    )


if __name__ == "__main__":
    main()
