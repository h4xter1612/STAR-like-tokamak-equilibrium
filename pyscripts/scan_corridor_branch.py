#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scan_corridor_branch.py

Coarse branch search for STAR-like DN equilibria.

This is NOT a local optimizer.

Purpose
-------
The local optimizers keep falling into the same topology where the divertor leg
exits through the PF5-PF6 corridor. This script performs a coarse, pattern-based
branch scan with hard gates, aiming to find any alternative branch where:

    - ok_sep=True
    - A, kappa, chamfer remain acceptable
    - PF6 is not allowed to dominate
    - the separatrix leg moves closer to LEG_UPPER_WIN / LEG_LOWER_WIN

It reuses:
    - fit_simplified_dn_corridor.eval_case_corridor(...)
    - fit_simplified_dn_toposafe helpers

Outputs
-------
results/scan_corridor_branch_results.jsonl
results/scan_corridor_branch_best.json
results/scan_corridor_branch_accepted.json

Recommended first run
---------------------
py .\\scan_corridor_branch.py ^
  --target .\\results\\targets\\star_simplified_dn_target.json ^
  --seed .\\results\\seeds\\branch_probe_low_pf6.json ^
  --dxf .\\cad\\star_baseline.dxf ^
  --n-cases 96 ^
  --workers 6 ^
  --timeout 180
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

import fit_simplified_dn_toposafe as base
import fit_simplified_dn_corridor as corridor


FAMILIES_ALL = base.FAMILIES_ALL


# ---------------------------------------------------------------------
# JSON helpers
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


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_json_dumps_safe(obj, indent=2))


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(_json_dumps_safe(obj) + "\n")


def _results_dir() -> Path:
    d = Path(__file__).resolve().parent / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------
# Current sampling
# ---------------------------------------------------------------------
def _currents_MA_to_A(curr_MA: Dict[str, float]) -> Dict[str, float]:
    return {k: float(curr_MA.get(k, 0.0)) * 1e6 for k in FAMILIES_ALL}


def _currents_A_to_MA(curr_A: Dict[str, float]) -> Dict[str, float]:
    return {k: float(curr_A.get(k, 0.0)) / 1e6 for k in FAMILIES_ALL}


def _clip_MA(x: Dict[str, float], bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    out = {}
    for k in FAMILIES_ALL:
        lo, hi = bounds[k]
        out[k] = float(np.clip(float(x.get(k, 0.0)), lo, hi))
    return out


def _sample_uniform(bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    return {k: random.uniform(bounds[k][0], bounds[k][1]) for k in FAMILIES_ALL}


def _sample_gaussian_template(
    template: Dict[str, float],
    sigmas: Dict[str, float],
    bounds: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    x = {}
    for k in FAMILIES_ALL:
        val = random.gauss(float(template.get(k, 0.0)), float(sigmas.get(k, 1.0)))
        lo, hi = bounds[k]
        x[k] = float(np.clip(val, lo, hi))
    return x


def _make_templates() -> List[Dict[str, float]]:
    """
    Templates in MA.

    These deliberately reduce PF6 relative to the previously found local branch
    and push PF4/PF5/PF2/PF3 harder to look for a different divertor-leg topology.
    """
    return [
        {
            "CS": 2.0,
            "PF1": 0.0,
            "PF2": -6.5,
            "PF3": 4.0,
            "PF4": 6.0,
            "PF5": -4.0,
            "PF6": 6.0,
        },
        {
            "CS": 2.2,
            "PF1": -0.5,
            "PF2": -7.5,
            "PF3": 4.8,
            "PF4": 6.8,
            "PF5": -5.0,
            "PF6": 5.0,
        },
        {
            "CS": 1.6,
            "PF1": 0.0,
            "PF2": -8.5,
            "PF3": 5.5,
            "PF4": 7.5,
            "PF5": -5.8,
            "PF6": 4.0,
        },
        {
            "CS": 2.8,
            "PF1": -1.0,
            "PF2": -6.0,
            "PF3": 5.0,
            "PF4": 8.0,
            "PF5": -4.5,
            "PF6": 3.0,
        },
        {
            "CS": 1.2,
            "PF1": 0.5,
            "PF2": -9.0,
            "PF3": 6.2,
            "PF4": 5.5,
            "PF5": -6.5,
            "PF6": 7.0,
        },
        {
            "CS": 3.2,
            "PF1": -1.5,
            "PF2": -5.5,
            "PF3": 3.5,
            "PF4": 7.0,
            "PF5": -3.5,
            "PF6": 8.0,
        },
    ]


def generate_candidates(
    *,
    n_cases: int,
    bounds: Dict[str, Tuple[float, float]],
    seed_currents_A: Dict[str, float],
) -> List[Dict[str, float]]:
    candidates_MA: List[Dict[str, float]] = []

    seed_MA = _currents_A_to_MA(seed_currents_A)
    seed_MA = _clip_MA(seed_MA, bounds)

    templates = _make_templates()

    sigmas = {
        "CS": 0.55,
        "PF1": 0.75,
        "PF2": 1.10,
        "PF3": 0.90,
        "PF4": 1.00,
        "PF5": 1.00,
        "PF6": 1.40,
    }

    # Always include seed and templates.
    candidates_MA.append(seed_MA)
    for t in templates:
        candidates_MA.append(_clip_MA(t, bounds))

    # Pattern-biased samples.
    while len(candidates_MA) < int(0.75 * n_cases):
        t = random.choice(templates)
        candidates_MA.append(_sample_gaussian_template(t, sigmas, bounds))

    # Fully uniform exploration over restricted bounds.
    while len(candidates_MA) < int(n_cases):
        candidates_MA.append(_sample_uniform(bounds))

    # Dedupe roughly by rounded currents.
    seen = set()
    out_MA = []
    for x in candidates_MA:
        x = _clip_MA(x, bounds)
        key = tuple(round(float(x[k]), 2) for k in FAMILIES_ALL)
        if key in seen:
            continue
        seen.add(key)
        out_MA.append(x)

    return [_currents_MA_to_A(x) for x in out_MA[: int(n_cases)]]


# ---------------------------------------------------------------------
# Gating / ranking
# ---------------------------------------------------------------------
def _sf(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def score_candidate_with_gates(
    res: Dict[str, Any],
    *,
    max_pf6_MA: float,
    min_A: float,
    min_kappa: float,
    min_delta: float,
    max_chamfer: float,
    max_leg_dist: float,
    max_abs_axis_shift_R: float,
    max_abs_axis_shift_Z: float,
) -> Tuple[float, Dict[str, Any]]:
    si = res.get("score_info", {})
    if not isinstance(si, dict):
        si = {}

    curr_MA = res.get("currents_MA", {})
    if not isinstance(curr_MA, dict):
        curr_MA = {}

    has_sep = bool(si.get("has_true_sep", False))
    shape_reason = str(si.get("shape_reason", "")).lower()

    A = _sf(si.get("A", np.nan))
    kappa = _sf(si.get("kappa", np.nan))
    delta = _sf(si.get("delta_bar", np.nan))
    chamfer = _sf(si.get("boundary_chamfer_m", np.nan))
    Rax = _sf(si.get("Rax", np.nan))
    Zax = _sf(si.get("Zax", np.nan))
    legU = _sf(si.get("leg_upper_distance_m", np.inf))
    legL = _sf(si.get("leg_lower_distance_m", np.inf))
    PF6 = abs(_sf(curr_MA.get("PF6", np.inf)))

    fail: List[str] = []

    if not has_sep or shape_reason != "ok":
        fail.append("no_true_sep")
    if not np.isfinite(A) or A < min_A:
        fail.append("A_below_min")
    if not np.isfinite(kappa) or kappa < min_kappa:
        fail.append("kappa_below_min")
    if not np.isfinite(delta) or delta < min_delta:
        fail.append("delta_below_min")
    if not np.isfinite(chamfer) or chamfer > max_chamfer:
        fail.append("chamfer_above_max")
    if not np.isfinite(PF6) or PF6 > max_pf6_MA:
        fail.append("PF6_above_max")
    if not np.isfinite(Rax) or abs(Rax - 4.0) > max_abs_axis_shift_R:
        fail.append("Raxis_outside")
    if not np.isfinite(Zax) or abs(Zax - 0.0) > max_abs_axis_shift_Z:
        fail.append("Zaxis_outside")

    # Corridor is not a hard reject by default, but it gets strong ranking pressure.
    if not np.isfinite(legU) or not np.isfinite(legL):
        fail.append("missing_leg_distance")
    elif max(legU, legL) > max_leg_dist:
        fail.append("leg_distance_above_soft_max")

    accepted = (
        ("no_true_sep" not in fail)
        and ("A_below_min" not in fail)
        and ("kappa_below_min" not in fail)
        and ("chamfer_above_max" not in fail)
        and ("PF6_above_max" not in fail)
        and ("Raxis_outside" not in fail)
        and ("Zaxis_outside" not in fail)
    )

    base_score = _sf(si.get("score_total", res.get("score", 1e18)), 1e18)

    # Custom branch rank:
    # - first avoid failures
    # - then prefer closer leg corridors
    # - then shape/chamfer
    # - then lower PF6
    gate_penalty = 0.0
    hard_fail_names = {
        "no_true_sep",
        "A_below_min",
        "kappa_below_min",
        "chamfer_above_max",
        "PF6_above_max",
        "Raxis_outside",
        "Zaxis_outside",
    }

    for f in fail:
        if f in hard_fail_names:
            gate_penalty += 1.0e7
        else:
            gate_penalty += 1.0e5

    leg_rank = 0.0
    if np.isfinite(legU) and np.isfinite(legL):
        leg_rank = 1500.0 * ((legU / 0.60) ** 2 + (legL / 0.60) ** 2)
    else:
        leg_rank = 1.0e6

    pf6_rank = 500.0 * (PF6 / max(max_pf6_MA, 1e-6)) ** 2 if np.isfinite(PF6) else 1.0e6

    # Shape target pressure, not too dominant.
    shape_rank = 0.0
    if np.isfinite(A):
        shape_rank += 300.0 * ((A - 1.85) / 0.35) ** 2
    else:
        shape_rank += 1.0e6

    if np.isfinite(kappa):
        shape_rank += 300.0 * ((kappa - 2.15) / 0.30) ** 2
    else:
        shape_rank += 1.0e6

    if np.isfinite(delta):
        shape_rank += 250.0 * ((delta - 0.42) / 0.25) ** 2
    else:
        shape_rank += 1.0e6

    chamfer_rank = 1500.0 * (chamfer / max(max_chamfer, 1e-6)) ** 2 if np.isfinite(chamfer) else 1.0e6

    rank_score = float(gate_penalty + base_score + leg_rank + pf6_rank + shape_rank + chamfer_rank)

    info = {
        "accepted": bool(accepted),
        "fail_reasons": fail,
        "rank_score": float(rank_score),
        "base_score": float(base_score),
        "A": A,
        "kappa": kappa,
        "delta_bar": delta,
        "chamfer": chamfer,
        "Rax": Rax,
        "Zax": Zax,
        "legU": legU,
        "legL": legL,
        "PF6_MA": PF6,
    }

    return float(rank_score), info


# ---------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------
def run_scan(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(int(args.seed_rng))
    np.random.seed(int(args.seed_rng))

    target_path = str(args.target)
    seed_path = str(args.seed)
    dxf_path = base._resolve_path_maybe(args.dxf)

    target = base._load_json(target_path)
    seed_currents_A, physics = base._load_seed_currents(seed_path)

    bounds = {
        "CS": (args.cs_min, args.cs_max),
        "PF1": (args.pf1_min, args.pf1_max),
        "PF2": (args.pf2_min, args.pf2_max),
        "PF3": (args.pf3_min, args.pf3_max),
        "PF4": (args.pf4_min, args.pf4_max),
        "PF5": (args.pf5_min, args.pf5_max),
        "PF6": (args.pf6_min, args.pf6_max),
    }

    candidates = generate_candidates(
        n_cases=int(args.n_cases),
        bounds=bounds,
        seed_currents_A=seed_currents_A,
    )

    cfg_overrides: Dict[str, Any] = {}
    if not args.no_coarse:
        cfg_overrides.update({
            "nx_eq": 65,
            "ny_eq": 129,
            "blanket_enabled": False,
            "blanket_n_filaments": 0,
            "target_rel_tol_ramp": 1.0e-4,
            "target_rel_tol": 5.0e-5,
            "f_list_equilibrium": (0.10, 0.25, 0.45, 0.70, 1.00),
        })

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = _results_dir()

    log_jsonl = out_dir / "scan_corridor_branch_results.jsonl"
    best_json = out_dir / "scan_corridor_branch_best.json"
    accepted_json = out_dir / "scan_corridor_branch_accepted.json"
    run_json = out_dir / f"scan_corridor_branch_run_{run_tag}.json"

    print("[INFO] scan_corridor_branch started")
    print(f"[INFO] target={Path(target_path).resolve()}")
    print(f"[INFO] seed={Path(seed_path).resolve()}")
    print(f"[INFO] dxf={dxf_path}")
    print(f"[INFO] n_candidates={len(candidates)}")
    print(f"[INFO] workers={args.workers} timeout={args.timeout}")
    print(f"[INFO] bounds [MA]:")
    for k in FAMILIES_ALL:
        print(f"  {k:>3s}: [{bounds[k][0]:+.3f}, {bounds[k][1]:+.3f}]")

    best: Optional[Dict[str, Any]] = None
    best_rank = float("inf")
    best_info: Dict[str, Any] = {}
    accepted: List[Dict[str, Any]] = []

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = {}

        for i, curr_A in enumerate(candidates):
            fut = ex.submit(
                corridor.eval_case_corridor,
                curr_A,
                target=target,
                dxf_path=dxf_path,
                cfg_overrides=cfg_overrides,
                physics=physics,
                ref_currents_A=seed_currents_A,
                seed_currents_A=seed_currents_A,
                free_keys=list(FAMILIES_ALL),
                stage="release-cs",
                timeout_s=float(args.timeout),
                redirect_solver_noise=True,
                corridor_weight=float(args.corridor_weight),
                corridor_sigma_m=float(args.corridor_sigma),
                corridor_roi_pad_m=float(args.corridor_roi_pad),
                missing_corridor_penalty=float(args.missing_corridor_penalty),
            )
            futs[fut] = (i, curr_A)

        done = 0
        ok_solve = 0
        ok_sep = 0

        for fut in as_completed(futs):
            idx, curr_A = futs[fut]
            done += 1

            try:
                res = fut.result()
            except Exception as e:
                res = {
                    "ok": False,
                    "ok_solve": False,
                    "score": 1e18,
                    "error": repr(e),
                    "currents_A": curr_A,
                    "currents_MA": _currents_A_to_MA(curr_A),
                }

            if bool(res.get("ok_solve", False)):
                ok_solve += 1

            si = res.get("score_info", {})
            if isinstance(si, dict) and bool(si.get("has_true_sep", False)):
                ok_sep += 1

            rank, gate_info = score_candidate_with_gates(
                res,
                max_pf6_MA=float(args.max_pf6_gate),
                min_A=float(args.min_A),
                min_kappa=float(args.min_kappa),
                min_delta=float(args.min_delta),
                max_chamfer=float(args.max_chamfer),
                max_leg_dist=float(args.max_leg_dist),
                max_abs_axis_shift_R=float(args.max_Raxis_shift),
                max_abs_axis_shift_Z=float(args.max_Zaxis_shift),
            )

            record = {
                "kind": "eval",
                "index": int(idx),
                "rank_score": float(rank),
                "gate_info": gate_info,
                "result": res,
            }

            _append_jsonl(log_jsonl, record)

            if gate_info["accepted"]:
                accepted.append(record)

            if rank < best_rank:
                best_rank = float(rank)
                best = res
                best_info = gate_info

                _save_json(best_json, {
                    "schema": "scan_corridor_branch_best.v1",
                    "best_rank_score": float(best_rank),
                    "best_gate_info": best_info,
                    "best_currents_A": res.get("currents_A", {}),
                    "best_currents_MA": res.get("currents_MA", {}),
                    "best_result": res,
                    "physics": physics,
                    "target_path": str(Path(target_path).resolve()),
                    "seed_path": str(Path(seed_path).resolve()),
                    "dxf_path": dxf_path,
                    "bounds_MA": bounds,
                    "log_jsonl": str(log_jsonl),
                })

            if done % max(1, int(args.print_every)) == 0 or done == len(candidates):
                print(
                    f"[{done:04d}/{len(candidates):04d}] "
                    f"ok={ok_solve} sep={ok_sep} accepted={len(accepted)} "
                    f"best_rank={best_rank:.6g} "
                    f"A={best_info.get('A', np.nan):.3f} "
                    f"k={best_info.get('kappa', np.nan):.3f} "
                    f"d={best_info.get('delta_bar', np.nan):.3f} "
                    f"ch={best_info.get('chamfer', np.nan):.3f} "
                    f"legU={best_info.get('legU', np.nan):.3f} "
                    f"legL={best_info.get('legL', np.nan):.3f} "
                    f"PF6={best_info.get('PF6_MA', np.nan):.3f}"
                )

    accepted.sort(key=lambda r: float(r.get("rank_score", 1e18)))

    _save_json(accepted_json, {
        "schema": "scan_corridor_branch_accepted.v1",
        "n_accepted": int(len(accepted)),
        "accepted": accepted[: int(args.keep_accepted)],
    })

    out = {
        "schema": "scan_corridor_branch_run.v1",
        "n_candidates": int(len(candidates)),
        "n_accepted": int(len(accepted)),
        "best_rank_score": float(best_rank),
        "best_gate_info": best_info,
        "best_currents_A": best.get("currents_A", {}) if best else {},
        "best_currents_MA": best.get("currents_MA", {}) if best else {},
        "best_result": best,
        "physics": physics,
        "target_path": str(Path(target_path).resolve()),
        "seed_path": str(Path(seed_path).resolve()),
        "dxf_path": dxf_path,
        "bounds_MA": bounds,
        "log_jsonl": str(log_jsonl),
        "best_json": str(best_json),
        "accepted_json": str(accepted_json),
        "runtime_s": float(time.time() - t0),
    }

    _save_json(run_json, out)

    print("\n[OK] finished scan_corridor_branch")
    print(f"[OK] n_accepted={len(accepted)}")
    print(f"[OK] best_rank={best_rank:.6g}")
    print(f"[OK] best currents [MA]={out['best_currents_MA']}")
    print(f"[OK] best json: {best_json}")
    print(f"[OK] accepted: {accepted_json}")
    print(f"[OK] run json : {run_json}")

    return out


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--target", type=str, default=str(_results_dir() / "targets" / "star_simplified_dn_target.json"))
    ap.add_argument("--seed", type=str, default=str(_results_dir() / "seeds" / "branch_probe_low_pf6.json"))
    ap.add_argument("--dxf", type=str, default=str(Path(__file__).resolve().parent / "cad" / "star_baseline.dxf"))

    ap.add_argument("--n-cases", type=int, default=96)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--seed-rng", type=int, default=17)
    ap.add_argument("--print-every", type=int, default=6)
    ap.add_argument("--keep-accepted", type=int, default=25)
    ap.add_argument("--no-coarse", action="store_true")

    # Restricted branch-search bounds [MA].
    ap.add_argument("--cs-min", type=float, default=0.5)
    ap.add_argument("--cs-max", type=float, default=4.0)

    ap.add_argument("--pf1-min", type=float, default=-2.0)
    ap.add_argument("--pf1-max", type=float, default=1.0)

    ap.add_argument("--pf2-min", type=float, default=-10.0)
    ap.add_argument("--pf2-max", type=float, default=-3.0)

    ap.add_argument("--pf3-min", type=float, default=2.0)
    ap.add_argument("--pf3-max", type=float, default=7.0)

    ap.add_argument("--pf4-min", type=float, default=3.0)
    ap.add_argument("--pf4-max", type=float, default=8.5)

    ap.add_argument("--pf5-min", type=float, default=-7.0)
    ap.add_argument("--pf5-max", type=float, default=-1.0)

    ap.add_argument("--pf6-min", type=float, default=1.0)
    ap.add_argument("--pf6-max", type=float, default=10.0)

    # Hard gates / acceptance thresholds.
    ap.add_argument("--max-pf6-gate", type=float, default=10.0)
    ap.add_argument("--min-A", type=float, default=1.50)
    ap.add_argument("--min-kappa", type=float, default=1.90)
    ap.add_argument("--min-delta", type=float, default=0.22)
    ap.add_argument("--max-chamfer", type=float, default=0.50)
    ap.add_argument("--max-leg-dist", type=float, default=2.25)
    ap.add_argument("--max-Raxis-shift", type=float, default=0.35)
    ap.add_argument("--max-Zaxis-shift", type=float, default=0.25)

    # Corridor objective used inside evaluator, intentionally milder than before.
    ap.add_argument("--corridor-weight", type=float, default=45.0)
    ap.add_argument("--corridor-sigma", type=float, default=0.35)
    ap.add_argument("--corridor-roi-pad", type=float, default=0.90)
    ap.add_argument("--missing-corridor-penalty", type=float, default=2.0e5)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_scan(args)


if __name__ == "__main__":
    main()
