# search_highip_branch_star.py
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CURRENT_KEYS = ["CS", "CS_MID", "CS_END", "PF1", "PF2", "PF3", "PF4", "PF5", "PF6"]


# ----------------------------------------------------------------------
# Basic utilities
# ----------------------------------------------------------------------
def now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def isfinite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def recursive_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            got = recursive_get(v, key, None)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = recursive_get(v, key, None)
            if got is not None:
                return got
    return default


def extract_currents_A(d: Dict[str, Any]) -> Dict[str, float]:
    for key in ("currents_A", "best_currents_A"):
        obj = recursive_get(d, key, None)
        if isinstance(obj, dict) and obj:
            return {k: safe_float(obj.get(k, 0.0), 0.0) for k in CURRENT_KEYS}

    for key in ("currents_MA", "best_currents_MA"):
        obj = recursive_get(d, key, None)
        if isinstance(obj, dict) and obj:
            return {k: 1.0e6 * safe_float(obj.get(k, 0.0), 0.0) for k in CURRENT_KEYS}

    # Some older seeds may be flat.
    out = {}
    found_any = False
    for k in CURRENT_KEYS:
        for kk in (k, f"{k}_current", f"{k}_current_A"):
            if kk in d:
                out[k] = safe_float(d.get(kk, 0.0), 0.0)
                found_any = True
                break
        if k not in out:
            out[k] = 0.0
    if found_any:
        return out

    raise KeyError("Could not find currents_A/currents_MA in seed JSON.")


def extract_score_info(d: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("score_info", "best_score_info", "summary", "metrics"):
        obj = recursive_get(d, key, None)
        if isinstance(obj, dict):
            return obj

    # Fallback: collect top-level metrics.
    return d


def score_metric(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    si = extract_score_info(d)
    if key in si:
        return si[key]
    val = recursive_get(d, key, None)
    return default if val is None else val


def parse_ip_paxis_from_filename(name: str) -> Tuple[float, float]:
    m = re.search(r"Ip(?P<ip>[0-9p]+)MA_p(?P<p>[0-9p]+)Pa", name)
    if not m:
        return float("nan"), float("nan")
    return float(m.group("ip").replace("p", ".")), float(m.group("p").replace("p", "."))


def ma_dict(currents_A: Dict[str, float]) -> Dict[str, float]:
    return {k: float(currents_A.get(k, 0.0)) / 1.0e6 for k in CURRENT_KEYS}


def make_seed_json(
    *,
    path: Path,
    currents_A: Dict[str, float],
    name: str,
    source: str,
    Ip_A: float,
    paxis_Pa: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    obj = {
        "schema": "star_highip_branch_seed_v1",
        "name": name,
        "source": source,
        "physics": {
            "Ip_A": float(Ip_A),
            "paxis_Pa": float(paxis_Pa),
        },
        "currents_A": {k: float(currents_A.get(k, 0.0)) for k in CURRENT_KEYS},
        "currents_MA": ma_dict(currents_A),
        "metadata": metadata or {},
    }
    save_json(path, obj)


# ----------------------------------------------------------------------
# Seed generation
# ----------------------------------------------------------------------
def scale_currents(currents_A: Dict[str, float], factor: float, fixed_keys: List[str]) -> Dict[str, float]:
    out = dict(currents_A)
    for k in CURRENT_KEYS:
        if k in fixed_keys:
            out[k] = currents_A.get(k, 0.0)
        else:
            out[k] = factor * currents_A.get(k, 0.0)
    return out


def perturb_currents(
    currents_A: Dict[str, float],
    rng: random.Random,
    *,
    sigma_frac: float,
    fixed_keys: List[str],
    bounds_MA: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    out = dict(currents_A)
    for k in CURRENT_KEYS:
        if k in fixed_keys:
            continue
        base = float(currents_A.get(k, 0.0))
        # At least 0.5 MA scale, otherwise small currents never move.
        scale = max(abs(base), 0.5e6)
        val = base + rng.gauss(0.0, sigma_frac * scale)

        lo_MA, hi_MA = bounds_MA.get(k, (-100.0, 100.0))
        val = min(max(val, lo_MA * 1.0e6), hi_MA * 1.0e6)
        out[k] = val
    return out


def apply_push(
    currents_A: Dict[str, float],
    push_MA: Dict[str, float],
    fixed_keys: List[str],
    bounds_MA: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    out = dict(currents_A)
    for k, dv_MA in push_MA.items():
        if k in fixed_keys:
            continue
        val = float(out.get(k, 0.0)) + float(dv_MA) * 1.0e6
        lo_MA, hi_MA = bounds_MA.get(k, (-100.0, 100.0))
        out[k] = min(max(val, lo_MA * 1.0e6), hi_MA * 1.0e6)
    return out


def default_bounds_MA(args: argparse.Namespace) -> Dict[str, Tuple[float, float]]:
    """
    Exploratory, not strict engineering limits.
    Use reasonably wide windows but avoid completely absurd values.
    """
    cs_lim = float(args.cs_abs_limit_ma)
    pf_lim = float(args.pf_abs_limit_ma)

    return {
        "CS": (0.0, 0.0),  # parent fixed by default
        "CS_MID": (-cs_lim, cs_lim),
        "CS_END": (-cs_lim, cs_lim),
        "PF1": (-pf_lim, pf_lim),
        "PF2": (-pf_lim, pf_lim),
        "PF3": (-pf_lim, pf_lim),
        "PF4": (-pf_lim, pf_lim),
        "PF5": (-pf_lim, pf_lim),
        "PF6": (-pf_lim, pf_lim),
    }


def generate_seed_bank(
    *,
    base_seed_files: List[Path],
    outdir: Path,
    args: argparse.Namespace,
    Ip_A: float,
    paxis_Pa: float,
) -> List[Path]:
    rng = random.Random(args.seed_random)
    bounds_MA = default_bounds_MA(args)
    fixed_keys = [x.strip() for x in args.fixed_keys.split(",") if x.strip()]

    seed_dir = outdir / "generated_seeds"
    seed_dir.mkdir(parents=True, exist_ok=True)

    seed_paths: List[Path] = []

    # Read base seeds.
    base_currents: List[Tuple[str, Dict[str, float]]] = []
    for p in base_seed_files:
        if not p.exists():
            print(f"[WARN] base seed not found, skipping: {p}")
            continue
        try:
            d = load_json(p)
            cur = extract_currents_A(d)
            base_currents.append((p.stem, cur))
        except Exception as e:
            print(f"[WARN] could not load seed {p}: {repr(e)}")

    if not base_currents:
        raise RuntimeError("No valid base seeds found.")

    # 1. Raw copied seeds.
    for name, cur in base_currents:
        path = seed_dir / f"seed_raw_{name}.json"
        make_seed_json(
            path=path,
            currents_A=cur,
            name=f"raw_{name}",
            source=name,
            Ip_A=Ip_A,
            paxis_Pa=paxis_Pa,
            metadata={"kind": "raw"},
        )
        seed_paths.append(path)

    # 2. Scaled seeds.
    for name, cur in base_currents:
        for fac in args.scale_factors:
            cur2 = scale_currents(cur, fac, fixed_keys=fixed_keys)
            path = seed_dir / f"seed_scaled_{name}_x{str(fac).replace('.', 'p')}.json"
            make_seed_json(
                path=path,
                currents_A=cur2,
                name=f"scaled_{name}_x{fac}",
                source=name,
                Ip_A=Ip_A,
                paxis_Pa=paxis_Pa,
                metadata={"kind": "scaled", "factor": fac},
            )
            seed_paths.append(path)

    # 3. Directed pushes around each base seed.
    directed_pushes = [
        # CS-segment variations.
        {"CS_MID": -5.0},
        {"CS_MID": -10.0},
        {"CS_MID": +5.0},
        {"CS_END": -5.0},
        {"CS_END": +5.0},
        {"CS_MID": -5.0, "CS_END": -5.0},
        {"CS_MID": +5.0, "CS_END": +5.0},

        # PF shaping variations.
        {"PF2": -2.0},
        {"PF2": +2.0},
        {"PF4": -2.0},
        {"PF4": +2.0},
        {"PF5": -2.0},
        {"PF5": +2.0},
        {"PF6": -2.0},
        {"PF6": +2.0},

        # Combined shaping.
        {"PF2": -2.0, "PF4": +2.0},
        {"PF2": -2.0, "PF5": +2.0},
        {"PF2": +2.0, "PF6": -2.0},
        {"PF4": +2.0, "PF6": -2.0},
        {"PF4": -2.0, "PF5": +2.0},
    ]

    for name, cur in base_currents:
        for i, push in enumerate(directed_pushes):
            cur2 = apply_push(cur, push, fixed_keys=fixed_keys, bounds_MA=bounds_MA)
            path = seed_dir / f"seed_push_{name}_{i:03d}.json"
            make_seed_json(
                path=path,
                currents_A=cur2,
                name=f"push_{name}_{i:03d}",
                source=name,
                Ip_A=Ip_A,
                paxis_Pa=paxis_Pa,
                metadata={"kind": "directed_push", "push_MA": push},
            )
            seed_paths.append(path)

    # 4. Random perturbations.
    n_random_per_base = int(args.random_per_base)
    for name, cur in base_currents:
        for i in range(n_random_per_base):
            cur2 = perturb_currents(
                cur,
                rng,
                sigma_frac=float(args.seed_sigma_frac),
                fixed_keys=fixed_keys,
                bounds_MA=bounds_MA,
            )
            path = seed_dir / f"seed_rand_{name}_{i:03d}.json"
            make_seed_json(
                path=path,
                currents_A=cur2,
                name=f"rand_{name}_{i:03d}",
                source=name,
                Ip_A=Ip_A,
                paxis_Pa=paxis_Pa,
                metadata={
                    "kind": "random",
                    "sigma_frac": float(args.seed_sigma_frac),
                },
            )
            seed_paths.append(path)

    # Limit total seeds.
    if args.max_seeds > 0 and len(seed_paths) > args.max_seeds:
        # rng.shuffle(seed_paths)
        seed_paths = seed_paths[: args.max_seeds]

    return seed_paths


# ----------------------------------------------------------------------
# Run fit wrapper
# ----------------------------------------------------------------------
def run_fit_for_seed(
    *,
    seed_path: Path,
    attempt_dir: Path,
    args: argparse.Namespace,
    Ip_A: float,
    paxis_Pa: float,
    attempt_index: int,
) -> Dict[str, Any]:
    attempt_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["STAR_IP_A"] = str(float(Ip_A))
    env["STAR_PAXIS_PA"] = str(float(paxis_Pa))
    env["STAR_PASSIVE_STRUCTURES"] = "1" if args.keep_passives else "0"
    env["STAR_PASSIVE_USE_STAR_VESSEL"] = "1" if args.keep_passives else "0"
    env["STAR_PLOT_PASSIVES"] = "1" if args.keep_passives else "0"
    env["STAR_EQ_DOMAIN_SOURCE"] = str(args.domain_source)

    # Scoring weights / branch gates.
    env["STAR_WALL_GAP_TARGET_M"] = str(args.wall_gap_target_m)
    env["STAR_SHAPE_WEIGHT_SCALE"] = str(args.shape_weight_scale)
    env["STAR_BOUNDARY_WEIGHT_SCALE"] = str(args.boundary_weight_scale)
    env["STAR_XPOINT_WEIGHT_SCALE"] = str(args.xpoint_weight_scale)
    env["STAR_SYM_WEIGHT_SCALE"] = str(args.sym_weight_scale)
    env["STAR_CURRENT_REG_SCALE"] = str(args.current_reg_scale)
    env["STAR_CS_REG_SCALE"] = str(args.cs_reg_scale)
    env["STAR_STEP_REG_SCALE"] = str(args.step_reg_scale)

    env["STAR_BRANCH_RAX_MIN"] = str(args.axis_R_min)
    env["STAR_BRANCH_RAX_MAX"] = str(args.axis_R_max)
    env["STAR_BRANCH_ZAX_MAX"] = str(args.axis_Z_max)
    env["STAR_BRANCH_AREA_MIN"] = str(args.area_min)
    env["STAR_BRANCH_AREA_MAX"] = str(args.area_max)
    env["STAR_BRANCH_TERM_WEIGHT"] = str(args.branch_term_weight)

    cmd = [
        sys.executable,
        "-u",
        str(Path(args.fit_script)),
        "--target",
        str(Path(args.target)),
        "--seed",
        str(seed_path),
        "--stage",
        "release-cs",
        "--iters",
        str(int(args.iters)),
        "--pop",
        str(int(args.pop)),
        "--workers",
        str(int(args.workers)),
        "--timeout",
        str(float(args.eval_timeout)),
        "--widen",
        str(float(args.widen)),
        "--sigma-frac",
        str(float(args.sigma_frac)),
        "--free-keys",
        str(args.free_keys),
        "--dxf",
        str(Path(args.dxf)),
        "--dshape-weight",
        str(float(args.dshape_weight)),
        "--inboard-weight",
        str(float(args.inboard_weight)),
        "--target-rms-weight",
        str(float(args.target_rms_weight)),
        "--delta-weight",
        str(float(args.delta_weight)),
        "--delta-target",
        str(float(args.target_delta)),
        "--kappa-min",
        str(float(args.kappa_min)),
        "--A-min",
        str(float(args.A_min)),
        "--chamfer-max-m",
        str(float(args.chamfer_max_m)),
    ]

    if args.no_early_stop:
        cmd.append("--no-early-stop")

    log_path = attempt_dir / "stdout_stderr.log"

    print("\n" + "=" * 100)
    print(f"[ATTEMPT {attempt_index:04d}] seed={seed_path.name}")
    print(f"[RUN] {' '.join(cmd)}")
    print("=" * 100)

    t0 = _dt.datetime.now()

    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as logf:
            proc = subprocess.run(
                cmd,
                cwd=str(Path.cwd()),
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=float(args.command_timeout),
            )
        returncode = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        returncode = -999
        timed_out = True

    dt = (_dt.datetime.now() - t0).total_seconds()

    # Copy the global fit outputs if they exist. Your current wrapper writes to ./results.
    global_best = Path("results") / "fit_simplified_dn_toposafe_best.json"
    global_run_candidates = sorted(Path("results").glob("fit_simplified_dn_toposafe_run_*.json"))
    global_log = Path("results") / "fit_simplified_dn_toposafe_results.jsonl"

    copied_best = attempt_dir / "best.json"
    copied_run = attempt_dir / "run.json"
    copied_jsonl = attempt_dir / "results.jsonl"

    if global_best.exists():
        shutil.copy2(global_best, copied_best)
    if global_run_candidates:
        shutil.copy2(global_run_candidates[-1], copied_run)
    if global_log.exists():
        shutil.copy2(global_log, copied_jsonl)

    result: Dict[str, Any] = {
        "attempt_index": attempt_index,
        "seed_path": str(seed_path),
        "attempt_dir": str(attempt_dir),
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_s": dt,
        "log_path": str(log_path),
        "best_json": str(copied_best) if copied_best.exists() else "",
        "run_json": str(copied_run) if copied_run.exists() else "",
    }

    if copied_best.exists():
        try:
            best = load_json(copied_best)
            result.update(summarize_candidate(best))
        except Exception as e:
            result["summary_error"] = repr(e)

    return result


def summarize_candidate(best: Dict[str, Any]) -> Dict[str, Any]:
    si = extract_score_info(best)

    def g(key: str, default: Any = None) -> Any:
        if key in si:
            return si[key]
        return recursive_get(best, key, default)

    currents_A = extract_currents_A(best)

    out = {
        "score": safe_float(g("score", recursive_get(best, "score", float("nan")))),
        "ok": bool(g("ok", False)),
        "ok_sep": bool(g("ok_sep", g("has_true_sep", g("has_true_separatrix", False)))),
        "has_true_separatrix": bool(g("has_true_separatrix", g("has_true_sep", False))),
        "inside_WALL_INNER": bool(g("inside_WALL_INNER", False)),
        "signed_gap_to_WALL_INNER_m": safe_float(g("signed_gap_to_WALL_INNER_m", float("nan"))),
        "outside_frac": safe_float(g("outside_frac", 1.0), 1.0),
        "branch_term": safe_float(g("branch_term", 0.0), 0.0),
        "Rax": safe_float(g("Rax", g("R_axis_m", float("nan")))),
        "Zax": safe_float(g("Zax", g("Z_axis_m", float("nan")))),
        "R0": safe_float(g("R0", float("nan"))),
        "A": safe_float(g("A", float("nan"))),
        "kappa": safe_float(g("kappa", float("nan"))),
        "delta": safe_float(g("delta", g("delta_bar", float("nan")))),
        "area": safe_float(g("area", g("area_m2", float("nan")))),
        "n_xpoints": int(safe_float(g("n_xpoints", 0), 0)),
        "CS_MID_MA": currents_A.get("CS_MID", 0.0) / 1.0e6,
        "CS_END_MA": currents_A.get("CS_END", 0.0) / 1.0e6,
        "PF1_MA": currents_A.get("PF1", 0.0) / 1.0e6,
        "PF2_MA": currents_A.get("PF2", 0.0) / 1.0e6,
        "PF3_MA": currents_A.get("PF3", 0.0) / 1.0e6,
        "PF4_MA": currents_A.get("PF4", 0.0) / 1.0e6,
        "PF5_MA": currents_A.get("PF5", 0.0) / 1.0e6,
        "PF6_MA": currents_A.get("PF6", 0.0) / 1.0e6,
    }
    return out


def accepted_by_search(result: Dict[str, Any], args: argparse.Namespace) -> bool:
    if result.get("timed_out", False):
        return False

    if not bool(result.get("has_true_separatrix", False)) and not bool(result.get("ok_sep", False)):
        return False

    if not bool(result.get("inside_WALL_INNER", False)):
        return False

    outside = safe_float(result.get("outside_frac", 1.0), 1.0)
    if not isfinite(outside) or outside > float(args.max_outside_frac):
        return False

    gap = safe_float(result.get("signed_gap_to_WALL_INNER_m", float("nan")))
    if not isfinite(gap) or gap < float(args.min_gap_m):
        return False

    branch = safe_float(result.get("branch_term", 0.0), 0.0)
    if isfinite(branch) and branch > 0.0:
        return False

    Rax = safe_float(result.get("Rax", float("nan")))
    Zax = safe_float(result.get("Zax", float("nan")))
    area = safe_float(result.get("area", float("nan")))
    A = safe_float(result.get("A", float("nan")))
    kap = safe_float(result.get("kappa", float("nan")))

    if not (float(args.axis_R_min) <= Rax <= float(args.axis_R_max)):
        return False
    if not (abs(Zax) <= float(args.axis_Z_max)):
        return False
    if not (float(args.area_min) <= area <= float(args.area_max)):
        return False
    if not (A >= float(args.A_min)):
        return False
    if not (kap >= float(args.kappa_min)):
        return False

    return True


def result_sort_key(r: Dict[str, Any]) -> Tuple[int, float, float, float]:
    accepted = 0 if r.get("accepted", False) else 1
    score = safe_float(r.get("score", 1.0e99), 1.0e99)
    # Prefer larger gap after score.
    gap = -safe_float(r.get("signed_gap_to_WALL_INNER_m", -1.0), -1.0)
    outside = safe_float(r.get("outside_frac", 1.0), 1.0)
    return accepted, score, gap, outside


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    keys = [
        "accepted",
        "attempt_index",
        "score",
        "ok",
        "has_true_separatrix",
        "inside_WALL_INNER",
        "signed_gap_to_WALL_INNER_m",
        "outside_frac",
        "branch_term",
        "Rax",
        "Zax",
        "R0",
        "A",
        "kappa",
        "delta",
        "area",
        "n_xpoints",
        "CS_MID_MA",
        "CS_END_MA",
        "PF1_MA",
        "PF2_MA",
        "PF3_MA",
        "PF4_MA",
        "PF5_MA",
        "PF6_MA",
        "elapsed_s",
        "returncode",
        "timed_out",
        "seed_path",
        "best_json",
        "attempt_dir",
        "log_path",
    ]

    extra = sorted({k for r in rows for k in r.keys()} - set(keys))
    keys = keys + extra

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="High-Ip branch search for STAR-like free-boundary equilibria."
    )

    ap.add_argument("--target", required=True)
    ap.add_argument("--dxf", default=r".\cad\star_baseline.dxf")
    ap.add_argument("--fit-script", default=r".\fit_simplified_dn_dshape_first.py")
    ap.add_argument("--outdir", default=None)

    ap.add_argument("--Ip-MA", type=float, default=13.2)
    ap.add_argument("--paxis-Pa", type=float, default=2.0e3)

    ap.add_argument(
        "--base-seeds",
        required=True,
        help="Comma-separated seed JSON files. Use baseline 4MA, last 7.15MA, ugly 13.2MA, etc.",
    )

    ap.add_argument("--max-seeds", type=int, default=48)
    ap.add_argument("--random-per-base", type=int, default=8)
    ap.add_argument("--seed-sigma-frac", type=float, default=0.35)
    ap.add_argument("--seed-random", type=int, default=123)
    ap.add_argument(
        "--scale-factors",
        type=float,
        nargs="*",
        default=[0.75, 1.0, 1.25, 1.5, 2.0],
    )

    ap.add_argument("--free-keys", default="CS_MID,CS_END,PF1,PF2,PF3,PF4,PF5,PF6")
    ap.add_argument("--fixed-keys", default="CS")
    ap.add_argument("--cs-abs-limit-ma", type=float, default=80.0)
    ap.add_argument("--pf-abs-limit-ma", type=float, default=25.0)

    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pop", type=int, default=36)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--eval-timeout", type=float, default=150.0)
    ap.add_argument("--command-timeout", type=float, default=3600.0)

    ap.add_argument("--widen", type=float, default=0.12)
    ap.add_argument("--sigma-frac", type=float, default=0.16)
    ap.add_argument("--no-early-stop", action="store_true")

    # Objective weights passed to dshape wrapper.
    ap.add_argument("--dshape-weight", type=float, default=0.02)
    ap.add_argument("--inboard-weight", type=float, default=2.0)
    ap.add_argument("--target-rms-weight", type=float, default=0.0)
    ap.add_argument("--delta-weight", type=float, default=0.2)
    ap.add_argument("--target-delta", type=float, default=0.4695)
    ap.add_argument("--kappa-min", type=float, default=1.25)
    ap.add_argument("--A-min", type=float, default=1.15)
    ap.add_argument("--chamfer-max-m", type=float, default=0.32)

    # Environment scoring scales.
    ap.add_argument("--wall-gap-target-m", type=float, default=0.005)
    ap.add_argument("--shape-weight-scale", type=float, default=0.05)
    ap.add_argument("--boundary-weight-scale", type=float, default=0.03)
    ap.add_argument("--xpoint-weight-scale", type=float, default=0.02)
    ap.add_argument("--sym-weight-scale", type=float, default=0.04)
    ap.add_argument("--current-reg-scale", type=float, default=0.02)
    ap.add_argument("--cs-reg-scale", type=float, default=0.008)
    ap.add_argument("--step-reg-scale", type=float, default=0.012)

    # Acceptance gates.
    ap.add_argument("--min-gap-m", type=float, default=0.001)
    ap.add_argument("--max-outside-frac", type=float, default=0.0)
    ap.add_argument("--axis-R-min", type=float, default=3.2)
    ap.add_argument("--axis-R-max", type=float, default=5.2)
    ap.add_argument("--axis-Z-max", type=float, default=0.8)
    ap.add_argument("--area-min", type=float, default=15.0)
    ap.add_argument("--area-max", type=float, default=50.0)
    ap.add_argument("--branch-term-weight", type=float, default=1.0e6)

    ap.add_argument("--keep-passives", action="store_true")
    ap.add_argument("--domain-source", default="outer", choices=["outer", "machine", "inner"])

    ap.add_argument(
        "--stop-after-accepted",
        type=int,
        default=3,
        help="Stop once this many accepted candidates are found. Use 0 to run all.",
    )

    args = ap.parse_args()

    Ip_A = float(args.Ip_MA) * 1.0e6
    paxis_Pa = float(args.paxis_Pa)

    outdir = Path(args.outdir) if args.outdir else Path("results") / f"highip_branch_search_{now_tag()}"
    outdir.mkdir(parents=True, exist_ok=True)

    base_seed_files = [Path(s.strip()) for s in args.base_seeds.split(",") if s.strip()]

    print("\n" + "=" * 100)
    print("[STAR high-Ip branch search]")
    print(f"outdir     = {outdir.resolve()}")
    print(f"Ip         = {args.Ip_MA:.4f} MA")
    print(f"paxis      = {paxis_Pa:.4e} Pa")
    print(f"target     = {Path(args.target).resolve()}")
    print(f"dxf        = {Path(args.dxf).resolve()}")
    print(f"base seeds = {len(base_seed_files)}")
    for p in base_seed_files:
        print(f"  - {p}")
    print("=" * 100 + "\n")

    save_json(outdir / "search_config.json", vars(args))

    seed_paths = generate_seed_bank(
        base_seed_files=base_seed_files,
        outdir=outdir,
        args=args,
        Ip_A=Ip_A,
        paxis_Pa=paxis_Pa,
    )

    print(f"[INFO] generated seeds: {len(seed_paths)}")

    manifest_path = outdir / "manifest.jsonl"
    rows: List[Dict[str, Any]] = []
    accepted_count = 0

    for i, seed_path in enumerate(seed_paths, start=1):
        attempt_dir = outdir / "attempts" / f"attempt_{i:04d}_{seed_path.stem}"

        result = run_fit_for_seed(
            seed_path=seed_path,
            attempt_dir=attempt_dir,
            args=args,
            Ip_A=Ip_A,
            paxis_Pa=paxis_Pa,
            attempt_index=i,
        )

        result["accepted"] = accepted_by_search(result, args)
        rows.append(result)

        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if result["accepted"]:
            accepted_count += 1

            accepted_dir = outdir / "accepted"
            accepted_dir.mkdir(parents=True, exist_ok=True)

            src = Path(result.get("best_json", ""))
            if src.exists():
                dst = accepted_dir / f"accepted_{accepted_count:03d}_attempt_{i:04d}.json"
                shutil.copy2(src, dst)
                result["accepted_copy"] = str(dst)

            print(
                f"[ACCEPTED] attempt={i:04d} "
                f"score={result.get('score')} "
                f"gap={result.get('signed_gap_to_WALL_INNER_m')} "
                f"A={result.get('A')} k={result.get('kappa')} "
                f"Rax={result.get('Rax')} area={result.get('area')}"
            )
        else:
            print(
                f"[REJECT] attempt={i:04d} "
                f"score={result.get('score')} "
                f"sep={result.get('has_true_separatrix')} "
                f"in={result.get('inside_WALL_INNER')} "
                f"gap={result.get('signed_gap_to_WALL_INNER_m')} "
                f"out={result.get('outside_frac')} "
                f"Rax={result.get('Rax')} area={result.get('area')}"
            )

        write_summary_csv(outdir / "summary.csv", rows)

        if args.stop_after_accepted > 0 and accepted_count >= args.stop_after_accepted:
            print(f"[STOP] Found {accepted_count} accepted candidates.")
            break

    rows_sorted = sorted(rows, key=result_sort_key)
    save_json(outdir / "best_ranked_results.json", {"results": rows_sorted[:20]})
    write_summary_csv(outdir / "summary_ranked.csv", rows_sorted)

    print("\n" + "=" * 100)
    print("[DONE]")
    print(f"outdir          = {outdir.resolve()}")
    print(f"manifest        = {manifest_path.resolve()}")
    print(f"summary         = {(outdir / 'summary.csv').resolve()}")
    print(f"ranked summary  = {(outdir / 'summary_ranked.csv').resolve()}")
    print(f"accepted count  = {accepted_count}")
    print("=" * 100)

    if accepted_count == 0:
        print(
            "\n[NOTE] No accepted high-Ip branch found. "
            "This does not prove non-existence; it means none was found within this seed bank and bounds."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
