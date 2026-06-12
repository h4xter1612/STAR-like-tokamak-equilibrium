#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
continue_ip_paxis_star_incremental.py

Coupled Ip + paxis ramp-up for the STAR-like branch.

Instead of doing Ip first and pressure later, this script advances both
together. At every stage it runs repeated local current scans until either:
  - the current stage satisfies containment + shape criteria, or
  - a maximum number of fit iterations is reached.

Designed for the corrected analyze_star.py policy where rejected limiter and
rejected psi_bndry fallback curves are not accepted as physical LCFS.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

DSHAPE_BEST = RESULTS / "fit_simplified_dn_toposafe_best.json"
DSHAPE_JSONL = RESULTS / "fit_simplified_dn_toposafe_results.jsonl"


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _as_path(p: str | Path) -> Path:
    p = Path(p)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
        return y if y == y else default
    except Exception:
        return default


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _label_number(x: float, nd: int = 4) -> str:
    s = f"{x:.{nd}g}"
    return s.replace("+", "").replace("-", "m").replace(".", "p")


def _stage_label(ip_ma: float, paxis_pa: float) -> str:
    return f"Ip{_label_number(ip_ma)}MA_p{_label_number(paxis_pa)}Pa"


def _parse_pair_stages(s: str) -> List[Tuple[float, float]]:
    """Parse explicit stages: '4.0:2e3,5.0:1e4,6.5:5e4'."""
    out: List[Tuple[float, float]] = []
    if not s.strip():
        return out
    for tok in re.split(r"[,\s]+", s.strip()):
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(f"Bad stage token {tok!r}; expected IpMA:paxisPa")
        a, b = tok.split(":", 1)
        out.append((float(a), float(b)))
    return out


def _make_coupled_stages(args: argparse.Namespace) -> List[Tuple[float, float]]:
    explicit = _parse_pair_stages(args.stages)
    if explicit:
        return explicit

    n = int(args.n_stages)
    if n < 2:
        raise ValueError("--n-stages must be >= 2 unless --stages is used")

    ip0, ip1 = float(args.ip_start_ma), float(args.ip_end_ma)
    p0, p1 = float(args.paxis_start_pa), float(args.paxis_end_pa)
    mode = str(args.paxis_schedule).lower().strip()

    out: List[Tuple[float, float]] = []
    for k in range(n):
        t = k / float(n - 1)
        ip = ip0 + t * (ip1 - ip0)
        if mode == "linear":
            p = p0 + t * (p1 - p0)
        else:
            p0c = max(p0, 1.0e-30)
            p1c = max(p1, 1.0e-30)
            p = p0c * (p1c / p0c) ** t
        out.append((float(ip), float(p)))
    return out


def _extract_result_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(d.get("best_result"), dict):
        return d["best_result"]
    if isinstance(d.get("result"), dict):
        return d["result"]
    return d


def _get_nested(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Recursive dict lookup.

    Needed because fit_simplified_dn_toposafe_best.json stores useful metrics
    inside best_result.score_info and best_result.diag, not always at top level.
    """
    if not isinstance(d, dict):
        return default

    if key in d:
        return d[key]

    # Common subdicts first.
    for sub in (
        "plasma_diag",
        "diag",
        "shape",
        "metrics",
        "score_info",
        "best_result",
        "result",
        "diagnostics",
        "containment",
    ):
        x = d.get(sub)
        if isinstance(x, dict):
            v = _get_nested(x, key, None)
            if v is not None:
                return v

    # Full recursive fallback.
    for v in d.values():
        if isinstance(v, dict):
            found = _get_nested(v, key, None)
            if found is not None:
                return found

    return default

def _summarize_best(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    try:
        d = _load_json(path)
    except Exception as e:
        return {"exists": True, "path": str(path), "read_error": repr(e)}

    br = _extract_result_dict(d)
    out: Dict[str, Any] = {"exists": True, "path": str(path)}

    for key in ("best_currents_A", "currents_A", "family_currents_A"):
        if isinstance(d.get(key), dict):
            out["currents_A"] = d[key]
            break
        if isinstance(br, dict) and isinstance(br.get(key), dict):
            out["currents_A"] = br[key]
            break

    for key in ("best_currents_MA", "currents_MA"):
        if isinstance(d.get(key), dict):
            out["currents_MA"] = d[key]
            break
        if isinstance(br, dict) and isinstance(br.get(key), dict):
            out["currents_MA"] = br[key]
            break

    for k in (
        "score", "ok", "sep", "ok_sep", "has_separatrix", "has_usable_sep",
        "has_true_separatrix", "has_freegs_psibndry_sep", "sep_source",
        "reason", "shape_reason", "n_xpoints", "nxp", "inside_WALL_INNER",
        "signed_gap_to_WALL_INNER_m", "min_wall_gap_m", "outside_frac",
        "has_true_sep",
        "boundary_chamfer_m",
        "fail_reasons",
    ):
        v = _get_nested(br, k, None)
        if v is not None:
            out[k] = v

    aliases = {
        "Rax": ("Rax", "R_ax", "R_axis_m", "R_axis"),
        "Zax": ("Zax", "Z_ax", "Z_axis_m", "Z_axis"),

        "R0": ("R0", "R0_m", "R0_plasma"),
        "A": ("A", "A_plasma"),
        "kappa": ("kappa", "kappa_plasma"),
        "delta_u": ("delta_u",),
        "delta_l": ("delta_l",),
        "delta": ("delta", "delta_bar", "delta_bar_plasma"),
        "area": ("area", "area_m2", "plasma_area_m2"),

        "Rmin": ("Rmin",),
        "Rmax": ("Rmax",),
        "Zmin": ("Zmin",),
        "Zmax": ("Zmax",),
    }
    for outk, keys in aliases.items():
        for k in keys:
            v = _get_nested(br, k, None)
            if v is not None:
                out[outk] = v
                break

    if "delta" not in out:
        du = _safe_float(out.get("delta_u"), float("nan"))
        dl = _safe_float(out.get("delta_l"), float("nan"))
        if du == du and dl == dl:
            out["delta"] = 0.5 * (du + dl)

    pd = br.get("plasma_diag") if isinstance(br, dict) else None
    if isinstance(pd, dict):
        out["plasma_diag_ok"] = pd.get("ok")
        out["plasma_diag_method"] = pd.get("method")
        out["plasma_diag_reason"] = pd.get("reason")

    return out


def _stage_satisfied(summary: Dict[str, Any], args: argparse.Namespace) -> Tuple[bool, Dict[str, Any]]:
    comp: Dict[str, Any] = {}
    sep_source = str(summary.get("sep_source", "") or "")
    reason = str(summary.get("reason", summary.get("shape_reason", "")) or "")

    rejected = (
        "rejected" in sep_source.lower()
        or "rejected" in reason.lower()
        or "fallback_only" in reason.lower()
        or "no_physical_separatrix" in reason.lower()
        or "far_from" in reason.lower()
        or "not_near" in reason.lower()
    )

    has_sep = bool(
        summary.get("ok_sep", False)
        or summary.get("has_usable_sep", False)
        or summary.get("has_true_separatrix", False)
        or summary.get("has_true_sep", False)
        or summary.get("has_freegs_psibndry_sep", False)
    ) and not rejected

    fail_reasons = summary.get("fail_reasons", [])
    if not isinstance(fail_reasons, list):
        fail_reasons = []

    gap = _safe_float(
        summary.get(
            "signed_gap_to_WALL_INNER_m",
            summary.get("min_wall_gap_m", float("nan")),
        ),
        float("nan"),
    )
    outside_frac = _safe_float(summary.get("outside_frac", 1.0), 1.0)

    inside_raw = summary.get("inside_WALL_INNER", None)

    if inside_raw is None:
        # Do NOT infer wall containment from boundary_chamfer_m or fail_reasons.
        # If explicit WALL_INNER containment is missing, reject the stage.
        inside = False
    else:
        inside = bool(inside_raw)

    R0 = _safe_float(summary.get("R0"), float("nan"))
    A = _safe_float(summary.get("A"), float("nan"))
    kappa = _safe_float(summary.get("kappa"), float("nan"))
    delta = _safe_float(summary.get("delta"), float("nan"))
    Rax = _safe_float(summary.get("Rax"), float("nan"))
    Zax = _safe_float(summary.get("Zax"), float("nan"))
    area = _safe_float(summary.get("area"), float("nan"))

    comp.update({
        "has_sep": has_sep,
        "inside_WALL_INNER": inside,
        "gap": gap,
        "outside_frac": outside_frac,
        "Rax": Rax,
        "Zax": Zax,
        "area": area,
        "R0": R0,
        "A": A,
        "kappa": kappa,
        "delta": delta,
        "sep_source": sep_source,
        "reason": reason,
    })

    if not has_sep:
        comp["fail"] = "no_physical_usable_sep"
        return False, comp
    if not inside:
        comp["fail"] = "not_inside_WALL_INNER"
        return False, comp
    if not (outside_frac == outside_frac) or outside_frac > float(getattr(args, "max_outside_frac", 0.0)):
        comp["fail"] = f"outside_frac_nonzero:{outside_frac}"
        return False, comp
    if not (gap == gap) or gap < float(args.min_gap_m):
        comp["fail"] = f"gap_too_small:{gap}"
        return False, comp

    # Branch-lock gates: reject branch jumps even if a separatrix-like curve exists.
    for name, val in [
        ("Rax", Rax),
        ("Zax", Zax),
        ("area", area),
        ("R0", R0),
        ("A", A),
        ("kappa", kappa),
        ("delta", delta),
    ]:
        if not (val == val):
            comp["fail"] = f"{name}_nan"
            return False, comp

    if not (float(args.axis_R_min) <= Rax <= float(args.axis_R_max)):
        comp["fail"] = f"bad_axis_R:{Rax}"
        return False, comp

    if abs(Zax) > float(args.axis_Z_max):
        comp["fail"] = f"bad_axis_Z:{Zax}"
        return False, comp

    if not (float(args.area_min) <= area <= float(args.area_max)):
        comp["fail"] = f"bad_area:{area}"
        return False, comp

    if abs(R0 - float(args.target_R0)) > float(args.tol_R0):
        comp["fail"] = f"R0_out:{R0}"
        return False, comp
    if abs(A - float(args.target_A)) > float(args.tol_A):
        comp["fail"] = f"A_out:{A}"
        return False, comp
    if abs(kappa - float(args.target_kappa)) > float(args.tol_kappa):
        comp["fail"] = f"kappa_out:{kappa}"
        return False, comp
    if abs(delta - float(args.target_delta)) > float(args.tol_delta):
        comp["fail"] = f"delta_out:{delta}"
        return False, comp

    comp["fail"] = None
    return True, comp


def _run(cmd: List[str], env: Dict[str, str], log_path: Path, timeout_s: Optional[float]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        log.write("\n" + "=" * 100 + "\n")
        log.write("[RUN] " + " ".join(cmd) + "\n")
        log.write("=" * 100 + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = 124
            msg = f"\n[TIMEOUT] command killed after {timeout_s} s\n"
            print(msg)
            log.write(msg)
        except KeyboardInterrupt:
            proc.kill()
            rc = 130
            msg = "\n[INTERRUPTED] command killed by user\n"
            print(msg)
            log.write(msg)
            raise
        finally:
            dt = time.time() - t0
            log.write(f"\n[EXIT] rc={rc} elapsed_s={dt:.1f}\n")
            log.flush()
    return int(rc)


def _stage_env(args: argparse.Namespace, ip_ma: float, paxis_pa: float) -> Dict[str, str]:
    env = os.environ.copy()
    env["STAR_IP_A"] = f"{float(ip_ma) * 1e6:.9g}"
    env["STAR_PAXIS_PA"] = f"{float(paxis_pa):.9g}"
    if not args.keep_passives:
        env["STAR_PASSIVE_STRUCTURES"] = "0"
        env["STAR_PASSIVE_USE_STAR_VESSEL"] = "0"
        env["STAR_PLOT_PASSIVES"] = "0"
        env["STAR_EQ_DOMAIN_SOURCE"] = str(args.domain_source)
    if args.nx:
        env["STAR_NX_EQ"] = str(int(args.nx))
    if args.ny:
        env["STAR_NY_EQ"] = str(int(args.ny))
    return env


def _append_manifest(outdir: Path, rec: Dict[str, Any]) -> None:
    with open(outdir / "manifest.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _find_latest_success(outdir: Path) -> Optional[Tuple[int, Path]]:
    manifest = outdir / "manifest.jsonl"
    if not manifest.exists():
        return None
    latest: Optional[Tuple[int, Path]] = None
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not rec.get("accepted", False):
                continue
            p = Path(str(rec.get("best_json", "")))
            if not p.exists():
                continue
            idx = int(rec.get("stage_index", -1))
            if latest is None or idx > latest[0]:
                latest = (idx, p)
    return latest


def _run_dshape_round(args: argparse.Namespace, seed: Path, target: Path, ip_ma: float, paxis_pa: float, stage_dir: Path, round_idx: int, env: Dict[str, str]) -> Optional[Path]:
    label = _stage_label(ip_ma, paxis_pa)
    log = stage_dir / f"{label}_round_{round_idx:02d}.log"
    cmd = [
        sys.executable, "-u", ".\\fit_simplified_dn_dshape_first.py",
        "--target", str(target),
        "--seed", str(seed),
        "--stage", "release-cs",
        "--iters", str(int(args.iters_per_round)),
        "--pop", str(int(args.pop)),
        "--workers", str(int(args.workers)),
        "--timeout", str(float(args.eval_timeout)),
        "--widen", str(float(args.widen)),
        "--sigma-frac", str(float(args.sigma_frac)),
        "--free-keys", str(args.free_keys),
        "--no-early-stop",
        "--dxf", str(args.dxf),
        "--dshape-weight", str(float(args.dshape_weight)),
        "--inboard-weight", str(float(args.inboard_weight)),
        "--target-rms-weight", str(float(args.target_rms_weight)),
        "--delta-weight", str(float(args.delta_weight)),
        "--delta-target", str(float(args.target_delta)),
        "--kappa-min", str(float(args.kappa_min)),
        "--A-min", str(float(args.A_min)),
        "--chamfer-max-m", str(float(args.chamfer_max_m)),
    ]
    rc = _run(cmd, env=env, log_path=log, timeout_s=float(args.command_timeout))
    if rc != 0:
        print(f"[WARN] dshape round failed rc={rc}")
        return None
    dst = stage_dir / f"{label}_round_{round_idx:02d}_best.json"
    if not _copy_if_exists(DSHAPE_BEST, dst):
        print(f"[WARN] expected best not found: {DSHAPE_BEST}")
        return None
    _copy_if_exists(DSHAPE_JSONL, stage_dir / f"{label}_round_{round_idx:02d}_results.jsonl")
    print(f"[SAVED] {dst}")
    return dst

def _accept_frozen_stage(
    *,
    outdir: Path,
    stage_dir: Path,
    current_seed: Path,
    label: str,
    idx: int,
    ip_ma: float,
    paxis_pa: float,
    target: Path,
) -> Path:
    """
    Accept current_seed directly as the best for this stage.

    Used to prevent the optimizer from deforming the known-good baseline
    at the first low-Ip/low-pressure point.
    """
    canonical = outdir / f"{label}_best.json"
    shutil.copy2(current_seed, canonical)

    summary = _summarize_best(canonical)

    rec: Dict[str, Any] = {
        "stage_index": idx,
        "ip_ma": ip_ma,
        "paxis_pa": paxis_pa,
        "seed_in": str(current_seed),
        "target": str(target),
        "started": time.ctime(),
        "ended": time.ctime(),
        "accepted": True,
        "status": "frozen_seed_accepted",
        "best_json": str(canonical),
        "summary": summary,
        "rounds": [],
    }

    _append_manifest(outdir, rec)
    _write_json(stage_dir / "stage_record.json", rec)

    print("\n" + "-" * 100)
    print(f"[FROZEN ACCEPTED] {label}")
    print(f"[BEST] {canonical}")
    print("[SUMMARY]", json.dumps(summary, indent=2, ensure_ascii=False)[:2500])
    print("-" * 100 + "\n")

    return canonical

def main() -> None:
    ap = argparse.ArgumentParser(description="Coupled STAR Ip+paxis ramp with repeated local optimization per stage.")
    ap.add_argument("--seed", type=str, required=False, help="Initial good 4MA/low-p seed JSON.")
    ap.add_argument("--target", type=str, required=True, help="Branch target JSON.")
    ap.add_argument("--dxf", type=str, default=".\\cad\\star_baseline.dxf")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--stages", type=str, default="", help='Explicit stages "IpMA:paxisPa,IpMA:paxisPa,..."')
    ap.add_argument("--n-stages", type=int, default=28)
    ap.add_argument("--freeze-first-stage", action="store_true", 
                    help="Copy the initial seed as accepted best for the first stage without re-optimizing it.")
    ap.add_argument("--ip-start-ma", type=float, default=4.0)
    ap.add_argument("--ip-end-ma", type=float, default=13.2)
    ap.add_argument("--paxis-start-pa", type=float, default=2.0e3)
    ap.add_argument("--paxis-end-pa", type=float, default=1.2e6)
    ap.add_argument("--paxis-schedule", type=str, default="geom", choices=["geom", "linear"])
    ap.add_argument("--iters-per-round", type=int, default=5)
    ap.add_argument("--max-fit-iters-per-stage", type=int, default=50)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-timeout", type=float, default=150.0)
    ap.add_argument("--command-timeout", type=float, default=3600.0)
    ap.add_argument("--widen", type=float, default=0.008)
    ap.add_argument("--sigma-frac", type=float, default=0.016)
    ap.add_argument("--free-keys", type=str, default="CS_MID,CS_END,PF1,PF2,PF3,PF4,PF5,PF6")
    ap.add_argument("--dshape-weight", type=float, default=1.25)
    ap.add_argument("--inboard-weight", type=float, default=90.0)
    ap.add_argument("--target-rms-weight", type=float, default=0.0)
    ap.add_argument("--delta-weight", type=float, default=20.0)
    ap.add_argument("--chamfer-max-m", type=float, default=0.32)
    ap.add_argument("--target-R0", type=float, default=4.2083)
    ap.add_argument("--target-A", type=float, default=2.0275)
    ap.add_argument("--target-kappa", type=float, default=2.16)
    ap.add_argument("--target-delta", type=float, default=0.4695)
    ap.add_argument("--keep-passives", action="store_true")
    ap.add_argument("--tol-R0", type=float, default=0.16)
    ap.add_argument("--tol-A", type=float, default=0.12)
    ap.add_argument("--tol-kappa", type=float, default=0.14)
    ap.add_argument("--tol-delta", type=float, default=0.09)
    ap.add_argument("--min-gap-m", type=float, default=0.05)
    ap.add_argument("--A-min", type=float, default=1.92)
    ap.add_argument("--kappa-min", type=float, default=2.03)
    ap.add_argument("--domain-source", type=str, default="outer", choices=["outer", "machine", "inner"])
    ap.add_argument("--nx", type=int, default=0)
    ap.add_argument("--ny", type=int, default=0)
    ap.add_argument("--max-outside-frac", type=float, default=0.0)
    ap.add_argument("--axis-R-min", type=float, default=3.5)
    ap.add_argument("--axis-R-max", type=float, default=5.0)
    ap.add_argument("--axis-Z-max", type=float, default=0.8)

    ap.add_argument("--area-min", type=float, default=18.0)
    ap.add_argument("--area-max", type=float, default=45.0)
    args = ap.parse_args()

    target = _as_path(args.target)
    if not target.exists():
        raise FileNotFoundError(f"Target not found: {target}")
    stages = _make_coupled_stages(args)
    outdir = _as_path(args.outdir) if args.outdir else RESULTS / f"ramp_ip_paxis_coupled_{_now_tag()}"
    outdir.mkdir(parents=True, exist_ok=True)

    if args.restart and (outdir / "manifest.jsonl").exists():
        backup = outdir / f"manifest_backup_{_now_tag()}.jsonl"
        shutil.copy2(outdir / "manifest.jsonl", backup)
        (outdir / "manifest.jsonl").unlink()
        print(f"[RESTART] old manifest backed up to {backup}")

    seed: Optional[Path] = None
    start_stage_idx = 0
    if args.resume and not args.restart:
        latest = _find_latest_success(outdir)
        if latest is not None:
            idx, seed_path = latest
            seed = seed_path
            start_stage_idx = idx + 1
            print(f"[RESUME] latest accepted: stage={idx} seed={seed}")
        else:
            print("[RESUME] no accepted stage found; starting from --seed")
    if seed is None:
        if not args.seed:
            raise ValueError("--seed is required unless --resume finds a previous accepted stage.")
        seed = _as_path(args.seed)
        if not seed.exists():
            raise FileNotFoundError(f"Seed not found: {seed}")

    _write_json(outdir / "run_config.json", {
        "created": time.ctime(),
        "seed_initial": str(seed),
        "target": str(target),
        "dxf": str(args.dxf),
        "stages": [{"ip_ma": ip, "paxis_pa": p} for ip, p in stages],
        "free_keys": args.free_keys,
        "fixed_by_omission": ["CS"],
        "target_geometry": {"R0": args.target_R0, "A": args.target_A, "kappa": args.target_kappa, "delta": args.target_delta},
        "acceptance": {"tol_R0": args.tol_R0, "tol_A": args.tol_A, "tol_kappa": args.tol_kappa, "tol_delta": args.tol_delta, "min_gap_m": args.min_gap_m},
    })

    print("\n" + "=" * 100)
    print("[STAR coupled Ip+paxis ramp]")
    print(f"outdir       = {outdir}")
    print(f"target       = {target}")
    print(f"seed         = {seed}")
    print(f"stages left  = {len(stages[start_stage_idx:])}")
    print(f"first stage  = {stages[start_stage_idx] if start_stage_idx < len(stages) else None}")
    print(f"last stage   = {stages[-1] if stages else None}")
    print(f"free_keys    = {args.free_keys}")
    print(f"accept       = physical sep + inside WALL_INNER + gap >= {args.min_gap_m} m + shape tolerances")
    print("=" * 100 + "\n")

    current_seed = seed
    for idx, (ip_ma, paxis_pa) in enumerate(stages[start_stage_idx:], start=start_stage_idx):
        label = _stage_label(ip_ma, paxis_pa)
        stage_dir = outdir / f"stage_{idx:03d}_{label}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "#" * 100)
        if bool(args.freeze_first_stage) and idx == 0 and not args.resume:
            current_seed = _accept_frozen_stage(
                outdir=outdir,
                stage_dir=stage_dir,
                current_seed=current_seed,
                label=label,
                idx=idx,
                ip_ma=ip_ma,
                paxis_pa=paxis_pa,
                target=target,
            )
            continue
        print(f"[STAGE {idx+1}/{len(stages)}] Ip={ip_ma:.4f} MA  paxis={paxis_pa:.4e} Pa")
        print(f"[SEED] {current_seed}")
        print("#" * 100 + "\n")

        env = _stage_env(args, ip_ma, paxis_pa)
        rec: Dict[str, Any] = {
            "stage_index": idx, "ip_ma": ip_ma, "paxis_pa": paxis_pa,
            "seed_in": str(current_seed), "target": str(target), "started": time.ctime(),
            "accepted": False, "rounds": [],
        }

        max_rounds = max(1, int(math.ceil(float(args.max_fit_iters_per_stage) / max(1, int(args.iters_per_round)))))
        candidate_seed = current_seed
        accepted = False
        last_best: Optional[Path] = None
        last_summary: Dict[str, Any] = {}

        for r in range(1, max_rounds + 1):
            best = _run_dshape_round(args, candidate_seed, target, ip_ma, paxis_pa, stage_dir, r, env)
            if best is None:
                rec["rounds"].append({"round": r, "status": "failed"})
                print("[WARN] round failed; stopping this stage.")
                break
            summary = _summarize_best(best)
            ok, accept_info = _stage_satisfied(summary, args)
            rec["rounds"].append({"round": r, "best": str(best), "summary": summary, "accept_info": accept_info, "accepted": ok})
            _write_json(stage_dir / "stage_record_partial.json", rec)
            print(
                f"[CHECK] round={r}/{max_rounds} accepted={ok} fail={accept_info.get('fail')} "
                f"sep={accept_info.get('has_sep')} in={accept_info.get('inside_WALL_INNER')} "
                f"outside_frac={accept_info.get('outside_frac')} "
                f"signed_gap={accept_info.get('gap')} "
                f"Rax={accept_info.get('Rax')} Zax={accept_info.get('Zax')} area={accept_info.get('area')} "
                f"R0={accept_info.get('R0')} A={accept_info.get('A')} "
                f"k={accept_info.get('kappa')} d={accept_info.get('delta')}"
            )
            last_best = best
            last_summary = summary

            gap_val = _safe_float(accept_info.get("gap", float("nan")), float("nan"))
            outside_val = _safe_float(accept_info.get("outside_frac", 1.0), 1.0)

            seed_feasible = bool(
                accept_info.get("has_sep", False)
                and accept_info.get("inside_WALL_INNER", False)
                and outside_val <= float(args.max_outside_frac)
                and gap_val == gap_val
                and gap_val >= float(args.min_gap_m)
                and accept_info.get("branch_term", 0.0) in (0, 0.0, None)
            )

            if seed_feasible:
                candidate_seed = best
            else:
                print(
                    "[SEED GUARD] best rejected as next seed: "
                    f"gap={gap_val}, outside_frac={outside_val}, fail={accept_info.get('fail')}"
                )

            if ok:
                accepted = True
                print(f"[STAGE ACCEPTED EARLY] round={r}")
                break

        if last_best is None:
            rec["status"] = "stage_failed_no_best"
            rec["ended"] = time.ctime()
            _append_manifest(outdir, rec)
            print("[STOP] no usable best generated; inspect logs.")
            break

        if not accepted:
            rec["status"] = "max_rounds_reached_not_accepted"
            rec["ended"] = time.ctime()
            rec["last_best"] = str(last_best)
            rec["last_summary"] = last_summary
            _append_manifest(outdir, rec)
            _write_json(stage_dir / "stage_record.json", rec)

            print("\n" + "!" * 100)
            print(f"[STOP] Stage not accepted: {label}")
            print(f"[LAST BEST] {last_best}")
            print("[LAST SUMMARY]", json.dumps(last_summary, indent=2, ensure_ascii=False)[:2500])
            print("Reason: refusing to carry a non-accepted branch forward.")
            print("!" * 100 + "\n")
            break

        canonical = outdir / f"{label}_best.json"
        shutil.copy2(last_best, canonical)
        rec["best_json"] = str(canonical)
        rec["summary"] = last_summary
        rec["accepted"] = True
        rec["status"] = "accepted"
        rec["ended"] = time.ctime()
        _append_manifest(outdir, rec)
        _write_json(stage_dir / "stage_record.json", rec)

        print("\n" + "-" * 100)
        print(f"[ACCEPTED] {label}")
        print(f"[BEST] {canonical}")
        print("[SUMMARY]", json.dumps(last_summary, indent=2, ensure_ascii=False)[:2500])
        print("-" * 100 + "\n")
        current_seed = canonical

    print("\n[DONE]")
    print(f"Run directory: {outdir}")
    print(f"Manifest: {outdir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
