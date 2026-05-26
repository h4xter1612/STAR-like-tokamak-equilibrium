#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
continue_ip_star_incremental.py

Incremental Ip ramp-up for the corrected STAR-like equilibrium branch.

Overnight version: fixed CS and CS_END; only CS_MID + PF1..PF6 are scanned.
The script performs a small local current scan/refine at each Ip step, saves
the best current set with a unique name, and uses that best seed as the initial
condition for the next Ip step.

Default strategy:
- passive structures OFF for speed
- fixed CS and CS_END by default
- 5-iteration dshape-first scan every stage
- optional divertor/leg polish every N stages
- resume support through manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
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

DIV_BEST = RESULTS / "fit_simplified_dn_divertor_first_best.json"
DIV_JSONL = RESULTS / "fit_simplified_dn_divertor_first_results.jsonl"


def _as_path(p: str | Path) -> Path:
    p = Path(p)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ip_label(ip_ma: float) -> str:
    return f"{ip_ma:.2f}".replace(".", "p")


def _parse_stages(args: argparse.Namespace) -> List[float]:
    if args.ip_stages_ma:
        return [float(x) for x in re.split(r"[,\s]+", args.ip_stages_ma.strip()) if x]

    start = float(args.ip_start_ma)
    end = float(args.ip_end_ma)
    step = float(args.ip_step_ma)
    if step <= 0:
        raise ValueError("--ip-step-ma must be positive")

    stages: List[float] = []
    x = start
    while x <= end + 1e-9:
        stages.append(round(x, 10))
        x += step
    if stages and stages[-1] < end - 1e-9:
        stages.append(end)
    return stages


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _json_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False}

    try:
        d = _load_json(path)
    except Exception as e:
        return {"exists": True, "read_error": repr(e)}

    out: Dict[str, Any] = {"exists": True}

    for key in ("best_currents_MA", "currents_MA"):
        if isinstance(d.get(key), dict):
            out["currents_MA"] = d[key]
            break

    br = d.get("best_result") if isinstance(d.get("best_result"), dict) else d

    for k in (
        "score", "ok", "ok_sep", "sep", "has_separatrix", "has_true_separatrix",
        "R0", "R0_m", "A", "kappa", "delta", "delta_bar", "delta_u", "delta_l",
        "min_wall_gap_m", "min_abs_distance_to_WALL_INNER_m",
        "inside_WALL_INNER", "n_xpoints", "nxp",
    ):
        if isinstance(br, dict) and k in br:
            out[k] = br[k]

    pd = br.get("plasma_diag") if isinstance(br, dict) else None
    if isinstance(pd, dict):
        for k in ("R0", "A", "kappa", "delta_u", "delta_l", "Rmin", "Rmax", "Zmin", "Zmax"):
            if k in pd and k not in out:
                out[k] = pd[k]

    return out


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _run(cmd: List[str], env: Dict[str, str], log_path: Path, timeout_s: Optional[float] = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        log.write("\n" + "=" * 100 + "\n")
        log.write("[RUN] " + " ".join(cmd) + "\n")
        log.write("=" * 100 + "\n")
        log.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
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


def _stage_env(args: argparse.Namespace, ip_ma: float) -> Dict[str, str]:
    env = os.environ.copy()
    env["STAR_IP_A"] = f"{ip_ma * 1e6:.9g}"
    env["STAR_PAXIS_PA"] = f"{float(args.paxis_pa):.9g}"

    if not args.keep_passives:
        env["STAR_PASSIVE_STRUCTURES"] = "0"
        env["STAR_PASSIVE_USE_STAR_VESSEL"] = "0"
        env["STAR_PLOT_PASSIVES"] = "0"
        env["STAR_EQ_DOMAIN_SOURCE"] = "outer"

    if args.nx:
        env["STAR_NX_EQ"] = str(int(args.nx))
    if args.ny:
        env["STAR_NY_EQ"] = str(int(args.ny))

    return env


def _find_latest_success(outdir: Path) -> Optional[Tuple[int, float, Path]]:
    manifest = outdir / "manifest.jsonl"
    if not manifest.exists():
        return None

    latest: Optional[Tuple[int, float, Path]] = None
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not rec.get("accepted", False):
                continue
            seed = Path(rec.get("best_json", ""))
            if not seed.exists():
                continue
            idx = int(rec.get("stage_index", -1))
            ip = float(rec.get("ip_ma", float("nan")))
            if latest is None or idx > latest[0]:
                latest = (idx, ip, seed)
    return latest


def _append_manifest(outdir: Path, rec: Dict[str, Any]) -> None:
    with open(outdir / "manifest.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _run_dshape(args: argparse.Namespace, seed: Path, target: Path, ip_ma: float, stage_dir: Path, env: Dict[str, str]) -> Optional[Path]:
    label = _ip_label(ip_ma)
    log = stage_dir / f"ip_{label}_dshape.log"

    cmd = [
        sys.executable, "-u", ".\\fit_simplified_dn_dshape_first.py",
        "--target", str(target),
        "--seed", str(seed),
        "--stage", "release-cs",
        "--iters", str(int(args.iters)),
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
        "--delta-target", str(float(args.delta_target)),
        "--kappa-min", str(float(args.kappa_min)),
        "--A-min", str(float(args.A_min)),
        "--chamfer-max-m", str(float(args.chamfer_max_m)),
    ]

    rc = _run(cmd, env=env, log_path=log, timeout_s=float(args.command_timeout))
    if rc != 0:
        print(f"[WARN] dshape command failed rc={rc}")
        return None

    dst = stage_dir / f"ip_{label}_dshape_best.json"
    if not _copy_if_exists(DSHAPE_BEST, dst):
        print(f"[WARN] expected best not found: {DSHAPE_BEST}")
        return None

    _copy_if_exists(DSHAPE_JSONL, stage_dir / f"ip_{label}_dshape_results.jsonl")
    print(f"[SAVED] {dst}")
    return dst


def _run_divertor(args: argparse.Namespace, seed: Path, target: Path, ip_ma: float, stage_dir: Path, env: Dict[str, str]) -> Optional[Path]:
    label = _ip_label(ip_ma)
    log = stage_dir / f"ip_{label}_divertor.log"

    cmd = [
        sys.executable, "-u", ".\\fit_simplified_dn_divertor_first.py",
        "--target", str(target),
        "--seed", str(seed),
        "--stage", "release-cs",
        "--iters", str(int(args.div_iters)),
        "--pop", str(int(args.div_pop)),
        "--workers", str(int(args.workers)),
        "--timeout", str(float(args.eval_timeout)),
        "--widen", str(float(args.div_widen)),
        "--sigma-frac", str(float(args.div_sigma_frac)),
        "--free-keys", str(args.free_keys),
        "--dxf", str(args.dxf),
        "--strike-in-weight", str(float(args.strike_in_weight)),
        "--strike-out-weight", str(float(args.strike_out_weight)),
        "--strike-max-weight", str(float(args.strike_max_weight)),
        "--leg-weight", str(float(args.leg_weight)),
        "--leg-sigma", str(float(args.leg_sigma)),
        "--shape-weight", str(float(args.shape_weight)),
        "--min-A-soft", str(float(args.min_A_soft)),
        "--min-kappa-soft", str(float(args.min_kappa_soft)),
        "--max-chamfer-soft", str(float(args.max_chamfer_soft)),
        "--current-reg-weight", str(float(args.current_reg_weight)),
    ]

    if args.hard_wall_gate:
        cmd.extend(["--hard-wall-gate", "--hard-wall-chamfer", str(float(args.hard_wall_chamfer))])

    rc = _run(cmd, env=env, log_path=log, timeout_s=float(args.command_timeout))
    if rc != 0:
        print(f"[WARN] divertor command failed rc={rc}")
        return None

    dst = stage_dir / f"ip_{label}_divertor_best.json"
    if not _copy_if_exists(DIV_BEST, dst):
        print(f"[WARN] expected best not found: {DIV_BEST}")
        return None

    _copy_if_exists(DIV_JSONL, stage_dir / f"ip_{label}_divertor_results.jsonl")
    print(f"[SAVED] {dst}")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description="Incremental STAR Ip ramp-up with small local current scans.")

    ap.add_argument("--seed", type=str, required=False, help="Initial seed JSON.")
    ap.add_argument("--target", type=str, required=True, help="Updated branch target JSON.")
    ap.add_argument("--dxf", type=str, default=".\\star_baseline.dxf", help="DXF path passed to fit scripts.")
    ap.add_argument("--outdir", type=str, default="", help="Output directory. If omitted, creates timestamped run dir.")
    ap.add_argument("--resume", action="store_true", help="Resume from latest accepted stage in --outdir.")
    ap.add_argument("--restart", action="store_true", help="Ignore previous manifest and restart.")

    ap.add_argument("--ip-stages-ma", type=str, default="", help="Comma-separated Ip stages in MA.")
    ap.add_argument("--ip-start-ma", type=float, default=4.2)
    ap.add_argument("--ip-end-ma", type=float, default=13.2)
    ap.add_argument("--ip-step-ma", type=float, default=0.2)
    ap.add_argument("--paxis-pa", type=float, default=2.0e3)

    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--eval-timeout", type=float, default=120.0)
    ap.add_argument("--command-timeout", type=float, default=3600.0)

    ap.add_argument("--widen", type=float, default=0.006)
    ap.add_argument("--sigma-frac", type=float, default=0.012)
    ap.add_argument("--free-keys", type=str, default="CS_MID,CS_END,PF1,PF2,PF3,PF4,PF5,PF6", help="Default fixes CS; scans only CS_MID + PFs.")

    ap.add_argument("--dshape-weight", type=float, default=0.75)
    ap.add_argument("--inboard-weight", type=float, default=55.0)
    ap.add_argument("--target-rms-weight", type=float, default=0.0)
    ap.add_argument("--delta-weight", type=float, default=14.0)
    ap.add_argument("--delta-target", type=float, default=0.4695)
    ap.add_argument("--kappa-min", type=float, default=2.04)
    ap.add_argument("--A-min", type=float, default=1.90)
    ap.add_argument("--chamfer-max-m", type=float, default=0.32)

    ap.add_argument("--skip-divertor", action="store_true")
    ap.add_argument("--div-every", type=int, default=4, help="Run divertor polish every N stages. 1 = every stage. Default 4 for overnight speed.")
    ap.add_argument("--div-iters", type=int, default=4)
    ap.add_argument("--div-pop", type=int, default=10)
    ap.add_argument("--div-widen", type=float, default=0.005)
    ap.add_argument("--div-sigma-frac", type=float, default=0.010)
    ap.add_argument("--strike-in-weight", type=float, default=4.0)
    ap.add_argument("--strike-out-weight", type=float, default=0.35)
    ap.add_argument("--strike-max-weight", type=float, default=4.0)
    ap.add_argument("--leg-weight", type=float, default=1.2)
    ap.add_argument("--leg-sigma", type=float, default=0.32)
    ap.add_argument("--shape-weight", type=float, default=0.80)
    ap.add_argument("--min-A-soft", type=float, default=1.88)
    ap.add_argument("--min-kappa-soft", type=float, default=2.02)
    ap.add_argument("--max-chamfer-soft", type=float, default=0.36)
    ap.add_argument("--current-reg-weight", type=float, default=0.25)
    ap.add_argument("--hard-wall-gate", action="store_true", default=True)
    ap.add_argument("--hard-wall-chamfer", type=float, default=0.055)

    ap.add_argument("--keep-passives", action="store_true", help="Do not force passive structures OFF.")
    ap.add_argument("--nx", type=int, default=0, help="Optional STAR_NX_EQ override.")
    ap.add_argument("--ny", type=int, default=0, help="Optional STAR_NY_EQ override.")

    args = ap.parse_args()

    target = _as_path(args.target)
    if not target.exists():
        raise FileNotFoundError(f"Target not found: {target}")

    stages = _parse_stages(args)

    if args.outdir:
        outdir = _as_path(args.outdir)
    else:
        outdir = RESULTS / f"ramp_ip_incremental_fixed_CS_CSEND_{_now_tag()}"
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
            idx, ip, seed_path = latest
            seed = seed_path
            start_stage_idx = idx + 1
            print(f"[RESUME] latest accepted: stage={idx} Ip={ip:.3f} MA seed={seed}")
        else:
            print("[RESUME] no accepted stage found; starting from --seed")

    if seed is None:
        if not args.seed:
            raise ValueError("--seed is required unless --resume finds a previous successful stage.")
        seed = _as_path(args.seed)
        if not seed.exists():
            raise FileNotFoundError(f"Seed not found: {seed}")

    _write_json(outdir / "run_config.json", {
        "created": time.ctime(),
        "seed_initial": str(seed),
        "target": str(target),
        "dxf": str(args.dxf),
        "ip_stages_ma": stages,
        "paxis_pa": args.paxis_pa,
        "workers": args.workers,
        "iters": args.iters,
        "pop": args.pop,
        "skip_divertor": args.skip_divertor,
        "div_every": args.div_every,
        "keep_passives": args.keep_passives,
        "free_keys": args.free_keys,
        "fixed_by_omission": ["CS"],
    })

    print("\n" + "=" * 100)
    print("[STAR Ip incremental ramp]")
    print(f"outdir  = {outdir}")
    print(f"target  = {target}")
    print(f"seed    = {seed}")
    print(f"stages  = {stages[start_stage_idx:]}")
    print(f"free_keys = {args.free_keys}")
    if "CS_END" not in args.free_keys:
        print("[INFO] CS and CS_END fixed; scanning CS_MID + selected PF families.")
    print("=" * 100 + "\n")

    current_seed = seed

    for idx, ip_ma in enumerate(stages[start_stage_idx:], start=start_stage_idx):
        label = _ip_label(ip_ma)
        stage_dir = outdir / f"stage_{idx:03d}_Ip_{label}MA"
        stage_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "#" * 100)
        print(f"[STAGE {idx+1}/{len(stages)}] Ip={ip_ma:.3f} MA  paxis={args.paxis_pa:.3e} Pa")
        print(f"[SEED] {current_seed}")
        print("#" * 100 + "\n")

        env = _stage_env(args, ip_ma)

        rec: Dict[str, Any] = {
            "stage_index": idx,
            "ip_ma": ip_ma,
            "paxis_pa": args.paxis_pa,
            "seed_in": str(current_seed),
            "target": str(target),
            "started": time.ctime(),
            "accepted": False,
        }

        dshape_best = _run_dshape(args, current_seed, target, ip_ma, stage_dir, env)
        if dshape_best is None:
            rec["status"] = "dshape_failed"
            rec["ended"] = time.ctime()
            _append_manifest(outdir, rec)
            print("[STOP] dshape failed; stopping ramp so you can inspect/backtrack.")
            break

        rec["dshape_best"] = str(dshape_best)
        rec["dshape_summary"] = _json_summary(dshape_best)

        candidate_seed = dshape_best

        do_div = (not args.skip_divertor) and (args.div_every > 0) and ((idx - start_stage_idx) % args.div_every == 0)
        if do_div:
            div_best = _run_divertor(args, dshape_best, target, ip_ma, stage_dir, env)
            if div_best is not None:
                candidate_seed = div_best
                rec["divertor_best"] = str(div_best)
                rec["divertor_summary"] = _json_summary(div_best)
            else:
                print("[WARN] divertor polish failed; accepting dshape best and continuing.")
                rec["divertor_failed_but_accepted_dshape"] = True

        canonical = outdir / f"ip_{label}MA_best.json"
        shutil.copy2(candidate_seed, canonical)

        rec["best_json"] = str(canonical)
        rec["summary"] = _json_summary(canonical)
        rec["accepted"] = True
        rec["status"] = "accepted"
        rec["ended"] = time.ctime()
        _append_manifest(outdir, rec)
        _write_json(stage_dir / "stage_record.json", rec)

        print(f"\n[ACCEPTED] Ip={ip_ma:.3f} MA")
        print(f"[BEST] {canonical}")
        print("[SUMMARY]", json.dumps(rec["summary"], indent=2, ensure_ascii=False)[:2500])

        current_seed = canonical

    print("\n[DONE]")
    print(f"Run directory: {outdir}")
    print(f"Manifest: {outdir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
