#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
continue_ip_star_shape_locked_v3.py

Ip ramp-up driver for the corrected STAR-like CAD/free-boundary pipeline.

Purpose
-------
Ramp Ip from the current successful low-pressure branch to STAR-like Ip while
preserving the *current corrected branch*, not the old simplified/CAD target:

    R0 ~ 4.2 m, A ~ 2.0, kappa ~ 2.15-2.2, delta ~ 0.47,
    LCFS inside WALL_INNER with finite gap, and divertor legs connected.

Important design choice
-----------------------
The old results/targets/star_simplified_dn_target.json had scalar targets such as
R0=4.0, kappa=2.23, delta=0.62. This wrapper DOES NOT use those old shape scalars
as the main objective. It creates a patched runtime target JSON that keeps useful
marker/window information from the old target, but replaces scalar targets with
the current branch targets and suppresses old-boundary RMS as a strong term.

Wrapped scripts
---------------
  fit_simplified_dn_dshape_first.py
  fit_simplified_dn_divertor_first.py

Required config patch
---------------------
Your config_star_bean.py must allow Ip/paxis overrides via environment variables.
Add this after the normal Ip_A and paxis definitions:

    import os
    Ip_A = float(os.environ.get("STAR_IP_A", Ip_A))
    paxis = float(os.environ.get("STAR_PAXIS_PA", paxis))

If your variable names are different, adapt those two lines accordingly.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_IP_STAGES_MA = [4.0, 5.0, 6.5, 8.0, 10.0, 11.5, 13.2]
DEFAULT_PAXIS_PA = 2.0e3

# Current successful branch, not the old target.
DEFAULT_TARGET_R0_M = 4.2083
DEFAULT_TARGET_A = 2.0275
DEFAULT_TARGET_KAPPA = 2.1636
DEFAULT_TARGET_DELTA = 0.4695
DEFAULT_MIN_WALL_GAP_M = 0.075  # current baseline ~0.0978 m; keep a margin but allow numerical motion


def _now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _as_path(p: str | Path, cwd: Path) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (cwd / q).resolve()


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> int:
    print("\n[RUN]", " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write("[RUN] " + " ".join(cmd) + "\n")
        f.write("=" * 100 + "\n")
        f.flush()
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        f.write(f"\n[EXIT_CODE] {p.returncode}\n")
    return int(p.returncode)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[COPIED] {src} -> {dst}")
        return True
    print(f"[WARN] missing file: {src}")
    return False


def _find_best_json(results_dir: Path, preferred: Iterable[str]) -> Optional[Path]:
    for name in preferred:
        p = results_dir / name
        if p.exists():
            return p
    candidates = sorted(results_dir.glob("*best*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _make_runtime_target(
    original_target: Path,
    outdir: Path,
    *,
    R0: float,
    A: float,
    kappa: float,
    delta: float,
    min_wall_gap_m: float,
) -> Path:
    """Create a patched target JSON for this ramp.

    It keeps useful CAD marker/windows from the old file but replaces scalar targets
    that would otherwise pull the optimizer back to the obsolete target.
    """
    if original_target.exists():
        with original_target.open("r", encoding="utf-8") as f:
            target = json.load(f)
    else:
        target = {}

    a = R0 / A if A else 2.0
    target["description"] = (
        "Runtime branch target for Ip ramp-up. Scalars updated from the corrected "
        "STAR-like baseline; old simplified CAD plasma boundary is not used as a strong shape target."
    )
    target["scalars_target"] = {
        "R0": float(R0),
        "Z0": 0.0,
        "A": float(A),
        "a": float(a),
        "kappa": float(kappa),
        "delta_u": float(delta),
        "delta_l": float(delta),
        "delta_bar": float(delta),
        "vertical_half_height": float(kappa * a),
        "min_wall_gap_m": float(min_wall_gap_m),
    }

    # Do not let target-level default weights revive the old CAD target boundary.
    ow = target.get("objective_weights", {}) if isinstance(target.get("objective_weights", {}), dict) else {}
    ow.update({
        "target_rms": 0.0,
        "shape_rms": 0.0,
        "boundary_rms": 0.0,
        "cad_target_rms": 0.0,
        "legacy_boundary_rms": 0.0,
        "wall_containment": max(float(ow.get("wall_containment", 1.0) or 1.0), 10.0),
        "leg": max(float(ow.get("leg", 1.0) or 1.0), 3.0),
        "strike": max(float(ow.get("strike", 1.0) or 1.0), 3.0),
    })
    target["objective_weights"] = ow

    tol = target.get("shape_tolerances", {}) if isinstance(target.get("shape_tolerances", {}), dict) else {}
    tol.update({
        "R0_sigma_m": 0.20,
        "A_sigma": 0.12,
        "kappa_sigma": 0.15,
        "delta_sigma": 0.08,
        "min_wall_gap_m": float(min_wall_gap_m),
    })
    target["shape_tolerances"] = tol

    prov = target.get("provenance", {}) if isinstance(target.get("provenance", {}), dict) else {}
    prov.update({
        "runtime_generated_by": "continue_ip_star_shape_locked_v3.py",
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "source_target": str(original_target),
        "note": "Old target marker windows may be retained, but scalar shape targets are branch-locked to the corrected baseline.",
    })
    target["provenance"] = prov

    out = outdir / "runtime_branch_target_A2p03_k2p16_d0p47.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(target, f, indent=2)
    print(f"[TARGET] runtime branch target written: {out}")
    return out


def _write_stage_manifest(
    outdir: Path,
    *,
    stage_index: int,
    ip_ma: float,
    paxis_pa: float,
    seed_in: Path,
    seed_out: Optional[Path],
    dshape_exit: int,
    divertor_exit: int,
    target_R0: float,
    target_A: float,
    target_kappa: float,
    target_delta: float,
    min_wall_gap_m: float,
) -> None:
    manifest = {
        "stage_index": stage_index,
        "Ip_MA": ip_ma,
        "Ip_A": ip_ma * 1.0e6,
        "paxis_Pa": paxis_pa,
        "seed_in": str(seed_in),
        "seed_out": str(seed_out) if seed_out else None,
        "dshape_exit_code": dshape_exit,
        "divertor_exit_code": divertor_exit,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "shape_targets": {
            "R0_m": target_R0,
            "A": target_A,
            "kappa": target_kappa,
            "delta": target_delta,
            "min_wall_gap_m": min_wall_gap_m,
        },
        "priority": [
            "preserve corrected STAR-like branch, not old CAD target",
            "LCFS inside WALL_INNER with finite gap",
            "divertor legs connected to divertor regions",
            "A around 2 and kappa around 2.16-2.2",
            "delta around current successful branch",
            "do not optimize q aggressively during Ip ramp",
        ],
    }
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / f"stage_{stage_index:02d}_{ip_ma:.2f}MA_manifest.json".replace(".", "p")).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=r".\results\targets\star_branch_A2p03_k2p16_d0p47_target.json",
                    help="Updated branch target JSON. If missing, the old simplified target can still be supplied; scalar targets are patched at runtime.")
    ap.add_argument("--seed", default=r".\results\seeds\star_A2p03_k2p16_d0p47_baseline_seed.json", help="Initial successful baseline seed JSON. Default uses the corrected baseline seed.")
    ap.add_argument("--dxf", default=r".\star_baseline.dxf")
    ap.add_argument("--outdir", default=None,
                    help="Output directory. Default: results/ramp_ip_shape_locked_<timestamp>")
    ap.add_argument("--ip-stages-ma", default=",".join(str(x) for x in DEFAULT_IP_STAGES_MA),
                    help="Comma-separated Ip stages in MA, e.g. 4,5,6.5,8,10,11.5,13.2")
    ap.add_argument("--paxis-pa", type=float, default=DEFAULT_PAXIS_PA,
                    help="Fixed low pressure during Ip ramp. Default: 2e3 Pa.")
    ap.add_argument("--target-R0", type=float, default=DEFAULT_TARGET_R0_M)
    ap.add_argument("--target-A", type=float, default=DEFAULT_TARGET_A)
    ap.add_argument("--target-kappa", type=float, default=DEFAULT_TARGET_KAPPA)
    ap.add_argument("--target-delta", type=float, default=DEFAULT_TARGET_DELTA)
    ap.add_argument("--min-wall-gap", type=float, default=DEFAULT_MIN_WALL_GAP_M,
                    help="Desired hard-wall chamfer/gap in m. Baseline was about 0.0978 m; default 0.075 m.")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--free-keys", default="CS_MID,CS_END,PF1,PF2,PF3,PF4,PF5,PF6")
    ap.add_argument("--skip-divertor-polish", action="store_true",
                    help="Only run D-shape stage at each Ip. Not recommended.")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    cwd = Path.cwd()
    results_dir = cwd / "results"
    seed = _as_path(args.seed, cwd)
    if not seed.exists():
        raise FileNotFoundError(f"Initial seed not found: {seed}")

    original_target = _as_path(args.target, cwd)
    ip_stages = [float(x.strip()) for x in args.ip_stages_ma.split(",") if x.strip()]
    if not ip_stages:
        raise ValueError("No Ip stages provided.")

    outdir = Path(args.outdir) if args.outdir else (results_dir / f"ramp_ip_shape_locked_{_now_tag()}")
    if not outdir.is_absolute():
        outdir = (cwd / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    runtime_target = _make_runtime_target(
        original_target,
        outdir,
        R0=args.target_R0,
        A=args.target_A,
        kappa=args.target_kappa,
        delta=args.target_delta,
        min_wall_gap_m=args.min_wall_gap,
    )

    with (outdir / "ramp_plan.json").open("w", encoding="utf-8") as f:
        json.dump({
            "ip_stages_MA": ip_stages,
            "paxis_Pa": args.paxis_pa,
            "initial_seed": str(seed),
            "original_target": str(original_target),
            "runtime_target": str(runtime_target),
            "dxf": str(args.dxf),
            "free_keys": args.free_keys,
            "workers": args.workers,
            "timeout": args.timeout,
            "shape_targets": {
                "R0_m": args.target_R0,
                "A": args.target_A,
                "kappa": args.target_kappa,
                "delta": args.target_delta,
                "min_wall_gap_m": args.min_wall_gap,
            },
            "notes": [
                "Ip ramp only; keep paxis low.",
                "Runtime target patches obsolete scalar target values from the old CAD/Miller target.",
                "Divertor leg quality and WALL_INNER confinement are prioritized over q during this phase.",
                "Old target boundary RMS is intentionally suppressed.",
            ],
        }, f, indent=2)

    current_seed = seed

    for i, ip_ma in enumerate(ip_stages, start=1):
        stage_tag = f"stage_{i:02d}_{ip_ma:.2f}MA".replace(".", "p")
        stage_dir = outdir / stage_tag
        stage_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["STAR_IP_A"] = f"{ip_ma * 1.0e6:.12g}"
        env["STAR_PAXIS_PA"] = f"{args.paxis_pa:.12g}"

        log_path = stage_dir / f"{stage_tag}.log"

        print("\n" + "#" * 100)
        print(f"[STAGE {i}/{len(ip_stages)}] Ip = {ip_ma:.3f} MA, paxis = {args.paxis_pa:.3e} Pa")
        print(f"[SEED] {current_seed}")
        print(f"[TARGET] {runtime_target}")
        print("#" * 100)

        # Shape preservation step. Do not use old CAD target RMS as strong objective.
        dshape_cmd = [
            args.python, r".\fit_simplified_dn_dshape_first.py",
            "--target", str(runtime_target),
            "--seed", str(current_seed),
            "--stage", "release-cs",
            "--iters", "14",
            "--pop", "24",
            "--workers", str(args.workers),
            "--timeout", str(args.timeout),
            "--widen", "0.016",
            "--sigma-frac", "0.026",
            "--free-keys", args.free_keys,
            "--no-early-stop",
            "--dxf", args.dxf,
            "--dshape-weight", "1.00",
            "--inboard-weight", "80",
            "--target-rms-weight", "0.0",
            "--delta-weight", "26",
            "--delta-target", f"{args.target_delta:.4f}",
            "--kappa-min", f"{max(2.05, args.target_kappa - 0.08):.3f}",
            "--A-min", "1.88",
            "--chamfer-max-m", "0.30",
        ]
        dshape_exit = _run(dshape_cmd, cwd=cwd, env=env, log_path=log_path)

        dshape_best = _find_best_json(results_dir, [
            "fit_simplified_dn_dshape_first_best.json",
            "fit_simplified_dn_toposafe_best.json",
        ])
        dshape_saved = None
        if dshape_best:
            dshape_saved = stage_dir / f"{stage_tag}_A_dshape_best.json"
            _copy_if_exists(dshape_best, dshape_saved)

        polish_seed = dshape_saved if dshape_saved and dshape_saved.exists() else current_seed

        # Divertor-leg/containment step. X-points are weak; legs and first-wall confinement dominate.
        divertor_exit = 0
        divertor_saved = None
        if not args.skip_divertor_polish:
            divertor_cmd = [
                args.python, r".\fit_simplified_dn_divertor_first.py",
                "--target", str(runtime_target),
                "--seed", str(polish_seed),
                "--stage", "release-cs",
                "--iters", "10",
                "--pop", "22",
                "--workers", str(args.workers),
                "--timeout", str(args.timeout),
                "--widen", "0.012",
                "--sigma-frac", "0.020",
                "--free-keys", args.free_keys,
                "--dxf", args.dxf,
                "--strike-in-weight", "6.0",
                "--strike-out-weight", "3.0",
                "--strike-max-weight", "8.0",
                "--leg-weight", "5.0",
                "--leg-sigma", "0.28",
                "--xpt-weight", "0.25",
                "--xpt-sigma", "0.60",
                "--shape-weight", "0.35",
                "--min-A-soft", "1.88",
                "--min-kappa-soft", f"{max(2.03, args.target_kappa - 0.10):.3f}",
                "--max-chamfer-soft", "0.30",
                "--hard-wall-gate",
                "--hard-wall-chamfer", f"{args.min_wall_gap:.4f}",
                "--penalty-hard-wall", "1.0e9",
                "--penalty-no-sep", "1.0e10",
                "--current-reg-weight", "0.18",
            ]
            divertor_exit = _run(divertor_cmd, cwd=cwd, env=env, log_path=log_path)

            divertor_best = _find_best_json(results_dir, [
                "fit_simplified_dn_divertor_first_best.json",
                "fit_simplified_dn_toposafe_best.json",
                "fit_simplified_dn_dshape_first_best.json",
            ])
            if divertor_best:
                divertor_saved = stage_dir / f"{stage_tag}_B_divertor_best.json"
                _copy_if_exists(divertor_best, divertor_saved)

        next_seed = divertor_saved if divertor_saved and divertor_saved.exists() else dshape_saved
        if next_seed is None or not next_seed.exists():
            print(f"[STOP] No successful best JSON found at Ip={ip_ma:.3f} MA.")
            _write_stage_manifest(
                stage_dir,
                stage_index=i,
                ip_ma=ip_ma,
                paxis_pa=args.paxis_pa,
                seed_in=current_seed,
                seed_out=None,
                dshape_exit=dshape_exit,
                divertor_exit=divertor_exit,
                target_R0=args.target_R0,
                target_A=args.target_A,
                target_kappa=args.target_kappa,
                target_delta=args.target_delta,
                min_wall_gap_m=args.min_wall_gap,
            )
            break

        final_stage_seed = outdir / f"ip_{ip_ma:.2f}MA_best.json".replace(".", "p")
        _copy_if_exists(next_seed, final_stage_seed)

        for fig_name in ["STAR_bean_equilibrium.png", "STAR_machine_setup.png"]:
            src = results_dir / fig_name
            if src.exists():
                _copy_if_exists(src, stage_dir / f"{stage_tag}_{fig_name}")

        _write_stage_manifest(
            stage_dir,
            stage_index=i,
            ip_ma=ip_ma,
            paxis_pa=args.paxis_pa,
            seed_in=current_seed,
            seed_out=final_stage_seed,
            dshape_exit=dshape_exit,
            divertor_exit=divertor_exit,
            target_R0=args.target_R0,
            target_A=args.target_A,
            target_kappa=args.target_kappa,
            target_delta=args.target_delta,
            min_wall_gap_m=args.min_wall_gap,
        )

        current_seed = final_stage_seed

    print("\n[DONE]")
    print(f"Ramp outputs saved in:\n  {outdir}")
    print("\nRecommended next checks:")
    print("  1) Rebuild/plot each ip_*MA_best.json using your best-case plotter.")
    print("  2) Review logs for WALL_INNER gap, legs, A/kappa/delta, and current utilization.")
    print("  3) If a jump fails, rerun with denser --ip-stages-ma.")


if __name__ == "__main__":
    main()
