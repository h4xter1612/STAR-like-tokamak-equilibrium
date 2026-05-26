#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_star_branch_target.py

Create an updated branch-locked target JSON for the corrected STAR-like baseline.
It keeps useful marker/windows from the old target JSON, but replaces obsolete
shape targets (e.g. delta=0.62, xpoints at R=2.76) with the current baseline:
R0≈4.208 m, A≈2.03, kappa≈2.16, delta≈0.47.

Usage from pyscripts:
  py .\make_star_branch_target.py --old-target .\results\targets\star_simplified_dn_target.json --seed .\results\seeds\star_A2p03_k2p16_d0p47_baseline_seed.json --out .\results\targets\star_branch_A2p03_k2p16_d0p47_target.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


def _load_json(path: Path, default):
    if path and path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _seed_geometry(seed: dict) -> dict:
    g = seed.get("geometry_diagnostics", {}) or seed.get("best_result", {}).get("score_info", {}) or {}
    return {
        "R0": float(g.get("R0_m", g.get("R0", 4.2083))),
        "Z0": 0.0,
        "A": float(g.get("A", 2.0275)),
        "a": float(g.get("a_m", g.get("a", 2.0756))),
        "kappa": float(g.get("kappa", 2.1636)),
        "delta_u": float(g.get("delta_u", 0.4695)),
        "delta_l": float(g.get("delta_l", 0.4695)),
        "delta_bar": float(g.get("delta_bar", 0.4695)),
        "min_wall_gap_m": float(g.get("min_abs_distance_to_WALL_INNER_m", 0.0978)),
    }


def make_target(old_target: Path, seed_path: Path, out_path: Path, min_wall_gap: float | None = None) -> Path:
    target = _load_json(old_target, {})
    seed = _load_json(seed_path, {})
    geom = _seed_geometry(seed)
    if min_wall_gap is not None:
        geom["min_wall_gap_m"] = float(min_wall_gap)

    R0 = geom["R0"]
    A = geom["A"]
    a = geom["a"] if geom["a"] > 0 else R0 / A
    kappa = geom["kappa"]
    delta = geom["delta_bar"]

    target["schema"] = "star_branch_target.v1"
    target["description"] = (
        "Updated target for the corrected STAR-like branch. This target replaces the old "
        "simplified/Miller-like plasma target and is intended for Ip ramp-up around the "
        "validated low-pressure DN branch."
    )
    target["scalars_target"] = {
        "R0": R0,
        "Z0": 0.0,
        "A": A,
        "a": a,
        "kappa": kappa,
        "delta_u": geom["delta_u"],
        "delta_l": geom["delta_l"],
        "delta_bar": delta,
        "vertical_half_height": kappa * a,
        "min_wall_gap_m": geom["min_wall_gap_m"],
    }

    # Keep old marker windows if present, but demote old xpoint targets.
    target["topology_requirements"] = {
        **(target.get("topology_requirements", {}) if isinstance(target.get("topology_requirements", {}), dict) else {}),
        "require_lcfs": True,
        "require_wall_inner_containment": True,
        "prefer_double_null": True,
        "prioritize_divertor_legs_over_exact_xpoint_position": True,
        "do_not_use_old_miller_boundary_as_primary_target": True,
    }

    target["shape_tolerances"] = {
        "R0_sigma_m": 0.20,
        "A_sigma": 0.12,
        "kappa_sigma": 0.15,
        "delta_sigma": 0.08,
        "min_wall_gap_m": max(0.06, 0.75 * geom["min_wall_gap_m"]),
        "preferred_wall_gap_m": geom["min_wall_gap_m"],
    }

    ow = target.get("objective_weights", {}) if isinstance(target.get("objective_weights", {}), dict) else {}
    ow.update({
        "target_rms": 0.0,
        "shape_rms": 0.0,
        "boundary_rms": 0.0,
        "cad_target_rms": 0.0,
        "legacy_boundary_rms": 0.0,
        "wall_containment": max(float(ow.get("wall_containment", 1.0) or 1.0), 10.0),
        "leg": max(float(ow.get("leg", 1.0) or 1.0), 5.0),
        "strike": max(float(ow.get("strike", 1.0) or 1.0), 5.0),
        "xpoint_exact": min(float(ow.get("xpoint_exact", 0.25) or 0.25), 0.25),
    })
    target["objective_weights"] = ow

    target["branch_reference"] = {
        "seed_path": str(seed_path),
        "baseline_currents_A": seed.get("best_currents_A", {}),
        "baseline_geometry": geom,
        "reference_policy": "Preserve corrected branch shape and containment; do not pull toward old delta=0.62 target.",
    }
    target["provenance"] = {
        **(target.get("provenance", {}) if isinstance(target.get("provenance", {}), dict) else {}),
        "generated_by": "make_star_branch_target.py",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "old_target": str(old_target),
        "seed": str(seed_path),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(target, f, indent=2)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-target", default=r".\results\targets\star_simplified_dn_target.json")
    ap.add_argument("--seed", default=r".\results\seeds\star_A2p03_k2p16_d0p47_baseline_seed.json")
    ap.add_argument("--out", default=r".\results\targets\star_branch_A2p03_k2p16_d0p47_target.json")
    ap.add_argument("--min-wall-gap", type=float, default=None)
    args = ap.parse_args()

    cwd = Path.cwd()
    old_target = Path(args.old_target)
    seed = Path(args.seed)
    out = Path(args.out)
    if not old_target.is_absolute(): old_target = (cwd / old_target).resolve()
    if not seed.is_absolute(): seed = (cwd / seed).resolve()
    if not out.is_absolute(): out = (cwd / out).resolve()

    p = make_target(old_target, seed, out, args.min_wall_gap)
    print(f"[OK] wrote updated branch target:\n  {p}")


if __name__ == "__main__":
    main()
