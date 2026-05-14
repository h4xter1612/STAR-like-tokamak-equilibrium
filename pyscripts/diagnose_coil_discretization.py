#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_coil_discretization.py

Diagnostic script to inspect how STAR-like coils are represented
magnetically after importing CAD/config.

Goal:
- Count magnetic coil objects / filaments.
- Estimate whether each family is represented as one point-like coil or many filaments.
- Print R, Z, current per element if available.
- Plot magnetic centers if possible.

Usage:
    py .\diagnose_coil_discretization.py --dxf .\cad\star_baseline.dxf --plot
"""

from __future__ import annotations

import argparse
import inspect
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


FAMILIES = ["CS", "PF1", "PF2", "PF3", "PF4", "PF5", "PF6"]


def get_attr_any(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            try:
                return getattr(obj, name)
            except Exception:
                pass
    return default


def is_number(x: Any) -> bool:
    try:
        float(x)
        return math.isfinite(float(x))
    except Exception:
        return False


def guess_family(name: str) -> str:
    u = str(name).upper()

    # Order matters: PF10 should not match PF1, though you probably do not have PF10.
    for fam in ["PF6", "PF5", "PF4", "PF3", "PF2", "PF1", "CS"]:
        if u.startswith(fam) or f"_{fam}" in u or fam in u:
            return fam

    return "OTHER"


def flatten_possible_coils(obj: Any, path: str = "root", max_depth: int = 5) -> List[Tuple[str, Any]]:
    """
    Generic recursive extractor for coil-like objects.

    It looks for objects/dicts with R/Z-like fields and returns them.
    This is intentionally broad because FreeGS/freegsnke object structures vary.
    """
    out: List[Tuple[str, Any]] = []

    if max_depth < 0:
        return out

    # Dict case.
    if isinstance(obj, dict):
        # If this dict itself looks coil-like.
        keys = {str(k).lower() for k in obj.keys()}
        if any(k in keys for k in ["r", "R".lower(), "x"]) and any(k in keys for k in ["z", "Z".lower(), "y"]):
            out.append((path, obj))

        for k, v in obj.items():
            out.extend(flatten_possible_coils(v, f"{path}.{k}", max_depth - 1))
        return out

    # List/tuple case.
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.extend(flatten_possible_coils(v, f"{path}[{i}]", max_depth - 1))
        return out

    # Object case.
    # If it has R/Z attributes, it may be a coil.
    R = get_attr_any(obj, ["R", "r", "x", "Rcentre", "Rcenter", "Rc"])
    Z = get_attr_any(obj, ["Z", "z", "y", "Zcentre", "Zcenter", "Zc"])
    if R is not None and Z is not None:
        out.append((path, obj))

    # Recurse into selected attributes only, avoiding huge recursion.
    if max_depth > 0 and not isinstance(obj, (str, bytes, int, float, np.ndarray)):
        for attr in ["coils", "_coils", "coilset", "coil_set", "coil_sets", "tokamak", "machine", "components"]:
            if hasattr(obj, attr):
                try:
                    v = getattr(obj, attr)
                    out.extend(flatten_possible_coils(v, f"{path}.{attr}", max_depth - 1))
                except Exception:
                    pass

    return out


def extract_RZI(name: str, coil: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Extract R, Z, I from either dict or object.
    """
    if isinstance(coil, dict):
        R = None
        Z = None
        I = None

        for k in ["R", "r", "x", "Rcentre", "Rcenter", "Rc"]:
            if k in coil:
                R = coil[k]
                break

        for k in ["Z", "z", "y", "Zcentre", "Zcenter", "Zc"]:
            if k in coil:
                Z = coil[k]
                break

        for k in ["current", "I", "i", "current_A", "I_A"]:
            if k in coil:
                I = coil[k]
                break

        return (
            float(R) if is_number(R) else None,
            float(Z) if is_number(Z) else None,
            float(I) if is_number(I) else None,
        )

    R = get_attr_any(coil, ["R", "r", "x", "Rcentre", "Rcenter", "Rc"])
    Z = get_attr_any(coil, ["Z", "z", "y", "Zcentre", "Zcenter", "Zc"])
    I = get_attr_any(coil, ["current", "I", "i", "current_A", "I_A"])

    # Some libraries use methods.
    if callable(I):
        try:
            I = I()
        except Exception:
            I = None

    return (
        float(R) if is_number(R) else None,
        float(Z) if is_number(Z) else None,
        float(I) if is_number(I) else None,
    )


def try_build_machine_from_star_machine_cad(dxf: Optional[str]) -> Any:
    import star_machine_cad

    if not hasattr(star_machine_cad, "make_star_machine_from_cad"):
        raise RuntimeError("star_machine_cad.py no tiene make_star_machine_from_cad")

    print("\n[INFO] Calling star_machine_cad.make_star_machine_from_cad(...)")

    if dxf is None:
        out = star_machine_cad.make_star_machine_from_cad()
    else:
        out = star_machine_cad.make_star_machine_from_cad(dxf_path=dxf)

    # Tu función regresa (tokamak, geom)
    if isinstance(out, tuple) and len(out) == 2:
        tokamak, geom = out
        tokamak.geom = geom
        return tokamak

    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dxf", default=None, help="Path to CAD DXF, if your machine builder needs it.")
    ap.add_argument("--plot", action="store_true", help="Plot detected magnetic coil centers.")
    ap.add_argument("--max-print", type=int, default=300, help="Maximum individual coil-like elements to print.")
    args = ap.parse_args()

    dxf = str(Path(args.dxf)) if args.dxf else None

    print("[INFO] diagnose_coil_discretization started")
    print(f"[INFO] dxf = {dxf}")

    machine = try_build_machine_from_star_machine_cad(dxf)

    print("\n=== MACHINE OBJECT ===")
    print(type(machine))
    print(machine)

    print("\n=== MACHINE DIR HINTS ===")
    for attr in ["coils", "_coils", "coilset", "coil_set", "coil_sets", "tokamak", "machine", "components"]:
        if hasattr(machine, attr):
            try:
                v = getattr(machine, attr)
                print(f"{attr}: {type(v)}")
            except Exception as e:
                print(f"{attr}: error reading: {e}")

    coil_like = flatten_possible_coils(machine, max_depth=7)

    # Deduplicate by object id/path roughness.
    seen = set()
    unique = []
    for path, obj in coil_like:
        key = id(obj)
        if key in seen:
            continue
        seen.add(key)
        unique.append((path, obj))

    print("\n=== DETECTED COIL-LIKE OBJECTS ===")
    print(f"n_detected = {len(unique)}")

    rows = []
    for path, obj in unique:
        name = path.split(".")[-1]
        R, Z, I = extract_RZI(name, obj)
        fam = guess_family(path)
        rows.append(
            {
                "path": path,
                "family": fam,
                "R": R,
                "Z": Z,
                "I": I,
                "type": type(obj).__name__,
            }
        )

    # Print individual elements.
    print("\n=== INDIVIDUAL ELEMENTS ===")
    for row in rows[: args.max_print]:
        I_MA = row["I"] / 1e6 if row["I"] is not None else None
        print(
            f"{row['family']:6s} "
            f"R={row['R']} Z={row['Z']} "
            f"I_MA={I_MA} "
            f"path={row['path']} "
            f"type={row['type']}"
        )
    if len(rows) > args.max_print:
        print(f"... truncated, printed {args.max_print}/{len(rows)}")

    # Summary per family.
    byfam = defaultdict(list)
    for row in rows:
        byfam[row["family"]].append(row)

    print("\n=== FAMILY SUMMARY ===")
    for fam in sorted(byfam.keys()):
        rr = byfam[fam]
        Rs = np.array([x["R"] for x in rr if x["R"] is not None], dtype=float)
        Zs = np.array([x["Z"] for x in rr if x["Z"] is not None], dtype=float)
        Is = np.array([x["I"] for x in rr if x["I"] is not None], dtype=float)

        n = len(rr)
        Rrng = (float(np.min(Rs)), float(np.max(Rs))) if Rs.size else (None, None)
        Zrng = (float(np.min(Zs)), float(np.max(Zs))) if Zs.size else (None, None)
        Itot = float(np.sum(Is)) if Is.size else None

        print(
            f"{fam:8s} n={n:4d} "
            f"R_range={Rrng} Z_range={Zrng} "
            f"Itotal_MA={(Itot/1e6 if Itot is not None else None)}"
        )

    if args.plot:
        fig, ax = plt.subplots(figsize=(8, 10))

        for fam in sorted(byfam.keys()):
            rr = byfam[fam]
            Rs = [x["R"] for x in rr if x["R"] is not None and x["Z"] is not None]
            Zs = [x["Z"] for x in rr if x["R"] is not None and x["Z"] is not None]
            if not Rs:
                continue
            ax.scatter(Rs, Zs, s=18, label=f"{fam} n={len(Rs)}")

            # Label single-point families.
            if len(Rs) <= 3:
                for R, Z in zip(Rs, Zs):
                    ax.text(R, Z, fam, fontsize=8)

        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_title("Detected magnetic coil / filament centers")
        ax.grid(True, alpha=0.35)
        ax.axis("equal")
        ax.legend(loc="best", fontsize=8)

        out = Path("results") / "coil_discretization_debug.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"\n[OK] Saved plot: {out.resolve()}")
        plt.show()


if __name__ == "__main__":
    main()
