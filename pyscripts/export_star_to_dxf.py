#!/usr/bin/env python3
"""
export_star_to_dxf.py

Export the current STAR-like analytic geometry (from star_machine.star_geometry)
to a 2D DXF for AutoCAD.

Mapping:
  DXF X axis = R [m]
  DXF Y axis = Z [m]

Default output:
  pyscripts/cad/star_baseline.dxf

Layers:
  - WALL_OUTER
  - WALL_INNER
  - PLASMA_TARGET
  - COIL_<NAME>     (deterministic per-coil layers, e.g., COIL_PF1L)
  - COILS           (optional copy of all coil rectangles)
  - COIL_LABELS     (optional TEXT labels at coil centers)
  - AXES            (optional reference lines)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from star_machine import star_geometry


def _ensure_closed_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    pts = np.column_stack([x, y])
    if len(pts) < 2:
        return pts
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    return pts


def _add_closed_polyline(msp, pts_xy: np.ndarray, layer: str):
    points = [(float(p[0]), float(p[1])) for p in pts_xy]
    msp.add_lwpolyline(points, dxfattribs={"layer": layer, "closed": True})


def _coil_rect_vertices(Rc: float, Zc: float, dR: float, dZ: float) -> np.ndarray:
    # dR/dZ are HALF-extents (consistent with your current convention)
    x0, x1 = Rc - dR, Rc + dR
    y0, y1 = Zc - dZ, Zc + dZ
    return np.array([[x0, y0],
                     [x1, y0],
                     [x1, y1],
                     [x0, y1],
                     [x0, y0]], dtype=float)


def _add_centered_text(msp, text_str: str, x: float, y: float, layer: str, height: float):
    """
    Place centered text in a way that works across ezdxf versions.
    """
    t = msp.add_text(text_str, dxfattribs={"layer": layer, "height": float(height)})

    # Preferred (newer ezdxf)
    if hasattr(t, "set_placement"):
        try:
            try:
                from ezdxf.enums import TextEntityAlignment
                t.set_placement((float(x), float(y)), align=TextEntityAlignment.MIDDLE_CENTER)
            except Exception:
                t.set_placement((float(x), float(y)), align="MIDDLE_CENTER")
            return
        except Exception:
            pass

    # Fallbacks
    t.dxf.insert = (float(x), float(y))
    try:
        t.dxf.halign = 1  # center
        t.dxf.valign = 2  # middle
        t.dxf.align_point = (float(x), float(y))
    except Exception:
        pass


def _resolve_out_path(user_out: str | None) -> Path:
    """
    Default output: pyscripts/cad/star_baseline.dxf (relative to this script).
    If user_out is relative, resolve it relative to pyscripts/ (script directory).
    """
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir / "cad" / "star_baseline.dxf"

    if not user_out:
        return default_out

    p = Path(user_out)
    if p.is_absolute():
        return p
    return (script_dir / p).resolve()


def export_star_geometry_to_dxf(
    out_path: Path,
    R0: float = 4.0,
    A: float = 1.7,
    kappa: float = 1.8,
    delta: float = 0.30,
    add_axes: bool = True,
    insunits: str = "m",
    also_export_coils_layer: bool = True,
    export_labels: bool = True,
):
    try:
        import ezdxf
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: ezdxf\n"
            "Install it in your venv:\n"
            "  pip install ezdxf\n"
        ) from e

    geom = star_geometry(R0=R0, A=A, kappa=kappa, delta=delta)

    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    # Units metadata ($INSUNITS)
    insunits_map = {"in": 1, "ft": 2, "mm": 4, "cm": 5, "m": 6}
    doc.header["$INSUNITS"] = insunits_map.get(insunits.lower(), 6)

    # Base layers
    base_layers = ["WALL_OUTER", "WALL_INNER", "PLASMA_TARGET", "AXES"]
    if also_export_coils_layer:
        base_layers.append("COILS")
    if export_labels:
        base_layers.append("COIL_LABELS")

    for layer_name in base_layers:
        if layer_name not in doc.layers:
            doc.layers.new(name=layer_name)

    # Walls / plasma target
    outer = _ensure_closed_xy(geom["R_outer"], geom["Z_outer"])
    inner = _ensure_closed_xy(geom["R_inner"], geom["Z_inner"])
    plasma = _ensure_closed_xy(geom["R_plasma"], geom["Z_plasma"])

    _add_closed_polyline(msp, outer, layer="WALL_OUTER")
    _add_closed_polyline(msp, inner, layer="WALL_INNER")
    _add_closed_polyline(msp, plasma, layer="PLASMA_TARGET")

    # Coils: export deterministically per layer (COIL_<NAME>)
    for name, (Rc, Zc, dR, dZ) in geom["coils"].items():
        label = str(name).strip().upper()
        coil_layer = f"COIL_{label}"
        if coil_layer not in doc.layers:
            doc.layers.new(name=coil_layer)

        rect = _coil_rect_vertices(float(Rc), float(Zc), float(dR), float(dZ))
        pts = [(float(x), float(y)) for x, y in rect]

        # The key part: per-coil layer
        msp.add_lwpolyline(pts, dxfattribs={"layer": coil_layer, "closed": True})

        # Optional: also export all coils into a common layer for quick viewing
        if also_export_coils_layer:
            msp.add_lwpolyline(pts, dxfattribs={"layer": "COILS", "closed": True})

        # Optional: labels
        if export_labels:
            _add_centered_text(
                msp,
                text_str=label,
                x=float(Rc),
                y=float(Zc),
                layer="COIL_LABELS",
                height=0.15,  # 15 cm in meters
            )

    # Axes
    if add_axes:
        Rmin = min(float(outer[:, 0].min()), 0.0)
        Rmax = float(outer[:, 0].max())
        Zmin = float(outer[:, 1].min())
        Zmax = float(outer[:, 1].max())

        msp.add_line((Rmin, 0.0), (Rmax, 0.0), dxfattribs={"layer": "AXES"})
        msp.add_line((0.0, Zmin), (0.0, Zmax), dxfattribs={"layer": "AXES"})

        if "R0" in geom:
            msp.add_line((float(geom["R0"]), Zmin), (float(geom["R0"]), Zmax), dxfattribs={"layer": "AXES"})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(out_path))
    print(f"[OK] Wrote DXF: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Export STAR-like analytic geometry to DXF for AutoCAD.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output DXF path. If relative, resolved relative to pyscripts/. "
                         "Default: cad/star_baseline.dxf")
    ap.add_argument("--R0", type=float, default=4.0)
    ap.add_argument("--A", type=float, default=1.7)
    ap.add_argument("--kappa", type=float, default=1.8)
    ap.add_argument("--delta", type=float, default=0.30)
    ap.add_argument("--no-axes", action="store_true")
    ap.add_argument("--insunits", type=str, default="m", choices=["m", "mm", "cm", "in", "ft"])
    ap.add_argument("--no-coils-layer", action="store_true",
                    help="Do not also write all coil rectangles into the common COILS layer (per-coil layers still written).")
    ap.add_argument("--no-labels", action="store_true",
                    help="Do not export TEXT/MTEXT coil labels.")

    args = ap.parse_args()
    out_path = _resolve_out_path(args.out)

    export_star_geometry_to_dxf(
        out_path=out_path,
        R0=args.R0,
        A=args.A,
        kappa=args.kappa,
        delta=args.delta,
        add_axes=(not args.no_axes),
        insunits=args.insunits,
        also_export_coils_layer=(not args.no_coils_layer),
        export_labels=(not args.no_labels),
    )


if __name__ == "__main__":
    main()

