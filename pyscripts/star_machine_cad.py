"""
star_machine_cad.py

Build a FreeGSNKE machine from a CAD DXF (2D poloidal cross-section).

Conventions:
- DXF X axis = R [m]
- DXF Y axis = Z [m]
- Required layers:
    WALL_OUTER   : closed polyline for conducting boundary used by GS
- Optional layers:
    WALL_INNER   : closed polyline for limiter/inner wall
    PLASMA_TARGET: closed polyline reference
- Coils (recommended, deterministic):
    One rectangle polyline per coil on layers: COIL_CS, COIL_PF1U, COIL_PF1L, ...
  Fallback (legacy, less robust):
    Rectangles on COILS + labels on COIL_LABELS

Stability improvements (CAD-realistic ready):
  - Sequential duplicate removal / closure enforcement
  - CCW enforcement for walls (area > 0); if area < 0 reverse
  - Optional resampling policy: "auto" | "always" | "never"
  - Canonical start point: outboard midplane (max R, then min |Z|)
  - POLYLINE vertex API compatibility (list vs callable)
  - Optional entity flattening via ezdxf.path (if available)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import matplotlib.pyplot as plt

from freegs4e import machine

try:
    import ezdxf
except Exception:
    ezdxf = None

try:
    from ezdxf import path as ezpath
except Exception:
    ezpath = None


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class CADLayers:
    wall_outer: str = "WALL_OUTER"
    wall_inner: str = "WALL_INNER"
    plasma_target: str = "PLASMA_TARGET"

    coil_layer_prefix: str = "COIL_"          # Recommended: COIL_CS, COIL_PF1U, ...
    coils_layer: str = "COILS"                # Fallback: all rectangles here
    coil_labels_layer: str = "COIL_LABELS"    # Fallback: text labels here


@dataclass(frozen=True)
class CADImportOptions:
    # If unit_scale is None, infer from $INSUNITS. Example: mm -> 1e-3.
    unit_scale: Optional[float] = None

    # Wall resampling policy: "auto" | "always" | "never"
    # - "auto": resample only if len(points) < min_wall_pts
    # - "always": always resample to n_wall/n_inner/n_plasma
    # - "never": keep vertices as-is (strict retrocompat mode)
    resample_walls: str = "auto"

    # Target points used if resampling is active
    n_wall: int = 801
    n_inner: int = 801
    n_plasma: int = 400
    min_wall_pts: int = 200

    # Enforce CCW orientation for wall polylines (recommended)
    enforce_ccw: bool = True

    # Rotate start to outboard midplane for consistency (recommended)
    canonical_start: bool = True

    # Flattening chord-length target (meters) for SPLINE/ARC/ELLIPSE fallback (if ezdxf.path available)
    flatten_distance: float = 0.01

    # Fallback label matching tolerance (legacy mode)
    label_match_factor: float = 2.0  # radius ~ factor * max(dR,dZ)


# -----------------------------
# Helpers
# -----------------------------

def _normalize_label(s: str) -> str:
    return str(s).strip().upper()


def _infer_unit_scale_from_insunits(insunits_code: int) -> float:
    # AutoCAD $INSUNITS: 0=unitless, 1=in, 2=ft, 4=mm, 5=cm, 6=m
    mapping = {0: 1.0, 1: 0.0254, 2: 0.3048, 4: 1e-3, 5: 1e-2, 6: 1.0}
    return mapping.get(int(insunits_code), 1.0)


def _ensure_closed(xy: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    pts = np.asarray(xy, dtype=float)
    if len(pts) == 0:
        return pts
    if np.linalg.norm(pts[0] - pts[-1]) > tol:
        pts = np.vstack([pts, pts[0]])
    return pts


def _drop_duplicate_endpoint(xy: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    pts = np.asarray(xy, dtype=float)
    if len(pts) < 2:
        return pts
    if np.linalg.norm(pts[0] - pts[-1]) <= tol:
        return pts[:-1]
    return pts


def _dedupe_sequential(xy: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    pts = np.asarray(xy, dtype=float)
    if len(pts) <= 1:
        return pts
    d = pts[1:] - pts[:-1]
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) > eps * eps
    return pts[keep]


def _polygon_area(xy: np.ndarray) -> float:
    pts = _ensure_closed(np.asarray(xy, dtype=float))
    if len(pts) < 4:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _rotate_to_outboard_midplane(xy: np.ndarray) -> np.ndarray:
    """
    Rotate closed contour so the first point is at outboard midplane:
      maximize R, and among ties minimize |Z|.
    """
    pts_open = _drop_duplicate_endpoint(xy)
    if len(pts_open) == 0:
        return _ensure_closed(pts_open)
    score = pts_open[:, 0] - 1e-6 * np.abs(pts_open[:, 1])
    idx = int(np.argmax(score))
    pts_rot = np.roll(pts_open, -idx, axis=0)
    return _ensure_closed(pts_rot)


def _enforce_ccw(xy: np.ndarray) -> np.ndarray:
    """
    Ensure positive signed area (CCW). Keeps closure.
    """
    pts_open = _drop_duplicate_endpoint(xy)
    if _polygon_area(pts_open) < 0:
        pts_open = pts_open[::-1].copy()
    return _ensure_closed(pts_open)


def _resample_closed_curve(xy: np.ndarray, n: int) -> np.ndarray:
    """
    Uniform arc-length resampling of a closed polyline.
    Returns a CLOSED polyline (first point repeated at end).
    """
    pts = _ensure_closed(_dedupe_sequential(np.asarray(xy, dtype=float)))
    if len(pts) < 4:
        return pts

    seg = pts[1:] - pts[:-1]
    ds = np.sqrt(np.sum(seg**2, axis=1))
    s = np.hstack([[0.0], np.cumsum(ds)])
    total = s[-1]
    if total <= 0:
        return pts

    s_new = np.linspace(0.0, total, int(n), endpoint=False)
    x_new = np.interp(s_new, s, pts[:, 0])
    y_new = np.interp(s_new, s, pts[:, 1])
    out = np.column_stack([x_new, y_new])
    out = _ensure_closed(_dedupe_sequential(out))
    return out


def _maybe_resample(xy: np.ndarray, n: int, policy: str, min_pts: int) -> np.ndarray:
    pol = str(policy).strip().lower()
    pts_open = _drop_duplicate_endpoint(xy)
    if pol == "never":
        return _ensure_closed(pts_open)
    if pol == "always":
        return _resample_closed_curve(pts_open, n)
    # auto
    if len(pts_open) < int(min_pts):
        return _resample_closed_curve(pts_open, n)
    return _ensure_closed(pts_open)


def _bbox_from_poly(xy: np.ndarray) -> Tuple[float, float, float, float]:
    pts = np.asarray(xy, dtype=float)
    xmin = float(np.min(pts[:, 0]))
    xmax = float(np.max(pts[:, 0]))
    ymin = float(np.min(pts[:, 1]))
    ymax = float(np.max(pts[:, 1]))
    return xmin, xmax, ymin, ymax


def _coil_from_rect_poly(xy: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Convert rectangle polyline to (Rc, Zc, dR, dZ) using bounding box.
    dR/dZ are HALF-extents.
    """
    xmin, xmax, ymin, ymax = _bbox_from_poly(xy)
    Rc = 0.5 * (xmin + xmax)
    Zc = 0.5 * (ymin + ymax)
    dR = 0.5 * (xmax - xmin)
    dZ = 0.5 * (ymax - ymin)
    return Rc, Zc, dR, dZ


def _entity_to_xy(entity, opts: CADImportOptions) -> np.ndarray:
    """
    Extract XY vertices from LWPOLYLINE / POLYLINE.
    Fallback: flatten via ezdxf.path for SPLINE/ARC/ELLIPSE if available.
    """
    et = entity.dxftype()

    if et == "LWPOLYLINE":
        pts = [(p[0], p[1]) for p in entity.get_points("xy")]
        return np.array(pts, dtype=float)

    if et == "POLYLINE":
        # ezdxf version differences:
        #   - entity.vertices() (callable)
        #   - entity.vertices (list)
        verts = getattr(entity, "vertices", None)
        if callable(verts):
            verts = verts()
        if verts is None:
            # last attempt: method call
            try:
                verts = entity.vertices()
            except Exception:
                verts = []
        pts = [(float(v.dxf.location.x), float(v.dxf.location.y)) for v in verts]
        return np.array(pts, dtype=float)

    if ezpath is not None:
        try:
            p = ezpath.make_path(entity)
            pts = [(float(v.x), float(v.y)) for v in p.flattening(distance=float(opts.flatten_distance))]
            return np.array(pts, dtype=float)
        except Exception:
            pass

    raise ValueError(f"Unsupported entity type '{et}'. Use (LW)POLYLINE in DXF.")


def _text_entities_from_layer(msp, layer: str) -> List[Tuple[str, float, float]]:
    out: List[Tuple[str, float, float]] = []
    for e in msp.query(f'TEXT[layer=="{layer}"]'):
        try:
            s = str(e.dxf.text).strip()
            x, y = float(e.dxf.insert.x), float(e.dxf.insert.y)
            if s:
                out.append((s, x, y))
        except Exception:
            continue

    for e in msp.query(f'MTEXT[layer=="{layer}"]'):
        try:
            s = str(e.plain_text()).strip()
            x, y = float(e.dxf.insert.x), float(e.dxf.insert.y)
            if s:
                out.append((s, x, y))
        except Exception:
            continue
    return out


# -----------------------------
# DXF -> geom
# -----------------------------

def load_geom_from_dxf(
    dxf_path: Path,
    layers: CADLayers = CADLayers(),
    opts: CADImportOptions = CADImportOptions(),
) -> Dict:
    """
    Read DXF and return a geom dict compatible with your pipeline:
      geom["R_outer"], geom["Z_outer"], geom["R_inner"], geom["Z_inner"],
      geom["R_plasma"], geom["Z_plasma"], geom["coils"] = {label: (Rc,Zc,dR,dZ)}
    """
    if ezdxf is None:
        raise RuntimeError("Missing dependency: ezdxf. Install: pip install ezdxf")

    dxf_path = Path(dxf_path)
    if not dxf_path.exists():
        raise FileNotFoundError(f"DXF not found: {dxf_path}")

    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    # Units
    if opts.unit_scale is None:
        try:
            insunits = int(doc.header.get("$INSUNITS", 6))
        except Exception:
            insunits = 6
        unit_scale = _infer_unit_scale_from_insunits(insunits)
    else:
        unit_scale = float(opts.unit_scale)

    def polylines_in_layer(layer_name: str) -> List[np.ndarray]:
        polys: List[np.ndarray] = []
        # robust: query all entities on layer and attempt conversion
        for e in msp.query(f'*[layer=="{layer_name}"]'):
            try:
                polys.append(_entity_to_xy(e, opts))
            except Exception:
                continue
        return polys

    # ---- Walls
    outer_candidates = polylines_in_layer(layers.wall_outer)
    if len(outer_candidates) == 0:
        raise ValueError(f"No polyline-like entities found on layer '{layers.wall_outer}'.")

    outer_xy = max(outer_candidates, key=lambda xy: abs(_polygon_area(xy)))
    outer_xy = outer_xy * unit_scale
    outer_xy = _ensure_closed(_dedupe_sequential(outer_xy))

    # CCW + canonical start + resample policy
    if opts.enforce_ccw:
        outer_xy = _enforce_ccw(outer_xy)
    if opts.canonical_start:
        outer_xy = _rotate_to_outboard_midplane(outer_xy)
    outer_xy = _maybe_resample(outer_xy, opts.n_wall, opts.resample_walls, opts.min_wall_pts)

    inner_xy = None
    inner_candidates = polylines_in_layer(layers.wall_inner)
    if len(inner_candidates) > 0:
        inner_xy = max(inner_candidates, key=lambda xy: abs(_polygon_area(xy)))
        inner_xy = inner_xy * unit_scale
        inner_xy = _ensure_closed(_dedupe_sequential(inner_xy))
        if opts.enforce_ccw:
            inner_xy = _enforce_ccw(inner_xy)
        if opts.canonical_start:
            inner_xy = _rotate_to_outboard_midplane(inner_xy)
        inner_xy = _maybe_resample(inner_xy, opts.n_inner, opts.resample_walls, opts.min_wall_pts)

    plasma_xy = None
    plasma_candidates = polylines_in_layer(layers.plasma_target)
    if len(plasma_candidates) > 0:
        plasma_xy = max(plasma_candidates, key=lambda xy: abs(_polygon_area(xy)))
        plasma_xy = plasma_xy * unit_scale
        plasma_xy = _ensure_closed(_dedupe_sequential(plasma_xy))
        # For plasma target, CCW/canonical is harmless; keep consistent
        if opts.enforce_ccw:
            plasma_xy = _enforce_ccw(plasma_xy)
        if opts.canonical_start:
            plasma_xy = _rotate_to_outboard_midplane(plasma_xy)
        plasma_xy = _maybe_resample(plasma_xy, opts.n_plasma, opts.resample_walls, opts.min_wall_pts)

    # ---- Coils (preferred): COIL_* layers
    coils: Dict[str, Tuple[float, float, float, float]] = {}

    coil_polys: List[Tuple[str, np.ndarray]] = []
    for e in msp.query("LWPOLYLINE"):
        layer = str(getattr(e.dxf, "layer", ""))
        if layer.startswith(layers.coil_layer_prefix):
            coil_polys.append((layer, _entity_to_xy(e, opts)))
    for e in msp.query("POLYLINE"):
        layer = str(getattr(e.dxf, "layer", ""))
        if layer.startswith(layers.coil_layer_prefix):
            coil_polys.append((layer, _entity_to_xy(e, opts)))

    if len(coil_polys) > 0:
        for layer, xy in coil_polys:
            label = _normalize_label(layer[len(layers.coil_layer_prefix):])
            Rc, Zc, dR, dZ = _coil_from_rect_poly(xy * unit_scale)

            if label in coils:
                raise ValueError(
                    f"Duplicate coil label '{label}' found in DXF (layers). "
                    "Ensure each coil is on its own unique COIL_<NAME> layer."
                )
            coils[label] = (float(Rc), float(Zc), float(dR), float(dZ))

    else:
        # Fallback: COILS layer + text labels
        coil_rects = polylines_in_layer(layers.coils_layer)
        labels = _text_entities_from_layer(msp, layers.coil_labels_layer)

        lab_names = [_normalize_label(t[0]) for t in labels]
        lab_xy = np.array([(t[1], t[2]) for t in labels], dtype=float) * unit_scale if labels else None

        for i, xy in enumerate(coil_rects):
            Rc, Zc, dR, dZ = _coil_from_rect_poly(xy * unit_scale)
            label = f"COIL{i+1}"

            if lab_xy is not None and len(lab_xy) > 0:
                center = np.array([Rc, Zc], dtype=float)
                dist = np.sqrt(np.sum((lab_xy - center) ** 2, axis=1))
                j = int(np.argmin(dist))
                tol = float(opts.label_match_factor) * max(float(dR), float(dZ))
                if float(dist[j]) <= tol:
                    label = lab_names[j]

            label = _normalize_label(label)
            if label in coils:
                raise ValueError(
                    f"Duplicate coil label '{label}' inferred from COILS/COIL_LABELS. "
                    "This mode is ambiguous. Prefer per-coil layers COIL_<NAME>."
                )
            coils[label] = (float(Rc), float(Zc), float(dR), float(dZ))

    # Assemble geom dict
    geom: Dict = {
        "R_outer": outer_xy[:, 0],
        "Z_outer": outer_xy[:, 1],
        "coils": coils,
        "cad_path": str(dxf_path),
        "unit_scale": float(unit_scale),
    }
    if inner_xy is not None:
        geom["R_inner"] = inner_xy[:, 0]
        geom["Z_inner"] = inner_xy[:, 1]
    if plasma_xy is not None:
        geom["R_plasma"] = plasma_xy[:, 0]
        geom["Z_plasma"] = plasma_xy[:, 1]

    # Basic derived numbers (nice-to-have)
    if "R_plasma" in geom:
        Rmin, Rmax = float(np.min(geom["R_plasma"])), float(np.max(geom["R_plasma"]))
        Zmin, Zmax = float(np.min(geom["Z_plasma"])), float(np.max(geom["Z_plasma"]))
        geom["R0"] = 0.5 * (Rmin + Rmax)
        geom["a"] = 0.5 * (Rmax - Rmin)
        geom["kappa_geom_est"] = (Zmax - Zmin) / (2.0 * geom["a"]) if geom["a"] > 0 else np.nan
    else:
        Rmin, Rmax = float(np.min(geom["R_outer"])), float(np.max(geom["R_outer"]))
        geom["R0"] = 0.5 * (Rmin + Rmax)

    # Sanity check: outer wall should have R > 0
    if float(np.min(geom["R_outer"])) <= 0.0:
        raise ValueError("Outer wall contains R <= 0. Check DXF coordinates and units (X must be R > 0).")

    return geom


# -----------------------------
# Build FreeGSNKE machine
# -----------------------------

def _build_machine_compat(coils_for_machine, vessel_wall):
    """
    Minimal compatibility: prefer Machine(coils, wall=...)
    fallback to Machine(coils, vessel_wall)
    """
    try:
        return machine.Machine(coils_for_machine, wall=vessel_wall)
    except TypeError:
        return machine.Machine(coils_for_machine, vessel_wall)


def make_star_machine_from_cad(
    dxf_path: Optional[str] = None,
    layers: CADLayers = CADLayers(),
    opts: CADImportOptions = CADImportOptions(),
    strict_expected: bool = False,
    expected_coils: Optional[Set[str]] = None,
):
    """
    Build a FreeGSNKE machine from CAD DXF.

    Parameters
    ----------
    dxf_path:
        If None, defaults to pyscripts/cad/star_baseline.dxf
        If relative, resolved relative to pyscripts/cad/
    strict_expected:
        If True, raise if expected coils are missing.
    expected_coils:
        Set of expected coil labels (already normalized, e.g., {"CS","PF1U",...}).
        If None and strict_expected=True, uses the standard STAR baseline set.

    Returns
    -------
    tokamak : machine.Machine
    geom : dict
    """
    script_dir = Path(__file__).resolve().parent
    cad_dir = script_dir / "cad"
    cad_dir.mkdir(parents=True, exist_ok=True)

    if dxf_path is None:
        dxf = cad_dir / "star_baseline.dxf"
    else:
        dxf = Path(dxf_path)
        if not dxf.is_absolute():
            dxf = (cad_dir / dxf).resolve()

    geom = load_geom_from_dxf(dxf, layers=layers, opts=opts)

    # Optional strict coil check (useful for retrocompatibility debugging)
    if strict_expected:
        if expected_coils is None:
            expected_coils = {"CS", "PF1U", "PF1L", "PF2U", "PF2L", "PF3U", "PF3L"}
        have = set(_normalize_label(k) for k in geom["coils"].keys())
        missing = sorted(list(set(expected_coils) - have))
        if missing:
            raise ValueError(f"Missing expected coils in CAD import: {missing}")

    # Walls
    vessel_wall = machine.Wall(geom["R_outer"], geom["Z_outer"])
    limiter = None
    if "R_inner" in geom and "Z_inner" in geom:
        limiter = machine.Wall(geom["R_inner"], geom["Z_inner"])

    # Coils
    coils_for_machine = []
    for label, (Rc, Zc, dR, dZ) in geom["coils"].items():
        lab = _normalize_label(label)
        c = machine.MultiCoil(float(Rc), float(Zc), float(dR), float(dZ))
        try:
            c.label = lab
        except Exception:
            pass
        coils_for_machine.append((lab, c))

    tokamak = _build_machine_compat(coils_for_machine, vessel_wall)

    # Keep your original behavior: set limiter as attribute (works with your env)
    if limiter is not None:
        try:
            tokamak.limiter = limiter
        except Exception:
            pass

    # Convenience attributes (to match your existing usage)
    tokamak.active_coils = [label for label, _ in coils_for_machine]
    tokamak.passive_coils = []
    tokamak.R0 = float(geom.get("R0", np.nan))
    tokamak.geom = geom
    tokamak.coils_dict = {label: coil for label, coil in coils_for_machine}

    return tokamak, geom


# -----------------------------
# Optional: quick plot
# -----------------------------

def plot_cad_geometry(geom: Dict, show: bool = True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 9))

    ax.plot(geom["R_outer"], geom["Z_outer"], "k-", lw=2, label="CAD outer wall")
    if "R_inner" in geom:
        ax.plot(geom["R_inner"], geom["Z_inner"], "k--", lw=1.5, label="CAD inner wall")
    if "R_plasma" in geom:
        ax.plot(geom["R_plasma"], geom["Z_plasma"], color="tab:orange", lw=1.5, label="CAD plasma target")

    for name, (Rc, Zc, dR, dZ) in geom["coils"].items():
        x0, x1 = Rc - dR, Rc + dR
        y0, y1 = Zc - dZ, Zc + dZ
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k-", lw=1)
        ax.text(Rc, Zc, _normalize_label(name), ha="center", va="center", fontsize=8)

    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.grid(True)
    ax.legend(loc="upper right")

    if show:
        plt.tight_layout()
        plt.show()
    return ax


if __name__ == "__main__":
    # Smoke test: baseline import (keep plot identical if baseline already dense)
    opts = CADImportOptions(
        unit_scale=None,          # infer from INSUNITS
        resample_walls="auto",    # safe default for CAD-realistic (won't change dense baselines)
        n_wall=801,
        n_inner=801,
        min_wall_pts=200,
        enforce_ccw=True,
        canonical_start=True,
    )

    tokamak, geom = make_star_machine_from_cad(opts=opts, strict_expected=True)
    print("[OK] Loaded CAD machine from:", geom.get("cad_path"))
    print("[INFO] Coils found:", sorted(list(geom["coils"].keys())))
    print("[INFO] outer wall points:", len(geom["R_outer"]), "| inner wall points:", len(geom.get("R_inner", [])))
    area = _polygon_area(np.column_stack([geom["R_outer"], geom["Z_outer"]]))
    print(f"[INFO] outer wall area (signed, CCW+): {float(area):.6e}")

    plot_cad_geometry(geom, show=True)

