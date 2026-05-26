"""
star_machine_cad.py

Build a FreeGSNKE/freegs4e machine from a CAD DXF (2D poloidal cross-section).

Conventions:
- DXF X axis = R [m]
- DXF Y axis = Z [m]
- Required layers:
    WALL_OUTER   : closed polyline for conducting boundary used by GS
- Optional layers:
    WALL_INNER   : closed polyline for limiter/inner wall
    PLASMA_TARGET: closed polyline reference
- Coils (recommended, deterministic):
    One rectangle polyline per coil on layers:
        COIL_CS, COIL_PF1U, COIL_PF1L, ...
    You may also use multiple CS segments as distinct layers:
        COIL_CS1M, COIL_CS2U, COIL_CS2L, ...
    These will be treated as separate coils but grouped under family "CS".
  Fallback (legacy, less robust):
    Rectangles on COILS + labels on COIL_LABELS

Key features:
- Robust wall import: CCW enforcement, canonical start, resampling.
- "Smooth CAD" option: densify polyline segments by max segment length (meters).
- AUTO plasma target: Miller-like boundary fit inside inner wall (preferred) or outer wall.
- Geometric xpoints markers + strike rays (plot-only).
- Blanket passive filaments: fill region between WALL_INNER and WALL_OUTER uniformly.
- Coil family grouping + area weights.
- Coil area metrics + recommended Imax by family: Imax ≈ J_eng * fill_factor * A_eff.

Notes:
- "Imax recommended" is an engineering *order-of-magnitude* bound based on cross-section
  and an assumed engineering current density J_eng. It is NOT a hard physics limit.
- For ITER-like SC coils, J_eng can be ~30–60 A/mm^2 depending on design choices
  (stabilizer, coolant, insulation, winding pack fill, peak field margins, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

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

    # Passive / structural CAD layers
    # STAR_VESSEL is treated as real passive metal geometry, not active PF/OH current.
    star_vessel: str = "STAR_VESSEL"
    passive_layer_prefix: str = "PASSIVE_"

    coil_layer_prefix: str = "COIL_"          # Recommended: COIL_CS, COIL_PF1U, COIL_CS1M, ...
    coils_layer: str = "COILS"                # Fallback: all rectangles here
    coil_labels_layer: str = "COIL_LABELS"    # Fallback: text labels here

    # --- NEW: divertor windows / markers ---
    xpt_lower_win: str = "XPT_LOWER_WIN"
    xpt_upper_win: str = "XPT_UPPER_WIN"
    strike_lower_win: str = "STRIKE_LOWER_WIN"
    strike_upper_win: str = "STRIKE_UPPER_WIN"

    strike_upper_in: str = "STRIKE_U_IN"
    strike_lower_in: str = "STRIKE_L_IN"
    strike_upper_out: str = "STRIKE_U_OUT"
    strike_lower_out: str = "STRIKE_L_OUT"

    # --- Separatrix leg corridor windows ---
    # These are not strike targets. They indicate where the divertor legs
    # should pass, e.g. through the PF4-PF5 corridor rather than PF5-PF6.
    leg_lower_win: str = "LEG_LOWER_WIN"
    leg_upper_win: str = "LEG_UPPER_WIN"

@dataclass(frozen=True)
class CADImportOptions:
    # If unit_scale is None, infer from $INSUNITS. Example: mm -> 1e-3.
    unit_scale: Optional[float] = None

    # Wall resampling policy (arc-length): "auto" | "always" | "never"
    resample_walls: str = "auto"

    # Target points used if arc-length resampling is active
    n_wall: int = 1601
    n_inner: int = 2001
    n_plasma: int = 501
    min_wall_pts: int = 400

    # Enforce CCW orientation for wall polylines (recommended)
    enforce_ccw: bool = True

    # Rotate start to outboard midplane for consistency (recommended)
    canonical_start: bool = True

    # -------------------------
    # "Smooth CAD" densification (recommended for low-poly CAD)
    # -------------------------
    # Prefer ezdxf.path flattening even for polylines (if available).
    prefer_path_flattening: bool = True

    # Chord-length target (meters) used by ezdxf.path flattening for SPLINE/ARC/ELLIPSE
    flatten_distance: float = 0.002

    # Subdivide polyline edges so max segment length <= this value (meters)
    # Applied after units scaling, before arc-length resampling.
    max_seg_len_wall: float = 0.008
    max_seg_len_plasma: float = 0.008

    # -------------------------
    # Fallback label matching tolerance (legacy mode)
    # -------------------------
    label_match_factor: float = 2.0  # radius ~ factor * max(dR,dZ)

    # -------------------------
    # AUTO plasma target
    # -------------------------
    plasma_target_mode: str = "auto"  # "cad" | "auto"
    plasma_fit_to_inner_if_available: bool = True

    # Nominal target parameters
    plasma_R0: float = 4.0
    plasma_A: float = 2.0
    plasma_kappa: float = 2.5
    plasma_Z0: float = 0.0

    # Triangularity scan
    plasma_delta_max: float = 0.70
    plasma_delta_grid: int = 17
    plasma_delta_symmetric: bool = True   # True => scan du=dl only (fast). False => scan du,dl grid (slower)

    # Shrink search
    plasma_shrink_iters: int = 20
    plasma_scale_safety: float = 0.999   # keep a tiny margin

    # Strict inside check: radius<0 shrinks polygon so "on boundary" counts as outside
    containment_radius: float = -1e-9

    # If (R0,Z0) not inside fit wall, move center to nearest sampled interior point (optional)
    fix_center_if_outside: bool = True
    center_search_samples: int = 800
    center_search_seed: int = 0

    # Strike line ray length cap (m) if intersection fails
    strike_ray_fallback_len: float = 3.0

    # -------------------------
    # BLANKET / passive filaments (AUTO between inner & outer wall)
    # -------------------------
    blanket_enabled: bool = False
    blanket_n_filaments: int = 0
    blanket_distribution: str = "stratified"   # "stratified" (recommended), "grid", "random"
    blanket_seed: int = 0
    blanket_wall_margin_m: float = 0.01
    blanket_filament_dR: float = 0.004
    blanket_filament_dZ: float = 0.004
    blanket_bins_R: int = 0
    blanket_bins_Z: int = 0
    blanket_pitch_mode: str = "auto"   # "auto" or "manual"
    blanket_pitch_R: float = 0.03
    blanket_pitch_Z: float = 0.03
    blanket_label_prefix: str = "BLK"
    blanket_containment_radius: float = -1e-9

    # -------------------------
    # STAR_VESSEL / passive structure discretization
    # -------------------------
    # New preferred passive model: discretize real CAD solids on STAR_VESSEL
    # and PASSIVE_* layers. These replace the old artificial blanket fill.
    passive_structures_enabled: bool = True
    passive_use_star_vessel: bool = True
    passive_use_passive_prefix: bool = True

    # Passive cell target size [m]. Keep coarse enough to avoid thousands of coils.
    passive_target_dR_m: float = 0.10
    passive_target_dZ_m: float = 0.10
    passive_nR_max: int = 80
    passive_nZ_max: int = 160
    passive_min_cell_area_m2: float = 1.0e-6
    passive_containment_radius: float = -1e-9

    # Effective material metadata. Static FreeGS/FreeGSNKE forward equilibria do not
    # use resistivity directly, but the metadata is attached for passive/dynamic models
    # and for diagnostics.
    star_vessel_material: str = "SS316L"
    star_vessel_resistivity_ohm_m: float = 0.75e-6
    passive_default_material: str = "SS316L"
    passive_default_resistivity_ohm_m: float = 0.75e-6
    first_wall_material: str = "EUROFER97"
    first_wall_resistivity_ohm_m: float = 1.0e-6
    blanket_outer_material: str = "EUROFER97"
    blanket_outer_resistivity_ohm_m: float = 1.0e-6

    # Semantic wall mapping for the updated CAD:
    # WALL_INNER = first wall / limiter; WALL_OUTER = blanket outer/back plate.
    # Machine wall remains WALL_OUTER for compatibility unless your solver later
    # needs a separate computational envelope.
    machine_wall_source: str = "outer"   # "outer" | "inner"
    limiter_source: str = "inner"        # "inner" | "outer"

    # -------------------------
    # Coil Imax recommendation (engineering)
    # -------------------------
    # Effective conductor area = fill_factor * geometric rectangle area.
    # Imax ≈ Jeng * Aeff.
    fill_factor: float = 0.75
    Jeng_default_A_per_mm2: float = 40.0
    family_mode: str = "min"  # "min" (conservative) or "sum"
    # Optional per-family overrides, e.g. {"CS": 35.0, "PF6": 45.0}
    family_J_override_A_per_mm2: Optional[Dict[str, float]] = None

    # -------------------------
    # Active coil discretization
    # -------------------------
    # If True, each CAD rectangular active coil is expanded into multiple
    # smaller magnetic elements before building the FreeGS/freegs4e Machine.
    coil_discretize_active: bool = True

    # Target physical cell size for PF coils [m]
    coil_target_dR_m: float = 0.12
    coil_target_dZ_m: float = 0.12

    # Target physical cell size for CS [m]
    coil_target_dR_CS_m: float = 0.08
    coil_target_dZ_CS_m: float = 0.18

    # Safety clamps to avoid excessive filament counts
    coil_nR_min: int = 2
    coil_nZ_min: int = 2
    coil_nR_max: int = 8
    coil_nZ_max: int = 48

    # -------------------------
    # Optional effective CS segmentation
    # -------------------------
    # Experimental model:
    #   CS_MID = central part of the CS
    #   CS_END = upper + lower ends of the CS, kept symmetric
    #
    # This does not modify the CAD. It only creates additional current-control
    # groups after the CS has been discretized into filaments.
    cs_segmented: bool = True

    # Fraction of the full CS height assigned to CS_MID.
    # Example:
    #   0.45 means central 45% of the full CS height is CS_MID,
    #   and the remaining 55% is CS_END, split between top and bottom.
    cs_mid_fraction: float = 1.0 #0.45

    # Optional absolute Z cutoff [m].
    # If None, the code computes:
    #   zcut = cs_mid_fraction * max(abs(Z_center) + dZ)
    cs_segment_zcut_m: Optional[float] = None

    # Keep the original CS group for backward compatibility.
    # If True:
    #   coil_groups contains CS, CS_MID, CS_END.
    # If False:
    #   coil_groups contains only CS_MID, CS_END for the CS.
    #
    # I recommend True for now so old scripts using "CS" still work.
    cs_segment_keep_parent: bool = True


# -----------------------------
# Helpers
# -----------------------------

def _normalize_label(s: str) -> str:
    return str(s).strip().upper()

def _polyline_length_open(xy: np.ndarray) -> float:
    P = np.asarray(xy, float)
    if P.ndim != 2 or P.shape[0] < 2:
        return 0.0
    d = P[1:] - P[:-1]
    return float(np.sum(np.sqrt(np.sum(d * d, axis=1))))

def _is_closed_polyline(xy: np.ndarray, tol: float = 1e-9) -> bool:
    P = np.asarray(xy, float)
    if P.shape[0] < 3:
        return False
    return bool(np.linalg.norm(P[0] - P[-1]) <= tol)

def _prep_marker_polyline(xy: np.ndarray, *, unit_scale: float, opts: CADImportOptions) -> Dict[str, Any]:
    """
    Scale + dedupe + optional densify (using max_seg_len_plasma).
    Keeps closure if the input is closed.
    Returns dict {xy, closed, length_m}.
    """
    P = np.asarray(xy, float) * float(unit_scale)
    P = _dedupe_sequential(P)

    closed = _is_closed_polyline(P)
    if closed:
        P = _ensure_closed(P)

    # Densify a bit for stable distance computations
    maxlen = float(getattr(opts, "max_seg_len_plasma", 0.0))
    if np.isfinite(maxlen) and maxlen > 0:
        if closed:
            P = _subdivide_closed_polyline_by_maxseg(P, maxlen)
        else:
            P = _subdivide_open_polyline_by_maxseg(P, maxlen)

    P = _dedupe_sequential(P)
    if closed:
        P = _ensure_closed(P)

    L = _polyline_length_open(_drop_duplicate_endpoint(P))
    return {"xy": P, "closed": bool(closed), "length_m": float(L)}

def _pick_best_marker(candidates: List[np.ndarray], *, unit_scale: float, opts: CADImportOptions) -> Optional[Dict[str, Any]]:
    """
    Pick the 'best' (longest) marker polyline from candidates.
    """
    best = None
    bestL = -1.0
    for xy in candidates:
        try:
            pack = _prep_marker_polyline(xy, unit_scale=unit_scale, opts=opts)
            L = float(pack.get("length_m", 0.0))
            if L > bestL:
                bestL = L
                best = pack
        except Exception:
            continue
    return best

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

def _active_coil_discretization_shape(label: str, dR: float, dZ: float, opts: CADImportOptions) -> Tuple[int, int]:
    """
    Choose nR, nZ from physical coil size, not from a fixed global grid.

    dR, dZ are half-width and half-height of the CAD rectangular coil.
    Full coil dimensions are:
        width  = 2*dR
        height = 2*dZ
    """
    fam = _coil_family(label)

    width = 2.0 * float(dR)
    height = 2.0 * float(dZ)

    if width <= 0.0 or height <= 0.0:
        raise ValueError(
            f"Invalid active coil size for {label}: dR={dR}, dZ={dZ}"
        )

    if fam == "CS":
        target_R = float(getattr(opts, "coil_target_dR_CS_m", 0.08))
        target_Z = float(getattr(opts, "coil_target_dZ_CS_m", 0.18))
    else:
        target_R = float(getattr(opts, "coil_target_dR_m", 0.12))
        target_Z = float(getattr(opts, "coil_target_dZ_m", 0.12))

    target_R = max(target_R, 1.0e-6)
    target_Z = max(target_Z, 1.0e-6)

    nR = int(np.ceil(width / target_R))
    nZ = int(np.ceil(height / target_Z))

    nR_min = int(getattr(opts, "coil_nR_min", 2))
    nZ_min = int(getattr(opts, "coil_nZ_min", 2))
    nR_max = int(getattr(opts, "coil_nR_max", 8))
    nZ_max = int(getattr(opts, "coil_nZ_max", 48))

    nR = max(nR_min, min(nR, nR_max))
    nZ = max(nZ_min, min(nZ, nZ_max))

    return nR, nZ


def _discretize_rect_coil_to_filaments(
    label: str,
    Rc: float,
    Zc: float,
    dR: float,
    dZ: float,
    *,
    opts: CADImportOptions,
) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Convert one CAD rectangular active coil into many smaller magnetic elements.

    Returns:
        sub_label -> (R_sub, Z_sub, dR_sub, dZ_sub)

    Important:
    - dR_sub and dZ_sub are kept as geometry metadata.
    - They should be used for area weights.
    - They should NOT be passed positionally to MultiCoil as current/turns.
    """
    label = _normalize_label(label)

    Rc = float(Rc)
    Zc = float(Zc)
    dR = float(dR)
    dZ = float(dZ)

    if dR <= 0.0 or dZ <= 0.0:
        raise ValueError(
            f"Invalid coil dimensions for {label}: "
            f"Rc={Rc}, Zc={Zc}, dR={dR}, dZ={dZ}"
        )

    if not bool(getattr(opts, "coil_discretize_active", True)):
        return {label: (Rc, Zc, dR, dZ)}

    nR, nZ = _active_coil_discretization_shape(label, dR, dZ, opts)

    if nR == 1 and nZ == 1:
        return {label: (Rc, Zc, dR, dZ)}

    Rmin = Rc - dR
    Rmax = Rc + dR
    Zmin = Zc - dZ
    Zmax = Zc + dZ

    # Cell centers.
    Rs = np.linspace(Rmin, Rmax, nR + 2)[1:-1]
    Zs = np.linspace(Zmin, Zmax, nZ + 2)[1:-1]

    # Half-size of each subcell.
    sub_dR = dR / float(nR)
    sub_dZ = dZ / float(nZ)

    out: Dict[str, Tuple[float, float, float, float]] = {}

    k = 0
    for i, R in enumerate(Rs):
        for j, Z in enumerate(Zs):
            k += 1
            sub_label = f"{label}_F{k:03d}"
            out[sub_label] = (
                float(R),
                float(Z),
                float(sub_dR),
                float(sub_dZ),
            )

    return out


def _add_discretized_active_coil(
    coils: Dict[str, Tuple[float, float, float, float]],
    label: str,
    Rc: float,
    Zc: float,
    dR: float,
    dZ: float,
    *,
    opts: CADImportOptions,
) -> None:
    """
    Add one CAD active coil to the coils dictionary, either as a single element
    or as discretized sub-elements.
    """
    subcoils = _discretize_rect_coil_to_filaments(
        label,
        float(Rc),
        float(Zc),
        float(dR),
        float(dZ),
        opts=opts,
    )

    for sub_label, pack in subcoils.items():
        if sub_label in coils:
            raise ValueError(f"Duplicate active coil label after discretization: {sub_label}")
        coils[sub_label] = pack

def _subdivide_open_polyline_by_maxseg(poly_open: np.ndarray, max_len: float) -> np.ndarray:
    """
    Densify an OPEN polyline so that each segment length <= max_len.
    Keeps endpoints, inserts linear points.
    """
    P = np.asarray(poly_open, float)
    if len(P) < 2:
        return P
    Lmax = float(max_len)
    if not np.isfinite(Lmax) or Lmax <= 0:
        return P

    out = [P[0].copy()]
    for i in range(len(P) - 1):
        A = P[i]
        B = P[i + 1]
        d = B - A
        seg = float(np.linalg.norm(d))
        if seg <= 1e-15:
            continue
        nsub = int(np.ceil(seg / Lmax))
        nsub = max(1, nsub)
        for k in range(1, nsub + 1):
            t = k / nsub
            out.append(A + t * d)
    out = np.asarray(out, float)
    out = _dedupe_sequential(out)
    return out


def _subdivide_closed_polyline_by_maxseg(xy_closed: np.ndarray, max_len: float) -> np.ndarray:
    """
    Densify CLOSED polyline using max segment length; returns CLOSED polyline.
    """
    Popen = _drop_duplicate_endpoint(xy_closed)
    if len(Popen) < 2:
        return _ensure_closed(Popen)
    Pwrap = np.vstack([Popen, Popen[0]])
    Psub_open = _subdivide_open_polyline_by_maxseg(Pwrap, max_len)
    Psub_open = _drop_duplicate_endpoint(Psub_open)
    return _ensure_closed(Psub_open)


def _entity_to_xy(entity, opts: CADImportOptions) -> np.ndarray:
    """
    Extract XY vertices from LWPOLYLINE / POLYLINE / LINE.
    Fallback: flatten via ezdxf.path for SPLINE/ARC/ELLIPSE if available.
    """
    et = entity.dxftype()

    # If requested, prefer path flattening even for polylines (handles bulges).
    if bool(getattr(opts, "prefer_path_flattening", False)) and ezpath is not None:
        try:
            p = ezpath.make_path(entity)
            pts = [(float(v.x), float(v.y)) for v in p.flattening(distance=float(opts.flatten_distance))]
            if len(pts) >= 2:
                return np.array(pts, dtype=float)
        except Exception:
            pass

    if et == "LWPOLYLINE":
        pts = [(p[0], p[1]) for p in entity.get_points("xy")]
        return np.array(pts, dtype=float)

    if et == "POLYLINE":
        verts = getattr(entity, "vertices", None)
        if callable(verts):
            verts = verts()
        if verts is None:
            try:
                verts = entity.vertices()
            except Exception:
                verts = []
        pts = [(float(v.dxf.location.x), float(v.dxf.location.y)) for v in verts]
        return np.array(pts, dtype=float)

    # --- NEW: support LINE (common for windows) ---
    if et == "LINE":
        try:
            x1, y1 = float(entity.dxf.start.x), float(entity.dxf.start.y)
            x2, y2 = float(entity.dxf.end.x), float(entity.dxf.end.y)
            return np.array([[x1, y1], [x2, y2]], dtype=float)
        except Exception:
            pass

    if ezpath is not None:
        try:
            p = ezpath.make_path(entity)
            pts = [(float(v.x), float(v.y)) for v in p.flattening(distance=float(opts.flatten_distance))]
            return np.array(pts, dtype=float)
        except Exception:
            pass

    raise ValueError(f"Unsupported entity type '{et}'. Use (LW)POLYLINE/LINE in DXF.")

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


def _strip_auto_suffix(label: str) -> str:
    return re.sub(r"_[0-9]+$", "", str(label).strip().upper())


# def _coil_family(label: str) -> str:
#     lab = _strip_auto_suffix(_normalize_label(label))
#     if lab.startswith("CS"):
#         return "CS"
#     m = re.match(r"^(PF[0-9]+)", lab)
#     if m:
#         return m.group(1)
#     return lab

def _coil_family(label: str) -> str:
    """
    Map detailed CAD/subcoil labels to current-control families.

    Examples:
        PF4U        -> PF4
        PF4U_F001   -> PF4
        PF4L_F017   -> PF4
        CS1M        -> CS
        CS1M_F012   -> CS
    """
    s = _normalize_label(label).upper()

    if s.startswith("CS"):
        return "CS"

    for k in range(1, 10):
        fam = f"PF{k}"
        if s.startswith(fam):
            return fam

    return s

def _area_from_dR_dZ(dR: float, dZ: float) -> float:
    a = 4.0 * float(dR) * float(dZ)
    if not np.isfinite(a) or a <= 0:
        return 1.0
    return a


# -----------------------------
# BLANKET filament generator
# -----------------------------

def _blanket_region_mask(
    pts: np.ndarray,
    outer_open: np.ndarray,
    inner_open: Optional[np.ndarray],
    *,
    margin: float,
    outer_radius: float,
    inner_radius: float,
) -> np.ndarray:
    """
    Region = inside OUTER (shrunken by margin) AND outside INNER (inflated by margin).
    """
    pts = np.asarray(pts, float)
    outer_path = MplPath(outer_open, closed=True)
    inside_outer = outer_path.contains_points(pts, radius=float(outer_radius) - float(margin))

    if inner_open is None or len(inner_open) < 3:
        return inside_outer

    inner_path = MplPath(inner_open, closed=True)
    inside_inner = inner_path.contains_points(pts, radius=float(inner_radius) + float(margin))
    return inside_outer & (~inside_inner)


def _generate_blanket_centers(
    outer_open: np.ndarray,
    inner_open: Optional[np.ndarray],
    *,
    n: int,
    distribution: str,
    seed: int,
    margin: float,
    bins_R: int,
    bins_Z: int,
    pitch_mode: str,
    pitch_R: float,
    pitch_Z: float,
    containment_radius: float,
) -> np.ndarray:
    """
    Returns (n,2) centers (R,Z) approximately uniformly covering blanket region.
    """
    outer_open = np.asarray(outer_open, float)
    inner_open = None if inner_open is None else np.asarray(inner_open, float)

    xmin, xmax, ymin, ymax = _bbox_from_poly(outer_open)
    W = xmax - xmin
    H = ymax - ymin
    if W <= 0 or H <= 0:
        raise ValueError("Outer wall bbox degenerate.")

    rng = np.random.default_rng(int(seed))
    dist = str(distribution).strip().lower()

    outer_r = float(containment_radius)
    inner_r = float(containment_radius)

    def accept(P: np.ndarray) -> np.ndarray:
        return _blanket_region_mask(
            P, outer_open, inner_open,
            margin=margin,
            outer_radius=outer_r,
            inner_radius=inner_r,
        )

    # ---- GRID
    if dist == "grid":
        pm = str(pitch_mode).strip().lower()
        if pm == "manual":
            dx = float(pitch_R)
            dy = float(pitch_Z)
        else:
            area_bbox = W * H
            pitch = np.sqrt(area_bbox / max(1, int(n)) / 1.25)
            dx = dy = max(1e-6, float(pitch))

        for _ in range(12):
            xs = np.arange(xmin + 0.5 * dx, xmax, dx)
            ys = np.arange(ymin + 0.5 * dy, ymax, dy)
            XX, YY = np.meshgrid(xs, ys, indexing="xy")
            P = np.column_stack([XX.ravel(), YY.ravel()])
            M = accept(P)
            Pin = P[M]
            if Pin.shape[0] >= n:
                idx = np.linspace(0, Pin.shape[0] - 1, n).astype(int)
                return Pin[idx]
            dx *= 0.85
            dy *= 0.85

        dist = "stratified"

    # ---- STRATIFIED
    if dist == "stratified":
        if int(bins_R) <= 0 or int(bins_Z) <= 0:
            aspect = W / (H + 1e-30)
            nR = int(np.ceil(np.sqrt(n * aspect)))
            nZ = int(np.ceil(n / max(1, nR)))
        else:
            nR = int(bins_R)
            nZ = int(bins_Z)

        dx = W / max(1, nR)
        dy = H / max(1, nZ)

        out: List[Tuple[float, float]] = []
        for pass_id in range(20):
            cells = [(i, j) for i in range(nR) for j in range(nZ)]
            rng.shuffle(cells)

            for (i, j) in cells:
                x0 = xmin + i * dx
                y0 = ymin + j * dy
                x = x0 + rng.random() * dx
                y = y0 + rng.random() * dy
                P = np.array([[x, y]], float)
                if accept(P)[0]:
                    out.append((float(x), float(y)))
                    if len(out) >= n:
                        return np.asarray(out, float)

            if len(out) < n and pass_id in (3, 7, 11):
                nR = int(np.ceil(nR * 1.25))
                nZ = int(np.ceil(nZ * 1.25))
                dx = W / max(1, nR)
                dy = H / max(1, nZ)

        dist = "random"

    # ---- RANDOM
    out2: List[Tuple[float, float]] = []
    for _ in range(250):
        batch = max(2000, 5 * (n - len(out2)))
        P = np.column_stack([rng.uniform(xmin, xmax, batch), rng.uniform(ymin, ymax, batch)])
        M = accept(P)
        Pin = P[M]
        for p in Pin:
            out2.append((float(p[0]), float(p[1])))
            if len(out2) >= n:
                return np.asarray(out2, float)

    raise ValueError("Could not generate enough blanket filament centers; check walls/margin/filament size.")


def _coil_rect_area_from_pack(pack: Tuple[float, float, float, float]) -> float:
    """
    Area proxy for a rectangular coil/subcoil pack.

    pack = (Rc, Zc, dR, dZ)
    where dR and dZ are half-width and half-height.
    """
    _, _, dR, dZ = pack
    return max(0.0, 4.0 * float(dR) * float(dZ))


def _normalized_area_weights_for_labels(
    coils: Dict[str, Tuple[float, float, float, float]],
    labels: List[str],
) -> Dict[str, float]:
    """
    Build normalized area weights for a list of coil labels.

    The weights satisfy:
        sum(weights.values()) = 1.0

    This is used so a group current, e.g. CS_MID_current, is distributed
    across the filaments belonging to that group.
    """
    labels = [_normalize_label(x) for x in labels]

    areas = {}
    for lab in labels:
        if lab not in coils:
            continue
        areas[lab] = _coil_rect_area_from_pack(coils[lab])

    total = float(sum(areas.values()))

    if total <= 0.0:
        if len(labels) == 0:
            return {}
        w = 1.0 / float(len(labels))
        return {lab: w for lab in labels if lab in coils}

    return {lab: area / total for lab, area in areas.items()}

def _group_packs_by_family(
    packs: Dict[str, Tuple[float, float, float, float]]
) -> Dict[str, List[str]]:
    """
    Group physical CAD coils or filament packs by current-control family.
    """
    groups: Dict[str, List[str]] = {}
    for lab in packs.keys():
        fam = _coil_family(lab)
        groups.setdefault(fam, []).append(_normalize_label(lab))
    return groups


def _safe_pack_area_m2(pack: Tuple[float, float, float, float]) -> float:
    """
    Geometric cross-section area of a rectangular winding pack/subpack.
    pack = (Rc, Zc, dR, dZ), with dR/dZ half-extents.
    """
    try:
        _, _, dR, dZ = pack
        area = 4.0 * float(dR) * float(dZ)
        if np.isfinite(area) and area > 0.0:
            return float(area)
    except Exception:
        pass
    return 0.0


def _compute_imax_metrics_from_physical_packs(
    physical_packs: Dict[str, Tuple[float, float, float, float]],
    opts: CADImportOptions,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Compute recommended Imax using physical CAD coil rectangles, not the
    discretized filament rectangles.

    This fixes the bug where family_mode='min' becomes the area of the
    smallest filament instead of the area of the smallest real coil.
    """
    fill = float(getattr(opts, "fill_factor", 0.75))
    Jdef = float(getattr(opts, "Jeng_default_A_per_mm2", 40.0))
    fam_mode = str(getattr(opts, "family_mode", "min")).strip().lower()
    J_over = getattr(opts, "family_J_override_A_per_mm2", None) or {}

    family_groups = _group_packs_by_family(physical_packs)

    coil_family_metrics: Dict[str, Dict[str, float]] = {}
    Imax_recommended_MA: Dict[str, float] = {}

    for fam, labs in family_groups.items():
        areas_m2 = []
        for lab in labs:
            pack = physical_packs.get(lab)
            if pack is None:
                continue
            area = _safe_pack_area_m2(pack)
            if area > 0.0:
                areas_m2.append(area)

        areas_m2 = np.asarray(areas_m2, dtype=float)

        if len(areas_m2) == 0:
            A_min_m2 = 0.0
            A_sum_m2 = 0.0
        else:
            A_min_m2 = float(np.min(areas_m2))
            A_sum_m2 = float(np.sum(areas_m2))

        Aeff_min_m2 = fill * A_min_m2
        Aeff_sum_m2 = fill * A_sum_m2

        Aeff_min_mm2 = Aeff_min_m2 * 1.0e6
        Aeff_sum_mm2 = Aeff_sum_m2 * 1.0e6

        J = float(J_over.get(fam, Jdef))

        if fam_mode == "sum":
            Aeff_mm2 = Aeff_sum_mm2
        else:
            Aeff_mm2 = Aeff_min_mm2

        Imax_A = J * Aeff_mm2
        Imax_MA = Imax_A / 1.0e6

        coil_family_metrics[fam] = {
            "n_physical_coils": float(len(labs)),
            "A_min_m2": float(A_min_m2),
            "A_sum_m2": float(A_sum_m2),
            "fill_factor": float(fill),
            "Aeff_min_mm2": float(Aeff_min_mm2),
            "Aeff_sum_mm2": float(Aeff_sum_mm2),
            "J_A_per_mm2": float(J),
            "family_mode_min": 1.0 if fam_mode != "sum" else 0.0,
            "family_mode_sum": 1.0 if fam_mode == "sum" else 0.0,
            "area_basis_physical_cad": 1.0,
            "Imax_recommended_MA": float(Imax_MA),
        }

        Imax_recommended_MA[fam] = float(Imax_MA)

    return coil_family_metrics, Imax_recommended_MA


def _add_segmented_group_imax_metrics(
    geom: Dict,
    opts: CADImportOptions,
) -> None:
    """
    Add effective Imax estimates for artificial segmented groups such as
    CS_MID and CS_END.

    These are not independent physical CAD coils. Their area basis is the
    sum of the discretized subcoil areas belonging to each group.
    """
    coils = geom.get("coils", {}) or {}
    coil_groups = geom.get("coil_groups", {}) or {}

    coil_family_metrics = geom.setdefault("coil_family_metrics", {})
    Imax_recommended_MA = geom.setdefault("Imax_recommended_MA", {})

    fill = float(getattr(opts, "fill_factor", 0.75))
    Jdef = float(getattr(opts, "Jeng_default_A_per_mm2", 40.0))
    J_over = getattr(opts, "family_J_override_A_per_mm2", None) or {}

    for fam in ["CS_MID", "CS_END"]:
        labs = list(coil_groups.get(fam, []))
        if not labs:
            continue

        areas_m2 = []
        for lab in labs:
            pack = coils.get(_normalize_label(lab))
            if pack is None:
                continue
            area = _safe_pack_area_m2(pack)
            if area > 0.0:
                areas_m2.append(area)

        areas_m2 = np.asarray(areas_m2, dtype=float)

        A_sum_m2 = float(np.sum(areas_m2)) if len(areas_m2) else 0.0
        A_min_m2 = float(np.min(areas_m2)) if len(areas_m2) else 0.0

        Aeff_sum_m2 = fill * A_sum_m2
        Aeff_min_m2 = fill * A_min_m2

        Aeff_sum_mm2 = Aeff_sum_m2 * 1.0e6
        Aeff_min_mm2 = Aeff_min_m2 * 1.0e6

        # By default use CS J override for CS_MID/CS_END if present.
        J = float(J_over.get(fam, J_over.get("CS", Jdef)))

        Imax_A = J * Aeff_sum_mm2
        Imax_MA = Imax_A / 1.0e6

        coil_family_metrics[fam] = {
            "n_filaments": float(len(labs)),
            "A_min_m2": float(A_min_m2),
            "A_sum_m2": float(A_sum_m2),
            "fill_factor": float(fill),
            "Aeff_min_mm2": float(Aeff_min_mm2),
            "Aeff_sum_mm2": float(Aeff_sum_mm2),
            "J_A_per_mm2": float(J),
            "family_mode_min": 0.0,
            "family_mode_sum": 1.0,
            "area_basis_segmented_filament_sum": 1.0,
            "Imax_recommended_MA": float(Imax_MA),
        }

        Imax_recommended_MA[fam] = float(Imax_MA)


def _attach_imax_metrics_to_geom(
    geom: Dict,
    opts: CADImportOptions,
) -> None:
    """
    Attach corrected Imax metrics to geom.

    Parent families use original physical CAD rectangles.
    Artificial segmented groups use partitioned filament area sums.
    """
    physical_packs = (
        geom.get("base_coils", None)
        or geom.get("physical_coils", None)
        or geom.get("coil_base_rects", None)
        or geom.get("coils", {})
    )

    metrics, imax = _compute_imax_metrics_from_physical_packs(
        physical_packs,
        opts,
    )

    geom["coil_family_metrics"] = metrics
    geom["Imax_recommended_MA"] = imax

    geom["Imax_assumptions"] = {
        "fill_factor": float(getattr(opts, "fill_factor", 0.75)),
        "Jeng_default_A_per_mm2": float(getattr(opts, "Jeng_default_A_per_mm2", 40.0)),
        "family_mode": str(getattr(opts, "family_mode", "min")),
        "area_basis": "physical_CAD_rectangles_for_parent_families",
        "segmented_groups_area_basis": "sum_of_discretized_filament_areas",
    }

    _add_segmented_group_imax_metrics(geom, opts)

def _add_segmented_cs_groups_to_geom(
    geom: Dict,
    opts: CADImportOptions,
) -> None:
    """
    Add effective segmented-CS current-control groups to geom.

    This does not change the CAD geometry and does not create new coils.
    It only partitions the already-discretized CS filaments into:

        CS_MID : central part of the CS around Z = 0
        CS_END : upper + lower ends of the CS, kept symmetric

    Recommended initial partition:
        cs_mid_fraction = 0.45

    Meaning:
        central 45% of the full CS height -> CS_MID
        remaining 55% -> CS_END
    """
    if not bool(getattr(opts, "cs_segmented", False)):
        return

    coils = geom.get("coils", {})
    if not coils:
        return

    coil_groups = geom.setdefault("coil_groups", {})
    coil_group_weights = geom.setdefault("coil_group_weights", {})

    # Prefer existing CS group if already built.
    cs_labels = list(coil_groups.get("CS", []))

    # Fallback: detect CS filaments from labels.
    if not cs_labels:
        cs_labels = [
            _normalize_label(k)
            for k in coils.keys()
            if _coil_family(k) == "CS"
        ]

    cs_labels = [_normalize_label(x) for x in cs_labels]

    if not cs_labels:
        raise ValueError(
            "cs_segmented=True was requested, but no CS filaments were found."
        )

    # Estimate full CS half-height using filament extents.
    z_extent = 0.0
    for lab in cs_labels:
        Rc, Zc, dR, dZ = coils[lab]
        z_extent = max(z_extent, abs(float(Zc)) + abs(float(dZ)))

    if z_extent <= 0.0:
        raise ValueError(
            "Could not determine CS vertical extent for segmented-CS model."
        )

    zcut_abs = getattr(opts, "cs_segment_zcut_m", None)

    if zcut_abs is None:
        mid_frac = float(getattr(opts, "cs_mid_fraction", 0.45))
        mid_frac = max(0.05, min(mid_frac, 0.95))
        zcut = mid_frac * z_extent
    else:
        zcut = abs(float(zcut_abs))

    cs_mid = []
    cs_end = []

    for lab in cs_labels:
        Rc, Zc, dR, dZ = coils[lab]
        if abs(float(Zc)) <= zcut:
            cs_mid.append(lab)
        else:
            cs_end.append(lab)

    if not cs_mid:
        raise ValueError(
            f"CS_MID group is empty. zcut={zcut:.4f} m is too small."
        )

    if not cs_end:
        raise ValueError(
            f"CS_END group is empty. zcut={zcut:.4f} m is too large."
        )

    # Optional: remove parent CS group if you want fully segmented control only.
    # I recommend keeping it for backward compatibility.
    keep_parent = bool(getattr(opts, "cs_segment_keep_parent", True))
    if not keep_parent:
        coil_groups.pop("CS", None)
        coil_group_weights.pop("CS", None)

    coil_groups["CS_MID"] = list(cs_mid)
    coil_groups["CS_END"] = list(cs_end)

    coil_group_weights["CS_MID"] = _normalized_area_weights_for_labels(coils, cs_mid)
    coil_group_weights["CS_END"] = _normalized_area_weights_for_labels(coils, cs_end)

    geom["cs_segment_info"] = {
        "enabled": True,
        "z_extent_m": float(z_extent),
        "zcut_m": float(zcut),
        "mid_fraction_effective": float(zcut / z_extent),
        "n_mid": int(len(cs_mid)),
        "n_end": int(len(cs_end)),
        "keep_parent": bool(keep_parent),
    }

def _build_blanket_filaments(
    outer_xy_closed: np.ndarray,
    inner_xy_closed: Optional[np.ndarray],
    *,
    opts: CADImportOptions,
) -> List[Tuple[str, float, float, float, float]]:
    """
    Returns list of (label, Rc, Zc, dR, dZ)
    """
    n = int(getattr(opts, "blanket_n_filaments", 0))
    if n <= 0:
        return []

    dR = float(getattr(opts, "blanket_filament_dR", 0.004))
    dZ = float(getattr(opts, "blanket_filament_dZ", 0.004))

    wall_margin = float(getattr(opts, "blanket_wall_margin_m", 0.0))
    eff_margin = wall_margin + 1.05 * max(dR, dZ)

    outer_open = _drop_duplicate_endpoint(outer_xy_closed)
    inner_open = None if inner_xy_closed is None else _drop_duplicate_endpoint(inner_xy_closed)

    centers = _generate_blanket_centers(
        outer_open,
        inner_open,
        n=n,
        distribution=str(getattr(opts, "blanket_distribution", "stratified")),
        seed=int(getattr(opts, "blanket_seed", 0)),
        margin=eff_margin,
        bins_R=int(getattr(opts, "blanket_bins_R", 0)),
        bins_Z=int(getattr(opts, "blanket_bins_Z", 0)),
        pitch_mode=str(getattr(opts, "blanket_pitch_mode", "auto")),
        pitch_R=float(getattr(opts, "blanket_pitch_R", 0.03)),
        pitch_Z=float(getattr(opts, "blanket_pitch_Z", 0.03)),
        containment_radius=float(getattr(opts, "blanket_containment_radius", -1e-9)),
    )

    prefix = str(getattr(opts, "blanket_label_prefix", "BLK")).strip().upper()
    fil = []
    for i, (Rc, Zc) in enumerate(centers, start=1):
        lab = f"{prefix}{i:05d}"
        fil.append((lab, float(Rc), float(Zc), float(dR), float(dZ)))
    return fil



# -----------------------------
# Passive STAR_VESSEL / structural filaments
# -----------------------------

def _prepare_closed_cad_polyline(
    xy: np.ndarray,
    *,
    unit_scale: float,
    opts: CADImportOptions,
    n_resample: Optional[int] = None,
) -> np.ndarray:
    """
    Scale, close, orient, densify, and optionally resample one CAD closed polyline.
    """
    P = np.asarray(xy, float) * float(unit_scale)
    P = _ensure_closed(_dedupe_sequential(P))

    if opts.enforce_ccw:
        P = _enforce_ccw(P)
    if opts.canonical_start:
        P = _rotate_to_outboard_midplane(P)

    maxseg = float(getattr(opts, "max_seg_len_wall", 0.0))
    if np.isfinite(maxseg) and maxseg > 0.0:
        P = _subdivide_closed_polyline_by_maxseg(P, maxseg)

    if n_resample is not None and int(n_resample) > 0:
        P = _maybe_resample(
            P,
            int(n_resample),
            str(getattr(opts, "resample_walls", "auto")),
            int(getattr(opts, "min_wall_pts", 400)),
        )

    return _ensure_closed(_dedupe_sequential(P))


def _collect_closed_polygons_from_layer(
    msp,
    layer_name: str,
    *,
    unit_scale: float,
    opts: CADImportOptions,
    min_abs_area_m2: float = 1.0e-8,
) -> List[np.ndarray]:
    """
    Collect all closed polygon-like entities from a DXF layer.
    Returns closed, scaled, CCW polylines in meters.
    """
    polys: List[np.ndarray] = []
    for e in msp.query(f'*[layer=="{layer_name}"]'):
        try:
            xy = _entity_to_xy(e, opts)
            P = _prepare_closed_cad_polyline(
                xy,
                unit_scale=unit_scale,
                opts=opts,
                n_resample=None,
            )
            if len(_drop_duplicate_endpoint(P)) >= 3 and abs(_polygon_area(P)) >= float(min_abs_area_m2):
                polys.append(P)
        except Exception:
            continue
    return polys


def _collect_passive_polygons(
    msp,
    layers: CADLayers,
    *,
    unit_scale: float,
    opts: CADImportOptions,
) -> List[Dict[str, Any]]:
    """
    Read STAR_VESSEL and PASSIVE_* polygon layers as passive structural solids.
    """
    out: List[Dict[str, Any]] = []

    if bool(getattr(opts, "passive_use_star_vessel", True)):
        layer = str(getattr(layers, "star_vessel", "STAR_VESSEL"))
        for i, poly in enumerate(
            _collect_closed_polygons_from_layer(msp, layer, unit_scale=unit_scale, opts=opts),
            start=1,
        ):
            out.append({
                "layer": layer,
                "name": f"STAR_VESSEL_{i:03d}",
                "xy": poly,
                "material": str(getattr(opts, "star_vessel_material", "SS316L")),
                "resistivity_ohm_m": float(getattr(opts, "star_vessel_resistivity_ohm_m", 0.75e-6)),
            })

    if bool(getattr(opts, "passive_use_passive_prefix", True)):
        prefix = str(getattr(layers, "passive_layer_prefix", "PASSIVE_")).upper()
        layer_names = sorted({str(e.dxf.layer) for e in msp if str(e.dxf.layer).upper().startswith(prefix)})
        for layer in layer_names:
            for i, poly in enumerate(
                _collect_closed_polygons_from_layer(msp, layer, unit_scale=unit_scale, opts=opts),
                start=1,
            ):
                lname = _normalize_label(layer)
                out.append({
                    "layer": layer,
                    "name": f"{lname}_{i:03d}",
                    "xy": poly,
                    "material": str(getattr(opts, "passive_default_material", "SS316L")),
                    "resistivity_ohm_m": float(getattr(opts, "passive_default_resistivity_ohm_m", 0.75e-6)),
                })

    return out


def _discretize_passive_polygon(
    poly_closed: np.ndarray,
    *,
    label_prefix: str,
    material: str,
    resistivity_ohm_m: float,
    opts: CADImportOptions,
) -> List[Tuple[str, float, float, float, float, str, float]]:
    """
    Discretize one passive CAD polygon into rectangular filament cells.

    Returns:
        [(label, Rc, Zc, dR, dZ, material, resistivity_ohm_m), ...]

    These are passive structural elements: current=0, control=False.
    """
    P = _drop_duplicate_endpoint(np.asarray(poly_closed, float))
    if len(P) < 3:
        return []

    xmin, xmax, zmin, zmax = _bbox_from_poly(P)
    width = xmax - xmin
    height = zmax - zmin
    if width <= 0.0 or height <= 0.0:
        return []

    target_R = max(1.0e-6, float(getattr(opts, "passive_target_dR_m", 0.10)))
    target_Z = max(1.0e-6, float(getattr(opts, "passive_target_dZ_m", 0.10)))

    nR = int(np.ceil(width / target_R))
    nZ = int(np.ceil(height / target_Z))
    nR = max(1, min(nR, int(getattr(opts, "passive_nR_max", 80))))
    nZ = max(1, min(nZ, int(getattr(opts, "passive_nZ_max", 160))))

    cell_w = width / float(nR)
    cell_h = height / float(nZ)
    dR = 0.5 * cell_w
    dZ = 0.5 * cell_h

    if cell_w * cell_h < float(getattr(opts, "passive_min_cell_area_m2", 1.0e-6)):
        return []

    xs = xmin + (np.arange(nR) + 0.5) * cell_w
    zs = zmin + (np.arange(nZ) + 0.5) * cell_h
    XX, ZZ = np.meshgrid(xs, zs, indexing="xy")
    pts = np.column_stack([XX.ravel(), ZZ.ravel()])

    path = MplPath(P, closed=True)
    radius = float(getattr(opts, "passive_containment_radius", -1e-9))
    inside = path.contains_points(pts, radius=radius)

    fil: List[Tuple[str, float, float, float, float, str, float]] = []
    k = 0
    prefix = _normalize_label(label_prefix)
    for (R, Z), ok in zip(pts, inside):
        if not bool(ok):
            continue
        k += 1
        lab = f"{prefix}_P{k:04d}"
        fil.append((lab, float(R), float(Z), float(dR), float(dZ), str(material), float(resistivity_ohm_m)))

    return fil


def _build_passive_filaments_from_structures(
    passive_structures: List[Dict[str, Any]],
    *,
    opts: CADImportOptions,
) -> List[Tuple[str, float, float, float, float, str, float]]:
    """
    Convert CAD passive structures into passive filament cells.
    """
    filaments: List[Tuple[str, float, float, float, float, str, float]] = []
    used: Set[str] = set()

    for s in passive_structures:
        base = _normalize_label(s.get("name", "PASSIVE"))
        poly = np.asarray(s.get("xy"), float)
        material = str(s.get("material", getattr(opts, "passive_default_material", "SS316L")))
        rho = float(s.get("resistivity_ohm_m", getattr(opts, "passive_default_resistivity_ohm_m", 0.75e-6)))
        local = _discretize_passive_polygon(
            poly,
            label_prefix=base,
            material=material,
            resistivity_ohm_m=rho,
            opts=opts,
        )
        for lab, Rc, Zc, dR, dZ, mat, rr in local:
            lab0 = lab
            kk = 1
            while lab in used:
                kk += 1
                lab = f"{lab0}_{kk}"
            used.add(lab)
            filaments.append((lab, Rc, Zc, dR, dZ, mat, rr))

    return filaments


def _passive_structures_bbox(passive_structures: List[Dict[str, Any]]) -> Optional[Tuple[float, float, float, float]]:
    """
    Combined bbox for passive structures.
    """
    bbs = []
    for s in passive_structures:
        try:
            bbs.append(_bbox_from_poly(np.asarray(s["xy"], float)))
        except Exception:
            continue
    if not bbs:
        return None
    return (
        float(min(b[0] for b in bbs)),
        float(max(b[1] for b in bbs)),
        float(min(b[2] for b in bbs)),
        float(max(b[3] for b in bbs)),
    )

# -----------------------------
# Plasma geometry pack
# -----------------------------

def _poly_perimeter(xy_closed: np.ndarray) -> float:
    pts = _ensure_closed(np.asarray(xy_closed, float))
    if len(pts) < 4:
        return 0.0
    d = pts[1:] - pts[:-1]
    return float(np.sum(np.sqrt(np.sum(d * d, axis=1))))


def _poly_centroid(xy_closed: np.ndarray) -> Tuple[float, float]:
    pts = _ensure_closed(np.asarray(xy_closed, float))
    if len(pts) < 4:
        c = np.mean(pts, axis=0) if len(pts) else np.array([np.nan, np.nan])
        return float(c[0]), float(c[1])

    x = pts[:, 0]
    y = pts[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    A2 = float(np.sum(cross))  # 2*Area signed
    if abs(A2) < 1e-14:
        c = np.mean(pts[:-1], axis=0)
        return float(c[0]), float(c[1])

    cx = float(np.sum((x[:-1] + x[1:]) * cross) / (3.0 * A2))
    cy = float(np.sum((y[:-1] + y[1:]) * cross) / (3.0 * A2))
    return cx, cy


def _horizontal_intersections_R(poly_open: np.ndarray, Zc: float, tol: float = 1e-12) -> np.ndarray:
    P = np.asarray(poly_open, float)
    if len(P) < 3:
        return np.array([], float)

    R1 = P[:, 0]
    Z1 = P[:, 1]
    R2 = np.roll(R1, -1)
    Z2 = np.roll(Z1, -1)

    out = []
    for r1, z1, r2, z2 in zip(R1, Z1, R2, Z2):
        dz = z2 - z1
        if abs(dz) < tol:
            continue
        t = (Zc - z1) / dz
        if t < -1e-12 or t > 1.0 + 1e-12:
            continue
        t = min(1.0, max(0.0, t))
        r = r1 + t * (r2 - r1)
        out.append(float(r))

    if not out:
        return np.array([], float)

    out = np.array(sorted(out), float)
    keep = [out[0]]
    for v in out[1:]:
        if abs(v - keep[-1]) > 1e-9:
            keep.append(v)
    return np.array(keep, float)


def compute_plasma_geom_params(
    xy_closed: np.ndarray,
    *,
    Z0_ref: Optional[float] = None,
    R0_ref: Optional[float] = None,
) -> Dict[str, float]:
    pts = _drop_duplicate_endpoint(np.asarray(xy_closed, float))
    if len(pts) < 3:
        return {"valid": 0.0}

    R = pts[:, 0]
    Z = pts[:, 1]
    Rmin, Rmax = float(np.min(R)), float(np.max(R))
    Zmin, Zmax = float(np.min(Z)), float(np.max(Z))

    R0_bbox = 0.5 * (Rmin + Rmax)
    Z0_bbox = 0.5 * (Zmin + Zmax)

    Z0_use = float(Z0_ref) if Z0_ref is not None else Z0_bbox

    interR = _horizontal_intersections_R(pts, Z0_use)
    if len(interR) >= 2:
        Rin_mid = float(np.min(interR))
        Rout_mid = float(np.max(interR))
    else:
        Rin_mid = float(Rmin)
        Rout_mid = float(Rmax)

    R0_mid = 0.5 * (Rin_mid + Rout_mid)
    a_mid = 0.5 * (Rout_mid - Rin_mid)
    b = 0.5 * (Zmax - Zmin)

    kappa_mid = (b / a_mid) if a_mid > 0 else np.nan
    A_mid = (R0_mid / a_mid) if a_mid > 0 else np.nan

    i_top = int(np.argmax(Z))
    i_bot = int(np.argmin(Z))
    R_top = float(R[i_top])
    R_bot = float(R[i_bot])

    R0_for_delta = float(R0_ref) if R0_ref is not None else float(R0_mid)
    delta_u = (R0_for_delta - R_top) / a_mid if a_mid > 0 else np.nan
    delta_l = (R0_for_delta - R_bot) / a_mid if a_mid > 0 else np.nan
    delta_avg = 0.5 * (delta_u + delta_l) if np.isfinite(delta_u) and np.isfinite(delta_l) else np.nan

    area = abs(_polygon_area(pts))
    perim = _poly_perimeter(_ensure_closed(pts))
    Rc_c, Zc_c = _poly_centroid(_ensure_closed(pts))

    vol_torus = (2.0 * np.pi * R0_mid * area) if np.isfinite(R0_mid) else np.nan
    surf_torus = (2.0 * np.pi * R0_mid * perim) if np.isfinite(R0_mid) else np.nan

    return {
        "valid": 1.0,
        "Rmin": Rmin, "Rmax": Rmax, "Zmin": Zmin, "Zmax": Zmax,
        "R0_bbox": R0_bbox, "Z0_bbox": Z0_bbox,
        "Z0_used": Z0_use,
        "Rin_mid": Rin_mid, "Rout_mid": Rout_mid,
        "R0_mid": R0_mid, "a_mid": a_mid, "b": b,
        "kappa_mid": kappa_mid, "A_mid": A_mid,
        "R_top": R_top, "R_bot": R_bot,
        "delta_u": delta_u, "delta_l": delta_l, "delta_avg": delta_avg,
        "area_poloidal": float(area),
        "perimeter_poloidal": float(perim),
        "Rc_centroid": float(Rc_c),
        "Zc_centroid": float(Zc_c),
        "volume_torus_approx": float(vol_torus),
        "surface_torus_approx": float(surf_torus),
    }


# -----------------------------
# AUTO plasma target
# -----------------------------

def _miller_boundary_ud(
    *,
    R0: float,
    a: float,
    kappa: float,
    delta_u: float,
    delta_l: float,
    Z0: float,
    n: int,
    scale: float,
) -> np.ndarray:
    R0 = float(R0); a = float(a); kappa = float(kappa)
    du = float(delta_u); dl = float(delta_l)
    Z0 = float(Z0); n = int(max(160, n)); s = float(scale)

    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    st = np.sin(t)
    dt = np.where(st >= 0.0, du, dl)

    aa = s * a
    bb = s * kappa * a

    R = R0 + aa * np.cos(t + dt * st)
    Z = Z0 + bb * np.sin(t)

    xy = np.column_stack([R, Z])
    xy = _ensure_closed(_dedupe_sequential(xy))
    xy = _enforce_ccw(xy)
    xy = _rotate_to_outboard_midplane(xy)
    return xy


def _fits_inside_wall_strict(
    xy_plasma: np.ndarray,
    *,
    wall_open: np.ndarray,
    containment_radius: float,
) -> bool:
    pts = _drop_duplicate_endpoint(xy_plasma)
    if len(pts) == 0:
        return False
    wall_path = MplPath(wall_open, closed=True)
    inside = wall_path.contains_points(pts, radius=float(containment_radius))
    return bool(np.all(inside))


def _max_scale_for_delta_fast(
    *,
    wall_open: np.ndarray,
    R0: float,
    a: float,
    kappa: float,
    du: float,
    dl: float,
    Z0: float,
    n: int,
    iters: int,
    containment_radius: float,
) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        xy = _miller_boundary_ud(R0=R0, a=a, kappa=kappa, delta_u=du, delta_l=dl, Z0=Z0, n=n, scale=mid)
        ok = _fits_inside_wall_strict(xy, wall_open=wall_open, containment_radius=containment_radius)
        if ok:
            lo = mid
        else:
            hi = mid
    return float(lo)


def _find_interior_point_near_target(
    wall_open: np.ndarray,
    target: Tuple[float, float],
    *,
    samples: int,
    seed: int,
) -> Tuple[float, float]:
    wall_open = np.asarray(wall_open, float)
    wall_path = MplPath(wall_open, closed=True)

    xmin, xmax, ymin, ymax = _bbox_from_poly(wall_open)
    rng = np.random.default_rng(int(seed))
    tgt = np.array([float(target[0]), float(target[1])], dtype=float)

    best = None  # (dist2, point)
    for _ in range(int(samples)):
        p = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)], dtype=float)
        if wall_path.contains_point((float(p[0]), float(p[1]))):
            d2 = float(np.sum((p - tgt) ** 2))
            if (best is None) or (d2 < best[0]):
                best = (d2, p)

    if best is None:
        c = np.mean(wall_open, axis=0)
        return float(c[0]), float(c[1])
    return float(best[1][0]), float(best[1][1])


def _estimate_from_boundary(xy_closed: np.ndarray) -> Dict[str, float]:
    pts = _drop_duplicate_endpoint(xy_closed)
    R = pts[:, 0]; Z = pts[:, 1]
    Rmin, Rmax = float(np.min(R)), float(np.max(R))
    Zmin, Zmax = float(np.min(Z)), float(np.max(Z))
    R0 = 0.5 * (Rmin + Rmax)
    a = 0.5 * (Rmax - Rmin)
    kappa = (Zmax - Zmin) / (2.0 * a) if a > 0 else np.nan
    A = R0 / a if a > 0 else np.nan

    i_top = int(np.argmax(Z))
    i_bot = int(np.argmin(Z))
    R_top = float(R[i_top])
    R_bot = float(R[i_bot])

    du = (R0 - R_top) / a if a > 0 else np.nan
    dl = (R0 - R_bot) / a if a > 0 else np.nan

    return {
        "R0_est": float(R0),
        "a_est": float(a),
        "A_est": float(A),
        "kappa_est": float(kappa),
        "delta_u_est": float(du),
        "delta_l_est": float(dl),
        "Zmin": float(Zmin),
        "Zmax": float(Zmax),
    }


def _auto_plasma_target_fit_inner(
    *,
    wall_open: np.ndarray,
    R0_t: float,
    A_t: float,
    kappa_t: float,
    Z0_t: float,
    n: int,
    delta_max: float,
    delta_grid: int,
    symmetric: bool,
    iters: int,
    containment_radius: float,
    scale_safety: float,
    fix_center_if_outside: bool,
    center_samples: int,
    center_seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    wall_open = np.asarray(wall_open, float)
    if len(wall_open) < 3:
        raise ValueError("Fit wall has <3 points. Check WALL_INNER/WALL_OUTER layer import.")

    wall_path = MplPath(wall_open, closed=True)

    R0_use = float(R0_t)
    Z0_use = float(Z0_t)

    if fix_center_if_outside and (not wall_path.contains_point((R0_use, Z0_use))):
        R0_use, Z0_use = _find_interior_point_near_target(
            wall_open, (R0_use, Z0_use), samples=center_samples, seed=center_seed
        )

    a_nom = float(R0_use) / float(A_t)
    deltas = np.linspace(0.0, float(delta_max), int(max(3, delta_grid)))

    best = None  # (score_tuple, du, dl, scale)
    for du in deltas:
        dl_list = [float(du)] if symmetric else deltas
        for dl in dl_list:
            s = _max_scale_for_delta_fast(
                wall_open=wall_open,
                R0=R0_use, a=a_nom, kappa=float(kappa_t),
                du=float(du), dl=float(dl), Z0=Z0_use,
                n=int(n), iters=int(iters),
                containment_radius=float(containment_radius),
            )
            if s <= 1e-8:
                continue

            score = (s, 0.10 * (du + dl), -0.02 * abs(du - dl))
            if (best is None) or (score > best[0]):
                best = (score, float(du), float(dl), float(s))

    if best is None:
        raise ValueError("AUTO plasma target: no feasible delta/scale. Check wall orientation / self-intersection.")

    _, du_best, dl_best, s_best = best
    s_final = float(scale_safety) * float(s_best)

    xy = _miller_boundary_ud(
        R0=R0_use, a=a_nom, kappa=float(kappa_t),
        delta_u=du_best, delta_l=dl_best,
        Z0=Z0_use, n=int(n), scale=s_final
    )

    if not _fits_inside_wall_strict(xy, wall_open=wall_open, containment_radius=float(containment_radius)):
        xy = _miller_boundary_ud(
            R0=R0_use, a=a_nom, kappa=float(kappa_t),
            delta_u=du_best, delta_l=dl_best,
            Z0=Z0_use, n=int(n), scale=float(0.995 * s_final)
        )

    meta: Dict[str, float] = {
        "mode": 1.0,
        "R0_target": float(R0_t),
        "A_target": float(A_t),
        "kappa_target": float(kappa_t),
        "Z0_target": float(Z0_t),
        "R0_used": float(R0_use),
        "Z0_used": float(Z0_use),
        "center_moved": 1.0 if (abs(R0_use - R0_t) > 1e-9 or abs(Z0_use - Z0_t) > 1e-9) else 0.0,
        "scale": float(s_final),
        "delta_u": float(du_best),
        "delta_l": float(dl_best),
    }

    meta.update(_estimate_from_boundary(xy))

    geom_pack = compute_plasma_geom_params(
        xy,
        Z0_ref=float(Z0_use),
        R0_ref=float(R0_t),
    )
    for k, v in geom_pack.items():
        if isinstance(v, (int, float, np.floating)):
            meta[f"geom_{k}"] = float(v)

    A_mid = meta.get("geom_A_mid", np.nan)
    k_mid = meta.get("geom_kappa_mid", np.nan)
    R0_mid = meta.get("geom_R0_mid", np.nan)

    meta["err_R0_mid"] = float(R0_mid - float(R0_t)) if np.isfinite(R0_mid) else np.nan
    meta["err_A_mid"] = float(A_mid - float(A_t)) if np.isfinite(A_mid) else np.nan
    meta["err_kappa_mid"] = float(k_mid - float(kappa_t)) if np.isfinite(k_mid) else np.nan

    meta["rel_err_A_mid"] = float((A_mid - float(A_t)) / float(A_t)) if (np.isfinite(A_mid) and float(A_t) != 0.0) else np.nan
    meta["rel_err_kappa_mid"] = float((k_mid - float(kappa_t)) / float(kappa_t)) if (np.isfinite(k_mid) and float(kappa_t) != 0.0) else np.nan

    return xy, meta


# -----------------------------
# Xpoints + strike lines (geometric markers)
# -----------------------------

def _poly_segments(poly_open: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    P = np.asarray(poly_open, dtype=float)
    return P, np.roll(P, -1, axis=0)


def _ray_segment_intersection(O: np.ndarray, d: np.ndarray, P: np.ndarray, Q: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Solve O + t d = P + u (Q-P), with t>=0, u in [0,1].
    Returns (t,u) if intersects, else None.
    """
    O = np.asarray(O, float); d = np.asarray(d, float)
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    v = Q - P

    A = np.array([[d[0], -v[0]], [d[1], -v[1]]], dtype=float)
    b = P - O
    det = float(np.linalg.det(A))
    if abs(det) < 1e-14:
        return None
    sol = np.linalg.solve(A, b)
    t = float(sol[0]); u = float(sol[1])
    if (t >= 0.0) and (u >= 0.0) and (u <= 1.0):
        return t, u
    return None


def _ray_to_polygon_first_hit(O: np.ndarray, d: np.ndarray, poly_open: np.ndarray) -> Optional[np.ndarray]:
    """
    Return the closest intersection point of ray (O,d) with polygon edges.
    """
    P0, P1 = _poly_segments(poly_open)
    best_t = None
    best_pt = None
    for A, B in zip(P0, P1):
        hit = _ray_segment_intersection(O, d, A, B)
        if hit is None:
            continue
        t, _u = hit
        if (best_t is None) or (t < best_t):
            best_t = t
            best_pt = O + t * d
    return best_pt


def _compute_xpoints_and_strike_lines(
    xy_closed: np.ndarray,
    wall_open: np.ndarray,
    *,
    fallback_len: float,
) -> Tuple[List[Tuple[float, float, str]], List[np.ndarray]]:
    """
    Geometric markers:
      - xpoints = top/bottom extrema of the TARGET boundary
      - strike lines = 2 rays from each extremum towards outer wall
    """
    pts = _drop_duplicate_endpoint(xy_closed)
    wall = np.asarray(wall_open, float)

    i_top = int(np.argmax(pts[:, 1]))
    i_bot = int(np.argmin(pts[:, 1]))

    xpoints: List[Tuple[float, float, str]] = []
    lines: List[np.ndarray] = []

    def _two_rays(i: int, kind: str):
        O = pts[i].copy()
        p_prev = pts[(i - 1) % len(pts)]
        p_next = pts[(i + 1) % len(pts)]

        dirs = []
        for p in (p_prev, p_next):
            v = p - O
            if np.linalg.norm(v) < 1e-12:
                continue

            if kind == "lower":
                v[1] = -abs(v[1])
            else:
                v[1] = abs(v[1])

            if np.linalg.norm(v) < 1e-12:
                continue
            v = v / np.linalg.norm(v)
            dirs.append(v)

        if len(dirs) < 2:
            dirs = [
                np.array([+1.0, -1.0 if kind == "lower" else +1.0]),
                np.array([-1.0, -1.0 if kind == "lower" else +1.0]),
            ]
            dirs = [d / np.linalg.norm(d) for d in dirs]

        for d in dirs[:2]:
            hit = _ray_to_polygon_first_hit(O, d, wall)
            if hit is None:
                hit = O + float(fallback_len) * d
            lines.append(np.vstack([O, hit]))

    xpoints.append((float(pts[i_bot, 0]), float(pts[i_bot, 1]), "lower_geom"))
    _two_rays(i_bot, "lower")

    xpoints.append((float(pts[i_top, 0]), float(pts[i_top, 1]), "upper_geom"))
    _two_rays(i_top, "upper")

    return xpoints, lines


# -----------------------------
# DXF -> geom
# -----------------------------

def load_geom_from_dxf(
    dxf_path: Path,
    layers: CADLayers = CADLayers(),
    opts: CADImportOptions = CADImportOptions(),
) -> Dict:
    """
    Read DXF and return a geom dict.
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

    if opts.enforce_ccw:
        outer_xy = _enforce_ccw(outer_xy)
    if opts.canonical_start:
        outer_xy = _rotate_to_outboard_midplane(outer_xy)

    # Smooth CAD densify
    outer_xy = _subdivide_closed_polyline_by_maxseg(outer_xy, float(getattr(opts, "max_seg_len_wall", 0.0)))

    # Arc-length resample
    outer_xy = _maybe_resample(outer_xy, int(opts.n_wall), str(opts.resample_walls), int(opts.min_wall_pts))

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

        inner_xy = _subdivide_closed_polyline_by_maxseg(inner_xy, float(getattr(opts, "max_seg_len_wall", 0.0)))
        inner_xy = _maybe_resample(inner_xy, int(opts.n_inner), str(opts.resample_walls), int(opts.min_wall_pts))

    # ---- Passive structural polygons / filaments
    # Preferred new model: use real CAD passive solids (STAR_VESSEL and PASSIVE_*),
    # not the old artificial fill between WALL_INNER and WALL_OUTER.
    passive_structures: List[Dict[str, Any]] = []
    passive_filaments: List[Tuple[str, float, float, float, float, str, float]] = []
    if bool(getattr(opts, "passive_structures_enabled", True)):
        passive_structures = _collect_passive_polygons(
            msp,
            layers,
            unit_scale=unit_scale,
            opts=opts,
        )
        passive_filaments = _build_passive_filaments_from_structures(
            passive_structures,
            opts=opts,
        )

    # ---- Legacy blanket filaments (AUTO between inner & outer)
    # Kept only as fallback. With STAR_VESSEL available, leave blanket_enabled=False.
    blanket_filaments = []
    if bool(getattr(opts, "blanket_enabled", False)) and not passive_filaments:
        blanket_filaments = _build_blanket_filaments(outer_xy, inner_xy, opts=opts)

    # ---- Plasma target: CAD or AUTO
    plasma_xy = None
    plasma_candidates = polylines_in_layer(layers.plasma_target)
    if len(plasma_candidates) > 0:
        plasma_xy = max(plasma_candidates, key=lambda xy: abs(_polygon_area(xy)))
        plasma_xy = plasma_xy * unit_scale
        plasma_xy = _ensure_closed(_dedupe_sequential(plasma_xy))
        if opts.enforce_ccw:
            plasma_xy = _enforce_ccw(plasma_xy)
        if opts.canonical_start:
            plasma_xy = _rotate_to_outboard_midplane(plasma_xy)
        plasma_xy = _subdivide_closed_polyline_by_maxseg(plasma_xy, float(getattr(opts, "max_seg_len_plasma", 0.0)))
        plasma_xy = _maybe_resample(plasma_xy, int(opts.n_plasma), str(opts.resample_walls), int(opts.min_wall_pts))

    plasma_meta: Dict[str, float] = {}
    xpoints_target: List[Tuple[float, float, str]] = []
    strike_lines_target: List[np.ndarray] = []
    markers: Dict[str, Any] = {}
    geom_markers_meta: Dict[str, Any] = {"found": [], "layers": {}}

    mode_pt = str(getattr(opts, "plasma_target_mode", "cad")).strip().lower()
    want_auto = (mode_pt == "auto") or (plasma_xy is None and mode_pt == "cad")

    if want_auto:
        fit_to_inner = bool(getattr(opts, "plasma_fit_to_inner_if_available", True)) and (inner_xy is not None)
        fit_wall = inner_xy if fit_to_inner else outer_xy

        wall_open = _drop_duplicate_endpoint(fit_wall)
        if len(wall_open) < 3:
            wall_open = _drop_duplicate_endpoint(outer_xy)
            fit_to_inner = False

        plasma_xy, plasma_meta = _auto_plasma_target_fit_inner(
            wall_open=wall_open,
            R0_t=float(getattr(opts, "plasma_R0", 4.0)),
            A_t=float(getattr(opts, "plasma_A", 2.0)),
            kappa_t=float(getattr(opts, "plasma_kappa", 2.5)),
            Z0_t=float(getattr(opts, "plasma_Z0", 0.0)),
            n=int(getattr(opts, "n_plasma", 501)),
            delta_max=float(getattr(opts, "plasma_delta_max", 0.70)),
            delta_grid=int(getattr(opts, "plasma_delta_grid", 17)),
            symmetric=bool(getattr(opts, "plasma_delta_symmetric", True)),
            iters=int(getattr(opts, "plasma_shrink_iters", 20)),
            containment_radius=float(getattr(opts, "containment_radius", -1e-9)),
            scale_safety=float(getattr(opts, "plasma_scale_safety", 0.999)),
            fix_center_if_outside=bool(getattr(opts, "fix_center_if_outside", True)),
            center_samples=int(getattr(opts, "center_search_samples", 800)),
            center_seed=int(getattr(opts, "center_search_seed", 0)),
        )
        plasma_meta["fit_wall_is_inner"] = 1.0 if fit_to_inner else 0.0

        # Optional smoothing of plasma target too (after generation)
        plasma_xy = _subdivide_closed_polyline_by_maxseg(plasma_xy, float(getattr(opts, "max_seg_len_plasma", 0.0)))
        plasma_xy = _maybe_resample(plasma_xy, int(getattr(opts, "n_plasma", 501)), str(getattr(opts, "resample_walls", "auto")), int(getattr(opts, "min_wall_pts", 400)))

        # xpoints + strike lines to OUTER wall for visualization
        xpoints_target, strike_lines_target = _compute_xpoints_and_strike_lines(
            plasma_xy,
            wall_open=_drop_duplicate_endpoint(outer_xy),
            fallback_len=float(getattr(opts, "strike_ray_fallback_len", 3.0)),
        )

        # ---- NEW: import divertor windows/markers from CAD (XPT/STRIKE)
        # helper reusing the local polylines_in_layer()
        def _load_marker(layer_name: str) -> Optional[Dict[str, Any]]:
            cand = polylines_in_layer(layer_name)
            if not cand:
                return None
            return _pick_best_marker(cand, unit_scale=unit_scale, opts=opts)

        xptL = _load_marker(layers.xpt_lower_win)
        xptU = _load_marker(layers.xpt_upper_win)
        stL  = _load_marker(layers.strike_lower_win)
        stU  = _load_marker(layers.strike_upper_win)

        strikeUI = _load_marker(layers.strike_upper_in)
        strikeLI = _load_marker(layers.strike_lower_in)
        strikeUO = _load_marker(layers.strike_upper_out)
        strikeLO = _load_marker(layers.strike_lower_out)


        # New: desired separatrix-leg corridor windows.
        # These are meant to guide where the upper/lower divertor legs should pass.
        legL = _load_marker(layers.leg_lower_win)
        legU = _load_marker(layers.leg_upper_win)

        if xptL is not None:
            markers["xpt_lower"] = xptL
        if xptU is not None:
            markers["xpt_upper"] = xptU
        if stL is not None:
            markers["strike_lower"] = stL
        if stU is not None:
            markers["strike_upper"] = stU

        if strikeLI is not None:
            markers["STRIKE_L_IN"] = strikeLI
        if strikeUI is not None:
            markers["STRIKE_U_IN"] = strikeUI
        if strikeLO is not None:
            markers["STRIKE_L_OUT"] = strikeLO
        if strikeUO is not None:
            markers["STRIKE_U_OUT"] = strikeUO

        if legL is not None:
            markers["leg_lower"] = legL
        if legU is not None:
            markers["leg_upper"] = legU

        if markers:
            geom_markers_meta = {
                "found": sorted(list(markers.keys())),
                "layers": {
                    "xpt_lower": layers.xpt_lower_win,
                    "xpt_upper": layers.xpt_upper_win,
                    "strike_lower": layers.strike_lower_win,
                    "strike_upper": layers.strike_upper_win,
                    "leg_lower": layers.leg_lower_win,
                    "leg_upper": layers.leg_upper_win,

                    "STRIKE_U_IN": layers.strike_upper_in,
                    "STRIKE_L_IN": layers.strike_lower_in,
                    "STRIKE_U_OUT": layers.strike_upper_out,
                    "STRIKE_L_OUT": layers.strike_lower_out,

                }
            }
        else:
            geom_markers_meta = {
                "found": [],
                "layers": {
                    "xpt_lower": layers.xpt_lower_win,
                    "xpt_upper": layers.xpt_upper_win,
                    "strike_lower": layers.strike_lower_win,
                    "strike_upper": layers.strike_upper_win,
                    "leg_lower": layers.leg_lower_win,
                    "leg_upper": layers.leg_upper_win,

                    "STRIKE_U_IN": layers.strike_upper_in,
                    "STRIKE_L_IN": layers.strike_lower_in,
                    "STRIKE_U_OUT": layers.strike_upper_out,
                    "STRIKE_L_OUT": layers.strike_lower_out,
                }
            }

    # ---- Coils (active)
    coils: Dict[str, Tuple[float, float, float, float]] = {}

    # Physical CAD rectangles before discretization.
    # Used only for Imax/engineering cross-section estimates.
    base_coils: Dict[str, Tuple[float, float, float, float]] = {}

    coil_polys: List[Tuple[str, np.ndarray]] = []
    for e in msp.query("LWPOLYLINE"):
        layer = str(getattr(e.dxf, "layer", ""))
        if layer.startswith(layers.coil_layer_prefix):
            try:
                coil_polys.append((layer, _entity_to_xy(e, opts)))
            except Exception:
                continue

    for e in msp.query("POLYLINE"):
        layer = str(getattr(e.dxf, "layer", ""))
        if layer.startswith(layers.coil_layer_prefix):
            try:
                coil_polys.append((layer, _entity_to_xy(e, opts)))
            except Exception:
                continue

    if len(coil_polys) > 0:
        for layer, xy in coil_polys:
            base_label = _normalize_label(layer[len(layers.coil_layer_prefix):])
            Rc, Zc, dR, dZ = _coil_from_rect_poly(xy * unit_scale)

            label = base_label
            k = 1

            # Avoid duplicate physical labels.
            while label in base_coils:
                k += 1
                label = f"{base_label}_{k}"

            base_coils[label] = (float(Rc), float(Zc), float(dR), float(dZ))

            _add_discretized_active_coil(
                coils,
                label,
                float(Rc),
                float(Zc),
                float(dR),
                float(dZ),
                opts=opts,
            )

    else:
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
            if label in base_coils:
                raise ValueError(
                    f"Duplicate coil label '{label}' inferred from COILS/COIL_LABELS. "
                    "This mode is ambiguous. Prefer per-coil layers COIL_<NAME>."
                )

            base_coils[label] = (float(Rc), float(Zc), float(dR), float(dZ))

            _add_discretized_active_coil(
                coils,
                label,
                float(Rc),
                float(Zc),
                float(dR),
                float(dZ),
                opts=opts,
            )

    # ---- Coil grouping (families + area weights)
    coil_groups: Dict[str, List[str]] = {}
    coil_group_weights: Dict[str, Dict[str, float]] = {}

    for lab, (_Rc, _Zc, dR, dZ) in coils.items():
        fam = _coil_family(lab)
        coil_groups.setdefault(fam, []).append(lab)

    for fam, labs in coil_groups.items():
        areas = []
        for lab in labs:
            _Rc, _Zc, dR, dZ = coils[lab]
            areas.append(_area_from_dR_dZ(dR, dZ))
        areas = np.array(areas, dtype=float)
        w = areas / float(np.sum(areas)) if float(np.sum(areas)) > 0 else np.ones_like(areas) / len(areas)
        coil_group_weights[fam] = {lab: float(wi) for lab, wi in zip(labs, w)}

    # # ---- Coil family metrics + Imax recommendation
    # fill = float(getattr(opts, "fill_factor", 0.75))
    # Jdef = float(getattr(opts, "Jeng_default_A_per_mm2", 40.0))
    # fam_mode = str(getattr(opts, "family_mode", "min")).strip().lower()
    # J_over = getattr(opts, "family_J_override_A_per_mm2", None) or {}

    # coil_family_metrics: Dict[str, Dict[str, float]] = {}
    # Imax_recommended_MA: Dict[str, float] = {}
    #
    # for fam, labs in coil_groups.items():
    #     areas_m2 = []
    #     for lab in labs:
    #         _Rc, _Zc, dR, dZ = coils[lab]
    #         areas_m2.append(_area_from_dR_dZ(dR, dZ))
    #     areas_m2 = np.asarray(areas_m2, float)
    #
    #     A_min_m2 = float(np.min(areas_m2)) if len(areas_m2) else 0.0
    #     A_sum_m2 = float(np.sum(areas_m2)) if len(areas_m2) else 0.0
    #
    #     Aeff_min_m2 = float(fill) * A_min_m2
    #     Aeff_sum_m2 = float(fill) * A_sum_m2
    #
    #     # convert to mm^2
    #     Aeff_min_mm2 = Aeff_min_m2 * 1.0e6
    #     Aeff_sum_mm2 = Aeff_sum_m2 * 1.0e6
    #
    #     J = float(J_over.get(fam, Jdef))
    #
    #     Aeff_mm2 = Aeff_min_mm2 if fam_mode == "min" else Aeff_sum_mm2
    #     Imax_A = J * Aeff_mm2
    #     Imax_MA = Imax_A / 1.0e6
    #
    #     coil_family_metrics[fam] = {
    #         "n_coils": float(len(labs)),
    #         "A_min_m2": float(A_min_m2),
    #         "A_sum_m2": float(A_sum_m2),
    #         "fill_factor": float(fill),
    #         "Aeff_min_mm2": float(Aeff_min_mm2),
    #         "Aeff_sum_mm2": float(Aeff_sum_mm2),
    #         "J_A_per_mm2": float(J),
    #         "family_mode_min": 1.0 if fam_mode == "min" else 0.0,
    #         "Imax_recommended_MA": float(Imax_MA),
    #     }
    #     Imax_recommended_MA[fam] = float(Imax_MA)

    # Assemble geom dict
    geom: Dict = {
        # Updated CAD semantics:
        #   WALL_INNER -> first wall / limiter / plasma-facing boundary.
        #   WALL_OUTER -> blanket outer/back plate.
        #   STAR_VESSEL/PASSIVE_* -> passive structural metal.
        "R_outer": outer_xy[:, 0],
        "Z_outer": outer_xy[:, 1],
        "R_blanket_outer": outer_xy[:, 0],
        "Z_blanket_outer": outer_xy[:, 1],
        "coils": coils,
        "base_coils": base_coils,
        "physical_coils": base_coils,
        "coil_groups": coil_groups,
        "coil_group_weights": coil_group_weights,
        "cad_path": str(dxf_path),
        "unit_scale": float(unit_scale),
        "materials_meta": dict(
            first_wall_material=str(getattr(opts, "first_wall_material", "EUROFER97")),
            first_wall_resistivity_ohm_m=float(getattr(opts, "first_wall_resistivity_ohm_m", 1.0e-6)),
            blanket_outer_material=str(getattr(opts, "blanket_outer_material", "EUROFER97")),
            blanket_outer_resistivity_ohm_m=float(getattr(opts, "blanket_outer_resistivity_ohm_m", 1.0e-6)),
            star_vessel_material=str(getattr(opts, "star_vessel_material", "SS316L")),
            star_vessel_resistivity_ohm_m=float(getattr(opts, "star_vessel_resistivity_ohm_m", 0.75e-6)),
        ),
        "geometry_semantics": dict(
            WALL_INNER="first_wall_limiter_plasma_boundary",
            WALL_OUTER="blanket_outer_back_plate",
            STAR_VESSEL="passive_structural_vessel_support",
            active_coils="COIL_* layers",
            legacy_blanket_fill_enabled=bool(getattr(opts, "blanket_enabled", False)),
            passive_structures_enabled=bool(getattr(opts, "passive_structures_enabled", True)),
        ),
        "sampling_meta": dict(
            prefer_path_flattening=bool(getattr(opts, "prefer_path_flattening", True)),
            flatten_distance=float(getattr(opts, "flatten_distance", 0.002)),
            max_seg_len_wall=float(getattr(opts, "max_seg_len_wall", 0.008)),
            max_seg_len_plasma=float(getattr(opts, "max_seg_len_plasma", 0.008)),
            resample_walls=str(getattr(opts, "resample_walls", "auto")),
            n_wall=int(getattr(opts, "n_wall", 1601)),
            n_inner=int(getattr(opts, "n_inner", 2001)),
            n_plasma=int(getattr(opts, "n_plasma", 501)),
        ),
        # "coil_family_metrics": dict(coil_family_metrics),
        # "Imax_recommended_MA": dict(Imax_recommended_MA),
        # "Imax_assumptions": dict(
        #     fill_factor=float(fill),
        #     Jeng_default_A_per_mm2=float(Jdef),
        #     family_mode=str(fam_mode),
        # ),
    }

    _add_segmented_cs_groups_to_geom(geom, opts)
    _attach_imax_metrics_to_geom(geom, opts)

    # Attach marker windows (if any)
    if markers:
        # Store as numpy arrays (consistent with rest of geom)
        # Each entry: {"xy": np.ndarray, "closed": bool, "length_m": float}
        geom["marker_windows"] = {}
        for k, v in markers.items():
            geom["marker_windows"][k] = {
                "xy": np.asarray(v["xy"], float),
                "closed": bool(v.get("closed", False)),
                "length_m": float(v.get("length_m", 0.0)),
            }
        geom["marker_windows_meta"] = geom_markers_meta

    if inner_xy is not None:
        geom["R_inner"] = inner_xy[:, 0]
        geom["Z_inner"] = inner_xy[:, 1]
        geom["R_limiter"] = inner_xy[:, 0]
        geom["Z_limiter"] = inner_xy[:, 1]
        geom["limiter_material"] = str(getattr(opts, "first_wall_material", "EUROFER97"))
        geom["limiter_resistivity_ohm_m"] = float(getattr(opts, "first_wall_resistivity_ohm_m", 1.0e-6))

    # Blanket region is treated as a forbidden/solid region for plasma geometry.
    # It is not automatically discretized as conducting filaments.
    if inner_xy is not None:
        geom["blanket_region"] = dict(
            inner_layer=str(layers.wall_inner),
            outer_layer=str(layers.wall_outer),
            material=str(getattr(opts, "blanket_outer_material", "EUROFER97")),
            resistivity_ohm_m=float(getattr(opts, "blanket_outer_resistivity_ohm_m", 1.0e-6)),
            forbidden_for_plasma=True,
            note="Region between WALL_INNER and WALL_OUTER represents blanket/solid region; LCFS should stay inside WALL_INNER.",
        )

    if passive_structures:
        geom["passive_structures"] = [
            {
                "layer": str(p.get("layer", "")),
                "name": str(p.get("name", "")),
                "xy": np.asarray(p.get("xy"), float),
                "material": str(p.get("material", "")),
                "resistivity_ohm_m": float(p.get("resistivity_ohm_m", np.nan)),
                "area_m2": float(abs(_polygon_area(np.asarray(p.get("xy"), float)))),
            }
            for p in passive_structures
        ]
        geom["solid_regions"] = list(geom["passive_structures"])

    if passive_filaments:
        geom["passive_filaments"] = list(passive_filaments)
        geom["passive_meta"] = dict(
            n_structures=int(len(passive_structures)),
            n_filaments=int(len(passive_filaments)),
            target_dR_m=float(getattr(opts, "passive_target_dR_m", 0.10)),
            target_dZ_m=float(getattr(opts, "passive_target_dZ_m", 0.10)),
            default_material=str(getattr(opts, "passive_default_material", "SS316L")),
            default_resistivity_ohm_m=float(getattr(opts, "passive_default_resistivity_ohm_m", 0.75e-6)),
        )

    if plasma_xy is not None:
        geom["R_plasma"] = plasma_xy[:, 0]
        geom["Z_plasma"] = plasma_xy[:, 1]

    if plasma_meta:
        geom["plasma_auto_meta"] = dict(plasma_meta)
    if xpoints_target:
        geom["xpoints_target"] = list(xpoints_target)
    if strike_lines_target:
        geom["strike_lines_target"] = [np.asarray(L, float) for L in strike_lines_target]

    if blanket_filaments:
        geom["blanket_filaments"] = list(blanket_filaments)
        geom["blanket_meta"] = dict(
            n=int(len(blanket_filaments)),
            distribution=str(getattr(opts, "blanket_distribution", "stratified")),
            seed=int(getattr(opts, "blanket_seed", 0)),
            wall_margin_m=float(getattr(opts, "blanket_wall_margin_m", 0.0)),
            filament_dR=float(getattr(opts, "blanket_filament_dR", 0.004)),
            filament_dZ=float(getattr(opts, "blanket_filament_dZ", 0.004)),
            has_inner=bool(inner_xy is not None),
        )

    # Basic derived numbers
    if "R_plasma" in geom:
        Rmin, Rmax = float(np.min(geom["R_plasma"])), float(np.max(geom["R_plasma"]))
        Zmin, Zmax = float(np.min(geom["Z_plasma"])), float(np.max(geom["Z_plasma"]))
        geom["R0"] = 0.5 * (Rmin + Rmax)
        geom["a"] = 0.5 * (Rmax - Rmin)
        geom["kappa_geom_est"] = (Zmax - Zmin) / (2.0 * geom["a"]) if geom["a"] > 0 else np.nan
    else:
        Rmin, Rmax = float(np.min(geom["R_outer"])), float(np.max(geom["R_outer"]))
        geom["R0"] = 0.5 * (Rmin + Rmax)

    if float(np.min(geom["R_outer"])) <= 0.0:
        raise ValueError("Outer wall contains R <= 0. Check DXF coordinates and units (X must be R > 0).")

    return geom


# -----------------------------
# Build FreeGSNKE/freegs4e machine
# -----------------------------

def _build_machine_compat(coils_for_machine, vessel_wall):
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
    script_dir = Path(__file__).resolve().parent
    cad_dir = script_dir / "cad"
    cad_dir.mkdir(parents=True, exist_ok=True)

    if dxf_path is None:
        dxf = cad_dir / "star_baseline.dxf"
    else:
        dxf = Path(dxf_path)
        if not dxf.is_absolute():
            dxf = (cad_dir).resolve() # checkcheck

    geom = load_geom_from_dxf(dxf, layers=layers, opts=opts)

    # ------------------------------------------------------------------
    # Optional strict coil check (ACTIVE coils only)
    #
    # Compatible with both:
    #   old labels: PF1U, PF1L, CS1M, ...
    #   discretized labels: PF1U_F001, PF1U_F002, CS1M_F001, ...
    #
    # The check accepts an expected coil if:
    #   1) the exact label exists,
    #   2) its family exists in coil_groups / have_fams,
    #   3) at least one actual coil label starts with expected_label + "_"
    #      e.g. PF1U is satisfied by PF1U_F001.
    # ------------------------------------------------------------------
    if strict_expected:
        if expected_coils is None:
            # After active-coil discretization, checking by family is more robust
            # than requiring exact upper/lower labels.
            expected_coils = {"CS", "PF1", "PF2", "PF3", "PF4", "PF5", "PF6"}

        have = set(_normalize_label(k) for k in geom["coils"].keys())
        have_fams = set(_coil_family(k) for k in have)

        coil_groups = geom.get("coil_groups", {})
        have_group_fams = set(_normalize_label(k) for k in coil_groups.keys())

        missing = []

        for exp in expected_coils:
            expn = _normalize_label(exp)
            expfam = _coil_family(expn)

            # Exact label exists.
            if expn in have:
                continue

            # Family exists, e.g. expected PF1 and have PF1U_F001/PF1L_F001.
            if expn in have_fams or expn in have_group_fams:
                continue

            # Expected upper/lower prefix exists after discretization,
            # e.g. expected PF1U and have PF1U_F001.
            prefix_match = any(
                lab == expn or lab.startswith(expn + "_")
                for lab in have
            )
            if prefix_match:
                continue

            # Expected family exists even if expected was PF1U/PF1L.
            if expfam in have_fams or expfam in have_group_fams:
                continue

            missing.append(expn)

        if missing:
            raise ValueError(
                f"Missing expected coils/families in CAD import: {sorted(missing)}"
            )

    wall_source = str(getattr(opts, "machine_wall_source", "outer")).strip().lower()
    if wall_source == "inner" and "R_inner" in geom and "Z_inner" in geom:
        vessel_wall = machine.Wall(geom["R_inner"], geom["Z_inner"])
    else:
        vessel_wall = machine.Wall(geom["R_outer"], geom["Z_outer"])

    limiter = None
    limiter_source = str(getattr(opts, "limiter_source", "inner")).strip().lower()
    if limiter_source == "outer":
        limiter = machine.Wall(geom["R_outer"], geom["Z_outer"])
    elif "R_inner" in geom and "Z_inner" in geom:
        limiter = machine.Wall(geom["R_inner"], geom["Z_inner"])

    coils_for_machine = []

    for label, (Rc, Zc, dR, dZ) in geom["coils"].items():
        lab = _normalize_label(label)

        # Important:
        # MultiCoil's third and fourth arguments are current and turns,
        # not physical dR/dZ. Geometry was already discretized into centers.
        # Current is assigned later by apply_group_currents(...).
        c = machine.MultiCoil(
            float(Rc),
            float(Zc),
            current=0.0,
            turns=1.0,
        )

        try:
            c.label = lab
        except Exception:
            pass

        coils_for_machine.append((lab, c))

    # ---- Passive structures / legacy blanket filaments (current=0, control=False)
    passive_labels: List[str] = []

    # New preferred passive structures from STAR_VESSEL / PASSIVE_*
    if "passive_filaments" in geom:
        for item in geom["passive_filaments"]:
            lab0, Rc, Zc, dR, dZ, material, resistivity = item
            lab = _normalize_label(lab0)

            c = machine.MultiCoil(
                float(Rc),
                float(Zc),
                current=0.0,
                turns=1.0,
            )

            try:
                c.label = lab
            except Exception:
                pass
            try:
                c.current = 0.0
            except Exception:
                pass
            try:
                c.control = False
            except Exception:
                pass
            try:
                c.material = str(material)
                c.resistivity_ohm_m = float(resistivity)
                c.dR = float(dR)
                c.dZ = float(dZ)
            except Exception:
                pass

            coils_for_machine.append((lab, c))
            passive_labels.append(lab)

    # Legacy artificial blanket fill, used only if explicitly enabled and no
    # STAR_VESSEL passive filaments were generated.
    if "blanket_filaments" in geom:
        for (lab0, Rc, Zc, dR, dZ) in geom["blanket_filaments"]:
            lab = _normalize_label(lab0)
            c = machine.MultiCoil(
                float(Rc),
                float(Zc),
                current=0.0,
                turns=1.0,
            )

            try:
                c.label = lab
                c.control = False
                c.material = "LEGACY_BLANKET_FILL"
                c.resistivity_ohm_m = float(getattr(opts, "blanket_outer_resistivity_ohm_m", 1.0e-6))
                c.dR = float(dR)
                c.dZ = float(dZ)
            except Exception:
                pass

            coils_for_machine.append((lab, c))
            passive_labels.append(lab)

    tokamak = _build_machine_compat(coils_for_machine, vessel_wall)

    if limiter is not None:
        try:
            tokamak.limiter = limiter
        except Exception:
            pass

    active_labels = [_normalize_label(x) for x in geom["coils"].keys()]

    tokamak.active_coils = list(active_labels)
    tokamak.passive_coils = list(passive_labels)

    tokamak.R0 = float(geom.get("R0", np.nan))
    tokamak.geom = geom

    # Includes active + passive coils.
    tokamak.coils_dict = {label: coil for label, coil in coils_for_machine}

    # These are critical for apply_group_currents(...).
    tokamak.coil_groups = dict(geom.get("coil_groups", {}))
    tokamak.coil_group_weights = dict(geom.get("coil_group_weights", {}))

    return tokamak, geom

# -----------------------------
# Optional: apply grouped currents
# -----------------------------

def apply_group_currents(tokamak, group_currents: Dict[str, float], *, mode: str = "area"):
    """
    Apply current-control family currents to discretized coils.

    Important semantics:
    - Active coil currents are reset to zero before applying families.
    - Family currents are added, not overwritten.
    - This is required because CS, CS_MID and CS_END can overlap:
        CS      = parent/full central solenoid group
        CS_MID  = subset of CS filaments
        CS_END  = subset of CS filaments

      Therefore:
        CS_current != 0, CS_MID_current = 0, CS_END_current = 0
      correctly gives a uniform CS current.

      And:
        CS_current = 0, CS_MID_current != 0, CS_END_current != 0
      correctly gives segmented control.

      If all three are nonzero, the result is:
        total filament current = parent CS contribution + local segment contribution.
    """
    mode = str(mode).lower().strip()
    groups = getattr(tokamak, "coil_groups", {}) or {}
    weights = getattr(tokamak, "coil_group_weights", {}) or {}
    coils_dict = getattr(tokamak, "coils_dict", {}) or {}

    # Normalize coil dictionary keys.
    coil_map = {str(k).strip().upper(): v for k, v in coils_dict.items()}

    # Reset active coils only. Passive coils remain zero / untouched here.
    active = getattr(tokamak, "active_coils", None)
    if active is None:
        active = list(coil_map.keys())

    for lab in active:
        UL = str(lab).strip().upper()
        c = coil_map.get(UL, None)
        if c is not None:
            try:
                c.current = 0.0
            except Exception:
                pass

    for fam, Itot in group_currents.items():
        fam = _normalize_label(fam)
        Itot = float(Itot)

        labs = groups.get(fam, [])
        labs = [_normalize_label(x) for x in labs]

        if not labs:
            continue

        if mode == "same":
            # Add same current to each filament.
            for lab in labs:
                c = coil_map.get(lab, None)
                if c is not None:
                    try:
                        c.current = float(getattr(c, "current", 0.0)) + Itot
                    except Exception:
                        pass
            continue

        if mode == "equal":
            w = {lab: 1.0 / max(1, len(labs)) for lab in labs}
        else:
            w = weights.get(fam, None)
            if not w:
                w = {lab: 1.0 / max(1, len(labs)) for lab in labs}

        # Normalize weight keys defensively.
        w = {_normalize_label(k): float(v) for k, v in dict(w).items()}

        for lab in labs:
            c = coil_map.get(lab, None)
            if c is None:
                continue
            try:
                c.current = float(getattr(c, "current", 0.0)) + Itot * float(w.get(lab, 0.0))
            except Exception:
                pass

# -----------------------------
# Plot with right-side parameter box
# -----------------------------

def _format_plasma_meta_text(meta: Dict[str, float]) -> str:
    def g(k, default=np.nan):
        return meta.get(k, default)

    lines = []
    lines.append("PLASMA TARGET (geom)")
    lines.append("")

    fit = "INNER" if int(meta.get("fit_wall_is_inner", 0.0)) == 1 else "OUTER"
    moved = "YES" if int(meta.get("center_moved", 0.0)) == 1 else "NO"
    lines.append(f"fit_wall     : {fit}")
    lines.append(f"center_moved : {moved}")
    lines.append(f"scale        : {g('scale'):.4f}")
    lines.append(f"du/dl        : {g('delta_u'):.3f} / {g('delta_l'):.3f}")
    lines.append("")

    lines.append("TARGET")
    lines.append(f"R0,A,k,Z0    : {g('R0_target'):.3f}, {g('A_target'):.3f}, {g('kappa_target'):.3f}, {g('Z0_target'):.3f}")
    lines.append("USED CENTER")
    lines.append(f"R0_used,Z0   : {g('R0_used'):.3f}, {g('Z0_used'):.3f}")
    lines.append("")

    lines.append("MIDPLANE GEOM (Z=Z0_used)")
    lines.append(f"R0_mid       : {g('geom_R0_mid'):.3f}")
    lines.append(f"a_mid        : {g('geom_a_mid'):.3f}")
    lines.append(f"A_mid        : {g('geom_A_mid'):.3f}")
    lines.append(f"kappa_mid    : {g('geom_kappa_mid'):.3f}")
    lines.append(f"delta_u/l    : {g('geom_delta_u'):.3f} / {g('geom_delta_l'):.3f}")
    lines.append("")

    lines.append("BBOX")
    lines.append(f"Rmin/Rmax    : {g('geom_Rmin'):.3f} / {g('geom_Rmax'):.3f}")
    lines.append(f"Zmin/Zmax    : {g('geom_Zmin'):.3f} / {g('geom_Zmax'):.3f}")
    lines.append("")

    lines.append("AREA/PERIM")
    lines.append(f"area [m^2]   : {g('geom_area_poloidal'):.4f}")
    lines.append(f"perim [m]    : {g('geom_perimeter_poloidal'):.4f}")
    lines.append("")

    lines.append("TOROIDAL APPROX")
    lines.append(f"V~ [m^3]     : {g('geom_volume_torus_approx'):.4f}")
    lines.append(f"S~ [m^2]     : {g('geom_surface_torus_approx'):.4f}")
    lines.append("")

    lines.append("ERRORS (midplane)")
    lines.append(f"err_R0_mid   : {g('err_R0_mid'):+.4f} m")
    lines.append(f"err_A_mid    : {g('err_A_mid'):+.4f}")
    lines.append(f"err_kappa    : {g('err_kappa_mid'):+.4f}")
    lines.append(f"rel_err_A    : {100.0*g('rel_err_A_mid'):+.2f} %")
    lines.append(f"rel_err_k    : {100.0*g('rel_err_kappa_mid'):+.2f} %")

    return "\n".join(lines)


def plot_cad_geometry(geom: Dict, show: bool = True, ax=None):
    meta = geom.get("plasma_auto_meta", None)

    if ax is None:
        fig, (ax, axr) = plt.subplots(
            1, 2,
            figsize=(11, 9),
            gridspec_kw={"width_ratios": [3.3, 1.7], "wspace": 0.05},
        )
        axr.axis("off")

        if isinstance(meta, dict) and meta:
            txt = _format_plasma_meta_text(meta)
            axr.text(
                0.02, 0.98, txt,
                va="top", ha="left",
                family="monospace", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.3", alpha=0.95),
                transform=axr.transAxes,
            )
    else:
        fig = ax.figure
        if isinstance(meta, dict) and meta:
            txt = _format_plasma_meta_text(meta)
            ax.text(
                1.02, 0.98, txt,
                va="top", ha="left",
                family="monospace", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.3", alpha=0.95),
                transform=ax.transAxes,
                clip_on=False,
            )

    ax.plot(geom["R_outer"], geom["Z_outer"], "k-", lw=2, label="WALL_OUTER / blanket outer")
    if "R_inner" in geom:
        ax.plot(geom["R_inner"], geom["Z_inner"], "k--", lw=1.5, label="WALL_INNER / limiter")

    if "passive_structures" in geom:
        for ps in geom["passive_structures"]:
            try:
                xy = np.asarray(ps.get("xy"), float)
                ax.fill(xy[:, 0], xy[:, 1], color="0.70", alpha=0.25, lw=0.5, edgecolor="0.25")
            except Exception:
                pass

    if "R_plasma" in geom:
        ax.plot(geom["R_plasma"], geom["Z_plasma"], color="tab:orange", lw=1.8, label="Plasma target (AUTO)")

    # passive / blanket filaments
    if "passive_filaments" in geom:
        for item in geom["passive_filaments"]:
            _lab, Rc, Zc, dR, dZ, _mat, _rho = item
            x0, x1 = Rc - dR, Rc + dR
            y0, y1 = Zc - dZ, Zc + dZ
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="0.25", lw=0.15, alpha=0.55)

    if "blanket_filaments" in geom:
        for (_lab, Rc, Zc, dR, dZ) in geom["blanket_filaments"]:
            x0, x1 = Rc - dR, Rc + dR
            y0, y1 = Zc - dZ, Zc + dZ
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], lw=0.2)

    # xpoints + strike lines
    if "xpoints_target" in geom:
        for (Rx, Zx, kind) in geom["xpoints_target"]:
            ax.plot([Rx], [Zx], marker="x", ms=10, mew=2, linestyle="None")
            ax.text(Rx, Zx, f" {kind}", fontsize=9, va="center")

    if "strike_lines_target" in geom:
        for L in geom["strike_lines_target"]:
            L = np.asarray(L, float)
            ax.plot(L[:, 0], L[:, 1], "-", lw=1.2)

    # coils
    for name, (Rc, Zc, dR, dZ) in geom["coils"].items():
        x0, x1 = Rc - dR, Rc + dR
        y0, y1 = Zc - dZ, Zc + dZ
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k-", lw=0.3)
        # ax.text(Rc, Zc, _normalize_label(name), ha="center", va="center", fontsize=8)

    # --- NEW: marker windows visualization ---
    if "marker_windows" in geom and isinstance(geom["marker_windows"], dict):
        for name, pack in geom["marker_windows"].items():
            try:
                xy = np.asarray(pack.get("xy", None), float)
                if xy.ndim == 2 and xy.shape[0] >= 2:
                    ax.plot(xy[:, 0], xy[:, 1], lw=3.0, alpha=0.8, label=f"WIN:{name}")
                    # small label near first point
                    ax.text(float(xy[0, 0]), float(xy[0, 1]), f" {name}", fontsize=9, va="center")
            except Exception:
                pass

    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.grid(True)
    # ax.legend(loc="upper left")

    if show:
        # tight_layout can warn with side text axes; it's safe. If you want, replace with subplots_adjust.
        plt.tight_layout()
        plt.show()
    return ax


# -----------------------------
# Main (smoke test)
# -----------------------------

if __name__ == "__main__":
    opts = CADImportOptions(
        unit_scale=None,

        # Smooth CAD + sampling
        prefer_path_flattening=True,
        flatten_distance=0.002,
        max_seg_len_wall=0.008,
        max_seg_len_plasma=0.008,

        resample_walls="always",
        n_wall=1601,
        n_inner=2001,
        n_plasma=501,
        min_wall_pts=400,

        enforce_ccw=True,
        canonical_start=True,

        # Plasma target AUTO
        plasma_target_mode="auto",
        plasma_fit_to_inner_if_available=True,
        plasma_R0=4.0,
        plasma_A=2.0,
        plasma_kappa=2.5,
        plasma_Z0=0.0,
        plasma_delta_max=0.70,
        plasma_delta_grid=17,
        plasma_delta_symmetric=True,
        plasma_shrink_iters=20,
        plasma_scale_safety=0.999,
        containment_radius=-1e-9,
        fix_center_if_outside=True,
        center_search_samples=800,
        center_search_seed=0,
        strike_ray_fallback_len=3.0,

        # Passive structures from STAR_VESSEL/PASSIVE_* are preferred.
        # Legacy artificial blanket fill is disabled.
        passive_structures_enabled=True,
        passive_use_star_vessel=True,
        passive_target_dR_m=0.10,
        passive_target_dZ_m=0.10,

        # Legacy blanket passive filaments
        blanket_enabled=False,
        blanket_n_filaments=0,
        blanket_distribution="stratified",
        blanket_seed=0,
        blanket_wall_margin_m=0.01,
        blanket_filament_dR=0.004,
        blanket_filament_dZ=0.004,

        # Imax engineering
        fill_factor=0.75,
        Jeng_default_A_per_mm2=40.0,
        family_mode="min",
        family_J_override_A_per_mm2=None,
    )

    tokamak, geom = make_star_machine_from_cad(opts=opts, strict_expected=True)
    print("[OK] Loaded CAD machine from:", geom.get("cad_path"))

    if "sampling_meta" in geom:
        sm = geom["sampling_meta"]
        print(f"[INFO] outer points: {len(geom['R_outer'])} inner: {len(geom.get('R_inner', []))}")
        print("[INFO] sampling meta:", sm)

    print("[INFO] Coils found:", sorted(list(geom["coils"].keys())))
    print("[INFO] Families:", {k: len(v) for k, v in (geom.get("coil_groups", {}) or {}).items()})

    if "blanket_meta" in geom:
        print("[INFO] blanket_meta:", geom["blanket_meta"])
    if "passive_meta" in geom:
        print("[INFO] passive_meta:", geom["passive_meta"])

    if "plasma_auto_meta" in geom:
        meta = geom["plasma_auto_meta"]
        print("[INFO] plasma_auto_meta (selected):")
        keys = [
            "fit_wall_is_inner", "center_moved", "scale", "delta_u", "delta_l",
            "R0_target", "A_target", "kappa_target", "Z0_target",
            "R0_used", "Z0_used",
            "geom_R0_mid", "geom_a_mid", "geom_A_mid", "geom_kappa_mid",
            "geom_delta_u", "geom_delta_l",
            "geom_area_poloidal", "geom_perimeter_poloidal",
            "err_R0_mid", "err_A_mid", "err_kappa_mid",
            "rel_err_A_mid", "rel_err_kappa_mid",
        ]
        for k in keys:
            if k in meta:
                print(f"  {k:>22s}: {meta[k]}")

    if "xpoints_target" in geom:
        print("[INFO] xpoints_target:", geom["xpoints_target"])

    # Imax summary
    if "Imax_recommended_MA" in geom:
        ass = geom.get("Imax_assumptions", {})
        print("\n[INFO] Imax recommended by family (Jeng * Aeff):")
        print(f"  assumptions: fill_factor={ass.get('fill_factor')}, Jeng_default={ass.get('Jeng_default_A_per_mm2')} A/mm^2, family_mode={ass.get('family_mode')}")
        fams = sorted(list(geom["Imax_recommended_MA"].keys()))
        for fam in fams:
            Imax = geom["Imax_recommended_MA"][fam]
            met = (geom.get("coil_family_metrics", {}) or {}).get(fam, {})
            Aeff_min = met.get("Aeff_min_mm2", np.nan)
            J = met.get("J_A_per_mm2", np.nan)
            print(f"  {fam:>4s}: Imax~{Imax:7.3f} MA  |  Aeff_min={Aeff_min:,.0f} mm^2  |  J={J:.1f} A/mm^2")

    plot_cad_geometry(geom, show=True)
