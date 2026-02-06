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
        COIL_CS1M, COIL_CS2U, COIL_CS2L, COIL_CS3U, ...
    These will be treated as separate coils but grouped under family "CS".
  Fallback (legacy, less robust):
    Rectangles on COILS + labels on COIL_LABELS

Stability improvements:
  - Sequential duplicate removal / closure enforcement
  - CCW enforcement for walls
  - Optional resampling policy: "auto" | "always" | "never"
  - Canonical start point: outboard midplane (max R, then min |Z|)
  - POLYLINE vertex API compatibility (list vs callable)
  - Optional entity flattening via ezdxf.path (if available)

AUTO plasma target + side-panel parameters:
  - plasma_target_mode: "cad" | "auto"
  - Auto Miller-like target with (R0,A,kappa,Z0) + triangularity scan + shrink
  - Fits inside WALL_INNER if available (else inside WALL_OUTER)
  - Adds geometric xpoints markers + strike rays (plot-only, not magnetic)
  - Computes a full pack of geometric parameters from the boundary and prints/stores them

NOTE:
  - These are geometric (boundary) parameters, NOT equilibrium (psi-based) diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
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

    coil_layer_prefix: str = "COIL_"          # Recommended: COIL_CS, COIL_PF1U, COIL_CS1M, ...
    coils_layer: str = "COILS"                # Fallback: all rectangles here
    coil_labels_layer: str = "COIL_LABELS"    # Fallback: text labels here


@dataclass(frozen=True)
class CADImportOptions:
    # If unit_scale is None, infer from $INSUNITS. Example: mm -> 1e-3.
    unit_scale: Optional[float] = None

    # Wall resampling policy: "auto" | "always" | "never"
    resample_walls: str = "auto"

    # Target points used if resampling is active
    n_wall: int = 801
    n_inner: int = 801
    n_plasma: int = 320
    min_wall_pts: int = 200

    # Enforce CCW orientation for wall polylines (recommended)
    enforce_ccw: bool = True

    # Rotate start to outboard midplane for consistency (recommended)
    canonical_start: bool = True

    # Flattening chord-length target (meters) for SPLINE/ARC/ELLIPSE fallback (if ezdxf.path available)
    flatten_distance: float = 0.01

    # Fallback label matching tolerance (legacy mode)
    label_match_factor: float = 2.0  # radius ~ factor * max(dR,dZ)

    # -------------------------
    # AUTO plasma target
    # -------------------------
    # "cad"  -> use PLASMA_TARGET if exists (else fallback to auto)
    # "auto" -> always generate auto target
    plasma_target_mode: str = "auto"

    # Fit wall preference: if WALL_INNER exists, fit inside it
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


def _strip_auto_suffix(label: str) -> str:
    return re.sub(r"_[0-9]+$", "", str(label).strip().upper())


def _coil_family(label: str) -> str:
    lab = _strip_auto_suffix(_normalize_label(label))
    if lab.startswith("CS"):
        return "CS"
    m = re.match(r"^(PF[0-9]+)", lab)
    if m:
        return m.group(1)
    return lab


def _area_from_dR_dZ(dR: float, dZ: float) -> float:
    a = 4.0 * float(dR) * float(dZ)
    if not np.isfinite(a) or a <= 0:
        return 1.0
    return a


# -----------------------------
# Geometry pack for plasma target
# -----------------------------

def _poly_perimeter(xy_closed: np.ndarray) -> float:
    pts = _ensure_closed(np.asarray(xy_closed, float))
    if len(pts) < 4:
        return 0.0
    d = pts[1:] - pts[:-1]
    return float(np.sum(np.sqrt(np.sum(d * d, axis=1))))


def _poly_centroid(xy_closed: np.ndarray) -> Tuple[float, float]:
    """
    Centroid of a simple polygon (area-weighted). If degenerate, returns mean of vertices.
    """
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
    """
    Intersections of polygon edges with horizontal line Z=Zc.
    Returns sorted unique R intersections.
    """
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

    # dedupe near-equal intersections (vertex hits)
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
    """
    Compute geometric (purely boundary-based) parameters of a plasma contour.

    Key results:
      - bbox extrema (Rmin,Rmax,Zmin,Zmax), bbox center (R0_bbox, Z0_bbox)
      - midplane cut (Z=Z0_ref or Z0_bbox): Rin_mid, Rout_mid, R0_mid, a_mid
      - kappa_mid, b (vertical semi-axis), aspect ratio A_mid
      - triangularity delta_u/delta_l w.r.t. R0_ref (or R0_mid)
      - area_poloidal, perimeter_poloidal
      - centroid (Rc_centroid,Zc_centroid)
      - toroidal approximations: volume ~ 2π R0_mid * area, surface ~ 2π R0_mid * perimeter
    """
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
# AUTO plasma target (fit inside inner wall + xpoints markers)
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
    """
    Miller-like boundary with separate upper/lower triangularity.
    Strictly geometric target.
    """
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
    """
    If (R0,Z0) is not inside the fit wall, pick an interior point (random sampling)
    closest to the target. Helps feasibility when CAD wall is shifted.
    """
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

            # Primary objective: maximize scale (closest to requested A)
            # Secondary: prefer larger delta (more D-like), then symmetry
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

    # Final safety (if numeric jitter)
    if not _fits_inside_wall_strict(xy, wall_open=wall_open, containment_radius=float(containment_radius)):
        xy = _miller_boundary_ud(
            R0=R0_use, a=a_nom, kappa=float(kappa_t),
            delta_u=du_best, delta_l=dl_best,
            Z0=Z0_use, n=int(n), scale=float(0.995 * s_final)
        )

    meta: Dict[str, float] = {
        "mode": 1.0,  # numeric tag to keep meta float-friendly
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

    # Old quick estimates
    meta.update(_estimate_from_boundary(xy))

    # Full geometry pack (prefix geom_)
    geom_pack = compute_plasma_geom_params(
        xy,
        Z0_ref=float(Z0_use),
        R0_ref=float(R0_t),  # triangularity reported w.r.t. target R0
    )
    for k, v in geom_pack.items():
        if isinstance(v, (int, float, np.floating)):
            meta[f"geom_{k}"] = float(v)

    # Errors vs targets (midplane-based)
    A_mid = meta.get("geom_A_mid", np.nan)
    k_mid = meta.get("geom_kappa_mid", np.nan)
    R0_mid = meta.get("geom_R0_mid", np.nan)

    meta["err_R0_mid"] = float(R0_mid - float(R0_t)) if np.isfinite(R0_mid) else np.nan
    meta["err_A_mid"] = float(A_mid - float(A_t)) if np.isfinite(A_mid) else np.nan
    meta["err_kappa_mid"] = float(k_mid - float(kappa_t)) if np.isfinite(k_mid) else np.nan

    meta["rel_err_A_mid"] = float((A_mid - float(A_t)) / float(A_t)) if (np.isfinite(A_mid) and float(A_t) != 0.0) else np.nan
    meta["rel_err_kappa_mid"] = float((k_mid - float(kappa_t)) / float(kappa_t)) if (np.isfinite(k_mid) and float(kappa_t) != 0.0) else np.nan

    return xy, meta


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
        plasma_xy = _maybe_resample(plasma_xy, opts.n_plasma, opts.resample_walls, opts.min_wall_pts)

    plasma_meta: Dict[str, float] = {}
    xpoints_target = []
    strike_lines_target = []

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
            n=int(getattr(opts, "n_plasma", 320)),
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

        # xpoints + strike lines to OUTER wall for visualization
        xpoints_target, strike_lines_target = _compute_xpoints_and_strike_lines(
            plasma_xy,
            wall_open=_drop_duplicate_endpoint(outer_xy),
            fallback_len=float(getattr(opts, "strike_ray_fallback_len", 3.0)),
        )

    # ---- Coils (UNCHANGED from your original)
    coils: Dict[str, Tuple[float, float, float, float]] = {}

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
            while label in coils:
                k += 1
                label = f"{base_label}_{k}"

            coils[label] = (float(Rc), float(Zc), float(dR), float(dZ))

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
            if label in coils:
                raise ValueError(
                    f"Duplicate coil label '{label}' inferred from COILS/COIL_LABELS. "
                    "This mode is ambiguous. Prefer per-coil layers COIL_<NAME>."
                )
            coils[label] = (float(Rc), float(Zc), float(dR), float(dZ))

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

    # Assemble geom dict
    geom: Dict = {
        "R_outer": outer_xy[:, 0],
        "Z_outer": outer_xy[:, 1],
        "coils": coils,
        "coil_groups": coil_groups,
        "coil_group_weights": coil_group_weights,
        "cad_path": str(dxf_path),
        "unit_scale": float(unit_scale),
    }
    if inner_xy is not None:
        geom["R_inner"] = inner_xy[:, 0]
        geom["Z_inner"] = inner_xy[:, 1]
    if plasma_xy is not None:
        geom["R_plasma"] = plasma_xy[:, 0]
        geom["Z_plasma"] = plasma_xy[:, 1]

    if plasma_meta:
        geom["plasma_auto_meta"] = dict(plasma_meta)
    if xpoints_target:
        geom["xpoints_target"] = list(xpoints_target)
    if strike_lines_target:
        geom["strike_lines_target"] = [np.asarray(L, float) for L in strike_lines_target]

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
            dxf = (cad_dir / dxf).resolve()

    geom = load_geom_from_dxf(dxf, layers=layers, opts=opts)

    # Optional strict coil check
    if strict_expected:
        if expected_coils is None:
            expected_coils = {"CS", "PF1U", "PF1L", "PF2U", "PF2L", "PF3U", "PF3L"}

        have = set(_normalize_label(k) for k in geom["coils"].keys())
        have_fams = set(_coil_family(k) for k in have)

        missing = []
        for exp in expected_coils:
            expn = _normalize_label(exp)
            if expn in have:
                continue
            if expn in have_fams:
                continue
            missing.append(expn)

        if missing:
            raise ValueError(f"Missing expected coils in CAD import: {sorted(missing)}")

    vessel_wall = machine.Wall(geom["R_outer"], geom["Z_outer"])
    limiter = None
    if "R_inner" in geom and "Z_inner" in geom:
        limiter = machine.Wall(geom["R_inner"], geom["Z_inner"])

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

    if limiter is not None:
        try:
            tokamak.limiter = limiter
        except Exception:
            pass

    tokamak.active_coils = [label for label, _ in coils_for_machine]
    tokamak.passive_coils = []
    tokamak.R0 = float(geom.get("R0", np.nan))
    tokamak.geom = geom
    tokamak.coils_dict = {label: coil for label, coil in coils_for_machine}
    tokamak.coil_groups = dict(geom.get("coil_groups", {}))
    tokamak.coil_group_weights = dict(geom.get("coil_group_weights", {}))

    return tokamak, geom


# -----------------------------
# Optional: apply grouped currents
# -----------------------------

def apply_group_currents(tokamak, group_currents: Dict[str, float], *, mode: str = "area"):
    mode = str(mode).lower().strip()
    groups = getattr(tokamak, "coil_groups", {}) or {}
    weights = getattr(tokamak, "coil_group_weights", {}) or {}

    for fam, Itot in group_currents.items():
        fam = _normalize_label(fam)
        labs = groups.get(fam, [])
        if not labs:
            continue

        if mode == "same":
            for lab in labs:
                if lab in tokamak.coils_dict:
                    tokamak.coils_dict[lab].current = float(Itot)
            continue

        if mode == "equal":
            w = {lab: 1.0 / len(labs) for lab in labs}
        else:  # "area"
            w = weights.get(fam, None)
            if not w:
                w = {lab: 1.0 / len(labs) for lab in labs}

        for lab in labs:
            if lab in tokamak.coils_dict:
                tokamak.coils_dict[lab].current = float(Itot) * float(w.get(lab, 0.0))


# -----------------------------
# Plot with right-side parameter box
# -----------------------------

def _format_plasma_meta_text(meta: Dict[str, float]) -> str:
    """
    Create a compact, readable block with the most important geometric params + errors.
    """
    def g(k, default=np.nan):
        v = meta.get(k, default)
        return v

    # Prefer midplane-based geometry (geom_*)
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
    """
    If ax is None, creates a 2-panel figure (plot + right-side text box).
    If ax is provided, plots on it and writes the box outside the axes on the right.
    """
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

    ax.plot(geom["R_outer"], geom["Z_outer"], "k-", lw=2, label="CAD outer wall")
    if "R_inner" in geom:
        ax.plot(geom["R_inner"], geom["Z_inner"], "k--", lw=1.5, label="CAD inner wall")
    if "R_plasma" in geom:
        ax.plot(geom["R_plasma"], geom["Z_plasma"], color="tab:orange", lw=1.8, label="Plasma target (AUTO)")

    # xpoints + strike lines (if present)
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
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k-", lw=1)
        ax.text(Rc, Zc, _normalize_label(name), ha="center", va="center", fontsize=8)

    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.grid(True)
    ax.legend(loc="upper left")

    if show:
        plt.tight_layout()
        plt.show()
    return ax


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    # Smoke test: AUTO plasma target fit inside WALL_INNER (if exists)
    opts = CADImportOptions(
        unit_scale=None,
        resample_walls="auto",
        n_wall=801,
        n_inner=801,
        n_plasma=320,
        min_wall_pts=200,
        enforce_ccw=True,
        canonical_start=True,

        plasma_target_mode="auto",
        plasma_fit_to_inner_if_available=True,
        plasma_R0=4.0,
        plasma_A=2.0,
        plasma_kappa=2.5,
        plasma_Z0=0.0,
        plasma_delta_max=0.70,
        plasma_delta_grid=17,
        plasma_delta_symmetric=True,   # set False if you want du/dl grid (slower)
        plasma_shrink_iters=20,
        plasma_scale_safety=0.999,
        containment_radius=-1e-9,
        fix_center_if_outside=True,
        center_search_samples=800,
        center_search_seed=0,
        strike_ray_fallback_len=3.0,
    )

    tokamak, geom = make_star_machine_from_cad(opts=opts, strict_expected=True)
    print("[OK] Loaded CAD machine from:", geom.get("cad_path"))
    print("[INFO] Coils found:", sorted(list(geom["coils"].keys())))
    print("[INFO] Families:", {k: len(v) for k, v in (geom.get("coil_groups", {}) or {}).items()})
    print("[INFO] outer wall points:", len(geom["R_outer"]), "| inner wall points:", len(geom.get("R_inner", [])))

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

    plot_cad_geometry(geom, show=True)

