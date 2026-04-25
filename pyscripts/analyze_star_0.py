# analyze_star.py
# Unified analysis: separatrix/X-point when possible; otherwise LCFS-limiter (robust).
# JSON-friendly output, Windows-safe (Agg only when needed).

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import contourpy  # fast contouring, no pyplot needed
except Exception:
    contourpy = None

try:
    from matplotlib.path import Path
except Exception:
    Path = None


# -------------------------
# Basic helpers
# -------------------------
def _as_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _get_attr_or_call(obj: Any, name: str) -> Any:
    if not hasattr(obj, name):
        return None
    v = getattr(obj, name)
    try:
        return v() if callable(v) else v
    except TypeError:
        return None
    except Exception:
        return None

def _psi_from_eq(eq: Any) -> np.ndarray:
    psi = _get_attr_or_call(eq, "psi")
    if psi is None:
        psi = _get_attr_or_call(eq, "Psi")
    if psi is None:
        raise AttributeError("Could not obtain psi from eq (eq.psi / eq.Psi).")
    psi = np.asarray(psi, dtype=float)
    return psi

def _grids_from_eq(eq: Any) -> Tuple[np.ndarray, np.ndarray]:
    R = _get_attr_or_call(eq, "R")
    Z = _get_attr_or_call(eq, "Z")
    if R is None or Z is None:
        raise AttributeError("Could not obtain grids from eq (eq.R / eq.Z).")
    return np.asarray(R, float), np.asarray(Z, float)

def _infer_axes_and_orient(Rm: np.ndarray, Zm: np.ndarray, psi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (R1, Z1, psi_RZ) where:
      R1: (nR,) increasing
      Z1: (nZ,) increasing
      psi_RZ: (nR, nZ) consistent with R1,Z1

    Handles both meshgrid conventions:
      - indexing='ij' => Rm[:,0] ~ R axis, Zm[0,:] ~ Z axis
      - indexing='xy' => Rm[0,:] ~ R axis, Zm[:,0] ~ Z axis
    """
    if Rm.ndim != 2 or Zm.ndim != 2:
        # already 1D? (rare)
        R1 = np.asarray(Rm, float).ravel()
        Z1 = np.asarray(Zm, float).ravel()
        psi_RZ = np.asarray(psi, float)
        if psi_RZ.shape != (R1.size, Z1.size):
            raise ValueError("Non-2D grids but psi shape mismatch.")
        return R1, Z1, psi_RZ

    # Detect 'ij' vs 'xy' by checking which direction is constant
    # ij: Rm columns are ~identical; Zm rows are ~identical
    is_ij = False
    try:
        if Rm.shape[1] > 1 and Zm.shape[0] > 1:
            is_ij = np.allclose(Rm[:, 0], Rm[:, 1]) and np.allclose(Zm[0, :], Zm[1, :])
    except Exception:
        is_ij = False

    if is_ij:
        R1 = np.asarray(Rm[:, 0], float)
        Z1 = np.asarray(Zm[0, :], float)
        psi_RZ = np.asarray(psi, float)
        if psi_RZ.shape != (R1.size, Z1.size):
            # maybe transposed
            if psi_RZ.shape == (Z1.size, R1.size):
                psi_RZ = psi_RZ.T
            else:
                raise ValueError(f"psi shape mismatch: {psi_RZ.shape} vs {(R1.size,Z1.size)}")
    else:
        # assume xy
        R1 = np.asarray(Rm[0, :], float)
        Z1 = np.asarray(Zm[:, 0], float)
        psi_xy = np.asarray(psi, float)
        # want (nR,nZ): transpose xy layout (nZ,nR) -> (nR,nZ)
        if psi_xy.shape == (Z1.size, R1.size):
            psi_RZ = psi_xy.T
        elif psi_xy.shape == (R1.size, Z1.size):
            psi_RZ = psi_xy
        else:
            raise ValueError(f"psi shape mismatch: {psi_xy.shape} vs {(Z1.size,R1.size)} or {(R1.size,Z1.size)}")

    # enforce increasing axes + flip psi accordingly
    if R1.size > 1 and R1[0] > R1[-1]:
        R1 = R1[::-1].copy()
        psi_RZ = psi_RZ[::-1, :].copy()
    if Z1.size > 1 and Z1[0] > Z1[-1]:
        Z1 = Z1[::-1].copy()
        psi_RZ = psi_RZ[:, ::-1].copy()

    return R1, Z1, psi_RZ

def _bilinear_interp(R1: np.ndarray, Z1: np.ndarray, F_RZ: np.ndarray, Rp: np.ndarray, Zp: np.ndarray) -> np.ndarray:
    """
    Bilinear interpolation on rect grid.
    F_RZ shape (nR,nZ).
    """
    R1 = np.asarray(R1, float)
    Z1 = np.asarray(Z1, float)
    F = np.asarray(F_RZ, float)
    Rp = np.asarray(Rp, float)
    Zp = np.asarray(Zp, float)

    nR, nZ = F.shape
    if nR != R1.size or nZ != Z1.size:
        raise ValueError(f"F_RZ shape mismatch: {F.shape} vs {(R1.size, Z1.size)}")

    Rp = np.clip(Rp, R1[0], R1[-1])
    Zp = np.clip(Zp, Z1[0], Z1[-1])

    i = np.searchsorted(R1, Rp, side="right") - 1
    j = np.searchsorted(Z1, Zp, side="right") - 1
    i = np.clip(i, 0, nR - 2)
    j = np.clip(j, 0, nZ - 2)

    R0 = R1[i]; R2 = R1[i + 1]
    Z0 = Z1[j]; Z2 = Z1[j + 1]

    t = (Rp - R0) / (R2 - R0 + 1e-30)
    u = (Zp - Z0) / (Z2 - Z0 + 1e-30)

    f00 = F[i, j]
    f10 = F[i + 1, j]
    f01 = F[i, j + 1]
    f11 = F[i + 1, j + 1]

    return (1 - t) * (1 - u) * f00 + t * (1 - u) * f10 + (1 - t) * u * f01 + t * u * f11

def _polyline_length(xy: np.ndarray) -> float:
    d = np.diff(xy, axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))

def _polygon_area(xy: np.ndarray) -> float:
    # expects closed polygon (last==first) but works ok if not perfectly closed
    x = xy[:, 0]; y = xy[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))

def _contains_point(poly: np.ndarray, pt: Tuple[float, float]) -> bool:
    if Path is not None:
        try:
            return bool(Path(poly).contains_point(pt))
        except Exception:
            pass
    # fallback ray casting
    x, y = pt
    n = poly.shape[0]
    inside = False
    x0, y0 = poly[0, 0], poly[0, 1]
    for i in range(1, n + 1):
        x1, y1 = poly[i % n, 0], poly[i % n, 1]
        if ((y0 > y) != (y1 > y)) and (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-30) + x0):
            inside = not inside
        x0, y0 = x1, y1
    return inside

def _close_if_needed(seg: np.ndarray) -> np.ndarray:
    if seg.shape[0] < 3:
        return seg
    if float(np.linalg.norm(seg[0] - seg[-1])) > 0:
        return np.vstack([seg, seg[0]])
    return seg


# -------------------------
# Contours
# -------------------------
def _contour_segments(R1: np.ndarray, Z1: np.ndarray, psi_RZ: np.ndarray, level: float) -> List[np.ndarray]:
    """
    Returns list of segments (N,2) in (R,Z).
    """
    lvl = float(level)
    if not np.isfinite(lvl):
        return []

    # contourpy expects z shape (len(y), len(x)) = (nZ,nR)
    z = np.asarray(psi_RZ, float).T

    if contourpy is not None:
        cg = contourpy.contour_generator(x=R1, y=Z1, z=z, name="serial")
        segs = cg.lines(lvl)
        out = []
        for s in segs:
            s = np.asarray(s, float)
            if s.ndim == 2 and s.shape[1] == 2 and s.shape[0] >= 20:
                out.append(s)
        return out

    # matplotlib fallback
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        cs = ax.contour(R1, Z1, z, levels=[lvl])
        segs = cs.allsegs[0] if (cs is not None and cs.allsegs and cs.allsegs[0]) else []
        out = []
        for s in segs:
            s = np.asarray(s, float)
            if s.ndim == 2 and s.shape[1] == 2 and s.shape[0] >= 20:
                out.append(s)
        return out
    finally:
        plt.close(fig)


def _pick_boundary(segs: List[np.ndarray], R_ax: float, Z_ax: float, close_tol: float) -> Optional[np.ndarray]:
    """
    Pick the "best" closed-ish contour that encloses the axis.
    Choose the one with largest area among valid candidates.
    """
    cand = []
    for s in segs:
        s = np.asarray(s, float)
        if s.ndim != 2 or s.shape[1] != 2 or s.shape[0] < 20:
            continue
        gap = float(np.linalg.norm(s[0] - s[-1]))
        L = _polyline_length(s)
        if not (gap <= close_tol or (L > 0 and gap / L < 0.02)):
            continue
        poly = _close_if_needed(s)
        if not _contains_point(poly, (float(R_ax), float(Z_ax))):
            continue
        area = abs(_polygon_area(poly))
        cand.append((area, poly))
    if not cand:
        return None
    cand.sort(key=lambda t: t[0], reverse=True)
    return cand[0][1]


# -------------------------
# Axis + X-point detection
# -------------------------
def _magnetic_axis(eq: Any, R1: np.ndarray, Z1: np.ndarray, psi_RZ: np.ndarray) -> Tuple[float, float, float]:
    # best: eq.magneticAxis()
    if hasattr(eq, "magneticAxis"):
        try:
            ax = eq.magneticAxis()
            if isinstance(ax, (tuple, list)) and len(ax) >= 2:
                R_ax = float(ax[0]); Z_ax = float(ax[1])
                psi_ax = float(ax[2]) if len(ax) >= 3 else float("nan")
                if not np.isfinite(psi_ax):
                    psi_ax = float(_bilinear_interp(R1, Z1, psi_RZ, np.array([R_ax]), np.array([Z_ax]))[0])
                return R_ax, Z_ax, psi_ax
        except Exception:
            pass

    # fallback: pick extremum closer to center
    nR, nZ = psi_RZ.shape
    i0, j0 = nR // 2, nZ // 2
    imin = int(np.nanargmin(psi_RZ))
    imax = int(np.nanargmax(psi_RZ))
    i1, j1 = np.unravel_index(imin, psi_RZ.shape)
    i2, j2 = np.unravel_index(imax, psi_RZ.shape)
    d1 = (i1 - i0) ** 2 + (j1 - j0) ** 2
    d2 = (i2 - i0) ** 2 + (j2 - j0) ** 2
    i, j = (i1, j1) if d1 <= d2 else (i2, j2)
    return float(R1[i]), float(Z1[j]), float(psi_RZ[i, j])

def _find_xpoints_saddle(
    R1: np.ndarray,
    Z1: np.ndarray,
    psi_RZ: np.ndarray,
    *,
    max_points: int = 10,
    cand_count: int = 80,
    grad_q: float = 0.0025,
    min_sep_m: float = 0.18,
) -> List[Dict[str, float]]:
    """
    Saddle heuristic on psi:
      - compute grad^2
      - pick lowest grad^2 candidates
      - keep those with Hessian det < 0
      - cluster by min distance
    """
    psi = np.asarray(psi_RZ, float)
    nR, nZ = psi.shape

    # gradients (axis0=R, axis1=Z)
    dpsi_dR, dpsi_dZ = np.gradient(psi, R1, Z1, edge_order=1)
    g2 = dpsi_dR**2 + dpsi_dZ**2

    # ignore edges
    g2[:2, :] = np.inf
    g2[-2:, :] = np.inf
    g2[:, :2] = np.inf
    g2[:, -2:] = np.inf

    finite = np.isfinite(g2)
    if not np.any(finite):
        return []

    thr = np.quantile(g2[finite], float(grad_q))
    cand_mask = g2 <= thr
    cand_idx = np.argwhere(cand_mask)

    if cand_idx.shape[0] < 5:
        flat = g2.ravel()
        k = min(int(cand_count), flat.size)
        sel = np.argpartition(flat, k - 1)[:k]
        cand_idx = np.column_stack(np.unravel_index(sel, g2.shape))
    elif cand_idx.shape[0] > cand_count:
        vals = g2[cand_mask]
        order = np.argsort(vals)[:int(cand_count)]
        cand_idx = cand_idx[order]

    # Hessian (using gradients)
    d2psi_dRR, d2psi_dRZ = np.gradient(dpsi_dR, R1, Z1, edge_order=1)
    d2psi_dZR, d2psi_dZZ = np.gradient(dpsi_dZ, R1, Z1, edge_order=1)

    pts: List[Tuple[float, float, float]] = []
    for (i, j) in cand_idx:
        det = d2psi_dRR[i, j] * d2psi_dZZ[i, j] - 0.25 * (d2psi_dRZ[i, j] + d2psi_dZR[i, j]) ** 2
        if not np.isfinite(det) or det >= 0:
            continue
        R = float(R1[int(i)])
        Z = float(Z1[int(j)])
        p = float(psi[i, j])
        pts.append((R, Z, p))

    if not pts:
        return []

    out: List[Dict[str, float]] = []
    for R, Z, p in pts:
        keep = True
        for q in out:
            if math.hypot(R - q["R"], Z - q["Z"]) < float(min_sep_m):
                keep = False
                break
        if keep:
            out.append({"R": float(R), "Z": float(Z), "psi": float(p)})
        if len(out) >= int(max_points):
            break
    return out


# -------------------------
# Metrics from boundary
# -------------------------
def _metrics_from_boundary(R_sep: np.ndarray, Z_sep: np.ndarray) -> Dict[str, float]:
    R = np.asarray(R_sep, float)
    Z = np.asarray(Z_sep, float)

    Rmin = float(np.nanmin(R)); Rmax = float(np.nanmax(R))
    Zmin = float(np.nanmin(Z)); Zmax = float(np.nanmax(Z))

    R0 = 0.5 * (Rmax + Rmin)
    a = 0.5 * (Rmax - Rmin)
    if not np.isfinite(a) or a <= 1e-6:
        return dict(R0_plasma=float("nan"), a_plasma=float("nan"), A_plasma=float("nan"),
                    kappa_plasma=float("nan"), delta_u=float("nan"), delta_l=float("nan"))

    A = R0 / a
    # kappa = Zspan/(Rspan) == Zspan/(2a)
    kappa = (Zmax - Zmin) / (Rmax - Rmin + 1e-30)

    iu = int(np.argmax(Z)); il = int(np.argmin(Z))
    Ru = float(R[iu]); Rl = float(R[il])
    delta_u = (R0 - Ru) / a
    delta_l = (R0 - Rl) / a

    return dict(R0_plasma=float(R0), a_plasma=float(a), A_plasma=float(A),
                kappa_plasma=float(kappa), delta_u=float(delta_u), delta_l=float(delta_l))


# -------------------------
# LCFS-limiter (robust level selection)
# -------------------------
def _lcfs_limiter(
    R1: np.ndarray,
    Z1: np.ndarray,
    psi_RZ: np.ndarray,
    R_ax: float,
    Z_ax: float,
    psi_ax: float,
    geom: Dict[str, Any],
    *,
    prefer_inner: bool = True,
    edge_pad_cells: int = 2,
    psi_percentile: Optional[float] = 0.5,  # <=0 -> extreme
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any], float]:
    """
    Returns (R_sep, Z_sep, info, psi_lcfs_used)
    Chooses psi_lcfs by scanning candidate percentiles to ensure LCFS "touches" limiter (not too inside).
    """
    info = {"ok": False, "source": "lcfs_limiter"}

    if prefer_inner and ("R_inner" in geom and "Z_inner" in geom):
        R_lim = np.asarray(geom["R_inner"], float)
        Z_lim = np.asarray(geom["Z_inner"], float)
        touch_ref = float(np.nanmin(R_lim))  # inner limiter ~ smallest R
        touch_mode = "Rmin"
    else:
        R_lim = np.asarray(geom["R_outer"], float)
        Z_lim = np.asarray(geom["Z_outer"], float)
        touch_ref = float(np.nanmax(R_lim))  # outer limiter ~ largest R
        touch_mode = "Rmax"

    psi_lim = _bilinear_interp(R1, Z1, psi_RZ, R_lim, Z_lim)
    psi_lim = psi_lim[np.isfinite(psi_lim)]
    if psi_lim.size < 10:
        return None, None, {"ok": False, "source": "lcfs_limiter", "why": "psi_lim_empty"}, float("nan")

    # determine whether axis is min or max relative to edge
    psi_edge = float(np.nanmean(np.r_[psi_RZ[0, :], psi_RZ[-1, :], psi_RZ[:, 0], psi_RZ[:, -1]]))
    axis_is_min = bool(psi_edge > psi_ax)

    # build candidate levels (percentile scan)
    # If psi_percentile <= 0 => allow extreme (min/max) first.
    cand_p = []
    if psi_percentile is None:
        cand_p = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    else:
        p0 = float(psi_percentile)
        if p0 <= 0:
            cand_p = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
        else:
            # center scan around requested percentile
            cand_p = sorted(set([0.0, max(0.0, p0 / 2), p0, min(50.0, 2 * p0), 5.0, 10.0]))

    # closure tolerance based on grid
    dR = float(np.min(np.diff(R1))) if R1.size > 1 else 0.0
    dZ = float(np.min(np.diff(Z1))) if Z1.size > 1 else 0.0
    close_tol = 5.0 * max(dR, dZ, 1e-6)

    # edge-pad sanity
    Rmin_dom, Rmax_dom = float(R1[0]), float(R1[-1])
    Zmin_dom, Zmax_dom = float(Z1[0]), float(Z1[-1])
    padR = edge_pad_cells * dR
    padZ = edge_pad_cells * dZ

    best = None  # (touch_err, seg, psi_lcfs)
    for p in cand_p:
        p = float(max(0.0, min(50.0, p)))
        if axis_is_min:
            psi_lcfs = float(np.nanpercentile(psi_lim, p))
        else:
            psi_lcfs = float(np.nanpercentile(psi_lim, 100.0 - p))

        segs = _contour_segments(R1, Z1, psi_RZ, psi_lcfs)
        seg = _pick_boundary(segs, R_ax, Z_ax, close_tol=close_tol)
        if seg is None:
            continue

        R_sep = seg[:, 0]; Z_sep = seg[:, 1]

        # reject if on domain edge
        if (R_sep.min() <= Rmin_dom + padR) or (R_sep.max() >= Rmax_dom - padR) or (Z_sep.min() <= Zmin_dom + padZ) or (Z_sep.max() >= Zmax_dom - padZ):
            continue

        # touch error heuristic (avoid “LCFS too inside”)
        if touch_mode == "Rmin":
            touch_err = abs(float(R_sep.min()) - touch_ref)
        else:
            touch_err = abs(float(R_sep.max()) - touch_ref)

        if best is None or touch_err < best[0]:
            best = (touch_err, seg, psi_lcfs)

    if best is None:
        return None, None, {"ok": False, "source": "lcfs_limiter", "why": "no_valid_contour"}, float("nan")

    _, seg_best, psi_best = best
    info["ok"] = True
    info["touch_mode"] = touch_mode
    info["psi_percentile_scan"] = cand_p
    return np.asarray(seg_best[:, 0], float), np.asarray(seg_best[:, 1], float), info, float(psi_best)


# -------------------------
# Public API
# -------------------------
def analyze_star(
    eq: Any,
    geom: Dict[str, Any],
    *,
    require_two_x: bool = False,
    null_prefer: str = "lower",   # "lower" | "upper" | "any"
    max_xpoints: int = 10,
    # LCFS fallback controls
    prefer_inner_lcfs: bool = True,
    psi_percentile_lcfs: Optional[float] = 0.5,  # <=0 => extreme
    edge_pad_cells: int = 2,
) -> Dict[str, Any]:
    """
    Output keys (backward-compatible with your scan):
      ok_sep, reason, xpoints, R_sep, Z_sep,
      R0_plasma, A_plasma, kappa_plasma, delta_u, delta_l, a_plasma,
      R_ax, Z_ax, psi_ax, psi_sep, fallback_lcfs

    IMPORTANT FIX:
      - ok_sep == True ONLY when a true separatrix contour at psi_sep is found.
      - fallback LCFS (limiter-like) does NOT set ok_sep=True.
      - We add:
          has_true_separatrix (bool)
          has_closed_lcfs     (bool)
          psi_lcfs            (float, only in fallback mode)
    """
    out: Dict[str, Any] = dict(
        ok_sep=False,                 # True ONLY for true separatrix
        has_true_separatrix=False,
        has_closed_lcfs=False,
        reason="init",
        xpoints=[],
        R_sep=[],
        Z_sep=[],
        R0_plasma=float("nan"),
        A_plasma=float("nan"),
        kappa_plasma=float("nan"),
        delta_u=float("nan"),
        delta_l=float("nan"),
        a_plasma=float("nan"),
        R_ax=float("nan"),
        Z_ax=float("nan"),
        psi_ax=float("nan"),
        psi_sep=float("nan"),
        psi_lcfs=float("nan"),        # only meaningful in fallback mode
        fallback_lcfs=None,
    )

    try:
        psi = _psi_from_eq(eq)
        Rm, Zm = _grids_from_eq(eq)
        R1, Z1, psi_RZ = _infer_axes_and_orient(Rm, Zm, psi)

        if not np.isfinite(psi_RZ).all():
            out["reason"] = "non_finite_psi"
            return out

        # Axis
        R_ax, Z_ax, psi_ax = _magnetic_axis(eq, R1, Z1, psi_RZ)
        out["R_ax"] = float(R_ax)
        out["Z_ax"] = float(Z_ax)
        out["psi_ax"] = float(psi_ax)

        # X-point candidates (saddles)
        xps = _find_xpoints_saddle(
            R1, Z1, psi_RZ,
            max_points=int(max_xpoints),
            cand_count=80,
            grad_q=0.0025,
            min_sep_m=0.18,
        )
        out["xpoints"] = xps

        if require_two_x and len(xps) < 2:
            out["reason"] = "require_two_x_not_met"
            # don't return early; still allow fallback LCFS diagnostics below

        # Choose psi_sep from preferred X-point (if any)
        psi_sep = None
        if xps:
            pref = str(null_prefer).strip().lower()
            if pref == "upper":
                xp = max(xps, key=lambda d: float(d["Z"]))
            elif pref == "any":
                xp = xps[0]
            else:
                xp = min(xps, key=lambda d: float(d["Z"]))
            psi_sep = float(xp["psi"])
            out["psi_sep"] = float(psi_sep)

        # closure tolerance
        dR = float(np.min(np.diff(R1))) if R1.size > 1 else 0.0
        dZ = float(np.min(np.diff(Z1))) if Z1.size > 1 else 0.0
        close_tol = 5.0 * max(dR, dZ, 1e-6)

        # ---- TRUE separatrix attempt
        if psi_sep is not None and np.isfinite(psi_sep):
            segs = _contour_segments(R1, Z1, psi_RZ, psi_sep)
            seg = _pick_boundary(segs, R_ax, Z_ax, close_tol=close_tol)
            if seg is not None:
                R_sep = np.asarray(seg[:, 0], float)
                Z_sep = np.asarray(seg[:, 1], float)
                met = _metrics_from_boundary(R_sep, Z_sep)
                out.update(met)
                out["R_sep"] = [float(x) for x in R_sep.tolist()]
                out["Z_sep"] = [float(x) for x in Z_sep.tolist()]

                out["ok_sep"] = True
                out["has_true_separatrix"] = True
                out["has_closed_lcfs"] = True
                out["reason"] = "ok"
                out["fallback_lcfs"] = None
                return out

            # separatrix contour failed -> go fallback
            out["reason"] = "separatrix_contour_failed"
        else:
            out["reason"] = "no_xpoint_for_psi_sep"

        # ---- Fallback LCFS-limiter (NOT a true separatrix)
        Rf, Zf, info, psi_lcfs = _lcfs_limiter(
            R1, Z1, psi_RZ, R_ax, Z_ax, psi_ax, geom,
            prefer_inner=bool(prefer_inner_lcfs),
            edge_pad_cells=int(edge_pad_cells),
            psi_percentile=psi_percentile_lcfs,
        )
        out["fallback_lcfs"] = info
        out["psi_lcfs"] = float(psi_lcfs)

        if Rf is not None and Zf is not None and isinstance(info, dict) and info.get("ok", False):
            met = _metrics_from_boundary(Rf, Zf)
            out.update(met)
            out["R_sep"] = [float(x) for x in np.asarray(Rf, float).tolist()]
            out["Z_sep"] = [float(x) for x in np.asarray(Zf, float).tolist()]

            # KEY FIX:
            # This is a closed LCFS-like contour, but NOT a separatrix.
            out["ok_sep"] = False
            out["has_true_separatrix"] = False
            out["has_closed_lcfs"] = True
            # preserve earlier reason if separatrix failed; otherwise mark fallback
            if out.get("reason") in ("init", "no_xpoint_for_psi_sep"):
                out["reason"] = "fallback_lcfs_ok"
            return out

        out["ok_sep"] = False
        out["has_true_separatrix"] = False
        out["has_closed_lcfs"] = False
        out["reason"] = "fallback_lcfs_failed"
        return out

    except Exception as e:
        out["ok_sep"] = False
        out["has_true_separatrix"] = False
        out["has_closed_lcfs"] = False
        out["reason"] = f"exception:{repr(e)}"
        return out
