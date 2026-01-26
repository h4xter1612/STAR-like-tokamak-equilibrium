# analyze_star_lcfs.py
"""
Robust LCFS extraction for FreeGSNKE when "critical points" (X/O points)
aren't reliably found (common with limiter-like equilibria).

Method:
- Take inner wall (limiter) if available; else outer wall.
- Sample psi on that wall.
- Define boundary psi_b as a robust extreme of psi on the wall (percentile).
- Extract contour psi=psi_b and choose the closed contour that contains the
  magnetic axis (largest area).
- Compute R0, a, A, kappa, and triangularity from that contour.

This version is compatible with Matplotlib variants where QuadContourSet
may not expose `.collections`. We use `cs.allsegs` as the primary path source.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath


def _bilinear_on_rect_grid(R1d, Z1d, F, rq, zq) -> float:
    R1d = np.asarray(R1d, float)
    Z1d = np.asarray(Z1d, float)
    F = np.asarray(F, float)
    rq = float(rq)
    zq = float(zq)

    if rq <= R1d[0]:
        i = 0
    elif rq >= R1d[-1]:
        i = len(R1d) - 2
    else:
        i = int(np.searchsorted(R1d, rq) - 1)

    if zq <= Z1d[0]:
        j = 0
    elif zq >= Z1d[-1]:
        j = len(Z1d) - 2
    else:
        j = int(np.searchsorted(Z1d, zq) - 1)

    r0, r1 = R1d[i], R1d[i + 1]
    z0, z1 = Z1d[j], Z1d[j + 1]
    if (r1 - r0) == 0 or (z1 - z0) == 0:
        return float(F[j, i])

    tx = (rq - r0) / (r1 - r0)
    tz = (zq - z0) / (z1 - z0)

    f00 = F[j, i]
    f10 = F[j, i + 1]
    f01 = F[j + 1, i]
    f11 = F[j + 1, i + 1]

    return float((1 - tx) * (1 - tz) * f00 + tx * (1 - tz) * f10 + (1 - tx) * tz * f01 + tx * tz * f11)


def _poly_area(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3:
        return 0.0
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _close_xy(x, y, tol=1e-12):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) == 0:
        return x, y
    if (abs(x[0] - x[-1]) + abs(y[0] - y[-1])) > tol:
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
    return x, y


def _extract_contour_segments(cs) -> list[np.ndarray]:
    """
    Return list of (N,2) arrays of contour vertices for the first (and only) level.
    Compatible across Matplotlib variants.

    Prefer `cs.allsegs` which is stable:
      cs.allsegs[level_index] -> list of arrays (N,2)
    Fallback to `cs.collections` if present.
    """
    segs: list[np.ndarray] = []

    # Primary: allsegs
    if hasattr(cs, "allsegs") and cs.allsegs:
        # With levels=[psi_b], there is exactly one level -> index 0
        try:
            for s in cs.allsegs[0]:
                s = np.asarray(s, float)
                if s.ndim == 2 and s.shape[1] == 2 and s.shape[0] >= 4:
                    segs.append(s)
            if segs:
                return segs
        except Exception:
            pass

    # Fallback: collections -> paths
    if hasattr(cs, "collections"):
        try:
            for col in cs.collections:
                for p in col.get_paths():
                    v = np.asarray(p.vertices, float)
                    if v.ndim == 2 and v.shape[1] == 2 and v.shape[0] >= 4:
                        segs.append(v)
        except Exception:
            pass

    return segs


def shape_from_lcfs_limiter(eq, geom, *, prefer_inner=True, psi_percentile=5.0):
    psi = eq.psi()
    R = eq.R
    Z = eq.Z

    # 1D axes (assumes meshgrid-like)
    R1d = np.asarray(R[0, :], float)
    Z1d = np.asarray(Z[:, 0], float)

    # Magnetic axis
    R_ax, Z_ax = eq.magneticAxis()[:2]
    psi_ax = _bilinear_on_rect_grid(R1d, Z1d, psi, R_ax, Z_ax)

    # Choose limiter polyline
    if prefer_inner and ("R_inner" in geom) and ("Z_inner" in geom):
        Rw = np.asarray(geom["R_inner"], float)
        Zw = np.asarray(geom["Z_inner"], float)
    else:
        Rw = np.asarray(geom["R_outer"], float)
        Zw = np.asarray(geom["Z_outer"], float)

    # psi on wall/limiter
    psi_w = np.array([_bilinear_on_rect_grid(R1d, Z1d, psi, r, z) for r, z in zip(Rw, Zw)], float)

    # Determine whether axis is "min-like" or "max-like" relative to wall
    med_w = float(np.median(psi_w))
    axis_is_min = (psi_ax < med_w)

    # Robust extreme on wall (percentile avoids a single bad point)
    if axis_is_min:
        psi_b = float(np.percentile(psi_w, float(psi_percentile)))
    else:
        psi_b = float(np.percentile(psi_w, 100.0 - float(psi_percentile)))

    # Extract contour at psi_b
    fig, ax = plt.subplots()
    cs = ax.contour(R, Z, psi, levels=[psi_b])
    plt.close(fig)

    segs = _extract_contour_segments(cs)
    if not segs:
        raise RuntimeError("LCFS extraction failed: no contour segments found (Matplotlib contour empty).")

    best = None
    best_area = -np.inf

    # Prefer contour containing the magnetic axis
    for v in segs:
        if v.shape[0] < 20:
            continue
        poly = MplPath(v)
        if not poly.contains_point((R_ax, Z_ax)):
            continue
        x, y = v[:, 0], v[:, 1]
        x, y = _close_xy(x, y)
        area = abs(_poly_area(x, y))
        if area > best_area:
            best_area = area
            best = (x, y)

    # Fallback: largest area contour
    if best is None:
        for v in segs:
            if v.shape[0] < 20:
                continue
            x, y = v[:, 0], v[:, 1]
            x, y = _close_xy(x, y)
            area = abs(_poly_area(x, y))
            if area > best_area:
                best_area = area
                best = (x, y)

    if best is None:
        raise RuntimeError("LCFS extraction failed: no suitable contour found.")

    R_sep, Z_sep = best

    # Shape metrics
    Rmin, Rmax = float(np.min(R_sep)), float(np.max(R_sep))
    Zmin, Zmax = float(np.min(Z_sep)), float(np.max(Z_sep))
    R0_pl = 0.5 * (Rmin + Rmax)
    a_pl = 0.5 * (Rmax - Rmin)
    if a_pl <= 0:
        raise RuntimeError("Degenerate LCFS: a_plasma <= 0.")

    A_pl = R0_pl / a_pl
    kappa = (Zmax - Zmin) / (2.0 * a_pl)

    # Triangularity (top/bottom)
    i_top = int(np.argmax(Z_sep))
    i_bot = int(np.argmin(Z_sep))
    R_top = float(R_sep[i_top])
    R_bot = float(R_sep[i_bot])
    delta_u = (R0_pl - R_top) / a_pl
    delta_l = (R0_pl - R_bot) / a_pl

    return dict(
        R_sep=R_sep,
        Z_sep=Z_sep,
        R0_plasma=float(R0_pl),
        a_plasma=float(a_pl),
        A_plasma=float(A_pl),
        kappa_plasma=float(kappa),
        delta_u=float(delta_u),
        delta_l=float(delta_l),
    )

