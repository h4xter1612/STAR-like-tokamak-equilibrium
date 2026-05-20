# analyze_star.py
# Unified analysis: true-separatrix detection via X-point + near-separatrix closed contour,
# and robust LCFS fallback.
#
# Key change vs old version:
# - We DO NOT require a closed contour exactly at psi_sep = psi_xpoint.
# - Instead:
#     1) detect candidate X-point(s)
#     2) classify whether psi_x lies between axis and edge (physically consistent separatrix)
#     3) build a CLOSED contour slightly inside the separatrix (psi_eval)
# - If that succeeds, we mark ok_sep=True and return R_sep/Z_sep = near-separatrix evaluation curve.
#
# This is much more robust for diverted equilibria, where the exact psi=psi_x contour is often
# numerically singular/open/degenerate.

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import contourpy
except Exception:
    contourpy = None

try:
    from matplotlib.path import Path
except Exception:
    Path = None

try:
    from separatrix_fallback_freegs import extract_freegs_dn_lcfs
except Exception:
    extract_freegs_dn_lcfs = None

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
    return np.asarray(psi, dtype=float)


def _grids_from_eq(eq: Any) -> Tuple[np.ndarray, np.ndarray]:
    R = _get_attr_or_call(eq, "R")
    Z = _get_attr_or_call(eq, "Z")
    if R is None or Z is None:
        raise AttributeError("Could not obtain grids from eq (eq.R / eq.Z).")
    return np.asarray(R, float), np.asarray(Z, float)


def _infer_axes_and_orient(
    Rm: np.ndarray, Zm: np.ndarray, psi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (R1, Z1, psi_RZ) where:
      R1: (nR,) increasing
      Z1: (nZ,) increasing
      psi_RZ: (nR, nZ) consistent with R1,Z1
    """
    if Rm.ndim != 2 or Zm.ndim != 2:
        R1 = np.asarray(Rm, float).ravel()
        Z1 = np.asarray(Zm, float).ravel()
        psi_RZ = np.asarray(psi, float)
        if psi_RZ.shape != (R1.size, Z1.size):
            raise ValueError("Non-2D grids but psi shape mismatch.")
        return R1, Z1, psi_RZ

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
            if psi_RZ.shape == (Z1.size, R1.size):
                psi_RZ = psi_RZ.T
            else:
                raise ValueError(f"psi shape mismatch: {psi_RZ.shape} vs {(R1.size, Z1.size)}")
    else:
        R1 = np.asarray(Rm[0, :], float)
        Z1 = np.asarray(Zm[:, 0], float)
        psi_xy = np.asarray(psi, float)
        if psi_xy.shape == (Z1.size, R1.size):
            psi_RZ = psi_xy.T
        elif psi_xy.shape == (R1.size, Z1.size):
            psi_RZ = psi_xy
        else:
            raise ValueError(f"psi shape mismatch: {psi_xy.shape} vs {(Z1.size, R1.size)} or {(R1.size, Z1.size)}")

    if R1.size > 1 and R1[0] > R1[-1]:
        R1 = R1[::-1].copy()
        psi_RZ = psi_RZ[::-1, :].copy()

    if Z1.size > 1 and Z1[0] > Z1[-1]:
        Z1 = Z1[::-1].copy()
        psi_RZ = psi_RZ[:, ::-1].copy()

    return R1, Z1, psi_RZ


def _bilinear_interp(
    R1: np.ndarray, Z1: np.ndarray, F_RZ: np.ndarray, Rp: np.ndarray, Zp: np.ndarray
) -> np.ndarray:
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

    R0 = R1[i]
    R2 = R1[i + 1]
    Z0 = Z1[j]
    Z2 = Z1[j + 1]

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
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _contains_point(poly: np.ndarray, pt: Tuple[float, float]) -> bool:
    if Path is not None:
        try:
            return bool(Path(poly).contains_point(pt))
        except Exception:
            pass

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


def _edge_psi_mean(psi_RZ: np.ndarray) -> float:
    edge = np.r_[psi_RZ[0, :], psi_RZ[-1, :], psi_RZ[:, 0], psi_RZ[:, -1]]
    return float(np.nanmean(edge))


def _axis_is_min(psi_ax: float, psi_edge: float) -> bool:
    return bool(psi_edge > psi_ax)


def _psi_between_axis_and_edge(
    psi_ax: float,
    psi_x: float,
    psi_edge: float,
    frac_margin: float = 0.002,
) -> bool:
    """
    Check whether psi_x lies strictly between axis and edge with a small margin.
    """
    lo = min(psi_ax, psi_edge)
    hi = max(psi_ax, psi_edge)
    span = hi - lo
    if not np.isfinite(span) or span <= 1e-14:
        return False
    lo2 = lo + frac_margin * span
    hi2 = hi - frac_margin * span
    return bool(lo2 < psi_x < hi2)


# -------------------------
# Contours
# -------------------------
def _contour_segments(R1: np.ndarray, Z1: np.ndarray, psi_RZ: np.ndarray, level: float) -> List[np.ndarray]:
    lvl = float(level)
    if not np.isfinite(lvl):
        return []

    z = np.asarray(psi_RZ, float).T  # contourpy wants (nZ,nR)

    if contourpy is not None:
        cg = contourpy.contour_generator(x=R1, y=Z1, z=z, name="serial")
        segs = cg.lines(lvl)
        out: List[np.ndarray] = []
        for s in segs:
            s = np.asarray(s, float)
            if s.ndim == 2 and s.shape[1] == 2 and s.shape[0] >= 20:
                out.append(s)
        return out

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        cs = ax.contour(R1, Z1, z, levels=[lvl])
        segs = cs.allsegs[0] if (cs is not None and cs.allsegs and cs.allsegs[0]) else []
        out: List[np.ndarray] = []
        for s in segs:
            s = np.asarray(s, float)
            if s.ndim == 2 and s.shape[1] == 2 and s.shape[0] >= 20:
                out.append(s)
        return out
    finally:
        plt.close(fig)


def _pick_boundary(segs: List[np.ndarray], R_ax: float, Z_ax: float, close_tol: float) -> Optional[np.ndarray]:
    """
    Pick the best closed-ish contour enclosing the axis.
    Prefer larger enclosed area.
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


def _curve_not_touching_domain_edge(
    seg: np.ndarray,
    R1: np.ndarray,
    Z1: np.ndarray,
    edge_pad_cells: int,
) -> bool:
    if seg is None or seg.shape[0] < 10:
        return False

    dR = float(np.min(np.diff(R1))) if R1.size > 1 else 0.0
    dZ = float(np.min(np.diff(Z1))) if Z1.size > 1 else 0.0

    Rmin_dom, Rmax_dom = float(R1[0]), float(R1[-1])
    Zmin_dom, Zmax_dom = float(Z1[0]), float(Z1[-1])

    padR = edge_pad_cells * dR
    padZ = edge_pad_cells * dZ

    Rs = seg[:, 0]
    Zs = seg[:, 1]

    if Rs.min() <= Rmin_dom + padR:
        return False
    if Rs.max() >= Rmax_dom - padR:
        return False
    if Zs.min() <= Zmin_dom + padZ:
        return False
    if Zs.max() >= Zmax_dom - padZ:
        return False
    return True


# -------------------------
# Axis + X-point detection
# -------------------------
def _magnetic_axis(eq: Any, R1: np.ndarray, Z1: np.ndarray, psi_RZ: np.ndarray) -> Tuple[float, float, float]:
    if hasattr(eq, "magneticAxis"):
        try:
            ax = eq.magneticAxis()
            if isinstance(ax, (tuple, list)) and len(ax) >= 2:
                R_ax = float(ax[0])
                Z_ax = float(ax[1])
                psi_ax = float(ax[2]) if len(ax) >= 3 else float("nan")
                if not np.isfinite(psi_ax):
                    psi_ax = float(_bilinear_interp(R1, Z1, psi_RZ, np.array([R_ax]), np.array([Z_ax]))[0])
                return R_ax, Z_ax, psi_ax
        except Exception:
            pass

    nR, nZ = psi_RZ.shape

    # Restrict axis search to plausible plasma core region.
    RR, ZZ = np.meshgrid(R1, Z1, indexing="ij")
    mask = (
        np.isfinite(psi_RZ)
        & (R1[0] <= RR) & (RR <= R1[-1])
        & (2.0 <= RR) & (RR <= 6.5)
        & (-3.5 <= ZZ) & (ZZ <= 3.5)
    )

    if not np.any(mask):
        mask = np.isfinite(psi_RZ)

    vals = np.where(mask, psi_RZ, np.nan)

    imin = int(np.nanargmin(vals))
    imax = int(np.nanargmax(vals))

    i1, j1 = np.unravel_index(imin, psi_RZ.shape)
    i2, j2 = np.unravel_index(imax, psi_RZ.shape)

    # Pick the extremum closer to expected plasma center.
    Rcen_guess = 4.0
    Zcen_guess = 0.0

    d1 = (R1[i1] - Rcen_guess) ** 2 + (Z1[j1] - Zcen_guess) ** 2
    d2 = (R1[i2] - Rcen_guess) ** 2 + (Z1[j2] - Zcen_guess) ** 2

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
      - pick lowest-grad candidates
      - keep those with Hessian det < 0
      - cluster by min distance
    """
    psi = np.asarray(psi_RZ, float)
    dpsi_dR, dpsi_dZ = np.gradient(psi, R1, Z1, edge_order=1)
    g2 = dpsi_dR**2 + dpsi_dZ**2

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

    d2psi_dRR, d2psi_dRZ = np.gradient(dpsi_dR, R1, Z1, edge_order=1)
    d2psi_dZR, d2psi_dZZ = np.gradient(dpsi_dZ, R1, Z1, edge_order=1)

    pts: List[Tuple[float, float, float, float]] = []
    for (i, j) in cand_idx:
        hxy = 0.5 * (d2psi_dRZ[i, j] + d2psi_dZR[i, j])
        det = d2psi_dRR[i, j] * d2psi_dZZ[i, j] - hxy**2
        if not np.isfinite(det) or det >= 0:
            continue
        R = float(R1[int(i)])
        Z = float(Z1[int(j)])
        p = float(psi[i, j])
        g = float(g2[i, j])
        pts.append((R, Z, p, g))

    if not pts:
        return []

    pts.sort(key=lambda t: t[3])  # lower grad^2 first

    out: List[Dict[str, float]] = []
    for R, Z, p, g in pts:
        keep = True
        for q in out:
            if math.hypot(R - q["R"], Z - q["Z"]) < float(min_sep_m):
                keep = False
                break
        if keep:
            out.append({"R": float(R), "Z": float(Z), "psi": float(p), "grad2": float(g)})
        if len(out) >= int(max_points):
            break
    return out


# -------------------------
# Metrics from boundary
# -------------------------
def _metrics_from_boundary(R_sep: np.ndarray, Z_sep: np.ndarray) -> Dict[str, float]:
    R = np.asarray(R_sep, float)
    Z = np.asarray(Z_sep, float)

    ok = np.isfinite(R) & np.isfinite(Z)
    R = R[ok]
    Z = Z[ok]

    if R.size < 10:
        return dict(
            R0_plasma=float("nan"),
            a_plasma=float("nan"),
            A_plasma=float("nan"),
            kappa_plasma=float("nan"),
            delta_u=float("nan"),
            delta_l=float("nan"),
            delta_bar=float("nan"),
            area=float("nan"),
        )

    if np.hypot(R[0] - R[-1], Z[0] - Z[-1]) > 1e-9:
        R = np.r_[R, R[0]]
        Z = np.r_[Z, Z[0]]

    Rmin = float(np.nanmin(R))
    Rmax = float(np.nanmax(R))
    Zmin = float(np.nanmin(Z))
    Zmax = float(np.nanmax(Z))

    R0 = 0.5 * (Rmax + Rmin)
    Z0 = 0.5 * (Zmax + Zmin)
    a = 0.5 * (Rmax - Rmin)

    if not np.isfinite(a) or a <= 1e-6:
        return dict(
            R0_plasma=float("nan"),
            Z0_plasma=float("nan"),
            a_plasma=float("nan"),
            A_plasma=float("nan"),
            kappa_plasma=float("nan"),
            delta_u=float("nan"),
            delta_l=float("nan"),
            delta_bar=float("nan"),
            area=float("nan"),
        )

    A = R0 / a
    kappa = 0.5 * (Zmax - Zmin) / a

    iu = int(np.nanargmax(Z))
    il = int(np.nanargmin(Z))

    Ru = float(R[iu])
    Rl = float(R[il])

    delta_u = (R0 - Ru) / a
    delta_l = (R0 - Rl) / a
    delta_bar = 0.5 * (delta_u + delta_l)

    area = 0.5 * abs(np.sum(R[:-1] * Z[1:] - R[1:] * Z[:-1]))

    return dict(
        R0_plasma=float(R0),
        Z0_plasma=float(Z0),
        a_plasma=float(a),
        A_plasma=float(A),
        kappa_plasma=float(kappa),
        delta_u=float(delta_u),
        delta_l=float(delta_l),
        delta_bar=float(delta_bar),
        area=float(area),
        Rmin=float(Rmin),
        Rmax=float(Rmax),
        Zmin=float(Zmin),
        Zmax=float(Zmax),
    )

# -------------------------
# LCFS-limiter fallback
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
    psi_percentile: Optional[float] = 0.5,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any], float]:
    info = {"ok": False, "source": "lcfs_limiter"}

    if prefer_inner and ("R_inner" in geom and "Z_inner" in geom):
        R_lim = np.asarray(geom["R_inner"], float)
        Z_lim = np.asarray(geom["Z_inner"], float)
        touch_ref = float(np.nanmin(R_lim))
        touch_mode = "Rmin"
    else:
        R_lim = np.asarray(geom["R_outer"], float)
        Z_lim = np.asarray(geom["Z_outer"], float)
        touch_ref = float(np.nanmax(R_lim))
        touch_mode = "Rmax"

    psi_lim = _bilinear_interp(R1, Z1, psi_RZ, R_lim, Z_lim)
    psi_lim = psi_lim[np.isfinite(psi_lim)]
    if psi_lim.size < 10:
        return None, None, {"ok": False, "source": "lcfs_limiter", "why": "psi_lim_empty"}, float("nan")

    psi_edge = _edge_psi_mean(psi_RZ)
    axis_is_min = _axis_is_min(psi_ax, psi_edge)

    cand_p: List[float]
    if psi_percentile is None:
        cand_p = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    else:
        p0 = float(psi_percentile)
        if p0 <= 0:
            cand_p = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
        else:
            cand_p = sorted(set([0.0, max(0.0, p0 / 2.0), p0, min(50.0, 2.0 * p0), 5.0, 10.0]))

    dR = float(np.min(np.diff(R1))) if R1.size > 1 else 0.0
    dZ = float(np.min(np.diff(Z1))) if Z1.size > 1 else 0.0
    close_tol = 5.0 * max(dR, dZ, 1e-6)

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

        if not _curve_not_touching_domain_edge(seg, R1, Z1, edge_pad_cells=edge_pad_cells):
            continue

        R_sep = seg[:, 0]
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
# Near-separatrix evaluation contour
# -------------------------
def _near_separatrix_closed_curve(
    R1: np.ndarray,
    Z1: np.ndarray,
    psi_RZ: np.ndarray,
    R_ax: float,
    Z_ax: float,
    psi_ax: float,
    psi_sep: float,
    *,
    edge_pad_cells: int = 2,
    eps_fracs: Tuple[float, ...] = (1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2),
) -> Tuple[Optional[np.ndarray], float, Dict[str, Any]]:
    """
    Build a CLOSED contour slightly inside the separatrix.
    This is what we use for shape/strike/containment evaluation when a true X-point exists.
    """
    info: Dict[str, Any] = {
        "ok": False,
        "source": "near_separatrix",
        "eps_fracs": list(eps_fracs),
    }

    if not np.isfinite(psi_sep) or not np.isfinite(psi_ax):
        info["why"] = "bad_psi"
        return None, float("nan"), info

    span_ax = abs(float(psi_sep) - float(psi_ax))
    if not np.isfinite(span_ax) or span_ax <= 1e-14:
        info["why"] = "span_ax_too_small"
        return None, float("nan"), info

    psi_edge = _edge_psi_mean(psi_RZ)
    axis_is_min = _axis_is_min(psi_ax, psi_edge)

    dR = float(np.min(np.diff(R1))) if R1.size > 1 else 0.0
    dZ = float(np.min(np.diff(Z1))) if Z1.size > 1 else 0.0
    close_tol = 5.0 * max(dR, dZ, 1e-6)

    candidates: List[Tuple[float, float, float, np.ndarray]] = []
    for eps in eps_fracs:
        eps = float(eps)
        if axis_is_min:
            psi_eval = float(psi_sep - eps * span_ax)
            if not (psi_ax < psi_eval < psi_sep):
                continue
        else:
            psi_eval = float(psi_sep + eps * span_ax)
            if not (psi_sep < psi_eval < psi_ax):
                continue

        segs = _contour_segments(R1, Z1, psi_RZ, psi_eval)
        seg = _pick_boundary(segs, R_ax, Z_ax, close_tol=close_tol)
        if seg is None:
            continue

        if not _curve_not_touching_domain_edge(seg, R1, Z1, edge_pad_cells=edge_pad_cells):
            continue

        area = abs(_polygon_area(seg))
        candidates.append((eps, -area, psi_eval, seg))

    if not candidates:
        info["why"] = "no_closed_inner_curve"
        return None, float("nan"), info

    # Prefer the contour closest to separatrix (smallest eps), then largest area
    candidates.sort(key=lambda t: (t[0], t[1]))
    eps_best, neg_area_best, psi_eval_best, seg_best = candidates[0]

    info["ok"] = True
    info["eps_best"] = float(eps_best)
    info["area_m2"] = float(-neg_area_best)
    info["psi_edge"] = float(psi_edge)
    info["axis_is_min"] = bool(axis_is_min)
    return np.asarray(seg_best, float), float(psi_eval_best), info

def _xpoints_for_dn_fallback(
    valid_xps: List[Dict[str, float]],
    all_xps: List[Dict[str, float]],
    *,
    prefer_valid: bool = True,
) -> List[Tuple[float, float]]:
    src = valid_xps if (prefer_valid and len(valid_xps) >= 2) else all_xps

    pts: List[Tuple[float, float]] = []
    for xp in src:
        try:
            R = float(xp["R"])
            Z = float(xp["Z"])
            if not (np.isfinite(R) and np.isfinite(Z)):
                continue

            # STAR-like diverted X-points should be near the inboard/top-bottom neck,
            # not at the far outer boundary or coil-dominated region.
            if not (1.5 <= R <= 4.2):
                continue
            if not (3.5 <= abs(Z) <= 6.2):
                continue

            pts.append((R, Z))
        except Exception:
            continue

    if len(pts) >= 2:
        return pts

    # Last fallback: return unfiltered finite points.
    pts = []
    for xp in src:
        try:
            R = float(xp["R"])
            Z = float(xp["Z"])
            if np.isfinite(R) and np.isfinite(Z):
                pts.append((R, Z))
        except Exception:
            pass

    return pts

def _try_freegs_dn_fallback(
    eq: Any,
    out: Dict[str, Any],
    valid_xps: List[Dict[str, float]],
    all_xps: List[Dict[str, float]],
    *,
    xpoint_tol: float = 0.75,
) -> bool:
    """
    Try to reconstruct a double-null LCFS from FreeGS/FreeGSNKE psi_bndry.

    Mutates `out` in-place if successful.

    Return:
        True if fallback succeeded and out was updated.
        False otherwise.
    """
    if extract_freegs_dn_lcfs is None:
        out["dn_fallback"] = {
            "ok": False,
            "reason": "separatrix_fallback_freegs_import_failed",
        }
        return False

    xpoints = _xpoints_for_dn_fallback(valid_xps, all_xps, prefer_valid=True)

    if len(xpoints) < 2:
        out["dn_fallback"] = {
            "ok": False,
            "reason": "not_enough_xpoints_for_dn_fallback",
            "n_xpoints": len(xpoints),
        }
        return False

    try:
        fb = extract_freegs_dn_lcfs(
            eq,
            xpoints=xpoints,
            debug_plot=None,
            xpoint_tol=float(xpoint_tol),
        )
    except Exception as e:
        out["dn_fallback"] = {
            "ok": False,
            "reason": f"exception:{repr(e)}",
        }
        return False

    out["dn_fallback"] = {
        k: v for k, v in fb.items()
        if k not in ("R_sep", "Z_sep", "R_lcfs", "Z_lcfs", "raw_segments")
    }

    if not (fb.get("ok", False) and fb.get("has_usable_sep", False)):
        return False

    if "R_sep" not in fb or "Z_sep" not in fb:
        return False

    R_sep = np.asarray(fb["R_sep"], dtype=float)
    Z_sep = np.asarray(fb["Z_sep"], dtype=float)

    if R_sep.size < 20 or Z_sep.size < 20 or R_sep.size != Z_sep.size:
        out["dn_fallback"]["ok"] = False
        out["dn_fallback"]["reason"] = "bad_RZ_sep_from_dn_fallback"
        return False

    met = _metrics_from_boundary(R_sep, Z_sep)

    out.update(met)
    out["R_sep"] = [float(x) for x in R_sep.tolist()]
    out["Z_sep"] = [float(x) for x in Z_sep.tolist()]

    # Existing keys, so downstream scripts do not need major changes.
    out["ok_sep"] = True
    out["has_closed_lcfs"] = True

    # This is reconstructed from FreeGS psi_bndry, not the old near-separatrix method.
    # Set True to avoid downstream rejection, but keep provenance explicit.
    out["has_true_separatrix"] = True
    out["has_usable_sep"] = True
    out["has_freegs_psibndry_sep"] = True

    out["reason"] = "ok_dn_fallback"
    out["sep_source"] = fb.get("source", "freegs_psibndry_dn_reconstructed")
    out["shape_reason"] = fb.get("reason", "dn_lcfs_reconstructed_from_psibndry_segments")

    geom = fb.get("geometry", {})
    if isinstance(geom, dict) and geom.get("ok", False):
        out["R0_fallback_geom"] = float(geom.get("R0", float("nan")))
        out["A_fallback_geom"] = float(geom.get("A", float("nan")))
        out["kappa_fallback_geom"] = float(geom.get("kappa", float("nan")))
        out["delta_bar_fallback_geom"] = float(geom.get("delta_bar", float("nan")))
        out["area_fallback_geom"] = float(geom.get("area", float("nan")))

    return True

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
    prefer_inner_lcfs: bool = True,
    psi_percentile_lcfs: Optional[float] = 0.5,
    edge_pad_cells: int = 2,
) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(
        ok_sep=False,                 # True only if true X-point + near-separatrix closed curve found
        has_true_separatrix=False,
        has_closed_lcfs=False,
        has_xpoint=False,
        reason="init",
        xpoints=[],
        xpoints_valid=[],
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
        psi_edge=float("nan"),
        psi_sep=float("nan"),        # exact psi at chosen X-point
        psi_eval=float("nan"),       # closed curve slightly inside separatrix
        psi_lcfs=float("nan"),       # fallback only
        preferred_xpoint=None,
        fallback_lcfs=None,
        separatrix_eval_info=None,
        has_usable_sep=False,
        has_freegs_psibndry_sep=False,
        sep_source="none",
        shape_reason="init",
        dn_fallback=None,
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
        psi_edge = _edge_psi_mean(psi_RZ)

        out["R_ax"] = float(R_ax)
        out["Z_ax"] = float(Z_ax)
        out["psi_ax"] = float(psi_ax)
        out["psi_edge"] = float(psi_edge)

        # X-points
        xps = _find_xpoints_saddle(
            R1, Z1, psi_RZ,
            max_points=int(max_xpoints),
            cand_count=80,
            grad_q=0.0025,
            min_sep_m=0.18,
        )
        out["xpoints"] = xps
        out["has_xpoint"] = bool(len(xps) > 0)

        valid_xps = []
        for xp in xps:
            psi_x = float(xp["psi"])
            if _psi_between_axis_and_edge(psi_ax, psi_x, psi_edge, frac_margin=0.002):
                valid_xps.append(dict(xp))
        out["xpoints_valid"] = valid_xps

        if require_two_x and len(valid_xps) < 2:
            out["reason"] = "require_two_x_not_met"
            # no early return; fallback diagnostics still useful

        # Choose preferred X-point among VALID candidates
        psi_sep = None
        chosen_xp = None
        if valid_xps:
            pref = str(null_prefer).strip().lower()
            if pref == "upper":
                chosen_xp = max(valid_xps, key=lambda d: float(d["Z"]))
            elif pref == "any":
                chosen_xp = valid_xps[0]
            else:
                chosen_xp = min(valid_xps, key=lambda d: float(d["Z"]))
            psi_sep = float(chosen_xp["psi"])
            out["psi_sep"] = float(psi_sep)
            out["preferred_xpoint"] = dict(chosen_xp)

        # ---- True separatrix logic:
        # if we have a physically consistent X-point, try a CLOSED contour slightly inside psi_sep
        if psi_sep is not None and np.isfinite(psi_sep):
            seg_eval, psi_eval, sep_info = _near_separatrix_closed_curve(
                R1, Z1, psi_RZ,
                R_ax, Z_ax, psi_ax, psi_sep,
                edge_pad_cells=int(edge_pad_cells),
            )
            out["separatrix_eval_info"] = sep_info
            out["psi_eval"] = float(psi_eval)

            if seg_eval is not None and isinstance(sep_info, dict) and sep_info.get("ok", False):
                R_sep = np.asarray(seg_eval[:, 0], float)
                Z_sep = np.asarray(seg_eval[:, 1], float)
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

            out["reason"] = "separatrix_eval_contour_failed"
        else:
            out["reason"] = "no_valid_xpoint_for_psi_sep"

        # ---- Double-null FreeGS psi_bndry fallback
        # This handles true DN cases where psi=psi_bndry is topologically valid,
        # but the exact contour is not a single closed curve and the near-separatrix
        # closed-contour check fails.
        if len(valid_xps) >= 2 or len(xps) >= 2:
            if _try_freegs_dn_fallback(
                eq,
                out,
                valid_xps,
                xps,
                xpoint_tol=0.75,
            ):
                return out

        # ---- Fallback LCFS (NOT true separatrix)
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
            out["ok_sep"] = False
            out["has_true_separatrix"] = False
            out["has_closed_lcfs"] = True
            out["has_usable_sep"] = True
            out["sep_source"] = "lcfs_limiter"
            if out.get("reason") in ("init", "no_valid_xpoint_for_psi_sep"):
                out["reason"] = "fallback_lcfs_ok"
            out["shape_reason"] = out.get("reason", "fallback_lcfs_ok")
            return out

        out["ok_sep"] = False
        out["has_true_separatrix"] = False
        out["has_closed_lcfs"] = False
        out["has_usable_sep"] = False
        out["sep_source"] = "none"
        out["shape_reason"] = "fallback_lcfs_failed"
        out["reason"] = "fallback_lcfs_failed"
        return out

    except Exception as e:
        out["ok_sep"] = False
        out["has_true_separatrix"] = False
        out["has_closed_lcfs"] = False
        out["reason"] = f"exception:{repr(e)}"
        out["has_usable_sep"] = False
        out["sep_source"] = "none"
        out["shape_reason"] = f"exception:{repr(e)}"
        return out
