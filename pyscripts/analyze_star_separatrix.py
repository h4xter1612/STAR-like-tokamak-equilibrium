"""
analyze_star_separatrix.py

Robust separatrix + X-point analysis for FreeGSNKE equilibria.

Key fixes vs fragile versions:
  - Uses QuadContourSet.allsegs (NOT cs.collections) => avoids Matplotlib API issues
  - Handles eq.psi as either method or attribute
  - X-point detection via saddle-point heuristic on psi gradients/Hessian
  - Optional fallback to LCFS-limiter extractor (if analyze_star_lcfs is available),
    called with signature introspection to avoid unexpected-kw errors.

Public API:
  - shape_from_separatrix(eq, geom, require_two_x=False, null_prefer="lower", ...)

Returned dict is JSON-friendly (lists + floats; may contain NaN if extraction fails).
"""

from __future__ import annotations

import math
import inspect
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


# -------------------------
# Utilities: safe access
# -------------------------

def _as_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _get_attr_or_call(obj: Any, names: Tuple[str, ...]) -> Any:
    """
    Try a list of attribute names; if attribute is callable, call it with no args.
    """
    for n in names:
        if not hasattr(obj, n):
            continue
        v = getattr(obj, n)
        try:
            return v() if callable(v) else v
        except TypeError:
            # callable but requires args -> ignore
            continue
        except Exception:
            continue
    return None

def _grid_from_eq(eq: Any) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns:
      R2, Z2  : 2D grids
      R1, Z1  : 1D coordinate arrays if we can infer them (monotone), else None

    Assumes typical meshgrid layout:
      R2[0, :] is R axis; Z2[:, 0] is Z axis
    """
    R2 = _get_attr_or_call(eq, ("R",))
    Z2 = _get_attr_or_call(eq, ("Z",))
    if R2 is None or Z2 is None:
        # try alternatives
        R2 = _get_attr_or_call(eq, ("Rg", "RR", "Rgrid"))
        Z2 = _get_attr_or_call(eq, ("Zg", "ZZ", "Zgrid"))

    if R2 is None or Z2 is None:
        raise AttributeError("Could not find eq.R and eq.Z grids")

    R2 = np.asarray(R2, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)

    R1 = None
    Z1 = None

    try:
        r = np.asarray(R2[0, :], dtype=float)
        z = np.asarray(Z2[:, 0], dtype=float)
        # check monotonic
        if (np.all(np.diff(r) > 0) or np.all(np.diff(r) < 0)) and (np.all(np.diff(z) > 0) or np.all(np.diff(z) < 0)):
            # enforce ascending for interpolation/gradient spacing
            if np.diff(r).mean() < 0:
                r = r[::-1]
                R2 = R2[:, ::-1]
                # Z2 and psi should be flipped consistently by caller if needed
            if np.diff(z).mean() < 0:
                z = z[::-1]
                Z2 = Z2[::-1, :]
            R1 = r
            Z1 = z
    except Exception:
        pass

    return R2, Z2, R1, Z1

def _psi_from_eq(eq: Any) -> np.ndarray:
    """
    Get psi grid robustly:
      - eq.psi() method
      - eq.psi attribute
      - eq.Psi / eq.psi_grid / eq.plasma_psi etc
    """
    psi = _get_attr_or_call(eq, ("psi", "Psi", "psi_grid", "plasma_psi", "psiRZ", "psi_func"))
    if psi is None:
        raise AttributeError("Could not obtain psi from equilibrium (eq.psi/eq.Psi/...)")
    psi = np.asarray(psi, dtype=float)
    return psi

def _magnetic_axis(eq: Any, R2: np.ndarray, Z2: np.ndarray, psi: np.ndarray) -> Tuple[float, float, float]:
    """
    Try eq.magneticAxis(); else approximate as global extremum near center.

    Returns R_ax, Z_ax, psi_ax
    """
    # Preferred
    if hasattr(eq, "magneticAxis"):
        try:
            ax = eq.magneticAxis()
            # could be (R,Z) or (R,Z,psi)
            if isinstance(ax, (tuple, list)) and len(ax) >= 2:
                R_ax = float(ax[0])
                Z_ax = float(ax[1])
                psi_ax = float(ax[2]) if len(ax) >= 3 else float(np.nan)
                if not np.isfinite(psi_ax):
                    # interpolate psi at axis approximately by nearest cell
                    i = int(np.argmin((R2 - R_ax) ** 2 + (Z2 - Z_ax) ** 2))
                    ii, jj = np.unravel_index(i, psi.shape)
                    psi_ax = float(psi[ii, jj])
                return R_ax, Z_ax, psi_ax
        except Exception:
            pass

    # Fallback: choose global extremum; often axis is min/max of psi inside domain
    # Pick extremum closer to middle of domain to avoid boundary artifacts.
    nr, nc = psi.shape
    ic0 = nr // 2
    jc0 = nc // 2

    # consider both min and max
    imin = int(np.nanargmin(psi))
    imax = int(np.nanargmax(psi))
    i1, j1 = np.unravel_index(imin, psi.shape)
    i2, j2 = np.unravel_index(imax, psi.shape)

    d1 = (i1 - ic0) ** 2 + (j1 - jc0) ** 2
    d2 = (i2 - ic0) ** 2 + (j2 - jc0) ** 2
    if d1 <= d2:
        i, j = i1, j1
    else:
        i, j = i2, j2

    return float(R2[i, j]), float(Z2[i, j]), float(psi[i, j])


# -------------------------
# Bilinear interpolation (for wall sampling / sanity checks)
# -------------------------

def _bilinear_interp_rect(R1: np.ndarray, Z1: np.ndarray, F: np.ndarray, Rq: np.ndarray, Zq: np.ndarray) -> np.ndarray:
    """
    Bilinear interpolation for rectilinear grid:
      F shape (len(Z1), len(R1))
    """
    Rq = np.asarray(Rq, float)
    Zq = np.asarray(Zq, float)

    # clip inside domain
    Rq = np.clip(Rq, R1.min(), R1.max())
    Zq = np.clip(Zq, Z1.min(), Z1.max())

    j = np.searchsorted(R1, Rq, side="right") - 1
    i = np.searchsorted(Z1, Zq, side="right") - 1
    j = np.clip(j, 0, len(R1) - 2)
    i = np.clip(i, 0, len(Z1) - 2)

    R0 = R1[j]
    R1n = R1[j + 1]
    Z0 = Z1[i]
    Z1n = Z1[i + 1]

    t = (Rq - R0) / np.maximum(R1n - R0, 1e-30)
    u = (Zq - Z0) / np.maximum(Z1n - Z0, 1e-30)

    f00 = F[i, j]
    f10 = F[i, j + 1]
    f01 = F[i + 1, j]
    f11 = F[i + 1, j + 1]

    return (1 - t) * (1 - u) * f00 + t * (1 - u) * f10 + (1 - t) * u * f01 + t * u * f11


# -------------------------
# Contour extraction (Matplotlib robust)
# -------------------------

def _contour_segments(R2: np.ndarray, Z2: np.ndarray, F: np.ndarray, level: float) -> List[np.ndarray]:
    """
    Returns a list of segments, each segment is (N,2) array of [R,Z].
    Uses QuadContourSet.allsegs => robust vs Matplotlib API changes.
    """
    fig = plt.figure()
    try:
        cs = plt.contour(R2, Z2, F, levels=[float(level)])
        segs = cs.allsegs[0] if (cs is not None and len(cs.allsegs) > 0) else []
        out = []
        for s in segs:
            s = np.asarray(s, dtype=float)
            if s.ndim == 2 and s.shape[1] == 2 and s.shape[0] >= 10:
                out.append(s)
        return out
    finally:
        plt.close(fig)

def _polyline_length(seg: np.ndarray) -> float:
    d = np.diff(seg, axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))

def _is_closed(seg: np.ndarray, tol: float = 1e-2) -> bool:
    if seg.shape[0] < 3:
        return False
    return float(np.hypot(seg[0, 0] - seg[-1, 0], seg[0, 1] - seg[-1, 1])) <= float(tol)

def _point_in_poly(x: float, y: float, poly: np.ndarray) -> bool:
    """
    Ray casting algorithm. poly is (N,2).
    """
    n = poly.shape[0]
    inside = False
    x0, y0 = poly[0, 0], poly[0, 1]
    for i in range(1, n + 1):
        x1, y1 = poly[i % n, 0], poly[i % n, 1]
        # check edge crossing
        if ((y0 > y) != (y1 > y)) and (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-30) + x0):
            inside = not inside
        x0, y0 = x1, y1
    return inside

def _choose_separatrix_segment(
    segs: List[np.ndarray],
    R_ax: float,
    Z_ax: float,
) -> Optional[np.ndarray]:
    """
    Choose a "best" separatrix segment:
      - prefer closed segments that contain the magnetic axis
      - among those, choose the longest
      - else choose the segment with smallest distance to axis (but mark as not ok later)
    """
    if not segs:
        return None

    closed_containing = []
    for s in segs:
        if _is_closed(s, tol=5e-2):
            if _point_in_poly(R_ax, Z_ax, s):
                closed_containing.append(s)

    if closed_containing:
        return max(closed_containing, key=_polyline_length)

    # fallback: nearest to axis
    def dist_to_axis(s: np.ndarray) -> float:
        d = np.hypot(s[:, 0] - R_ax, s[:, 1] - Z_ax)
        return float(np.min(d))

    return min(segs, key=dist_to_axis)


# -------------------------
# X-point detection (saddle heuristic)
# -------------------------

def _find_xpoints_saddle(
    R2: np.ndarray,
    Z2: np.ndarray,
    R1: Optional[np.ndarray],
    Z1: Optional[np.ndarray],
    psi: np.ndarray,
    *,
    max_points: int = 10,
    cand_count: int = 80,
    grad_q: float = 0.003,
    min_sep_m: float = 0.15,
) -> List[Dict[str, float]]:
    """
    Heuristic:
      - compute grad(psi) and Hessian
      - candidates = smallest grad^2 cells (excluding edges)
      - keep those with Hessian determinant < 0 (saddle)
      - cluster by distance

    Returns list of dicts: {R,Z,psi}
    """
    psi = np.asarray(psi, float)
    nr, nc = psi.shape

    # spacing-aware gradients if 1D coords exist
    if R1 is not None and Z1 is not None and (len(R1) == nc) and (len(Z1) == nr):
        dpsi_dZ, dpsi_dR = np.gradient(psi, Z1, R1, edge_order=1)
    else:
        dpsi_dZ, dpsi_dR = np.gradient(psi, edge_order=1)

    g2 = dpsi_dR**2 + dpsi_dZ**2

    # ignore edges
    g2[:2, :] = np.inf
    g2[-2:, :] = np.inf
    g2[:, :2] = np.inf
    g2[:, -2:] = np.inf

    finite = np.isfinite(g2)
    if not np.any(finite):
        return []

    # threshold using quantile (more stable than absolute)
    thr = np.quantile(g2[finite], float(grad_q))
    cand_mask = g2 <= thr
    cand_idx = np.argwhere(cand_mask)

    # if too many/too few, take argpartition on flattened
    if cand_idx.shape[0] < 5:
        flat = g2.ravel()
        k = min(int(cand_count), flat.size)
        sel = np.argpartition(flat, k - 1)[:k]
        cand_idx = np.column_stack(np.unravel_index(sel, g2.shape))
    elif cand_idx.shape[0] > cand_count:
        # keep best cand_count by g2
        vals = g2[cand_mask]
        order = np.argsort(vals)[:int(cand_count)]
        cand_idx = cand_idx[order]

    # Hessian
    if R1 is not None and Z1 is not None and (len(R1) == nc) and (len(Z1) == nr):
        d2psi_dZZ, d2psi_dZR = np.gradient(dpsi_dZ, Z1, R1, edge_order=1)
        d2psi_dRZ, d2psi_dRR = np.gradient(dpsi_dR, Z1, R1, edge_order=1)
    else:
        d2psi_dZZ, d2psi_dZR = np.gradient(dpsi_dZ, edge_order=1)
        d2psi_dRZ, d2psi_dRR = np.gradient(dpsi_dR, edge_order=1)

    # evaluate candidates
    pts: List[Tuple[float, float, float]] = []
    for (i, j) in cand_idx:
        det = d2psi_dRR[i, j] * d2psi_dZZ[i, j] - 0.25 * (d2psi_dRZ[i, j] + d2psi_dZR[i, j])**2
        if not np.isfinite(det):
            continue
        if det >= 0.0:
            continue  # not a saddle

        R = float(R2[i, j])
        Z = float(Z2[i, j])
        p = float(psi[i, j])
        pts.append((R, Z, p))

    if not pts:
        return []

    # sort by increasing grad^2 or |det|? We'll use grad^2 implicit by earlier selection
    # cluster distinct points
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
# Shape metrics from boundary
# -------------------------

def _nearest_index(arr: np.ndarray, val: float) -> int:
    return int(np.argmin(np.abs(arr - float(val))))

def _plasma_metrics_from_sep(R_sep: np.ndarray, Z_sep: np.ndarray) -> Dict[str, float]:
    """
    Compute STAR-like geometric metrics from separatrix curve.
    Returns R0_plasma, a_plasma, A_plasma, kappa_plasma, delta_u, delta_l
    """
    R = np.asarray(R_sep, float)
    Z = np.asarray(Z_sep, float)

    Rmax = float(np.nanmax(R))
    Rmin = float(np.nanmin(R))
    Zmax = float(np.nanmax(Z))
    Zmin = float(np.nanmin(Z))

    R0 = 0.5 * (Rmax + Rmin)
    a = 0.5 * (Rmax - Rmin)
    if not np.isfinite(a) or a <= 1e-6:
        return {
            "R0_plasma": float("nan"),
            "a_plasma": float("nan"),
            "A_plasma": float("nan"),
            "kappa_plasma": float("nan"),
            "delta_u": float("nan"),
            "delta_l": float("nan"),
        }

    A = R0 / a
    kappa = (Zmax - Zmin) / (2.0 * a)

    # delta from R at upper/lower extrema (nearest-point method)
    iu = _nearest_index(Z, Zmax)
    il = _nearest_index(Z, Zmin)
    Ru = float(R[iu])
    Rl = float(R[il])
    delta_u = (R0 - Ru) / a
    delta_l = (R0 - Rl) / a

    return {
        "R0_plasma": float(R0),
        "a_plasma": float(a),
        "A_plasma": float(A),
        "kappa_plasma": float(kappa),
        "delta_u": float(delta_u),
        "delta_l": float(delta_l),
    }


# -------------------------
# Fallback: LCFS-limiter (optional)
# -------------------------

def _fallback_lcfs(eq: Any, geom: Dict[str, Any], *, prefer_inner: bool = True, psi_percentile: float = 5.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Any]:
    """
    Try analyze_star_lcfs.shape_from_lcfs_limiter(eq, geom, ...)
    Returns (R_sep, Z_sep, info)
    """
    try:
        from analyze_star_lcfs import shape_from_lcfs_limiter  # type: ignore
    except Exception as e:
        return None, None, f"no_lcfs_module:{repr(e)}"

    try:
        sig = inspect.signature(shape_from_lcfs_limiter)
        kwargs = {"prefer_inner": bool(prefer_inner)}
        if "psi_percentile" in sig.parameters:
            kwargs["psi_percentile"] = float(psi_percentile)
        shp = shape_from_lcfs_limiter(eq, geom, **kwargs)
        # common keys
        R_sep = np.asarray(shp.get("R_sep", []), float) if isinstance(shp, dict) else None
        Z_sep = np.asarray(shp.get("Z_sep", []), float) if isinstance(shp, dict) else None
        if R_sep is None or Z_sep is None or R_sep.size < 10 or Z_sep.size < 10:
            return None, None, f"lcfs_returned_empty:{type(shp)}"
        return R_sep, Z_sep, {"ok": True, "source": "lcfs_limiter"}
    except Exception as e:
        return None, None, f"lcfs_failed:{repr(e)}"


# -------------------------
# Public API
# -------------------------

def shape_from_separatrix(
    eq: Any,
    geom: Dict[str, Any],
    *,
    require_two_x: bool = False,
    null_prefer: str = "lower",       # "lower" | "upper" | "any"
    max_xpoints: int = 10,
    prefer_inner_fallback: bool = True,
    psi_percentile_fallback: float = 5.0,
) -> Dict[str, Any]:
    """
    Extract:
      - xpoints: list[{R,Z,psi}]
      - separatrix boundary: R_sep, Z_sep (if possible)
      - metrics: R0_plasma, a_plasma, A_plasma, kappa_plasma, delta_u, delta_l
      - axis: R_ax, Z_ax, psi_ax
      - psi_sep: chosen separatrix level (from preferred xpoint if available)

    Returns dict with:
      ok_sep (bool), reason (str), xpoints (list), R_sep/Z_sep (list),
      R0_plasma/A_plasma/kappa_plasma/delta_u/delta_l/a_plasma,
      R_ax/Z_ax/psi_ax/psi_sep, fallback_lcfs (info or None)
    """
    out: Dict[str, Any] = {
        "ok_sep": False,
        "reason": "init",
        "xpoints": [],
        "R_sep": [],
        "Z_sep": [],
        "R0_plasma": float("nan"),
        "A_plasma": float("nan"),
        "kappa_plasma": float("nan"),
        "delta_u": float("nan"),
        "delta_l": float("nan"),
        "a_plasma": float("nan"),
        "R_ax": float("nan"),
        "Z_ax": float("nan"),
        "psi_ax": float("nan"),
        "psi_sep": float("nan"),
        "fallback_lcfs": None,
    }

    try:
        R2, Z2, R1, Z1 = _grid_from_eq(eq)
        psi = _psi_from_eq(eq)

        if psi.shape != R2.shape:
            # attempt to coerce
            psi = np.asarray(psi, float)
            if psi.shape != R2.shape:
                raise ValueError(f"psi shape {psi.shape} != grid shape {R2.shape}")

        # sanity: non-finite psi => can't analyze
        if not np.isfinite(psi).all():
            out["reason"] = "non_finite_psi"
            return out

        R_ax, Z_ax, psi_ax = _magnetic_axis(eq, R2, Z2, psi)
        out["R_ax"] = float(R_ax)
        out["Z_ax"] = float(Z_ax)
        out["psi_ax"] = float(psi_ax)

        # Detect xpoints
        xps = _find_xpoints_saddle(
            R2, Z2, R1, Z1, psi,
            max_points=int(max_xpoints),
            cand_count=80,
            grad_q=0.0025,
            min_sep_m=0.18,
        )
        out["xpoints"] = xps

        if require_two_x and len(xps) < 2:
            out["reason"] = "require_two_x_not_met"
            # do not early-return: we can still try boundary extraction
        # Choose psi_sep:
        psi_sep = None
        if xps:
            if str(null_prefer).lower().strip() == "upper":
                xp = max(xps, key=lambda d: float(d["Z"]))
            elif str(null_prefer).lower().strip() == "any":
                xp = xps[0]
            else:
                xp = min(xps, key=lambda d: float(d["Z"]))
            psi_sep = float(xp["psi"])
            out["psi_sep"] = float(psi_sep)

        # If we don't have xpoints, estimate a "wall" flux level as fallback contour target.
        # This is not a true separatrix, but may still yield a closed boundary for metrics.
        if psi_sep is None:
            psi_sep = float("nan")

            if R1 is not None and Z1 is not None:
                # sample on outer wall if present
                Rw = geom.get("R_outer", None)
                Zw = geom.get("Z_outer", None)
                if Rw is not None and Zw is not None:
                    Rw = np.asarray(Rw, float)
                    Zw = np.asarray(Zw, float)
                    try:
                        psw = _bilinear_interp_rect(R1, Z1, psi, Rw, Zw)
                        psw = psw[np.isfinite(psw)]
                        if psw.size > 10:
                            # take median-ish boundary flux
                            psi_sep = float(np.median(psw))
                            out["psi_sep"] = float(psi_sep)
                    except Exception:
                        pass

        # If still NaN, can't contour meaningfully
        if not np.isfinite(psi_sep):
            out["reason"] = "no_psi_sep_level"
            # try LCFS-limiter fallback
            Rf, Zf, info = _fallback_lcfs(eq, geom, prefer_inner=prefer_inner_fallback, psi_percentile=psi_percentile_fallback)
            out["fallback_lcfs"] = info
            if Rf is not None and Zf is not None:
                met = _plasma_metrics_from_sep(Rf, Zf)
                out.update(met)
                out["R_sep"] = [float(x) for x in np.asarray(Rf, float).tolist()]
                out["Z_sep"] = [float(x) for x in np.asarray(Zf, float).tolist()]
                out["ok_sep"] = True
                out["reason"] = "fallback_lcfs_ok"
            return out

        # Extract contour(s) at psi_sep
        segs = _contour_segments(R2, Z2, psi, psi_sep)
        seg = _choose_separatrix_segment(segs, R_ax, Z_ax)
        if seg is None:
            out["reason"] = "no_contour_segments"
            # try LCFS-limiter fallback
            Rf, Zf, info = _fallback_lcfs(eq, geom, prefer_inner=prefer_inner_fallback, psi_percentile=psi_percentile_fallback)
            out["fallback_lcfs"] = info
            if Rf is not None and Zf is not None:
                met = _plasma_metrics_from_sep(Rf, Zf)
                out.update(met)
                out["R_sep"] = [float(x) for x in np.asarray(Rf, float).tolist()]
                out["Z_sep"] = [float(x) for x in np.asarray(Zf, float).tolist()]
                out["ok_sep"] = True
                out["reason"] = "fallback_lcfs_ok"
            return out

        R_sep = np.asarray(seg[:, 0], float)
        Z_sep = np.asarray(seg[:, 1], float)

        # Require "reasonable" segment:
        # If it doesn't contain the axis and isn't closed, we mark ok_sep False but still return geometry
        ok_closed = _is_closed(seg, tol=7e-2)
        ok_contains = False
        if ok_closed:
            try:
                ok_contains = _point_in_poly(R_ax, Z_ax, seg)
            except Exception:
                ok_contains = False

        out["R_sep"] = [float(x) for x in R_sep.tolist()]
        out["Z_sep"] = [float(x) for x in Z_sep.tolist()]

        met = _plasma_metrics_from_sep(R_sep, Z_sep)
        out.update(met)

        if ok_closed and ok_contains and np.isfinite(out["R0_plasma"] + out["A_plasma"] + out["kappa_plasma"] + out["a_plasma"]):
            out["ok_sep"] = True
            out["reason"] = "ok"
        else:
            out["ok_sep"] = False
            out["reason"] = f"weak_sep(closed={ok_closed},contains_axis={ok_contains})"

        # If metrics still NaN, try LCFS fallback as last resort
        if not np.isfinite(out["R0_plasma"] + out["A_plasma"] + out["kappa_plasma"] + out["a_plasma"]):
            Rf, Zf, info = _fallback_lcfs(eq, geom, prefer_inner=prefer_inner_fallback, psi_percentile=psi_percentile_fallback)
            out["fallback_lcfs"] = info
            if Rf is not None and Zf is not None:
                met2 = _plasma_metrics_from_sep(Rf, Zf)
                if np.isfinite(met2["R0_plasma"] + met2["A_plasma"] + met2["kappa_plasma"] + met2["a_plasma"]):
                    out.update(met2)
                    out["R_sep"] = [float(x) for x in np.asarray(Rf, float).tolist()]
                    out["Z_sep"] = [float(x) for x in np.asarray(Zf, float).tolist()]
                    out["ok_sep"] = True
                    out["reason"] = "fallback_lcfs_ok"

        return out

    except Exception as e:
        out["ok_sep"] = False
        out["reason"] = f"exception:{repr(e)}"
        return out

