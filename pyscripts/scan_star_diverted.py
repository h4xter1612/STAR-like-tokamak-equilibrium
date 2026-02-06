"""
scan_star_diverted.py

Robust diverted scan for STAR-like equilibria using FreeGSNKE + CAD/DXF geometry.
Optimizes ONLY coil-family currents (CS, PF1, PF2, PF3) with fixed geometry.

Fixes included:
  - No KeyError when timeouts happen (timeouts include currents_MA/currents_A)
  - FreeGSNKE NoneType copy bug patched via freegsnke.copying.copy_into
  - Matplotlib contour extraction robust (uses QuadContourSet.allsegs, not .collections)
  - Avoids float(method) errors by calling callables defensively
  - CEM updates only on valid (ok_solve && ok_sep) elites

Outputs:
  - results/scan_diverted_results.jsonl
  - results/scan_diverted_best.txt
"""

from __future__ import annotations

import os
import time
import json
import math
import argparse
from pathlib import Path
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from freegsnke import equilibrium_update, GSstaticsolver
from freegsnke.jtor_update import ConstrainPaxisIp

import config_star_bean as cfg


# -------------------------
# Defaults / bounds (MA)
# -------------------------

BOUNDS_DEFAULT_MA = {
    "CS":  (0.2, 2.5),
    "PF1": (-2.0, 0.6),
    "PF2": (-2.0, 1.2),
    "PF3": (-0.5, 2.5),
}

# Penalties
P_BAD_SOLVE = 120.0     # solver failed
P_BAD_SEP   = 105.0     # separatrix/shape extraction failed or invalid
P_NO_X      = 105.0     # no X-points found

# Shape soft penalties
P_THIN      = 5.0
P_NEG_DELTA = 2.0

# X-point target weight (meters -> misfit)
W_XPOINT = 35.0


# -------------------------
# Small helpers
# -------------------------

def _default_dxf() -> str:
    here = Path(__file__).resolve().parent
    return str((here / "cad" / "star_baseline.dxf").resolve())

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if callable(x):
            x = x()
        return float(x)
    except Exception:
        return float(default)

def _bar(frac: float, width: int = 28) -> str:
    frac = max(0.0, min(1.0, float(frac)))
    k = int(round(frac * width))
    return "[" + "#" * k + "-" * (width - k) + "]"

def _status_line(tag: str, done: int, total: int, solve_ok: int, sep_ok: int, best: float, rate: float, last: Optional[float]) -> str:
    frac = 0.0 if total <= 0 else done / total
    last_s = "" if last is None else f" last={last:7.2f}"
    best_s = f"{best:7.2f}" if np.isfinite(best) else "9999.00"
    return f"{tag:>6s} {_bar(frac)} {done:4d}/{total:<4d} solve_ok={solve_ok:4d} sep_ok={sep_ok:4d} best={best_s}{last_s} {rate:5.2f}/s"


# -------------------------
# FreeGSNKE patches (robust)
# -------------------------

def _patch_profiles_copy_once() -> None:
    """
    Ensure ConstrainPaxisIp.copy() doesn't blow up due to missing masks.
    """
    try:
        from freegsnke.jtor_update import ConstrainPaxisIp as _C  # type: ignore
    except Exception:
        return

    if getattr(_C, "_safe_copy_patched", False):
        return

    _orig_copy = _C.copy

    def _safe_copy(self, *args, **kwargs):  # type: ignore
        eq = getattr(self, "eq", None) or getattr(self, "_eq", None)
        if eq is not None:
            try:
                R = getattr(eq, "R", None)
                if R is not None:
                    mask = np.ones(np.asarray(R).shape, dtype=bool)
                else:
                    mask = None
            except Exception:
                mask = None

            if mask is not None:
                for name in ("diverted_core_mask", "limiter_core_mask", "diverted_mask", "limiter_mask", "core_mask"):
                    try:
                        if getattr(self, name, None) is None:
                            setattr(self, name, mask.copy())
                    except Exception:
                        pass

                for attr in dir(self):
                    if attr.endswith("_mask") or attr.endswith("_core_mask"):
                        try:
                            if getattr(self, attr, None) is None:
                                setattr(self, attr, mask.copy())
                        except Exception:
                            pass

        return _orig_copy(self, *args, **kwargs)

    _C.copy = _safe_copy  # type: ignore
    _C._safe_copy_patched = True  # type: ignore


def _patch_copy_into_allow_none_once() -> None:
    """
    Patch freegsnke.copying.copy_into to tolerate None when strict=False
    (fixes: TypeError("Cannot copy <class 'NoneType'> without deepcopying")).
    """
    try:
        import freegsnke.copying as _copying  # type: ignore
    except Exception:
        return

    if getattr(_copying, "_allow_none_patched", False):
        return

    _orig = _copying.copy_into

    def _copy_into_patched(src, dst, name, *args, **kwargs):  # type: ignore
        # Determine strict
        strict = kwargs.get("strict", True)
        if len(args) >= 2:
            strict = args[1]

        try:
            val = getattr(src, name)
        except Exception:
            val = None

        if (not strict) and (val is None):
            try:
                setattr(dst, name, None)
            except Exception:
                pass
            return

        return _orig(src, dst, name, *args, **kwargs)

    _copying.copy_into = _copy_into_patched  # type: ignore
    _copying._allow_none_patched = True  # type: ignore

    # Some versions also expose copy_into via jtor_update
    try:
        import freegsnke.jtor_update as _jtor  # type: ignore
        _jtor.copy_into = _copy_into_patched  # type: ignore
    except Exception:
        pass


def _ensure_profile_masks(profiles: Any, eq: Any, *, force: bool = True) -> None:
    """
    Ensure any *_mask / *_core_mask fields that exist are not None (and shaped).
    """
    try:
        R = getattr(eq, "R", None)
        if R is None:
            return
        mask = np.ones(np.asarray(R).shape, dtype=bool)
    except Exception:
        return

    candidates = (
        "diverted_core_mask",
        "limiter_core_mask",
        "diverted_mask",
        "limiter_mask",
        "core_mask",
        "_core_mask",
        "_edge_mask",
        "_vacuum_mask",
    )

    for name in candidates:
        if hasattr(profiles, name):
            try:
                v = getattr(profiles, name)
                if force or (v is None) or (np.asarray(v).shape != mask.shape):
                    setattr(profiles, name, mask.copy())
            except Exception:
                pass

    for attr in dir(profiles):
        if not (attr.endswith("_mask") or attr.endswith("_core_mask")):
            continue
        try:
            v = getattr(profiles, attr)
        except Exception:
            continue
        if v is None:
            try:
                setattr(profiles, attr, mask.copy())
            except Exception:
                pass


# -------------------------
# Coil currents application
# -------------------------

def apply_star_family_currents(tokamak: Any, CS_MA: float, PF1_MA: float, PF2_MA: float, PF3_MA: float, *, mode: str = "area") -> None:
    """
    Prefer tokamak.apply_group_currents(...) if present; fallback to direct coil labels.
    Currents are in MA; FreeGSNKE coil.current usually expects A (depends on your CAD wrapper),
    BUT your own apply_group_currents has handled this historically. We keep consistent with your setup:
      - If apply_group_currents exists -> pass MA and let it distribute.
      - Else: assign directly to coil.current in MA (your legacy path used MA).
    """
    family = {"CS": float(CS_MA), "PF1": float(PF1_MA), "PF2": float(PF2_MA), "PF3": float(PF3_MA)}

    if hasattr(tokamak, "apply_group_currents") and callable(getattr(tokamak, "apply_group_currents")):
        tokamak.apply_group_currents(family, mode=str(mode))
        return

    # Support both tokamak.coils = [(label, coil), ...] and tokamak.coils = [coil,...] with coil.label
    coils = getattr(tokamak, "coils", [])
    if not coils:
        return

    mapping = {
        "CS": family["CS"],
        "PF1U": family["PF1"], "PF1L": family["PF1"],
        "PF2U": family["PF2"], "PF2L": family["PF2"],
        "PF3U": family["PF3"], "PF3L": family["PF3"],
        "PF1": family["PF1"], "PF2": family["PF2"], "PF3": family["PF3"],
    }

    # tuple-style
    if isinstance(coils[0], (tuple, list)) and len(coils[0]) == 2:
        for label, coil in coils:
            lab = str(label).strip().upper()
            if lab in mapping:
                try:
                    coil.current = float(mapping[lab])
                except Exception:
                    pass
        return

    # object-style
    for coil in coils:
        lab = str(getattr(coil, "label", "")).strip().upper()
        if lab in mapping:
            try:
                coil.current = float(mapping[lab])
            except Exception:
                pass


# -------------------------
# Solver
# -------------------------

def solve_with_continuation(
    tokamak: Any,
    geom: Dict[str, Any],
    *,
    CS_MA: float, PF1_MA: float, PF2_MA: float, PF3_MA: float,
    nx: int, ny: int,
    Ip: float, paxis: float, fvac: float,
    alpha_m: float, alpha_n: float,
    target_rel_tol: float,
    margin_RZ: float,
    f_list: Tuple[float, ...],
    coil_group_mode: str,
    silence_solver: bool = True,
) -> Any:
    R_outer = np.asarray(geom["R_outer"], dtype=float)
    Z_outer = np.asarray(geom["Z_outer"], dtype=float)
    Rmin = float(R_outer.min() - margin_RZ)
    Rmax = float(R_outer.max() + margin_RZ)
    Zmin = float(Z_outer.min() - margin_RZ)
    Zmax = float(Z_outer.max() + margin_RZ)

    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=Rmin, Rmax=Rmax,
        Zmin=Zmin, Zmax=Zmax,
        nx=int(nx), ny=int(ny),
    )
    solver = GSstaticsolver.NKGSsolver(eq)

    ctx = None
    if silence_solver:
        import contextlib
        ctx = contextlib.ExitStack()
        devnull = ctx.enter_context(open(os.devnull, "w"))
        ctx.enter_context(contextlib.redirect_stdout(devnull))
        ctx.enter_context(contextlib.redirect_stderr(devnull))

    try:
        for f in f_list:
            apply_star_family_currents(
                tokamak,
                f * CS_MA, f * PF1_MA, f * PF2_MA, f * PF3_MA,
                mode=str(coil_group_mode),
            )

            profiles = ConstrainPaxisIp(
                eq=eq,
                paxis=float(f * paxis),
                Ip=float(f * Ip),
                fvac=float(fvac),
                alpha_m=float(alpha_m),
                alpha_n=float(alpha_n),
            )
            _ensure_profile_masks(profiles, eq, force=True)

            # Two attempts to tolerate internal copy hiccups
            for attempt in range(2):
                try:
                    solver.solve(
                        eq=eq,
                        profiles=profiles,
                        constrain=None,
                        target_relative_tolerance=float(target_rel_tol),
                        verbose=False,
                    )
                    break
                except TypeError as e:
                    msg = str(e)
                    if ("Cannot copy" in msg and "NoneType" in msg) or ("without deepcopying" in msg):
                        _ensure_profile_masks(profiles, eq, force=True)
                        if attempt == 1:
                            raise
                        continue
                    raise

    finally:
        if ctx is not None:
            ctx.close()

    return eq


# -------------------------
# Robust separatrix + X-point extraction (self-contained)
# -------------------------

def _grid_from_eq(eq: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (R1d, Z1d, psi2d) with consistent shapes.
    """
    # psi
    psi = getattr(eq, "psi", None)
    if psi is None:
        psi = getattr(eq, "Psi", None)
    if psi is None:
        psi = getattr(eq, "plasma_psi", None)
    if psi is None:
        raise RuntimeError("eq has no psi array")

    psi2d = np.asarray(psi, dtype=float)
    if psi2d.ndim != 2:
        raise RuntimeError(f"psi is not 2D (ndim={psi2d.ndim})")

    # coordinates
    R = getattr(eq, "R", None)
    Z = getattr(eq, "Z", None)
    if R is None or Z is None:
        raise RuntimeError("eq missing R/Z grid")

    Rg = np.asarray(R, dtype=float)
    Zg = np.asarray(Z, dtype=float)

    if Rg.ndim == 2 and Zg.ndim == 2:
        # Assume mesh
        R1d = np.asarray(Rg[:, 0], dtype=float)
        Z1d = np.asarray(Zg[0, :], dtype=float)
    elif Rg.ndim == 1 and Zg.ndim == 1:
        R1d, Z1d = Rg, Zg
    else:
        # Try best effort
        if Rg.ndim == 2:
            R1d = np.asarray(Rg[:, 0], dtype=float)
        else:
            R1d = np.asarray(Rg, dtype=float).ravel()
        if Zg.ndim == 2:
            Z1d = np.asarray(Zg[0, :], dtype=float)
        else:
            Z1d = np.asarray(Zg, dtype=float).ravel()

    return R1d, Z1d, psi2d


def _bilinear(R1d: np.ndarray, Z1d: np.ndarray, F: np.ndarray, r: float, z: float) -> float:
    """
    Bilinear interpolation on a rectilinear grid.
    """
    # clamp
    r = float(np.clip(r, R1d.min(), R1d.max()))
    z = float(np.clip(z, Z1d.min(), Z1d.max()))

    i = int(np.searchsorted(R1d, r) - 1)
    j = int(np.searchsorted(Z1d, z) - 1)
    i = max(0, min(i, len(R1d) - 2))
    j = max(0, min(j, len(Z1d) - 2))

    r0, r1 = float(R1d[i]), float(R1d[i + 1])
    z0, z1 = float(Z1d[j]), float(Z1d[j + 1])
    t = 0.0 if (r1 == r0) else (r - r0) / (r1 - r0)
    u = 0.0 if (z1 == z0) else (z - z0) / (z1 - z0)

    f00 = float(F[i, j])
    f10 = float(F[i + 1, j])
    f01 = float(F[i, j + 1])
    f11 = float(F[i + 1, j + 1])

    return (1 - t) * (1 - u) * f00 + t * (1 - u) * f10 + (1 - t) * u * f01 + t * u * f11


def _point_in_poly(x: float, y: float, poly: np.ndarray) -> bool:
    """
    Ray casting point-in-polygon for Nx2 vertices.
    """
    n = poly.shape[0]
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i, 0]), float(poly[i, 1])
        xj, yj = float(poly[j, 0]), float(poly[j, 1])
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-30) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside


def _contours_at_level(R1d: np.ndarray, Z1d: np.ndarray, psi2d: np.ndarray, level: float) -> List[np.ndarray]:
    """
    Matplotlib-version-robust contour extraction.
    Returns list of polylines Nx2 in (R,Z).
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # Build mesh consistent with psi2d indexing:
    # psi2d is assumed shaped (len(R1d), len(Z1d))
    RR, ZZ = np.meshgrid(R1d, Z1d, indexing="ij")

    fig = plt.figure()
    try:
        cs = plt.contour(RR, ZZ, psi2d, levels=[float(level)])
        segs: List[np.ndarray] = []

        # Stable API
        if hasattr(cs, "allsegs") and cs.allsegs and len(cs.allsegs) > 0:
            for seg in cs.allsegs[0]:
                if seg is None:
                    continue
                v = np.asarray(seg, dtype=float)
                if v.ndim == 2 and v.shape[0] >= 10:
                    segs.append(v)
            return segs

        # Fallback
        if hasattr(cs, "get_paths"):
            for p in cs.get_paths():
                v = getattr(p, "vertices", None)
                if v is None:
                    continue
                vv = np.asarray(v, dtype=float)
                if vv.ndim == 2 and vv.shape[0] >= 10:
                    segs.append(vv)

        return segs
    finally:
        plt.close(fig)


def _shape_metrics_from_boundary(R_sep: np.ndarray, Z_sep: np.ndarray) -> Dict[str, float]:
    """
    STAR-like shape metrics from separatrix boundary points.
    """
    R = np.asarray(R_sep, dtype=float)
    Z = np.asarray(Z_sep, dtype=float)

    if R.size < 20 or Z.size < 20:
        return dict(R0_plasma=np.nan, A_plasma=np.nan, kappa_plasma=np.nan, delta_u=np.nan, delta_l=np.nan, a_plasma=np.nan)

    Rmin, Rmax = float(R.min()), float(R.max())
    Zmin, Zmax = float(Z.min()), float(Z.max())

    R0 = 0.5 * (Rmax + Rmin)
    a = 0.5 * (Rmax - Rmin)
    if a <= 1e-6:
        return dict(R0_plasma=np.nan, A_plasma=np.nan, kappa_plasma=np.nan, delta_u=np.nan, delta_l=np.nan, a_plasma=np.nan)

    # Aspect ratio A ~ R0/a
    A = R0 / a

    # Elongation kappa ~ (Zmax-Zmin)/(2a)
    kappa = (Zmax - Zmin) / (2.0 * a)

    # Triangularity: use R at top/bottom extremes
    # Find boundary points near Zmax and Zmin
    def _R_at_Zext(zext: float) -> float:
        idx = np.argmin(np.abs(Z - zext))
        return float(R[idx])

    R_u = _R_at_Zext(Zmax)
    R_l = _R_at_Zext(Zmin)

    delta_u = (R0 - R_u) / a
    delta_l = (R0 - R_l) / a

    return dict(R0_plasma=float(R0), A_plasma=float(A), kappa_plasma=float(kappa),
                delta_u=float(delta_u), delta_l=float(delta_l), a_plasma=float(a))


def _find_saddles(R1d: np.ndarray, Z1d: np.ndarray, psi2d: np.ndarray) -> List[Tuple[int, int]]:
    """
    Crude saddle detection via Hessian determinant < 0 on interior grid.
    """
    dR = np.gradient(R1d)
    dZ = np.gradient(Z1d)

    # First derivatives
    dpsi_dR, dpsi_dZ = np.gradient(psi2d, dR, dZ, edge_order=1)

    # Second derivatives
    d2psi_dR2 = np.gradient(dpsi_dR, dR, axis=0, edge_order=1)
    d2psi_dZ2 = np.gradient(dpsi_dZ, dZ, axis=1, edge_order=1)
    d2psi_dRdZ = np.gradient(dpsi_dR, dZ, axis=1, edge_order=1)

    g2 = dpsi_dR**2 + dpsi_dZ**2
    # Candidate low-gradient points
    thr = np.percentile(g2, 0.05)  # very small gradients
    mask = (g2 <= thr)

    saddles: List[Tuple[int, int]] = []
    ii, jj = np.where(mask)
    for i, j in zip(ii.tolist(), jj.tolist()):
        if i <= 1 or j <= 1 or i >= psi2d.shape[0] - 2 or j >= psi2d.shape[1] - 2:
            continue
        # Hessian determinant
        det = float(d2psi_dR2[i, j] * d2psi_dZ2[i, j] - d2psi_dRdZ[i, j] ** 2)
        if det < 0.0:
            saddles.append((i, j))

    # De-duplicate by coarse binning
    if not saddles:
        return []

    # Keep unique by rounding grid coords
    seen = set()
    uniq: List[Tuple[int, int]] = []
    for i, j in saddles:
        key = (int(i // 2), int(j // 2))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((i, j))
    return uniq


def shape_from_separatrix_robust(eq: Any, geom: Dict[str, Any], *, require_two_x: bool, null_prefer: str) -> Dict[str, Any]:
    """
    Returns a JSON-serializable dict with:
      ok_sep, reason, xpoints, R_sep, Z_sep, metrics...
    """
    out: Dict[str, Any] = {
        "ok_sep": False,
        "reason": "",
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
        R1d, Z1d, psi2d = _grid_from_eq(eq)

        # Magnetic axis
        ma = getattr(eq, "magneticAxis", None)
        if callable(ma):
            ma_val = ma()
        else:
            ma_val = ma
        if ma_val is None:
            raise RuntimeError("eq.magneticAxis unavailable")

        # ma_val often (R,Z,psi) or more
        R_ax = _safe_float(ma_val[0])
        Z_ax = _safe_float(ma_val[1])
        out["R_ax"] = float(R_ax)
        out["Z_ax"] = float(Z_ax)
        try:
            out["psi_ax"] = float(_safe_float(ma_val[2]))
        except Exception:
            out["psi_ax"] = float(_bilinear(R1d, Z1d, psi2d, R_ax, Z_ax))

        psi_ax = float(out["psi_ax"])

        # Find saddle candidates (X-points)
        saddles = _find_saddles(R1d, Z1d, psi2d)
        if not saddles:
            out["reason"] = "no_saddles_found"
            return out

        # Build xpoint list with (R,Z,psi)
        xps = []
        for i, j in saddles:
            r = float(R1d[i])
            z = float(Z1d[j])
            ps = float(psi2d[i, j])
            xps.append((r, z, ps))

        # Filter by prefer: lower/upper/any
        if null_prefer == "lower":
            xps.sort(key=lambda t: t[1])  # Z ascending
        elif null_prefer == "upper":
            xps.sort(key=lambda t: -t[1])
        else:
            # any: sort by proximity to axis (not perfect)
            xps.sort(key=lambda t: (t[0] - R_ax) ** 2 + (t[1] - Z_ax) ** 2)

        # Take top few
        xps = xps[:12]

        out["xpoints"] = [{"R": float(r), "Z": float(z), "psi": float(ps)} for (r, z, ps) in xps]

        if require_two_x and len(xps) < 2:
            out["reason"] = "require_two_x_not_met"
            return out

        # Choose psi_sep from best xpoint
        psi_sep = float(xps[0][2])
        out["psi_sep"] = float(psi_sep)

        # Extract separatrix contour at psi_sep
        segs = _contours_at_level(R1d, Z1d, psi2d, psi_sep)
        if not segs:
            out["reason"] = "no_contours_at_psi_sep"
            return out

        # Choose segment that contains axis; fallback largest length
        axis_point = np.array([R_ax, Z_ax], dtype=float)

        best_seg = None
        best_len = -1.0
        for seg in segs:
            seg = np.asarray(seg, dtype=float)
            if seg.shape[0] < 20:
                continue
            # Ensure closed-ish: allow open but we still compute
            contains = False
            try:
                contains = _point_in_poly(float(axis_point[0]), float(axis_point[1]), seg)
            except Exception:
                contains = False

            # polyline length
            d = np.diff(seg, axis=0)
            L = float(np.sum(np.sqrt(np.sum(d * d, axis=1))))
            score = L + (1e6 if contains else 0.0)

            if score > best_len:
                best_len = score
                best_seg = seg

        if best_seg is None:
            out["reason"] = "no_valid_segment"
            return out

        out["R_sep"] = [float(x) for x in best_seg[:, 0].tolist()]
        out["Z_sep"] = [float(x) for x in best_seg[:, 1].tolist()]

        # Metrics
        m = _shape_metrics_from_boundary(best_seg[:, 0], best_seg[:, 1])
        out.update({k: float(v) for k, v in m.items()})

        # Basic sanity
        if not np.isfinite(out["R0_plasma"] + out["A_plasma"] + out["kappa_plasma"] + out["a_plasma"]):
            out["reason"] = "metrics_nan"
            return out

        out["ok_sep"] = True
        out["reason"] = "ok"
        return out

    except Exception as e:
        out["ok_sep"] = False
        out["reason"] = f"exception:{repr(e)}"
        return out


# -------------------------
# Misfit (diverted)
# -------------------------

def _dist(a: Optional[Tuple[float, float]], b: Tuple[float, float]) -> float:
    if a is None:
        return 0.0
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))

def _choose_xpoint(shape: Dict[str, Any], prefer: str) -> Optional[Tuple[float, float]]:
    xps = shape.get("xpoints", []) or []
    if not xps:
        return None
    pts = [(float(x["R"]), float(x["Z"])) for x in xps]
    if prefer == "any":
        return pts[0]
    if prefer == "lower":
        return min(pts, key=lambda p: p[1])
    if prefer == "upper":
        return max(pts, key=lambda p: p[1])
    return pts[0]

def compute_misfit_diverted(
    shape: Dict[str, Any],
    *,
    targets: Dict[str, float],
    x_target: Optional[Tuple[float, float]],
    require_two_x: bool,
    null_prefer: str,
) -> float:
    # If separatrix extraction failed, penalize
    if not shape or (not bool(shape.get("ok_sep", False))):
        return float(P_BAD_SEP)

    xps = shape.get("xpoints", []) or []
    if not xps:
        return float(P_NO_X)
    if require_two_x and len(xps) < 2:
        return float(P_NO_X + 20.0)

    R0 = float(shape.get("R0_plasma", np.nan))
    A  = float(shape.get("A_plasma", np.nan))
    k  = float(shape.get("kappa_plasma", np.nan))
    du = float(shape.get("delta_u", np.nan))
    dl = float(shape.get("delta_l", np.nan))
    a  = float(shape.get("a_plasma", np.nan))

    if not np.isfinite(R0 + A + k + du + dl + a):
        return float(P_BAD_SEP)

    mis = 0.0
    mis += 4.0 * abs(R0 - targets["R0_target"]) / 0.25
    mis += 5.0 * abs(A  - targets["A_target"])  / 0.20
    mis += 5.0 * abs(k  - targets["kappa_target"]) / 0.25

    # delta soft
    dmean = 0.5 * (du + dl)
    mis += 1.0 * abs(dmean - targets.get("delta_target", 0.2)) / 0.15
    if dmean < 0.0:
        mis += P_NEG_DELTA * abs(dmean)

    # thin plasma penalty
    a_min = float(targets.get("a_min", 0.7))
    if a < a_min:
        mis += P_THIN * (a_min - a) / 0.2

    # X-point target pull
    if x_target is not None:
        xp = _choose_xpoint(shape, null_prefer)
        if xp is None:
            mis += P_NO_X
        else:
            mis += W_XPOINT * _dist(xp, x_target)

    return float(mis)


# -------------------------
# Adaptive sampler (CEM-ish)
# -------------------------

@dataclass
class CEMState:
    mu: np.ndarray    # (4,)
    sigma: np.ndarray # (4,)

def _pack(CS: float, PF1: float, PF2: float, PF3: float) -> np.ndarray:
    return np.array([CS, PF1, PF2, PF3], dtype=float)

def _clip_to_bounds(x: np.ndarray, bounds: Dict[str, Tuple[float, float]]) -> np.ndarray:
    lo = np.array([bounds["CS"][0], bounds["PF1"][0], bounds["PF2"][0], bounds["PF3"][0]], dtype=float)
    hi = np.array([bounds["CS"][1], bounds["PF1"][1], bounds["PF2"][1], bounds["PF3"][1]], dtype=float)
    return np.minimum(np.maximum(x, lo), hi)

def _sample_uniform(rng: np.random.Generator, n: int, bounds: Dict[str, Tuple[float, float]]) -> np.ndarray:
    lo = np.array([bounds["CS"][0], bounds["PF1"][0], bounds["PF2"][0], bounds["PF3"][0]], dtype=float)
    hi = np.array([bounds["CS"][1], bounds["PF1"][1], bounds["PF2"][1], bounds["PF3"][1]], dtype=float)
    return lo + (hi - lo) * rng.random((n, 4))

def _sample_cem(rng: np.random.Generator, state: CEMState, n: int, bounds: Dict[str, Tuple[float, float]], mix_uniform: float) -> np.ndarray:
    n_u = int(round(mix_uniform * n))
    n_g = n - n_u
    X = []
    if n_g > 0:
        Z = rng.standard_normal((n_g, 4))
        Xg = state.mu[None, :] + Z * state.sigma[None, :]
        X.append(_clip_to_bounds(Xg, bounds))
    if n_u > 0:
        Xu = _sample_uniform(rng, n_u, bounds)
        X.append(Xu)
    return np.vstack(X)

def _update_cem(state: CEMState, elite: np.ndarray, *, damp: float = 0.6, min_sigma: float = 0.03) -> CEMState:
    mu_new = elite.mean(axis=0)
    sig_new = elite.std(axis=0)
    sig_new = np.maximum(sig_new, min_sigma)
    mu = damp * state.mu + (1 - damp) * mu_new
    sigma = damp * state.sigma + (1 - damp) * sig_new
    return CEMState(mu=mu, sigma=sigma)


# -------------------------
# Persistent workers (timeout-safe)
# -------------------------

@dataclass
class Worker:
    proc: mp.Process
    in_q: mp.Queue
    busy: bool = False
    case_id: Optional[int] = None
    t_start: float = 0.0


def _worker_main(in_q: mp.Queue, out_q: mp.Queue, init_payload: Dict[str, Any]) -> None:
    # Headless for safety
    import matplotlib
    matplotlib.use("Agg", force=True)

    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered*")

    from star_machine_cad import make_star_machine_from_cad, CADImportOptions, CADLayers

    opts = CADImportOptions(
        unit_scale=float(init_payload["unit_scale"]),
        resample_walls=str(init_payload["resample_walls"]),
        n_wall=int(init_payload["n_wall"]),
        n_inner=int(init_payload["n_inner"]),
        n_plasma=int(init_payload["n_plasma"]),
        min_wall_pts=int(init_payload["min_wall_pts"]),
        enforce_ccw=bool(init_payload["enforce_ccw"]),
        canonical_start=bool(init_payload["canonical_start"]),
        flatten_distance=float(init_payload["flatten_distance"]),
        label_match_factor=float(init_payload["label_match_factor"]),
    )

    tokamak, geom = make_star_machine_from_cad(
        dxf_path=str(init_payload["dxf_path"]),
        layers=CADLayers(),
        opts=opts,
        strict_expected=True,
    )

    _patch_profiles_copy_once()
    _patch_copy_into_allow_none_once()

    while True:
        msg = in_q.get()
        if msg is None:
            break

        case_id, solver_payload, meta = msg
        t0 = time.perf_counter()

        # Always define currents dicts for serialization even on failure
        CS_MA = _safe_float(solver_payload.get("CS_MA"))
        PF1_MA = _safe_float(solver_payload.get("PF1_MA"))
        PF2_MA = _safe_float(solver_payload.get("PF2_MA"))
        PF3_MA = _safe_float(solver_payload.get("PF3_MA"))

        try:
            eq = solve_with_continuation(
                tokamak, geom,
                CS_MA=float(CS_MA),
                PF1_MA=float(PF1_MA),
                PF2_MA=float(PF2_MA),
                PF3_MA=float(PF3_MA),
                nx=int(solver_payload["nx"]),
                ny=int(solver_payload["ny"]),
                Ip=float(solver_payload["Ip"]),
                paxis=float(solver_payload["paxis"]),
                fvac=float(solver_payload["fvac"]),
                alpha_m=float(solver_payload["alpha_m"]),
                alpha_n=float(solver_payload["alpha_n"]),
                target_rel_tol=float(solver_payload["target_rel_tol"]),
                margin_RZ=float(solver_payload["margin_RZ"]),
                f_list=tuple(solver_payload["f_list"]),
                silence_solver=bool(solver_payload["silence_solver"]),
                coil_group_mode=str(solver_payload["coil_group_mode"]),
            )

            shp = shape_from_separatrix_robust(
                eq, geom,
                require_two_x=bool(meta.get("require_two_x", False)),
                null_prefer=str(meta.get("null_prefer", "lower")),
            )

            elapsed = time.perf_counter() - t0

            out_q.put((case_id, {
                "ok_solve": True,
                "tag": str(meta.get("tag", "coarse")),
                "elapsed_s": float(elapsed),
                "currents_MA": {"CS": float(CS_MA), "PF1": float(PF1_MA), "PF2": float(PF2_MA), "PF3": float(PF3_MA)},
                "currents_A":  {"CS": float(CS_MA * 1e6), "PF1": float(PF1_MA * 1e6), "PF2": float(PF2_MA * 1e6), "PF3": float(PF3_MA * 1e6)},
                "shape": shp,
            }))

        except Exception as e:
            elapsed = time.perf_counter() - t0
            out_q.put((case_id, {
                "ok_solve": False,
                "tag": str(meta.get("tag", "coarse")),
                "elapsed_s": float(elapsed),
                "error": repr(e),
                "currents_MA": {"CS": float(CS_MA), "PF1": float(PF1_MA), "PF2": float(PF2_MA), "PF3": float(PF3_MA)},
                "currents_A":  {"CS": float(CS_MA * 1e6), "PF1": float(PF1_MA * 1e6), "PF2": float(PF2_MA * 1e6), "PF3": float(PF3_MA * 1e6)},
                "shape": {
                    "ok_sep": False, "reason": f"solve_failed:{repr(e)}",
                    "xpoints": [], "R_sep": [], "Z_sep": [], "fallback_lcfs": None,
                    "R0_plasma": float("nan"), "A_plasma": float("nan"), "kappa_plasma": float("nan"),
                    "delta_u": float("nan"), "delta_l": float("nan"), "a_plasma": float("nan"),
                    "R_ax": float("nan"), "Z_ax": float("nan"), "psi_ax": float("nan"), "psi_sep": float("nan"),
                },
            }))


def _start_worker(ctx: mp.context.BaseContext, init_payload: Dict[str, Any], out_q: mp.Queue) -> Worker:
    in_q: mp.Queue = ctx.Queue(maxsize=2)
    p = ctx.Process(target=_worker_main, args=(in_q, out_q, init_payload), daemon=True)
    p.start()
    return Worker(proc=p, in_q=in_q)

def _kill_worker(w: Worker) -> None:
    try:
        if w.proc.is_alive():
            w.proc.terminate()
    except Exception:
        pass
    try:
        w.proc.join(timeout=1.0)
    except Exception:
        pass


def _dispatch_and_collect(
    ctx: mp.context.BaseContext,
    workers: List[Worker],
    out_q: mp.Queue,
    init_payload: Dict[str, Any],
    tasks: List[Tuple[int, Dict[str, Any], Dict[str, Any]]],
    *,
    timeout_s: float,
    tag: str,
    jsonl_path: Path,
    best_path: Path,
    targets: Dict[str, float],
    x_target: Optional[Tuple[float, float]],
    require_two_x: bool,
    null_prefer: str,
) -> List[Dict[str, Any]]:
    """
    Run tasks with persistent workers, enforce timeout, write JSONL as results arrive,
    and print dynamic status line.
    """
    t_global0 = time.time()
    done = 0
    solve_ok = 0
    sep_ok = 0
    best = float("inf")
    last_mis: Optional[float] = None

    # map case_id -> worker index
    owner: Dict[int, int] = {}

    # payload cache to build timeout records safely
    task_payload: Dict[int, Dict[str, Any]] = {int(cid): payload for (cid, payload, _meta) in tasks}

    pending = tasks.copy()
    results: List[Dict[str, Any]] = []

    f = open(jsonl_path, "a", encoding="utf-8")

    try:
        while pending or any(w.busy for w in workers):
            # Assign
            for wi, w in enumerate(workers):
                if not pending:
                    break
                if not w.busy:
                    case_id, solver_payload, meta = pending.pop()
                    owner[int(case_id)] = wi
                    w.busy = True
                    w.case_id = int(case_id)
                    w.t_start = time.time()
                    w.in_q.put((int(case_id), solver_payload, meta))

            # Collect
            got_one = False
            rec = None
            got_case_id = None
            try:
                got_case_id, rec = out_q.get(timeout=0.15)
                got_one = True
            except Exception:
                pass

            # Handle timeouts
            now = time.time()
            for wi, w in enumerate(workers):
                if w.busy and (now - w.t_start) > timeout_s:
                    cid = int(w.case_id) if w.case_id is not None else -1

                    # kill & restart
                    _kill_worker(w)
                    workers[wi] = _start_worker(ctx, init_payload=init_payload, out_q=out_q)

                    pl = task_payload.get(cid, {})
                    CS_MA = _safe_float(pl.get("CS_MA"))
                    PF1_MA = _safe_float(pl.get("PF1_MA"))
                    PF2_MA = _safe_float(pl.get("PF2_MA"))
                    PF3_MA = _safe_float(pl.get("PF3_MA"))

                    trec = {
                        "ok_solve": False,
                        "tag": tag,
                        "elapsed_s": float(timeout_s),
                        "error": f"TimeoutError({timeout_s}s)",
                        "case_id": int(cid),
                        "currents_MA": {"CS": float(CS_MA), "PF1": float(PF1_MA), "PF2": float(PF2_MA), "PF3": float(PF3_MA)},
                        "currents_A":  {"CS": float(CS_MA * 1e6), "PF1": float(PF1_MA * 1e6), "PF2": float(PF2_MA * 1e6), "PF3": float(PF3_MA * 1e6)},
                        "shape": {
                            "ok_sep": False,
                            "reason": "timeout",
                            "xpoints": [],
                            "R_sep": [],
                            "Z_sep": [],
                            "fallback_lcfs": None,
                            "R0_plasma": float("nan"), "A_plasma": float("nan"), "kappa_plasma": float("nan"),
                            "delta_u": float("nan"), "delta_l": float("nan"), "a_plasma": float("nan"),
                            "R_ax": float("nan"), "Z_ax": float("nan"), "psi_ax": float("nan"), "psi_sep": float("nan"),
                        },
                    }

                    # misfit
                    trec["misfit"] = float(P_BAD_SEP)
                    last_mis = float(trec["misfit"])

                    f.write(json.dumps(trec) + "\n")
                    f.flush()

                    done += 1
                    results.append(trec)

                    # mark worker slot cleared
                    if cid in owner:
                        owner.pop(cid, None)
                    continue

            if got_one and rec is not None:
                cid = int(got_case_id)

                wi = owner.pop(cid, None)
                if wi is not None and 0 <= wi < len(workers):
                    ww = workers[wi]
                    ww.busy = False
                    ww.case_id = None

                rec["case_id"] = cid

                # Compute misfit in MAIN
                if bool(rec.get("ok_solve", False)):
                    solve_ok += 1
                    shp = rec.get("shape") or {}
                    if bool(shp.get("ok_sep", False)):
                        sep_ok += 1
                    mis = compute_misfit_diverted(
                        shp,
                        targets=targets,
                        x_target=x_target,
                        require_two_x=require_two_x,
                        null_prefer=null_prefer,
                    )
                else:
                    mis = float(P_BAD_SOLVE)

                rec["misfit"] = float(mis)

                done += 1
                last_mis = float(mis)

                if float(mis) < best:
                    best = float(mis)
                    with open(best_path, "w", encoding="utf-8") as bf:
                        bf.write(json.dumps(rec, indent=2) + "\n")

                f.write(json.dumps(rec) + "\n")
                f.flush()

                results.append(rec)

            # Status line
            dt = max(1e-6, time.time() - t_global0)
            rate = done / dt
            print("\r" + _status_line(tag, done, len(tasks), solve_ok, sep_ok, best, rate, last_mis), end="", flush=True)

        print()
        return results

    finally:
        f.close()


# -------------------------
# X-target inference (optional)
# -------------------------

def infer_x_target_from_geom(geom: Dict[str, Any], *, prefer: str = "lower") -> Optional[Tuple[float, float]]:
    Rin = geom.get("R_inner", None)
    Zin = geom.get("Z_inner", None)
    if Rin is None or Zin is None:
        return None
    R = np.asarray(Rin, dtype=float)
    Z = np.asarray(Zin, dtype=float)
    if R.size < 10:
        return None
    if prefer == "lower":
        k = int(np.argmin(Z))
        return (float(R[k] + 0.10), float(Z[k] - 0.05))
    if prefer == "upper":
        k = int(np.argmax(Z))
        return (float(R[k] + 0.10), float(Z[k] + 0.05))
    k = int(np.argmin(np.abs(Z)))
    return (float(R[k] + 0.10), float(Z[k]))


# -------------------------
# Main
# -------------------------

def main() -> None:
    mp.freeze_support()

    ap = argparse.ArgumentParser()
    ap.add_argument("--dxf", type=str, default=_default_dxf())
    ap.add_argument("--outdir", type=str, default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=120.0)

    ap.add_argument("--n1", type=int, default=200)
    ap.add_argument("--n2", type=int, default=300)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--elite_frac", type=float, default=0.12)

    ap.add_argument("--infer_x", action="store_true")
    ap.add_argument("--x_target", type=str, default="")
    ap.add_argument("--null", type=str, default="lower", choices=["lower","upper","any"])
    ap.add_argument("--require_two_x", action="store_true")

    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    # Targets (STAR-like)
    targets = {
        "R0_target": float(getattr(cfg, "R0_target", 4.0)),
        "A_target": float(getattr(cfg, "A_target", 2.0)),
        "kappa_target": float(getattr(cfg, "kappa_target", 2.5)),
        "delta_target": float(getattr(cfg, "delta_target", 0.2)),
        "a_min": float(getattr(cfg, "a_min", 0.7)),
    }

    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    jsonl_path = outdir / "scan_diverted_results.jsonl"
    best_path  = outdir / "scan_diverted_best.txt"
    if (not args.append) and jsonl_path.exists():
        jsonl_path.unlink(missing_ok=True)

    rng = np.random.default_rng(int(args.seed))

    # Build CAD once in main too (for infer_x)
    from star_machine_cad import make_star_machine_from_cad, CADImportOptions, CADLayers

    init_payload: Dict[str, Any] = {
        "dxf_path": args.dxf,
        "unit_scale": float(getattr(cfg, "unit_scale", 1.0)),
        "resample_walls": str(getattr(cfg, "resample_walls", "auto")),
        "n_wall": int(getattr(cfg, "n_wall", 801)),
        "n_inner": int(getattr(cfg, "n_inner", 801)),
        "n_plasma": int(getattr(cfg, "n_plasma", 801)),
        "min_wall_pts": int(getattr(cfg, "min_wall_pts", 200)),
        "enforce_ccw": bool(getattr(cfg, "enforce_ccw", True)),
        "canonical_start": bool(getattr(cfg, "canonical_start", True)),
        "flatten_distance": float(getattr(cfg, "flatten_distance", 0.0)),
        "label_match_factor": float(getattr(cfg, "label_match_factor", 0.92)),
    }

    opts = CADImportOptions(
        unit_scale=float(init_payload["unit_scale"]),
        resample_walls=str(init_payload["resample_walls"]),
        n_wall=int(init_payload["n_wall"]),
        n_inner=int(init_payload["n_inner"]),
        n_plasma=int(init_payload["n_plasma"]),
        min_wall_pts=int(init_payload["min_wall_pts"]),
        enforce_ccw=bool(init_payload["enforce_ccw"]),
        canonical_start=bool(init_payload["canonical_start"]),
        flatten_distance=float(init_payload["flatten_distance"]),
        label_match_factor=float(init_payload["label_match_factor"]),
    )
    _, geom_main = make_star_machine_from_cad(
        dxf_path=args.dxf,
        layers=CADLayers(),
        opts=opts,
        strict_expected=True,
    )

    # X target selection
    x_target: Optional[Tuple[float, float]] = None
    if args.x_target.strip():
        parts = args.x_target.replace(" ", "").split(",")
        if len(parts) == 2:
            x_target = (float(parts[0]), float(parts[1]))
    elif args.infer_x:
        x_target = infer_x_target_from_geom(geom_main, prefer=args.null)

    # Bounds (MA)
    bounds = dict(BOUNDS_DEFAULT_MA)

    # Initial CEM state centered mid-bounds
    mid = np.array([(bounds[k][0] + bounds[k][1]) * 0.5 for k in ("CS","PF1","PF2","PF3")], dtype=float)
    sig = np.array([(bounds[k][1] - bounds[k][0]) * 0.35 for k in ("CS","PF1","PF2","PF3")], dtype=float)
    state = CEMState(mu=mid, sigma=sig)

    # Solver constants from cfg
    base_solver = {
        "Ip": float(getattr(cfg, "Ip", 14.5e6)),
        "paxis": float(getattr(cfg, "paxis", 1.0e6)),
        "fvac": float(getattr(cfg, "fvac", 0.0)),
        "alpha_m": float(getattr(cfg, "alpha_m", 2.0)),
        "alpha_n": float(getattr(cfg, "alpha_n", 2.0)),
        "margin_RZ": float(getattr(cfg, "margin_RZ", 0.35)),
        "coil_group_mode": str(getattr(cfg, "coil_group_mode", "area")),
        "silence_solver": bool(getattr(cfg, "silence_solver", True)),
    }

    f_list_coarse = tuple(getattr(cfg, "f_list_coarse", (0.2, 0.55, 1.0)))
    f_list_refine = tuple(getattr(cfg, "f_list_refine", (0.15, 0.35, 0.65, 1.0)))

    # Grid params
    nx1 = int(getattr(cfg, "nx_coarse", 97))
    ny1 = int(getattr(cfg, "ny_coarse", 193))
    nx2 = int(getattr(cfg, "nx_refine", 129))
    ny2 = int(getattr(cfg, "ny_refine", 257))

    tol1 = float(getattr(cfg, "target_rel_tol_coarse", 3e-3))
    tol2 = float(getattr(cfg, "target_rel_tol_refine", 1e-3))

    # Start workers
    ctx = mp.get_context("spawn")
    out_q: mp.Queue = ctx.Queue()
    workers: List[Worker] = []
    for _ in range(max(1, int(args.workers))):
        workers.append(_start_worker(ctx, init_payload, out_q))

    def build_tasks(X: np.ndarray, *, tag: str, nx: int, ny: int, tol: float, f_list: Tuple[float, ...], start_id: int) -> List[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        tasks: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
        for i in range(X.shape[0]):
            CS_MA, PF1_MA, PF2_MA, PF3_MA = map(float, X[i, :])

            solver_payload = {
                "CS_MA": CS_MA, "PF1_MA": PF1_MA, "PF2_MA": PF2_MA, "PF3_MA": PF3_MA,
                "nx": int(nx), "ny": int(ny),
                "Ip": base_solver["Ip"],
                "paxis": base_solver["paxis"],
                "fvac": base_solver["fvac"],
                "alpha_m": base_solver["alpha_m"],
                "alpha_n": base_solver["alpha_n"],
                "target_rel_tol": float(tol),
                "margin_RZ": base_solver["margin_RZ"],
                "f_list": tuple(float(x) for x in f_list),
                "silence_solver": base_solver["silence_solver"],
                "coil_group_mode": base_solver["coil_group_mode"],
            }

            meta = {
                "tag": tag,
                "require_two_x": bool(args.require_two_x),
                "null_prefer": str(args.null),
            }

            tasks.append((start_id + i, solver_payload, meta))
        return tasks

    try:
        print("[INFO] scan_star_diverted started")
        print(f"[INFO] dxf={args.dxf}")
        print(f"[INFO] workers={args.workers} n1={args.n1} n2={args.n2} pop={args.pop} timeout={args.timeout}")
        print(f"[INFO] targets: R0={targets['R0_target']} A={targets['A_target']} kappa={targets['kappa_target']} delta={targets['delta_target']}")
        if x_target is not None:
            print(f"[INFO] x_target=({x_target[0]:.3f},{x_target[1]:.3f}) null_prefer={args.null} require_two_x={args.require_two_x}")
        else:
            print(f"[INFO] x_target=None null_prefer={args.null} require_two_x={args.require_two_x}")

        # -------- Stage 1 (coarse, dynamic) --------
        case_id0 = 0
        remaining = int(args.n1)
        all1: List[Dict[str, Any]] = []

        while remaining > 0:
            n = min(int(args.pop), remaining)
            X = _sample_cem(rng, state, n, bounds, mix_uniform=0.55)

            tasks = build_tasks(X, tag="coarse", nx=nx1, ny=ny1, tol=tol1, f_list=f_list_coarse, start_id=case_id0)
            case_id0 += len(tasks)
            remaining -= len(tasks)

            res = _dispatch_and_collect(
                ctx, workers, out_q, init_payload, tasks,
                timeout_s=float(args.timeout),
                tag="coarse",
                jsonl_path=jsonl_path,
                best_path=best_path,
                targets=targets,
                x_target=x_target,
                require_two_x=bool(args.require_two_x),
                null_prefer=str(args.null),
            )
            all1.extend(res)

            # Update CEM using elite of VALID results only
            ok_res = [
                r for r in res
                if r.get("ok_solve", False)
                and isinstance(r.get("shape"), dict)
                and r["shape"].get("ok_sep", False)
                and np.isfinite(r.get("misfit", np.inf))
                and isinstance(r.get("currents_MA"), dict)
            ]

            if len(ok_res) >= max(6, int(0.15 * len(res))):
                ok_res.sort(key=lambda r: float(r["misfit"]))
                elite_n = max(4, int(math.ceil(float(args.elite_frac) * len(ok_res))))
                elite = ok_res[:elite_n]
                E = np.vstack([
                    np.array([e["currents_MA"]["CS"], e["currents_MA"]["PF1"], e["currents_MA"]["PF2"], e["currents_MA"]["PF3"]], dtype=float)
                    for e in elite
                ])
                state = _update_cem(state, E, damp=0.55)
            else:
                # widen slightly if too many fails
                state.sigma = np.minimum(state.sigma * 1.10, np.array([0.9, 1.3, 1.3, 1.3], dtype=float))

        # Seed refine around best from stage 1 (if any)
        ok1 = [r for r in all1 if r.get("ok_solve", False) and isinstance(r.get("shape"), dict) and r["shape"].get("ok_sep", False)]
        if ok1:
            ok1.sort(key=lambda r: float(r["misfit"]))
            best1 = ok1[0]
            cm = best1["currents_MA"]
            state.mu = _pack(float(cm["CS"]), float(cm["PF1"]), float(cm["PF2"]), float(cm["PF3"]))
            state.sigma = np.maximum(state.sigma * 0.55, 0.04)

        # -------- Stage 2 (refine, dynamic) --------
        remaining = int(args.n2)
        while remaining > 0:
            n = min(int(args.pop), remaining)
            X = _sample_cem(rng, state, n, bounds, mix_uniform=0.20)

            tasks = build_tasks(X, tag="refine", nx=nx2, ny=ny2, tol=tol2, f_list=f_list_refine, start_id=case_id0)
            case_id0 += len(tasks)
            remaining -= len(tasks)

            res = _dispatch_and_collect(
                ctx, workers, out_q, init_payload, tasks,
                timeout_s=float(args.timeout),
                tag="refine",
                jsonl_path=jsonl_path,
                best_path=best_path,
                targets=targets,
                x_target=x_target,
                require_two_x=bool(args.require_two_x),
                null_prefer=str(args.null),
            )

            ok_res = [
                r for r in res
                if r.get("ok_solve", False)
                and isinstance(r.get("shape"), dict)
                and r["shape"].get("ok_sep", False)
                and np.isfinite(r.get("misfit", np.inf))
                and isinstance(r.get("currents_MA"), dict)
            ]

            if len(ok_res) >= max(6, int(0.15 * len(res))):
                ok_res.sort(key=lambda r: float(r["misfit"]))
                elite_n = max(4, int(math.ceil(float(args.elite_frac) * len(ok_res))))
                elite = ok_res[:elite_n]
                E = np.vstack([
                    np.array([e["currents_MA"]["CS"], e["currents_MA"]["PF1"], e["currents_MA"]["PF2"], e["currents_MA"]["PF3"]], dtype=float)
                    for e in elite
                ])
                state = _update_cem(state, E, damp=0.50)
            else:
                state.sigma = np.minimum(state.sigma * 1.12, np.array([0.9, 1.3, 1.3, 1.3], dtype=float))

        print("[DONE] Finished scan. Best written to:", str(best_path))

    finally:
        # stop workers
        for w in workers:
            try:
                w.in_q.put(None)
            except Exception:
                pass
        for w in workers:
            _kill_worker(w)


if __name__ == "__main__":
    main()

