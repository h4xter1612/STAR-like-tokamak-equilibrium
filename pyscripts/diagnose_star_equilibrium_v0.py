#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_star_equilibrium.py

Paper-style diagnostics for the STAR-like free-boundary Grad-Shafranov equilibrium.

Designed for the current CAD-based pipeline:
  - star_machine_cad.py builds active coils + STAR_VESSEL passive structures + limiter/wall.
  - star_equilibrium.py builds the solved FreeGSNKE equilibrium and returns (eq, tokamak, geom, shape).
  - analyze_star.py supplies the reconstructed LCFS/separatrix in shape["R_sep"], shape["Z_sep"].

Outputs, by default, are written to ../results/diagnostics_<timestamp>/:
  - summary.json
  - summary_paper.md
  - geometry_table.csv
  - coil_currents_table.csv
  - profiles_q_pressure.csv
  - midplane_fields.csv
  - figures/*.png

Usage examples:
  py .\diagnose_star_equilibrium.py --dxf .\star_baseline.dxf
  py .\diagnose_star_equilibrium.py --dxf .\star_baseline.dxf --outdir .\results\diagnostics_baseline --plot-passives

Notes:
  - Br and Bz are computed from psi by the axisymmetric relations:
        B_R = -(1/R) dpsi/dZ,   B_Z = (1/R) dpsi/dR.
  - B_tor is approximated as fvac/R unless a usable f(psi) method is found.
  - Passive-structure resistivities/materials are reported as CAD metadata. In static equilibrium,
    zero-current passives usually do not alter the field unless a dynamic/passive-current solve is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Rectangle, Polygon as MplPolygon

MU0 = 4.0e-7 * math.pi


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_results_dir() -> Path:
    p = _script_dir().parent / "results" / f"diagnostics_{_now_tag()}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _as_array(x: Any) -> np.ndarray:
    if callable(x):
        x = x()
    return np.asarray(x, dtype=float)


def _get_eq_arrays(eq: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = _as_array(getattr(eq, "R"))
    Z = _as_array(getattr(eq, "Z"))
    psi_attr = getattr(eq, "psi")
    psi = _as_array(psi_attr)
    return R, Z, psi


def _get_axis(shape: Dict[str, Any], eq: Any, R: np.ndarray, Z: np.ndarray, psi: np.ndarray) -> Tuple[float, float]:
    diag = shape.get("plasma_diag", {}) or {}
    for kr, kz in (("R_ax", "Z_ax"), ("R_axis", "Z_axis"), ("Raxis", "Zaxis")):
        if kr in shape and kz in shape:
            return float(shape[kr]), float(shape[kz])
        if kr in diag and kz in diag:
            return float(diag[kr]), float(diag[kz])

    # Fallback: use O-point if present.
    for name in ("o_points", "opoints", "O_points"):
        pts = shape.get(name, None)
        if pts:
            try:
                return float(pts[0][0]), float(pts[0][1])
            except Exception:
                pass

    # Last fallback: minimum psi point.
    idx = np.unravel_index(np.nanargmin(psi), psi.shape)
    return float(R[idx]), float(Z[idx])


def _get_lcfs(shape: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    for rk, zk in (("R_sep", "Z_sep"), ("R_lcfs", "Z_lcfs"), ("R_boundary", "Z_boundary")):
        if rk in shape and zk in shape:
            R = np.asarray(shape[rk], dtype=float).ravel()
            Z = np.asarray(shape[zk], dtype=float).ravel()
            good = np.isfinite(R) & np.isfinite(Z)
            R = R[good]
            Z = Z[good]
            if len(R) >= 8:
                return R, Z
    raise RuntimeError("No usable LCFS/separatrix curve found in shape dict. Expected R_sep/Z_sep.")


def _polygon_area(R: np.ndarray, Z: np.ndarray) -> float:
    if len(R) < 3:
        return float("nan")
    return 0.5 * abs(float(np.dot(R, np.roll(Z, -1)) - np.dot(Z, np.roll(R, -1))))


def _polygon_perimeter(R: np.ndarray, Z: np.ndarray) -> float:
    if len(R) < 2:
        return float("nan")
    dR = np.diff(np.r_[R, R[0]])
    dZ = np.diff(np.r_[Z, Z[0]])
    return float(np.sum(np.hypot(dR, dZ)))


def _inside_polygon_mask(Rgrid: np.ndarray, Zgrid: np.ndarray, Rpoly: np.ndarray, Zpoly: np.ndarray) -> np.ndarray:
    poly = np.vstack([Rpoly, Zpoly]).T
    path = MplPath(poly, closed=True)
    points = np.vstack([Rgrid.ravel(), Zgrid.ravel()]).T
    return path.contains_points(points).reshape(Rgrid.shape)


def _interp_at_RZ(grid: np.ndarray, Rgrid: np.ndarray, Zgrid: np.ndarray, R0: float, Z0: float) -> float:
    # Nearest-neighbour fallback. Good enough for axis psi normalization diagnostics.
    d2 = (Rgrid - R0) ** 2 + (Zgrid - Z0) ** 2
    idx = np.unravel_index(np.nanargmin(d2), d2.shape)
    return float(grid[idx])


def _psi_boundary(eq: Any, R_sep: np.ndarray, Z_sep: np.ndarray, Rgrid: np.ndarray, Zgrid: np.ndarray, psi: np.ndarray) -> float:
    for attr in ("psi_bndry", "psi_boundary", "psibndry"):
        if hasattr(eq, attr):
            try:
                return float(getattr(eq, attr))
            except Exception:
                pass
    # fallback: median psi sampled at closest grid points along LCFS
    vals = []
    stride = max(1, len(R_sep) // 200)
    for r, z in zip(R_sep[::stride], Z_sep[::stride]):
        vals.append(_interp_at_RZ(psi, Rgrid, Zgrid, float(r), float(z)))
    return float(np.nanmedian(vals))


def _midplane_crossings(R_sep: np.ndarray, Z_sep: np.ndarray, z0: float) -> List[float]:
    Rs: List[float] = []
    R = np.asarray(R_sep, float)
    Z = np.asarray(Z_sep, float) - float(z0)
    n = len(R)
    for i in range(n):
        j = (i + 1) % n
        z1, z2 = Z[i], Z[j]
        r1, r2 = R[i], R[j]
        if not (np.isfinite(z1) and np.isfinite(z2) and np.isfinite(r1) and np.isfinite(r2)):
            continue
        if z1 == 0.0:
            Rs.append(float(r1))
        if z1 * z2 < 0.0:
            t = -z1 / (z2 - z1)
            Rs.append(float(r1 + t * (r2 - r1)))
    # Deduplicate close crossings.
    out = []
    for r in sorted(Rs):
        if not out or abs(r - out[-1]) > 1e-3:
            out.append(r)
    return out


def _point_segment_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    c2 = vx * vx + vy * vy
    if c2 <= 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / c2))
    qx, qy = ax + t * vx, ay + t * vy
    return math.hypot(px - qx, py - qy)


def _min_distance_polyline(points_R: np.ndarray, points_Z: np.ndarray, wall_R: np.ndarray, wall_Z: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    best = float("inf")
    best_pt = (float("nan"), float("nan"))
    WR = np.asarray(wall_R, float).ravel()
    WZ = np.asarray(wall_Z, float).ravel()
    if len(WR) < 2:
        return best, best_pt
    for pr, pz in zip(points_R, points_Z):
        for i in range(len(WR) - 1):
            d = _point_segment_dist(float(pr), float(pz), float(WR[i]), float(WZ[i]), float(WR[i+1]), float(WZ[i+1]))
            if d < best:
                best = d
                best_pt = (float(pr), float(pz))
    return float(best), best_pt


def _contains_all(poly_R: np.ndarray, poly_Z: np.ndarray, pts_R: np.ndarray, pts_Z: np.ndarray, radius_tol: float = 0.0) -> bool:
    # For a conservative check, test points. Positive tolerance is not geometrically expanded here;
    # it is reported separately via min distance. This keeps the logic dependency-free.
    path = MplPath(np.vstack([poly_R, poly_Z]).T, closed=True)
    pts = np.vstack([pts_R, pts_Z]).T
    return bool(np.all(path.contains_points(pts, radius=radius_tol)))


def _compute_geometry(shape: Dict[str, Any], eq: Any, geom: Dict[str, Any]) -> Dict[str, Any]:
    Rgrid, Zgrid, psi = _get_eq_arrays(eq)
    R_sep, Z_sep = _get_lcfs(shape)
    R_ax, Z_ax = _get_axis(shape, eq, Rgrid, Zgrid, psi)

    Rmin, Rmax = float(np.nanmin(R_sep)), float(np.nanmax(R_sep))
    Zmin, Zmax = float(np.nanmin(Z_sep)), float(np.nanmax(Z_sep))
    R0_mid = 0.5 * (Rmin + Rmax)
    a_mid = 0.5 * (Rmax - Rmin)
    A_mid = R0_mid / a_mid if a_mid > 0 else float("nan")
    kappa = 0.5 * (Zmax - Zmin) / a_mid if a_mid > 0 else float("nan")

    # Use top/bottom extremal points for triangularity.
    itop = int(np.nanargmax(Z_sep))
    ibot = int(np.nanargmin(Z_sep))
    R_top = float(R_sep[itop])
    R_bot = float(R_sep[ibot])
    delta_u = (R0_mid - R_top) / a_mid if a_mid > 0 else float("nan")
    delta_l = (R0_mid - R_bot) / a_mid if a_mid > 0 else float("nan")
    delta_bar = 0.5 * (delta_u + delta_l)

    area = _polygon_area(R_sep, Z_sep)
    perim = _polygon_perimeter(R_sep, Z_sep)
    volume_shell = 2.0 * math.pi * R0_mid * area if np.isfinite(area) else float("nan")

    # Prefer plasma_diag values if present because that is what the equilibrium pipeline reports.
    diag = shape.get("plasma_diag", {}) or {}
    out = {
        "R_axis_m": R_ax,
        "Z_axis_m": Z_ax,
        "R0_m": float(diag.get("R0", diag.get("R0_plasma", R0_mid))),
        "a_m": float(diag.get("a", diag.get("a_plasma", a_mid))),
        "A": float(diag.get("A", diag.get("A_plasma", A_mid))),
        "kappa": float(diag.get("kappa", diag.get("kappa_plasma", kappa))),
        "delta_u": float(diag.get("delta_u", delta_u)),
        "delta_l": float(diag.get("delta_l", delta_l)),
        "delta_bar": float(diag.get("delta_bar", delta_bar)),
        "Rmin_m": Rmin,
        "Rmax_m": Rmax,
        "Zmin_m": Zmin,
        "Zmax_m": Zmax,
        "area_m2": float(diag.get("area", area)),
        "perimeter_m": perim,
        "volume_toroidal_m3": volume_shell,
        "R_top_m": R_top,
        "R_bottom_m": R_bot,
        "source_method": str(diag.get("method", shape.get("source", shape.get("reason", "unknown")))),
    }
    return out


def _compute_containment(geom: Dict[str, Any], shape: Dict[str, Any]) -> Dict[str, Any]:
    R_sep, Z_sep = _get_lcfs(shape)
    out: Dict[str, Any] = {}
    if "R_inner" in geom and "Z_inner" in geom:
        Rw = np.asarray(geom["R_inner"], float).ravel()
        Zw = np.asarray(geom["Z_inner"], float).ravel()
        inside = _contains_all(Rw, Zw, R_sep, Z_sep)
        dmin, pt = _min_distance_polyline(R_sep, Z_sep, Rw, Zw)
        out.update({
            "inside_WALL_INNER": inside,
            "min_abs_distance_to_WALL_INNER_m": dmin,
            "closest_lcfs_point_R_m": pt[0],
            "closest_lcfs_point_Z_m": pt[1],
        })
    if "R_outer" in geom and "Z_outer" in geom:
        Rw = np.asarray(geom["R_outer"], float).ravel()
        Zw = np.asarray(geom["Z_outer"], float).ravel()
        out["inside_WALL_OUTER"] = _contains_all(Rw, Zw, R_sep, Z_sep)
        dmin, _ = _min_distance_polyline(R_sep, Z_sep, Rw, Zw)
        out["min_abs_distance_to_WALL_OUTER_m"] = dmin
    return out


def _grid_spacing(R: np.ndarray, Z: np.ndarray) -> Tuple[float, float]:
    # FreeGSNKE commonly stores R varying along axis 0 and Z along axis 1.
    dR = float(np.nanmedian(np.abs(np.diff(R[:, 0])))) if R.ndim == 2 and R.shape[0] > 1 else 1.0
    dZ = float(np.nanmedian(np.abs(np.diff(Z[0, :])))) if Z.ndim == 2 and Z.shape[1] > 1 else 1.0
    if not np.isfinite(dR) or dR <= 0:
        dR = 1.0
    if not np.isfinite(dZ) or dZ <= 0:
        dZ = 1.0
    return dR, dZ


def _compute_fields(eq: Any, shape: Dict[str, Any], cfg: Any) -> Dict[str, Any]:
    R, Z, psi = _get_eq_arrays(eq)
    dR, dZ = _grid_spacing(R, Z)
    dpsi_dR, dpsi_dZ = np.gradient(psi, dR, dZ, edge_order=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        Br = -dpsi_dZ / R
        Bz = dpsi_dR / R
        Bp = np.hypot(Br, Bz)

    fvac = float(getattr(cfg, "fvac", 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        Bt = fvac / R
        Btot = np.sqrt(Bp * Bp + Bt * Bt)

    return {"R": R, "Z": Z, "psi": psi, "Br": Br, "Bz": Bz, "Bp": Bp, "Bt": Bt, "Btot": Btot, "dR": dR, "dZ": dZ}


def _sample_midplane(fields: Dict[str, Any], shape: Dict[str, Any]) -> Dict[str, np.ndarray]:
    R = fields["R"]
    Z = fields["Z"]
    R_sep, Z_sep = _get_lcfs(shape)
    R_ax, Z_ax = _get_axis(shape, None, R, Z, fields["psi"])

    # Select row/column closest to Z_axis. With R,Z mesh shape [nR,nZ], Z varies axis 1.
    if Z.ndim == 2:
        j = int(np.nanargmin(np.abs(Z[0, :] - Z_ax)))
        rline = R[:, j]
        zline = Z[:, j]
        idx = (slice(None), j)
    else:
        raise RuntimeError("Expected 2D R/Z arrays for midplane sampling.")

    crossings = _midplane_crossings(R_sep, Z_sep, Z_ax)
    return {
        "R": np.asarray(rline, float),
        "Z": np.asarray(zline, float),
        "Br": np.asarray(fields["Br"][idx], float),
        "Bz": np.asarray(fields["Bz"][idx], float),
        "Bp": np.asarray(fields["Bp"][idx], float),
        "Bt": np.asarray(fields["Bt"][idx], float),
        "Btot": np.asarray(fields["Btot"][idx], float),
        "R_axis": np.asarray([R_ax]),
        "Z_axis": np.asarray([Z_ax]),
        "R_lcfs_crossings": np.asarray(crossings, float),
    }


def _safe_q_profile(eq: Any) -> Optional[Dict[str, np.ndarray]]:
    if not hasattr(eq, "q"):
        return None
    psin = np.linspace(0.02, 0.98, 160)
    try:
        q = np.asarray(eq.q(psin), dtype=float)
        good = np.isfinite(q)
        if np.count_nonzero(good) < 4:
            return None
        return {"psi_norm": psin[good], "q": q[good]}
    except Exception:
        return None


def _safe_pressure_profile(eq: Any) -> Optional[Dict[str, np.ndarray]]:
    if not hasattr(eq, "pressure"):
        return None
    psin = np.linspace(0.0, 0.98, 160)
    try:
        p = np.asarray(eq.pressure(psin), dtype=float)
        good = np.isfinite(p)
        if np.count_nonzero(good) < 4:
            return None
        return {"psi_norm": psin[good], "pressure_Pa": p[good]}
    except Exception:
        return None


def _make_profiles(eq: Any, cfg: Any) -> Optional[Any]:
    try:
        from freegsnke.jtor_update import ConstrainPaxisIp
        profiles = ConstrainPaxisIp(
            eq=eq,
            paxis=float(getattr(cfg, "paxis", 0.0)),
            Ip=float(getattr(cfg, "Ip", 0.0)),
            fvac=float(getattr(cfg, "fvac", 1.0)),
            alpha_m=float(getattr(cfg, "alpha_m", 1.5)),
            alpha_n=float(getattr(cfg, "alpha_n", 1.1)),
        )
        try:
            profiles.diverted_core_mask = np.ones_like(_get_eq_arrays(eq)[2], dtype=bool)
        except Exception:
            pass
        return profiles
    except Exception:
        return None


def _compute_jtor(eq: Any, cfg: Any, shape: Dict[str, Any]) -> Optional[np.ndarray]:
    profiles = _make_profiles(eq, cfg)
    if profiles is None or not hasattr(profiles, "Jtor"):
        return None
    try:
        R, Z, psi = _get_eq_arrays(eq)
        R_sep, Z_sep = _get_lcfs(shape)
        psib = _psi_boundary(eq, R_sep, Z_sep, R, Z, psi)
        jtor = profiles.Jtor(R, Z, psi, psib)
        return np.asarray(jtor, dtype=float)
    except Exception:
        return None


def _compute_global(eq: Any, cfg: Any, geom: Dict[str, Any], shape: Dict[str, Any], fields: Dict[str, Any]) -> Dict[str, Any]:
    R, Z, psi = fields["R"], fields["Z"], fields["psi"]
    R_sep, Z_sep = _get_lcfs(shape)
    inside = _inside_polygon_mask(R, Z, R_sep, Z_sep)
    R_ax, Z_ax = _get_axis(shape, eq, R, Z, psi)
    psi_axis = _interp_at_RZ(psi, R, Z, R_ax, Z_ax)
    psi_b = _psi_boundary(eq, R_sep, Z_sep, R, Z, psi)
    denom = psi_b - psi_axis
    if abs(denom) < 1e-30:
        psin = np.full_like(psi, np.nan)
    else:
        psin = np.clip((psi - psi_axis) / denom, 0.0, 1.0)

    try:
        p_grid = np.asarray(eq.pressure(psin), dtype=float)
    except Exception:
        p_grid = np.zeros_like(psi)

    dR, dZ = fields["dR"], fields["dZ"]
    dV = 2.0 * math.pi * R * dR * dZ
    V = float(np.nansum(dV[inside]))
    int_p_dV = float(np.nansum(p_grid[inside] * dV[inside])) if V > 0 else float("nan")
    pavg = int_p_dV / V if V > 0 else float("nan")
    Wth = 1.5 * int_p_dV if np.isfinite(int_p_dV) else float("nan")

    # B0 at R0 from fvac/R0.
    geometry = _compute_geometry(shape, eq, geom)
    R0 = geometry["R0_m"]
    fvac = float(getattr(cfg, "fvac", 1.0))
    B0 = fvac / R0 if R0 > 0 else float("nan")
    beta_t = 2.0 * MU0 * pavg / (B0 * B0) if B0 and np.isfinite(B0) else float("nan")

    out = {
        "plasma_volume_m3": V,
        "pressure_volume_integral_J_like": int_p_dV,
        "thermal_energy_1p5p_J": Wth,
        "pressure_avg_Pa": pavg,
        "B0_toroidal_T": B0,
        "beta_t_volume_avg": beta_t,
    }
    try:
        out["beta_p"] = float(eq.poloidalBeta1())
    except Exception:
        pass
    return out


def _family_from_label(label: str) -> str:
    s = str(label).upper()
    if s.startswith("CS_MID"):
        return "CS_MID"
    if s.startswith("CS_END"):
        return "CS_END"
    if s.startswith("CS"):
        return "CS"
    for fam in ("PF1", "PF2", "PF3", "PF4", "PF5", "PF6"):
        if s.startswith(fam):
            return fam
    if s.startswith("PASSIVE") or s.startswith("STAR_VESSEL") or s.startswith("SV"):
        return "PASSIVE"
    return "OTHER"


def _coil_current_table(tokamak: Any, geom: Dict[str, Any]) -> List[Dict[str, Any]]:
    groups = geom.get("coil_groups", {}) or getattr(tokamak, "coil_groups", {}) or {}
    coils_dict = getattr(tokamak, "coils_dict", {}) or {}
    # Normalize labels.
    cdict = {str(k).upper(): v for k, v in coils_dict.items()}
    imax = geom.get("Imax_recommended_MA", {}) or {}
    rows: List[Dict[str, Any]] = []
    for fam, labs in groups.items():
        famu = str(fam).upper()
        total_A = 0.0
        n_found = 0
        for lab in labs:
            coil = cdict.get(str(lab).upper())
            if coil is None:
                continue
            try:
                total_A += float(getattr(coil, "current", 0.0))
                n_found += 1
            except Exception:
                pass
        I_MA = total_A / 1e6
        Imax_MA = float(imax.get(famu, imax.get(fam, np.nan))) if imax else float("nan")
        util = abs(I_MA) / Imax_MA * 100.0 if np.isfinite(Imax_MA) and Imax_MA > 0 else float("nan")
        rows.append({"family": famu, "n_filaments": int(n_found), "I_MA": I_MA, "Imax_MA": Imax_MA, "utilization_percent": util})
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_profile_csv(path: Path, qprof: Optional[Dict[str, np.ndarray]], pprof: Optional[Dict[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    psis = np.linspace(0.0, 0.98, 160)
    rows = []
    qpsi = qprof["psi_norm"] if qprof else np.array([])
    qval = qprof["q"] if qprof else np.array([])
    ppsi = pprof["psi_norm"] if pprof else np.array([])
    pval = pprof["pressure_Pa"] if pprof else np.array([])
    for x in psis:
        qx = float(np.interp(x, qpsi, qval)) if len(qpsi) else float("nan")
        px = float(np.interp(x, ppsi, pval)) if len(ppsi) else float("nan")
        rows.append({"psi_norm": float(x), "q": qx, "pressure_Pa": px})
    _write_csv(path, rows)


def _write_midplane_csv(path: Path, mid: Dict[str, np.ndarray]) -> None:
    rows = []
    for i in range(len(mid["R"])):
        rows.append({
            "R_m": float(mid["R"][i]),
            "Z_m": float(mid["Z"][i]),
            "Br_T": float(mid["Br"][i]),
            "Bz_T": float(mid["Bz"][i]),
            "Bp_T": float(mid["Bp"][i]),
            "Bt_T": float(mid["Bt"][i]),
            "Btotal_T": float(mid["Btot"][i]),
        })
    _write_csv(path, rows)


def _plot_rect(ax, Rc: float, Zc: float, dR: float, dZ: float, **kwargs) -> None:
    ax.add_patch(Rectangle((Rc - 0.5 * dR, Zc - 0.5 * dZ), dR, dZ, fill=False, **kwargs))


def plot_equilibrium_clean(figdir: Path, eq: Any, geom: Dict[str, Any], shape: Dict[str, Any], fields: Dict[str, Any], args: argparse.Namespace) -> None:
    figdir.mkdir(parents=True, exist_ok=True)
    R, Z, psi = fields["R"], fields["Z"], fields["psi"]
    R_sep, Z_sep = _get_lcfs(shape)
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.contour(R, Z, psi, levels=40, linewidths=0.8)

    if args.plot_passives:
        for item in geom.get("passive_filaments", [])[:args.max_passive_plot]:
            try:
                lab, Rc, Zc, dR, dZ = item[:5]
                _plot_rect(ax, float(Rc), float(Zc), float(dR), float(dZ), linewidth=0.25, alpha=0.25)
            except Exception:
                continue
    else:
        # Try plotting passive polygons if present; otherwise skip.
        for poly in geom.get("star_vessel_polys", []) or geom.get("passive_polygons", []) or []:
            try:
                arr = np.asarray(poly, float)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    ax.add_patch(MplPolygon(arr[:, :2], closed=True, facecolor="0.8", edgecolor="0.55", alpha=0.25, linewidth=0.5))
            except Exception:
                pass

    # Active coils as rectangles.
    label_done = set()
    for lab, val in (geom.get("coils", {}) or {}).items():
        try:
            Rc, Zc, dR, dZ = map(float, val[:4])
        except Exception:
            continue
        fam = _family_from_label(lab)
        label = f"{fam} active" if fam not in label_done else None
        _plot_rect(ax, Rc, Zc, dR, dZ, linewidth=0.35, alpha=0.55, label=label)
        label_done.add(fam)

    if "R_outer" in geom and "Z_outer" in geom:
        ax.plot(geom["R_outer"], geom["Z_outer"], "-", lw=2.0, label="WALL_OUTER / blanket outer")
    if "R_inner" in geom and "Z_inner" in geom:
        ax.plot(geom["R_inner"], geom["Z_inner"], "--", lw=2.0, label="WALL_INNER / limiter")
    if "R_plasma" in geom and "Z_plasma" in geom:
        ax.plot(geom["R_plasma"], geom["Z_plasma"], lw=1.3, alpha=0.8, label="CAD target")

    ax.plot(R_sep, Z_sep, lw=2.7, label="LCFS / separatrix")
    Rax, Zax = _get_axis(shape, eq, R, Z, psi)
    ax.plot([Rax], [Zax], marker="x", ms=9, mew=2, linestyle="None", label="magnetic axis")

    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("STAR-like equilibrium: paper-style overview")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "equilibrium_overview.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_midplane_fields(figdir: Path, mid: Dict[str, np.ndarray], args: argparse.Namespace) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = [
        ("Br", r"$B_R$ [T]", r"Radial field $B_R$ at magnetic midplane"),
        ("Bz", r"$B_Z$ [T]", r"Vertical field $B_Z$ at magnetic midplane"),
        ("Bp", r"$B_p$ [T]", r"Poloidal field $B_p$ at magnetic midplane"),
        ("Bt", r"$B_\phi$ [T]", r"Toroidal field $B_\phi$ at magnetic midplane"),
    ]
    R = mid["R"]
    Rax = float(mid["R_axis"][0])
    crossings = list(mid.get("R_lcfs_crossings", []))
    for ax, (key, ylabel, title) in zip(axs.ravel(), panels):
        ax.plot(R, mid[key], lw=1.8)
        ax.axvline(Rax, ls=":", lw=1.5, label="magnetic axis")
        for k, rc in enumerate(crossings[:2]):
            ax.axvline(float(rc), ls="--", lw=1.2, label="LCFS midplane" if k == 0 else None)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.30)
    axs[-1, 0].set_xlabel("R [m]")
    axs[-1, 1].set_xlabel("R [m]")
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "midplane_magnetic_fields.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_profiles(figdir: Path, qprof: Optional[Dict[str, np.ndarray]], pprof: Optional[Dict[str, np.ndarray]], args: argparse.Namespace) -> None:
    if qprof is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(qprof["psi_norm"], qprof["q"], lw=1.8)
        ax.set_xlabel(r"$\psi_N$")
        ax.set_ylabel("q")
        ax.set_title("Safety factor profile")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figdir / "q_profile.png", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
    if pprof is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(pprof["psi_norm"], pprof["pressure_Pa"], lw=1.8)
        ax.set_xlabel(r"$\psi_N$")
        ax.set_ylabel("p [Pa]")
        ax.set_title("Pressure profile")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figdir / "pressure_profile.png", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)


def plot_jtor(figdir: Path, eq: Any, geom: Dict[str, Any], shape: Dict[str, Any], jtor: Optional[np.ndarray], args: argparse.Namespace) -> None:
    if jtor is None:
        return
    R, Z, psi = _get_eq_arrays(eq)
    R_sep, Z_sep = _get_lcfs(shape)
    inside = _inside_polygon_mask(R, Z, R_sep, Z_sep)
    data = np.where(inside, jtor / 1e6, np.nan)
    vmax = float(np.nanmax(np.abs(data))) if np.any(np.isfinite(data)) else 1.0
    fig, ax = plt.subplots(figsize=(6, 8))
    pcm = ax.pcolormesh(R, Z, data, shading="auto", vmin=-vmax, vmax=vmax)
    fig.colorbar(pcm, ax=ax, label=r"$j_\phi$ [MA/m$^2$]")
    ax.plot(R_sep, Z_sep, lw=2.0, label="LCFS")
    if "R_inner" in geom and "Z_inner" in geom:
        ax.plot(geom["R_inner"], geom["Z_inner"], "--", lw=1.8, label="limiter")
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(r"Toroidal current density $j_\phi$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "jtor_map.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_coil_utilization(figdir: Path, rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    rows2 = [r for r in rows if np.isfinite(float(r.get("utilization_percent", np.nan)))]
    if not rows2:
        return
    fam = [r["family"] for r in rows2]
    util = [float(r["utilization_percent"]) for r in rows2]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(fam, util)
    ax.axhline(100.0, ls="--", lw=1.2)
    ax.set_ylabel("|I| / Imax [%]")
    ax.set_title("Coil current utilization")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "coil_current_utilization.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(rows: List[Tuple[str, Any, str]], title: str) -> str:
    lines = [f"### {title}", "", "| Quantity | Value | Unit/Note |", "|---|---:|---|"]
    for k, v, unit in rows:
        if isinstance(v, float):
            if abs(v) >= 1e4 or (abs(v) < 1e-3 and v != 0):
                s = f"{v:.6e}"
            else:
                s = f"{v:.5f}"
        else:
            s = str(v)
        lines.append(f"| {k} | {s} | {unit} |")
    return "\n".join(lines) + "\n"


def write_paper_summary(path: Path, geometry: Dict[str, Any], containment: Dict[str, Any], global_numbers: Dict[str, Any], coil_rows: List[Dict[str, Any]], qprof: Optional[Dict[str, np.ndarray]], pprof: Optional[Dict[str, np.ndarray]], geom: Dict[str, Any]) -> None:
    rows_geom = [
        ("R_axis", geometry.get("R_axis_m"), "m"),
        ("Z_axis", geometry.get("Z_axis_m"), "m"),
        ("R0", geometry.get("R0_m"), "m"),
        ("a", geometry.get("a_m"), "m"),
        ("A", geometry.get("A"), "-"),
        ("kappa", geometry.get("kappa"), "-"),
        ("delta_u", geometry.get("delta_u"), "-"),
        ("delta_l", geometry.get("delta_l"), "-"),
        ("area", geometry.get("area_m2"), "m^2"),
        ("volume", geometry.get("volume_toroidal_m3"), "m^3, shell approx."),
        ("R bounds", f"[{geometry.get('Rmin_m'):.4f}, {geometry.get('Rmax_m'):.4f}]", "m"),
        ("Z bounds", f"[{geometry.get('Zmin_m'):.4f}, {geometry.get('Zmax_m'):.4f}]", "m"),
    ]
    rows_cont = [(k, v, "") for k, v in containment.items()]
    rows_glob = [(k, v, "") for k, v in global_numbers.items()]

    lines = ["# STAR-like equilibrium diagnostics", ""]
    lines.append("Generated by `diagnose_star_equilibrium.py`. Values are intended for paper-style reporting and internal validation.")
    lines.append("")
    lines.append(_markdown_table(rows_geom, "Equilibrium geometry"))
    lines.append(_markdown_table(rows_cont, "Containment / wall metrics"))
    lines.append(_markdown_table(rows_glob, "Global plasma estimates"))

    if qprof is not None:
        ps = qprof["psi_norm"]; q = qprof["q"]
        q95 = float(np.interp(0.95, ps, q)) if len(ps) else float("nan")
        qmin = float(np.nanmin(q)) if len(q) else float("nan")
        lines.append(_markdown_table([("q95", q95, "interpolated"), ("qmin", qmin, "profile min")], "Safety factor"))

    if pprof is not None:
        ps = pprof["psi_norm"]; p = pprof["pressure_Pa"]
        p0 = float(np.interp(0.0, ps, p)) if len(ps) else float("nan")
        lines.append(_markdown_table([("p_axis", p0, "Pa")], "Pressure"))

    lines.append("### Coil currents")
    lines.append("")
    lines.append("| Family | n filaments | I [MA] | Imax [MA] | Utilization [%] |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in coil_rows:
        lines.append(f"| {r['family']} | {r['n_filaments']} | {r['I_MA']:.5f} | {r['Imax_MA']:.5f} | {r['utilization_percent']:.2f} |")
    lines.append("")

    if geom.get("geometry_semantics"):
        lines.append("### CAD semantics")
        lines.append("")
        lines.append("```text")
        for k, v in geom.get("geometry_semantics", {}).items():
            lines.append(f"{k}: {v}")
        lines.append("```")
        lines.append("")
    if geom.get("passive_meta"):
        lines.append("### Passive structures metadata")
        lines.append("")
        lines.append("```text")
        for k, v in geom.get("passive_meta", {}).items():
            lines.append(f"{k}: {v}")
        lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-style diagnostics for STAR-like equilibrium.")
    parser.add_argument("--dxf", default=None, help="DXF path passed to star_equilibrium.build_equilibrium")
    parser.add_argument("--outdir", default=None, help="Output directory. Default: ../results/diagnostics_<timestamp>")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--plot-passives", action="store_true", help="Plot passive filaments individually, limited by --max-passive-plot")
    parser.add_argument("--max-passive-plot", type=int, default=5000)
    parser.add_argument("--no-solver-noise-redirect", action="store_true", help="Do not redirect solver noise when building equilibrium")
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve() if args.outdir else _default_results_dir().resolve()
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    # Import local pipeline after parsing so it uses user's current working tree.
    import config_star_bean as cfg
    import star_equilibrium as se

    print(f"[diagnostics] outdir = {outdir}")
    print("[diagnostics] building equilibrium using star_equilibrium.build_equilibrium(...)")
    eq, tokamak, geom, shape = se.build_equilibrium(
        verbose=True,
        redirect_solver_noise=(not args.no_solver_noise_redirect),
        dxf_path=args.dxf,
    )

    R, Z, psi = _get_eq_arrays(eq)
    R_sep, Z_sep = _get_lcfs(shape)
    geometry = _compute_geometry(shape, eq, geom)
    containment = _compute_containment(geom, shape)
    fields = _compute_fields(eq, shape, cfg)
    mid = _sample_midplane(fields, shape)
    qprof = _safe_q_profile(eq)
    pprof = _safe_pressure_profile(eq)
    jtor = _compute_jtor(eq, cfg, shape)
    global_numbers = _compute_global(eq, cfg, geom, shape, fields)
    coil_rows = _coil_current_table(tokamak, geom)

    # Console summary.
    print("\n=== PAPER-STYLE GEOMETRY ===")
    for k in ["R_axis_m", "Z_axis_m", "R0_m", "a_m", "A", "kappa", "delta_u", "delta_l", "area_m2", "volume_toroidal_m3"]:
        print(f"{k:24s} = {geometry.get(k)}")
    print("\n=== CONTAINMENT ===")
    for k, v in containment.items():
        print(f"{k:34s} = {v}")
    print("\n=== COIL UTILIZATION ===")
    for r in coil_rows:
        print(f"{r['family']:8s} I={r['I_MA']:+9.4f} MA  Imax={r['Imax_MA']:9.4f} MA  util={r['utilization_percent']:7.2f}%")

    # Files.
    _write_csv(outdir / "geometry_table.csv", [{k: v for k, v in geometry.items()}])
    _write_csv(outdir / "containment_table.csv", [{k: v for k, v in containment.items()}])
    _write_csv(outdir / "global_numbers_table.csv", [{k: v for k, v in global_numbers.items()}])
    _write_csv(outdir / "coil_currents_table.csv", coil_rows)
    _write_profile_csv(outdir / "profiles_q_pressure.csv", qprof, pprof)
    _write_midplane_csv(outdir / "midplane_fields.csv", mid)

    summary = {
        "geometry": geometry,
        "containment": containment,
        "global_numbers": global_numbers,
        "coil_currents": coil_rows,
        "cad_semantics": geom.get("geometry_semantics", {}),
        "passive_meta": geom.get("passive_meta", {}),
        "materials_meta": geom.get("materials_meta", {}),
        "midplane_lcfs_crossings_R_m": [float(x) for x in mid.get("R_lcfs_crossings", [])],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_paper_summary(outdir / "summary_paper.md", geometry, containment, global_numbers, coil_rows, qprof, pprof, geom)

    # Figures.
    plot_equilibrium_clean(figdir, eq, geom, shape, fields, args)
    plot_midplane_fields(figdir, mid, args)
    plot_profiles(figdir, qprof, pprof, args)
    plot_jtor(figdir, eq, geom, shape, jtor, args)
    plot_coil_utilization(figdir, coil_rows, args)

    print("\n[diagnostics] saved:")
    print(f"  {outdir / 'summary.json'}")
    print(f"  {outdir / 'summary_paper.md'}")
    print(f"  {figdir}")


if __name__ == "__main__":
    main()
