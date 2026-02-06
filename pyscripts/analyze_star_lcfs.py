# analyze_star_lcfs.py
from __future__ import annotations

import math
import numpy as np

try:
    import contourpy  # suele venir con matplotlib
except Exception:
    contourpy = None

from matplotlib.path import Path


def _grid_1d_from_mesh(M: np.ndarray, axis: int) -> np.ndarray:
    """
    Para meshgrid con indexing='ij':
      axis=0 -> R (varía en filas): M[:,0]
      axis=1 -> Z (varía en columnas): M[0,:]
    """
    M = np.asarray(M)
    if M.ndim == 1:
        return M.astype(float).copy()
    if axis == 0:
        return np.asarray(M[:, 0], dtype=float)
    return np.asarray(M[0, :], dtype=float)


def _orient_psi_RZ(psi: np.ndarray, R1d: np.ndarray, Z1d: np.ndarray) -> np.ndarray:
    """
    Devuelve psi_RZ con shape (nR, nZ), consistente con R1d, Z1d.
    """
    psi = np.asarray(psi, float)
    nR = R1d.size
    nZ = Z1d.size
    if psi.shape == (nR, nZ):
        return psi
    if psi.shape == (nZ, nR):
        return psi.T
    raise ValueError(f"psi grid shape mismatch: psi.shape={psi.shape} vs (nR,nZ)=({nR},{nZ})")


def _ensure_increasing_axes(R1d: np.ndarray, Z1d: np.ndarray, psi_RZ: np.ndarray):
    """
    searchsorted requiere ejes crecientes. Si vienen decrecientes, se invierten y
    se invierte psi_RZ consistentemente.
    """
    R1d = np.asarray(R1d, float)
    Z1d = np.asarray(Z1d, float)
    psi_RZ = np.asarray(psi_RZ, float)

    if R1d[0] > R1d[-1]:
        R1d = R1d[::-1].copy()
        psi_RZ = psi_RZ[::-1, :].copy()

    if Z1d[0] > Z1d[-1]:
        Z1d = Z1d[::-1].copy()
        psi_RZ = psi_RZ[:, ::-1].copy()

    return R1d, Z1d, psi_RZ


def _bilinear_interp(R1d, Z1d, psi_RZ, Rp, Zp):
    """
    Interp bilineal vectorizada sobre grilla rectangular.
    R1d: (nR,), Z1d: (nZ,), psi_RZ: (nR,nZ)
    """
    R1d = np.asarray(R1d, float)
    Z1d = np.asarray(Z1d, float)
    F = np.asarray(psi_RZ, float)

    nR = R1d.size
    nZ = Z1d.size
    if F.shape != (nR, nZ):
        raise ValueError(f"psi_RZ shape mismatch: {F.shape} vs {(nR,nZ)}")

    Rp = np.asarray(Rp, float)
    Zp = np.asarray(Zp, float)

    # clamp
    Rp = np.clip(Rp, R1d[0], R1d[-1])
    Zp = np.clip(Zp, Z1d[0], Z1d[-1])

    i = np.searchsorted(R1d, Rp, side="right") - 1
    j = np.searchsorted(Z1d, Zp, side="right") - 1
    i = np.clip(i, 0, nR - 2)
    j = np.clip(j, 0, nZ - 2)

    R0 = R1d[i]
    R1 = R1d[i + 1]
    Z0 = Z1d[j]
    Z1 = Z1d[j + 1]

    t = (Rp - R0) / (R1 - R0 + 1e-30)
    u = (Zp - Z0) / (Z1 - Z0 + 1e-30)

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


def _pick_closed_contour_enclosing_axis(segments, R_ax, Z_ax, close_tol: float) -> np.ndarray | None:
    """
    segments: lista de arrays (N,2) con (R,Z)
    Selecciona el contorno "casi cerrado" que encierra el eje.
    close_tol: tolerancia absoluta para gap entre endpoints.
    """
    cand = []
    for seg in segments:
        if seg is None:
            continue
        seg = np.asarray(seg, float)
        if seg.ndim != 2 or seg.shape[1] != 2 or seg.shape[0] < 20:
            continue

        gap = float(np.linalg.norm(seg[0] - seg[-1]))
        L = _polyline_length(seg)

        # criterio robusto: gap pequeño comparado con L o con close_tol
        if not (gap <= close_tol or (L > 0 and gap / L < 0.02)):
            continue

        # cerrar explícitamente para Path/area
        if gap > 0:
            seg = np.vstack([seg, seg[0]])

        try:
            if not Path(seg).contains_point((R_ax, Z_ax)):
                continue
        except Exception:
            continue

        area = abs(_polygon_area(seg))
        cand.append((area, seg))

    if not cand:
        return None

    # mayor área = boundary más externa que sigue encerrando el eje
    cand.sort(key=lambda t: t[0], reverse=True)
    return cand[0][1]


def shape_from_lcfs_limiter(
    eq,
    geom: dict,
    *,
    prefer_inner: bool = True,
    edge_pad_cells: int = 2,
    psi_percentile: float | None = None,
) -> dict:
    """
    LCFS para plasma limitado (limiter).
    - psi_lcfs se toma desde psi sobre el limiter:
        * si psi_axis es mínimo => psi_lcfs ~ min(psi_lim)
        * si psi_axis es máximo => psi_lcfs ~ max(psi_lim)
      con opción robusta por percentil (psi_percentile).

    Retorna dict con:
      R_sep, Z_sep, R0_plasma, a_plasma, A_plasma, kappa_plasma, delta_u, delta_l,
      R_ax, Z_ax, psi_axis, psi_lcfs, axis_is_min
    """
    # psi puede ser método o atributo
    psi_raw = eq.psi() if callable(getattr(eq, "psi", None)) else getattr(eq, "psi")
    psi_raw = np.asarray(psi_raw, float)

    Rm = np.asarray(eq.R, float)
    Zm = np.asarray(eq.Z, float)

    R1d = _grid_1d_from_mesh(Rm, axis=0)
    Z1d = _grid_1d_from_mesh(Zm, axis=1)

    psi_RZ = _orient_psi_RZ(psi_raw, R1d, Z1d)
    R1d, Z1d, psi_RZ = _ensure_increasing_axes(R1d, Z1d, psi_RZ)

    # axis
    R_ax, Z_ax = eq.magneticAxis()[:2]
    R_ax = float(R_ax)
    Z_ax = float(Z_ax)
    psi_axis = float(_bilinear_interp(R1d, Z1d, psi_RZ, np.array([R_ax]), np.array([Z_ax]))[0])

    # limiter polyline
    if prefer_inner and ("R_inner" in geom and "Z_inner" in geom):
        R_lim = np.asarray(geom["R_inner"], float)
        Z_lim = np.asarray(geom["Z_inner"], float)
    else:
        R_lim = np.asarray(geom["R_outer"], float)
        Z_lim = np.asarray(geom["Z_outer"], float)

    psi_lim = _bilinear_interp(R1d, Z1d, psi_RZ, R_lim, Z_lim)
    psi_lim = psi_lim[np.isfinite(psi_lim)]
    if psi_lim.size < 10:
        raise RuntimeError("psi_lim demasiado corto/no finito.")

    # axis min vs max (comparación con borde del dominio)
    psi_edge = float(np.nanmean(np.r_[psi_RZ[0, :], psi_RZ[-1, :], psi_RZ[:, 0], psi_RZ[:, -1]]))
    axis_is_min = (psi_edge > psi_axis)

    # robustez por percentil (si se pide)
    if psi_percentile is None:
        psi_lcfs = float(np.min(psi_lim) if axis_is_min else np.max(psi_lim))
    else:
        p = float(psi_percentile)
        p = min(max(p, 0.0), 50.0)
        if axis_is_min:
            psi_lcfs = float(np.nanpercentile(psi_lim, p))
        else:
            psi_lcfs = float(np.nanpercentile(psi_lim, 100.0 - p))

    # contouring: contourpy espera z con shape (nZ,nR)
    z_for_contour = psi_RZ.T  # (nZ, nR)

    segments = []
    if contourpy is not None:
        cg = contourpy.contour_generator(x=R1d, y=Z1d, z=z_for_contour, name="serial")
        segments = cg.lines(psi_lcfs)
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        cs = ax.contour(R1d, Z1d, z_for_contour, levels=[psi_lcfs])
        if hasattr(cs, "allsegs") and cs.allsegs and cs.allsegs[0]:
            segments = [np.asarray(s, float) for s in cs.allsegs[0]]
        plt.close(fig)

    # tolerancia de cierre basada en malla
    dR = float(R1d[1] - R1d[0]) if R1d.size > 1 else 0.0
    dZ = float(Z1d[1] - Z1d[0]) if Z1d.size > 1 else 0.0
    close_tol = 5.0 * max(dR, dZ, 1e-6)

    seg = _pick_closed_contour_enclosing_axis(segments, R_ax, Z_ax, close_tol=close_tol)
    if seg is None:
        raise RuntimeError("No se encontró un contorno LCFS (casi cerrado) que encierre al eje.")

    R_sep = seg[:, 0].copy()
    Z_sep = seg[:, 1].copy()

    # Sanity: evitar LCFS pegada al borde del dominio numérico
    Rmin, Rmax = float(R1d[0]), float(R1d[-1])
    Zmin, Zmax = float(Z1d[0]), float(Z1d[-1])
    padR = edge_pad_cells * dR
    padZ = edge_pad_cells * dZ
    if (R_sep.min() <= Rmin + padR) or (R_sep.max() >= Rmax - padR) or (Z_sep.min() <= Zmin + padZ) or (Z_sep.max() >= Zmax - padZ):
        raise RuntimeError("LCFS tocó el borde del dominio (probable psi_lcfs mal elegido o contorno no físico).")

    # Métricas geométricas
    R_lo = float(np.min(R_sep))
    R_hi = float(np.max(R_sep))
    Z_lo = float(np.min(Z_sep))
    Z_hi = float(np.max(Z_sep))

    R0_pl = 0.5 * (R_hi + R_lo)
    a_pl = 0.5 * (R_hi - R_lo)
    if a_pl <= 0:
        raise RuntimeError("a_plasma <= 0 (LCFS degenerada).")

    A_pl = R0_pl / a_pl
    kappa = (Z_hi - Z_lo) / (R_hi - R_lo + 1e-30)

    i_top = int(np.argmax(Z_sep))
    i_bot = int(np.argmin(Z_sep))
    R_top = float(R_sep[i_top])
    R_bot = float(R_sep[i_bot])
    delta_u = (R0_pl - R_top) / a_pl
    delta_l = (R0_pl - R_bot) / a_pl

    return dict(
        R_sep=R_sep, Z_sep=Z_sep,
        R0_plasma=float(R0_pl),
        a_plasma=float(a_pl),
        A_plasma=float(A_pl),
        kappa_plasma=float(kappa),
        delta_u=float(delta_u),
        delta_l=float(delta_l),
        R_ax=float(R_ax),
        Z_ax=float(Z_ax),
        psi_axis=float(psi_axis),
        psi_lcfs=float(psi_lcfs),
        axis_is_min=bool(axis_is_min),
    )

