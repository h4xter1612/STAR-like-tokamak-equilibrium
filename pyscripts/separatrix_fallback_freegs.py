# separatrix_fallback_freegs.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path


def _as_2d_RZ(eq):
    """
    Return RR, ZZ meshgrid arrays compatible with psi.
    Handles common FreeGS/FreeGSNKE layouts.
    """
    R = np.asarray(getattr(eq, "R"))
    Z = np.asarray(getattr(eq, "Z"))

    psi = np.asarray(eq.psi())

    if R.ndim == 2 and Z.ndim == 2:
        return R, Z, psi

    if R.ndim == 1 and Z.ndim == 1:
        # Most common: psi shape is (len(R), len(Z)) or (len(Z), len(R)).
        RR, ZZ = np.meshgrid(R, Z, indexing="ij")
        if psi.shape == RR.shape:
            return RR, ZZ, psi

        RR2, ZZ2 = np.meshgrid(R, Z, indexing="xy")
        if psi.shape == RR2.shape:
            return RR2, ZZ2, psi

    raise ValueError(
        f"Could not align R/Z/psi shapes: R={R.shape}, Z={Z.shape}, psi={psi.shape}"
    )


def _axis_position(eq):
    """
    Robustly obtain magnetic axis position.
    """
    for rname, zname in [
        ("magneticAxis", None),
        ("axis", None),
        ("R_axis", "Z_axis"),
        ("Rax", "Zax"),
        ("Rmag", "Zmag"),
    ]:
        if zname is None:
            obj = getattr(eq, rname, None)
            if obj is None:
                continue

            if callable(obj):
                try:
                    obj = obj()
                except TypeError:
                    pass

            try:
                if len(obj) >= 2:
                    return float(obj[0]), float(obj[1])
            except Exception:
                pass
        else:
            if hasattr(eq, rname) and hasattr(eq, zname):
                return float(getattr(eq, rname)), float(getattr(eq, zname))

    # fallback: minimum/maximum psi location is not always correct, but useful for debug
    RR, ZZ, psi = _as_2d_RZ(eq)
    idx = np.unravel_index(np.nanargmin(np.abs(psi - np.nanmin(psi))), psi.shape)
    return float(RR[idx]), float(ZZ[idx])


def _get_psi_bndry(eq):
    for name in ["psi_bndry", "psi_boundary", "psibndry", "psi_bnd"]:
        if hasattr(eq, name):
            val = getattr(eq, name)
            if callable(val):
                val = val()
            return float(val)
    raise AttributeError("Could not find eq.psi_bndry or equivalent attribute.")


def _segment_properties(Rs, Zs, Rax, Zax, RR=None, ZZ=None):
    Rs = np.asarray(Rs, dtype=float)
    Zs = np.asarray(Zs, dtype=float)

    if len(Rs) < 5:
        return None

    dclose = float(np.hypot(Rs[0] - Rs[-1], Zs[0] - Zs[-1]))
    chord = float(np.nanmax(np.hypot(Rs - np.nanmean(Rs), Zs - np.nanmean(Zs))))
    closed = dclose < max(1e-3, 0.03 * chord)

    # Polygon containment if possible.
    contains_axis = False
    if len(Rs) >= 4:
        try:
            poly = Path(np.column_stack([Rs, Zs]))
            contains_axis = bool(poly.contains_point((Rax, Zax)))
        except Exception:
            contains_axis = False

    dist_axis = float(np.nanmin(np.hypot(Rs - Rax, Zs - Zax)))
    length = float(np.nansum(np.hypot(np.diff(Rs), np.diff(Zs))))

    Rmin = float(np.nanmin(Rs))
    Rmax = float(np.nanmax(Rs))
    Zmin = float(np.nanmin(Zs))
    Zmax = float(np.nanmax(Zs))

    touches_domain_edge = False
    if RR is not None and ZZ is not None:
        rlo, rhi = float(np.nanmin(RR)), float(np.nanmax(RR))
        zlo, zhi = float(np.nanmin(ZZ)), float(np.nanmax(ZZ))
        tolR = 0.01 * max(1e-9, rhi - rlo)
        tolZ = 0.01 * max(1e-9, zhi - zlo)
        touches_domain_edge = (
            np.nanmin(np.abs(Rs - rlo)) < tolR
            or np.nanmin(np.abs(Rs - rhi)) < tolR
            or np.nanmin(np.abs(Zs - zlo)) < tolZ
            or np.nanmin(np.abs(Zs - zhi)) < tolZ
        )

    return {
        "n": int(len(Rs)),
        "closed": bool(closed),
        "contains_axis": bool(contains_axis),
        "dclose": dclose,
        "dist_axis": dist_axis,
        "length": length,
        "Rmin": Rmin,
        "Rmax": Rmax,
        "Zmin": Zmin,
        "Zmax": Zmax,
        "touches_domain_edge": bool(touches_domain_edge),
    }


def extract_freegs_psibndry_contours(eq, debug_plot=None):
    """
    Extract contours of psi = eq.psi_bndry.

    Returns:
        {
            "ok": bool,
            "R_sep": np.ndarray,
            "Z_sep": np.ndarray,
            "source": "closed_lcfs_from_psibndry" or "freegs_psibndry_fallback",
            "psi_bndry": float,
            "R_axis": float,
            "Z_axis": float,
            "segments": list[dict],
            "reason": str,
        }
    """
    RR, ZZ, psi = _as_2d_RZ(eq)
    psi_bndry = _get_psi_bndry(eq)
    Rax, Zax = _axis_position(eq)

    fig, ax = plt.subplots()
    try:
        cs = ax.contour(RR, ZZ, psi, levels=[psi_bndry])
        segments = []

        # Matplotlib version compatibility.
        paths = []
        if hasattr(cs, "allsegs") and cs.allsegs and len(cs.allsegs[0]) > 0:
            for arr in cs.allsegs[0]:
                if arr is not None and len(arr) > 3:
                    paths.append(arr)
        else:
            for coll in cs.collections:
                for path in coll.get_paths():
                    arr = path.vertices
                    if arr is not None and len(arr) > 3:
                        paths.append(arr)

        for i, arr in enumerate(paths):
            Rs = np.asarray(arr[:, 0], dtype=float)
            Zs = np.asarray(arr[:, 1], dtype=float)
            props = _segment_properties(Rs, Zs, Rax, Zax, RR=RR, ZZ=ZZ)
            if props is None:
                continue

            props["idx"] = int(i)
            props["R"] = Rs
            props["Z"] = Zs
            segments.append(props)

        if not segments:
            return {
                "ok": False,
                "reason": "no_psibndry_contour_segments",
                "psi_bndry": psi_bndry,
                "R_axis": Rax,
                "Z_axis": Zax,
                "segments": [],
            }

        # Prefer a closed contour containing the magnetic axis.
        closed_axis = [
            s for s in segments
            if s["closed"] and s["contains_axis"] and not s["touches_domain_edge"]
        ]

        if closed_axis:
            best = max(closed_axis, key=lambda s: s["length"])
            source = "closed_lcfs_from_psibndry"
            reason = "closed_axis_contour_found"
        else:
            # Fallback for double-null: choose long non-edge segment close to the axis.
            # This is not certified closed LCFS, but it is still the FreeGS psi_bndry contour.
            candidates = [s for s in segments if not s["touches_domain_edge"]]
            if not candidates:
                candidates = segments

            # Score: long, near axis, vertically extended, not tiny.
            def score(s):
                zspan = s["Zmax"] - s["Zmin"]
                rspan = s["Rmax"] - s["Rmin"]
                return (
                    2.0 * s["length"]
                    + 4.0 * zspan
                    + 2.0 * rspan
                    - 3.0 * s["dist_axis"]
                    - (20.0 if s["touches_domain_edge"] else 0.0)
                )

            best = max(candidates, key=score)
            source = "freegs_psibndry_fallback"
            reason = "no_closed_axis_contour_using_best_psibndry_segment"

        if debug_plot:
            ax.clear()
            ax.contour(RR, ZZ, psi, levels=40, linewidths=0.7)
            for s in segments:
                ax.plot(s["R"], s["Z"], lw=1.0)
                mid = len(s["R"]) // 2
                ax.text(s["R"][mid], s["Z"][mid], str(s["idx"]), fontsize=8)
            ax.plot(best["R"], best["Z"], "r-", lw=2.5)
            ax.plot([Rax], [Zax], "gx", ms=10)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"psi_bndry contours; selected={best['idx']} source={source}")
            ax.set_xlabel("R [m]")
            ax.set_ylabel("Z [m]")
            fig.savefig(debug_plot, dpi=180, bbox_inches="tight")

        return {
            "ok": True,
            "reason": reason,
            "source": source,
            "psi_bndry": psi_bndry,
            "R_axis": Rax,
            "Z_axis": Zax,
            "R_sep": np.asarray(best["R"], dtype=float),
            "Z_sep": np.asarray(best["Z"], dtype=float),
            "selected_idx": int(best["idx"]),
            "selected_closed": bool(best["closed"]),
            "selected_contains_axis": bool(best["contains_axis"]),
            "selected_touches_domain_edge": bool(best["touches_domain_edge"]),
            "segments": [
                {k: v for k, v in s.items() if k not in ("R", "Z")}
                for s in segments
            ],
        }

    finally:
        plt.close(fig)
