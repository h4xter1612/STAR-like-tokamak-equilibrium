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
    Extract all contours of psi = eq.psi_bndry.

    This does not invent a new separatrix. It uses FreeGS/FreeGSNKE's native
    psi_bndry level and returns the contour segments.
    """
    RR, ZZ, psi = _as_2d_RZ(eq)
    psi_bndry = _get_psi_bndry(eq)
    Rax, Zax = _axis_position(eq)

    fig, ax = plt.subplots()
    try:
        cs = ax.contour(RR, ZZ, psi, levels=[psi_bndry])
        segments = []

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
                "raw_segments": [],
            }

        closed_axis = [
            s for s in segments
            if s["closed"] and s["contains_axis"] and not s["touches_domain_edge"]
        ]

        if closed_axis:
            best = max(closed_axis, key=lambda s: s["length"])
            source = "closed_lcfs_from_psibndry"
            reason = "closed_axis_contour_found"
        else:
            candidates = [s for s in segments if not s["touches_domain_edge"]]
            if not candidates:
                candidates = segments

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

        raw_segments = [
            {
                "R": np.asarray(s["R"], dtype=float),
                "Z": np.asarray(s["Z"], dtype=float),
                "idx": int(s["idx"]),
            }
            for s in segments
        ]

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
            "raw_segments": raw_segments,
            "segments": [
                {k: v for k, v in s.items() if k not in ("R", "Z")}
                for s in segments
            ],
        }

    finally:
        plt.close(fig)


def reconstruct_dn_lcfs_from_segments(raw_segments, xpt_upper, xpt_lower, tol=0.55):
    """
    Reconstruct a double-null LCFS from psi=psi_bndry contour segments.

    The desired closed LCFS is:
        outboard branch between upper/lower X-points
        + inboard branch between lower/upper X-points

    raw_segments:
        list of {"R": array, "Z": array}
    xpt_upper:
        (R, Z)
    xpt_lower:
        (R, Z)
    tol:
        distance tolerance to associate contour points with X-points [m]
    """
    def nearest_idx(R, Z, pt):
        d = np.hypot(R - pt[0], Z - pt[1])
        i = int(np.argmin(d))
        return i, float(d[i])

    candidate_branches = []

    for iseg, seg in enumerate(raw_segments):
        R = np.asarray(seg["R"], dtype=float)
        Z = np.asarray(seg["Z"], dtype=float)

        if len(R) < 8:
            continue

        iu, du = nearest_idx(R, Z, xpt_upper)
        il, dl = nearest_idx(R, Z, xpt_lower)

        if du > tol or dl > tol:
            continue

        if abs(iu - il) < 4:
            continue

        i0, i1 = sorted([iu, il])
        Rsub = R[i0:i1 + 1]
        Zsub = Z[i0:i1 + 1]

        if len(Rsub) < 8:
            continue

        length = float(np.sum(np.hypot(np.diff(Rsub), np.diff(Zsub))))
        zspan = float(np.max(Zsub) - np.min(Zsub))
        rspan = float(np.max(Rsub) - np.min(Rsub))

        required_zspan = 0.35 * abs(xpt_upper[1] - xpt_lower[1])
        if zspan < required_zspan:
            continue

        candidate_branches.append({
            "seg_id": int(seg.get("idx", iseg)),
            "R": Rsub,
            "Z": Zsub,
            "mean_R": float(np.mean(Rsub)),
            "min_R": float(np.min(Rsub)),
            "max_R": float(np.max(Rsub)),
            "z_span": zspan,
            "r_span": rspan,
            "length": length,
            "du": du,
            "dl": dl,
            "iu": int(iu),
            "il": int(il),
        })

    if len(candidate_branches) < 2:
        return {
            "ok": False,
            "reason": "could_not_find_two_xpoint_to_xpoint_branches",
            "n_candidate_branches": len(candidate_branches),
        }

    candidate_branches = sorted(candidate_branches, key=lambda b: b["mean_R"])
    inb = candidate_branches[0]
    outb = candidate_branches[-1]

    Rin = np.asarray(inb["R"], dtype=float)
    Zin = np.asarray(inb["Z"], dtype=float)
    Rout = np.asarray(outb["R"], dtype=float)
    Zout = np.asarray(outb["Z"], dtype=float)

    # Orient outboard branch upper -> lower.
    if Zout[0] < Zout[-1]:
        Rout = Rout[::-1]
        Zout = Zout[::-1]

    # Orient inboard branch lower -> upper.
    if Zin[0] > Zin[-1]:
        Rin = Rin[::-1]
        Zin = Zin[::-1]

    R_lcfs = np.concatenate([Rout, Rin, [Rout[0]]])
    Z_lcfs = np.concatenate([Zout, Zin, [Zout[0]]])

    if len(R_lcfs) < 20:
        return {
            "ok": False,
            "reason": "reconstructed_lcfs_too_few_points",
            "n_points": int(len(R_lcfs)),
        }

    if not np.all(np.isfinite(R_lcfs)) or not np.all(np.isfinite(Z_lcfs)):
        return {
            "ok": False,
            "reason": "reconstructed_lcfs_nonfinite",
        }

    area = 0.5 * abs(np.sum(R_lcfs[:-1] * Z_lcfs[1:] - R_lcfs[1:] * Z_lcfs[:-1]))

    if area <= 1e-3:
        return {
            "ok": False,
            "reason": "reconstructed_lcfs_zero_area",
            "area": float(area),
        }

    return {
        "ok": True,
        "reason": "dn_lcfs_reconstructed_from_psibndry_segments",
        "source": "freegs_psibndry_dn_reconstructed",
        "R_lcfs": R_lcfs,
        "Z_lcfs": Z_lcfs,
        "area": float(area),
        "branches": {
            "inboard": {k: v for k, v in inb.items() if k not in ("R", "Z")},
            "outboard": {k: v for k, v in outb.items() if k not in ("R", "Z")},
        },
        "n_candidate_branches": len(candidate_branches),
    }


def _coerce_xpoints(xpoints):
    """
    Convert xpoints to [(R,Z), ...].
    Accepts tuples/lists or dicts with common key names.
    """
    if xpoints is None:
        return []

    pts = []
    for p in xpoints:
        if isinstance(p, dict):
            R = p.get("R", p.get("r", p.get("Rx", p.get("rx", None))))
            Z = p.get("Z", p.get("z", p.get("Zx", p.get("zx", None))))
            if R is not None and Z is not None:
                pts.append((float(R), float(Z)))
        else:
            try:
                pts.append((float(p[0]), float(p[1])))
            except Exception:
                pass

    return pts


def _plot_dn_reconstruction_debug(eq, out, debug_plot):
    RR, ZZ, psi = _as_2d_RZ(eq)
    psi_bndry = _get_psi_bndry(eq)

    fig, ax = plt.subplots(figsize=(7, 9))

    ax.contour(RR, ZZ, psi, levels=40, linewidths=0.7)
    ax.contour(RR, ZZ, psi, levels=[psi_bndry], colors="k", linewidths=1.0)

    for seg in out.get("raw_segments", []):
        ax.plot(seg["R"], seg["Z"], lw=1.0, alpha=0.55)

    R = np.asarray(out["R_sep"])
    Z = np.asarray(out["Z_sep"])
    ax.plot(R, Z, "r-", lw=2.8, label=out.get("source", "selected sep"))

    xu = out.get("xpt_upper_used")
    xl = out.get("xpt_lower_used")
    if xu is not None:
        ax.plot([xu[0]], [xu[1]], "rx", ms=10, mew=2, label="upper X used")
    if xl is not None:
        ax.plot([xl[0]], [xl[1]], "bx", ms=10, mew=2, label="lower X used")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(f"FreeGS psi_bndry DN reconstruction: {out.get('reason')}")
    ax.legend(loc="best")
    fig.savefig(debug_plot, dpi=180, bbox_inches="tight")
    plt.close(fig)

def compute_lcfs_geometry(R_sep, Z_sep):
    """
    Compute basic LCFS geometry from a closed R-Z curve.

    Returns:
        R0, Z0, a, A, kappa, delta_u, delta_l, delta_bar, area, bounds.
    """
    import numpy as np

    R = np.asarray(R_sep, dtype=float)
    Z = np.asarray(Z_sep, dtype=float)

    ok = np.isfinite(R) & np.isfinite(Z)
    R = R[ok]
    Z = Z[ok]

    if len(R) < 10:
        return {
            "ok": False,
            "reason": "too_few_lcfs_points",
        }

    # Ensure closed for area.
    if np.hypot(R[0] - R[-1], Z[0] - Z[-1]) > 1e-6:
        R = np.r_[R, R[0]]
        Z = np.r_[Z, Z[0]]

    Rmin = float(np.nanmin(R))
    Rmax = float(np.nanmax(R))
    Zmin = float(np.nanmin(Z))
    Zmax = float(np.nanmax(Z))

    R0 = 0.5 * (Rmin + Rmax)
    Z0 = 0.5 * (Zmin + Zmax)
    a = 0.5 * (Rmax - Rmin)

    if a <= 0:
        return {
            "ok": False,
            "reason": "nonpositive_minor_radius",
        }

    A = R0 / a
    kappa = 0.5 * (Zmax - Zmin) / a

    # Upper/lower tips.
    iu = int(np.nanargmax(Z))
    il = int(np.nanargmin(Z))

    R_upper = float(R[iu])
    R_lower = float(R[il])

    # Tokamak convention for positive triangularity:
    # upper/lower tips shifted inward relative to R0.
    delta_u = (R0 - R_upper) / a
    delta_l = (R0 - R_lower) / a
    delta_bar = 0.5 * (delta_u + delta_l)

    area = 0.5 * abs(np.sum(R[:-1] * Z[1:] - R[1:] * Z[:-1]))

    return {
        "ok": True,
        "R0": float(R0),
        "Z0": float(Z0),
        "a": float(a),
        "A": float(A),
        "kappa": float(kappa),
        "delta_u": float(delta_u),
        "delta_l": float(delta_l),
        "delta_bar": float(delta_bar),
        "area": float(area),
        "Rmin": Rmin,
        "Rmax": Rmax,
        "Zmin": Zmin,
        "Zmax": Zmax,
        "bounds": {
            "Rmin": Rmin,
            "Rmax": Rmax,
            "Zmin": Zmin,
            "Zmax": Zmax,
        },
    }

def extract_freegs_dn_lcfs(eq, xpoints=None, debug_plot=None, xpoint_tol=0.55):
    """
    High-level fallback.

    1. Extract all psi=psi_bndry contour segments.
    2. If X-points are available, reconstruct DN LCFS from the two branches.
    3. If DN reconstruction fails, return the simpler psi_bndry fallback segment.

    Important:
        has_true_sep remains False for reconstructed DN fallback.
        has_usable_sep is True if the FreeGS psi_bndry contour is usable.
    """
    fb = extract_freegs_psibndry_contours(eq, debug_plot=None)

    if not fb.get("ok", False):
        return fb

    raw_segments = fb.get("raw_segments", [])
    pts = _coerce_xpoints(xpoints)

    if len(pts) >= 2:
        pts = sorted(pts, key=lambda q: q[1])
        xpt_lower = pts[0]
        xpt_upper = pts[-1]

        dn = reconstruct_dn_lcfs_from_segments(
            raw_segments,
            xpt_upper=xpt_upper,
            xpt_lower=xpt_lower,
            tol=xpoint_tol,
        )

        if dn.get("ok", False):
            out = dict(fb)
            out["ok"] = True
            out["source"] = "freegs_psibndry_dn_reconstructed"
            out["reason"] = dn["reason"]
            out["R_sep"] = np.asarray(dn["R_lcfs"], dtype=float)
            out["Z_sep"] = np.asarray(dn["Z_lcfs"], dtype=float)

            geom = compute_lcfs_geometry(out["R_sep"], out["Z_sep"])
            out["geometry"] = geom

            if geom.get("ok", False):
                out.update({
                    "R0": geom["R0"],
                    "Z0": geom["Z0"],
                    "a": geom["a"],
                    "A": geom["A"],
                    "kappa": geom["kappa"],
                    "delta_u": geom["delta_u"],
                    "delta_l": geom["delta_l"],
                    "delta_bar": geom["delta_bar"],
                    "area": geom["area"],
                    "Rmin": geom["Rmin"],
                    "Rmax": geom["Rmax"],
                    "Zmin": geom["Zmin"],
                    "Zmax": geom["Zmax"],
            })
            out["R_lcfs"] = np.asarray(dn["R_lcfs"], dtype=float)
            out["Z_lcfs"] = np.asarray(dn["Z_lcfs"], dtype=float)
            out["dn_reconstruction"] = dn
            out["has_freegs_psibndry_sep"] = True
            out["has_usable_sep"] = True
            out["has_true_sep"] = False
            out["xpt_upper_used"] = xpt_upper
            out["xpt_lower_used"] = xpt_lower

            if debug_plot:
                _plot_dn_reconstruction_debug(eq, out, debug_plot)

            return out

        fb["dn_reconstruction_failed"] = dn
        fb["xpt_upper_used"] = xpt_upper
        fb["xpt_lower_used"] = xpt_lower

    fb["has_freegs_psibndry_sep"] = True
    fb["has_usable_sep"] = True
    fb["has_true_sep"] = False

    if debug_plot:
        extract_freegs_psibndry_contours(eq, debug_plot=debug_plot)

    return fb
