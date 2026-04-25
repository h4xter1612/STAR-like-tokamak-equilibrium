#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inverse_star_native.py

Native inverse-equilibrium seed for STAR-like tokamak using FreeGSNKE's
Inverse_optimizer, driven by:

- null_points   <- CAD windows XPT_LOWER_WIN / XPT_UPPER_WIN
- isoflux_set   <- plasma target with MORE points near divertor windows

This is meant to replace the external finite-difference inverse seed as the
PRIMARY seed generator.

Outputs:
- ./results/inverse_star_native_best.json

Recommended usage:
  py .\inverse_star_native.py --null-mode double --fixed-keys CS,PF1 --free-keys PF2,PF3,PF4,PF5,PF6 --no-blanket
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from freegsnke import equilibrium_update, GSstaticsolver
from freegsnke.jtor_update import ConstrainPaxisIp
from freegsnke.inverse import Inverse_optimizer

from star_machine_cad import make_star_machine_from_cad, CADImportOptions, CADLayers, apply_group_currents
import config_star_bean as cfg


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def _results_dir() -> Path:
    d = Path(__file__).resolve().parent / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    def to_builtin(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, dict):
            return {str(k): to_builtin(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [to_builtin(v) for v in x]
        return x

    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(obj), f, indent=2)


# ---------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------
def _open_curve_from_closed(R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    P = np.column_stack([np.asarray(R, float), np.asarray(Z, float)])
    if P.shape[0] < 3:
        return P
    if np.linalg.norm(P[0] - P[-1]) < 1e-12:
        P = P[:-1]
    return P


def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    p = np.asarray(p, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ab = b - a
    den = float(np.dot(ab, ab))
    if den <= 1e-30:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / den)
    t = max(0.0, min(1.0, t))
    q = a + t * ab
    return float(np.linalg.norm(p - q))


def _point_to_polyline_distance(point: Tuple[float, float], poly: np.ndarray, closed: bool = False) -> float:
    P = np.asarray(poly, float)
    if P.ndim != 2 or P.shape[0] < 2:
        return float("inf")

    p = np.asarray(point, float)

    if bool(closed) and P.shape[0] >= 3:
        try:
            from matplotlib.path import Path as MplPath
            if MplPath(P, closed=True).contains_point((float(p[0]), float(p[1]))):
                return 0.0
        except Exception:
            pass

    dmin = float("inf")
    n = P.shape[0]
    m = n if closed else (n - 1)
    for i in range(m):
        a = P[i]
        b = P[(i + 1) % n]
        d = _point_to_segment_distance(p, a, b)
        if d < dmin:
            dmin = d
    return float(dmin)


def _extract_marker_window(geom: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    mw = geom.get("marker_windows", None)
    if not isinstance(mw, dict):
        return None
    pack = mw.get(key, None)
    if not isinstance(pack, dict):
        return None
    xy = pack.get("xy", None)
    if xy is None:
        return None
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or xy.shape[0] < 2 or xy.shape[1] != 2:
        return None
    return {
        "xy": xy,
        "closed": bool(pack.get("closed", False)),
        "length_m": float(pack.get("length_m", np.nan)),
    }


def _window_center(win: Dict[str, Any]) -> Tuple[float, float]:
    xy = np.asarray(win["xy"], float)
    c = np.mean(xy, axis=0)
    return float(c[0]), float(c[1])


def _curve_dist_to_window(P: np.ndarray, win: Dict[str, Any]) -> np.ndarray:
    xy = np.asarray(P, float)
    W = np.asarray(win["xy"], float)
    closed = bool(win.get("closed", False))
    d = np.empty(xy.shape[0], dtype=float)
    for i, p in enumerate(xy):
        d[i] = _point_to_polyline_distance((float(p[0]), float(p[1])), W, closed=closed)
    return d


def _unique_indices_keep_order(idxs: List[int]) -> List[int]:
    seen = set()
    out = []
    for i in idxs:
        ii = int(i)
        if ii not in seen:
            out.append(ii)
            seen.add(ii)
    return out


def _select_indices_near_window(
    P: np.ndarray,
    win: Optional[Dict[str, Any]],
    n_pick: int,
    min_sep_idx: int = 2,
) -> List[int]:
    if win is None or n_pick <= 0:
        return []

    d = _curve_dist_to_window(P, win)
    order = np.argsort(d)

    chosen: List[int] = []
    for idx in order:
        idx = int(idx)
        ok = True
        for j in chosen:
            if abs(idx - j) < int(min_sep_idx):
                ok = False
                break
        if ok:
            chosen.append(idx)
        if len(chosen) >= int(n_pick):
            break
    return chosen


def _select_uniform_indices(P: np.ndarray, n_pick: int) -> List[int]:
    if n_pick <= 0:
        return []
    n = P.shape[0]
    if n == 0:
        return []
    idx = np.linspace(0, n - 1, int(n_pick), endpoint=False)
    return [int(round(i)) for i in idx]


def _build_isoflux_from_target(
    geom: Dict[str, Any],
    *,
    n_strike_each: int = 8,
    n_xpt_each: int = 4,
    n_uniform: int = 8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build one isoflux set with more points near strike and x-point windows.
    Returns:
      isoflux_set: shape (1,2,N)
      meta: indices / counts / chosen points
    """
    if "R_plasma" not in geom or "Z_plasma" not in geom:
        raise ValueError("geom has no plasma target (R_plasma/Z_plasma).")

    P = _open_curve_from_closed(np.asarray(geom["R_plasma"], float), np.asarray(geom["Z_plasma"], float))
    if P.shape[0] < 20:
        raise ValueError("Plasma target has too few points.")

    win_xL = _extract_marker_window(geom, "xpt_lower")
    win_xU = _extract_marker_window(geom, "xpt_upper")
    win_sL = _extract_marker_window(geom, "strike_lower")
    win_sU = _extract_marker_window(geom, "strike_upper")

    idxs: List[int] = []
    idxs += _select_indices_near_window(P, win_sL, n_strike_each, min_sep_idx=2)
    idxs += _select_indices_near_window(P, win_sU, n_strike_each, min_sep_idx=2)
    idxs += _select_indices_near_window(P, win_xL, n_xpt_each, min_sep_idx=2)
    idxs += _select_indices_near_window(P, win_xU, n_xpt_each, min_sep_idx=2)
    idxs += _select_uniform_indices(P, n_uniform)

    idxs = _unique_indices_keep_order(sorted(idxs))
    Psel = P[idxs]

    isoflux_set = np.array([[
        Psel[:, 0].tolist(),
        Psel[:, 1].tolist(),
    ]], dtype=float)

    meta = {
        "n_total": int(Psel.shape[0]),
        "n_strike_each": int(n_strike_each),
        "n_xpt_each": int(n_xpt_each),
        "n_uniform": int(n_uniform),
        "indices": idxs,
        "points_RZ": Psel.tolist(),
    }
    return isoflux_set, meta


# ---------------------------------------------------------------------
# Tokamak / control helpers
# ---------------------------------------------------------------------
def _family_of_label(label: str) -> str:
    lab = str(label).strip().upper()
    if lab.startswith("CS"):
        return "CS"
    m = re.match(r"^(PF[0-9]+)", lab)
    if m:
        return m.group(1)
    return lab


def _set_control_flags(tokamak: Any, free_families: List[str], fixed_families: List[str]) -> Dict[str, List[str]]:
    """
    Sets coil.control on individual segment coils according to family membership.
    Returns a dict of {free_segments, fixed_segments, passive_segments}.
    """
    free_families = [str(k).strip().upper() for k in free_families]
    fixed_families = [str(k).strip().upper() for k in fixed_families]

    passive = set([str(k).strip().upper() for k in (getattr(tokamak, "passive_coils", []) or [])])

    free_segments: List[str] = []
    fixed_segments: List[str] = []
    passive_segments: List[str] = []

    coils_dict = getattr(tokamak, "coils_dict", {}) or {}
    for lab, coil in coils_dict.items():
        LAB = str(lab).strip().upper()
        fam = _family_of_label(LAB)

        is_passive = LAB in passive
        if is_passive:
            passive_segments.append(LAB)
            try:
                coil.control = False
            except Exception:
                pass
            continue

        if fam in free_families:
            free_segments.append(LAB)
            try:
                coil.control = True
            except Exception:
                pass
        else:
            fixed_segments.append(LAB)
            try:
                coil.control = False
            except Exception:
                pass

    return {
        "free_segments": free_segments,
        "fixed_segments": fixed_segments,
        "passive_segments": passive_segments,
    }


def _extract_individual_currents(tokamak: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    coils_dict = getattr(tokamak, "coils_dict", {}) or {}
    for lab, coil in coils_dict.items():
        try:
            out[str(lab).strip().upper()] = float(getattr(coil, "current"))
        except Exception:
            pass
    return out


def _extract_family_totals(tokamak: Any) -> Dict[str, float]:
    seg = _extract_individual_currents(tokamak)
    groups = getattr(tokamak, "coil_groups", {}) or {}
    if not groups:
        # fallback: infer by label
        out: Dict[str, float] = {}
        for lab, cur in seg.items():
            fam = _family_of_label(lab)
            out[fam] = out.get(fam, 0.0) + float(cur)
        return out

    out: Dict[str, float] = {}
    for fam, labs in groups.items():
        s = 0.0
        for lab in labs:
            LAB = str(lab).strip().upper()
            if LAB in seg:
                s += float(seg[LAB])
        out[str(fam).strip().upper()] = float(s)
    return out


# ---------------------------------------------------------------------
# Validation / plotting
# ---------------------------------------------------------------------
def _run_analysis(eq: Any, geom: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from analyze_star import analyze_star
        shape = analyze_star(
            eq, geom,
            require_two_x=False,
            null_prefer="lower",
            prefer_inner_lcfs=True,
            psi_percentile_lcfs=0.5,
            edge_pad_cells=2,
        )
        return shape
    except Exception as e:
        return {"ok_sep": False, "reason": f"analyze_failed:{repr(e)}"}


def _plot_inverse_setup(
    geom: Dict[str, Any],
    null_points: List[List[float]],
    isoflux_set: np.ndarray,
    outpath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 10))

    if "R_outer" in geom:
        ax.plot(geom["R_outer"], geom["Z_outer"], "k-", lw=1.5, label="Outer wall")
    if "R_inner" in geom:
        ax.plot(geom["R_inner"], geom["Z_inner"], "k--", lw=1.0, label="Inner wall")
    if "R_plasma" in geom:
        ax.plot(geom["R_plasma"], geom["Z_plasma"], color="tab:orange", lw=1.2, label="Plasma target")

    # marker windows
    mw = geom.get("marker_windows", {}) or {}
    for key, pack in mw.items():
        xy = np.asarray(pack["xy"], float)
        ax.plot(xy[:, 0], xy[:, 1], lw=2.8, label=key)

    # null points
    if len(null_points) == 2:
        ax.plot(null_points[0], null_points[1], "rx", ms=10, mew=2, label="Null points")

    # isoflux points
    try:
        Rf = np.asarray(isoflux_set[0, 0], float)
        Zf = np.asarray(isoflux_set[0, 1], float)
        ax.plot(Rf, Zf, "bo", ms=4, label="Isoflux points")
    except Exception:
        pass

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dxf", type=str, default=None)
    ap.add_argument("--null-mode", type=str, default="double", choices=["lower", "upper", "double"])

    ap.add_argument("--fixed-keys", type=str, default="CS,PF1")
    ap.add_argument("--free-keys", type=str, default="PF2,PF3,PF4,PF5,PF6")

    ap.add_argument("--n-strike-each", type=int, default=8)
    ap.add_argument("--n-xpt-each", type=int, default=4)
    ap.add_argument("--n-uniform", type=int, default=8)

    ap.add_argument("--no-blanket", action="store_true")
    ap.add_argument("--plot-setup", action="store_true")

    args = ap.parse_args()

    fixed_keys = [s.strip().upper() for s in str(args.fixed_keys).split(",") if s.strip()]
    free_keys = [s.strip().upper() for s in str(args.free_keys).split(",") if s.strip()]

    # --- Build CAD opts from cfg (matching your pipeline)
    opts = CADImportOptions(
        unit_scale=getattr(cfg, "unit_scale", None),
        resample_walls=str(getattr(cfg, "resample_walls", "always")),
        n_wall=int(getattr(cfg, "n_wall", 1601)),
        n_inner=int(getattr(cfg, "n_inner", 2001)),
        n_plasma=int(getattr(cfg, "n_plasma", 501)),
        min_wall_pts=int(getattr(cfg, "min_wall_pts", 200)),
        enforce_ccw=bool(getattr(cfg, "enforce_ccw", True)),
        canonical_start=bool(getattr(cfg, "canonical_start", True)),
        prefer_path_flattening=bool(getattr(cfg, "prefer_path_flattening", True)),
        flatten_distance=float(getattr(cfg, "flatten_distance", 0.002)),
        max_seg_len_wall=float(getattr(cfg, "max_seg_len_wall", 0.008)),
        max_seg_len_plasma=float(getattr(cfg, "max_seg_len_plasma", 0.008)),
        label_match_factor=float(getattr(cfg, "label_match_factor", 2.0)),

        plasma_target_mode=str(getattr(cfg, "plasma_target_mode", "auto")),
        plasma_fit_to_inner_if_available=bool(getattr(cfg, "plasma_fit_to_inner_if_available", True)),
        plasma_R0=float(getattr(cfg, "plasma_R0", 4.0)),
        plasma_A=float(getattr(cfg, "plasma_A", 2.0)),
        plasma_kappa=float(getattr(cfg, "plasma_kappa", 2.23)),
        plasma_Z0=float(getattr(cfg, "plasma_Z0", 0.0)),
        plasma_delta_max=float(getattr(cfg, "plasma_delta_max", 0.70)),
        plasma_delta_grid=int(getattr(cfg, "plasma_delta_grid", 17)),
        plasma_delta_symmetric=bool(getattr(cfg, "plasma_delta_symmetric", True)),
        plasma_shrink_iters=int(getattr(cfg, "plasma_shrink_iters", 20)),
        plasma_scale_safety=float(getattr(cfg, "plasma_scale_safety", 0.999)),
        containment_radius=float(getattr(cfg, "containment_radius", -1e-9)),
        fix_center_if_outside=bool(getattr(cfg, "fix_center_if_outside", True)),
        center_search_samples=int(getattr(cfg, "center_search_samples", 800)),
        center_search_seed=int(getattr(cfg, "center_search_seed", 0)),
        strike_ray_fallback_len=float(getattr(cfg, "strike_ray_fallback_len", 3.0)),

        blanket_enabled=(False if args.no_blanket else bool(getattr(cfg, "blanket_enabled", False))),
        blanket_n_filaments=(0 if args.no_blanket else int(getattr(cfg, "blanket_n_filaments", 0))),
        blanket_distribution=str(getattr(cfg, "blanket_distribution", "stratified")),
        blanket_seed=int(getattr(cfg, "blanket_seed", 0)),
        blanket_wall_margin_m=float(getattr(cfg, "blanket_wall_margin_m", 0.01)),
        blanket_filament_dR=float(getattr(cfg, "blanket_filament_dR", 0.004)),
        blanket_filament_dZ=float(getattr(cfg, "blanket_filament_dZ", 0.004)),
    )

    tokamak, geom = make_star_machine_from_cad(
        dxf_path=args.dxf,
        layers=CADLayers(),
        opts=opts,
        strict_expected=True,
    )

    # --- Set initial grouped currents from cfg
    init_family_currents = {
        "CS":  float(getattr(cfg, "CS_current", 0.0)),
        "PF1": float(getattr(cfg, "PF1_current", 0.0)),
        "PF2": float(getattr(cfg, "PF2_current", 0.0)),
        "PF3": float(getattr(cfg, "PF3_current", 0.0)),
        "PF4": float(getattr(cfg, "PF4_current", 0.0)),
        "PF5": float(getattr(cfg, "PF5_current", 0.0)),
        "PF6": float(getattr(cfg, "PF6_current", 0.0)),
    }
    apply_group_currents(tokamak, init_family_currents, mode=str(getattr(cfg, "coil_group_mode", "area")))

    # --- Set control flags: free families True, fixed False
    control_meta = _set_control_flags(tokamak, free_keys, fixed_keys)

    # --- Domain
    R_outer = np.asarray(geom["R_outer"], float)
    Z_outer = np.asarray(geom["Z_outer"], float)
    margin = float(getattr(cfg, "margin_RZ", 0.5))

    Rmin_raw = float(R_outer.min() - margin)
    Rmin = max(0.05, Rmin_raw)
    Rmax = float(R_outer.max() + margin)
    Zmin = float(Z_outer.min() - margin)
    Zmax = float(Z_outer.max() + margin)

    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=Rmin, Rmax=Rmax,
        Zmin=Zmin, Zmax=Zmax,
        nx=int(getattr(cfg, "nx_eq", 129)),
        ny=int(getattr(cfg, "ny_eq", 257)),
    )

    # --- Profiles
    Ip = float(getattr(cfg, "Ip", 13.6e6))
    paxis = float(getattr(cfg, "paxis", 8.0e5))
    fvac = float(getattr(cfg, "fvac", 20.8))
    alpha_m = float(getattr(cfg, "alpha_m", 1.8))
    alpha_n = float(getattr(cfg, "alpha_n", 1.2))

    profiles = ConstrainPaxisIp(
        eq=eq,
        paxis=paxis,
        Ip=Ip,
        fvac=fvac,
        alpha_m=alpha_m,
        alpha_n=alpha_n,
    )

    # --- Null points from marker windows
    win_xL = _extract_marker_window(geom, "xpt_lower")
    win_xU = _extract_marker_window(geom, "xpt_upper")

    if win_xL is None:
        raise ValueError("Missing marker window: xpt_lower")
    if str(args.null_mode).lower() == "double" and win_xU is None:
        raise ValueError("Missing marker window: xpt_upper (required for double-null)")

    xL = _window_center(win_xL)
    if win_xU is not None:
        xU = _window_center(win_xU)
    else:
        # reflect if only lower exists and user wants single-null lower
        xU = (xL[0], -xL[1])

    null_mode = str(args.null_mode).lower().strip()
    if null_mode == "lower":
        null_points = [[xL[0]], [xL[1]]]
    elif null_mode == "upper":
        null_points = [[xU[0]], [xU[1]]]
    else:
        null_points = [[xL[0], xU[0]], [xL[1], xU[1]]]

    # --- Isoflux set with MORE points near divertor
    isoflux_set, iso_meta = _build_isoflux_from_target(
        geom,
        n_strike_each=int(args.n_strike_each),
        n_xpt_each=int(args.n_xpt_each),
        n_uniform=int(args.n_uniform),
    )

    # Optional diagnostic plot of inverse targets
    if args.plot_setup:
        _plot_inverse_setup(
            geom=geom,
            null_points=null_points,
            isoflux_set=isoflux_set,
            outpath=_results_dir() / "inverse_star_native_setup.png",
        )

    # --- Build inverse constrain
    constrain = Inverse_optimizer(
        null_points=null_points,
        isoflux_set=isoflux_set,
    )

    # --- Solve
    solver = GSstaticsolver.NKGSsolver(eq)

    n_control = len(control_meta["free_segments"])
    l2_reg = np.array([1e-12] * max(1, n_control), dtype=float)

    solved_ok = False
    solve_error = None

    try:
        solver.solve(
            eq=eq,
            profiles=profiles,
            constrain=constrain,
            target_relative_tolerance=float(getattr(cfg, "target_rel_tol", 2e-5)),
            target_relative_psit_update=float(getattr(cfg, "target_rel_tol_ramp", 2e-5)),
            max_solving_iterations=100,
            max_iter_per_update=10,
            step_size=1.15,
            max_rel_update_size=0.10,
            full_jacobian_handover=[1e-2, 2e-1],
            forward_tolerance_increase=15,
            verbose=True,
            l2_reg=l2_reg,
        )
        solved_ok = True
    except TypeError:
        # fallback if your installed FreeGSNKE exposes a smaller solve signature
        try:
            solver.solve(
                eq=eq,
                profiles=profiles,
                constrain=constrain,
                target_relative_tolerance=float(getattr(cfg, "target_rel_tol", 2e-5)),
                verbose=True,
            )
            solved_ok = True
        except Exception as e:
            solve_error = repr(e)
    except Exception as e:
        solve_error = repr(e)

    # --- Postprocess
    shape = _run_analysis(eq, geom)
    individual = _extract_individual_currents(tokamak)
    family_totals = _extract_family_totals(tokamak)

    out = {
        "solved_ok": bool(solved_ok),
        "solve_error": solve_error,

        "null_mode": null_mode,
        "null_points": null_points,
        "isoflux_meta": iso_meta,

        "control_meta": control_meta,

        "physical_inputs": {
            "Ip_A": Ip,
            "paxis_Pa": paxis,
            "fvac": fvac,
            "alpha_m": alpha_m,
            "alpha_n": alpha_n,
        },

        "domain": {
            "Rmin": Rmin, "Rmax": Rmax,
            "Zmin": Zmin, "Zmax": Zmax,
            "nx": int(getattr(cfg, "nx_eq", 129)),
            "ny": int(getattr(cfg, "ny_eq", 257)),
        },

        "currents_individual_A": individual,
        "currents_family_A": family_totals,
        "currents_family_MA": {k: float(v / 1e6) for k, v in family_totals.items()},

        "analysis": shape,
    }

    outpath = _results_dir() / "inverse_star_native_best.json"
    _save_json(outpath, out)

    print("\n[DONE] inverse_star_native")
    print("solved_ok =", solved_ok)
    if solve_error:
        print("solve_error =", solve_error)

    print("\nFamily currents [MA]:")
    for fam in sorted(family_totals.keys()):
        print(f"  {fam:>4s}: {family_totals[fam] / 1e6:+.6f} MA")

    print("\nAnalysis:")
    print("  reason       =", shape.get("reason", "n/a"))
    print("  ok_sep       =", shape.get("ok_sep", "n/a"))
    print("  n_xpoints    =", len(shape.get("xpoints", []) or []))

    print("\n--- Paste into config_star_bean.py ---")
    for fam in ("CS", "PF1", "PF2", "PF3", "PF4", "PF5", "PF6"):
        if fam in family_totals:
            print(f"{fam}_current = {family_totals[fam]:.12g}")

    print(f"\n[SAVED] {outpath}")


if __name__ == "__main__":
    main()
