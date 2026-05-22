#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_simplified_dn_target.py

Build a simplified STAR-like double-null target for topology-safe equilibrium fitting.

Purpose
-------
This script DOES NOT solve Grad-Shafranov.

It only creates a clean, idealized target file that later scripts can use as a
physics/geometry reference before reintroducing full CAD constraints.

Main output:
    results/targets/star_simplified_dn_target.json

Optional plot:
    results/targets/star_simplified_dn_target.png

Target model
------------
Uses a Miller-like boundary:

    R(theta) = R0 + a cos(theta + delta(theta) sin(theta))
    Z(theta) = Z0 + kappa a sin(theta)

with separate upper/lower triangularity if desired.

Default values are read from config_star_bean.py when available:
    R0_geom, A_geom, kappa_geom, delta_geom

Recommended first run:
    py .\\make_simplified_dn_target.py --plot

More explicit:
    py .\\make_simplified_dn_target.py --R0 4.0 --A 2.0 --kappa 2.23 --delta-u 0.63 --delta-l 0.61 --plot

py .\make_simplified_dn_target.py --plot --overlay-cad --dxf .\cad\star_baseline.dxf

"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
def _here() -> Path:
    return Path(__file__).resolve().parent


def _results_dir() -> Path:
    d = _here() / "results" / "targets"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------
# JSON-safe helpers
# ---------------------------------------------------------------------
def _to_builtin(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]
    return obj


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(obj), f, indent=2)


# ---------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------
def _cfg_float(name: str, default: float) -> float:
    try:
        import config_star_bean as cfg

        return float(getattr(cfg, name))
    except Exception:
        return float(default)


def _cfg_dict(name: str, default: Dict[str, float]) -> Dict[str, float]:
    try:
        import config_star_bean as cfg

        d = getattr(cfg, name)
        if isinstance(d, dict) and d:
            return {str(k).upper(): float(v) for k, v in d.items()}
    except Exception:
        pass
    return dict(default)


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------
def _ensure_closed(R: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R = np.asarray(R, float)
    Z = np.asarray(Z, float)

    if R.size < 2:
        return R, Z

    if math.hypot(float(R[0] - R[-1]), float(Z[0] - Z[-1])) > 1e-12:
        R = np.r_[R, R[0]]
        Z = np.r_[Z, Z[0]]

    return R, Z


def _polygon_area(R: np.ndarray, Z: np.ndarray) -> float:
    R, Z = _ensure_closed(np.asarray(R, float), np.asarray(Z, float))
    if R.size < 4:
        return 0.0
    return 0.5 * float(np.sum(R[:-1] * Z[1:] - R[1:] * Z[:-1]))


def _arc_length_resample_closed(R: np.ndarray, Z: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    R, Z = _ensure_closed(R, Z)

    # Drop duplicate endpoint for periodic interpolation.
    R0 = R[:-1]
    Z0 = Z[:-1]

    if R0.size < 4:
        return _ensure_closed(R0, Z0)

    Rp = np.r_[R0, R0[0]]
    Zp = np.r_[Z0, Z0[0]]

    dR = np.diff(Rp)
    dZ = np.diff(Zp)
    ds = np.hypot(dR, dZ)
    s = np.r_[0.0, np.cumsum(ds)]
    total = float(s[-1])

    if total <= 0.0:
        return _ensure_closed(R0, Z0)

    s_new = np.linspace(0.0, total, int(n), endpoint=False)
    R_new = np.interp(s_new, s, Rp)
    Z_new = np.interp(s_new, s, Zp)

    return _ensure_closed(R_new, Z_new)


def make_miller_boundary(
    *,
    R0: float,
    Z0: float,
    A: float,
    kappa: float,
    delta_u: float,
    delta_l: float,
    n: int,
    resample: bool = True,
) -> Dict[str, Any]:
    """
    Generate a closed Miller-like plasma boundary.

    The delta value is smoothly switched between upper and lower half-plane.
    This avoids a discontinuity at Z=0 when delta_u != delta_l.
    """
    R0 = float(R0)
    Z0 = float(Z0)
    A = float(A)
    kappa = float(kappa)
    delta_u = float(delta_u)
    delta_l = float(delta_l)

    if A <= 0:
        raise ValueError("A must be positive.")
    if kappa <= 0:
        raise ValueError("kappa must be positive.")

    a = R0 / A

    theta = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)

    # Smooth transition:
    # sin(theta) > 0 => upper
    # sin(theta) < 0 => lower
    s = np.sin(theta)
    w_upper = 0.5 * (1.0 + np.tanh(8.0 * s))
    delta = w_upper * delta_u + (1.0 - w_upper) * delta_l

    R = R0 + a * np.cos(theta + delta * np.sin(theta))
    Z = Z0 + kappa * a * np.sin(theta)

    R, Z = _ensure_closed(R, Z)

    # Enforce CCW orientation.
    if _polygon_area(R, Z) < 0:
        R = R[::-1].copy()
        Z = Z[::-1].copy()
        R, Z = _ensure_closed(R, Z)

    if resample:
        R, Z = _arc_length_resample_closed(R, Z, n)

    R_min = float(np.min(R))
    R_max = float(np.max(R))
    Z_min = float(np.min(Z))
    Z_max = float(np.max(Z))

    a_eff = 0.5 * (R_max - R_min)
    R0_eff = 0.5 * (R_max + R_min)
    kappa_eff = 0.5 * (Z_max - Z_min) / max(a_eff, 1e-30)
    A_eff = R0_eff / max(a_eff, 1e-30)

    i_top = int(np.argmax(Z))
    i_bot = int(np.argmin(Z))
    delta_u_eff = (R0_eff - float(R[i_top])) / max(a_eff, 1e-30)
    delta_l_eff = (R0_eff - float(R[i_bot])) / max(a_eff, 1e-30)

    return {
        "R": R,
        "Z": Z,
        "theta": theta,
        "input_scalars": {
            "R0": R0,
            "Z0": Z0,
            "A": A,
            "a": a,
            "kappa": kappa,
            "delta_u": delta_u,
            "delta_l": delta_l,
        },
        "measured_scalars": {
            "R_min": R_min,
            "R_max": R_max,
            "Z_min": Z_min,
            "Z_max": Z_max,
            "R0": R0_eff,
            "a": a_eff,
            "A": A_eff,
            "kappa": kappa_eff,
            "delta_u": delta_u_eff,
            "delta_l": delta_l_eff,
            "area_m2": abs(_polygon_area(R, Z)),
        },
    }


def make_dn_markers(
    *,
    R0: float,
    Z0: float,
    A: float,
    kappa: float,
    delta_u: float,
    delta_l: float,
    xpoint_z_factor: float,
    xpoint_r_shift: float,
) -> Dict[str, Any]:
    """
    Approximate double-null X-point targets.

    These are intentionally simple geometric markers, not a claim that the
    Miller curve itself is a mathematically exact separatrix with cusps.

    For a STAR-like target:
      R_x ~ R0 - delta * a + shift
      Z_x ~ +/- xpoint_z_factor * kappa * a
    """
    a = float(R0) / float(A)

    R_xu = float(R0 - delta_u * a + xpoint_r_shift)
    R_xl = float(R0 - delta_l * a + xpoint_r_shift)

    Z_xu = float(Z0 + xpoint_z_factor * kappa * a)
    Z_xl = float(Z0 - xpoint_z_factor * kappa * a)

    return {
        "upper": {
            "R": R_xu,
            "Z": Z_xu,
            "kind": "upper",
        },
        "lower": {
            "R": R_xl,
            "Z": Z_xl,
            "kind": "lower",
        },
    }


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
def _resolve_dxf_for_cad_loader(dxf_path: str | None) -> str | None:
    """
    Robust DXF resolver for make_star_machine_from_cad().

    Important:
    star_machine_cad.make_star_machine_from_cad() interprets relative paths
    relative to ./cad. Therefore, if user passes ./cad/star_baseline.dxf,
    we convert it to an absolute path to avoid ./cad/cad/star_baseline.dxf.
    """
    if dxf_path is None:
        return None

    p = Path(str(dxf_path)).expanduser()

    if p.is_absolute():
        return str(p)

    # First interpret relative to current working directory.
    p_cwd = (Path.cwd() / p).resolve()
    if p_cwd.exists():
        return str(p_cwd)

    # Then interpret relative to script directory.
    p_script = (_here() / p).resolve()
    if p_script.exists():
        return str(p_script)

    # Last fallback: pass as-is; star_machine_cad may resolve it relative to ./cad.
    return str(dxf_path)

def load_marker_windows_from_cad(
    *,
    dxf_path: str | None,
    target_scalars: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Load marker windows from CAD and return JSON-safe marker_windows.

    This is used to export CAD marker windows into the target JSON.
    The plotting function already reads these windows from geom, but unless
    we explicitly copy them into target["marker_windows"], the optimizer cannot
    see them later.
    """
    if dxf_path is None:
        return {}

    try:
        from star_machine_cad import make_star_machine_from_cad, CADImportOptions, CADLayers

        dxf_resolved = _resolve_dxf_for_cad_loader(dxf_path)

        opts = CADImportOptions(
            unit_scale=None,
            resample_walls="always",
            n_wall=1601,
            n_inner=2001,
            n_plasma=501,
            min_wall_pts=400,
            enforce_ccw=True,
            canonical_start=True,
            prefer_path_flattening=True,
            flatten_distance=0.002,
            max_seg_len_wall=0.008,
            max_seg_len_plasma=0.008,
            plasma_target_mode="auto",
            plasma_fit_to_inner_if_available=True,
            plasma_R0=float(target_scalars["R0"]),
            plasma_A=float(target_scalars["A"]),
            plasma_kappa=float(target_scalars["kappa"]),
            plasma_Z0=float(target_scalars["Z0"]),
            plasma_delta_max=0.70,
            plasma_delta_grid=17,
            plasma_delta_symmetric=True,
            plasma_shrink_iters=20,
            plasma_scale_safety=0.999,
            containment_radius=-1e-9,
            fix_center_if_outside=True,
            center_search_samples=800,
            center_search_seed=0,
            strike_ray_fallback_len=3.0,
            blanket_enabled=False,
            blanket_n_filaments=0,
        )

        _, geom = make_star_machine_from_cad(
            dxf_path=dxf_resolved,
            layers=CADLayers(),
            opts=opts,
            strict_expected=False,
        )

        marker_windows = {}

        if isinstance(geom.get("marker_windows", None), dict):
            for name, pack in geom["marker_windows"].items():
                try:
                    xy = np.asarray(pack.get("xy", None), float)
                    if xy.ndim == 2 and xy.shape[0] >= 2:
                        key = str(name).upper()
                        marker_windows[key] = {
                            "xy": xy[:, :2].tolist(),
                            "closed": bool(pack.get("closed", False)),
                            "length_m": float(pack.get("length_m", 0.0)),
                        }
                except Exception:
                    pass

        return marker_windows

    except Exception as e:
        print(f"[WARN] Could not export CAD marker windows to target JSON: {repr(e)}")
        return {}

def plot_target(
    target: Dict[str, Any],
    path: Path,
    *,
    show_cad: bool = False,
    dxf_path: str | None = None,
    show_coils: bool = True,
    show_cad_auto_target: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    R = np.asarray(target["boundary"]["R"], float)
    Z = np.asarray(target["boundary"]["Z"], float)

    fig, ax = plt.subplots(figsize=(8, 10))

    if show_cad:
        try:
            from star_machine_cad import make_star_machine_from_cad, CADImportOptions, CADLayers

            dxf_resolved = _resolve_dxf_for_cad_loader(dxf_path)

            opts = CADImportOptions(
                unit_scale=None,
                resample_walls="always",
                n_wall=1601,
                n_inner=2001,
                n_plasma=501,
                min_wall_pts=400,
                enforce_ccw=True,
                canonical_start=True,
                prefer_path_flattening=True,
                flatten_distance=0.002,
                max_seg_len_wall=0.008,
                max_seg_len_plasma=0.008,
                plasma_target_mode="auto",
                plasma_fit_to_inner_if_available=True,
                plasma_R0=float(target["scalars_target"]["R0"]),
                plasma_A=float(target["scalars_target"]["A"]),
                plasma_kappa=float(target["scalars_target"]["kappa"]),
                plasma_Z0=float(target["scalars_target"]["Z0"]),
                plasma_delta_max=0.70,
                plasma_delta_grid=17,
                plasma_delta_symmetric=True,
                plasma_shrink_iters=20,
                plasma_scale_safety=0.999,
                containment_radius=-1e-9,
                fix_center_if_outside=True,
                center_search_samples=800,
                center_search_seed=0,
                strike_ray_fallback_len=3.0,
                blanket_enabled=False,
                blanket_n_filaments=0,
            )

            _, geom = make_star_machine_from_cad(
                dxf_path=dxf_resolved,
                layers=CADLayers(),
                opts=opts,
                strict_expected=False,
            )

            if "R_outer" in geom and "Z_outer" in geom:
                ax.plot(
                    np.asarray(geom["R_outer"], float),
                    np.asarray(geom["Z_outer"], float),
                    "k-",
                    lw=2.0,
                    label="CAD outer wall",
                )

            if "R_inner" in geom and "Z_inner" in geom:
                ax.plot(
                    np.asarray(geom["R_inner"], float),
                    np.asarray(geom["Z_inner"], float),
                    "k--",
                    lw=1.6,
                    label="CAD inner wall",
                )

            if show_cad_auto_target and "R_plasma" in geom and "Z_plasma" in geom:
                ax.plot(
                    np.asarray(geom["R_plasma"], float),
                    np.asarray(geom["Z_plasma"], float),
                    lw=1.6,
                    alpha=0.8,
                    label="CAD AUTO plasma target",
                )

            if show_coils and isinstance(geom.get("coils", None), dict):
                for name, box in geom["coils"].items():
                    try:
                        Rc, Zc, dR, dZ = [float(x) for x in box]
                        x0, x1 = Rc - dR, Rc + dR
                        z0, z1 = Zc - dZ, Zc + dZ
                        ax.plot(
                            [x0, x1, x1, x0, x0],
                            [z0, z0, z1, z1, z0],
                            "k-",
                            lw=0.9,
                            alpha=0.85,
                        )
                        # ax.text(
                        #     Rc,
                        #     Zc,
                        #     str(name).upper(),
                        #     ha="center",
                        #     va="center",
                        #     fontsize=7,
                        # )
                    except Exception:
                        pass

            if isinstance(geom.get("marker_windows", None), dict):
                for name, pack in geom["marker_windows"].items():
                    try:
                        xy = np.asarray(pack.get("xy", None), float)
                        if xy.ndim == 2 and xy.shape[0] >= 2:
                            ax.plot(
                                xy[:, 0],
                                xy[:, 1],
                                lw=2.5,
                                alpha=0.75,
                                label=f"WIN:{name}",
                            )
                    except Exception:
                        pass

            print("[OK] CAD overlay loaded.")
            print(f"     DXF: {geom.get('cad_path', dxf_resolved)}")

        except Exception as e:
            print(f"[WARN] Could not overlay CAD geometry: {repr(e)}")

    ax.plot(R, Z, lw=2.8, label="Simplified DN target")

    xpts = target["xpoints_target"]
    ax.plot(
        [xpts["upper"]["R"]],
        [xpts["upper"]["Z"]],
        marker="x",
        markersize=10,
        linestyle="None",
        label="upper X target",
    )
    ax.plot(
        [xpts["lower"]["R"]],
        [xpts["lower"]["Z"]],
        marker="x",
        markersize=10,
        linestyle="None",
        label="lower X target",
    )

    R0 = target["scalars_target"]["R0"]
    Z0 = target["scalars_target"]["Z0"]
    ax.plot([R0], [Z0], marker="o", linestyle="None", label="target center")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("Simplified STAR-like DN target vs CAD")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def build_target(args: argparse.Namespace) -> Dict[str, Any]:
    R0 = float(args.R0)
    Z0 = float(args.Z0)
    A = float(args.A)
    kappa = float(args.kappa)
    delta_u = float(args.delta_u)
    delta_l = float(args.delta_l)
    n = int(args.n)

    boundary = make_miller_boundary(
        R0=R0,
        Z0=Z0,
        A=A,
        kappa=kappa,
        delta_u=delta_u,
        delta_l=delta_l,
        n=n,
        resample=True,
    )

    xpts = make_dn_markers(
        R0=R0,
        Z0=Z0,
        A=A,
        kappa=kappa,
        delta_u=delta_u,
        delta_l=delta_l,
        xpoint_z_factor=float(args.xpoint_z_factor),
        xpoint_r_shift=float(args.xpoint_r_shift),
    )

    # Use cfg limits when available.
    default_imax = {
        "CS": 73.036e6,
        "PF1": 15.000e6,
        "PF2": 15.000e6,
        "PF3": 7.500e6,
        "PF4": 9.437e6,
        "PF5": 11.850e6,
        "PF6": 19.752e6,
    }
    imax_A = _cfg_dict("MAX_RECOMMENDED_FAMILY_CURRENTS_A", default_imax)

    operating_A = _cfg_dict(
        "OPERATING_FAMILY_CURRENT_LIMIT_A",
        {k: 0.35 * v for k, v in imax_A.items()},
    )

    a = R0 / A

    target = {
        "schema": "star_simplified_dn_target.v1",
        "description": (
            "Idealized STAR-like double-null target for topology-safe fitting. "
            "This is not a solved equilibrium and should not be treated as a CAD-final constraint."
        ),
        "boundary": {
            "type": "miller_like_near_separatrix",
            "R": boundary["R"],
            "Z": boundary["Z"],
            "units": "m",
            "closed": True,
            "n_points": int(len(boundary["R"])),
        },
        "scalars_target": {
            "R0": R0,
            "Z0": Z0,
            "A": A,
            "a": a,
            "kappa": kappa,
            "delta_u": delta_u,
            "delta_l": delta_l,
            "delta_bar": 0.5 * (delta_u + delta_l),
            "vertical_half_height": kappa * a,
        },
        "scalars_measured_from_boundary": boundary["measured_scalars"],
        "xpoints_target": xpts,
        "topology_requirements": {
            "require_true_separatrix": True,
            "reject_fallback_lcfs_as_success": True,
            "prefer_double_null": True,
            "min_valid_xpoints": 2,
            "require_axis_inside_main_plasma": True,
            "reject_domain_edge_touching_boundary": True,
            "reject_small_islands": True,
        },
        "shape_tolerances": {
            "sigma_R0_m": 0.20,
            "sigma_Z0_m": 0.15,
            "sigma_A": 0.20,
            "sigma_kappa": 0.18,
            "sigma_delta": 0.12,
            "sigma_boundary_chamfer_m": 0.18,
            "sigma_xpoint_m": 0.35,
        },
        "axis_window": {
            "R_min": R0 - 0.35,
            "R_max": R0 + 0.35,
            "Z_min": Z0 - 0.30,
            "Z_max": Z0 + 0.30,
        },
        "valid_plasma_window": {
            "R_min": R0 - 1.35 * a,
            "R_max": R0 + 1.35 * a,
            "Z_min": Z0 - 1.25 * kappa * a,
            "Z_max": Z0 + 1.25 * kappa * a,
            "min_area_m2": 0.35 * math.pi * a * kappa * a,
            "max_area_m2": 1.80 * math.pi * a * kappa * a,
        },
        "objective_weights": {
            "topology_fail": 1.0e8,
            "not_double_null": 2.0e5,
            "axis": 30.0,
            "shape_scalars": 20.0,
            "boundary_chamfer": 15.0,
            "xpoints": 10.0,
            "vertical_symmetry": 8.0,
            "current_regularization": 3.0,
            "cs_regularization_extra": 12.0,
            "current_step_regularization": 1.5,
        },
        "coil_regularization": {
            "imax_recommended_A": imax_A,
            "operating_limit_A": operating_A,
            "preferred_limit_mode": "operating",
            "cs_soft_fraction_of_recommended": 0.25,
            "pf_soft_fraction_of_recommended": 0.45,
            "hard_fraction_of_recommended": 0.80,
            "notes": (
                "The fit script should strongly discourage CS-dominated improvement. "
                "CS may be allowed, but not as the primary shape actuator."
            ),
        },
        "fit_schedule_hint": {
            "stage_1": {
                "name": "PF-only or CS-fixed topology recovery",
                "free_keys": ["PF2", "PF3", "PF4", "PF5", "PF6"],
                "fixed_or_heavily_penalized": ["CS", "PF1"],
            },
            "stage_2": {
                "name": "moderate CS release",
                "free_keys": ["PF1", "PF2", "PF3", "PF4", "PF5", "PF6"],
                "heavily_penalized": ["CS"],
            },
            "stage_3": {
                "name": "full but regularized local refine",
                "free_keys": ["CS", "PF1", "PF2", "PF3", "PF4", "PF5", "PF6"],
                "notes": "Only run if topology remains clean and CS does not dominate.",
            },
        },
        "provenance": {
            "script": "make_simplified_dn_target.py",
            "used_config_defaults": {
                "R0_geom": _cfg_float("R0_geom", 4.0),
                "A_geom": _cfg_float("A_geom", 2.0),
                "kappa_geom": _cfg_float("kappa_geom", 2.23),
                "delta_geom": _cfg_float("delta_geom", 0.62),
            },
        },
    }

    # Export CAD marker windows into target JSON.
    # These are required by fit_simplified_dn_divertor_first.py.
    marker_windows = load_marker_windows_from_cad(
        dxf_path=args.dxf,
        target_scalars=target["scalars_target"],
    )

    target["marker_windows"] = marker_windows

    if marker_windows:
        target["provenance"]["marker_windows_exported"] = sorted(marker_windows.keys())
    else:
        target["provenance"]["marker_windows_exported"] = []

    return target


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a simplified STAR-like double-null target JSON."
    )

    p.add_argument("--R0", type=float, default=_cfg_float("R0_geom", 4.0))
    p.add_argument("--Z0", type=float, default=0.0)
    p.add_argument("--A", type=float, default=_cfg_float("A_geom", 2.0))
    p.add_argument("--kappa", type=float, default=_cfg_float("kappa_geom", 2.23))

    # By default, use the config delta for both halves.
    d0 = _cfg_float("delta_geom", 0.62)
    p.add_argument("--delta-u", type=float, default=d0)
    p.add_argument("--delta-l", type=float, default=d0)

    p.add_argument("--n", type=int, default=720)

    # X-point targets are geometric hints, not exact boundary cusps.
    p.add_argument("--xpoint-z-factor", type=float, default=1.04)
    p.add_argument("--xpoint-r-shift", type=float, default=0.0)

    p.add_argument(
        "--out",
        type=str,
        default=str(_results_dir() / "star_simplified_dn_target.json"),
        help="Output JSON path.",
    )

    p.add_argument("--plot", action="store_true", help="Save a PNG plot of the target.")
    p.add_argument(
        "--plot-path",
        type=str,
        default=str(_results_dir() / "star_simplified_dn_target.png"),
        help="Output plot path.",
    )

    p.add_argument(
        "--overlay-cad",
        action="store_true",
        help="Try to overlay CAD walls using star_machine_cad.py.",
    )
    p.add_argument(
        "--dxf",
        type=str,
        default=None,
        help="Optional DXF path for CAD overlay. If omitted, star_machine_cad default behavior is used.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    target = build_target(args)

    out = Path(args.out).resolve()
    _save_json(out, target)

    print(f"[OK] Saved simplified DN target:")
    print(f"     {out}")

    s = target["scalars_target"]
    sm = target["scalars_measured_from_boundary"]
    xu = target["xpoints_target"]["upper"]
    xl = target["xpoints_target"]["lower"]

    print("\n=== Target scalars ===")
    print(f"  R0       = {s['R0']:.4f} m")
    print(f"  Z0       = {s['Z0']:.4f} m")
    print(f"  a        = {s['a']:.4f} m")
    print(f"  A        = {s['A']:.4f}")
    print(f"  kappa    = {s['kappa']:.4f}")
    print(f"  delta_u  = {s['delta_u']:.4f}")
    print(f"  delta_l  = {s['delta_l']:.4f}")

    print("\n=== Measured from generated boundary ===")
    print(f"  R0       = {sm['R0']:.4f} m")
    print(f"  a        = {sm['a']:.4f} m")
    print(f"  A        = {sm['A']:.4f}")
    print(f"  kappa    = {sm['kappa']:.4f}")
    print(f"  delta_u  = {sm['delta_u']:.4f}")
    print(f"  delta_l  = {sm['delta_l']:.4f}")
    print(f"  area     = {sm['area_m2']:.4f} m^2")

    print("\n=== Approximate X-point targets ===")
    print(f"  upper: R={xu['R']:.4f} m, Z={xu['Z']:.4f} m")
    print(f"  lower: R={xl['R']:.4f} m, Z={xl['Z']:.4f} m")

    if args.plot:
        plot_path = Path(args.plot_path).resolve()
        plot_target(
            target,
            plot_path,
            show_cad=bool(args.overlay_cad),
            dxf_path=args.dxf,
            show_coils=True,
            show_cad_auto_target=True,
        )
        print(f"\n[OK] Saved plot:")
        print(f"     {plot_path}")


if __name__ == "__main__":
    main()
