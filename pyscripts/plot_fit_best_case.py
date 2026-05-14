#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_fit_best_case.py

Rebuild and plot the best case stored in a fit JSON file,
e.g. results/fit_simplified_dn_toposafe_best.json

What it does
------------
- Loads best currents and physics from a fit JSON.
- Rebuilds the equilibrium through star_equilibrium.build_equilibrium(...)
- Plots:
    * equilibrium map (if available)
    * CAD outer / inner wall
    * CAD AUTO plasma target (if present)
    * simplified DN target from target JSON (optional)
    * obtained separatrix / LCFS
    * detected X-points
    * magnetic axis
    * coil boxes
    * text summary panel

Recommended usage
-----------------
py .\plot_fit_best_case.py `
  --fit-json .\results\fit_simplified_dn_toposafe_best.json `
  --target-json .\results\targets\star_simplified_dn_target.json `
  --dxf .\cad\star_baseline.dxf `
  --save .\results\fit_best_case_plot.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


FAMILIES_ALL = ("CS", "PF1", "PF2", "PF3", "PF4", "PF5", "PF6")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _here() -> Path:
    return Path(__file__).resolve().parent


def _resolve_path_maybe(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None

    p = Path(str(path)).expanduser()

    if p.is_absolute():
        return str(p)

    p_cwd = (Path.cwd() / p).resolve()
    if p_cwd.exists():
        return str(p_cwd)

    p_script = (_here() / p).resolve()
    if p_script.exists():
        return str(p_script)

    return str(path)


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _currents_A_to_MA(curr_A: Dict[str, float]) -> Dict[str, float]:
    return {k: float(curr_A.get(k, 0.0)) / 1e6 for k in FAMILIES_ALL}


def _fmt_currents_MA(curr_A: Dict[str, float], keys: Optional[List[str]] = None) -> str:
    cm = _currents_A_to_MA(curr_A)
    use = keys if keys else list(FAMILIES_ALL)
    return ", ".join(f"{k}={cm.get(k, 0.0):+.3f}" for k in use)


def _extract_fit_currents_and_physics(j: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """
    Supports:
    - fit_simplified_dn_toposafe_best.json structure
    - run json structure
    """
    currents_A = None
    physics = None
    meta = {}

    if isinstance(j.get("best_currents_A", None), dict):
        currents_A = {k: float(j["best_currents_A"].get(k, 0.0)) for k in FAMILIES_ALL}
    elif isinstance(j.get("best_result", None), dict) and isinstance(j["best_result"].get("currents_A", None), dict):
        currents_A = {k: float(j["best_result"]["currents_A"].get(k, 0.0)) for k in FAMILIES_ALL}
    else:
        raise ValueError("Could not find best currents in fit JSON.")

    if isinstance(j.get("physics", None), dict):
        physics = {
            "Ip_A": _safe_float(j["physics"].get("Ip_A", 4.0e6), 4.0e6),
            "paxis_Pa": _safe_float(j["physics"].get("paxis_Pa", 2.0e3), 2.0e3),
            "fvac": _safe_float(j["physics"].get("fvac", 20.8), 20.8),
            "alpha_m": _safe_float(j["physics"].get("alpha_m", 1.5), 1.5),
            "alpha_n": _safe_float(j["physics"].get("alpha_n", 1.1), 1.1),
        }
    elif isinstance(j.get("best_result", None), dict) and isinstance(j["best_result"].get("physics", None), dict):
        physics = {
            "Ip_A": _safe_float(j["best_result"]["physics"].get("Ip_A", 4.0e6), 4.0e6),
            "paxis_Pa": _safe_float(j["best_result"]["physics"].get("paxis_Pa", 2.0e3), 2.0e3),
            "fvac": _safe_float(j["best_result"]["physics"].get("fvac", 20.8), 20.8),
            "alpha_m": _safe_float(j["best_result"]["physics"].get("alpha_m", 1.5), 1.5),
            "alpha_n": _safe_float(j["best_result"]["physics"].get("alpha_n", 1.1), 1.1),
        }
    else:
        physics = {
            "Ip_A": 4.0e6,
            "paxis_Pa": 2.0e3,
            "fvac": 20.8,
            "alpha_m": 1.5,
            "alpha_n": 1.1,
        }

    meta["best_score"] = _safe_float(j.get("best_score", np.nan))
    meta["stage"] = j.get("stage", None)
    meta["score_info"] = {}
    if isinstance(j.get("best_result", None), dict) and isinstance(j["best_result"].get("score_info", None), dict):
        meta["score_info"] = j["best_result"]["score_info"]

    return currents_A, physics, meta


def _extract_lcfs_curve(shape: Dict[str, Any], diag: Dict[str, Any]) -> Optional[np.ndarray]:
    key_pairs = [
        ("R_sep", "Z_sep"),
        ("R_separatrix", "Z_separatrix"),
        ("R_lcfs", "Z_lcfs"),
        ("lcfs_R", "lcfs_Z"),
        ("R_LCFS", "Z_LCFS"),
    ]

    for rk, zk in key_pairs:
        if rk in shape and zk in shape:
            try:
                R = np.asarray(shape[rk], float)
                Z = np.asarray(shape[zk], float)
                if R.size >= 20 and Z.size >= 20 and R.size == Z.size:
                    P = np.column_stack([R, Z])
                    return P
            except Exception:
                pass

    if isinstance(diag, dict):
        try:
            R = np.asarray(diag.get("R_lcfs", []), float)
            Z = np.asarray(diag.get("Z_lcfs", []), float)
            if R.size >= 20 and Z.size >= 20 and R.size == Z.size:
                return np.column_stack([R, Z])
        except Exception:
            pass

    fb = shape.get("fallback_lcfs", None)
    if isinstance(fb, dict):
        try:
            R = np.asarray(fb.get("R", []), float)
            Z = np.asarray(fb.get("Z", []), float)
            if R.size >= 20 and Z.size >= 20 and R.size == Z.size:
                return np.column_stack([R, Z])
        except Exception:
            pass

    return None


def _extract_xpoints(shape: Dict[str, Any]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []

    xpv = shape.get("xpoints_valid", None)
    if isinstance(xpv, list) and xpv:
        src = xpv
    else:
        src = shape.get("xpoints", None) or shape.get("xpoint", None) or []

    if isinstance(src, dict):
        src = list(src.values())

    if not isinstance(src, list):
        return out

    for item in src:
        if isinstance(item, dict):
            R = _safe_float(item.get("R", item.get("r", np.nan)))
            Z = _safe_float(item.get("Z", item.get("z", np.nan)))
            if np.isfinite(R) and np.isfinite(Z):
                out.append((R, Z))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            R = _safe_float(item[0], np.nan)
            Z = _safe_float(item[1], np.nan)
            if np.isfinite(R) and np.isfinite(Z):
                out.append((R, Z))

    return out


def _load_target_boundary(target_json: Optional[str]) -> Optional[np.ndarray]:
    if target_json is None:
        return None
    try:
        j = _load_json(target_json)
        b = j.get("boundary", {})
        R = np.asarray(b.get("R", []), float)
        Z = np.asarray(b.get("Z", []), float)
        if R.size >= 20 and Z.size >= 20 and R.size == Z.size:
            return np.column_stack([R, Z])
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------
# Main plotter
# ---------------------------------------------------------------------
def rebuild_case(
    fit_json: str,
    dxf_path: Optional[str],
) -> Tuple[Any, Any, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, float], Dict[str, float]]:
    """
    Returns:
        eq, tokamak, geom, shape, meta, currents_A, physics
    """
    fit_json = _resolve_path_maybe(fit_json)
    dxf_path = _resolve_path_maybe(dxf_path)

    j = _load_json(fit_json)
    currents_A, physics, meta = _extract_fit_currents_and_physics(j)

    import config_star_bean as cfg
    import star_equilibrium as se

    # Apply physics
    cfg.Ip = float(physics["Ip_A"])
    cfg.paxis = float(physics["paxis_Pa"])
    cfg.fvac = float(physics["fvac"])
    cfg.alpha_m = float(physics["alpha_m"])
    cfg.alpha_n = float(physics["alpha_n"])
    cfg.vacuum_only = False

    # Apply currents
    for k, v in currents_A.items():
        try:
            setattr(cfg, f"{k}_current", float(v))
        except Exception:
            pass

    eq, tokamak, geom, shape = se.build_equilibrium(
        verbose=False,
        redirect_solver_noise=True,
        dxf_path=dxf_path,
    )

    diag = shape.get("plasma_diag", None) or {}
    if (not isinstance(diag, dict)) or (not diag.get("ok", False)):
        try:
            diag = se.plasma_diagnostics(eq, geom, shape)
        except Exception:
            diag = {"ok": False, "reason": "plasma_diagnostics_failed"}

    meta["diag"] = diag

    return eq, tokamak, geom, shape, meta, currents_A, physics


def plot_case(
    *,
    eq: Any,
    geom: Dict[str, Any],
    shape: Dict[str, Any],
    meta: Dict[str, Any],
    currents_A: Dict[str, float],
    physics: Dict[str, float],
    target_boundary: Optional[np.ndarray],
    save_path: Optional[str],
    title: str,
    show: bool,
) -> None:
    diag = meta.get("diag", {}) if isinstance(meta.get("diag", {}), dict) else {}
    score_info = meta.get("score_info", {}) if isinstance(meta.get("score_info", {}), dict) else {}

    fig, ax = plt.subplots(figsize=(10, 12))

    # ----------------------------------------------------------
    # Equilibrium map
    # ----------------------------------------------------------
    try:
        eq.plot(axis=ax, show=False)
    except Exception:
        # fallback: no eq.plot available or failed
        pass

    # ----------------------------------------------------------
    # CAD walls
    # ----------------------------------------------------------
    if "R_outer" in geom and "Z_outer" in geom:
        ax.plot(
            np.asarray(geom["R_outer"], float),
            np.asarray(geom["Z_outer"], float),
            "k-",
            lw=2.2,
            label="CAD outer wall",
        )

    if "R_inner" in geom and "Z_inner" in geom:
        ax.plot(
            np.asarray(geom["R_inner"], float),
            np.asarray(geom["Z_inner"], float),
            "k--",
            lw=1.8,
            label="CAD inner wall",
        )

    # ----------------------------------------------------------
    # CAD AUTO plasma target
    # ----------------------------------------------------------
    if "R_plasma" in geom and "Z_plasma" in geom:
        ax.plot(
            np.asarray(geom["R_plasma"], float),
            np.asarray(geom["Z_plasma"], float),
            lw=1.8,
            alpha=0.85,
            label="CAD AUTO plasma target",
        )

    # ----------------------------------------------------------
    # Simplified target
    # ----------------------------------------------------------
    if target_boundary is not None:
        ax.plot(
            target_boundary[:, 0],
            target_boundary[:, 1],
            lw=2.2,
            alpha=0.95,
            label="Simplified DN target",
        )

    # ----------------------------------------------------------
    # Separatrix / LCFS from reconstructed case
    # ----------------------------------------------------------
    lcfs = _extract_lcfs_curve(shape, diag)
    if lcfs is not None:
        ax.plot(
            lcfs[:, 0],
            lcfs[:, 1],
            color="tab:red",
            lw=2.6,
            label="Rebuilt best separatrix",
        )

    # ----------------------------------------------------------
    # X-points
    # ----------------------------------------------------------
    xps = _extract_xpoints(shape)
    if xps:
        xr = [p[0] for p in xps]
        zr = [p[1] for p in xps]
        ax.plot(
            xr,
            zr,
            linestyle="None",
            marker="x",
            markersize=10,
            markeredgewidth=2,
            label="Detected X-points",
        )

    # ----------------------------------------------------------
    # Magnetic axis
    # ----------------------------------------------------------
    Rax = _safe_float(diag.get("R_ax", shape.get("R_ax", np.nan)))
    Zax = _safe_float(diag.get("Z_ax", shape.get("Z_ax", np.nan)))
    if np.isfinite(Rax) and np.isfinite(Zax):
        ax.plot(
            [Rax],
            [Zax],
            linestyle="None",
            marker="o",
            markersize=8,
            label="Magnetic axis",
        )

    # ----------------------------------------------------------
    # Coils
    # ----------------------------------------------------------
    if isinstance(geom.get("coils", None), dict):
        for name, box in geom["coils"].items():
            try:
                Rc, Zc, dR, dZ = [float(x) for x in box]
                x0, x1 = Rc - dR, Rc + dR
                z0, z1 = Zc - dZ, Zc + dZ
                ax.plot(
                    [x0, x1, x1, x0, x0],
                    [z0, z0, z1, z1, z0],
                    "k-",
                    lw=0.8,
                    alpha=0.8,
                )
                ax.text(
                    Rc,
                    Zc,
                    str(name).upper(),
                    ha="center",
                    va="center",
                    fontsize=7,
                )
            except Exception:
                pass

    # ----------------------------------------------------------
    # Axes formatting
    # ----------------------------------------------------------
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(title)

    # ----------------------------------------------------------
    # Summary text
    # ----------------------------------------------------------
    best_score = _safe_float(meta.get("best_score", np.nan))
    stage = meta.get("stage", None)

    A = _safe_float(score_info.get("A", diag.get("A", np.nan)))
    kap = _safe_float(score_info.get("kappa", diag.get("kappa", np.nan)))
    du = _safe_float(score_info.get("delta_u", diag.get("delta_u", np.nan)))
    dl = _safe_float(score_info.get("delta_l", diag.get("delta_l", np.nan)))
    dbar = _safe_float(score_info.get("delta_bar", np.nan))
    chamfer = _safe_float(score_info.get("boundary_chamfer_m", np.nan))
    nxp = score_info.get("n_xpoints", len(xps))
    has_sep = score_info.get("has_true_sep", bool(shape.get("ok_sep", False)))
    reason = score_info.get("shape_reason", shape.get("reason", None))

    curr_MA = _currents_A_to_MA(currents_A)

    summary_lines = [
        f"best_score = {best_score:.3f}" if np.isfinite(best_score) else "best_score = n/a",
        f"stage = {stage}",
        "",
        f"has_true_sep = {has_sep}",
        f"shape_reason = {reason}",
        f"n_xpoints = {nxp}",
        "",
        f"Rax = {Rax:.3f} m" if np.isfinite(Rax) else "Rax = n/a",
        f"Zax = {Zax:.3f} m" if np.isfinite(Zax) else "Zax = n/a",
        f"A = {A:.3f}" if np.isfinite(A) else "A = n/a",
        f"kappa = {kap:.3f}" if np.isfinite(kap) else "kappa = n/a",
        f"delta_u = {du:.3f}" if np.isfinite(du) else "delta_u = n/a",
        f"delta_l = {dl:.3f}" if np.isfinite(dl) else "delta_l = n/a",
        f"delta_bar = {dbar:.3f}" if np.isfinite(dbar) else "delta_bar = n/a",
        f"boundary_chamfer = {chamfer:.3f} m" if np.isfinite(chamfer) else "boundary_chamfer = n/a",
        "",
        f"Ip = {physics['Ip_A']/1e6:.3f} MA",
        f"paxis = {physics['paxis_Pa']:.3e} Pa",
        f"fvac = {physics['fvac']:.3f}",
        f"alpha_m = {physics['alpha_m']:.3f}",
        f"alpha_n = {physics['alpha_n']:.3f}",
        "",
        "Currents [MA]:",
        f"CS  = {curr_MA['CS']:+.3f}",
        f"PF1 = {curr_MA['PF1']:+.3f}",
        f"PF2 = {curr_MA['PF2']:+.3f}",
        f"PF3 = {curr_MA['PF3']:+.3f}",
        f"PF4 = {curr_MA['PF4']:+.3f}",
        f"PF5 = {curr_MA['PF5']:+.3f}",
        f"PF6 = {curr_MA['PF6']:+.3f}",
    ]

    summary_text = "\n".join(summary_lines)

    ax.text(
        1.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    if save_path:
        save_path = _resolve_path_maybe(save_path)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
        print(f"[OK] Saved figure to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--fit-json",
        type=str,
        default=str(_here() / "results" / "fit_simplified_dn_toposafe_best.json"),
        help="Path to fit best JSON.",
    )
    ap.add_argument(
        "--target-json",
        type=str,
        default=str(_here() / "results" / "targets" / "star_simplified_dn_target.json"),
        help="Optional simplified target JSON.",
    )
    ap.add_argument(
        "--dxf",
        type=str,
        default=None,
        help="Optional DXF path. If omitted, star_equilibrium uses cfg/default behavior.",
    )
    ap.add_argument(
        "--save",
        type=str,
        default=str(_here() / "results" / "fit_best_case_plot.png"),
        help="Output PNG path.",
    )
    ap.add_argument(
        "--title",
        type=str,
        default="Rebuilt best fit case",
        help="Figure title.",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Show figure interactively.",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    fit_json = _resolve_path_maybe(args.fit_json)
    target_json = _resolve_path_maybe(args.target_json)
    dxf_path = _resolve_path_maybe(args.dxf)

    eq, tokamak, geom, shape, meta, currents_A, physics = rebuild_case(
        fit_json=fit_json,
        dxf_path=dxf_path,
    )

    target_boundary = _load_target_boundary(target_json)

    plot_case(
        eq=eq,
        geom=geom,
        shape=shape,
        meta=meta,
        currents_A=currents_A,
        physics=physics,
        target_boundary=target_boundary,
        save_path=args.save,
        title=args.title,
        show=bool(args.show),
    )


if __name__ == "__main__":
    main()
