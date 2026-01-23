"""
star_equilibrium.py

Refined STAR-like equilibrium using parameters in config_star_bean.py,
with CAD/DXF-backed machine geometry (star_machine_cad.py).

Key points
- Supports segmented coils in CAD (e.g., CS1M, CS2U, CS2L, ...) by treating them
  as ONE family current (CS_current) distributed across segments.
- Distribution modes: "equal" or "area" (recommended). ("same" exists but is NOT
  recommended for segmented coils.)
- Keeps console output clean: solver chatter is redirected to a log file while
  progress prints remain visible.
- Robust mask handling to avoid FreeGSNKE copy() failures when masks are None.
"""

from __future__ import annotations

import os
import types
import contextlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from freegsnke import equilibrium_update, GSstaticsolver
from freegsnke.jtor_update import ConstrainPaxisIp

from star_machine_cad import make_star_machine_from_cad, CADImportOptions
from analyze_star_shape import shape_from_separatrix
import config_star_bean as cfg


# -------------------------
# Small utilities
# -------------------------

def _norm(s: str) -> str:
    return str(s).strip().upper()


def _results_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "results"


@contextlib.contextmanager
def redirect_stdout_stderr(to_path: Optional[str]):
    """
    Redirect stdout+stderr to:
      - os.devnull if to_path is None
      - a file (append) if to_path is a path

    This is used to suppress solver spam like:
      "Update resizing triggered due to failure to find a critical points."
    while keeping your own progress prints visible.
    """
    if to_path is None:
        f = open(os.devnull, "w")
    else:
        p = Path(to_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        f = open(p, "a", buffering=1, encoding="utf-8")  # line-buffered
    try:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            yield
    finally:
        try:
            f.close()
        except Exception:
            pass


# -------------------------
# Coil grouping + currents
# -------------------------

def build_coil_families(tokamak) -> Dict[str, List[str]]:
    """
    Build family -> [coil_label, ...] mapping.

    Preferred: if tokamak already has .coil_groups, use it.
    Fallback: infer from labels.
    """
    if hasattr(tokamak, "coil_groups") and tokamak.coil_groups:
        out = {}
        for fam, labs in tokamak.coil_groups.items():
            out[_norm(fam)] = [_norm(x) for x in labs]
        return out

    families: Dict[str, List[str]] = {"CS": [], "PF1": [], "PF2": [], "PF3": []}
    for label, _coil in getattr(tokamak, "coils", []):
        lab = _norm(label)

        if lab.startswith("CS"):                      # CS, CS1M, CS2U, ...
            families["CS"].append(lab)
        elif lab in ("PF1U", "PF1L"):
            families["PF1"].append(lab)
        elif lab in ("PF2U", "PF2L"):
            families["PF2"].append(lab)
        elif lab in ("PF3U", "PF3L"):
            families["PF3"].append(lab)

    # Remove empties
    return {k: v for k, v in families.items() if v}


def _coil_area_from_geom(geom: Dict, label: str) -> float:
    """
    Use CAD rectangle half-extents to build a weight proxy ~ (2 dR)*(2 dZ) ~ 4 dR dZ.
    Constant factors cancel; we use dR*dZ.
    """
    lab = _norm(label)
    if "coils" not in geom or lab not in {_norm(k) for k in geom["coils"].keys()}:
        return 1.0

    # geom["coils"] keys may already be normalized; handle both
    for k, (Rc, Zc, dR, dZ) in geom["coils"].items():
        if _norm(k) == lab:
            return float(abs(dR) * abs(dZ)) if (dR is not None and dZ is not None) else 1.0
    return 1.0


def apply_group_currents(
    tokamak,
    geom: Dict,
    family_currents: Dict[str, float],
    *,
    mode: str = "area",
):
    """
    Distribute ONE family current across family segments.

    mode:
      - "equal": split equally across segments (sum segment currents = family current)
      - "area" : split proportional to segment area proxy (dR*dZ) from CAD geom
      - "same" : each segment gets full family current (NOT recommended for segmented coils)
    """
    mode = str(mode).strip().lower()
    fams = build_coil_families(tokamak)

    # Build a map label->coil object
    coil_map = {_norm(lbl): coil for (lbl, coil) in getattr(tokamak, "coils", [])}

    # Zero everything first (safe baseline)
    for lbl, coil in getattr(tokamak, "coils", []):
        try:
            coil.current = 0.0
        except Exception:
            pass

    for fam, total_I in family_currents.items():
        famN = _norm(fam)
        if famN not in fams:
            continue
        labs = fams[famN]
        n = len(labs)
        if n == 0:
            continue

        if mode == "same":
            weights = np.ones(n, dtype=float)
        elif mode == "equal":
            weights = np.ones(n, dtype=float) / float(n)
        else:  # "area" default
            w = np.array([_coil_area_from_geom(geom, lab) for lab in labs], dtype=float)
            if not np.all(np.isfinite(w)) or float(np.sum(w)) <= 0.0:
                weights = np.ones(n, dtype=float) / float(n)
            else:
                weights = w / float(np.sum(w))

        for lab, wi in zip(labs, weights):
            c = coil_map.get(_norm(lab), None)
            if c is None:
                continue
            try:
                if mode == "same":
                    c.current = float(total_I)
                else:
                    c.current = float(wi * float(total_I))
            except Exception:
                pass


def currents_sanity_check(tokamak, family_targets: Dict[str, float]) -> str:
    """
    Return a multi-line sanity report: sum(segment currents) vs target total per family.
    """
    fams = build_coil_families(tokamak)
    coil_map = {_norm(lbl): coil for (lbl, coil) in getattr(tokamak, "coils", [])}

    lines = []
    lines.append("--- Coil currents sanity check ---")
    lines.append(f"Has coil_groups: {bool(getattr(tokamak, 'coil_groups', None))}")
    for fam in ("PF1", "PF2", "PF3", "CS"):
        if fam not in fams:
            continue
        s = 0.0
        for lab in fams[fam]:
            c = coil_map.get(_norm(lab), None)
            if c is None:
                continue
            try:
                s += float(getattr(c, "current", 0.0))
            except Exception:
                pass
        tgt = float(family_targets.get(fam, np.nan))
        lines.append(f" {fam:>3s}: sum(segment currents) = {s/1e6: .6f} MA | target total = {tgt/1e6: .6f} MA")
    return "\n".join(lines)


def set_star_currents(tokamak, geom: Dict, CS=None, PF1=None, PF2=None, PF3=None):
    """
    Family-aware setter. Uses cfg defaults when args are None.
    """
    if CS is None:
        CS = cfg.CS_current
    if PF1 is None:
        PF1 = cfg.PF1_current
    if PF2 is None:
        PF2 = cfg.PF2_current
    if PF3 is None:
        PF3 = cfg.PF3_current

    mode = str(getattr(cfg, "coil_group_mode", "area")).strip().lower()
    apply_group_currents(tokamak, geom, {"CS": CS, "PF1": PF1, "PF2": PF2, "PF3": PF3}, mode=mode)


# -------------------------
# Masks / profiles robustness
# -------------------------

def _full_core_mask(eq) -> np.ndarray:
    # Robust core mask matching current grid; avoid relying on eq.psi() shape.
    return np.ones(eq.R.shape, dtype=bool)


def _ensure_masks(profiles, eq):
    """
    Ensure any *_core_mask fields exist and match current grid.
    This prevents failures in profiles.copy() in some FreeGSNKE builds.
    """
    base = _full_core_mask(eq)

    for name in ("diverted_core_mask", "limiter_core_mask"):
        val = getattr(profiles, name, None)
        if val is None:
            try:
                setattr(profiles, name, base.copy())
            except Exception:
                pass
        else:
            try:
                arr = np.asarray(val)
                if arr.shape != base.shape:
                    setattr(profiles, name, base.copy())
            except Exception:
                try:
                    setattr(profiles, name, base.copy())
                except Exception:
                    pass

    # Future-proof: any attr ending with _core_mask that is None -> fill it
    for attr in dir(profiles):
        if not attr.endswith("_core_mask"):
            continue
        try:
            v = getattr(profiles, attr)
        except Exception:
            continue
        if v is None:
            try:
                setattr(profiles, attr, base.copy())
            except Exception:
                pass


def _make_profiles(eq, *, paxis, Ip, fvac, alpha_m, alpha_n):
    profiles = ConstrainPaxisIp(
        eq=eq,
        paxis=float(paxis),
        Ip=float(Ip),
        fvac=float(fvac),
        alpha_m=float(alpha_m),
        alpha_n=float(alpha_n),
    )

    _ensure_masks(profiles, eq)

    # Patch copy() defensively
    _orig_copy = profiles.copy

    def _safe_copy(self, *args, **kwargs):
        _ensure_masks(self, eq)
        return _orig_copy(*args, **kwargs)

    profiles.copy = types.MethodType(_safe_copy, profiles)
    return profiles


# -------------------------
# Build equilibrium
# -------------------------

def build_equilibrium(
    verbose: bool = True,
    *,
    dxf_path: str | None = None,
    unit_scale: float | None = None,      # None => infer from INSUNITS
    resample_walls: str = "auto",         # "auto" | "always" | "never"
    n_wall: int = 801,
    n_inner: int = 801,
    min_wall_pts: int = 200,
    enforce_ccw: bool = True,
    canonical_start: bool = True,
    silence_solver_noise: bool = True,
    solver_noise_log: str | None = None,  # if None and silence_solver_noise True -> defaults to results/solver_noise.log
):
    """
    Build a STAR-like equilibrium using ONLY parameters in config_star_bean.py,
    using CAD machine geometry from star_machine_cad.py.
    """

    # 1) Geometry and Machine (CAD)
    opts = CADImportOptions(
        unit_scale=unit_scale,
        resample_walls=str(resample_walls),
        n_wall=int(n_wall),
        n_inner=int(n_inner),
        min_wall_pts=int(min_wall_pts),
        enforce_ccw=bool(enforce_ccw),
        canonical_start=bool(canonical_start),
    )

    tokamak, geom = make_star_machine_from_cad(
        dxf_path=dxf_path,
        opts=opts,
        strict_expected=True,
    )

    # Set currents from cfg (family-aware)
    set_star_currents(tokamak, geom)

    if verbose:
        # family sanity at full current
        fam_targets = {"CS": cfg.CS_current, "PF1": cfg.PF1_current, "PF2": cfg.PF2_current, "PF3": cfg.PF3_current}
        print(currents_sanity_check(tokamak, fam_targets), flush=True)

        print("\n--- CAD machine loaded ---", flush=True)
        print(f"CAD path         = {geom.get('cad_path', '(unknown)')}", flush=True)
        print(f"unit_scale       = {geom.get('unit_scale', unit_scale)}", flush=True)
        print(f"resample_walls   = {resample_walls} | n_wall={n_wall} n_inner={n_inner} min_wall_pts={min_wall_pts}", flush=True)
        print(f"enforce_ccw      = {enforce_ccw} | canonical_start={canonical_start}", flush=True)
        print(f"Coils found      = {sorted(list(geom.get('coils', {}).keys()))}", flush=True)

        fams = build_coil_families(tokamak)
        fam_sizes = {k: len(v) for k, v in fams.items()}
        print(f"Coil families    = {fam_sizes}", flush=True)
        print(f"coil_group_mode  = {getattr(cfg, 'coil_group_mode', 'area')}", flush=True)

        print("\n--- Target STAR-like settings (cfg) ---", flush=True)
        print(f"R0_geom    = {cfg.R0_geom:.3f} m", flush=True)
        print(f"A_geom     = {cfg.A_geom:.3f}", flush=True)
        print(f"kappa_geom = {cfg.kappa_geom:.3f}", flush=True)
        print(f"delta_geom = {cfg.delta_geom:.3f}", flush=True)

        print("\n--- PF/CS currents (cfg totals) ---", flush=True)
        print(f"CS  = {cfg.CS_current/1e6:.3f} MA", flush=True)
        print(f"PF1 = {cfg.PF1_current/1e6:.3f} MA", flush=True)
        print(f"PF2 = {cfg.PF2_current/1e6:.3f} MA", flush=True)
        print(f"PF3 = {cfg.PF3_current/1e6:.3f} MA", flush=True)

    # 2) Numerical domain (R, Z)
    R_outer = np.asarray(geom["R_outer"], dtype=float)
    Z_outer = np.asarray(geom["Z_outer"], dtype=float)

    margin = float(getattr(cfg, "margin_RZ", 0.5))
    Rmin = float(R_outer.min() - margin)
    Rmax = float(R_outer.max() + margin)
    Zmin = float(Z_outer.min() - margin)
    Zmax = float(Z_outer.max() + margin)

    if verbose:
        print("\n--- Numerical domain ---", flush=True)
        print(f"R in [{Rmin:.2f}, {Rmax:.2f}] m", flush=True)
        print(f"Z in [{Zmin:.2f}, {Zmax:.2f}] m", flush=True)
        print(f"Grid: nx = {int(cfg.nx_eq)}, ny = {int(cfg.ny_eq)}", flush=True)

    # 3) Equilibrium object
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=Rmin, Rmax=Rmax,
        Zmin=Zmin, Zmax=Zmax,
        nx=int(cfg.nx_eq),
        ny=int(cfg.ny_eq),
    )

    # 4) Solve with continuation
    solver = GSstaticsolver.NKGSsolver(eq)

    f_list = tuple(getattr(cfg, "f_list_equilibrium", (0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.00)))
    tol_ramp  = float(getattr(cfg, "target_rel_tol_ramp", 3e-6))
    tol_final = float(getattr(cfg, "target_rel_tol", 1e-8))

    if verbose:
        print("\n--- Solving equilibrium (Newton–Krylov) ---", flush=True)
        print(f"[DEBUG] f_list_equilibrium = {f_list} | len = {len(f_list)}", flush=True)

    # noise log policy
    if silence_solver_noise:
        if solver_noise_log is None:
            solver_noise_log = str(_results_dir() / "solver_noise.log")
    else:
        solver_noise_log = None

    for j, f in enumerate(f_list):
        this_tol = tol_final if (j == len(f_list) - 1) else tol_ramp

        # Apply scaled family currents (important: family total scales with f)
        set_star_currents(
            tokamak, geom,
            CS=f * cfg.CS_current,
            PF1=f * cfg.PF1_current,
            PF2=f * cfg.PF2_current,
            PF3=f * cfg.PF3_current,
        )

        # Optional: sanity print at this f (shows scaling, not a bug)
        if verbose:
            fam_targets = {
                "CS": f * cfg.CS_current,
                "PF1": f * cfg.PF1_current,
                "PF2": f * cfg.PF2_current,
                "PF3": f * cfg.PF3_current,
            }
            print("\n" + currents_sanity_check(tokamak, fam_targets), flush=True)
            print(f"[continuation] j={j+1}/{len(f_list)} f={f:.3f} tol={this_tol:.1e}", flush=True)

        profiles = _make_profiles(
            eq,
            paxis=f * cfg.paxis,
            Ip=f * cfg.Ip,
            fvac=cfg.fvac,
            alpha_m=cfg.alpha_m,
            alpha_n=cfg.alpha_n,
        )

        # Solve, but suppress spam to log/devnull
        with redirect_stdout_stderr(solver_noise_log):
            solver.solve(
                eq=eq,
                profiles=profiles,
                constrain=None,
                target_relative_tolerance=float(this_tol),
                verbose=False,
            )

        if verbose:
            R_ax, Z_ax = eq.magneticAxis()[:2]
            print(f"[done] f={f:.3f} axis=({R_ax:.6f},{Z_ax:.6e})", flush=True)

    if verbose:
        print("[OK] Continuation finished.", flush=True)
        if solver_noise_log is not None:
            print(f"[INFO] Solver noise logged to: {solver_noise_log}", flush=True)

    # 5) Separatrix geometry (only after finishing the whole ramp)
    shape = shape_from_separatrix(eq, geom)

    # Add magnetic axis in consistent place
    R_ax, Z_ax = eq.magneticAxis()[:2]
    shape["R_ax"] = float(R_ax)
    shape["Z_ax"] = float(Z_ax)

    if verbose:
        print("\n=== Plasma geometry (from separatrix) ===", flush=True)
        print(f"  R0_plasma    = {shape['R0_plasma']:.3f} m", flush=True)
        print(f"  a_plasma     = {shape['a_plasma']:.3f} m", flush=True)
        print(f"  A_plasma     = {shape['A_plasma']:.3f}", flush=True)
        print(f"  kappa_plasma = {shape['kappa_plasma']:.3f}", flush=True)
        print(f"  delta_u      = {shape['delta_u']:.3f}", flush=True)
        print(f"  delta_l      = {shape['delta_l']:.3f}", flush=True)
        print(f"  R_ax         = {shape['R_ax']:.3f} m", flush=True)
        print(f"  Z_ax         = {shape['Z_ax']:.3f} m", flush=True)

    return eq, tokamak, geom, shape


# -------------------------
# Plot
# -------------------------

def plot_equilibrium(eq, geom, shape, filename: str | None = None):
    plt.rcParams.update({
        "figure.figsize": (6, 10),
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    fig, ax = plt.subplots()
    eq.plot(axis=ax, show=False)

    ax.plot(geom["R_outer"], geom["Z_outer"], "k", lw=2, label="Vessel (outer wall)")

    if "R_inner" in geom and "Z_inner" in geom:
        ax.plot(geom["R_inner"], geom["Z_inner"], "k--", lw=1.5, label="Inner wall / limiter")

    if "R_plasma" in geom and "Z_plasma" in geom:
        ax.plot(geom["R_plasma"], geom["Z_plasma"], "k--", lw=1.5, label="Plasma target (CAD)")

    if "R_sep" in shape and "Z_sep" in shape:
        ax.plot(shape["R_sep"], shape["Z_sep"], color="tab:red", lw=2.2, label="Separatrix (eq)")

    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("STAR-like equilibrium – equilibrium map")

    txt = (
        rf"$R_0^{{\rm pl}} = {shape['R0_plasma']:.2f}\,\mathrm{{m}}$" "\n"
        rf"$a = {shape['a_plasma']:.2f}\,\mathrm{{m}},\ A = {shape['A_plasma']:.2f}$" "\n"
        rf"$\kappa = {shape['kappa_plasma']:.2f}$" "\n"
        rf"$\delta_u,\delta_l \approx {shape['delta_u']:.2f}, {shape['delta_l']:.2f}$"
    )
    ax.text(
        0.02, 0.02, txt,
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.legend(loc="upper right")
    plt.tight_layout()

    if filename is None:
        filename = getattr(cfg, "fig_equilibrium", "STAR_equilibrium.png")

    out_dir = _results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    print(f"Equilibrium figure saved to: {out_path}", flush=True)


def main():
    eq, tokamak, geom, shape = build_equilibrium(
        verbose=True,
        silence_solver_noise=True,   # keep console clean
        solver_noise_log=None,       # default results/solver_noise.log
    )
    plot_equilibrium(eq, geom, shape, filename=getattr(cfg, "fig_equilibrium", None))


if __name__ == "__main__":
    main()

