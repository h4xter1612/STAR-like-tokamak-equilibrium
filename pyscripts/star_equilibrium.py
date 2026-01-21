"""
star_equilibrium.py

Refined STAR-like "bean" equilibrium using the parameters defined in
config_star_bean.py, with CAD/DXF-backed machine geometry (star_machine_cad.py).
"""

from __future__ import annotations

import os
import types
import numpy as np
import matplotlib.pyplot as plt

from freegsnke import equilibrium_update, GSstaticsolver
from freegsnke.jtor_update import ConstrainPaxisIp

from star_machine_cad import make_star_machine_from_cad, CADImportOptions
from analyze_star_shape import shape_from_separatrix
import config_star_bean as cfg


# -------------------------
# Currents
# -------------------------

def set_star_currents(tokamak, CS=None, PF1=None, PF2=None, PF3=None):
    """
    Assign PF/CS currents to the tokamak Machine.

    Any current set to None is taken from config_star_bean.
    """
    if CS is None:
        CS = cfg.CS_current
    if PF1 is None:
        PF1 = cfg.PF1_current
    if PF2 is None:
        PF2 = cfg.PF2_current
    if PF3 is None:
        PF3 = cfg.PF3_current

    for label, coil in tokamak.coils:
        lab = str(label).strip().upper()
        if lab == "CS":
            coil.current = float(CS)
        elif lab in ("PF1U", "PF1L"):
            coil.current = float(PF1)
        elif lab in ("PF2U", "PF2L"):
            coil.current = float(PF2)
        elif lab in ("PF3U", "PF3L"):
            coil.current = float(PF3)
        else:
            coil.current = 0.0


def _full_core_mask(eq) -> np.ndarray:
    """
    Robust core mask for the current grid.
    Do NOT rely on eq.psi() (can be transient/uninitialized in some paths).
    """
    return np.ones(eq.R.shape, dtype=bool)


def _ensure_masks(profiles, eq):
    """
    FreeGSNKE sometimes tries to copy masks that can be None (esp. limiter_core_mask
    when an inner wall/limiter exists from CAD). Ensure they exist and match grid.
    """
    base = _full_core_mask(eq)

    # Common masks seen in copy() code paths
    for name in ("diverted_core_mask", "limiter_core_mask"):
        val = getattr(profiles, name, None)
        if val is None:
            setattr(profiles, name, base.copy())
        else:
            arr = np.asarray(val)
            if arr.shape != base.shape:
                setattr(profiles, name, base.copy())

    # Extra safety: if future versions add more "*_core_mask" fields,
    # populate any that are present and currently None.
    for attr in dir(profiles):
        if attr.endswith("_core_mask"):
            try:
                v = getattr(profiles, attr)
            except Exception:
                continue
            if v is None:
                setattr(profiles, attr, base.copy())


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

    # Patch copy so it never fails due to None masks.
    _orig_copy = profiles.copy

    def _safe_copy(self):
        _ensure_masks(self, eq)
        return _orig_copy()

    profiles.copy = types.MethodType(_safe_copy, profiles)
    return profiles


# -------------------------
# Build equilibrium
# -------------------------

def build_equilibrium(
    verbose: bool = True,
    *,
    dxf_path: str | None = None,
    unit_scale: float | None = None,     # None => infer from INSUNITS
    resample_walls: str = "auto",        # "auto" | "always" | "never"
    n_wall: int = 801,
    n_inner: int = 801,
    min_wall_pts: int = 200,
    enforce_ccw: bool = True,
    canonical_start: bool = True,
):
    """
    Build the refined STAR-like bean equilibrium using only parameters in config_star_bean.py.
    Uses CAD machine geometry from star_machine_cad.py.
    """

    # -------------------------
    # 1) Geometry and Machine (CAD)
    # -------------------------
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

    # Set currents from cfg
    set_star_currents(tokamak)

    if verbose:
        print("\n--- CAD machine loaded ---")
        print(f"CAD path      = {geom.get('cad_path', '(unknown)')}")
        print(f"unit_scale    = {geom.get('unit_scale', unit_scale)}")
        print(f"resample_walls= {resample_walls} | n_wall={n_wall} n_inner={n_inner} min_wall_pts={min_wall_pts}")
        print(f"enforce_ccw   = {enforce_ccw} | canonical_start={canonical_start}")
        print(f"Coils found   = {sorted(list(geom.get('coils', {}).keys()))}")
        print(f"Outer pts     = {len(geom['R_outer'])}")
        if "R_inner" in geom:
            print(f"Inner pts     = {len(geom['R_inner'])}")
        if "R_plasma" in geom:
            print(f"Plasma target pts = {len(geom['R_plasma'])}")

        print("\n--- Target STAR-like settings (cfg) ---")
        print(f"R0_geom    = {cfg.R0_geom:.3f} m")
        print(f"A_geom     = {cfg.A_geom:.3f}")
        print(f"kappa_geom = {cfg.kappa_geom:.3f}")
        print(f"delta_geom = {cfg.delta_geom:.3f}")

        print("\n--- PF/CS currents (cfg) ---")
        print(f"CS  = {cfg.CS_current/1e6:.3f} MA")
        print(f"PF1 = {cfg.PF1_current/1e6:.3f} MA")
        print(f"PF2 = {cfg.PF2_current/1e6:.3f} MA")
        print(f"PF3 = {cfg.PF3_current/1e6:.3f} MA")

    # -------------------------
    # 2) Numerical domain (R, Z)
    # -------------------------
    R_outer = np.asarray(geom["R_outer"], dtype=float)
    Z_outer = np.asarray(geom["Z_outer"], dtype=float)

    Rmin = float(R_outer.min() - cfg.margin_RZ)
    Rmax = float(R_outer.max() + cfg.margin_RZ)
    Zmin = float(Z_outer.min() - cfg.margin_RZ)
    Zmax = float(Z_outer.max() + cfg.margin_RZ)

    if verbose:
        print("\n--- Numerical domain ---")
        print(f"R in [{Rmin:.2f}, {Rmax:.2f}] m")
        print(f"Z in [{Zmin:.2f}, {Zmax:.2f}] m")
        print(f"Grid: nx = {cfg.nx_eq}, ny = {cfg.ny_eq}")

    # -------------------------
    # 3) Equilibrium object
    # -------------------------
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=Rmin, Rmax=Rmax,
        Zmin=Zmin, Zmax=Zmax,
        nx=int(cfg.nx_eq),
        ny=int(cfg.ny_eq),
    )

    # -------------------------
    # 4) Solve with continuation (robust)
    # -------------------------
    solver = GSstaticsolver.NKGSsolver(eq)

    if verbose:
        print("\n--- Solving equilibrium (Newton–Krylov) ---")

    f_list = getattr(cfg, "f_list_equilibrium", (0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.00))
    tol_ramp  = float(getattr(cfg, "target_rel_tol_ramp", 1e-5))
    tol_final = float(getattr(cfg, "target_rel_tol", 1e-8))

    for j, f in enumerate(f_list):
        set_star_currents(
            tokamak,
            CS=f * cfg.CS_current,
            PF1=f * cfg.PF1_current,
            PF2=f * cfg.PF2_current,
            PF3=f * cfg.PF3_current,
        )

        profiles = _make_profiles(
            eq,
            paxis=f * cfg.paxis,
            Ip=f * cfg.Ip,
            fvac=cfg.fvac,
            alpha_m=cfg.alpha_m,
            alpha_n=cfg.alpha_n,
        )

        this_tol = tol_final if (j == len(f_list) - 1) else tol_ramp

        if verbose:
            print(f"\n  [continuation] f={f:.2f} | tol={this_tol:.1e}")

        solver.solve(
            eq=eq,
            profiles=profiles,
            constrain=None,
            target_relative_tolerance=this_tol,
            verbose=verbose,
        )

    # -------------------------
    # 5) Separatrix geometry
    # -------------------------
    shape = shape_from_separatrix(eq, geom)

    if verbose:
        print("\n=== Plasma geometry (from separatrix) ===")
        print(f"  R0_plasma    = {shape['R0_plasma']:.3f} m")
        print(f"  a_plasma     = {shape['a_plasma']:.3f} m")
        print(f"  A_plasma     = {shape['A_plasma']:.3f}")
        print(f"  kappa_plasma = {shape['kappa_plasma']:.3f}")
        print(f"  delta_u      = {shape['delta_u']:.3f}")
        print(f"  delta_l      = {shape['delta_l']:.3f}")
        print(f"  R_ax         = {shape['R_ax']:.3f} m")
        print(f"  Z_ax         = {shape['Z_ax']:.3f} m")

    return eq, tokamak, geom, shape


# -------------------------
# Plot
# -------------------------

def plot_equilibrium(eq, tokamak, geom, shape, filename: str | None = None):
    plt.rcParams.update({
        "figure.figsize": (6, 10),
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    R = eq.R
    Z = eq.Z
    Rmin, Rmax = R.min(), R.max()
    Zmin, Zmax = Z.min(), Z.max()

    fig, ax = plt.subplots()

    eq.plot(axis=ax, show=False)

    ax.plot(geom["R_outer"], geom["Z_outer"], "k", lw=2, label="Vessel (outer wall)")

    if "R_inner" in geom and "Z_inner" in geom:
        ax.plot(geom["R_inner"], geom["Z_inner"], "k--", lw=1.5, label="Inner wall / limiter")

    if "R_plasma" in geom and "Z_plasma" in geom:
        ax.plot(geom["R_plasma"], geom["Z_plasma"], "k--", lw=1.5, label="Plasma target (CAD)")

    ax.plot(shape["R_sep"], shape["Z_sep"], color="tab:red", lw=2.2, label="Separatrix (eq)")

    ax.set_aspect("equal")
    ax.set_xlim(Rmin, Rmax)
    ax.set_ylim(Zmin, Zmax)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("STAR-like bean equilibrium – equilibrium map")

    txt = (
        rf"$R_0^{{\rm pl}} = {shape['R0_plasma']:.2f}\,\mathrm{{m}}$" "\n"
        rf"$a = {shape['a_plasma']:.2f}\,\mathrm{{m}},\ "
        rf"A = {shape['A_plasma']:.2f}$" "\n"
        rf"$\kappa = {shape['kappa_plasma']:.2f}$" "\n"
        rf"$\delta \approx {shape['delta_u']:.2f}$"
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
        filename = cfg.fig_equilibrium

    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)

    out_path = os.path.join(results_dir, filename)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Equilibrium figure saved to: {out_path}")


def main():
    eq, tokamak, geom, shape = build_equilibrium(verbose=True)
    plot_equilibrium(eq, tokamak, geom, shape, filename=cfg.fig_equilibrium)


if __name__ == "__main__":
    main()

