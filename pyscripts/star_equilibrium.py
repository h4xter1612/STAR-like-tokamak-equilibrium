"""
star_equilibrium.py

Refined STAR-like "bean" equilibrium using the parameters defined in
config_star_bean.py.

This script:
  - prints basic geometric and solver information to stdout
  - produces a publication-style figure with:
      * poloidal flux contours
      * X- and O-points
      * separatrix from the GS solution
      * target Miller geometry
      * vessel outline
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from freegsnke import equilibrium_update, GSstaticsolver
from freegsnke.jtor_update import ConstrainPaxisIp

from star_machine_cad import make_star_machine_from_cad
# from star_machine import make_star_machine
from analyze_star_shape import shape_from_separatrix
import config_star_bean as cfg


def set_star_currents(tokamak,
                      CS=None, PF1=None, PF2=None, PF3=None):
    """
    Assign PF/CS currents to the tokamak Machine.

    Any current set to ``None`` is taken from ``config_star_bean``.
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
        if label == "CS":
            coil.current = CS
        elif label in ("PF1U", "PF1L"):
            coil.current = PF1
        elif label in ("PF2U", "PF2L"):
            coil.current = PF2
        elif label in ("PF3U", "PF3L"):
            coil.current = PF3
        else:
            coil.current = 0.0


def build_equilibrium(verbose: bool = True):
    """
    Build the refined STAR-like bean equilibrium using only the parameters
    specified in ``config_star_bean.py``.

    Returns
    -------
    eq : freegsnke.equilibrium_update.Equilibrium
        Equilibrium object with the converged GS solution.
    tokamak : freegs4e.machine.Machine
        Machine object (geometry + coils).
    geom : dict
        Geometry dictionary returned by :func:`make_star_machine`.
    shape : dict
        Geometry of the separatrix, as returned by :func:`shape_from_separatrix`.
    """

    # -------------------------
    # 1) Geometry and Machine
    # -------------------------
    tokamak, geom = make_star_machine_from_cad()
    # tokamak, geom = make_star_machine(
    #     R0=cfg.R0_geom,
    #     A=cfg.A_geom,
    #     kappa=cfg.kappa_geom,
    #     delta=cfg.delta_geom,
    # )
    #
    # Refined coil currents
    set_star_currents(tokamak)

    if verbose:
        print("\n--- Target STAR-like geometry (Miller) ---")
        print(f"R0_geom    = {cfg.R0_geom:.3f} m")
        print(f"A_geom     = {cfg.A_geom:.3f}")
        print(f"kappa_geom = {cfg.kappa_geom:.3f}")
        print(f"delta_geom = {cfg.delta_geom:.3f}")

        print("\n--- PF/CS currents (refined STAR-like bean) ---")
        print(f"CS  = {cfg.CS_current/1e6:.3f} MA")
        print(f"PF1 = {cfg.PF1_current/1e6:.3f} MA")
        print(f"PF2 = {cfg.PF2_current/1e6:.3f} MA")
        print(f"PF3 = {cfg.PF3_current/1e6:.3f} MA")

    # -------------------------
    # 2) Numerical domain (R, Z)
    # -------------------------
    R_outer = geom["R_outer"]
    Z_outer = geom["Z_outer"]

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
        nx=cfg.nx_eq,
        ny=cfg.ny_eq,
    )

    # -------------------------
    # 4) Profiles jtor / pressure
    # -------------------------
    profiles = ConstrainPaxisIp(
        eq=eq,
        paxis=cfg.paxis,
        Ip=cfg.Ip,
        fvac=cfg.fvac,
        alpha_m=cfg.alpha_m,
        alpha_n=cfg.alpha_n,
    )
    # For this bean scenario we treat the whole core as diverted
    profiles.diverted_core_mask = np.ones_like(eq.psi(), dtype=bool)

    # -------------------------
    # 5) Newton–Krylov solver
    # -------------------------
    solver = GSstaticsolver.NKGSsolver(eq)

    if verbose:
        print("\n--- Solving equilibrium (Newton–Krylov) ---")

    solver.solve(
        eq=eq,
        profiles=profiles,
        constrain=None,
        target_relative_tolerance=cfg.target_rel_tol,
        verbose=verbose,
    )

    # -------------------------
    # 6) Separatrix geometry
    # -------------------------
    shape = shape_from_separatrix(eq, geom)

    if verbose:
        print("\n=== Plasma geometry (refined STAR-like bean) ===")
        print(f"  R0_plasma    = {shape['R0_plasma']:.3f} m")
        print(f"  a_plasma     = {shape['a_plasma']:.3f} m")
        print(f"  A_plasma     = {shape['A_plasma']:.3f}")
        print(f"  kappa_plasma = {shape['kappa_plasma']:.3f}")
        print(f"  delta_u      = {shape['delta_u']:.3f}")
        print(f"  delta_l      = {shape['delta_l']:.3f}")
        print(f"  R_ax         = {shape['R_ax']:.3f} m")
        print(f"  Z_ax         = {shape['Z_ax']:.3f} m")

    return eq, tokamak, geom, shape


def plot_equilibrium(eq, tokamak, geom, shape, filename: str | None = None):
    """
    Produce a publication-style equilibrium figure:

      - poloidal flux contours (including X- and O-points via eq.plot)
      - vessel
      - target plasma boundary (Miller geometry)
      - separatrix from the GS equilibrium
    """

    # Slightly cleaner style
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

    # Poloidal flux contours and X/O points
    eq.plot(axis=ax, show=False)

    # Vessel and target geometry
    ax.plot(geom["R_outer"], geom["Z_outer"], "k", lw=2, label="Vessel")
    ax.plot(
        geom["R_plasma"], geom["Z_plasma"],
        "k--", lw=1.5, label="Plasma target (geom)",
    )

    # Separatrix from the GS solution
    ax.plot(
        shape["R_sep"], shape["Z_sep"],
        color="tab:red", lw=2.2, label="Separatrix (eq)",
    )

    ax.set_aspect("equal")
    ax.set_xlim(Rmin, Rmax)
    ax.set_ylim(Zmin, Zmax)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("STAR-like bean equilibrium – equilibrium map")

    # Small box with key plasma parameters
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
    

    # Definir carpeta de resultados
    results_dir = "../results"

    # Crear la carpeta si no existe
    os.makedirs(results_dir, exist_ok=True)
    
    filename = os.path.join(results_dir, filename)
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Equilibrium figure saved to: {filename}")

    # For interactive inspection, uncomment:
    # plt.show()


def main():
    eq, tokamak, geom, shape = build_equilibrium(verbose=True)
    plot_equilibrium(eq, tokamak, geom, shape, filename=cfg.fig_equilibrium)


if __name__ == "__main__":
    main()

