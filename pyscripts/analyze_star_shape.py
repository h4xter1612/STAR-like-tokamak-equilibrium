# analyze_star_shape.py
#
# Compare the target STAR-like geometry (prescribed plasma boundary)
# with the actual plasma geometry (separatrix) from a FreeGSNKE equilibrium,
# using the optimal coil currents obtained from a scan.

import numpy as np
import matplotlib.pyplot as plt

from freegsnke import equilibrium_update, GSstaticsolver
from freegsnke.jtor_update import ConstrainPaxisIp

from star_machine import make_star_machine


# ------------------------------------------------------------
# 1. Optimal coil currents (from scan)
# ------------------------------------------------------------

def set_star_currents_opt(tokamak):
    """
    Set the coil currents that performed best in the micro-scan:

        CS  = 0.80 MA
        PF1 = -0.20 MA
        PF2 =  0.00 MA
        PF3 =  1.00 MA

    Parameters
    ----------
    tokamak : object
        Tokamak (or machine) object with a `coils` iterable of (label, coil) pairs.
        The `coil` objects are expected to have a `.current` attribute in Amperes.
    """
    for label, coil in tokamak.coils:
        if label == "CS":
            coil.current = 0.8e6
        elif label in ("PF1U", "PF1L"):
            coil.current = -0.20e6
        elif label in ("PF2U", "PF2L"):
            coil.current = 0.0e6
        elif label in ("PF3U", "PF3L"):
            coil.current = 1.00e6
        else:
            coil.current = 0.0


# ------------------------------------------------------------
# 2. Separatrix from a psi = psi_sep contour
# ------------------------------------------------------------

def shape_from_separatrix(eq, geom):
    """
    Extract the separatrix as a closed curve using a contour at psi = psi_sep,
    and compute basic geometric parameters of the plasma:

        R0_plasma, a_plasma, A_plasma, kappa_plasma, delta_u, delta_l

    We use `cs.allsegs` for compatibility with older versions of matplotlib.

    Parameters
    ----------
    eq : object
        Equilibrium object from FreeGSNKE. It is expected to provide:
          - eq.psi(): 2D array of poloidal flux ψ(R, Z)
          - eq.psi_bndry: scalar value of ψ at the plasma boundary
          - eq.R, eq.Z: 2D grids of R and Z coordinates
          - eq.magneticAxis(): returns (R_axis, Z_axis, psi_axis, ...)
    geom : dict
        Geometry dictionary (not used directly here, but kept for interface
        consistency and possible extensions).

    Returns
    -------
    results : dict
        Dictionary with:
          - "R_ax", "Z_ax": magnetic axis coordinates
          - "R0_plasma": major radius of plasma (m)
          - "a_plasma" : minor radius of plasma (m)
          - "A_plasma" : aspect ratio R0 / a
          - "kappa_plasma": elongation
          - "delta_u", "delta_l": upper and lower triangularities
          - "R_sep", "Z_sep": arrays for the separatrix contour
    """
    psi = eq.psi()
    psi_sep = eq.psi_bndry
    R = eq.R
    Z = eq.Z

    # Magnetic axis position
    R_ax, Z_ax = eq.magneticAxis()[:2]

    # --- 1. Compute psi = psi_sep contours using matplotlib ---
    fig, ax = plt.subplots()
    cs = ax.contour(R, Z, psi, levels=[psi_sep])
    plt.close(fig)  # do not display this figure

    segs = cs.allsegs[0]
    if len(segs) == 0:
        raise RuntimeError("Did not find any psi = psi_sep contour")

    # --- 2. Choose the segment that encloses the magnetic axis ---
    chosen = None
    for seg in segs:
        R_path = seg[:, 0]
        Z_path = seg[:, 1]
        if (R_path.min() < R_ax < R_path.max() and
                Z_path.min() < Z_ax < Z_path.max()):
            chosen = seg
            break

    # If none clearly encloses the axis, pick the one with closest centroid
    if chosen is None:
        best_d2 = None
        for seg in segs:
            Rc = seg[:, 0].mean()
            Zc = seg[:, 1].mean()
            d2 = (Rc - R_ax) ** 2 + (Zc - Z_ax) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                chosen = seg

    R_sep = chosen[:, 0]
    Z_sep = chosen[:, 1]

    # --- 3. Extremes to compute a, R0_plasma, kappa ---
    R_min = R_sep.min()
    R_max = R_sep.max()
    Z_min = Z_sep.min()
    Z_max = Z_sep.max()

    a_plasma = 0.5 * (R_max - R_min)
    R0_plasma = 0.5 * (R_max + R_min)
    A_plasma = R0_plasma / a_plasma
    kappa_plasma = 0.5 * (Z_max - Z_min) / a_plasma

    # --- 4. Triangularities using highest and lowest points ---
    idx_top = np.argmax(Z_sep)
    idx_bot = np.argmin(Z_sep)
    R_top = R_sep[idx_top]
    R_bot = R_sep[idx_bot]

    delta_u = (R0_plasma - R_top) / a_plasma
    delta_l = (R0_plasma - R_bot) / a_plasma

    results = {
        "R_ax": R_ax,
        "Z_ax": Z_ax,
        "R0_plasma": R0_plasma,
        "a_plasma": a_plasma,
        "A_plasma": A_plasma,
        "kappa_plasma": kappa_plasma,
        "delta_u": delta_u,
        "delta_l": delta_l,
        "R_sep": R_sep,
        "Z_sep": Z_sep,
    }
    return results


# ------------------------------------------------------------
# 3. Main script
# ------------------------------------------------------------

def main():
    # --- Machine definition and optimal currents ---
    tokamak, geom = make_star_machine(
        R0=4.0,
        A=1.7,
        kappa=1.8,
        delta=0.30,
    )
    set_star_currents_opt(tokamak)

    # --- Domain (same style as in star_equilibrium) ---
    R_outer = geom["R_outer"]
    Z_outer = geom["Z_outer"]
    Rmin = float(R_outer.min() - 0.5)
    Rmax = float(R_outer.max() + 0.5)
    Zmin = float(Z_outer.min() - 0.5)
    Zmax = float(Z_outer.max() + 0.5)

    print(f"Domain R = [{Rmin:.2f}, {Rmax:.2f}] m")
    print(f"Domain Z = [{Zmin:.2f}, {Zmax:.2f}] m")

    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=Rmin, Rmax=Rmax,
        Zmin=Zmin, Zmax=Zmax,
        nx=129, ny=257,
    )

    # --- Profiles (same as in star_equilibrium) ---
    Ip = 0.8e6      # 0.8 MA
    paxis = 2.0e3   # 2 kPa
    fvac = 0.5

    print(
        f"Using Ip = {Ip/1e6:.3f} MA, "
        f"paxis = {paxis/1e3:.2f} kPa, "
        f"fvac = {fvac:.2f}"
    )

    profiles = ConstrainPaxisIp(
        eq=eq,
        paxis=paxis,
        Ip=Ip,
        fvac=fvac,
        alpha_m=1.8,
        alpha_n=1.2,
    )
    # No diverted region: treat entire domain as core
    profiles.diverted_core_mask = np.ones_like(eq.psi(), dtype=bool)

    solver = GSstaticsolver.NKGSsolver(eq)

    solver.solve(
        eq=eq,
        profiles=profiles,
        constrain=None,
        target_relative_tolerance=1e-8,
        verbose=True,
    )

    # --- Extract separatrix shape ---
    shape = shape_from_separatrix(eq, geom)

    # --- Design geometric parameters (target oval) ---
    R0_geom = geom["R0"]
    a_geom = geom["a"]
    A_geom = geom["A"]
    kappa_geom = geom["kappa"]
    delta_geom = geom.get("delta", 0.0)

    # --- Print comparison ---
    print("\n=== Geometry comparison ===")
    print("Target geometry:")
    print(f"  R0_geom    = {R0_geom:.3f} m")
    print(f"  a_geom     = {a_geom:.3f} m")
    print(f"  A_geom     = {A_geom:.3f}")
    print(f"  kappa_geom = {kappa_geom:.3f}")
    print(f"  delta_geom = {delta_geom:.3f}")

    print("\nPlasma geometry (separatrix):")
    print(f"  R0_plasma   = {shape['R0_plasma']:.3f} m")
    print(f"  a_plasma    = {shape['a_plasma']:.3f} m")
    print(f"  A_plasma    = {shape['A_plasma']:.3f}")
    print(f"  kappa_plasma = {shape['kappa_plasma']:.3f}")
    print(f"  delta_u     = {shape['delta_u']:.3f}")
    print(f"  delta_l     = {shape['delta_l']:.3f}")
    print(
        f"\nMagnetic axis: R_ax = {shape['R_ax']:.3f} m, "
        f"Z_ax = {shape['Z_ax']:.3f} m"
    )

    # --- Final plot: compare everything ---
    fig, ax = plt.subplots(figsize=(5, 9))

    # Flux contours from the equilibrium
    eq.plot(axis=ax, show=False)

    # Extracted separatrix
    ax.plot(shape["R_sep"], shape["Z_sep"], "r-", lw=2, label="Separatrix (eq)")

    # Target plasma boundary (design oval)
    ax.plot(
        geom["R_plasma"],
        geom["Z_plasma"],
        "k--",
        lw=1.5,
        label="Plasma target (geom)",
    )

    # External vessel wall
    ax.plot(geom["R_outer"], geom["Z_outer"], "k", lw=2, label="Vessel")

    ax.set_aspect("equal")
    ax.set_xlim(Rmin, Rmax)
    ax.set_ylim(Zmin, Zmax)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("STAR-like equilibrium – geometric shape comparison")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

