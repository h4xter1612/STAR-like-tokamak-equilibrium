"""
diagnostics_star_bean.py

Diagnostics for the refined STAR-like "bean" equilibrium defined by
config_star_bean.py:

  - Separatrix geometry and equilibrium map
  - Safety factor profile q(psi_norm)
  - Pressure profile p(psi_norm) and poloidal beta
  - Coil forces (if available in this FreeGSNKE build)
  - Toroidal current density map j_phi(R, Z) inside the bean
  - Approximate global numbers (V_pl, W_p, <p>, beta_T, B0)
  - Approximate magnetic shear profile s_hat(psi)

Everything is built consistently from config_star_bean.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import os

from freegsnke import equilibrium_update, GSstaticsolver
from freegsnke.jtor_update import ConstrainPaxisIp

from star_machine import make_star_machine
from analyze_star_shape import shape_from_separatrix
import config_star_bean as cfg

# --------------------------------------------------------------------
# Directorio de resultados único
# --------------------------------------------------------------------
if "__file__" in globals():
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    _BASE_DIR = os.getcwd()

RESULTS_DIR = os.path.abspath(os.path.join(_BASE_DIR, "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

def _in_results(path_like):
    """Garantiza que el archivo vaya a RESULTS_DIR respetando solo el basename."""
    return os.path.join(RESULTS_DIR, os.path.basename(path_like))

# Figuras definidas en el config: forzamos que vayan a ../results
cfg.fig_shape = _in_results(cfg.fig_shape)
cfg.fig_q_profile = _in_results(cfg.fig_q_profile)
cfg.fig_pressure = _in_results(cfg.fig_pressure)

# Archivos extra de salida (todos en ../results)
FIG_JTOR_MAP = _in_results("STAR_bean_jtor_map.png")
FIG_SHEAR = _in_results("STAR_bean_shear_profile.png")
TXT_GLOBAL = _in_results("STAR_bean_global_numbers.txt")


# --------------------------------------------------------------------
# 0) Equilibrium construction (same physics for all diagnostics)
# --------------------------------------------------------------------
def set_star_currents(tokamak):
    """
    Assign PF/CS currents to the Machine using config_star_bean.py.
    """
    for label, coil in tokamak.coils:
        if label == "CS":
            coil.current = cfg.CS_current
        elif label in ("PF1U", "PF1L"):
            coil.current = cfg.PF1_current
        elif label in ("PF2U", "PF2L"):
            coil.current = cfg.PF2_current
        elif label in ("PF3U", "PF3L"):
            coil.current = cfg.PF3_current
        else:
            coil.current = 0.0


def build_equilibrium_and_profiles(verbose: bool = True):
    """
    Build the same equilibrium used in star_equilibrium.py, including
    the j_phi / pressure profiles object.

    Returns
    -------
    eq : freegsnke.equilibrium_update.Equilibrium
    tokamak : Machine
    geom : dict
    profiles : ConstrainPaxisIp
    """
    # Target STAR-like geometry (Miller)
    tokamak, geom = make_star_machine(
        R0=cfg.R0_geom,
        A=cfg.A_geom,
        kappa=cfg.kappa_geom,
        delta=cfg.delta_geom,
    )
    set_star_currents(tokamak)

    # Numerical domain from the outer vessel
    R_outer = geom["R_outer"]
    Z_outer = geom["Z_outer"]

    Rmin = float(R_outer.min() - cfg.margin_RZ)
    Rmax = float(R_outer.max() + cfg.margin_RZ)
    Zmin = float(Z_outer.min() - cfg.margin_RZ)
    Zmax = float(Z_outer.max() + cfg.margin_RZ)

    if verbose:
        print("\n[diagnostics] Numerical domain:")
        print(f"  R in [{Rmin:.2f}, {Rmax:.2f}] m")
        print(f"  Z in [{Zmin:.2f}, {Zmax:.2f}] m")
        print(f"  Grid nx={cfg.nx_eq}, ny={cfg.ny_eq}")

    # Equilibrium object
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=Rmin,
        Rmax=Rmax,
        Zmin=Zmin,
        Zmax=Zmax,
        nx=cfg.nx_eq,
        ny=cfg.ny_eq,
    )

    # j_phi / pressure profiles
    profiles = ConstrainPaxisIp(
        eq=eq,
        paxis=cfg.paxis,
        Ip=cfg.Ip,
        fvac=cfg.fvac,
        alpha_m=cfg.alpha_m,
        alpha_n=cfg.alpha_n,
    )
    profiles.diverted_core_mask = np.ones_like(eq.psi(), dtype=bool)

    # Newton–Krylov solver
    solver = GSstaticsolver.NKGSsolver(eq)
    solver.solve(
        eq=eq,
        profiles=profiles,
        constrain=None,
        target_relative_tolerance=cfg.target_rel_tol,
        verbose=verbose,
    )

    return eq, tokamak, geom, profiles


# --------------------------------------------------------------------
# 1) Separatrix geometry and equilibrium map
# --------------------------------------------------------------------
def analyze_shape(eq, tokamak, geom, save: bool = True):
    """
    Compute and plot the separatrix geometry for the current equilibrium.
    """
    shape = shape_from_separatrix(eq, geom)

    print("\n=== Plasma geometry (current equilibrium) ===")
    print(f"  R0_plasma    = {shape['R0_plasma']:.3f} m")
    print(f"  a_plasma     = {shape['a_plasma']:.3f} m")
    print(f"  A_plasma     = {shape['A_plasma']:.3f}")
    print(f"  kappa_plasma = {shape['kappa_plasma']:.3f}")
    print(f"  delta_u      = {shape['delta_u']:.3f}")
    print(f"  delta_l      = {shape['delta_l']:.3f}")
    print(f"  R_ax         = {shape['R_ax']:.3f} m")
    print(f"  Z_ax         = {shape['Z_ax']:.3f} m")

    R = eq.R
    Z = eq.Z
    Rmin, Rmax = R.min(), R.max()
    Zmin, Zmax = Z.min(), Z.max()

    plt.rcParams.update(
        {
            "figure.figsize": (6, 10),
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

    fig, ax = plt.subplots()

    # Poloidal flux + X/O-points
    eq.plot(axis=ax, show=False)

    # Vessel and target geometry
    ax.plot(geom["R_outer"], geom["Z_outer"], "k", lw=2, label="Vessel")
    ax.plot(
        geom["R_plasma"],
        geom["Z_plasma"],
        "k--",
        lw=1.5,
        label="Plasma target (geom)",
    )

    # Separatrix from the equilibrium
    ax.plot(
        shape["R_sep"],
        shape["Z_sep"],
        color="tab:red",
        lw=2.2,
        label="Separatrix (eq)",
    )

    ax.set_aspect("equal")
    ax.set_xlim(Rmin, Rmax)
    ax.set_ylim(Zmin, Zmax)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("STAR-like equilibrium – bean shape")

    ax.legend(loc="upper right")
    plt.tight_layout()

    if save:
        fig.savefig(cfg.fig_shape, dpi=200, bbox_inches="tight")
        print(f"Shape figure saved to: {cfg.fig_shape}")

    return shape


# --------------------------------------------------------------------
# 2) Safety factor profile q(psi_norm)
# --------------------------------------------------------------------
def analyze_q_profile(eq, save: bool = True):
    """
    Compute and plot q(psi_norm) avoiding the exact psi_norm = 0 and 1.
    """
    psis = np.linspace(0.01, 0.98, 200)
    psis_arr = np.array(psis)
    q_vals = np.array(eq.q(psis_arr))

    # Physical q_min (ignore ultra-central core)
    mask_core = psis_arr >= 0.02
    q_core = q_vals[mask_core]
    psis_core = psis_arr[mask_core]
    q_min_phys = float(q_core.min())
    psi_at_qmin = float(psis_core[q_core.argmin()])

    q_95 = float(np.interp(0.95, psis_arr, q_vals))

    print("\n=== Safety factor q(psi_norm) ===")
    print(
        f"  q_min (filtered) = {q_min_phys:.3f} "
        f"(at psi_norm ≈ {psi_at_qmin:.3f})"
    )
    print(f"  q_95              = {q_95:.3f}")
    for psi in [0.05, 0.25, 0.50, 0.75, 0.90]:
        q_val = float(np.interp(psi, psis_arr, q_vals))
        print(f"  q({psi:.2f}) = {q_val:.3f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(psis_arr, q_vals, "-")
    ax.set_xlabel(r"Normalised $\psi$")
    ax.set_ylabel(r"Safety factor $q$")
    ax.set_title("Safety factor profile q(ψ) – STAR-like bean")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fig.savefig(cfg.fig_q_profile, dpi=200, bbox_inches="tight")
        print(f"q-profile figure saved to: {cfg.fig_q_profile}")


# --------------------------------------------------------------------
# 3) Pressure profile and poloidal beta
# --------------------------------------------------------------------
def analyze_beta_and_pressure(eq, save: bool = True):
    """
    Compute pressure profile p(psi_norm) and poloidal beta.
    """
    betap = eq.poloidalBeta1()

    psis_arr = np.linspace(0.0, 0.98, 80)
    p_arr = np.array(eq.pressure(psis_arr))

    p_axis = float(p_arr[0])
    p_095 = float(np.interp(0.95, psis_arr, p_arr))

    print("\n=== Pressure and poloidal beta ===")
    print(f"  p_axis      = {p_axis:.3e} Pa")
    print(f"  p(psi=0.95) = {p_095:.3e} Pa")
    print(f"  beta_p(1)   = {betap:.3f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(psis_arr, p_arr, "-")
    ax.set_xlabel(r"Normalised $\psi$")
    ax.set_ylabel(r"p(ψ) [Pa]")
    ax.set_title("Pressure profile – STAR-like bean")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fig.savefig(cfg.fig_pressure, dpi=200, bbox_inches="tight")
        print(f"Pressure figure saved to: {cfg.fig_pressure}")


# --------------------------------------------------------------------
# 4) Coil forces
# --------------------------------------------------------------------
def analyze_coil_forces(eq):
    """
    Print coil forces if the FreeGSNKE build provides eq.printForces().
    """
    print("\n=== Coil forces (FreeGSNKE) ===")
    try:
        eq.printForces()
    except AttributeError:
        print("  printForces() is not available in this FreeGSNKE version.")


# --------------------------------------------------------------------
# 5) Toroidal current density map j_phi(R, Z)
# --------------------------------------------------------------------
def plot_jtor_map(eq, geom, profiles):
    """
    Plot j_phi(R,Z) in MA/m^2 restricted to the bean-shaped plasma region.
    """
    psi = eq.psi()
    psi_sep = eq.psi_bndry

    # Toroidal current density [A/m^2] from profiles
    jtor = profiles.Jtor(eq.R, eq.Z, psi, psi_sep)
    jtor_MA = jtor / 1e6  # MA/m^2

    # --- Mask: only the central bean (separatrix from shape_from_separatrix) ---
    shape = shape_from_separatrix(eq, geom)
    R_sep = shape["R_sep"]
    Z_sep = shape["Z_sep"]

    # Polygon for the separatrix
    poly = np.vstack((R_sep, Z_sep)).T
    path = Path(poly, closed=True)

    # Test if each (R,Z) grid point is inside the polygon
    points = np.vstack((eq.R.ravel(), eq.Z.ravel())).T
    inside = path.contains_points(points)
    plasma_mask = inside.reshape(eq.R.shape)

    # Keep j_phi only inside the bean
    jtor_plot = np.where(plasma_mask, jtor_MA, np.nan)

    jmax = float(np.nanmax(np.abs(jtor_plot)))
    print(f"j_phi max (in plasma) ≈ {jmax:.3e} MA/m^2")

    fig, ax = plt.subplots(figsize=(5, 9))

    pcm = ax.pcolormesh(
        eq.R,
        eq.Z,
        jtor_plot,
        shading="auto",
        vmin=-jmax,
        vmax=jmax,
    )
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r"$j_\phi$ [MA/m$^2$]")

    ax.plot(geom["R_outer"], geom["Z_outer"], "k-", lw=2, label="Vessel")
    ax.plot(R_sep, Z_sep, "r-", lw=2, label="Separatrix (eq)")

    ax.set_aspect("equal")
    ax.set_xlim(eq.R.min(), eq.R.max())
    ax.set_ylim(eq.Z.min(), eq.Z.max())
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("STAR-like bean – toroidal current density map")

    ax.legend(loc="lower left", framealpha=0.85)
    plt.tight_layout()

    fig.savefig(FIG_JTOR_MAP, dpi=200, bbox_inches="tight")
    print(f"j_phi figure saved to: {FIG_JTOR_MAP}")


# --------------------------------------------------------------------
# 6) Global approximate numbers (V_pl, W_p, <p>, beta_T, B0)
# --------------------------------------------------------------------
def compute_global_numbers(eq, geom, profiles=None):
    """
    Approximate global quantities:

      - Plasma volume V_pl
      - Stored energy W_p
      - Volume-averaged pressure <p>
      - Effective toroidal field B0 ~ f_vac / R0_pl
      - Volume-averaged beta_T

    Uses the same recipe that worked in your earlier script.
    `profiles` is kept as an optional argument for backward compatibility
    (it is not used here).
    """
    from scipy.constants import mu_0

    psi = eq.psi()
    psi_sep = eq.psi_bndry

    # Plasma region: everything "inside" the separatrix
    mask_plasma = psi <= psi_sep

    if not mask_plasma.any():
        print(
            "\n[WARNING] Plasma mask (psi<=psi_sep) is empty; "
            "cannot compute global numbers."
        )
        return

    # psi at magnetic axis (minimum inside plasma)
    psi_axis = psi[mask_plasma].min()

    # Normalised psi (0 at axis, 1 at separatrix)
    psinorm = (psi - psi_axis) / (psi_sep - psi_axis)
    psinorm = np.clip(psinorm, 0.0, 1.0)

    # Pressure on the full grid; we will integrate only inside mask_plasma
    p_grid = np.array(eq.pressure(psinorm))

    R = eq.R
    Z = eq.Z

    dR = float(R[1, 0] - R[0, 0])
    dZ = float(Z[0, 1] - Z[0, 0])
    dV = 2.0 * np.pi * R * dR * dZ

    V_pl = float(np.sum(dV * mask_plasma))
    W_p = float(np.sum(p_grid * dV * mask_plasma))

    if V_pl <= 0.0:
        print(
            "\n[WARNING] Plasma volume V_pl <= 0; "
            "cannot define <p> or beta_T."
        )
        return

    p_avg = W_p / V_pl

    # Effective toroidal field ~ f_vac / R0_pl (from separatrix geometry)
    shape = shape_from_separatrix(eq, geom)
    R0_pl = shape["R0_plasma"]
    B0 = cfg.fvac / R0_pl  # [T]

    beta_T = 2.0 * mu_0 * p_avg / (B0**2)

    print("\n=== Global numbers (approx) ===")
    print(f"  Plasma volume V_pl   = {V_pl:.3e} m^3")
    print(f"  Stored energy W_p    = {W_p:.3e} J")
    print(f"  <p> (volume)         = {p_avg:.3e} Pa")
    print(f"  B0_tor (approx)      = {B0:.3f} T")
    print(f"  beta_T, volume-avg   = {beta_T:.3f}")

    with open(TXT_GLOBAL, "w") as f:
        f.write("STAR-like bean – global numbers (approx)\n")
        f.write(f"V_pl   = {V_pl:.6e} m^3\n")
        f.write(f"W_p    = {W_p:.6e} J\n")
        f.write(f"<p>    = {p_avg:.6e} Pa\n")
        f.write(f"B0_tor = {B0:.6e} T\n")
        f.write(f"beta_T = {beta_T:.6e}\n")

    print(f"Global summary written to: {TXT_GLOBAL}")


# --------------------------------------------------------------------
# 7) Approximate magnetic shear profile s_hat(psi)
# --------------------------------------------------------------------
def compute_shear_profile(eq):
    """
    Compute a simple estimate of the magnetic shear:

        s_hat ~ (psi / q) dq/dpsi

    on psi_norm in [0.05, 0.95], to avoid edge artefacts.
    """
    psis = np.linspace(0.05, 0.95, 80)
    q_vals = np.array(eq.q(psis))

    dq_dpsi = np.gradient(q_vals, psis)
    s_hat = (psis / q_vals) * dq_dpsi

    print("\n=== Approximate magnetic shear s_hat(psi) ===")
    print(f"  s_hat(min) ≈ {s_hat.min():.3f}")
    print(f"  s_hat(max) ≈ {s_hat.max():.3f}")

    plt.figure(figsize=(8, 4))
    plt.plot(psis, s_hat, "-")
    plt.xlabel(r"Normalised $\psi$")
    plt.ylabel(r"$\hat{s}(\psi)$")
    plt.title("Magnetic shear profile (approx.) – STAR-like bean")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_SHEAR, dpi=200, bbox_inches="tight")
    print(f"Shear figure saved to: {FIG_SHEAR}")


# --------------------------------------------------------------------
def main():
    eq, tokamak, geom, profiles = build_equilibrium_and_profiles(verbose=True)

    # Core equilibrium / geometry diagnostics
    shape = analyze_shape(eq, tokamak, geom, save=True)
    analyze_q_profile(eq, save=True)
    analyze_beta_and_pressure(eq, save=True)
    analyze_coil_forces(eq)

    # Extra j_phi / global / shear diagnostics
    plot_jtor_map(eq, geom, profiles)
    compute_global_numbers(eq, geom, profiles)
    compute_shear_profile(eq)


if __name__ == "__main__":
    main()

