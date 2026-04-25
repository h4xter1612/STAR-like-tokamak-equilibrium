"""
star_machine.py

Builds a "STAR-like" spherical tokamak for FreeGS/FreeGSNKE with a
Miller-like D-shaped plasma boundary (finite elongation and triangularity)
and a simple set of PF/CS coils aimed at achieving
kappa ~ 1.8 and positive triangularity.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from freegs4e import machine


def _dshape(R0: float, a: float, kappa: float, delta: float, theta: np.ndarray):
    """
    Up–down symmetric Miller-like parameterisation:

        R(θ) = R0 + a cos(θ + δ sin θ)
        Z(θ) = κ a sin θ
    """
    R = R0 + a * np.cos(theta + delta * np.sin(theta))
    Z = kappa * a * np.sin(theta)
    return R, Z


def star_geometry(
    R0: float = 4.0,
    A: float = 1.7,       # more ST-like than A = 2
    kappa: float = 1.8,   # target elongation
    delta: float = 0.30,  # target triangularity
    wall_gap_R: float = 0.45,
    wall_gap_Z: float = 0.45,
    inner_gap_R: float = 0.35,
    inner_gap_Z: float = 0.45,
):
    """
    Build the STAR-like geometric data structure.

    Returns a dictionary with all (R, Z) arrays for:
      - plasma boundary (target, Miller D-shape)
      - inner wall
      - outer vessel
      - PF/CS coil rectangles

    Main design parameters:
      - R0    : major radius
      - A     : target geometric aspect ratio (R0 / a)
      - kappa : target elongation
      - delta : target triangularity (Miller-like)

    *_gap_* control radial/vertical clearance between plasma and walls.
    """

    # Target minor radius
    a = R0 / A

    # Fine angular mesh for smooth contours
    theta = np.linspace(0.0, 2.0 * np.pi, 400)

    # ---- Plasma boundary (Miller, D-shape) ----
    Rp, Zp = _dshape(R0, a, kappa, delta, theta)

    # ---- Inner and outer walls following the same D-shape with gaps ----
    a_inner_R = a + inner_gap_R
    kappa_inner = (kappa * a + inner_gap_Z) / a_inner_R
    R_inner, Z_inner = _dshape(R0, a_inner_R, kappa_inner, delta, theta)

    a_outer_R = a + inner_gap_R + wall_gap_R
    kappa_outer = (kappa * a + inner_gap_Z + wall_gap_Z) / a_outer_R
    R_outer, Z_outer = _dshape(R0, a_outer_R, kappa_outer, delta, theta)

    # -------------------------
    # Global extents of plasma and walls
    # -------------------------
    R_pl_inboard = Rp.min()
    R_pl_outboard = Rp.max()
    Z_pl_top = Zp.max()
    Z_pl_bot = Zp.min()

    # Inboard / outboard of the inner wall (size of the central column)
    R_inboard_mid = R_inner.min()
    R_inner_out_mid = R_inner.max()

    # Outboard of the outer wall (for PF1 outside the vessel)
    R_out_mid = R_outer.max()

    # Top of the vessel (for reference)
    Z_top_outer = Z_outer.max()

    # -------------------------
    # Central solenoid (CS)
    # -------------------------
    # Central column goes from R ~ 0 to R_inboard_mid.
    # Place the CS around ~0.5 R_inboard_mid with a thickness such that
    # it does not touch either the axis or the inner wall.
    R_cs = 0.5 * R_inboard_mid
    dR_cs = 0.3 * R_inboard_mid   # R_cs - dR_cs > 0 and R_cs + dR_cs < R_inboard_mid
    dZ_cs = 0.8 * a               # height somewhat smaller than plasma height

    # -------------------------
    # PF1: pair outside the vessel (main vertical field)
    # -------------------------
    # Symmetric coils above/below, relatively far away, to provide
    # global vertical field.
    R_pf1 = R_out_mid + 0.6
    Z_pf1 = 0.0
    dR_pf1 = 0.5
    dZ_pf1 = 0.8

    # -------------------------
    # PF2: outboard shaping pair near top of the plasma
    # -------------------------
    # Just outside the outboard plasma and slightly above the upper lobe,
    # but still inside the vessel.
    Z_shaping = Z_pl_top + 0.3          # ~30 cm above the target plasma
    Z_shaping = min(Z_shaping, Z_top_outer - 0.15)  # keep below the vessel roof

    R_pf2 = R_pl_outboard + 0.3
    dR_pf2 = 0.3
    dZ_pf2 = 0.6

    # -------------------------
    # PF3: inboard shaping pair near top of the plasma
    # -------------------------
    # On the high-field side, between CS and inner wall, at the same
    # height as PF2, to strengthen elongation and positive triangularity.
    R_cs_max = R_cs + dR_cs
    R_pf3 = max(R_pl_inboard - 0.3, R_cs_max + 0.15)
    dR_pf3 = 0.25
    Z_pf3 = Z_shaping
    dZ_pf3 = 0.6

    # -------------------------
    # Coil dictionary
    # -------------------------
    coils = {
        # PF1: vertical field coils outside the vessel
        "PF1U": (R_pf1, +Z_pf1, dR_pf1, dZ_pf1),
        "PF1L": (R_pf1, -Z_pf1, dR_pf1, dZ_pf1),

        # PF2: outboard shaping (near upper external lobe)
        "PF2U": (R_pf2, +Z_shaping, dR_pf2, dZ_pf2),
        "PF2L": (R_pf2, -Z_shaping, dR_pf2, dZ_pf2),

        # PF3: inboard shaping (high-field side)
        "PF3U": (R_pf3, +Z_shaping, dR_pf3, dZ_pf3),
        "PF3L": (R_pf3, -Z_shaping, dR_pf3, dZ_pf3),

        # Central solenoid
        "CS": (R_cs, 0.0, dR_cs, dZ_cs),
    }

    geom = {
        "R0": R0,
        "A": A,
        "kappa": kappa,
        "delta": delta,
        "a": a,
        "R_plasma": Rp,
        "Z_plasma": Zp,
        "R_inner": R_inner,
        "Z_inner": Z_inner,
        "R_outer": R_outer,
        "Z_outer": Z_outer,
        "coils": coils,
        # handy values for logging / debugging
        "R_pl_inboard": R_pl_inboard,
        "R_pl_outboard": R_pl_outboard,
        "Z_pl_top": Z_pl_top,
        "Z_pl_bot": Z_pl_bot,
        "R_inboard_mid": R_inboard_mid,
        "R_inner_out_mid": R_inner_out_mid,
        "R_out_mid": R_out_mid,
        "Z_top_outer": Z_top_outer,
    }
    return geom


def plot_star_geometry(geom, show: bool = True, ax=None):
    """
    Plot a quick overview of the geometry: vessel, inner wall,
    target plasma boundary and PF/CS coil positions.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 9))

    ax.plot(geom["R_outer"], geom["Z_outer"], "k-", label="Outer vessel")
    ax.plot(geom["R_inner"], geom["Z_inner"], "k--", label="Inner wall")
    ax.plot(
        geom["R_plasma"], geom["Z_plasma"],
        color="tab:orange", label="Plasma boundary (target)",
    )

    # Geometric major radius
    ax.axvline(
        geom["R0"], linestyle=":", color="tab:orange",
        alpha=0.7, label="R0",
    )

    # PF / CS coils as circles with labels
    for name, (Rc, Zc, dR, dZ) in geom["coils"].items():
        circle = plt.Circle((Rc, Zc), radius=min(dR, dZ), fill=False, color="k")
        ax.add_patch(circle)
        ax.text(Rc, Zc, name, ha="center", va="center", fontsize=9)

    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("Geometry sketch – STAR-like spherical tokamak (D-shape, κ≈1.8)")
    ax.grid(True)
    ax.legend(loc="upper right")

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def make_star_machine(
    R0: float = 4.0,
    A: float = 1.7,
    kappa: float = 1.8,   # consistent with default geometry
    delta: float = 0.30,
):
    """
    Build a FreeGS/FreeGSNKE Machine with the STAR-like D-shaped geometry.

    Returns
    -------
    tokamak : machine.Machine
        Machine object with walls, coils and limiter set.
    geom : dict
        Geometry dictionary returned by :func:`star_geometry`.
    """
    # --- Geometry (D-shape) ---
    geom = star_geometry(R0=R0, A=A, kappa=kappa, delta=delta)

    # --- Walls ---
    vessel_wall = machine.Wall(geom["R_outer"], geom["Z_outer"])
    inner_wall = machine.Wall(geom["R_inner"], geom["Z_inner"])
    limiter = inner_wall

    # --- Coils (PF + CS) ---
    coils_for_machine = []  # list of (label, coil)
    for label, (Rc, Zc, dR, dZ) in geom["coils"].items():
        coil = machine.MultiCoil(Rc, Zc, dR, dZ)
        coil.label = label
        coils_for_machine.append((label, coil))

    # --- Build Machine ---
    tokamak = machine.Machine(
        coils_for_machine,
        wall=vessel_wall,
    )

    tokamak.limiter = limiter
    tokamak.active_coils = [label for label, _ in coils_for_machine]
    tokamak.passive_coils = []
    tokamak.R0 = geom["R0"]
    tokamak.geom = geom
    tokamak.coils_dict = {label: coil for label, coil in coils_for_machine}

    return tokamak, geom


if __name__ == "__main__":
    # Quick demo to check that the geometry looks reasonable
    geom = star_geometry()
    print("R_pl_inboard         =", geom["R_pl_inboard"])
    print("R_inboard_mid (wall) =", geom["R_inboard_mid"])
    print("CS (Rc, Zc, dR, dZ)  =", geom["coils"]["CS"])
    plot_star_geometry(geom)

