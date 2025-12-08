"""
animate_star_bean.py

Animation of a quasi–static ramp-up of the STAR-like bean equilibrium.

The final state matches the refined micro-scan result:
  CS  = 0.80 MA
  PF1 = -0.23 MA
  PF2 = 0.00 MA
  PF3 = 1.10 MA

The ramp-up is implemented by scaling all currents and pressure by
a factor f(t) from F_START to F_END, solving a static equilibrium
at each frame.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from tqdm import tqdm  # <- para la barra de progreso

from freegsnke import equilibrium_update, GSstaticsolver
from freegsnke.jtor_update import ConstrainPaxisIp

from star_machine import make_star_machine
from analyze_star_shape import shape_from_separatrix


# =========================
# ANIMATION PARAMETERS
# =========================
T_TOTAL = 10.0   # total duration [s] (for the MP4 timing)
FPS     = 30     # frames per second
N_FRAMES = int(T_TOTAL * FPS)

# Ramp-up factor: from f_start to f_end
F_START = 0.3
F_END   = 1.0


def set_star_currents(tokamak, CS, PF1, PF2, PF3):
    """
    Assign STAR-like coil currents (in A) to the machine.
    """
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


def main():
    # -------------------------
    # 1) Geometry and machine
    # -------------------------
    tokamak, geom = make_star_machine(
        R0=4.0,
        A=1.7,
        kappa=1.8,   # same geometry as in star_equilibrium / diagnostics
        delta=0.30,
    )

    # Domain based on the outer wall
    R_outer = geom["R_outer"]
    Z_outer = geom["Z_outer"]
    Rmin = float(R_outer.min() - 0.5)
    Rmax = float(R_outer.max() + 0.5)
    Zmin = float(Z_outer.min() - 0.5)
    Zmax = float(Z_outer.max() + 0.5)

    # Equilibrium with a slightly coarser grid so the animation is faster
    nx, ny = 81, 161
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=Rmin,
        Rmax=Rmax,
        Zmin=Zmin,
        Zmax=Zmax,
        nx=nx,
        ny=ny,
    )
    solver = GSstaticsolver.NKGSsolver(eq)

    # -------------------------
    # 2) Final plasma and coil parameters
    # -------------------------
    Ip_final    = 8.0e5   # 0.8 MA
    paxis_final = 2.0e3   # 2 kPa
    fvac        = 0.5

    # <<< FINAL STATE = REFINED MICRO-SCAN CASE >>>
    CS_final  = 0.8e6
    PF1_final = -0.23e6
    PF2_final = 0.0
    PF3_final = 1.10e6

    # List of ramp-up factors from F_START to F_END
    f_list = np.linspace(F_START, F_END, N_FRAMES)

    # -------------------------
    # 3) Prepare figure
    # -------------------------
    fig, ax = plt.subplots(figsize=(5, 9))

    def init_axis():
        ax.clear()
        ax.set_xlim(Rmin, Rmax)
        ax.set_ylim(Zmin, Zmax)
        ax.set_aspect("equal")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_title("STAR-like bean ramp-up (quasi-static)")
        # Vessel and target geometry
        ax.plot(geom["R_outer"], geom["Z_outer"], "k-", lw=2, label="Vessel")
        ax.plot(
            geom["R_plasma"],
            geom["Z_plasma"],
            "--",
            color="tab:orange",
            lw=1.5,
            label="Plasma target (geom)",
        )
        # Coil positions (fixed)
        tokamak.plot(axis=ax, show=False)
        ax.legend(loc="upper right")

    init_axis()

    # -------------------------
    # 4) Per-frame update function
    # -------------------------
    def update(frame_index):
        f = f_list[frame_index]

        # Currents and profiles at this "time"
        Ip    = Ip_final * f
        paxis = paxis_final * f

        CS  = CS_final * f
        PF1 = PF1_final * f
        PF2 = PF2_final * f   # remains ≈ 0
        PF3 = PF3_final * f

        # Update currents in the machine
        set_star_currents(tokamak, CS, PF1, PF2, PF3)

        # Profiles for this state
        profiles = ConstrainPaxisIp(
            eq=eq,
            paxis=paxis,
            Ip=Ip,
            fvac=fvac,
            alpha_m=1.8,
            alpha_n=1.2,
        )
        profiles.diverted_core_mask = np.ones_like(eq.psi(), dtype=bool)

        # Solve using the previous psi as initial guess
        solver.solve(
            eq=eq,
            profiles=profiles,
            constrain=None,
            target_relative_tolerance=1e-6,
            verbose=False,
        )

        # Separatrix for this state (if any)
        try:
            shape = shape_from_separatrix(eq, geom)
            have_shape = True
        except Exception:
            # If there is not yet a closed separatrix (very small f),
            # skip plotting the shape
            have_shape = False

        # Redraw axes, vessel, target, and coils
        init_axis()

        # Poloidal flux contours
        eq.plot(axis=ax, show=False)

        # Current separatrix if available
        if have_shape:
            ax.plot(
                shape["R_sep"],
                shape["Z_sep"],
                "r-",
                lw=2,
                label="Separatrix (eq)",
            )
            kappa_pl = shape["kappa_plasma"]
            delta_pl = shape["delta_u"]
        else:
            kappa_pl = np.nan
            delta_pl = np.nan

        # Text overlay with parameters
        txt = rf"$f={f:.2f}$, " rf"$I_p={Ip/1e6:.2f}\,\mathrm{{MA}}$"
        if have_shape:
            txt += (
                rf", $\kappa={kappa_pl:.2f}$"
                rf", $\delta\approx{delta_pl:.2f}$"
            )

        ax.text(
            0.02,
            0.02,
            txt,
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

        return []

    # -------------------------
    # 5) Build animation
    # -------------------------
    anim = FuncAnimation(
        fig,
        update,
        frames=N_FRAMES,
        interval=1000 / FPS,
        blit=False,
        repeat=True,
    )
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    video = os.path.join(results_dir, "star_bean_ramp_refined.mp4")

    # ====== BARRA DE PROGRESO EN EL GUARDADO ======
    with tqdm(total=N_FRAMES, desc="Guardando animación") as pbar:
        def progress(current_frame, total_frames):
            # actualizamos +1 cada vez que se llama
            pbar.update(1)

        anim.save(
            video,
            fps=FPS,
            dpi=150,
            progress_callback=progress,
        )
    # =============================================

    print("Animation saved to star_bean_ramp_refined.mp4")
    # plt.show()


if __name__ == "__main__":
    main()

