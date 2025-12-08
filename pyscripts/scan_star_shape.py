"""
scan_star_shape.py

Local micro–scan of PF/CS currents (CS, PF1, PF3) to refine the
STAR-like bean equilibrium.

  - CS  = 0.8  MA (fixed)
  - PF2 = 0.0  MA (fixed)
  - PF1 scanned around ≈ -0.20 MA
  - PF3 scanned around ≈ +1.00 MA

Refined misfit metric:
  - Uses the magnetic axis R_ax vs. a target major radius R0_target = 4.0 m
  - Prefers a moderate average triangularity delta_bar ~ 0.4
  - Keeps elongation kappa close to a target value
  - Penalizes δ < 0, very thin plasmas, and large R0 shifts
"""

import time
import numpy as np
import multiprocessing as mp

from freegsnke import equilibrium_update, GSstaticsolver
from freegsnke.jtor_update import ConstrainPaxisIp

from star_machine import make_star_machine
from analyze_star_shape import shape_from_separatrix


# Global penalties for the misfit functional
PENALTY_NEG_DELTA = 10.0   # strong penalty for negative triangularity
PENALTY_THIN      = 5.0    # penalty for very thin plasmas
PENALTY_R0_SHIFT  = 3.0    # extra penalty for large major-radius shift


def set_star_currents(tokamak, CS, PF1, PF2, PF3):
    """
    Assign coil currents (in A) to the STAR-like machine:

      CS      -> central solenoid
      PF1U/L  -> PF1 pair
      PF2U/L  -> PF2 pair
      PF3U/L  -> PF3 pair

    All other coils are set to zero current.
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


# ---------- Worker that runs in a SUBPROCESS ----------

def _worker_solve(
    CS,
    PF1,
    PF2,
    PF3,
    Ip,
    paxis,
    fvac,
    nx,
    ny,
    geom_kwargs,
    out_queue,
):
    """
    This function is executed in a subprocess.

    It builds the machine, solves the equilibrium, evaluates the shape
    and the misfit, and pushes the result into out_queue.
    """
    try:
        # 1) Machine + geometry
        tokamak, geom = make_star_machine(**geom_kwargs)
        set_star_currents(tokamak, CS, PF1, PF2, PF3)

        # 2) Numerical domain
        R_outer = geom["R_outer"]
        Z_outer = geom["Z_outer"]
        Rmin = float(R_outer.min() - 0.5)
        Rmax = float(R_outer.max() + 0.5)
        Zmin = float(Z_outer.min() - 0.5)
        Zmax = float(Z_outer.max() + 0.5)

        eq = equilibrium_update.Equilibrium(
            tokamak=tokamak,
            Rmin=Rmin,
            Rmax=Rmax,
            Zmin=Zmin,
            Zmax=Zmax,
            nx=nx,
            ny=ny,
        )

        # 3) Profiles
        profiles = ConstrainPaxisIp(
            eq=eq,
            paxis=paxis,
            Ip=Ip,
            fvac=fvac,
            alpha_m=1.8,
            alpha_n=1.2,
        )
        profiles.diverted_core_mask = np.ones_like(eq.psi(), dtype=bool)

        # 4) Solver (silent here to avoid spam)
        solver = GSstaticsolver.NKGSsolver(eq)
        solver.solve(
            eq=eq,
            profiles=profiles,
            constrain=None,
            target_relative_tolerance=1e-5,
            verbose=False,
        )

        # 5) Geometric shape
        shape = shape_from_separatrix(eq, geom)

        required = (
            "R0_plasma",
            "kappa_plasma",
            "delta_u",
            "delta_l",
            "a_plasma",
        )
        if not all(k in shape for k in required):
            out_queue.put({"error": f"shape keys={list(shape.keys())}"})
            return

        # Geometric targets
        R0_target    = geom["R0"]   # ~ 4.0 m
        kappa_target = 2.3          # approximate STAR-like target
        delta_target = 0.4          # moderate positive triangularity

        # Plasma parameters
        R0_pl    = shape["R0_plasma"]
        kappa_pl = shape["kappa_plasma"]
        du       = shape["delta_u"]
        dl       = shape["delta_l"]
        a_pl     = shape["a_plasma"]

        # Magnetic axis (more relevant than geometric R0_plasma)
        R_ax, Z_ax = eq.magneticAxis()[:2]

        # --- Base misfit (dimensionless) ---------------------------------
        # Each term is normalized by a “typical tolerance” so that
        # contributions ~1 correspond to “reasonable error”.
        dR_ax = (R_ax - R0_target)          # want |dR_ax| ≲ 0.2 m
        term_Rax = (dR_ax / 0.2) ** 2

        dk_rel = (kappa_pl - kappa_target) / max(kappa_target, 1e-6)
        term_kappa = (dk_rel / 0.25) ** 2   # 25% error -> contribution ~1

        delta_bar = 0.5 * (du + dl)
        term_delta = ((delta_bar - delta_target) / 0.2) ** 2  # ±0.2 OK

        misfit = np.sqrt(term_Rax + term_kappa + term_delta)

        # --- Additional geometric penalties ------------------------------

        # 1) Negative triangularity (proportional to |δ^-|)
        neg_u = min(du, 0.0)
        neg_l = min(dl, 0.0)
        if neg_u < 0.0 or neg_l < 0.0:
            misfit += PENALTY_NEG_DELTA * (abs(neg_u) + abs(neg_l))

        # 2) Large shift of the geometric R0_plasma
        if abs(R0_pl - R0_target) > 1.0:
            misfit += PENALTY_R0_SHIFT * (abs(R0_pl - R0_target) - 1.0)

        # 3) Plasma too thin (small minor radius a_plasma)
        a_min = 1.0  # want at least ~1 m
        if a_pl < a_min:
            misfit += PENALTY_THIN * (a_min - a_pl)

        result = {
            "CS": CS,
            "PF1": PF1,
            "PF2": PF2,
            "PF3": PF3,
            "misfit": float(misfit),
            "shape": shape,
            "R_ax": float(R_ax),
            "Z_ax": float(Z_ax),
        }
        out_queue.put(result)

    except Exception as e:
        out_queue.put({"error": repr(e)})


# ---------- Wrapper in the main process ----------

def run_case(
    CS,
    PF1,
    PF2,
    PF3,
    Ip=8.0e5,
    paxis=2.0e3,
    fvac=0.5,
    nx=65,
    ny=129,
    max_time_s=25.0,
):
    """
    Launch _worker_solve in a subprocess with a hard timeout.

    If the computation takes longer than max_time_s, the subprocess
    is killed and None is returned.
    """
    geom_kwargs = dict(R0=4.0, A=1.7, kappa=1.8, delta=0.30)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_worker_solve,
        args=(CS, PF1, PF2, PF3, Ip, paxis, fvac, nx, ny, geom_kwargs, q),
    )

    t0 = time.time()
    p.start()
    p.join(max_time_s)
    elapsed = time.time() - t0

    if p.is_alive():
        # Hard timeout: kill the subprocess
        print(f"    → hard timeout (> {max_time_s:.0f} s), killing process")
        p.terminate()
        p.join()
        return None

    if q.empty():
        print("    → worker returned no result (internal error)")
        return None

    res = q.get()
    if "error" in res:
        print(f"    → error in worker: {res['error']}")
        return None

    print(f"    Solver + shape in {elapsed:5.1f} s")
    return res


# ---------- Main micro–scan loop ----------

def main():
    # Fixed CS close to a reasonable case
    CS0 = 0.8e6   # 0.800 MA
    PF2_0 = 0.0   # PF2 kept off in this refinement

    # Local micro–scan ranges for PF1 and PF3 (in MA)
    PF1_values = 1e6 * np.array([-0.25, -0.23, -0.20, -0.18, -0.15])
    PF3_values = 1e6 * np.array([0.80, 0.90, 1.00, 1.10])

    combos = [
        (CS0, pf1, PF2_0, pf3)
        for pf1 in PF1_values
        for pf3 in PF3_values
    ]

    print(f"Total combinations to test (local micro-scan): {len(combos)}\n")

    results = []

    for i, (CS, PF1, PF2, PF3) in enumerate(combos, start=1):
        print(
            f"[{i}/{len(combos)}] "
            f"CS={CS/1e6:.3f} MA, "
            f"PF1={PF1/1e6:.3f} MA, "
            f"PF2={PF2/1e6:.3f} MA, "
            f"PF3={PF3/1e6:.3f} MA"
        )

        res = run_case(
            CS,
            PF1,
            PF2,
            PF3,
            Ip=8.0e5,
            paxis=2.0e3,
            fvac=0.5,
            nx=65,
            ny=129,
            max_time_s=25.0,
        )

        if res is None:
            print("    → case discarded\n")
            continue

        shape = res["shape"]
        print(
            f"    → misfit={res['misfit']:.3e}, "
            f"R_ax={res['R_ax']:.3f}, "
            f"R0_pl={shape['R0_plasma']:.3f}, "
            f"a_pl={shape['a_plasma']:.3f}, "
            f"kappa={shape['kappa_plasma']:.3f}, "
            f"δu={shape['delta_u']:.3f}, δl={shape['delta_l']:.3f}\n"
        )

        results.append(res)

    if not results:
        print("\nNo valid cases found in this micro-scan.")
        return

    # Sort by misfit and report the best candidate
    results.sort(key=lambda r: r["misfit"])
    best = results[0]
    s = best["shape"]

    print("\n=== Best case found in the local micro-scan ===")
    print(f"CS  = {best['CS']/1e6:.3f} MA")
    print(f"PF1 = {best['PF1']/1e6:.3f} MA")
    print(f"PF2 = {best['PF2']/1e6:.3f} MA")
    print(f"PF3 = {best['PF3']/1e6:.3f} MA")
    print(f"misfit   = {best['misfit']:.3e}")
    print(f"R_ax     = {best['R_ax']:.3f} m")
    print(f"R0_pl    = {s['R0_plasma']:.3f} m")
    print(f"a_plasma = {s['a_plasma']:.3f} m")
    print(f"A_plasma = {s['A_plasma']:.3f}")
    print(f"kappa    = {s['kappa_plasma']:.3f}")
    print(f"delta_u  = {s['delta_u']:.3f}")
    print(f"delta_l  = {s['delta_l']:.3f}")


if __name__ == "__main__":
    main()

