"""
analyze_star_shape.py  (CAD-first)

Analyze and plot STAR-like equilibria using CAD/DXF-backed machine geometry.

- NO dependency on star_machine.py
- Loads machine from DXF via star_machine_cad.make_star_machine_from_cad
- Supports segmented coils (CS*, PF1*, PF2*, PF3*) via family-current distribution
- Provides shape_from_separatrix(eq, geom) as a lightweight utility that is safe to import
  from scan scripts (it does NOT import star_machine).

Run example (from project root):
    python analyze_star_shape.py --dxf cad/star_baseline.dxf

In Colab with your venv:
    /content/fgvenv/bin/python analyze_star_shape.py --dxf cad/star_baseline.dxf --silence-solver
"""

from __future__ import annotations

import os
import types
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from freegsnke import equilibrium_update, GSstaticsolver
from freegsnke.jtor_update import ConstrainPaxisIp

from star_machine_cad import make_star_machine_from_cad, CADImportOptions, plot_cad_geometry
import config_star_bean as cfg


# -----------------------------------------------------------------------------
# 1) Shape extraction: separatrix + basic geometry
# -----------------------------------------------------------------------------

def shape_from_separatrix(eq, geom=None):
    """
    Extract separatrix as a contour at psi = psi_bndry and compute:
      R0_plasma, a_plasma, A_plasma, kappa_plasma, delta_u, delta_l
    Also returns (R_sep, Z_sep) and (R_ax, Z_ax).

    Parameters
    ----------
    eq : FreeGSNKE Equilibrium
    geom : dict or None
        Kept for interface compatibility (not required for computation).

    Returns
    -------
    dict with keys:
      R_ax, Z_ax, R0_plasma, a_plasma, A_plasma, kappa_plasma,
      delta_u, delta_l, R_sep, Z_sep
    """
    psi = eq.psi()
    psi_sep = eq.psi_bndry
    R = eq.R
    Z = eq.Z

    R_ax, Z_ax = eq.magneticAxis()[:2]

    # Contours at psi = psi_sep
    fig, ax = plt.subplots()
    cs = ax.contour(R, Z, psi, levels=[psi_sep])
    plt.close(fig)

    segs = cs.allsegs[0]
    if not segs or len(segs) == 0:
        raise RuntimeError("Did not find any psi = psi_bndry contour (separatrix).")

    # Choose the segment that encloses the magnetic axis; fallback to closest centroid
    chosen = None
    for seg in segs:
        R_path = seg[:, 0]
        Z_path = seg[:, 1]
        if (R_path.min() < R_ax < R_path.max()) and (Z_path.min() < Z_ax < Z_path.max()):
            chosen = seg
            break

    if chosen is None:
        best_d2 = None
        for seg in segs:
            Rc = float(seg[:, 0].mean())
            Zc = float(seg[:, 1].mean())
            d2 = (Rc - R_ax) ** 2 + (Zc - Z_ax) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                chosen = seg

    R_sep = chosen[:, 0]
    Z_sep = chosen[:, 1]

    R_min = float(R_sep.min())
    R_max = float(R_sep.max())
    Z_min = float(Z_sep.min())
    Z_max = float(Z_sep.max())

    a_plasma = 0.5 * (R_max - R_min)
    if a_plasma <= 0:
        raise RuntimeError("Invalid separatrix: computed a_plasma <= 0.")

    R0_plasma = 0.5 * (R_max + R_min)
    A_plasma = R0_plasma / a_plasma
    kappa_plasma = 0.5 * (Z_max - Z_min) / a_plasma

    idx_top = int(np.argmax(Z_sep))
    idx_bot = int(np.argmin(Z_sep))
    R_top = float(R_sep[idx_top])
    R_bot = float(R_sep[idx_bot])

    delta_u = (R0_plasma - R_top) / a_plasma
    delta_l = (R0_plasma - R_bot) / a_plasma

    return dict(
        R_ax=float(R_ax),
        Z_ax=float(Z_ax),
        R0_plasma=float(R0_plasma),
        a_plasma=float(a_plasma),
        A_plasma=float(A_plasma),
        kappa_plasma=float(kappa_plasma),
        delta_u=float(delta_u),
        delta_l=float(delta_l),
        R_sep=np.asarray(R_sep, dtype=float),
        Z_sep=np.asarray(Z_sep, dtype=float),
    )


# -----------------------------------------------------------------------------
# 2) Family currents (segmented coils safe)
# -----------------------------------------------------------------------------

def apply_star_family_currents(tokamak, CS, PF1, PF2, PF3, *, mode: str = "area"):
    """
    Apply FAMILY total currents to a tokamak with possible segmented coils.

    mode:
      - "area"  : split family totals by segment area weights (recommended)
      - "equal" : equal split among segments
      - "same"  : each segment gets full family current (NOT recommended)
    """
    try:
        from star_machine_cad import apply_group_currents
    except Exception:
        apply_group_currents = None

    mode = str(mode).lower().strip()
    family = {"CS": float(CS), "PF1": float(PF1), "PF2": float(PF2), "PF3": float(PF3)}

    if apply_group_currents is not None and hasattr(tokamak, "coil_groups"):
        apply_group_currents(tokamak, family, mode=mode)

        # Zero other coils not in these families
        fam_set = {"CS", "PF1", "PF2", "PF3"}
        grouped = set()
        try:
            for fam, labs in (tokamak.coil_groups or {}).items():
                if str(fam).upper() in fam_set:
                    grouped |= set(labs)
        except Exception:
            grouped = set()

        for label, coil in tokamak.coils:
            lab = str(label).strip().upper()
            if lab not in grouped:
                try:
                    coil.current = 0.0
                except Exception:
                    pass
        return

    # Fallback legacy mapping
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


# -----------------------------------------------------------------------------
# 3) Robust masks/copy for ConstrainPaxisIp
# -----------------------------------------------------------------------------

def _full_core_mask(eq) -> np.ndarray:
    return np.ones(eq.R.shape, dtype=bool)


def _ensure_masks(profiles, eq):
    base = _full_core_mask(eq)

    # Common offenders
    for name in ("diverted_core_mask", "limiter_core_mask"):
        v = getattr(profiles, name, None)
        if v is None or np.asarray(v).shape != base.shape:
            setattr(profiles, name, base.copy())

    # Future-proof
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

    # Patch copy defensively
    _orig_copy = profiles.copy

    def _safe_copy(self):
        _ensure_masks(self, eq)
        return _orig_copy()

    profiles.copy = types.MethodType(_safe_copy, profiles)
    return profiles


# -----------------------------------------------------------------------------
# 4) Build & solve equilibrium using CAD machine
# -----------------------------------------------------------------------------

def build_equilibrium_from_cad(
    *,
    dxf_path: str | None = None,
    unit_scale: float | None = None,
    resample_walls: str = "auto",
    n_wall: int = 801,
    n_inner: int = 801,
    min_wall_pts: int = 200,
    enforce_ccw: bool = True,
    canonical_start: bool = True,
    nx: int | None = None,
    ny: int | None = None,
    target_rel_tol: float | None = None,
    silence_solver: bool = False,
    coil_group_mode: str | None = None,
):
    """
    Solve a STAR-like equilibrium using CAD geometry (DXF) and cfg parameters.

    Returns: (eq, tokamak, geom, shape)
    """
    # CAD import options
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

    # Grid and tolerance defaults
    if nx is None:
        nx = int(getattr(cfg, "nx_eq", 65))
    if ny is None:
        ny = int(getattr(cfg, "ny_eq", 129))
    if target_rel_tol is None:
        target_rel_tol = float(getattr(cfg, "target_rel_tol", 1e-8))

    if coil_group_mode is None:
        coil_group_mode = str(getattr(cfg, "coil_group_mode", "area")).lower().strip()

    # Domain from CAD outer wall
    R_outer = np.asarray(geom["R_outer"], dtype=float)
    Z_outer = np.asarray(geom["Z_outer"], dtype=float)
    margin = float(getattr(cfg, "margin_RZ", 0.5))

    Rmin = float(R_outer.min() - margin)
    Rmax = float(R_outer.max() + margin)
    Zmin = float(Z_outer.min() - margin)
    Zmax = float(Z_outer.max() + margin)

    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=Rmin, Rmax=Rmax,
        Zmin=Zmin, Zmax=Zmax,
        nx=int(nx), ny=int(ny),
    )

    solver = GSstaticsolver.NKGSsolver(eq)

    # Optional silencing of solver output
    ctx = None
    if silence_solver:
        import contextlib
        ctx = contextlib.ExitStack()
        devnull = ctx.enter_context(open(os.devnull, "w"))
        ctx.enter_context(contextlib.redirect_stdout(devnull))
        ctx.enter_context(contextlib.redirect_stderr(devnull))

    try:
        f_list = tuple(getattr(cfg, "f_list_equilibrium", (0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.00)))
        tol_ramp  = float(getattr(cfg, "target_rel_tol_ramp", 3e-6))
        tol_final = float(target_rel_tol)

        for j, f in enumerate(f_list):
            # Apply FAMILY totals (segmented-coil safe)
            apply_star_family_currents(
                tokamak,
                CS=f * float(getattr(cfg, "CS_current")),
                PF1=f * float(getattr(cfg, "PF1_current")),
                PF2=f * float(getattr(cfg, "PF2_current")),
                PF3=f * float(getattr(cfg, "PF3_current")),
                mode=coil_group_mode,
            )

            profiles = _make_profiles(
                eq,
                paxis=f * float(getattr(cfg, "paxis")),
                Ip=f * float(getattr(cfg, "Ip")),
                fvac=float(getattr(cfg, "fvac")),
                alpha_m=float(getattr(cfg, "alpha_m")),
                alpha_n=float(getattr(cfg, "alpha_n")),
            )

            this_tol = tol_final if (j == len(f_list) - 1) else tol_ramp

            solver.solve(
                eq=eq,
                profiles=profiles,
                constrain=None,
                target_relative_tolerance=float(this_tol),
                verbose=not silence_solver,
            )
    finally:
        if ctx is not None:
            ctx.close()

    shape = shape_from_separatrix(eq, geom)

    return eq, tokamak, geom, shape


# -----------------------------------------------------------------------------
# 5) Plot helper
# -----------------------------------------------------------------------------

def plot_analysis(eq, geom, shape, *, show=True, savepath: str | None = None, title: str = "STAR CAD equilibrium analysis"):
    fig, ax = plt.subplots(figsize=(6, 10))
    eq.plot(axis=ax, show=False)

    # CAD walls / targets
    ax.plot(geom["R_outer"], geom["Z_outer"], "k-", lw=2, label="CAD outer wall")
    if "R_inner" in geom and "Z_inner" in geom:
        ax.plot(geom["R_inner"], geom["Z_inner"], "k--", lw=1.5, label="CAD inner wall")
    if "R_plasma" in geom and "Z_plasma" in geom:
        ax.plot(geom["R_plasma"], geom["Z_plasma"], lw=1.5, label="CAD plasma target")

    # Separatrix
    ax.plot(shape["R_sep"], shape["Z_sep"], lw=2.2, label="Separatrix (eq)")

    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    txt = (
        rf"$R_0^{{pl}}={shape['R0_plasma']:.2f}\,$m  "
        rf"$a={shape['a_plasma']:.2f}\,$m  "
        rf"$A={shape['A_plasma']:.2f}$" "\n"
        rf"$\kappa={shape['kappa_plasma']:.2f}$  "
        rf"$\delta_u,\delta_l={shape['delta_u']:.2f},{shape['delta_l']:.2f}$" "\n"
        rf"$R_{ax}={shape['R_ax']:.2f}\,$m  "
        rf"$Z_{ax}={shape['Z_ax']:.2f}\,$m"
    )
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.legend(loc="upper right")
    plt.tight_layout()

    if savepath is not None:
        out = Path(savepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------------------------------------------------------
# 6) CLI
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dxf", default=None, help="Path to DXF (default: cad/star_baseline.dxf relative to this script)")
    ap.add_argument("--unit-scale", type=float, default=None, help="Override CAD units (m per DXF unit). Default: infer from INSUNITS.")
    ap.add_argument("--resample-walls", default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--n-wall", type=int, default=801)
    ap.add_argument("--n-inner", type=int, default=801)
    ap.add_argument("--min-wall-pts", type=int, default=200)
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--coil-group-mode", default=None, choices=["area", "equal", "same"])
    ap.add_argument("--silence-solver", action="store_true")
    ap.add_argument("--save", default=None, help="Save figure to this path (png).")
    args = ap.parse_args()

    # Default DXF: cad/star_baseline.dxf next to this script
    if args.dxf is None:
        here = Path(__file__).resolve().parent
        args.dxf = str((here / "cad" / "star_baseline.dxf").resolve())

    eq, tokamak, geom, shape = build_equilibrium_from_cad(
        dxf_path=args.dxf,
        unit_scale=args.unit_scale,
        resample_walls=args.resample_walls,
        n_wall=args.n_wall,
        n_inner=args.n_inner,
        min_wall_pts=args.min_wall_pts,
        nx=args.nx,
        ny=args.ny,
        target_rel_tol=args.tol,
        silence_solver=args.silence_solver,
        coil_group_mode=args.coil_group_mode,
    )

    # Print summary
    print("\n--- CAD machine ---")
    print("cad_path   =", geom.get("cad_path"))
    print("unit_scale =", geom.get("unit_scale"))
    print("coils      =", sorted(list(geom.get("coils", {}).keys())))
    if "coil_groups" in geom:
        fams = {k: len(v) for k, v in (geom.get("coil_groups", {}) or {}).items()}
        print("families   =", fams)

    print("\n--- Targets (cfg) ---")
    print(f"R0_geom={cfg.R0_geom:.3f}  A_geom={cfg.A_geom:.3f}  kappa_geom={cfg.kappa_geom:.3f}  delta_geom={cfg.delta_geom:.3f}")

    print("\n--- Plasma from separatrix ---")
    print(f"R0_plasma={shape['R0_plasma']:.3f}  a={shape['a_plasma']:.3f}  A={shape['A_plasma']:.3f}")
    print(f"kappa={shape['kappa_plasma']:.3f}  delta_u={shape['delta_u']:.3f}  delta_l={shape['delta_l']:.3f}")
    print(f"R_ax={shape['R_ax']:.3f}  Z_ax={shape['Z_ax']:.3f}")

    plot_analysis(eq, geom, shape, show=True, savepath=args.save)


if __name__ == "__main__":
    main()

