# star_equilibrium.py
"""
Refined STAR-like equilibrium using parameters in config_star_bean.py,
with CAD/DXF-backed geometry (star_machine_cad.py) and segmented-coil support.

What this version fixes/does:
- If CAD has segmented coils (CS1M, CS2U, ...), we apply a *single family total*
  (CS_current, PF1_current, ...) and distribute it across segments using
  star_machine_cad.apply_group_currents(..., mode=cfg.coil_group_mode).
- Continuation over f_list_equilibrium is printed cleanly; solver spam
  (e.g. "Update resizing triggered due to failure to find a critical points.")
  is redirected to results/solver_noise.log.
- Shape is computed from a limiter/wall-limited LCFS (robust even when no X-point
  critical points are found), via analyze_star_lcfs.shape_from_lcfs_limiter.
- Robust mask handling so FreeGSNKE copy() doesn't fail on None masks.
"""

from __future__ import annotations

import os
import types
import contextlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from freegsnke import equilibrium_update, GSstaticsolver
from freegsnke.jtor_update import ConstrainPaxisIp

from star_machine_cad import make_star_machine_from_cad, CADImportOptions
from analyze_star_lcfs import shape_from_lcfs_limiter  # <-- NEW robust shape extraction
import config_star_bean as cfg


# -------------------------
# Paths / logging
# -------------------------

def _repo_results_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "results"


def _solver_noise_log() -> Path:
    d = _repo_results_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "solver_noise.log"


@contextlib.contextmanager
def _redirect_solver_noise(enabled: bool = True, path: Path | None = None):
    """
    Redirect stdout/stderr (solver spam) into a log file, while keeping our own prints.
    """
    if not enabled:
        yield
        return

    if path is None:
        path = _solver_noise_log()

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f, \
         contextlib.redirect_stdout(f), \
         contextlib.redirect_stderr(f):
        yield


# -------------------------
# FreeGSNKE copy(None) hardening (extra safety)
# -------------------------

def _patch_profiles_copy_once():
    """
    Patch ConstrainPaxisIp.copy() to ensure common masks are never None.
    This prevents: TypeError("Cannot copy <class 'NoneType'> without deepcopying")
    """
    from freegsnke.jtor_update import ConstrainPaxisIp as _C

    if getattr(_C, "_safe_copy_patched", False):
        return

    _orig_copy = _C.copy

    def _safe_copy(self, *args, **kwargs):
        eq = getattr(self, "eq", None) or getattr(self, "_eq", None)
        if eq is not None:
            base = np.ones(eq.R.shape, dtype=bool)
            for name in ("diverted_core_mask", "limiter_core_mask"):
                try:
                    if getattr(self, name, None) is None:
                        setattr(self, name, base)
                except Exception:
                    pass
        return _orig_copy(self, *args, **kwargs)

    _C.copy = _safe_copy
    _C._safe_copy_patched = True


def _patch_copy_into_allow_none_once():
    """
    Patch freegsnke.copying.copy_into to allow copying None when strict=False,
    across multiple possible signatures (including allow_deepcopy kwarg).
    """
    import freegsnke.copying as _copying
    import freegsnke.jtor_update as _jtor

    if getattr(_copying, "_allow_none_patched", False):
        return

    _orig = _copying.copy_into

    def _copy_into_patched(src, dst, name, *args, **kwargs):
        # Extract strict/mutable from args/kwargs for compatibility
        mutable = kwargs.get("mutable", False)
        strict = kwargs.get("strict", True)

        if len(args) >= 1:
            mutable = args[0]
        if len(args) >= 2:
            strict = args[1]
        # allow_deepcopy may exist; accept it and forward unchanged

        try:
            val = getattr(src, name)
        except Exception:
            val = None

        if (not strict) and (val is None):
            try:
                setattr(dst, name, None)
            except Exception:
                pass
            return

        return _orig(src, dst, name, *args, **kwargs)

    _copying.copy_into = _copy_into_patched
    _copying._allow_none_patched = True

    # jtor_update often has its own imported reference
    try:
        _jtor.copy_into = _copy_into_patched
    except Exception:
        pass


# -------------------------
# Coil grouping / currents
# -------------------------

def _try_import_apply_group_currents():
    try:
        from star_machine_cad import apply_group_currents  # type: ignore
        return apply_group_currents
    except Exception:
        return None


def _coil_map(tokamak):
    """
    Build label->coil map from tokamak.coils iterable.
    """
    m = {}
    for label, coil in getattr(tokamak, "coils", []):
        m[str(label).strip().upper()] = coil
    return m


def _print_coil_family_sanity(tokamak, totals: dict[str, float]):
    """
    Print sum of segment currents per family, if grouping metadata exists.
    """
    has_groups = hasattr(tokamak, "coil_groups") and bool(getattr(tokamak, "coil_groups", None))
    print("\n--- Coil currents sanity check ---")
    print(f"Has coil_groups: {bool(has_groups)}")

    cmap = _coil_map(tokamak)

    if not has_groups:
        # legacy: just print direct labels
        for fam, tot in totals.items():
            lab = fam.upper()
            cur = float(getattr(cmap.get(lab, None), "current", np.nan))
            print(f" {lab}: current = {cur/1e6: .6f} MA | target = {tot/1e6: .6f} MA")
        return

    groups = getattr(tokamak, "coil_groups", {}) or {}
    for fam, tot in totals.items():
        fam_u = fam.upper()
        labs = groups.get(fam_u, []) or groups.get(fam_u.capitalize(), []) or groups.get(fam_u.lower(), [])
        s = 0.0
        for lab in labs:
            coil = cmap.get(str(lab).strip().upper())
            if coil is not None:
                s += float(getattr(coil, "current", 0.0))
        print(f" {fam_u:>3s}: sum(segment currents) = {s/1e6: .6f} MA | target total = {tot/1e6: .6f} MA")


def set_star_currents(tokamak, CS=None, PF1=None, PF2=None, PF3=None):
    """
    Assign PF/CS currents to the tokamak Machine.

    If CAD contains segmented coils (CS1M, CS2U, ...), we distribute the
    *family total current* using apply_group_currents().

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

    totals = {"CS": float(CS), "PF1": float(PF1), "PF2": float(PF2), "PF3": float(PF3)}

    apply_group_currents = _try_import_apply_group_currents()
    has_groups = hasattr(tokamak, "coil_groups") and bool(getattr(tokamak, "coil_groups", None))

    if apply_group_currents is not None and has_groups:
        mode = str(getattr(cfg, "coil_group_mode", "area")).lower().strip()
        apply_group_currents(tokamak, totals, mode=mode)

        # zero any other coils not in these families
        fam_set = {"CS", "PF1", "PF2", "PF3"}
        grouped = set()
        for fam, labs in (getattr(tokamak, "coil_groups", {}) or {}).items():
            if str(fam).strip().upper() in fam_set:
                grouped |= set(str(x).strip().upper() for x in (labs or []))

        for label, coil in getattr(tokamak, "coils", []):
            lab = str(label).strip().upper()
            if lab not in grouped:
                try:
                    coil.current = 0.0
                except Exception:
                    pass
        return

    # Fallback: legacy (NOT correct if CS is segmented and labels aren't CS)
    for label, coil in getattr(tokamak, "coils", []):
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


# -------------------------
# Masks / profiles robustness
# -------------------------

def _full_core_mask(eq) -> np.ndarray:
    return np.ones(eq.R.shape, dtype=bool)


def _ensure_masks(profiles, eq):
    """
    Ensure common mask fields exist and have the right shape.
    """
    base = _full_core_mask(eq)

    for name in ("diverted_core_mask", "limiter_core_mask"):
        try:
            v = getattr(profiles, name, None)
        except Exception:
            v = None

        if v is None:
            try:
                setattr(profiles, name, base.copy())
            except Exception:
                pass
        else:
            try:
                arr = np.asarray(v)
                if arr.shape != base.shape:
                    setattr(profiles, name, base.copy())
            except Exception:
                pass

    # Future-proof any *_core_mask attributes
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

    # Patch instance copy defensively
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
    unit_scale: float | None = None,      # None => infer from INSUNITS
    resample_walls: str = "auto",         # "auto" | "always" | "never"
    n_wall: int = 801,
    n_inner: int = 801,
    min_wall_pts: int = 200,
    enforce_ccw: bool = True,
    canonical_start: bool = True,
    redirect_solver_noise: bool = True,
):
    # Safety patches (copy(None) issues)
    _patch_profiles_copy_once()
    _patch_copy_into_allow_none_once()

    # 1) Geometry & machine (CAD)
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

    # Apply full currents once (sanity)
    set_star_currents(tokamak, CS=cfg.CS_current, PF1=cfg.PF1_current, PF2=cfg.PF2_current, PF3=cfg.PF3_current)
    if verbose:
        _print_coil_family_sanity(tokamak, {"CS": cfg.CS_current, "PF1": cfg.PF1_current, "PF2": cfg.PF2_current, "PF3": cfg.PF3_current})

    if verbose:
        print("\n--- CAD machine loaded ---")
        print(f"CAD path         = {geom.get('cad_path', '(unknown)')}")
        print(f"unit_scale       = {geom.get('unit_scale', unit_scale)}")
        print(f"resample_walls   = {resample_walls} | n_wall={n_wall} n_inner={n_inner} min_wall_pts={min_wall_pts}")
        print(f"enforce_ccw      = {enforce_ccw} | canonical_start={canonical_start}")
        print(f"Coils found      = {sorted(list(geom.get('coils', {}).keys()))}")
        if hasattr(tokamak, "coil_groups") and getattr(tokamak, "coil_groups", None):
            fams = {str(k).upper(): len(v) for k, v in (tokamak.coil_groups or {}).items()}
            print(f"Coil families    = {fams}")
            print(f"coil_group_mode  = {getattr(cfg, 'coil_group_mode', 'area')}")

        print("\n--- Target STAR-like settings (cfg) ---")
        print(f"R0_geom    = {cfg.R0_geom:.3f} m")
        print(f"A_geom     = {cfg.A_geom:.3f}")
        print(f"kappa_geom = {cfg.kappa_geom:.3f}")
        print(f"delta_geom = {cfg.delta_geom:.3f}")

        print("\n--- PF/CS currents (cfg totals) ---")
        print(f"CS  = {cfg.CS_current/1e6:.3f} MA")
        print(f"PF1 = {cfg.PF1_current/1e6:.3f} MA")
        print(f"PF2 = {cfg.PF2_current/1e6:.3f} MA")
        print(f"PF3 = {cfg.PF3_current/1e6:.3f} MA")

    # 2) Numerical domain
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

    # 3) Equilibrium object
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=Rmin, Rmax=Rmax,
        Zmin=Zmin, Zmax=Zmax,
        nx=int(cfg.nx_eq),
        ny=int(cfg.ny_eq),
    )

    solver = GSstaticsolver.NKGSsolver(eq)

    # 4) Continuation solve
    f_list = tuple(getattr(cfg, "f_list_equilibrium", (0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.00)))
    tol_ramp  = float(getattr(cfg, "target_rel_tol_ramp", 3e-6))
    tol_final = float(getattr(cfg, "target_rel_tol", 1e-8))

    if verbose:
        print("\n--- Solving equilibrium (Newton–Krylov) ---")
        print(f"[DEBUG] f_list_equilibrium = {f_list} | len = {len(f_list)}")

    noise_path = _solver_noise_log()
    if redirect_solver_noise:
        # fresh separator
        noise_path.parent.mkdir(parents=True, exist_ok=True)
        with open(noise_path, "a", encoding="utf-8") as f:
            f.write("\n" + "="*80 + "\n")
            f.write("New run: star_equilibrium.py\n")
            f.write(f"DXF: {geom.get('cad_path','(unknown)')}\n")
            f.write("="*80 + "\n")

    for j, f in enumerate(f_list, start=1):
        # Apply scaled *family totals*
        set_star_currents(
            tokamak,
            CS=f * cfg.CS_current,
            PF1=f * cfg.PF1_current,
            PF2=f * cfg.PF2_current,
            PF3=f * cfg.PF3_current,
        )

        if verbose:
            _print_coil_family_sanity(tokamak, {
                "CS": f * cfg.CS_current, "PF1": f * cfg.PF1_current,
                "PF2": f * cfg.PF2_current, "PF3": f * cfg.PF3_current
            })

        this_tol = tol_final if (j == len(f_list)) else tol_ramp
        if verbose:
            print(f"[continuation] j={j}/{len(f_list)} f={f:.3f} tol={this_tol:.1e}")

        profiles = _make_profiles(
            eq,
            paxis=f * cfg.paxis,
            Ip=f * cfg.Ip,
            fvac=cfg.fvac,
            alpha_m=cfg.alpha_m,
            alpha_n=cfg.alpha_n,
        )

        with _redirect_solver_noise(enabled=redirect_solver_noise, path=noise_path):
            solver.solve(
                eq=eq,
                profiles=profiles,
                constrain=None,
                target_relative_tolerance=this_tol,
                verbose=False,
            )

        # Minimal post-step info (clean)
        try:
            R_ax, Z_ax = eq.magneticAxis()[:2]
            if verbose:
                print(f"[done] f={f:.3f} axis=({R_ax:.6f},{Z_ax:.6e})")
        except Exception:
            if verbose:
                print(f"[done] f={f:.3f} axis=(unavailable)")

    if verbose:
        print("[OK] Continuation finished.")
        if redirect_solver_noise:
            print(f"[INFO] Solver noise logged to: {noise_path}")

    # 5) Robust shape extraction (LCFS limited by inner wall if present)
    shape = shape_from_lcfs_limiter(eq, geom, prefer_inner=True)

    # Add magnetic axis
    R_ax, Z_ax = eq.magneticAxis()[:2]
    shape["R_ax"] = float(R_ax)
    shape["Z_ax"] = float(Z_ax)

    if verbose:
        print("\n=== Plasma geometry (from LCFS-limiter) ===")
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
        ax.plot(geom["R_plasma"], geom["Z_plasma"], "k--", lw=1.2, label="Plasma target (CAD)")

    if "R_sep" in shape and "Z_sep" in shape:
        ax.plot(shape["R_sep"], shape["Z_sep"], color="tab:red", lw=2.2, label="LCFS (limiter-based)")

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

    out_dir = _repo_results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    print(f"Equilibrium figure saved to: {out_path}")


def main():
    eq, tokamak, geom, shape = build_equilibrium(verbose=True, redirect_solver_noise=True)
    plot_equilibrium(eq, geom, shape, filename=getattr(cfg, "fig_equilibrium", "STAR_equilibrium.png"))


if __name__ == "__main__":
    main()

