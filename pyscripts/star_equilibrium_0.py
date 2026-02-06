# star_equilibrium.py
from __future__ import annotations

import os
import time
import types
import contextlib
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from freegsnke import equilibrium_update, GSstaticsolver
from freegsnke.jtor_update import ConstrainPaxisIp

from star_machine_cad import make_star_machine_from_cad, CADImportOptions, CADLayers
import config_star_bean as cfg


# -------------------------
# Paths / results
# -------------------------
def _default_dxf() -> str:
    here = Path(__file__).resolve().parent
    return str((here / "cad" / "star_baseline.dxf").resolve())

def _results_dir() -> Path:
    d = Path(__file__).resolve().parent.parent / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _solver_noise_log() -> Path:
    return _results_dir() / "solver_noise.log"

@contextlib.contextmanager
def _redirect_solver_noise(enabled: bool, path: Path):
    if not enabled:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f, \
         contextlib.redirect_stdout(f), \
         contextlib.redirect_stderr(f):
        yield


# -------------------------
# FreeGSNKE robustness patches
# -------------------------
def _patch_profiles_copy_once():
    from freegsnke.jtor_update import ConstrainPaxisIp as _C
    if getattr(_C, "_safe_copy_patched", False):
        return
    _orig_copy = _C.copy

    def _safe_copy(self, *args, **kwargs):
        eq = getattr(self, "eq", None) or getattr(self, "_eq", None)
        if eq is not None:
            base = np.ones(eq.R.shape, dtype=bool)
            for name in ("diverted_core_mask", "limiter_core_mask", "core_mask"):
                try:
                    if getattr(self, name, None) is None:
                        setattr(self, name, base)
                except Exception:
                    pass
        return _orig_copy(self, *args, **kwargs)

    _C.copy = _safe_copy
    _C._safe_copy_patched = True

def _patch_copy_into_allow_none_once():
    import freegsnke.copying as _copying
    if getattr(_copying, "_allow_none_patched", False):
        return

    _orig = _copying.copy_into

    def _copy_into_patched(src, dst, name, *args, **kwargs):
        strict = kwargs.get("strict", True)
        if len(args) >= 2:
            strict = args[1]
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

    # Some versions import copy_into into jtor_update
    try:
        import freegsnke.jtor_update as _jtor
        _jtor.copy_into = _copy_into_patched
    except Exception:
        pass

def _ensure_profile_masks(profiles: Any, eq: Any) -> None:
    try:
        base = np.ones(eq.R.shape, dtype=bool)
    except Exception:
        return
    for name in ("diverted_core_mask", "limiter_core_mask", "core_mask"):
        try:
            v = getattr(profiles, name, None)
            if (v is None) or (np.asarray(v).shape != base.shape):
                setattr(profiles, name, base.copy())
        except Exception:
            pass
    for attr in dir(profiles):
        if attr.endswith("_core_mask") or attr.endswith("_mask"):
            try:
                if getattr(profiles, attr, None) is None:
                    setattr(profiles, attr, base.copy())
            except Exception:
                pass

def _make_profiles(eq: Any, *, paxis: float, Ip: float, fvac: float, alpha_m: float, alpha_n: float) -> Any:
    profiles = ConstrainPaxisIp(
        eq=eq,
        paxis=float(paxis),
        Ip=float(Ip),
        fvac=float(fvac),
        alpha_m=float(alpha_m),
        alpha_n=float(alpha_n),
    )
    _ensure_profile_masks(profiles, eq)

    # extra safety: instance-level copy wrapper
    _orig_copy = profiles.copy
    def _safe_copy(self):
        _ensure_profile_masks(self, eq)
        return _orig_copy()
    profiles.copy = types.MethodType(_safe_copy, profiles)
    return profiles


# -------------------------
# Coil currents (family totals -> distribute across segments)
# -------------------------
def _sanitize_mode(x: Any) -> str:
    s = str(x).strip().lower()
    return s if s in ("area", "equal", "same") else "area"

def _coil_items(tokamak: Any):
    coils = getattr(tokamak, "coils", None)
    if coils is None:
        return []
    if isinstance(coils, dict):
        return list(coils.items())
    return list(coils)

def _coil_map(tokamak: Any) -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    for item in _coil_items(tokamak):
        if isinstance(item, (tuple, list)) and len(item) == 2:
            label, coil = item
        else:
            coil = item
            label = getattr(coil, "label", getattr(coil, "name", ""))
        m[str(label).strip().upper()] = coil
    return m

def apply_family_currents(tokamak: Any, totals_A: Dict[str, float], mode: str) -> None:
    """
    Apply TOTAL family currents and distribute to segments if grouping exists.
    """
    mode = _sanitize_mode(mode)

    # Preferred path: tokamak has grouping API
    if hasattr(tokamak, "apply_group_currents") and hasattr(tokamak, "coil_groups"):
        tokamak.apply_group_currents(
            {"CS": totals_A["CS"], "PF1": totals_A["PF1"], "PF2": totals_A["PF2"], "PF3": totals_A["PF3"]},
            mode=mode,
        )
        return

    # Next: star_machine_cad helper (if exported)
    try:
        from star_machine_cad import apply_group_currents  # type: ignore
        if hasattr(tokamak, "coil_groups") and getattr(tokamak, "coil_groups", None):
            apply_group_currents(tokamak, totals_A, mode=mode)
            return
    except Exception:
        pass

    # Legacy fallback (only correct if your CAD labels match these)
    cmap = _coil_map(tokamak)
    for lab, coil in cmap.items():
        if lab == "CS":
            coil.current = float(totals_A["CS"])
        elif lab in ("PF1U", "PF1L"):
            coil.current = float(totals_A["PF1"])
        elif lab in ("PF2U", "PF2L"):
            coil.current = float(totals_A["PF2"])
        elif lab in ("PF3U", "PF3L"):
            coil.current = float(totals_A["PF3"])
        else:
            # If you don't want to zero-out "other" coils, delete this line.
            coil.current = 0.0

def _print_coil_family_sanity(tokamak: Any, totals_A: Dict[str, float]) -> None:
    print("\n--- Coil currents sanity check ---")
    has_groups = hasattr(tokamak, "coil_groups") and bool(getattr(tokamak, "coil_groups", None))
    print(f"Has coil_groups: {bool(has_groups)}")

    cmap = _coil_map(tokamak)

    if not has_groups:
        for fam, tot in totals_A.items():
            coil = cmap.get(fam.upper())
            cur = float(getattr(coil, "current", np.nan)) if coil is not None else np.nan
            print(f" {fam.upper():>3s}: current={cur/1e6: .6f} MA | target={tot/1e6: .6f} MA")
        return

    groups = getattr(tokamak, "coil_groups", {}) or {}
    for fam, tot in totals_A.items():
        labs = groups.get(fam, []) or groups.get(fam.upper(), []) or groups.get(fam.lower(), [])
        s = 0.0
        for lab in (labs or []):
            coil = cmap.get(str(lab).strip().upper())
            if coil is not None:
                s += float(getattr(coil, "current", 0.0))
        print(f" {fam.upper():>3s}: sum(segments)={s/1e6: .6f} MA | target={tot/1e6: .6f} MA")


# -------------------------
# Shape extraction (prefer unified analyze_star; fallback to analyze_star_lcfs)
# -------------------------
def compute_shape(eq: Any, geom: Dict[str, Any]) -> Dict[str, Any]:
    # Try unified analyze_star first
    try:
        from analyze_star import analyze_star  # type: ignore
        shp = analyze_star(
            eq, geom,
            require_two_x=False,
            null_prefer=str(getattr(cfg, "null_prefer", "lower")),
            prefer_inner_lcfs=True,
            psi_percentile_lcfs=float(getattr(cfg, "psi_percentile_lcfs", 0.5)),
            edge_pad_cells=2,
        )
        return shp
    except Exception:
        # Fallback to LCFS-limiter helper
        from analyze_star_lcfs import shape_from_lcfs_limiter  # type: ignore
        return shape_from_lcfs_limiter(eq, geom, prefer_inner=True)


# -------------------------
# Textbox: plasma summary
# -------------------------
def _try_get_Ip_from_eq(eq: Any) -> Optional[float]:
    """
    Try to read plasma current from the equilibrium object if available.
    Falls back to None if not found.
    """
    for name in ("plasmaCurrent", "plasma_current", "Ip", "ip"):
        if hasattr(eq, name):
            try:
                v = getattr(eq, name)
                v = v() if callable(v) else v
                v = float(v)
                if np.isfinite(v):
                    return v
            except Exception:
                pass
    return None

def _format_plasma_box(eq: Any, shape: Dict[str, Any]) -> str:
    def fnum(x, fmt=".3f"):
        try:
            v = float(x)
            return ("nan" if (not np.isfinite(v)) else format(v, fmt))
        except Exception:
            return "nan"

    # Geometry (from analyze_star)
    R0 = shape.get("R0_plasma", np.nan)
    a  = shape.get("a_plasma", np.nan)
    A  = shape.get("A_plasma", np.nan)
    k  = shape.get("kappa_plasma", np.nan)
    du = shape.get("delta_u", np.nan)
    dl = shape.get("delta_l", np.nan)

    nxp = len(shape.get("xpoints", []) or [])

    # Ip and paxis:
    # - Ip: try read from eq; else cfg.Ip (setpoint)
    # - paxis: use cfg.paxis (setpoint)
    Ip_eq = _try_get_Ip_from_eq(eq)
    Ip_use = Ip_eq if (Ip_eq is not None) else float(getattr(cfg, "Ip", np.nan))
    paxis_use = float(getattr(cfg, "paxis", np.nan))

    # Axis (if available)
    Rax = shape.get("R_ax", np.nan)
    Zax = shape.get("Z_ax", np.nan)

    lines = [
        "Plasma summary",
        f"R0 = {fnum(R0)} m",
        f"a  = {fnum(a)} m",
        f"A  = {fnum(A)}",
        f"κ  = {fnum(k)}",
        f"δu/δl = {fnum(du)}/{fnum(dl)}",
        f"X-points = {nxp}",
        f"Axis (R,Z) = ({fnum(Rax)},{fnum(Zax)})",
        "",
        f"Ip = {('nan' if not np.isfinite(Ip_use) else f'{Ip_use/1e6:.3f}')} MA",
        f"p_axis(set) = {('nan' if not np.isfinite(paxis_use) else f'{paxis_use/1e3:.2f}')} kPa",
    ]
    return "\n".join(lines)


# -------------------------
# Main equilibrium build
# -------------------------
def build_equilibrium(
    *,
    verbose: bool = True,
    redirect_solver_noise: bool = True,
    dxf_path: Optional[str] = None,
) -> Tuple[Any, Any, Dict[str, Any], Dict[str, Any]]:

    _patch_profiles_copy_once()
    _patch_copy_into_allow_none_once()

    # Resolve DXF path
    if dxf_path is None:
        dxf_path = str(getattr(cfg, "dxf_path", _default_dxf()))

    # CAD options (read from cfg with fallbacks)
    opts = CADImportOptions(
        unit_scale=float(getattr(cfg, "unit_scale", 1.0)) if getattr(cfg, "unit_scale", None) is not None else None,
        resample_walls=str(getattr(cfg, "resample_walls", "auto")),
        n_wall=int(getattr(cfg, "n_wall", 801)),
        n_inner=int(getattr(cfg, "n_inner", 801)),
        n_plasma=int(getattr(cfg, "n_plasma", 801)),
        min_wall_pts=int(getattr(cfg, "min_wall_pts", 200)),
        enforce_ccw=bool(getattr(cfg, "enforce_ccw", True)),
        canonical_start=bool(getattr(cfg, "canonical_start", True)),
        flatten_distance=float(getattr(cfg, "flatten_distance", 0.0)),
        label_match_factor=float(getattr(cfg, "label_match_factor", 0.92)),
    )

    tokamak, geom = make_star_machine_from_cad(
        dxf_path=str(dxf_path),
        layers=CADLayers(),
        opts=opts,
        strict_expected=True,
    )

    # Domain
    R_outer = np.asarray(geom["R_outer"], dtype=float)
    Z_outer = np.asarray(geom["Z_outer"], dtype=float)
    margin = float(getattr(cfg, "margin_RZ", 0.5))
    Rmin, Rmax = float(R_outer.min() - margin), float(R_outer.max() + margin)
    Zmin, Zmax = float(Z_outer.min() - margin), float(Z_outer.max() + margin)

    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=Rmin, Rmax=Rmax,
        Zmin=Zmin, Zmax=Zmax,
        nx=int(getattr(cfg, "nx_eq", 65)),
        ny=int(getattr(cfg, "ny_eq", 129)),
    )
    solver = GSstaticsolver.NKGSsolver(eq)

    # Continuation
    f_list = tuple(getattr(cfg, "f_list_equilibrium", (0.08, 0.15, 0.25, 0.40, 0.60, 0.78, 0.90, 1.00)))
    tol_ramp  = float(getattr(cfg, "target_rel_tol_ramp", 3e-5))
    tol_final = float(getattr(cfg, "target_rel_tol", 1e-5))

    mode = _sanitize_mode(getattr(cfg, "coil_group_mode", "area"))

    noise_path = _solver_noise_log()
    if redirect_solver_noise:
        with open(noise_path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 90 + "\n")
            f.write(f"New run: {time.ctime()}\n")
            f.write(f"DXF: {geom.get('cad_path', str(dxf_path))}\n")
            f.write("=" * 90 + "\n")

    if verbose:
        print("\n--- CAD machine loaded ---")
        print(f"CAD path   = {geom.get('cad_path', str(dxf_path))}")
        print(f"unit_scale = {geom.get('unit_scale', getattr(cfg, 'unit_scale', None))}")
        print(f"Domain R=[{Rmin:.2f},{Rmax:.2f}] Z=[{Zmin:.2f},{Zmax:.2f}]")
        print(f"Grid nx={getattr(cfg, 'nx_eq', 65)} ny={getattr(cfg, 'ny_eq', 129)}")
        print(f"coil_group_mode = {mode}")
        print("\n--- Currents (TOTAL family) ---")
        print(f"CS  = {cfg.CS_current/1e6:.6f} MA")
        print(f"PF1 = {cfg.PF1_current/1e6:.6f} MA")
        print(f"PF2 = {cfg.PF2_current/1e6:.6f} MA")
        print(f"PF3 = {cfg.PF3_current/1e6:.6f} MA")
        print("\n--- Profiles (setpoints) ---")
        print(f"Ip={cfg.Ip:.3e} A | paxis={cfg.paxis:.3e} Pa | fvac={cfg.fvac} | alpha_m={cfg.alpha_m} alpha_n={cfg.alpha_n}")
        print("\n--- Solve ---")
        print(f"f_list={f_list} | tol_ramp={tol_ramp:.1e} tol_final={tol_final:.1e}")

    for j, f in enumerate(f_list, start=1):
        this_tol = tol_final if (j == len(f_list)) else tol_ramp

        totals_A = {
            "CS":  float(f) * float(cfg.CS_current),
            "PF1": float(f) * float(cfg.PF1_current),
            "PF2": float(f) * float(cfg.PF2_current),
            "PF3": float(f) * float(cfg.PF3_current),
        }
        apply_family_currents(tokamak, totals_A, mode=mode)

        if verbose:
            print(f"\n[continuation] step {j}/{len(f_list)}  f={f:.3f}  tol={this_tol:.1e}")
            _print_coil_family_sanity(tokamak, totals_A)

        profiles = _make_profiles(
            eq,
            paxis=float(f) * float(cfg.paxis),
            Ip=float(f) * float(cfg.Ip),
            fvac=float(cfg.fvac),
            alpha_m=float(cfg.alpha_m),
            alpha_n=float(cfg.alpha_n),
        )

        with _redirect_solver_noise(redirect_solver_noise, noise_path):
            solver.solve(
                eq=eq,
                profiles=profiles,
                constrain=None,
                target_relative_tolerance=float(this_tol),
                verbose=False,
            )

        if verbose:
            try:
                R_ax, Z_ax = eq.magneticAxis()[:2]
                print(f"[done] axis=({R_ax:.6f},{Z_ax:.6e})")
            except Exception:
                print("[done] axis=(unavailable)")

    shape = compute_shape(eq, geom)
    try:
        R_ax, Z_ax = eq.magneticAxis()[:2]
        shape["R_ax"] = float(R_ax)
        shape["Z_ax"] = float(Z_ax)
    except Exception:
        pass

    if verbose:
        print("\n=== Plasma geometry ===")
        for k in ("R0_plasma", "a_plasma", "A_plasma", "kappa_plasma", "delta_u", "delta_l"):
            if k in shape:
                try:
                    print(f" {k:>12s} = {float(shape[k]):.6f}")
                except Exception:
                    pass
        if "xpoints" in shape:
            print(f" xpoints = {len(shape.get('xpoints', []) or [])}")
        if redirect_solver_noise:
            print(f"\n[INFO] solver noise log: {noise_path}")

    return eq, tokamak, geom, shape


# -------------------------
# Plot
# -------------------------
def plot_equilibrium(eq: Any, geom: Dict[str, Any], shape: Dict[str, Any], filename: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 10))
    eq.plot(axis=ax, show=False)

    ax.plot(geom["R_outer"], geom["Z_outer"], "k", lw=2, label="Outer wall")
    if "R_inner" in geom and "Z_inner" in geom:
        ax.plot(geom["R_inner"], geom["Z_inner"], "k--", lw=1.5, label="Inner wall")

    # LCFS/separatrix curve if available
    R_sep = shape.get("R_sep", None)
    Z_sep = shape.get("Z_sep", None)
    if R_sep is not None and Z_sep is not None and len(R_sep) > 10:
        ax.plot(R_sep, Z_sep, lw=2.2, label="LCFS/Separatrix")

    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("STAR-like equilibrium")

    ax.legend(loc="upper right")
    fig.tight_layout()

    # --- Textbox (legend-like) with plasma parameters ---
    box = _format_plasma_box(eq, shape)
    ax.text(
        0.02, 0.98, box,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.80, edgecolor="0.3"),
    )

    if filename is None:
        filename = str(getattr(cfg, "fig_equilibrium", "STAR_bean_equilibrium.png"))
    out = _results_dir() / filename
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[SAVED] {out}")


def main():
    eq, tokamak, geom, shape = build_equilibrium(verbose=True, redirect_solver_noise=True)
    plot_equilibrium(eq, geom, shape, filename=str(getattr(cfg, "fig_equilibrium", "STAR_bean_equilibrium.png")))

if __name__ == "__main__":
    main()

