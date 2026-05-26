# star_equilibrium.py
from __future__ import annotations

import time
import types
import contextlib
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Iterator, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

from freegsnke import equilibrium_update, GSstaticsolver
from freegsnke.jtor_update import ConstrainPaxisIp

from star_machine_cad import make_star_machine_from_cad, CADImportOptions, CADLayers
import config_star_bean as cfg


def _default_dxf() -> str:
    here = Path(__file__).resolve().parent
    return str((here  / "cad" / "star_baseline.dxf").resolve())


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
            # eq.R might be attribute (array) in many FreeGSNKE versions
            try:
                base = np.ones(eq.R.shape, dtype=bool)  # type: ignore[attr-defined]
            except Exception:
                # fallback if eq.R is callable
                try:
                    R = eq.R()  # type: ignore[operator]
                    base = np.ones(np.asarray(R).shape, dtype=bool)
                except Exception:
                    base = None

            if base is not None:
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

    try:
        import freegsnke.jtor_update as _jtor
        _jtor.copy_into = _copy_into_patched
    except Exception:
        pass


def _ensure_profile_masks(profiles: Any, eq: Any) -> None:
    try:
        base = np.ones(eq.R.shape, dtype=bool)  # type: ignore[attr-defined]
    except Exception:
        try:
            R = eq.R()  # type: ignore[operator]
            base = np.ones(np.asarray(R).shape, dtype=bool)
        except Exception:
            return

    for name in ("diverted_core_mask", "limiter_core_mask", "core_mask"):
        try:
            v = getattr(profiles, name, None)
            if (v is None) or (np.asarray(v).shape != base.shape):
                setattr(profiles, name, base.copy())
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

    _orig_copy = profiles.copy

    def _safe_copy(self):
        _ensure_profile_masks(self, eq)
        return _orig_copy()

    profiles.copy = types.MethodType(_safe_copy, profiles)
    return profiles


# -------------------------
# Coil currents helper
# -------------------------
def _sanitize_mode(x: Any) -> str:
    s = str(x).strip().lower()
    return s if s in ("area", "equal", "same") else "area"


def _zero_passive_currents(tokamak: Any) -> None:
    """
    Enforce I=0 on passive coils/filaments every time, no matter what.
    """
    passive = getattr(tokamak, "passive_coils", None) or []
    coils_dict = getattr(tokamak, "coils_dict", None) or {}

    # Normalize mapping keys
    norm_map = {str(k).upper(): v for k, v in coils_dict.items()}

    for lab in passive:
        UL = str(lab).upper()
        c = norm_map.get(UL, None)
        if c is None:
            continue
        try:
            c.current = 0.0
        except Exception:
            pass


def _family_from_label_for_current(label: str) -> str:
    UL = str(label).strip().upper()

    if UL.startswith("CS"):
        return "CS"

    for k in range(1, 10):
        fam = f"PF{k}"
        if UL.startswith(fam):
            return fam

    return UL


def apply_family_currents(tokamak: Any, totals_A: Dict[str, float], mode: str) -> None:
    mode = _sanitize_mode(mode)

    # Preferred path: use star_machine_cad group logic.
    try:
        from star_machine_cad import apply_group_currents
        if hasattr(tokamak, "coil_groups") and getattr(tokamak, "coil_groups", None):
            apply_group_currents(tokamak, totals_A, mode=mode)
            _zero_passive_currents(tokamak)
            return
    except Exception as e:
        print(f"[WARN] apply_group_currents failed; using fallback current assignment: {repr(e)}")

    # Robust fallback for discretized labels like CS1M_F001, PF4U_F012, etc.
    for lab, coil in getattr(tokamak, "coils_dict", {}).items():
        UL = str(lab).strip().upper()
        fam = _family_from_label_for_current(UL)

        try:
            coil.current = float(totals_A.get(fam, 0.0))
        except Exception:
            pass

    _zero_passive_currents(tokamak)

# -------------------------
# Coil grouping sanity check
# -------------------------
def _iter_coils_dict(tokamak: Any):
    # Prefer coils_dict if available
    d = getattr(tokamak, "coils_dict", None)
    if isinstance(d, dict) and d:
        for k, v in d.items():
            yield str(k).strip().upper(), v
        return
    # Fallback to tokamak.coils (tuple or object)
    for item in getattr(tokamak, "coils", []) or []:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            lab, coil = item
        else:
            coil = item
            lab = getattr(coil, "label", getattr(coil, "name", ""))
        yield str(lab).strip().upper(), coil


def _get_group_labels(tokamak: Any, fam: str) -> List[str]:
    fam = str(fam).strip().upper()
    groups = getattr(tokamak, "coil_groups", None) or {}
    labs = (
        groups.get(fam, None)
        or groups.get(fam.lower(), None)
        or groups.get(fam.upper(), None)
        or []
    )
    return [str(x).strip().upper() for x in (labs or []) if str(x).strip()]


def print_group_currents_sanity(tokamak: Any, totals_A: Dict[str, float], mode: str, header: str = "") -> None:
    """
    Prints:
      - family labels count
      - sum(segment currents) vs target total
      - warns if mismatch is large (suggesting wrong mode or labels mismatch)
    """
    mode = _sanitize_mode(mode)
    coil_map = {lab: coil for lab, coil in _iter_coils_dict(tokamak)}

    if header:
        print(header)

    if not getattr(tokamak, "coil_groups", None):
        print("[SANITY] tokamak.coil_groups not present; cannot sum per-family segments.")
        return

    # NUEVO: incluye PF4–PF6
    for fam in ("CS", "CS_MID", "CS_END", "PF1", "PF2", "PF3", "PF4", "PF5", "PF6"):
        target = float(totals_A.get(fam, 0.0))
        labs = _get_group_labels(tokamak, fam)

        if not labs:
            print(f"[SANITY] {fam}: no labels in coil_groups")
            continue

        s = 0.0
        n_ok = 0
        for lab in labs:
            c = coil_map.get(lab)
            if c is None:
                continue
            try:
                s += float(getattr(c, "current"))
                n_ok += 1
            except Exception:
                pass

        msg = (
            f"[SANITY] {fam}: n={len(labs)} (found {n_ok}) | "
            f"sum(segment currents)={s/1e6:+.6f} MA | "
            f"target direct={target/1e6:+.6f} MA"
        )

        # CS_MID/CS_END overlap with parent CS, so their sum can be nonzero
        # even if their direct target is zero.
        if fam in ("CS_MID", "CS_END") and abs(float(totals_A.get("CS", 0.0))) > 0:
            msg += "  [overlap subset; parent CS contribution included]"
            print(msg)
            continue

        denom = max(1.0, abs(target))
        rel = abs(s - target) / denom
        if rel > 0.02:
            msg += f"  <-- WARNING rel_err={rel:.3%} (mode={mode})"

        print(msg)

# -------------------------
# Shape extraction
# -------------------------
def compute_shape(eq: Any, geom: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from analyze_star import analyze_star
        shp = analyze_star(
            eq, geom,
            require_two_x=False,
            null_prefer=str(getattr(cfg, "null_prefer", "lower")),
            prefer_inner_lcfs=True,
            psi_percentile_lcfs=float(getattr(cfg, "psi_percentile_lcfs", 0.5)),
            edge_pad_cells=2,
        )
        # keep a slot for diagnostics in case caller adds it later
        if "plasma_diag" not in shp:
            shp["plasma_diag"] = None
        return shp
    except Exception as e:
        return {"ok_sep": False, "reason": f"analyze_star_failed:{repr(e)}", "plasma_diag": None}


# -------------------------
# LCFS fallback diagnostics (works even without separatrix)
# -------------------------
def _poly_area_xy(xy: np.ndarray) -> float:
    xy = np.asarray(xy, float)
    if xy.shape[0] < 3:
        return 0.0
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ensure_closed_xy(xy: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    xy = np.asarray(xy, float)
    if xy.shape[0] < 2:
        return xy
    if np.linalg.norm(xy[0] - xy[-1]) > tol:
        xy = np.vstack([xy, xy[0]])
    return xy


def _axis_candidates(eq: Any) -> List[Tuple[float, float, float, str]]:
    """
    Return two candidates: psi-min and psi-max extrema (axis can be either depending on sign convention).
    Each element is (R_ax, Z_ax, psi_ax, tag).
    """
    psi = eq.psi() if callable(getattr(eq, "psi", None)) else getattr(eq, "psi", None)
    R = eq.R() if callable(getattr(eq, "R", None)) else getattr(eq, "R", None)
    Z = eq.Z() if callable(getattr(eq, "Z", None)) else getattr(eq, "Z", None)

    psi = np.asarray(psi, float)
    R = np.asarray(R, float)
    Z = np.asarray(Z, float)

    if R.ndim == 2 and Z.ndim == 2:
        RR, ZZ = R, Z
    else:
        RR, ZZ = np.meshgrid(R, Z, indexing="xy")

    kmin = int(np.nanargmin(psi))
    imin, jmin = np.unravel_index(kmin, psi.shape)
    kmax = int(np.nanargmax(psi))
    imax, jmax = np.unravel_index(kmax, psi.shape)

    return [
        (float(RR[imin, jmin]), float(ZZ[imin, jmin]), float(psi[imin, jmin]), "min"),
        (float(RR[imax, jmax]), float(ZZ[imax, jmax]), float(psi[imax, jmax]), "max"),
    ]


def _closed_contours_containing_point(eq: Any, level: float, point: Tuple[float, float]) -> List[np.ndarray]:
    """
    Return list of closed contour polylines (Nx2) at psi=level that contain `point`.
    """
    psi = eq.psi() if callable(getattr(eq, "psi", None)) else getattr(eq, "psi", None)
    R = eq.R() if callable(getattr(eq, "R", None)) else getattr(eq, "R", None)
    Z = eq.Z() if callable(getattr(eq, "Z", None)) else getattr(eq, "Z", None)

    psi = np.asarray(psi, float)
    R = np.asarray(R, float)
    Z = np.asarray(Z, float)

    if R.ndim == 2 and Z.ndim == 2:
        RR, ZZ = R, Z
    else:
        RR, ZZ = np.meshgrid(R, Z, indexing="xy")

    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        cs = ax.contour(RR, ZZ, psi, levels=[float(level)])
        curves: List[np.ndarray] = []
        for col in cs.collections:
            for path in col.get_paths():
                v = np.asarray(path.vertices, float)
                if v.shape[0] < 20:
                    continue
                v = _ensure_closed_xy(v)
                try:
                    if MplPath(v).contains_point(point):
                        curves.append(v)
                except Exception:
                    continue
        return curves
    except Exception:
        return []
    finally:
        plt.close(fig)


def _infer_lcfs_closed(eq: Any, *, n_levels: int = 28) -> Optional[Dict[str, Any]]:
    """
    Fallback LCFS: search across psi levels and pick the LARGEST-area closed contour
    that contains the magnetic axis (trying both psi-min and psi-max extrema).
    """
    psi = eq.psi() if callable(getattr(eq, "psi", None)) else getattr(eq, "psi", None)
    psi = np.asarray(psi, float)
    if psi.size < 10:
        return None

    psi_lo = float(np.nanmin(psi))
    psi_hi = float(np.nanmax(psi))
    if not np.isfinite(psi_lo + psi_hi) or abs(psi_hi - psi_lo) < 1e-14:
        return None

    best: Optional[Dict[str, Any]] = None

    for (Rax, Zax, psi_ax, tag) in _axis_candidates(eq):
        # Move away from axis toward edge; direction depends on whether axis is min or max
        span = (psi_hi - psi_lo)
        if tag == "min":
            levels = np.linspace(psi_ax + 0.03 * span, psi_hi - 0.03 * span, int(n_levels))
        else:
            levels = np.linspace(psi_ax - 0.03 * span, psi_lo + 0.03 * span, int(n_levels))

        point = (Rax, Zax)

        for lv in levels:
            curves = _closed_contours_containing_point(eq, lv, point)
            if not curves:
                continue

            for v in curves:
                area = _poly_area_xy(v)
                if area <= 1e-6:
                    continue

                cand = {
                    "ok": True,
                    "method": "psi_closed_contour",
                    "R_ax": float(Rax),
                    "Z_ax": float(Zax),
                    "psi_ax": float(psi_ax),
                    "psi_level": float(lv),
                    "axis_tag": str(tag),
                    "curve_xy": v,
                    "area": float(area),
                }
                if (best is None) or (float(cand["area"]) > float(best["area"])):
                    best = cand

    if best is None:
        return None

    xy = np.asarray(best["curve_xy"], float)
    return {
        "ok": True,
        "method": "psi_closed_contour",
        "R_ax": float(best["R_ax"]),
        "Z_ax": float(best["Z_ax"]),
        "psi_ax": float(best["psi_ax"]),
        "psi_level": float(best["psi_level"]),
        "axis_tag": str(best["axis_tag"]),
        "R_lcfs": xy[:, 0].tolist(),
        "Z_lcfs": xy[:, 1].tolist(),
        "area_m2": float(best["area"]),
    }


def _plasma_params_from_curve(R: np.ndarray, Z: np.ndarray) -> Dict[str, Any]:
    R = np.asarray(R, float)
    Z = np.asarray(Z, float)
    if R.size < 20 or Z.size != R.size:
        return {"ok": False}

    iRmax = int(np.nanargmax(R))
    iRmin = int(np.nanargmin(R))
    iZmax = int(np.nanargmax(Z))
    iZmin = int(np.nanargmin(Z))

    Rmax = float(R[iRmax]); Rmin = float(R[iRmin])
    Zmax = float(Z[iZmax]); Zmin = float(Z[iZmin])

    a = 0.5 * (Rmax - Rmin)
    if not np.isfinite(a) or a <= 1e-6:
        return {"ok": False}

    R0 = 0.5 * (Rmax + Rmin)
    Z0 = 0.5 * (Zmax + Zmin)

    kappa = (Zmax - Zmin) / (2.0 * a)

    # triangularity (top/bottom)
    R_top = float(R[iZmax])
    R_bot = float(R[iZmin])
    delta_u = (R0 - R_top) / a
    delta_l = (R0 - R_bot) / a

    A = R0 / a
    area = float(_poly_area_xy(np.column_stack([R, Z])))

    return {
        "ok": True,
        "R0": float(R0),
        "Z0": float(Z0),
        "a": float(a),
        "A": float(A),
        "kappa": float(kappa),
        "delta_u": float(delta_u),
        "delta_l": float(delta_l),
        "area_m2": float(area),
        "Rmin": float(Rmin),
        "Rmax": float(Rmax),
        "Zmin": float(Zmin),
        "Zmax": float(Zmax),
    }


def plasma_diagnostics(eq: Any, geom: Dict[str, Any], shape: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefer:
      1) analyze_star separatrix (R_sep/Z_sep)
      2) analyze_star fallback_lcfs (if present)
      3) psi-closed-contour fallback containing the axis

    Returns dict with ok flag + method + geometric params whenever possible.
    """
    # 1) separatrix if present
    R_sep = shape.get("R_sep", None)
    Z_sep = shape.get("Z_sep", None)
    if R_sep is not None and Z_sep is not None:
        try:
            if len(R_sep) > 20 and len(Z_sep) == len(R_sep):
                params = _plasma_params_from_curve(np.asarray(R_sep, float), np.asarray(Z_sep, float))
                params["method"] = "analyze_star_separatrix"
                # carry axis if analyze_star provided
                if "R_ax" in shape and "Z_ax" in shape:
                    try:
                        params["R_ax"] = float(shape.get("R_ax", np.nan))
                        params["Z_ax"] = float(shape.get("Z_ax", np.nan))
                    except Exception:
                        pass
                return params
        except Exception:
            pass

    # 2) analyze_star fallback_lcfs (if analyze_star stored it)
    fb = shape.get("fallback_lcfs", None)
    if isinstance(fb, dict) and ("R" in fb) and ("Z" in fb):
        try:
            R = np.asarray(fb["R"], float)
            Z = np.asarray(fb["Z"], float)
            if R.size > 20 and Z.size == R.size:
                params = _plasma_params_from_curve(R, Z)
                params["method"] = "analyze_star_fallback_lcfs"
                if "R_ax" in shape and "Z_ax" in shape:
                    try:
                        params["R_ax"] = float(shape.get("R_ax", np.nan))
                        params["Z_ax"] = float(shape.get("Z_ax", np.nan))
                    except Exception:
                        pass
                params["R_lcfs"] = R.tolist()
                params["Z_lcfs"] = Z.tolist()
                return params
        except Exception:
            pass

    # 3) psi closed contour fallback
    nlv = int(getattr(cfg, "lcfs_fallback_nlevels", 28))
    fb2 = _infer_lcfs_closed(eq, n_levels=nlv)
    if fb2 is None or not fb2.get("ok", False):
        return {"ok": False, "method": "none", "reason": "no_closed_lcfs_found"}

    R = np.asarray(fb2["R_lcfs"], float)
    Z = np.asarray(fb2["Z_lcfs"], float)
    params = _plasma_params_from_curve(R, Z)
    params.update({
        "method": str(fb2.get("method", "psi_closed_contour")),
        "R_ax": float(fb2.get("R_ax", np.nan)),
        "Z_ax": float(fb2.get("Z_ax", np.nan)),
        "psi_ax": float(fb2.get("psi_ax", np.nan)),
        "psi_level": float(fb2.get("psi_level", np.nan)),
        "axis_tag": str(fb2.get("axis_tag", "")),
        "R_lcfs": fb2.get("R_lcfs", None),
        "Z_lcfs": fb2.get("Z_lcfs", None),
    })
    return params


def print_plasma_diagnostics(diag: Dict[str, Any]) -> None:
    if not diag.get("ok", False):
        print("\n[PLASMA] No LCFS diagnostic curve found.")
        if "reason" in diag:
            print("reason =", diag.get("reason"))
        return

    print("\n--- Plasma diagnostics (LCFS-based) ---")
    print(f"method   = {diag.get('method')}")
    if "R_ax" in diag and "Z_ax" in diag and np.isfinite(float(diag.get("R_ax", np.nan))):
        print(f"axis     = (R_ax={float(diag['R_ax']):.4f}, Z_ax={float(diag['Z_ax']):.4f})")
    if "psi_level" in diag and np.isfinite(float(diag.get("psi_level", np.nan))):
        print(f"psi_level= {float(diag['psi_level']):.6g}   (axis_tag={diag.get('axis_tag','')})")

    print(f"R0       = {float(diag['R0']):.4f} m")
    print(f"a        = {float(diag['a']):.4f} m")
    print(f"A=R0/a   = {float(diag['A']):.4f}")
    print(f"kappa    = {float(diag['kappa']):.4f}")
    print(f"delta_u  = {float(diag['delta_u']):.4f}")
    print(f"delta_l  = {float(diag['delta_l']):.4f}")
    print(f"area     = {float(diag['area_m2']):.4f} m^2")
    print(f"bounds   = R[{float(diag['Rmin']):.4f},{float(diag['Rmax']):.4f}]  Z[{float(diag['Zmin']):.4f},{float(diag['Zmax']):.4f}]")


# -------------------------
# Geometry/domain and plotting helpers for CAD/passive setup
# -------------------------
def _append_curve_points(points: List[np.ndarray], R: Any, Z: Any) -> None:
    try:
        r = np.asarray(R, dtype=float).ravel()
        z = np.asarray(Z, dtype=float).ravel()
        if r.size and z.size and r.size == z.size:
            points.append(np.column_stack([r, z]))
    except Exception:
        pass


def _machine_geometry_bbox(geom: Dict[str, Any], *, include_coils: bool = True, include_passives: bool = True) -> Tuple[float, float, float, float]:
    """
    Return a robust bbox for the whole CAD machine, not only the equilibrium domain.
    Includes:
      - WALL_OUTER / WALL_INNER
      - active coil rectangles/filaments
      - passive STAR_VESSEL structures/filaments
      - optional plasma target
    """
    pts: List[np.ndarray] = []

    _append_curve_points(pts, geom.get("R_outer"), geom.get("Z_outer"))
    _append_curve_points(pts, geom.get("R_inner"), geom.get("Z_inner"))
    _append_curve_points(pts, geom.get("R_limiter"), geom.get("Z_limiter"))
    _append_curve_points(pts, geom.get("R_blanket_outer"), geom.get("Z_blanket_outer"))
    _append_curve_points(pts, geom.get("R_plasma"), geom.get("Z_plasma"))

    if include_coils:
        for _lab, val in (geom.get("coils", {}) or {}).items():
            try:
                Rc, Zc, dR, dZ = map(float, val[:4])
                pts.append(np.array([
                    [Rc - dR, Zc - dZ],
                    [Rc + dR, Zc + dZ],
                ], dtype=float))
            except Exception:
                continue

    if include_passives:
        for ps in geom.get("passive_structures", []) or []:
            try:
                xy = np.asarray(ps.get("xy"), dtype=float)
                if xy.ndim == 2 and xy.shape[1] >= 2 and xy.shape[0] >= 3:
                    pts.append(xy[:, :2])
            except Exception:
                pass

        for item in geom.get("passive_filaments", []) or []:
            try:
                # (lab, Rc, Zc, dR, dZ, material, resistivity)
                _lab, Rc, Zc, dR, dZ = item[:5]
                Rc = float(Rc); Zc = float(Zc); dR = float(dR); dZ = float(dZ)
                pts.append(np.array([
                    [Rc - dR, Zc - dZ],
                    [Rc + dR, Zc + dZ],
                ], dtype=float))
            except Exception:
                continue

    if not pts:
        return (0.6, 8.0, -7.0, 7.0)

    P = np.vstack(pts)
    P = P[np.all(np.isfinite(P), axis=1)]
    if P.size == 0:
        return (0.6, 8.0, -7.0, 7.0)

    return float(np.min(P[:, 0])), float(np.max(P[:, 0])), float(np.min(P[:, 1])), float(np.max(P[:, 1]))


def _domain_from_geom(geom: Dict[str, Any], *, margin: float, source: str = "machine") -> Tuple[float, float, float, float]:
    """
    source:
      - "blanket" / "outer": old behavior, based on WALL_OUTER only.
      - "limiter" / "inner": based on WALL_INNER only.
      - "machine" / "all": based on walls + active coils + passives.
    """
    src = str(source).strip().lower()

    if src in ("blanket", "outer", "wall_outer"):
        R = np.asarray(geom["R_outer"], dtype=float)
        Z = np.asarray(geom["Z_outer"], dtype=float)
        Rmin_raw = float(np.nanmin(R) - margin)
        Rmax = float(np.nanmax(R) + margin)
        Zmin = float(np.nanmin(Z) - margin)
        Zmax = float(np.nanmax(Z) + margin)

    elif src in ("limiter", "inner", "wall_inner") and ("R_inner" in geom and "Z_inner" in geom):
        R = np.asarray(geom["R_inner"], dtype=float)
        Z = np.asarray(geom["Z_inner"], dtype=float)
        Rmin_raw = float(np.nanmin(R) - margin)
        Rmax = float(np.nanmax(R) + margin)
        Zmin = float(np.nanmin(Z) - margin)
        Zmax = float(np.nanmax(Z) + margin)

    else:
        Rmin0, Rmax0, Zmin0, Zmax0 = _machine_geometry_bbox(geom, include_coils=True, include_passives=True)
        Rmin_raw = Rmin0 - margin
        Rmax = Rmax0 + margin
        Zmin = Zmin0 - margin
        Zmax = Zmax0 + margin

    Rmin = max(0.05, float(Rmin_raw))
    return float(Rmin), float(Rmax), float(Zmin), float(Zmax)


def _plot_rect_outline(ax, Rc: float, Zc: float, dR: float, dZ: float, **kwargs) -> None:
    x0, x1 = Rc - dR, Rc + dR
    y0, y1 = Zc - dZ, Zc + dZ
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], **kwargs)


def _family_from_label(label: str) -> str:
    s = str(label).upper()
    if s.startswith("CS"):
        return "CS"
    for fam in ("PF1", "PF2", "PF3", "PF4", "PF5", "PF6"):
        if s.startswith(fam):
            return fam
    if s.startswith("PASSIVE") or s.startswith("STAR_VESSEL") or s.startswith("SV"):
        return "PASSIVE"
    return "OTHER"


def _plot_machine_cad_overlay(ax, geom: Dict[str, Any]) -> None:
    """
    Overlay MAST-U-like machine semantics:
      - passive structures/polygons
      - active coils
      - WALL_OUTER blanket back plate
      - WALL_INNER first wall/limiter
      - optional plasma target
    """
    # Passive solid polygons first, under everything else.
    passive_structs = geom.get("passive_structures", []) or []
    passive_poly_label_done = False
    for ps in passive_structs:
        try:
            xy = np.asarray(ps.get("xy"), float)
            if xy.ndim == 2 and xy.shape[0] >= 3:
                ax.fill(
                    xy[:, 0], xy[:, 1],
                    facecolor="0.75", edgecolor="0.35",
                    alpha=0.28, linewidth=0.35,
                    label="STAR_VESSEL passive structures" if not passive_poly_label_done else None,
                    zorder=0.5,
                )
                passive_poly_label_done = True
        except Exception:
            continue

    # Optional passive filament centers/rectangles. Full rectangles can be visually heavy.
    plot_passive_filaments = bool(getattr(cfg, "plot_passive_filaments", False))
    passive_filaments = geom.get("passive_filaments", []) or []
    if passive_filaments and plot_passive_filaments:
        max_passive_plot = int(getattr(cfg, "max_passive_filaments_plot", 1200))
        step = max(1, int(np.ceil(len(passive_filaments) / max_passive_plot)))
        label_done = False
        for item in passive_filaments[::step]:
            try:
                _lab, Rc, Zc, dR, dZ = item[:5]
                _plot_rect_outline(
                    ax, float(Rc), float(Zc), float(dR), float(dZ),
                    color="0.45", lw=0.20, alpha=0.30,
                    label="passive filaments" if not label_done else None,
                    zorder=0.7,
                )
                label_done = True
            except Exception:
                continue

    # Active coils: plot filament rectangles thinly, grouped by family color.
    fam_styles = {
        "CS":  dict(color="tab:red",    lw=0.22, alpha=0.75),
        "PF1": dict(color="tab:blue",   lw=0.25, alpha=0.80),
        "PF2": dict(color="tab:orange", lw=0.25, alpha=0.80),
        "PF3": dict(color="tab:green",  lw=0.25, alpha=0.80),
        "PF4": dict(color="tab:purple", lw=0.25, alpha=0.80),
        "PF5": dict(color="tab:brown",  lw=0.25, alpha=0.80),
        "PF6": dict(color="tab:pink",   lw=0.25, alpha=0.80),
        "OTHER": dict(color="0.25",     lw=0.20, alpha=0.60),
    }
    label_done_by_fam: Dict[str, bool] = {}
    for lab, val in (geom.get("coils", {}) or {}).items():
        try:
            Rc, Zc, dR, dZ = map(float, val[:4])
        except Exception:
            continue
        fam = _family_from_label(str(lab))
        style = fam_styles.get(fam, fam_styles["OTHER"])
        _plot_rect_outline(
            ax, Rc, Zc, dR, dZ,
            **style,
            label=f"{fam} active" if not label_done_by_fam.get(fam, False) else None,
            zorder=1.5,
        )
        label_done_by_fam[fam] = True

    # Walls / blanket / limiter.
    if "R_outer" in geom and "Z_outer" in geom:
        ax.plot(
            geom["R_outer"], geom["Z_outer"],
            color="0.15", lw=2.0, ls="-",
            label="WALL_OUTER / blanket outer",
            zorder=2.5,
        )

    if "R_inner" in geom and "Z_inner" in geom:
        ax.plot(
            geom["R_inner"], geom["Z_inner"],
            color="k", lw=1.8, ls="--",
            label="WALL_INNER / limiter",
            zorder=2.8,
        )

    # # Plasma target if present.
    # if "R_plasma" in geom and "Z_plasma" in geom:
    #     ax.plot(
    #         geom["R_plasma"], geom["Z_plasma"],
    #         color="tab:cyan", lw=1.4, alpha=0.8,
    #         label="CAD/AUTO plasma target",
    #         zorder=2.0,
    #     )
    #

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

    if dxf_path is None:
        dxf_path = str(getattr(cfg, "dxf_path", _default_dxf()))

    # Build CAD opts FROM CONFIG.
    #
    # MAST-U-like semantic setup:
    #   COIL_*      -> active coils
    #   STAR_VESSEL -> passive metallic structures
    #   WALL_INNER  -> first wall / limiter / plasma-accessible boundary
    #   WALL_OUTER  -> blanket outer/back plate
    #
    # Legacy artificial blanket fill stays disabled unless explicitly enabled in cfg.
    opts = CADImportOptions(
        unit_scale=getattr(cfg, "unit_scale", None),
        resample_walls=str(getattr(cfg, "resample_walls", "auto")),
        n_wall=int(getattr(cfg, "n_wall", 1601)),
        n_inner=int(getattr(cfg, "n_inner", 2001)),
        n_plasma=int(getattr(cfg, "n_plasma", 501)),
        min_wall_pts=int(getattr(cfg, "min_wall_pts", 400)),
        enforce_ccw=bool(getattr(cfg, "enforce_ccw", True)),
        canonical_start=bool(getattr(cfg, "canonical_start", True)),

        prefer_path_flattening=bool(getattr(cfg, "prefer_path_flattening", True)),
        flatten_distance=float(getattr(cfg, "flatten_distance", 0.002)),
        max_seg_len_wall=float(getattr(cfg, "max_seg_len_wall", 0.008)),
        max_seg_len_plasma=float(getattr(cfg, "max_seg_len_plasma", 0.008)),

        label_match_factor=float(getattr(cfg, "label_match_factor", 2.0)),

        plasma_target_mode=str(getattr(cfg, "plasma_target_mode", "auto")),
        plasma_fit_to_inner_if_available=bool(getattr(cfg, "plasma_fit_to_inner_if_available", True)),
        plasma_R0=float(getattr(cfg, "plasma_R0", 4.0)),
        plasma_A=float(getattr(cfg, "plasma_A", 2.0)),
        plasma_kappa=float(getattr(cfg, "plasma_kappa", 2.5)),
        plasma_Z0=float(getattr(cfg, "plasma_Z0", 0.0)),
        plasma_delta_max=float(getattr(cfg, "plasma_delta_max", 0.70)),
        plasma_delta_grid=int(getattr(cfg, "plasma_delta_grid", 17)),
        plasma_delta_symmetric=bool(getattr(cfg, "plasma_delta_symmetric", True)),
        plasma_shrink_iters=int(getattr(cfg, "plasma_shrink_iters", 20)),
        plasma_scale_safety=float(getattr(cfg, "plasma_scale_safety", 0.999)),
        containment_radius=float(getattr(cfg, "containment_radius", -1e-9)),
        fix_center_if_outside=bool(getattr(cfg, "fix_center_if_outside", True)),
        center_search_samples=int(getattr(cfg, "center_search_samples", 800)),
        center_search_seed=int(getattr(cfg, "center_search_seed", 0)),
        strike_ray_fallback_len=float(getattr(cfg, "strike_ray_fallback_len", 3.0)),

        # Legacy blanket fill: keep OFF for the new CAD architecture.
        blanket_enabled=bool(getattr(cfg, "blanket_enabled", False)),
        blanket_n_filaments=int(getattr(cfg, "blanket_n_filaments", 0)),
        blanket_distribution=str(getattr(cfg, "blanket_distribution", "stratified")),
        blanket_seed=int(getattr(cfg, "blanket_seed", 0)),
        blanket_wall_margin_m=float(getattr(cfg, "blanket_wall_margin_m", 0.01)),
        blanket_filament_dR=float(getattr(cfg, "blanket_filament_dR", 0.004)),
        blanket_filament_dZ=float(getattr(cfg, "blanket_filament_dZ", 0.004)),
        blanket_bins_R=int(getattr(cfg, "blanket_bins_R", 0)),
        blanket_bins_Z=int(getattr(cfg, "blanket_bins_Z", 0)),
        blanket_pitch_mode=str(getattr(cfg, "blanket_pitch_mode", "auto")),
        blanket_pitch_R=float(getattr(cfg, "blanket_pitch_R", 0.03)),
        blanket_pitch_Z=float(getattr(cfg, "blanket_pitch_Z", 0.03)),
        blanket_label_prefix=str(getattr(cfg, "blanket_label_prefix", "BLK")),
        blanket_containment_radius=float(getattr(cfg, "blanket_containment_radius", -1e-9)),

        # New preferred passive model: STAR_VESSEL/PASSIVE_* polygons.
        passive_structures_enabled=bool(getattr(cfg, "passive_structures_enabled", True)),
        passive_use_star_vessel=bool(getattr(cfg, "passive_use_star_vessel", True)),
        passive_use_passive_prefix=bool(getattr(cfg, "passive_use_passive_prefix", True)),
        passive_target_dR_m=float(getattr(cfg, "passive_target_dR_m", 0.10)),
        passive_target_dZ_m=float(getattr(cfg, "passive_target_dZ_m", 0.10)),
        passive_nR_max=int(getattr(cfg, "passive_nR_max", 80)),
        passive_nZ_max=int(getattr(cfg, "passive_nZ_max", 160)),
        passive_min_cell_area_m2=float(getattr(cfg, "passive_min_cell_area_m2", 1.0e-6)),
        passive_containment_radius=float(getattr(cfg, "passive_containment_radius", -1e-9)),
        star_vessel_material=str(getattr(cfg, "star_vessel_material", "SS316L")),
        star_vessel_resistivity_ohm_m=float(getattr(cfg, "star_vessel_resistivity_ohm_m", 0.75e-6)),
        passive_default_material=str(getattr(cfg, "passive_default_material", "SS316L")),
        passive_default_resistivity_ohm_m=float(getattr(cfg, "passive_default_resistivity_ohm_m", 0.75e-6)),
        first_wall_material=str(getattr(cfg, "first_wall_material", "EUROFER97")),
        first_wall_resistivity_ohm_m=float(getattr(cfg, "first_wall_resistivity_ohm_m", 1.0e-6)),
        blanket_outer_material=str(getattr(cfg, "blanket_outer_material", "EUROFER97")),
        blanket_outer_resistivity_ohm_m=float(getattr(cfg, "blanket_outer_resistivity_ohm_m", 1.0e-6)),
        machine_wall_source=str(getattr(cfg, "machine_wall_source", "outer")),
        limiter_source=str(getattr(cfg, "limiter_source", "inner")),

        # Active coil discretization.
        coil_discretize_active=bool(getattr(cfg, "coil_discretize_active", True)),
        coil_target_dR_m=float(getattr(cfg, "coil_target_dR_m", 0.12)),
        coil_target_dZ_m=float(getattr(cfg, "coil_target_dZ_m", 0.12)),
        coil_target_dR_CS_m=float(getattr(cfg, "coil_target_dR_CS_m", 0.08)),
        coil_target_dZ_CS_m=float(getattr(cfg, "coil_target_dZ_CS_m", 0.18)),
        coil_nR_min=int(getattr(cfg, "coil_nR_min", 2)),
        coil_nZ_min=int(getattr(cfg, "coil_nZ_min", 2)),
        coil_nR_max=int(getattr(cfg, "coil_nR_max", 8)),
        coil_nZ_max=int(getattr(cfg, "coil_nZ_max", 48)),

        # Segmented CS groups for shaping sensitivity.
        cs_segmented=bool(getattr(cfg, "cs_segmented", True)),
        cs_mid_fraction=float(getattr(cfg, "cs_mid_fraction", 0.45)),
        cs_segment_zcut_m=getattr(cfg, "cs_segment_zcut_m", None),
        cs_segment_keep_parent=bool(getattr(cfg, "cs_segment_keep_parent", True)),

        fill_factor=float(getattr(cfg, "fill_factor", 0.75)),
        Jeng_default_A_per_mm2=float(getattr(cfg, "Jeng_default_A_per_mm2", 40.0)),
        family_mode=str(getattr(cfg, "family_mode", "min")),
        family_J_override_A_per_mm2=getattr(cfg, "family_J_override_A_per_mm2", None),
    )

    tokamak, geom = make_star_machine_from_cad(
        dxf_path=str(dxf_path),
        layers=CADLayers(),
        opts=opts,
        strict_expected=True,
    )

    # Domain
    #
    # New default: include the full CAD machine (active coils + STAR_VESSEL passives)
    # instead of only WALL_OUTER. This is closer to the MAST-U-like setup where
    # active coils, passive structures, limiter and wall are all represented.
    margin = float(getattr(cfg, "margin_RZ", 0.5))
    domain_source = str(getattr(cfg, "eq_domain_source", "outer")) #machine outer
    Rmin, Rmax, Zmin, Zmax = _domain_from_geom(geom, margin=margin, source=domain_source)

    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=Rmin, Rmax=Rmax,
        Zmin=Zmin, Zmax=Zmax,
        nx=int(getattr(cfg, "nx_eq", 65)),
        ny=int(getattr(cfg, "ny_eq", 129)),
    )
    solver = GSstaticsolver.NKGSsolver(eq)

    f_list = tuple(getattr(cfg, "f_list_equilibrium", (1.0,)))
    tol_ramp = float(getattr(cfg, "target_rel_tol_ramp", 1e-9))
    tol_final = float(getattr(cfg, "target_rel_tol", 1e-8))
    mode = _sanitize_mode(getattr(cfg, "coil_group_mode", "area"))

    vacuum_only = bool(getattr(cfg, "vacuum_only", False))
    Ip_set = 0.0 if vacuum_only else float(getattr(cfg, "Ip", 0.0))
    paxis_set = 0.0 if vacuum_only else float(getattr(cfg, "paxis", 0.0))

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
        print(f"Domain R=[{Rmin:.2f},{Rmax:.2f}] Z=[{Zmin:.2f},{Zmax:.2f}]")
        print(f"Grid nx={getattr(cfg, 'nx_eq', 65)} ny={getattr(cfg, 'ny_eq', 129)}")
        print(f"coil_group_mode = {mode}")
        print(f"vacuum_only = {vacuum_only}  (Ip={Ip_set}  paxis={paxis_set})")
        print(f"domain_source = {domain_source}")
        print(f"active_coils = {len(getattr(tokamak, 'active_coils', []) or [])}")
        print(f"passive_coils = {len(getattr(tokamak, 'passive_coils', []) or [])}")
        if "geometry_semantics" in geom:
            print(f"[GEOM] {geom['geometry_semantics']}")
        if "materials_meta" in geom:
            print(f"[MATERIALS] {geom['materials_meta']}")
        if "passive_meta" in geom:
            print(f"[PASSIVE] {geom['passive_meta']}")
        if "blanket_meta" in geom:
            print(f"[BLANKET legacy] {geom['blanket_meta']}")
        if "blanket_region" in geom:
            print(f"[BLANKET region] {geom['blanket_region']}")

    for j, f in enumerate(f_list, start=1):
        this_tol = tol_final if (j == len(f_list)) else tol_ramp

        # NUEVO: incluye PF4–PF6 (mantiene PF1–PF3 como baseline desde cfg)
        seg_info = geom.get("cs_segment_info", {}) or {}
        cs_seg_enabled_runtime = bool(seg_info.get("enabled", False))

        totals_A = {
            "CS":  float(f) * float(getattr(cfg, "CS_current", 0.0)),
            "PF1": float(f) * float(getattr(cfg, "PF1_current", 0.0)),
            "PF2": float(f) * float(getattr(cfg, "PF2_current", 0.0)),
            "PF3": float(f) * float(getattr(cfg, "PF3_current", 0.0)),
            "PF4": float(f) * float(getattr(cfg, "PF4_current", 0.0)),
            "PF5": float(f) * float(getattr(cfg, "PF5_current", 0.0)),
            "PF6": float(f) * float(getattr(cfg, "PF6_current", 0.0)),
        }

        if cs_seg_enabled_runtime:
            totals_A["CS_MID"] = float(f) * float(getattr(cfg, "CS_MID_current", 0.0))
            totals_A["CS_END"] = float(f) * float(getattr(cfg, "CS_END_current", 0.0))

        apply_family_currents(tokamak, totals_A, mode=mode)
        if verbose:
            print_group_currents_sanity(tokamak, totals_A, mode, header="--- Coil currents sanity check ---")

        profiles = _make_profiles(
            eq,
            paxis=float(f) * float(paxis_set),
            Ip=float(f) * float(Ip_set),
            fvac=float(getattr(cfg, "fvac", 0.5)),
            alpha_m=float(getattr(cfg, "alpha_m", 1.8)),
            alpha_n=float(getattr(cfg, "alpha_n", 1.2)),
        )

        if verbose:
            print(f"\n[continuation] step {j}/{len(f_list)} f={f:.3f} tol={this_tol:.1e}")

        with _redirect_solver_noise(redirect_solver_noise, noise_path):
            solver.solve(
                eq=eq,
                profiles=profiles,
                constrain=None,
                target_relative_tolerance=float(this_tol),
                verbose=False,
            )

    shape = compute_shape(eq, geom)

    # NEW: Always compute plasma params from any closed LCFS that encloses the axis (even without separatrix)
    diag = plasma_diagnostics(eq, geom, shape)
    shape["plasma_diag"] = diag
    if verbose:
        print_plasma_diagnostics(diag)

    return eq, tokamak, geom, shape


# -------------------------
# Plot helpers (robust)
# -------------------------
def _safe_eq_plot(eq: Any, ax):
    try:
        eq.plot(axis=ax, show=False)
        return
    except Exception:
        pass

    # fallback: contour psi
    try:
        psi = eq.psi() if callable(getattr(eq, "psi", None)) else getattr(eq, "psi")
    except Exception:
        psi = None
    try:
        R = eq.R() if callable(getattr(eq, "R", None)) else getattr(eq, "R")
        Z = eq.Z() if callable(getattr(eq, "Z", None)) else getattr(eq, "Z")
    except Exception:
        R = Z = None

    if psi is None or R is None or Z is None:
        return

    psi = np.asarray(psi, float)
    R = np.asarray(R, float)
    Z = np.asarray(Z, float)

    if R.ndim == 2 and Z.ndim == 2:
        RR = R
        ZZ = Z
    else:
        RR, ZZ = np.meshgrid(R, Z, indexing="xy")

    ax.contour(RR, ZZ, psi, levels=30)


def _iter_blanket_rects(geom: Dict[str, Any]) -> Iterator[Tuple[float, float, float, float]]:
    """
    Yield (Rc, Zc, dR, dZ) for blanket filaments with maximum backward compatibility.

    Supported:
      - (label, Rc, Zc, dR, dZ)
      - (Rc, Zc, dR, dZ)
      - dict with keys Rc,Zc,dR,dZ
      - generic sequences where we can extract 4 numeric values
    """
    items = geom.get("blanket_filaments", None)
    if not items:
        return
        yield  # pragma: no cover

    for item in items:
        try:
            if isinstance(item, dict):
                Rc = float(item["Rc"]); Zc = float(item["Zc"]); dR = float(item["dR"]); dZ = float(item["dZ"])
                yield (Rc, Zc, dR, dZ)
                continue

            tup = list(item)

            # (Rc,Zc,dR,dZ)
            if len(tup) == 4:
                Rc, Zc, dR, dZ = map(float, tup)
                yield (Rc, Zc, dR, dZ)
                continue

            # (label,Rc,Zc,dR,dZ)
            if len(tup) >= 5 and isinstance(tup[0], (str, np.str_)):
                Rc, Zc, dR, dZ = map(float, tup[1:5])
                yield (Rc, Zc, dR, dZ)
                continue

            # last resort: first 4 numeric entries
            nums = []
            for x in tup:
                if isinstance(x, (int, float, np.floating)):
                    nums.append(float(x))
                if len(nums) >= 4:
                    break
            if len(nums) >= 4:
                Rc, Zc, dR, dZ = nums[:4]
                yield (Rc, Zc, dR, dZ)

        except Exception:
            continue


def plot_equilibrium(eq: Any, geom: Dict[str, Any], shape: Dict[str, Any], filename: Optional[str] = None) -> None:
    """
    Plot equilibrium + full CAD machine context.

    This is intentionally MAST-U-like:
      - active coils are visible
      - STAR_VESSEL passive structures are visible
      - WALL_INNER is the limiter/first wall
      - WALL_OUTER is the blanket outer/back plate
      - LCFS/separatrix is overlaid
    """
    fig, ax = plt.subplots(figsize=(7.2, 10.5))

    # FreeGSNKE psi contours. This only covers the equilibrium computational domain.
    _safe_eq_plot(eq, ax)

    # Full CAD overlay.
    _plot_machine_cad_overlay(ax, geom)

    # LCFS/separatrix from analyze_star if present.
    plotted = False
    R_sep = shape.get("R_sep", None)
    Z_sep = shape.get("Z_sep", None)
    if R_sep is not None and Z_sep is not None:
        try:
            if len(R_sep) > 10 and len(Z_sep) == len(R_sep):
                ax.plot(R_sep, Z_sep, color="tab:blue", lw=2.6, label="LCFS/Separatrix", zorder=4.0)
                plotted = True
        except Exception:
            pass

    # If no analyze_star separatrix, plot diagnostic LCFS fallback.
    if not plotted:
        diag = (shape.get("plasma_diag", None) or {})
        Rf = diag.get("R_lcfs", None)
        Zf = diag.get("Z_lcfs", None)
        try:
            if Rf is not None and Zf is not None and len(Rf) > 20:
                ax.plot(
                    Rf, Zf,
                    color="tab:blue", lw=2.2,
                    label=f"LCFS fallback ({diag.get('method','?')})",
                    zorder=4.0,
                )
        except Exception:
            pass

    # Magnetic axis if available.
    diag = (shape.get("plasma_diag", None) or {})
    try:
        Rax = float(diag.get("R_ax", shape.get("R_ax", np.nan)))
        Zax = float(diag.get("Z_ax", shape.get("Z_ax", np.nan)))
        if np.isfinite(Rax) and np.isfinite(Zax):
            ax.plot(Rax, Zax, "x", ms=8, mew=2, color="tab:green", label="Magnetic axis", zorder=5.0)
    except Exception:
        pass

    # X-points if analyze_star found them.
    for key, label in (("xpoints", "Detected X-points"), ("x_points", "Detected X-points")):
        xs = shape.get(key, None)
        if xs:
            done = False
            for p in xs:
                try:
                    if isinstance(p, dict):
                        xr, xz = float(p.get("R", p.get("r"))), float(p.get("Z", p.get("z")))
                    else:
                        xr, xz = float(p[0]), float(p[1])
                    ax.plot(xr, xz, marker="x", ms=7, mew=1.8, color="red", ls="", label=label if not done else None, zorder=5.0)
                    done = True
                except Exception:
                    pass
            break

    # Use full machine bbox for view so coils/passives are visible.
    Rmin0, Rmax0, Zmin0, Zmax0 = _machine_geometry_bbox(geom, include_coils=True, include_passives=True)
    pad_R = float(getattr(cfg, "plot_pad_R", 0.35))
    pad_Z = float(getattr(cfg, "plot_pad_Z", 0.35))
    ax.set_xlim(max(0.0, Rmin0 - pad_R), Rmax0 + pad_R)
    ax.set_ylim(Zmin0 - pad_Z, Zmax0 + pad_Z)

    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("STAR-like equilibrium: active coils + passive structures + limiter/wall")

    # Reduce duplicate legend entries.
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    handles2, labels2 = [], []
    for h, lab in zip(handles, labels):
        if not lab or lab in seen:
            continue
        seen.add(lab)
        handles2.append(h)
        labels2.append(lab)
    ax.legend(handles2, labels2, loc="upper right", fontsize=8)

    fig.tight_layout()

    if filename is None:
        filename = str(getattr(cfg, "fig_equilibrium", "STAR_bean_equilibrium.png"))
    out = _results_dir() / filename
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"[SAVED] {out}")


def plot_machine_setup(geom: Dict[str, Any], filename: str = "STAR_machine_setup.png") -> None:
    """
    CAD-only plot for checking the imported machine semantics without solving an equilibrium.
    """
    fig, ax = plt.subplots(figsize=(7.2, 10.5))
    _plot_machine_cad_overlay(ax, geom)

    Rmin0, Rmax0, Zmin0, Zmax0 = _machine_geometry_bbox(geom, include_coils=True, include_passives=True)
    pad_R = float(getattr(cfg, "plot_pad_R", 0.35))
    pad_Z = float(getattr(cfg, "plot_pad_Z", 0.35))
    ax.set_xlim(max(0.0, Rmin0 - pad_R), Rmax0 + pad_R)
    ax.set_ylim(Zmin0 - pad_Z, Zmax0 + pad_Z)
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("STAR CAD machine setup")

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    handles2, labels2 = [], []
    for h, lab in zip(handles, labels):
        if not lab or lab in seen:
            continue
        seen.add(lab)
        handles2.append(h)
        labels2.append(lab)
    ax.legend(handles2, labels2, loc="upper right", fontsize=8)
    fig.tight_layout()
    out = _results_dir() / filename
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"[SAVED] {out}")


def main():
    eq, tokamak, geom, shape = build_equilibrium(verbose=True, redirect_solver_noise=True)

    if bool(getattr(cfg, "plot_machine_setup", True)):
        plot_machine_setup(geom, filename=str(getattr(cfg, "fig_machine_setup", "STAR_machine_setup.png")))

    plot_equilibrium(eq, geom, shape, filename=str(getattr(cfg, "fig_equilibrium", "STAR_bean_equilibrium.png")))


if __name__ == "__main__":
    main()

