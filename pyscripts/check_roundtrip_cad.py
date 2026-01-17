# pyscripts/check_roundtrip_cad.py
from __future__ import annotations

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from star_machine import make_star_machine
from star_machine_cad import make_star_machine_from_cad


EXPECTED_COILS = ["CS", "PF1U", "PF1L", "PF2U", "PF2L", "PF3U", "PF3L"]


def _close_curve(R: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(R, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float).ravel()
    if len(R) < 3:
        raise ValueError("Curve must have >=3 points.")
    if not (np.isclose(R[0], R[-1]) and np.isclose(Z[0], Z[-1])):
        R = np.r_[R, R[0]]
        Z = np.r_[Z, Z[0]]
    return R, Z


def _resample_by_arclength(R: np.ndarray, Z: np.ndarray, n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    R, Z = _close_curve(R, Z)
    dR = np.diff(R)
    dZ = np.diff(Z)
    ds = np.sqrt(dR * dR + dZ * dZ)
    s = np.r_[0.0, np.cumsum(ds)]
    if s[-1] <= 0:
        raise ValueError("Degenerate curve with zero length.")
    su = np.linspace(0.0, s[-1], n, endpoint=False)
    Ru = np.interp(su, s, R)
    Zu = np.interp(su, s, Z)
    return Ru, Zu


def _curve_metrics(R: np.ndarray, Z: np.ndarray) -> dict:
    R, Z = _close_curve(R, Z)
    # polygon area (shoelace)
    area = 0.5 * np.sum(R[:-1] * Z[1:] - R[1:] * Z[:-1])
    bbox = (float(R.min()), float(R.max()), float(Z.min()), float(Z.max()))
    return {"area": float(area), "bbox": bbox, "npts": int(len(R))}


def _compare_curves(name: str, R1, Z1, R2, Z2) -> dict:
    R1u, Z1u = _resample_by_arclength(R1, Z1)
    R2u, Z2u = _resample_by_arclength(R2, Z2)
    dR = R2u - R1u
    dZ = Z2u - Z1u
    err = np.sqrt(dR * dR + dZ * dZ)
    out = {
        "name": name,
        "rms": float(np.sqrt(np.mean(err * err))),
        "max": float(np.max(err)),
        "mean_abs_dR": float(np.mean(np.abs(dR))),
        "mean_abs_dZ": float(np.mean(np.abs(dZ))),
    }
    out.update({f"{name}_ref": _curve_metrics(np.asarray(R1), np.asarray(Z1))})
    out.update({f"{name}_cad": _curve_metrics(np.asarray(R2), np.asarray(Z2))})
    return out


def _normalize_label(s: str) -> str:
    return str(s).strip().upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dxf", default=os.path.join("cad", "star_baseline.dxf"))
    ap.add_argument("--outfig", default=os.path.join("results", "roundtrip_overlay.png"))
    ap.add_argument("--outtxt", default=os.path.join("results", "roundtrip_report.txt"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.outfig), exist_ok=True)
    os.makedirs(os.path.dirname(args.outtxt), exist_ok=True)

    tok_ref, geom_ref = make_star_machine()
    tok_cad, geom_cad = make_star_machine_from_cad(args.dxf)

    # ---- label sanity
    labels_ref = sorted({_normalize_label(l) for l, _ in tok_ref.coils})
    labels_cad = sorted({_normalize_label(l) for l, _ in tok_cad.coils})

    missing = [c for c in EXPECTED_COILS if c not in labels_cad]
    extra = [c for c in labels_cad if c not in EXPECTED_COILS]

    # ---- curve comparisons
    rep = []
    rep.append(_compare_curves("outer_wall", geom_ref["R_outer"], geom_ref["Z_outer"],
                               geom_cad["R_outer"], geom_cad["Z_outer"]))
    rep.append(_compare_curves("inner_wall", geom_ref["R_inner"], geom_ref["Z_inner"],
                               geom_cad["R_inner"], geom_cad["Z_inner"]))

    # ---- coil comparisons
    coils_ref = geom_ref["coils"]
    coils_cad = geom_cad["coils"]
    coil_lines = []
    for name in EXPECTED_COILS:
        if name not in coils_ref or name not in coils_cad:
            continue
        Rc1, Zc1, dR1, dZ1 = map(float, coils_ref[name])
        Rc2, Zc2, dR2, dZ2 = map(float, coils_cad[name])
        coil_lines.append(
            (name, Rc2 - Rc1, Zc2 - Zc1, dR2 - dR1, dZ2 - dZ1)
        )

    # ---- write report
    with open(args.outtxt, "w", encoding="utf-8") as f:
        f.write("=== ROUNDTRIP CAD CHECK ===\n")
        f.write(f"DXF: {args.dxf}\n\n")

        f.write("Coil labels (ref): " + ", ".join(labels_ref) + "\n")
        f.write("Coil labels (cad): " + ", ".join(labels_cad) + "\n")
        f.write("Missing expected in CAD: " + ", ".join(missing) + "\n")
        f.write("Unexpected extras in CAD: " + ", ".join(extra) + "\n\n")

        for item in rep:
            f.write(f"[{item['name']}]\n")
            f.write(f"  RMS error [m]: {item['rms']:.6e}\n")
            f.write(f"  MAX error [m]: {item['max']:.6e}\n")
            f.write(f"  mean|dR| [m]:  {item['mean_abs_dR']:.6e}\n")
            f.write(f"  mean|dZ| [m]:  {item['mean_abs_dZ']:.6e}\n")
            f.write(f"  ref npts={item[item['name']+'_ref']['npts']} bbox={item[item['name']+'_ref']['bbox']} area={item[item['name']+'_ref']['area']:.6e}\n")
            f.write(f"  cad npts={item[item['name']+'_cad']['npts']} bbox={item[item['name']+'_cad']['bbox']} area={item[item['name']+'_cad']['area']:.6e}\n\n")

        f.write("[coil_deltas] (CAD - REF): dRc, dZc, ddR, ddZ in meters\n")
        for (name, dRc, dZc, ddR, ddZ) in coil_lines:
            f.write(f"  {name:5s}: {dRc:+.6e} {dZc:+.6e} {ddR:+.6e} {ddZ:+.6e}\n")

    # ---- overlay plot
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.plot(geom_ref["R_outer"], geom_ref["Z_outer"], "k-", lw=2, label="outer REF")
    ax.plot(geom_cad["R_outer"], geom_cad["Z_outer"], "r--", lw=1.5, label="outer CAD")
    ax.plot(geom_ref["R_inner"], geom_ref["Z_inner"], "k:", lw=2, label="inner REF")
    ax.plot(geom_cad["R_inner"], geom_cad["Z_inner"], "r-.", lw=1.5, label="inner CAD")

    for nm in EXPECTED_COILS:
        if nm in coils_ref:
            Rc, Zc, dR, dZ = map(float, coils_ref[nm])
            ax.add_patch(plt.Rectangle((Rc - dR, Zc - dZ), 2*dR, 2*dZ, fill=False))
        if nm in coils_cad:
            Rc, Zc, dR, dZ = map(float, coils_cad[nm])
            ax.add_patch(plt.Rectangle((Rc - dR, Zc - dZ), 2*dR, 2*dZ, fill=False, linestyle="--"))

    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.grid(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(args.outfig, dpi=200)
    print("Wrote:", args.outtxt)
    print("Saved:", args.outfig)


if __name__ == "__main__":
    main()

