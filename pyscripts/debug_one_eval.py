# debug_one_eval.py
import json
import numpy as np

import config_star_bean as cfg
import star_equilibrium as se

def _as_list(x):
    try:
        return x.tolist()
    except Exception:
        return x

def main():
    eq, tokamak, geom, shape = se.build_equilibrium(
        verbose=True,
        redirect_solver_noise=True,
        dxf_path=None,   # o pon tu ruta DXF si quieres forzar
    )

    # --- resumen mínimo (NO dumping completo de arrays enormes)
    out = {
        "geom_keys": sorted(list(geom.keys())) if isinstance(geom, dict) else str(type(geom)),
        "shape_keys": sorted(list(shape.keys())) if isinstance(shape, dict) else str(type(shape)),
        "has_inner_wall": ("R_inner" in geom and "Z_inner" in geom) if isinstance(geom, dict) else False,
        "n_outer": len(geom.get("R_outer", [])) if isinstance(geom, dict) else None,
        "n_inner": len(geom.get("R_inner", [])) if isinstance(geom, dict) else None,
        "n_plasma_target": len(geom.get("R_plasma", [])) if isinstance(geom, dict) else None,
        "shape_ok_sep": bool(shape.get("ok_sep", False)) if isinstance(shape, dict) else None,
        "fallback_lcfs": shape.get("fallback_lcfs", None) if isinstance(shape, dict) else None,
    }

    # --- intenta localizar LCFS/separatrix en shape (para saber qué keys reales tienes)
    candidates = [
        ("R_sep","Z_sep"),
        ("R_separatrix","Z_separatrix"),
        ("R_lcfs","Z_lcfs"),
        ("lcfs_R","lcfs_Z"),
        ("R_LCFS","Z_LCFS"),
    ]
    found = None
    for rk, zk in candidates:
        if isinstance(shape, dict) and rk in shape and zk in shape:
            found = (rk, zk, len(shape[rk]))
            break
    out["lcfs_key_found"] = found

    # guarda snapshot
    with open("results/_debug_one_eval.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n[DEBUG] wrote results/_debug_one_eval.json")
    print("[DEBUG] lcfs_key_found =", found)
    print("[DEBUG] has_inner_wall =", out["has_inner_wall"])

if __name__ == "__main__":
    main()

