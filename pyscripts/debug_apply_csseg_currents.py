import star_machine_cad as sm
from star_machine_cad import apply_group_currents

tok, geom = sm.make_star_machine_from_cad(dxf_path=r".\star_baseline.dxf")

totals_A = {
    "CS": 0.0,
    "CS_MID": -3.575e6,
    "CS_END": -1.925e6,
    "PF1": -1.7558498983813403e6,
    "PF2": -2.9980866201337054e6,
    "PF3": 1.6926804443442654e6,
    "PF4": 6.071663348636366e6,
    "PF5": 5.376091303354473e6,
    "PF6": 11.263833721367357e6,
}

apply_group_currents(tok, totals_A, mode="area")

coil_map = getattr(tok, "coils_dict", {})
groups = getattr(tok, "coil_groups", {})

print("\n=== GROUPS ===")
print({k: len(v) for k, v in groups.items() if k in ["CS", "CS_MID", "CS_END", "PF4", "PF5", "PF6"]})

print("\n=== CURRENT SUMS ===")
for fam in ["CS", "CS_MID", "CS_END", "PF1", "PF2", "PF3", "PF4", "PF5", "PF6"]:
    labs = groups.get(fam, [])
    s = 0.0
    n = 0
    for lab in labs:
        c = coil_map.get(lab)
        if c is None:
            continue
        s += float(getattr(c, "current", 0.0))
        n += 1
    print(f"{fam:7s} n={n:4d} sum_current={s/1e6:+.6f} MA target={totals_A.get(fam, 0.0)/1e6:+.6f} MA")