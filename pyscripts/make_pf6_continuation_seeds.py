# make_pf6_continuation_seeds.py
import json
from pathlib import Path

base_path = Path("results/FINAL_gap_v2_best_toposafe.json")
out_dir = Path("results/seeds")
out_dir.mkdir(parents=True, exist_ok=True)

j = json.loads(base_path.read_text(encoding="utf-8"))

if "best_currents_A" in j:
    currents = dict(j["best_currents_A"])
elif "best_result" in j and "currents_A" in j["best_result"]:
    currents = dict(j["best_result"]["currents_A"])
else:
    raise RuntimeError("No encontré best_currents_A/currents_A en el JSON base.")

physics = j.get("physics", {})
if not physics and "best_result" in j:
    physics = j["best_result"].get("physics", {})

Ip_A = float(physics.get("Ip_A", 4.0e6))
paxis_Pa = float(physics.get("paxis_Pa", 2.0e3))
fvac = float(physics.get("fvac", 20.8))
alpha_m = float(physics.get("alpha_m", 1.5))
alpha_n = float(physics.get("alpha_n", 1.1))

for pf6_MA in [13.5, 12.0, 10.5, 9.0, 7.5]:
    c = dict(currents)
    c["PF6"] = pf6_MA * 1e6

    seed = {
        "schema": "star_seed_currents.v1",
        "name": f"gap_v2_pf6_{pf6_MA:.1f}MA",
        "Ip_A": Ip_A,
        "paxis_Pa": paxis_Pa,
        "fvac": fvac,
        "alpha_m": alpha_m,
        "alpha_n": alpha_n,
        "currents_A": c,
        "currents_MA": {k: v / 1e6 for k, v in c.items()}
    }

    out = out_dir / f"gap_v2_pf6_{str(pf6_MA).replace('.', 'p')}MA.json"
    out.write_text(json.dumps(seed, indent=2), encoding="utf-8")
    print("wrote", out)
