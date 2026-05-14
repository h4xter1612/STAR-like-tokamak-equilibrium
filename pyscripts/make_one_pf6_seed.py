import argparse
import json
from pathlib import Path

def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def extract_currents(j):
    if "best_currents_A" in j:
        return dict(j["best_currents_A"])
    if "best_result" in j and "currents_A" in j["best_result"]:
        return dict(j["best_result"]["currents_A"])
    if "currents_A" in j:
        return dict(j["currents_A"])
    raise RuntimeError("No encontré currents_A / best_currents_A en el JSON.")

def extract_physics(j):
    physics = j.get("physics", {})
    if not physics and "best_result" in j:
        physics = j["best_result"].get("physics", {})
    return {
        "Ip_A": float(physics.get("Ip_A", j.get("Ip_A", 4.0e6))),
        "paxis_Pa": float(physics.get("paxis_Pa", j.get("paxis_Pa", 2.0e3))),
        "fvac": float(physics.get("fvac", j.get("fvac", 20.8))),
        "alpha_m": float(physics.get("alpha_m", j.get("alpha_m", 1.5))),
        "alpha_n": float(physics.get("alpha_n", j.get("alpha_n", 1.1))),
    }

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--pf6-ma", type=float, required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

j = load_json(args.base)
currents = extract_currents(j)
physics = extract_physics(j)

currents["PF6"] = args.pf6_ma * 1e6

seed = {
    "schema": "star_seed_currents.v1",
    "name": f"manual_pf6_{args.pf6_ma:.2f}MA",
    **physics,
    "currents_A": currents,
    "currents_MA": {k: float(v) / 1e6 for k, v in currents.items()},
}

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(seed, indent=2), encoding="utf-8")
print(f"[OK] wrote {out}")
