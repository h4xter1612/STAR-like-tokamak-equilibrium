import json
import numpy as np
from pathlib import Path

p = Path(r".\results\fit_simplified_dn_divertor_first_best.json")
j = json.loads(p.read_text(encoding="utf-8"))

out = []

def walk(o, path="root"):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, path + "." + str(k))
    elif isinstance(o, list):
        if len(o) > 10:
            try:
                arr = np.asarray(o, dtype=float)
                out.append((path, len(o), arr.shape, "numeric"))
            except Exception:
                arr = np.asarray(o, dtype=object)
                out.append((path, len(o), arr.shape, "non_numeric"))
        if len(o) < 20:
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")

walk(j.get("best_result", {}))

for item in out:
    print(item)