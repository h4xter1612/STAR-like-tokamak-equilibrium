#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def get_currents_ma(j):
    for key in ("best_currents_MA", "currents_MA"):
        if isinstance(j.get(key), dict):
            return dict(j[key])

    br = j.get("best_result", {})
    if isinstance(br, dict):
        for key in ("currents_MA", "best_currents_MA"):
            if isinstance(br.get(key), dict):
                return dict(br[key])

    raise KeyError("Could not find currents in JSON. Expected best_currents_MA or best_result.currents_MA.")


def set_currents_ma(j, currents):
    j["best_currents_MA"] = dict(currents)
    j["currents_MA"] = dict(currents)

    if "best_currents_A" in j:
        j["best_currents_A"] = {k: float(v) * 1e6 for k, v in currents.items()}

    br = j.get("best_result", {})
    if isinstance(br, dict):
        br["currents_MA"] = dict(currents)
        br["currents_A"] = {k: float(v) * 1e6 for k, v in currents.items()}
        j["best_result"] = br

    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out-dir", default="./results/seeds")
    ap.add_argument("--cs-list", default="-2,-4,-6")
    ap.add_argument("--pf6-list", default="")
    ap.add_argument("--tag", default="cs_branch")
    args = ap.parse_args()

    base = load_json(args.base)
    base_currents = get_currents_ma(base)

    cs_values = [float(x.strip()) for x in args.cs_list.split(",") if x.strip()]

    if args.pf6_list.strip():
        pf6_values = [float(x.strip()) for x in args.pf6_list.split(",") if x.strip()]
    else:
        pf6_values = [None]

    out_dir = Path(args.out_dir)
    made = []

    for cs in cs_values:
        for pf6 in pf6_values:
            j = copy.deepcopy(base)
            curr = dict(base_currents)

            curr["CS"] = float(cs)

            if pf6 is not None:
                curr["PF6"] = float(pf6)

            j = set_currents_ma(j, curr)

            label = f"{args.tag}_CS_{cs:+.1f}MA".replace("+", "p").replace("-", "m").replace(".", "p")
            if pf6 is not None:
                label += f"_PF6_{pf6:.1f}MA".replace(".", "p")

            out = out_dir / f"{label}.json"
            save_json(out, j)
            made.append(out)

    print("[OK] Created seeds:")
    for p in made:
        print("   ", p)


if __name__ == "__main__":
    main()
