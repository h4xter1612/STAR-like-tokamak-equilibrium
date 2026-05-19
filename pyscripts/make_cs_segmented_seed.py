#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
make_cs_segmented_seed.py

Generate seed JSON files for an effective segmented-CS model.

The segmented model assumes the CAD/import layer has created these groups:

    CS_MID
    CS_END

and, preferably, still keeps parent CS for backward compatibility.

This script modifies only the current dictionary in an existing best/seed JSON:

    CS -> 0.0
    CS_MID -> total_CS * mid_share
    CS_END -> total_CS * end_share

where:

    mid_share + end_share = 1.0

Example:
    base CS = -5.5 MA
    mid_share = 0.65

    CS_MID = -3.575 MA
    CS_END = -1.925 MA
    CS     = 0.0 MA

This avoids accidentally doubling the CS current.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        y = float(x)
        if math.isfinite(y):
            return y
    except Exception:
        pass
    return default


def _sanitize_float_for_name(x: float) -> str:
    """
    Convert -5.5 -> m5p5, 0.65 -> 0p65.
    """
    s = f"{x:.6g}"
    s = s.replace("-", "m")
    s = s.replace("+", "p")
    s = s.replace(".", "p")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    return s


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def _get_current_dict(seed: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract current dictionary from common seed/best JSON formats.

    Expected common key:
        best_currents_MA

    Fallbacks:
        currents_MA
        best_result.currents_MA
        best_result.best_currents_MA
    """
    candidates = [
        seed.get("best_currents_MA"),
        seed.get("currents_MA"),
        seed.get("best_result", {}).get("best_currents_MA")
        if isinstance(seed.get("best_result"), dict)
        else None,
        seed.get("best_result", {}).get("currents_MA")
        if isinstance(seed.get("best_result"), dict)
        else None,
    ]

    for cand in candidates:
        if isinstance(cand, dict) and cand:
            out: Dict[str, float] = {}
            for k, v in cand.items():
                fv = _safe_float(v, None)
                if fv is not None:
                    out[str(k)] = float(fv)
            if out:
                return out

    raise ValueError(
        "Could not find a valid current dictionary. Expected one of: "
        "best_currents_MA, currents_MA, best_result.currents_MA."
    )


def _set_current_dict_everywhere(seed, currents):
    currents_MA = {str(k): float(v) for k, v in currents.items()}
    currents_A = {str(k): float(v) * 1e6 for k, v in currents_MA.items()}

    seed["best_currents_MA"] = dict(currents_MA)
    seed["currents_MA"] = dict(currents_MA)
    seed["best_currents_A"] = dict(currents_A)
    seed["currents_A"] = dict(currents_A)

    if isinstance(seed.get("best_result"), dict):
        seed["best_result"]["best_currents_MA"] = dict(currents_MA)
        seed["best_result"]["currents_MA"] = dict(currents_MA)
        seed["best_result"]["best_currents_A"] = dict(currents_A)
        seed["best_result"]["currents_A"] = dict(currents_A)


def _parse_float_list(s: str) -> List[float]:
    vals: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


def _parse_pairs(s: str) -> List[Tuple[float, float]]:
    """
    Parse explicit pairs in MA:

        "-3.5,-2.0;-4.0,-1.5"

    meaning:

        CS_MID=-3.5, CS_END=-2.0
        CS_MID=-4.0, CS_END=-1.5
    """
    pairs: List[Tuple[float, float]] = []

    if not s.strip():
        return pairs

    for block in s.split(";"):
        block = block.strip()
        if not block:
            continue

        parts = [x.strip() for x in block.split(",") if x.strip()]
        if len(parts) != 2:
            raise ValueError(
                f"Invalid pair '{block}'. Expected format: mid,end;mid,end"
            )

        pairs.append((float(parts[0]), float(parts[1])))

    return pairs


def _make_segmented_currents(
    base_currents: Dict[str, float],
    *,
    total_cs_ma: float,
    mid_share: float,
    parent_cs_value: float = 0.0,
) -> Dict[str, float]:
    """
    Create a segmented CS current dictionary preserving total CS current.

    total_cs_ma is signed.
    Example:
        total_cs_ma = -5.5
        mid_share = 0.65

        CS_MID = -3.575
        CS_END = -1.925
        CS     = 0.0
    """
    mid_share = float(mid_share)
    mid_share = max(0.0, min(1.0, mid_share))
    end_share = 1.0 - mid_share

    out = dict(base_currents)

    out["CS"] = float(parent_cs_value)
    out["CS_MID"] = float(total_cs_ma * mid_share)
    out["CS_END"] = float(total_cs_ma * end_share)

    return out


def _make_explicit_segmented_currents(
    base_currents: Dict[str, float],
    *,
    cs_mid_ma: float,
    cs_end_ma: float,
    parent_cs_value: float = 0.0,
) -> Dict[str, float]:
    """
    Create a segmented CS current dictionary from explicit CS_MID and CS_END.
    """
    out = dict(base_currents)

    out["CS"] = float(parent_cs_value)
    out["CS_MID"] = float(cs_mid_ma)
    out["CS_END"] = float(cs_end_ma)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate segmented-CS seed JSON files from an existing best/seed JSON."
    )

    ap.add_argument(
        "--base",
        required=True,
        help="Base seed/best JSON file.",
    )
    ap.add_argument(
        "--out-dir",
        default=r".\results\seeds",
        help="Output directory.",
    )
    ap.add_argument(
        "--tag",
        default="csseg",
        help="Output filename tag.",
    )

    ap.add_argument(
        "--total-cs-ma",
        default=None,
        type=float,
        help=(
            "Total CS current in MA to repartition into CS_MID/CS_END. "
            "If omitted, uses base current CS from the JSON."
        ),
    )

    ap.add_argument(
        "--mid-shares",
        default="0.45,0.55,0.65,0.75",
        help=(
            "Comma-separated shares of total CS current assigned to CS_MID. "
            "CS_END gets 1-share. Example: 0.45,0.55,0.65"
        ),
    )

    ap.add_argument(
        "--total-scales",
        default="1.0",
        help=(
            "Comma-separated multipliers applied to total CS current. "
            "Example: 0.9,1.0,1.1"
        ),
    )

    ap.add_argument(
        "--explicit-pairs-ma",
        default="",
        help=(
            "Optional explicit CS_MID,CS_END pairs in MA. "
            "Example: \"-3.5,-2.0;-4.0,-1.5\". "
            "These are added in addition to share-based seeds."
        ),
    )

    ap.add_argument(
        "--parent-cs-value",
        default=0.0,
        type=float,
        help=(
            "Value assigned to parent CS key in the seed. "
            "Default 0.0 to avoid double-counting when CS_MID/CS_END are active."
        ),
    )

    ap.add_argument(
        "--also-write-summary",
        action="store_true",
        help="Write a small summary JSON with all generated cases.",
    )

    args = ap.parse_args()

    base_path = Path(args.base)
    out_dir = Path(args.out_dir)

    seed = _load_json(base_path)
    base_currents = _get_current_dict(seed)

    if args.total_cs_ma is None:
        if "CS" not in base_currents:
            raise ValueError(
                "Base JSON has no CS current. Pass --total-cs-ma explicitly."
            )
        total_cs_ma_base = float(base_currents["CS"])
    else:
        total_cs_ma_base = float(args.total_cs_ma)

    mid_shares = _parse_float_list(args.mid_shares)
    total_scales = _parse_float_list(args.total_scales)
    explicit_pairs = _parse_pairs(args.explicit_pairs_ma)

    generated: List[Dict[str, Any]] = []

    print("[INFO] make_cs_segmented_seed started")
    print(f"[INFO] base = {base_path}")
    print(f"[INFO] out_dir = {out_dir}")
    print(f"[INFO] tag = {args.tag}")
    print(f"[INFO] base total CS = {total_cs_ma_base:+.6f} MA")
    print(f"[INFO] mid_shares = {mid_shares}")
    print(f"[INFO] total_scales = {total_scales}")
    print(f"[INFO] parent CS value = {args.parent_cs_value:+.6f} MA")

    # Share-based cases.
    for scale in total_scales:
        total_cs_ma = float(total_cs_ma_base * scale)

        for mid_share in mid_shares:
            mid_share = float(mid_share)
            if not (0.0 < mid_share < 1.0):
                raise ValueError(
                    f"Invalid mid_share={mid_share}. Use 0 < mid_share < 1."
                )

            currents = _make_segmented_currents(
                base_currents,
                total_cs_ma=total_cs_ma,
                mid_share=mid_share,
                parent_cs_value=float(args.parent_cs_value),
            )

            case = copy.deepcopy(seed)
            _set_current_dict_everywhere(case, currents)

            case.setdefault("metadata", {})
            case["metadata"]["cs_segmented_seed"] = {
                "enabled": True,
                "mode": "share",
                "source_base": str(base_path),
                "total_cs_ma": float(total_cs_ma),
                "total_scale": float(scale),
                "mid_share": float(mid_share),
                "end_share": float(1.0 - mid_share),
                "CS": float(currents["CS"]),
                "CS_MID": float(currents["CS_MID"]),
                "CS_END": float(currents["CS_END"]),
                "note": (
                    "CS parent set to parent_cs_value to avoid double-counting. "
                    "Use CADImportOptions(cs_segmented=True) in the solver/optimizer."
                ),
            }

            fname = (
                f"{args.tag}_total_{_sanitize_float_for_name(total_cs_ma)}MA_"
                f"midfrac_{_sanitize_float_for_name(mid_share)}.json"
            )
            out_path = out_dir / fname
            _save_json(out_path, case)

            generated.append(
                {
                    "path": str(out_path),
                    "mode": "share",
                    "total_cs_ma": total_cs_ma,
                    "mid_share": mid_share,
                    "end_share": 1.0 - mid_share,
                    "CS": currents["CS"],
                    "CS_MID": currents["CS_MID"],
                    "CS_END": currents["CS_END"],
                }
            )

            print(
                "[OK] wrote "
                f"{out_path} | total={total_cs_ma:+.3f} MA "
                f"mid_share={mid_share:.3f} "
                f"CS_MID={currents['CS_MID']:+.3f} "
                f"CS_END={currents['CS_END']:+.3f} "
                f"CS={currents['CS']:+.3f}"
            )

    # Explicit CS_MID/CS_END cases.
    for cs_mid_ma, cs_end_ma in explicit_pairs:
        currents = _make_explicit_segmented_currents(
            base_currents,
            cs_mid_ma=float(cs_mid_ma),
            cs_end_ma=float(cs_end_ma),
            parent_cs_value=float(args.parent_cs_value),
        )

        case = copy.deepcopy(seed)
        _set_current_dict_everywhere(case, currents)

        total_cs_ma = float(cs_mid_ma + cs_end_ma)
        mid_share = float(cs_mid_ma / total_cs_ma) if total_cs_ma != 0.0 else math.nan

        case.setdefault("metadata", {})
        case["metadata"]["cs_segmented_seed"] = {
            "enabled": True,
            "mode": "explicit",
            "source_base": str(base_path),
            "total_cs_ma": float(total_cs_ma),
            "mid_share": float(mid_share) if math.isfinite(mid_share) else None,
            "CS": float(currents["CS"]),
            "CS_MID": float(currents["CS_MID"]),
            "CS_END": float(currents["CS_END"]),
            "note": (
                "Explicit CS_MID/CS_END values. "
                "CS parent set to parent_cs_value to avoid double-counting. "
                "Use CADImportOptions(cs_segmented=True) in the solver/optimizer."
            ),
        }

        fname = (
            f"{args.tag}_explicit_mid_{_sanitize_float_for_name(cs_mid_ma)}MA_"
            f"end_{_sanitize_float_for_name(cs_end_ma)}MA.json"
        )
        out_path = out_dir / fname
        _save_json(out_path, case)

        generated.append(
            {
                "path": str(out_path),
                "mode": "explicit",
                "total_cs_ma": total_cs_ma,
                "mid_share": mid_share if math.isfinite(mid_share) else None,
                "CS": currents["CS"],
                "CS_MID": currents["CS_MID"],
                "CS_END": currents["CS_END"],
            }
        )

        print(
            "[OK] wrote "
            f"{out_path} | explicit "
            f"CS_MID={currents['CS_MID']:+.3f} "
            f"CS_END={currents['CS_END']:+.3f} "
            f"total={total_cs_ma:+.3f} "
            f"CS={currents['CS']:+.3f}"
        )

    if args.also_write_summary:
        summary_path = out_dir / f"{args.tag}_summary.json"
        _save_json(
            summary_path,
            {
                "base": str(base_path),
                "tag": args.tag,
                "base_total_cs_ma": total_cs_ma_base,
                "parent_cs_value": args.parent_cs_value,
                "generated": generated,
            },
        )
        print(f"[OK] summary: {summary_path}")

    print(f"[OK] generated {len(generated)} segmented-CS seed files")


if __name__ == "__main__":
    main()
