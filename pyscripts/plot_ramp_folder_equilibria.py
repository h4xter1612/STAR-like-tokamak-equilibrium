# plot_ramp_folder_equilibria.py
from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import math
import os
import multiprocessing as mp
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CURRENT_KEYS = ["CS", "CS_MID", "CS_END", "PF1", "PF2", "PF3", "PF4", "PF5", "PF6"]


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _recursive_get(d: Any, key: str, default: Any = None) -> Any:
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            got = _recursive_get(v, key, None)
            if got is not None:
                return got
    elif isinstance(d, list):
        for v in d:
            got = _recursive_get(v, key, None)
            if got is not None:
                return got
    return default


def _extract_currents_A(d: Dict[str, Any]) -> Dict[str, float]:
    # Prefer amperes.
    for key in ("currents_A", "best_currents_A"):
        obj = _recursive_get(d, key, None)
        if isinstance(obj, dict) and obj:
            return {k: _safe_float(obj.get(k, 0.0), 0.0) for k in CURRENT_KEYS}

    # Fallback: MA.
    for key in ("currents_MA", "best_currents_MA"):
        obj = _recursive_get(d, key, None)
        if isinstance(obj, dict) and obj:
            return {k: 1e6 * _safe_float(obj.get(k, 0.0), 0.0) for k in CURRENT_KEYS}

    raise KeyError("No currents_A/currents_MA found in JSON")


def _parse_ip_paxis_from_name(name: str) -> Tuple[float, float]:
    """
    Examples:
      Ip5MA_p4050Pa_best.json
      Ip5p2MA_p4653Pa_best.json
      Ip6p2MA_p9315Pa_best.json
    """
    m = re.search(r"Ip(?P<ip>[0-9p]+)MA_p(?P<p>[0-9p]+)Pa", name)
    if not m:
        return float("nan"), float("nan")

    ip_ma = float(m.group("ip").replace("p", "."))
    paxis = float(m.group("p").replace("p", "."))
    return ip_ma, paxis


def _extract_physics(d: Dict[str, Any], path: Path) -> Tuple[float, float]:
    Ip_A = _recursive_get(d, "Ip_A", None)
    paxis = _recursive_get(d, "paxis_Pa", None)

    if Ip_A is not None and paxis is not None:
        return _safe_float(Ip_A), _safe_float(paxis)

    # Some files store physics nested.
    physics = _recursive_get(d, "physics", None)
    if isinstance(physics, dict):
        Ip_A = physics.get("Ip_A", physics.get("Ip", None))
        paxis = physics.get("paxis_Pa", physics.get("paxis", None))
        if Ip_A is not None and paxis is not None:
            return _safe_float(Ip_A), _safe_float(paxis)

    ip_ma, paxis_name = _parse_ip_paxis_from_name(path.name)
    if math.isfinite(ip_ma):
        return ip_ma * 1e6, paxis_name

    return float("nan"), float("nan")


def _extract_summary_from_json(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k in (
        "R0", "A", "kappa", "delta", "delta_u", "delta_l",
        "inside_WALL_INNER", "signed_gap_to_WALL_INNER_m",
        "outside_frac", "n_xpoints", "score", "ok", "reason",
    ):
        out[k] = _recursive_get(d, k, None)
    return out


def _patch_config_and_star_modules(
    *,
    cfg: Any,
    se: Any,
    currents_A: Dict[str, float],
    Ip_A: float,
    paxis_Pa: float,
    dxf_path: Path,
    args: argparse.Namespace,
) -> None:
    # Physics.
    for mod in (cfg, se):
        for name, val in [
            ("Ip", Ip_A),
            ("Ip_A", Ip_A),
            ("paxis", paxis_Pa),
            ("paxis_Pa", paxis_Pa),
        ]:
            try:
                setattr(mod, name, val)
            except Exception:
                pass

    # Currents.
    for k in CURRENT_KEYS:
        val = float(currents_A.get(k, 0.0))
        for attr in (f"{k}_current", f"{k}_current_A", k):
            try:
                setattr(cfg, attr, val)
            except Exception:
                pass
            try:
                setattr(se, attr, val)
            except Exception:
                pass

    # CAD/domain/passives.
    for mod in (cfg, se):
        for attr in ("dxf_path", "cad_dxf_path", "STAR_DXF_PATH", "default_dxf_path"):
            try:
                setattr(mod, attr, str(dxf_path))
            except Exception:
                pass

        for attr, val in [
            ("passive_structures_enabled", bool(args.passives)),
            ("passive_use_star_vessel", bool(args.passives)),
            ("plot_passives", bool(args.passives)),
            ("eq_domain_source", args.domain_source),
            ("nx", args.nx),
            ("ny", args.ny),
        ]:
            try:
                setattr(mod, attr, val)
            except Exception:
                pass

    # Env fallback.
    os.environ["STAR_IP_A"] = str(Ip_A)
    os.environ["STAR_PAXIS_PA"] = str(paxis_Pa)
    os.environ["STAR_EQ_DOMAIN_SOURCE"] = str(args.domain_source)
    os.environ["STAR_PASSIVE_STRUCTURES"] = "1" if args.passives else "0"
    os.environ["STAR_PASSIVE_USE_STAR_VESSEL"] = "1" if args.passives else "0"
    os.environ["STAR_PLOT_PASSIVES"] = "1" if args.passives else "0"


def _call_build_equilibrium(se: Any, dxf_path: Path) -> Tuple[Any, Any, Dict[str, Any], Dict[str, Any]]:
    fn = getattr(se, "build_equilibrium", None)
    if fn is None:
        raise AttributeError("star_equilibrium.build_equilibrium not found")

    sig = inspect.signature(fn)
    kwargs = {}

    # Only pass arguments that exist in your local function signature.
    for possible in ("dxf_path", "dxf", "cad_path"):
        if possible in sig.parameters:
            kwargs[possible] = str(dxf_path)

    for possible in ("make_plots", "plot", "do_plot", "save_plots"):
        if possible in sig.parameters:
            kwargs[possible] = False

    ret = fn(**kwargs)

    if isinstance(ret, tuple) and len(ret) >= 4:
        eq, tokamak, geom, shape = ret[:4]
        return eq, tokamak, geom, shape

    if isinstance(ret, dict):
        return ret.get("eq"), ret.get("tokamak"), ret.get("geom", {}), ret.get("shape", ret)

    raise RuntimeError(f"Unexpected build_equilibrium return type: {type(ret)}")


def _try_plot(se: Any, eq: Any, tokamak: Any, geom: Dict[str, Any], shape: Dict[str, Any], png: Path, title: str) -> bool:
    plot_names = [
        "plot_equilibrium",
        "plot_star_equilibrium",
        "plot_machine_equilibrium",
        "plot_equilibrium_with_machine",
        "plot_star_machine_equilibrium",
    ]

    for name in plot_names:
        fn = getattr(se, name, None)
        if fn is None:
            continue

        attempts = [
            ((eq, tokamak, geom, shape), {"save_path": str(png), "title": title}),
            ((eq, tokamak, geom, shape), {"out_path": str(png), "title": title}),
            ((eq, tokamak, geom, shape), {"fig_path": str(png), "title": title}),
            ((eq, tokamak, geom, shape), {"filename": str(png), "title": title}),
            ((eq, tokamak, geom, shape), {}),
            ((eq, geom, shape), {"save_path": str(png), "title": title}),
            ((eq, geom, shape), {}),
        ]

        for a, kw in attempts:
            try:
                plt.close("all")
                before = set(plt.get_fignums())
                ret = fn(*a, **kw)

                if hasattr(ret, "savefig"):
                    ret.savefig(png, dpi=220, bbox_inches="tight")
                    plt.close(ret)
                    return True

                after = set(plt.get_fignums())
                new_figs = list(after - before)
                if new_figs:
                    fig = plt.figure(new_figs[-1])
                    fig.savefig(png, dpi=220, bbox_inches="tight")
                    plt.close(fig)
                    return True

                # Some local plot functions save internally if save_path was provided.
                if png.exists() and png.stat().st_size > 0:
                    return True

            except TypeError:
                continue
            except Exception:
                continue

    return False


def _get_diag_from_shape(shape: Dict[str, Any]) -> Dict[str, Any]:
    for k in ("plasma_diag", "diag", "diagnostics"):
        obj = shape.get(k)
        if isinstance(obj, dict):
            return obj
    return {}


def _sort_key(p: Path) -> Tuple[float, float, str]:
    ip_ma, paxis = _parse_ip_paxis_from_name(p.name)
    if not math.isfinite(ip_ma):
        ip_ma = 1e9
    if not math.isfinite(paxis):
        paxis = 1e99
    return ip_ma, paxis, p.name
def _plot_one_worker(
    json_path_str: str,
    dxf_path_str: str,
    outdir_str: str,
    args_dict: Dict[str, Any],
    conn,
) -> None:
    """
    Build and plot one equilibrium in an isolated subprocess.

    This prevents one bad/stiff equilibrium from hanging the entire folder plotter.
    """
    try:
        json_path = Path(json_path_str)
        dxf_path = Path(dxf_path_str)
        outdir = Path(outdir_str)

        # Recreate a minimal argparse-like object.
        class ArgsObj:
            pass

        args = ArgsObj()
        for k, v in args_dict.items():
            setattr(args, k, v)

        # Import inside worker, not in parent.
        sys.path.insert(0, str(Path.cwd()))
        import config_star_bean as cfg
        import star_equilibrium as se

        d = _load_json(json_path)
        currents_A = _extract_currents_A(d)
        Ip_A, paxis_Pa = _extract_physics(d, json_path)
        js = _extract_summary_from_json(d)

        if not math.isfinite(Ip_A) or not math.isfinite(paxis_Pa):
            raise RuntimeError("Could not infer Ip/paxis")

        cfg = importlib.reload(cfg)
        se = importlib.reload(se)

        _patch_config_and_star_modules(
            cfg=cfg,
            se=se,
            currents_A=currents_A,
            Ip_A=Ip_A,
            paxis_Pa=paxis_Pa,
            dxf_path=dxf_path,
            args=args,
        )

        eq, tokamak, geom, shape = _call_build_equilibrium(se, dxf_path)
        diag = _get_diag_from_shape(shape)

        title = f"{json_path.stem} | Ip={Ip_A/1e6:.2f} MA, paxis={paxis_Pa:.3g} Pa"
        png = outdir / f"{json_path.stem}.png"

        ok_plot = _try_plot(se, eq, tokamak, geom, shape, png, title)

        row = {
            "file": str(json_path),
            "png": str(png) if png.exists() else "",
            "Ip_MA": Ip_A / 1e6,
            "paxis_Pa": paxis_Pa,
            "CS_MID_MA": currents_A.get("CS_MID", 0.0) / 1e6,
            "CS_END_MA": currents_A.get("CS_END", 0.0) / 1e6,
            "PF1_MA": currents_A.get("PF1", 0.0) / 1e6,
            "PF2_MA": currents_A.get("PF2", 0.0) / 1e6,
            "PF3_MA": currents_A.get("PF3", 0.0) / 1e6,
            "PF4_MA": currents_A.get("PF4", 0.0) / 1e6,
            "PF5_MA": currents_A.get("PF5", 0.0) / 1e6,
            "PF6_MA": currents_A.get("PF6", 0.0) / 1e6,

            "R0_json": js.get("R0"),
            "A_json": js.get("A"),
            "kappa_json": js.get("kappa"),
            "delta_json": js.get("delta"),
            "inside_json": js.get("inside_WALL_INNER"),
            "signed_gap_json": js.get("signed_gap_to_WALL_INNER_m"),
            "outside_frac_json": js.get("outside_frac"),

            "R0_rebuilt": diag.get("R0"),
            "A_rebuilt": diag.get("A"),
            "kappa_rebuilt": diag.get("kappa"),
            "delta_rebuilt": diag.get("delta", diag.get("delta_u")),
            "inside_rebuilt": diag.get("inside_WALL_INNER"),
            "signed_gap_rebuilt": diag.get("signed_gap_to_WALL_INNER_m"),
            "outside_frac_rebuilt": diag.get("outside_frac"),

            "status": "ok" if ok_plot else "built_but_plot_failed",
        }

        conn.send({"ok": True, "row": row})
        conn.close()

    except Exception as e:
        try:
            conn.send({
                "ok": False,
                "row": {
                    "file": str(json_path_str),
                    "status": "error",
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                },
            })
            conn.close()
        except Exception:
            pass


def _plot_one_with_timeout(
    *,
    json_path: Path,
    dxf_path: Path,
    outdir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Run one build+plot with timeout.
    If it hangs, terminate it and return a timeout row.
    """
    ctx = mp.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)

    args_dict = {
        "passives": int(args.passives),
        "domain_source": str(args.domain_source),
        "nx": args.nx,
        "ny": args.ny,
    }

    p = ctx.Process(
        target=_plot_one_worker,
        args=(
            str(json_path),
            str(dxf_path),
            str(outdir),
            args_dict,
            send_conn,
        ),
    )

    p.start()
    send_conn.close()

    timeout_s = float(args.build_timeout)

    if recv_conn.poll(timeout_s):
        try:
            msg = recv_conn.recv()
        except EOFError:
            msg = {
                "ok": False,
                "row": {
                    "file": str(json_path),
                    "status": "error",
                    "error": "EOFError receiving worker result",
                },
            }
    else:
        try:
            p.terminate()
        except Exception:
            pass

        msg = {
            "ok": False,
            "row": {
                "file": str(json_path),
                "status": "timeout",
                "error": f"build_timeout>{timeout_s:.1f}s",
            },
        }

    p.join(timeout=5.0)

    if p.is_alive():
        try:
            p.kill()
        except Exception:
            pass

    try:
        recv_conn.close()
    except Exception:
        pass

    return msg.get("row", {"file": str(json_path), "status": "unknown_error"})

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Ramp output folder, e.g. results/ramp_ip_paxis_coupled_...")
    ap.add_argument("--dxf", default=r".\cad\star_baseline.dxf")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--pattern", default="Ip*best.json")
    ap.add_argument("--passives", type=int, default=0)
    ap.add_argument("--domain-source", default="outer", choices=["outer", "machine", "inner"])
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--include-stage-files", action="store_true")
    ap.add_argument("--build-timeout", type=float, default=300.0, help="Seconds allowed per reconstructed equilibrium before skipping it.")
    args = ap.parse_args()

    folder = Path(args.folder).resolve()
    dxf_path = Path(args.dxf).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else folder / "equilibrium_plots"
    outdir.mkdir(parents=True, exist_ok=True)

    if not folder.exists():
        raise FileNotFoundError(folder)
    if not dxf_path.exists():
        raise FileNotFoundError(dxf_path)

    # Import after setting cwd/sys.path.
    sys.path.insert(0, str(Path.cwd()))
    import config_star_bean as cfg
    import star_equilibrium as se

    # Canonical accepted files.
    files = sorted(folder.glob(args.pattern), key=_sort_key)

    if args.include_stage_files:
        files += sorted(folder.glob("stage_*/*round_*_best.json"), key=_sort_key)

    # De-duplicate.
    seen = set()
    unique_files = []
    for p in files:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            unique_files.append(p)
    files = unique_files

    if args.limit and args.limit > 0:
        files = files[: args.limit]

    if not files:
        print(f"[ERROR] No JSON files found in {folder} with pattern={args.pattern}")
        return 2

    summary_csv = outdir / "ramp_plot_summary.csv"
    rows: List[Dict[str, Any]] = []

    print(f"[INFO] folder = {folder}")
    print(f"[INFO] dxf    = {dxf_path}")
    print(f"[INFO] outdir = {outdir}")
    print(f"[INFO] files  = {len(files)}")

    for i, p in enumerate(files, start=1):
        print("\n" + "=" * 100)
        print(f"[{i}/{len(files)}] {p.name}")

        row = _plot_one_with_timeout(
            json_path=p,
            dxf_path=dxf_path,
            outdir=outdir,
            args=args,
        )

        rows.append(row)

        status = row.get("status", "unknown")
        if status == "ok":
            print(f"[SAVED] {row.get('png', '')}")
            print(
                f"[DIAG] rebuilt: R0={row.get('R0_rebuilt')} "
                f"A={row.get('A_rebuilt')} "
                f"k={row.get('kappa_rebuilt')} "
                f"d={row.get('delta_rebuilt')} "
                f"inside={row.get('inside_rebuilt')} "
                f"gap={row.get('signed_gap_rebuilt')} "
                f"outside={row.get('outside_frac_rebuilt')}"
            )
        elif status == "built_but_plot_failed":
            print("[WARN] Build succeeded but plot failed.")
            print(
                f"[DIAG] rebuilt: R0={row.get('R0_rebuilt')} "
                f"A={row.get('A_rebuilt')} "
                f"k={row.get('kappa_rebuilt')} "
                f"d={row.get('delta_rebuilt')} "
                f"inside={row.get('inside_rebuilt')} "
                f"gap={row.get('signed_gap_rebuilt')} "
                f"outside={row.get('outside_frac_rebuilt')}"
            )
        elif status == "timeout":
            print(f"[TIMEOUT] {p.name}: {row.get('error')}")
        else:
            print(f"[ERROR] {p.name}: {row.get('error')}")
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[SAVED] {summary_csv}")

    print("\n[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
