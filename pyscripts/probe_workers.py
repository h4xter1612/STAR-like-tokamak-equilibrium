import os
import json
import time
import subprocess
from pathlib import Path

# Recomendado: evitar oversubscription de BLAS/OMP por proceso
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
SCAN = HERE / "scan_star.py"
OUTDIR = HERE / "results_probe"

def parse_jsonl(jsonl_path: Path):
    n = 0
    ok = 0
    timeouts = 0
    solve_fail = 0
    elapsed_sum = 0.0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            if r.get("ok_solve", False):
                ok += 1
            else:
                solve_fail += 1
                err = str(r.get("error", ""))
                if "TimeoutError" in err or (r.get("shape", {}).get("reason") == "timeout"):
                    timeouts += 1
            elapsed_sum += float(r.get("elapsed_s", 0.0) or 0.0)
    return {
        "n": n,
        "ok": ok,
        "solve_fail": solve_fail,
        "timeouts": timeouts,
        "mean_elapsed_s": (elapsed_sum / max(1, n)),
    }

def run_one(workers: int, n1: int = 24, n2: int = 0, timeout: int = 120):
    # limpia outdir
    OUTDIR.mkdir(parents=True, exist_ok=True)
    jsonl = OUTDIR / "scan_multigoal_results.jsonl"
    if jsonl.exists():
        jsonl.unlink()

    cmd = [
        "py", str(SCAN),
        "--outdir", str(OUTDIR),
        "--workers", str(workers),
        "--n1", str(n1),
        "--n2", str(n2),
        "--timeout", str(timeout),
        "--infer_x",
        "--null", "lower",
        "--prefer_inner",
    ]

    t0 = time.perf_counter()
    subprocess.run(cmd, check=False)
    wall = time.perf_counter() - t0

    stats = parse_jsonl(jsonl) if jsonl.exists() else {"n": 0, "ok": 0, "solve_fail": 0, "timeouts": 0, "mean_elapsed_s": 0.0}
    stats["wall_s"] = wall
    stats["throughput_cases_per_s"] = stats["n"] / max(1e-9, wall)
    stats["ok_rate"] = stats["ok"] / max(1, stats["n"])
    stats["timeout_rate"] = stats["timeouts"] / max(1, stats["n"])
    return stats

def main():
    # Prueba típica para 6C/12T
    candidates = [4, 6, 8, 10, 12]
    print("workers | cases/s | ok_rate | timeout_rate | mean_elapsed(s) | wall(s)")
    print("-"*74)
    for w in candidates:
        s = run_one(w)
        print(f"{w:7d} | {s['throughput_cases_per_s']:6.3f} | {s['ok_rate']:7.3f} | {s['timeout_rate']:11.3f} |"
              f" {s['mean_elapsed_s']:14.2f} | {s['wall_s']:6.1f}")

if __name__ == "__main__":
    main()

