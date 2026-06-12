import re
import sys
import subprocess
from pathlib import Path

RESULTS = Path("results")
TARGET = Path("results/targets/star_branch_A2p03_k2p16_d0p47_target.json")
DXF = Path("cad/star_baseline.dxf")

# Último seed bueno: 9.40 MA
seed = Path(
    r"results/highip_branch_search_20260611_014645/attempts/attempt_0003_seed_scaled_best_x1p025/best.json"
)

# No repitas 9.40; ya lo tienes.
# ip_stages = [9.50, 9.60, 9.75, 9.90, 10.10, 10.30, 10.50, 10.80, 11.10]
ip_stages = [11.30, 11.50, 11.70, 11.90, 12.10, 12.30, 12.50, 12.70, 12.90, 13.10, 13.20]

ACCEPT_RE = re.compile(
    r"\[ACCEPTED\]\s+attempt=(?P<attempt>\d+)\s+"
    r"score=(?P<score>[-+0-9.eE]+)\s+"
    r"gap=(?P<gap>[-+0-9.eE]+)\s+"
    r"A=(?P<A>[-+0-9.eE]+)\s+"
    r"k=(?P<kappa>[-+0-9.eE]+)\s+"
    r"Rax=(?P<Rax>[-+0-9.eE]+)\s+"
    r"area=(?P<area>[-+0-9.eE]+)"
)

OUTDIR_RE = re.compile(r"outdir\s+=\s+(?P<outdir>.+)$")


def parse_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def run_and_capture(cmd):
    accepted = []
    outdir = None

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None

    for line in proc.stdout:
        print(line, end="")

        m_out = OUTDIR_RE.search(line.strip())
        if m_out:
            outdir = Path(m_out.group("outdir").strip())

        m = ACCEPT_RE.search(line.strip())
        if m:
            d = m.groupdict()
            accepted.append(
                {
                    "attempt": int(d["attempt"]),
                    "score": parse_float(d["score"]),
                    "gap": parse_float(d["gap"]),
                    "A": parse_float(d["A"]),
                    "kappa": parse_float(d["kappa"]),
                    "Rax": parse_float(d["Rax"]),
                    "area": parse_float(d["area"]),
                }
            )

    ret = proc.wait()
    return ret, outdir, accepted


def valid_candidate(c):
    return (
        c["gap"] is not None
        and c["A"] is not None
        and c["kappa"] is not None
        and c["Rax"] is not None
        and c["area"] is not None
        and c["gap"] >= 0.005
        and 20.0 <= c["area"] <= 55.0
        and 3.8 <= c["Rax"] <= 5.6
        and 1.20 <= c["A"] <= 3.00
        and 1.35 <= c["kappa"] <= 3.00
    )


def candidate_quality(c):
    # Prioriza gap, pero penaliza alejarse demasiado de una rama tipo STAR.
    gap_term = 100.0 * c["gap"]
    shape_penalty = (
        1.0 * abs(c["A"] - 2.0)
        + 0.4 * abs(c["kappa"] - 2.1)
        + 0.03 * abs(c["area"] - 26.0)
        + 0.2 * abs(c["Rax"] - 4.2)
    )
    return gap_term - shape_penalty


def attempt_best_json(outdir, attempt_number):
    attempts_dir = outdir / "attempts"
    pattern = f"attempt_{attempt_number:04d}_*"
    matches = sorted(attempts_dir.glob(pattern))

    if not matches:
        return None

    best = matches[0] / "best.json"
    if best.exists():
        return best

    return None


for ip in ip_stages:
    print("\n" + "=" * 100)
    print(f"[AUTO] Searching Ip={ip:.2f} MA from seed:")
    print(f"       {seed}")
    print("=" * 100)

    if not seed.exists():
        print(f"[STOP] Seed does not exist: {seed}")
        break

    cmd = [
        sys.executable,
        "-u",
        "search_highip_branch_star.py",
        "--target",
        str(TARGET),
        "--dxf",
        str(DXF),
        "--Ip-MA",
        f"{ip:.2f}",
        "--paxis-Pa",
        "2.0e3",
        "--base-seeds",
        str(seed),
        "--scale-factors",
        "1.00",
        "1.025",
        "1.05",
        "1.075",
        "1.10",
        "1.125",
        "1.15",
        "1.20",
        "1.25",
        "1.30",
        "--max-seeds",
        "18",
        "--random-per-base",
        "1",
        "--seed-sigma-frac",
        "0.06",
        "--workers",
        "10",
        "--pop",
        "32",
        "--iters",
        "6",
        "--eval-timeout",
        "150",
        "--command-timeout",
        "4200",
        "--widen",
        "0.08", # 0.075
        "--sigma-frac",
        "0.08", # 0.075
        "--cs-abs-limit-ma",
        "140", # 130
        "--pf-abs-limit-ma",
        "48", # 45
        "--wall-gap-target-m",
        "0.025",
        "--shape-weight-scale",
        "0.012",
        "--boundary-weight-scale",
        "0.008",
        "--xpoint-weight-scale",
        "0.006",
        "--sym-weight-scale",
        "0.014",
        "--current-reg-scale",
        "0.030",
        "--cs-reg-scale",
        "0.015",
        "--step-reg-scale",
        "0.012",
        "--min-gap-m",
        "0.005",
        "--max-outside-frac",
        "0.0",
        "--axis-R-min",
        "3.8",
        "--axis-R-max",
        "5.6",
        "--axis-Z-max",
        "0.9",
        "--area-min",
        "20",
        "--area-max",
        "55",
        "--A-min",
        "1.20",
        "--kappa-min",
        "1.35",
        "--branch-term-weight",
        "100000000",
        "--dshape-weight",
        "0.002",
        "--inboard-weight",
        "0.2",
        "--target-rms-weight",
        "0",
        "--delta-weight",
        "0.02",
        "--stop-after-accepted",
        "1", # 2
    ]

    ret, outdir, accepted = run_and_capture(cmd)

    if ret != 0:
        print(f"[STOP] search_highip_branch_star.py failed at Ip={ip:.2f} MA")
        break

    if outdir is None:
        print(f"[STOP] Could not parse output directory at Ip={ip:.2f} MA")
        break

    valid = [c for c in accepted if valid_candidate(c)]

    if not valid:
        print(f"[STOP] No valid large-branch candidate found at Ip={ip:.2f} MA")
        print(f"[INFO] outdir = {outdir}")
        print("[INFO] accepted candidates reported by search:")
        for c in accepted:
            print("  ", c)
        break

    valid.sort(key=candidate_quality, reverse=True)
    chosen = valid[0]

    best_json = attempt_best_json(outdir, chosen["attempt"])
    if best_json is None:
        print(f"[STOP] Could not locate best.json for attempt {chosen['attempt']:04d}")
        print(f"[INFO] outdir = {outdir}")
        break

    print("\n" + "-" * 100)
    print(f"[ACCEPTED LARGE BRANCH] Ip={ip:.2f} MA")
    print(f"  chosen attempt = {chosen['attempt']:04d}")
    print(f"  best.json      = {best_json}")
    print(f"  gap            = {chosen['gap']}")
    print(f"  Rax            = {chosen['Rax']}")
    print(f"  A              = {chosen['A']}")
    print(f"  kappa          = {chosen['kappa']}")
    print(f"  area           = {chosen['area']}")
    print("-" * 100)

    seed = best_json

print("\n[DONE] Last usable seed:")
print(seed)
