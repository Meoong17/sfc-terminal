#!/usr/bin/env python3
"""
imbs_l1l2_calibration.py — Threshold calibration + crisis-window validation
for the IMBS Layer 1-2 liquidity-augmented signal.

Reads the cached time series produced by walk_forward_imbs_l1l2.py
(.walk_forward_imbs_l1l2.json) and runs two additional validations:

1) THRESHOLD CALIBRATION
   The existing pipeline buckets CALM<25 / ELEVATED 25-45 / STRESS>=45.
   Adding the liquidity index shifted many days into STRESS (n went from
   1208 -> 1757), so the fixed 45 cutoff may now be over-defensive. This
   script scans candidate STRESS cutoffs and reports which cutoff maximizes
   the CALM-vs-STRESS forward-return gap while keeping enough observations
   in each bucket. It does NOT silently change the live thresholds — it
   produces evidence for a human to act on.

2) CRISIS-WINDOW VALIDATION
   Checks that the IMBS signal actually ELEVATES (not just statistically
   separates) during the well-documented BTC crash windows, mirroring
   historical_backtest_m1m6.py's crisis sanity check. For each window it
   reports:
     - mean sfc_pct inside the window vs the overall / a 6-month control
     - % of days classified STRESS (>=45)
     - mean elevation over the surrounding 30-day baseline

USAGE:
    cd ~/sfc
    .venv/bin/python analysis/imbs_l1l2_calibration.py
    # or, to also force re-run the upstream walk-forward first:
    .venv/bin/python analysis/imbs_l1l2_calibration.py --recompute
"""
import json
import os
import random
import sys
from datetime import datetime, timedelta

SFC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERIES_FILE = os.path.join(SFC_ROOT, ".walk_forward_imbs_l1l2.json")
OUTPUT_FILE = os.path.join(SFC_ROOT, ".imbs_l1l2_calibration.json")

HORIZON = 30  # primary calibration horizon (30d gap is the strongest)
N_BOOTSTRAP = 2000

# Well-documented BTC crashes (same as historical_backtest_m1m6.py).
CRISIS_WINDOWS = {
    "2018 Bear Market Bottom": ("2018-11-01", "2018-12-31"),
    "COVID Crash (Mar 2020)": ("2020-03-08", "2020-03-20"),
    "Luna/UST Collapse (May 2022)": ("2022-05-07", "2022-05-16"),
    "FTX Collapse (Nov 2022)": ("2022-11-06", "2022-11-12"),
}

# Candidate STRESS cutoffs to scan (CALM stays <25, ELEVATED 25..cutoff).
CANDIDATE_CUTOFFS = [35, 40, 45, 50, 55]


def load_series():
    if not os.path.exists(SERIES_FILE):
        print(f"Series file not found: {SERIES_FILE}\n"
              "Run walk_forward_imbs_l1l2.py first (or --recompute).",
              file=sys.stderr)
        sys.exit(1)
    with open(SERIES_FILE) as f:
        return json.load(f)


def bucket_label(sfc_pct, stress_cutoff):
    if sfc_pct is None:
        return None
    if sfc_pct < 25:
        return "CALM"
    if sfc_pct < stress_cutoff:
        return "ELEVATED"
    return "STRESS"


def bootstrap_diff_ci(a, b, ci=0.90):
    if len(a) < 2 or len(b) < 2:
        return None, None, None
    na, nb = len(a), len(b)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        sa = [a[random.randrange(na)] for _ in range(na)]
        sb = [b[random.randrange(nb)] for _ in range(nb)]
        diffs.append(sum(sb) / nb - sum(sa) / na)
    diffs.sort()
    lo = int((1 - ci) / 2 * N_BOOTSTRAP)
    hi = int((1 + ci) / 2 * N_BOOTSTRAP) - 1
    return sum(b) / nb - sum(a) / na, diffs[lo], diffs[hi]


def run_threshold_calibration(series):
    print("=" * 70)
    print("1) THRESHOLD CALIBRATION  (horizon = 30d forward return)")
    print("=" * 70)
    fk = f"fwd_return_{HORIZON}d"
    print(f"\n  {'Cutoff':<7} {'n_CALM':<7} {'n_ELEV':<7} {'n_STRESS':<8} "
          f"{'gap(CALM-STRESS)':<20}  {'significant'}")
    print(f"  {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*20}  {'-'*11}")

    rows = []
    for cutoff in CANDIDATE_CUTOFFS:
        calm, elev, stress = [], [], []
        for p in series:
            fwd = p.get(fk)
            if fwd is None:
                continue
            lbl = bucket_label(p.get("sfc_pct_imbs"), cutoff)
            if lbl == "CALM":
                calm.append(fwd)
            elif lbl == "ELEVATED":
                elev.append(fwd)
            elif lbl == "STRESS":
                stress.append(fwd)
        est, lo_, hi_ = bootstrap_diff_ci(calm, stress)
        sig = hi_ < 0 if hi_ is not None else False
        rows.append({
            "cutoff": cutoff, "n_calm": len(calm), "n_elevated": len(elev),
            "n_stress": len(stress),
            "gap_pp": round(est, 2) if est is not None else None,
            "ci_lo": round(lo_, 2) if lo_ is not None else None,
            "ci_hi": round(hi_, 2) if hi_ is not None else None,
            "significant": sig,
        })
        print(f"  {cutoff:<7} {len(calm):<7} {len(elev):<7} {len(stress):<8} "
              f"{est:+.2f}pp [{lo_:+.2f}, {hi_:+.2f}]  "
              f"{'YES' if sig else 'NO'}")

    # Pick the best cutoff that keeps STRESS observations >= some floor.
    MIN_STRESS = 300  # avoid picking a cutoff with a tiny, unreliable bucket
    valid = [r for r in rows if r["significant"] and r["n_stress"] >= MIN_STRESS]
    if valid:
        best = min(valid, key=lambda r: r["gap_pp"])  # most negative gap = best
        print(f"\n  RECOMMENDED STRESS CUTOFF: {best['cutoff']} "
              f"(gap {best['gap_pp']}pp, n_STRESS={best['n_stress']})")
    else:
        best = None
        print("\n  No cutoff met the significance + minimum-n bar.")
    return rows, best


def run_crisis_validation(series):
    print("\n" + "=" * 70)
    print("2) CRISIS-WINDOW VALIDATION (IMBS L1-L2 signal)")
    print("=" * 70)
    results = {}
    for name, (start, end) in CRISIS_WINDOWS.items():
        # window values
        win = [p for p in series if start <= p["date"] <= end
               and p.get("sfc_pct_imbs") is not None]
        if not win:
            print(f"\n  {name}: NO DATA in window")
            results[name] = {"error": "no data"}
            continue

        # control: 6-month prior window (same length as crash window)
        s = datetime.strptime(start, "%Y-%m-%d")
        ctrl_start = (s - timedelta(days=180)).strftime("%Y-%m-%d")
        ctrl = [p for p in series if ctrl_start <= p["date"] < start
                and p.get("sfc_pct_imbs") is not None]

        win_mean = sum(p["sfc_pct_imbs"] for p in win) / len(win)
        win_stress = sum(1 for p in win if p["sfc_pct_imbs"] >= 45)
        win_stress_pct = win_stress / len(win) * 100
        ctrl_mean = (sum(p["sfc_pct_imbs"] for p in ctrl) / len(ctrl)
                     if ctrl else None)
        elev = win_mean - ctrl_mean if ctrl_mean is not None else None
        overall_mean = sum(p["sfc_pct_imbs"] for p in series) / len(series)

        results[name] = {
            "window_mean": round(win_mean, 1),
            "overall_mean": round(overall_mean, 1),
            "control_6m_mean": round(ctrl_mean, 1) if ctrl_mean is not None else None,
            "elevation_vs_control_pp": round(elev, 1) if elev is not None else None,
            "n_days": len(win),
            "n_stress": win_stress,
            "stress_pct": round(win_stress_pct, 1),
            "elevated": bool(win_mean > (ctrl_mean or overall_mean)),
        }
        ctrl_s = f"control {ctrl_mean:.1f}" if ctrl_mean is not None else "no control"
        print(f"\n  {name} ({start}..{end}):")
        print(f"    window mean sfc_pct : {win_mean:.1f}  (overall {overall_mean:.1f}, "
              f"{ctrl_s})")
        if elev is not None:
            print(f"    elevation vs control : {elev:+.1f}pp")
        print(f"    STRESS days (>=45)   : {win_stress}/{len(win)} "
              f"({win_stress_pct:.0f}%)")
    return results


def main():
    random.seed(42)
    if "--recompute" in sys.argv:
        print("Re-running upstream walk-forward first ...")
        import subprocess
        env = dict(os.environ)
        # load .env so FRED_API_KEY is present for the upstream run
        envfile = os.path.join(SFC_ROOT, ".env")
        if os.path.exists(envfile):
            with open(envfile) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        env.setdefault(k.strip(), v.strip().strip("\"'"))
        subprocess.run([sys.executable,
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "walk_forward_imbs_l1l2.py")],
                       env=env, check=True)

    series = load_series()
    print(f"Loaded {len(series)} observations "
          f"({series[0]['date']} .. {series[-1]['date']})")

    calib_rows, best = run_threshold_calibration(series)
    crisis_results = run_crisis_validation(series)

    summary = {
        "series": SERIES_FILE,
        "threshold_calibration": {"horizon_days": HORIZON, "rows": calib_rows,
                                  "recommended_cutoff": best["cutoff"] if best else None},
        "crisis_windows": crisis_results,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
