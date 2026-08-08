#!/usr/bin/env python3
"""
Objective robustness screen for the A/B/C/D execution_risk variants BEFORE any
production decision. Answers: is variant C's (no-squeeze) 14d significance a real,
stable edge or a multiple-comparison / outlier artefact?

Tests:
  T1. Two-sided bootstrap p-value for every variant x horizon gap.
  T2. Multiple-comparison correction: Bonferroni (0.10/12) + Benjamini-Hochberg q.
  T3. Era-split: first vs second half of window, C vs B at 7d/14d/30d.
  T4. Jackknife (leave-one-day-out) on the C 14d gap — is it one-day-driven?
  T5. Funding-source sensitivity: C gap with Binance-funding-only days vs all days.

Reuses extraction from ab_parallel_exec_risk.py (git snapshots + Binance cache).
"""
import os, sys, json, subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_parallel_exec_risk as ab  # load_snapshots, daily_resample, load_binance_cache, compute_variants, etc.

HORIZONS = [7, 14, 30]
VARIANTS = ["A", "B", "C", "D"]
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "ab_extracted.json")


def extract():
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            return json.load(f)
    snaps = ab.load_snapshots()
    daily = ab.daily_resample(snaps)
    binance = ab.load_binance_cache()
    def merged(s):
        row = dict(s); day = s["ts"][:10]
        if binance and day in binance:
            b = binance[day]
            if "funding_last" in b:
                row["funding"] = min(abs(b["funding_last"]) * 10, 1.0)
            if "close" in b:
                row["btc"] = b["close"]
        return row
    md = [merged(s) for s in daily]
    usable = [s for s in md if None not in (s["L"], s["S"], s["T"], s["density"])]
    out = {"days": [s["ts"][:10] for s in usable],
           "btc": [s["btc"] for s in usable],
           "rows": [{k: s[k] for k in ("L", "S", "T", "density", "funding")} for s in usable]}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(out, f)
    return out


def gap_and_bootstrap_p(exec_series, fwd, nboot=20000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(exec_series)
    q_hi, q_lo = np.quantile(exec_series, [0.75, 0.25])
    hi = exec_series >= q_hi; lo = exec_series <= q_lo
    if not hi.any() or not lo.any():
        return np.nan, 1.0, (np.nan, np.nan)
    obs = fwd[hi].mean() - fwd[lo].mean()
    idx = np.arange(n); gaps = []
    for _ in range(nboot):
        b = rng.choice(idx, n, replace=True)
        eb = exec_series[b]; fb = fwd[b]
        qh, ql = np.quantile(eb, [0.75, 0.25])
        mh = eb >= qh; ml = eb <= ql
        if mh.sum() and ml.sum():
            gaps.append(fb[mh].mean() - fb[ml].mean())
    gaps = np.array(gaps)
    # two-sided bootstrap p-value under H0: gap = 0 (distribution centered on obs)
    centered = gaps - gaps.mean()
    p2 = min(2 * np.mean(centered <= -abs(obs)), 2 * np.mean(centered >= abs(obs)))
    p2 = min(1.0, p2)
    return obs, p2, (np.percentile(gaps, 5), np.percentile(gaps, 95))


def bh_qvalues(pvals):
    p = np.array(pvals); n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    m = p[order][::-1]
    q[order[::-1]] = 1.0
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, p[order[i]] * n / (i + 1))
        q[order[i]] = running
    return q


def main():
    data = extract()
    rows = data["rows"]; btcs = np.array(data["btc"]); days = data["days"]
    variants = ab.compute_variants(rows)
    n = len(btcs)
    print(f"window: {len(days)} daily pts ({days[0]} -> {days[-1]})")

    # ---- T1 + T2: p-values + multiple-comparison ----
    exs = {v: np.array([r[v] for r in variants]) for v in VARIANTS}
    pvals = {}
    print(f"\nT1/T2 — two-sided bootstrap p + BH q (Bonferroni 0.10/12 = {0.10/12:.4f})")
    print(f"{'var':<4}{'horizon':>8}{'gap':>9}{'p':>9}{'q(BH)':>9}  sig@0.10?")
    all_p = []
    for v in VARIANTS:
        for h in HORIZONS:
            if h >= n: continue
            fwd = np.array([btcs[i+h]/btcs[i]-1.0 for i in range(n-h)])
            ex = exs[v][:len(fwd)]
            gap, p2, ci = gap_and_bootstrap_p(ex, fwd)
            all_p.append((v, h, gap, p2))
    qmap = {}
    qs = bh_qvalues([x[3] for x in all_p])
    for (v, h, gap, p2), q in zip(all_p, qs):
        qmap[(v, h)] = q
        bonf = p2 < 0.10/12
        sig = "***" if q < 0.10 else ("*" if p2 < 0.10 else "")
        print(f"{v:<4}{h:>8}{gap:>9.4f}{p2:>9.4f}{q:>9.4f}  {sig}")

    # ---- T3: era-split ----
    print(f"\nT3 — era-split (half1 vs half2) for C vs B, 14d gap")
    half = n // 2
    for lbl, lo_i in [("half1(awal)", 0), ("half2(akhir)", half)]:
        idx = slice(lo_i, lo_i + half)
        if half - 14 <= 0: 
            print(f"  {lbl}: too short"); continue
        fwd = np.array([btcs[i+14]/btcs[i]-1.0 for i in range(lo_i, lo_i+half-14)])
        for v in ["B", "C"]:
            ex = exs[v][idx][:len(fwd)]
            gap, p2, _ = gap_and_bootstrap_p(ex, fwd)
            print(f"  {lbl} {v}: 14d gap={gap:+.4f} p={p2:.3f}")

    # ---- T4: jackknife on C 14d ----
    print(f"\nT4 — jackknife (leave-one-day-out) C 14d gap")
    fwd14 = np.array([btcs[i+14]/btcs[i]-1.0 for i in range(n-14)])
    ex14 = exs["C"][:len(fwd14)]
    obs, _, _ = gap_and_bootstrap_p(ex14, fwd14)
    lo_jk = []
    for drop in range(len(ex14)):
        m = np.ones(len(ex14), bool); m[drop] = False
        g, _, _ = gap_and_bootstrap_p(ex14[m], fwd14[m], nboot=5000)
        lo_jk.append(g)
    lo_jk = np.array(lo_jk)
    print(f"  full gap={obs:+.4f} | jackknife min={lo_jk.min():+.4f} max={lo_jk.max():+.4f} "
          f"| max|drop single-day| impact={max(abs(lo_jk-obs)):.4f}")
    # how many drops flip sign or lose significance
    print(f"  n drops gap>=0: {np.sum(lo_jk>=0)}/{len(lo_jk)}  "
          f"(0 = stable direction)")

    # ---- T5: funding-source sensitivity for C ----
    print(f"\nT5 — funding-source sensitivity (C, 14d)")
    binance_days = set(ab.load_binance_cache().keys())
    use_bin = np.array([days[i] in binance_days for i in range(len(days))])
    idx = np.arange(n)
    # only days with binance funding
    sel = idx[use_bin]
    if len(sel) - 14 > 0:
        rows_sel = [variants[i] for i in sel]
        btcs_sel = np.array([btcs[i] for i in sel])
        exC = np.array([r["C"] for r in rows_sel])
        fwd = np.array([btcs_sel[i+14]/btcs_sel[i]-1.0 for i in range(len(btcs_sel)-14)])
        gap, p2, _ = gap_and_bootstrap_p(exC[:len(fwd)], fwd)
        print(f"  Binance-funding-only days (n={len(sel)}): C 14d gap={gap:+.4f} p={p2:.3f}")
    else:
        print("  binance-only days too few")


if __name__ == "__main__":
    main()
