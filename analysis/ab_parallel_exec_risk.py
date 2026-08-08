#!/usr/bin/env python3
"""
A/B/C/D parallel walk-forward comparison of execution_risk formula variants.

Goal: determine which candidate execution_risk formula produces better
OUT-OF-SAMPLE predictive information, using only historical SFC snapshots
(no production code change).

Variants (weights normalised where noted; all capped at 0.95):
  A  OLD double-count : cascade=imbalance*0.5 + total/5e9, squeeze=imbalance*density
                       exec = .40*cascade + .30*squeeze + .30*funding
  B  de-dup orthogonal: cascade=imbalance (direction), squeeze=density (magnitude)
                       exec = .40*cascade + .30*squeeze + .30*funding
  C  no squeeze       : exec = .40*cascade + .30*funding   (cascade=direction)
  D  funding-heavy    : exec = .30*cascade + .20*squeeze + .50*funding

Inputs (all from historical data.json snapshots in git):
  liq_long_vol, liq_short_vol, liq_total_24h  -> imbalance=|L-S|/T
  liq_density                                 -> magnitude
  funding_imbalance (real Deribit m13)        -> funding (0 when unavailable)

Out-of-sample metric per variant:
  - quantile tail gap: mean(forward return | top 25% exec) - mean( | bottom 25%)
    (polarity: high execution risk should precede LOWER forward return => negative)
  - Spearman IC: rank correlation of exec_risk with forward return
  - std of exec_risk series (amplitude sanity)
  Bootstrap CI 90% (seeded), daily-resampled to kill cron autocorrelation.

Verdict rule (per walk-forward skill): with ~17 real-funding days all CIs will
include zero -> report as EVIDENCE / DATA-TOO-SHORT, not a deploy decision.
"""
import subprocess, json, sys, os
import numpy as np

REPO = "/home/ubuntu/sfc"
HORIZONS = [7, 14, 30]

# ---- 1. Load every data.json snapshot from git ----
def load_snapshots():
    hashes = subprocess.run(
        ["git", "-C", REPO, "log", "--format=%H", "--", "data.json"],
        capture_output=True, text=True).stdout.split()
    snaps = []
    for c in hashes:
        try:
            raw = subprocess.run(["git", "-C", REPO, "show", f"{c}:data.json"],
                                 capture_output=True, text=True).stdout
            d = json.loads(raw)
        except Exception:
            continue
        ts = d.get("ts")
        btc = d.get("btc")
        if ts is None or btc is None:
            continue
        cc = d.get("confidence_components") or {}
        snaps.append({
            "ts": ts, "btc": float(btc),
            "L": d.get("liq_long_vol"), "S": d.get("liq_short_vol"),
            "T": d.get("liq_total_24h"), "density": d.get("liq_density"),
            "funding": cc.get("funding_imbalance"),
        })
    return snaps

def daily_resample(snaps):
    # keep last snapshot per UTC date (kill 5-min cron autocorrelation)
    byday = {}
    for s in snaps:
        day = s["ts"][:10]
        byday[day] = s  # last wins (chronological order preserved)
    return sorted(byday.values(), key=lambda s: s["ts"])

# ---- 2. Variant formulas ----
def compute_variants(rows):
    """rows: list of dicts with L,S,T,density,funding. Returns per-row dict of exec_risk per variant."""
    out = []
    for r in rows:
        L, S, T = r["L"], r["S"], r["T"]
        den = r["density"] if r["density"] is not None else 0.0
        fund = r["funding"] if r["funding"] is not None else 0.0
        # imbalance = one-sidedness |L-S|/T (0=balanced,1=one-sided)
        imbalance = 0.0
        if L is not None and S is not None and T is not None and T > 0:
            imbalance = abs(L - S) / float(T)
        # A: old double-count
        cascade_A = min(imbalance * 0.5 + (float(T) / 5e9 if T else 0.0), 0.95)
        squeeze_A = imbalance * den
        exec_A = min(0.40 * cascade_A + 0.30 * squeeze_A + 0.30 * fund, 0.95)
        # B: de-dup orthogonal
        cascade_B = min(imbalance, 0.95)
        squeeze_B = den
        exec_B = min(0.40 * cascade_B + 0.30 * squeeze_B + 0.30 * fund, 0.95)
        # C: no squeeze
        exec_C = min(0.40 * cascade_B + 0.30 * fund, 0.95)
        # D: funding-heavy
        exec_D = min(0.30 * cascade_B + 0.20 * squeeze_B + 0.50 * fund, 0.95)
        out.append({
            "A": exec_A, "B": exec_B, "C": exec_C, "D": exec_D,
            "imbalance": imbalance, "density": den, "funding": fund,
        })
    return out

# ---- 3. OOS metric with bootstrap CI ----
def bootstrap_ci_gap(exec_series, fwd, nboot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(exec_series)
    q_hi, q_lo = np.quantile(exec_series, [0.75, 0.25])
    hi = exec_series >= q_hi; lo = exec_series <= q_lo
    obs = fwd[hi].mean() - fwd[lo].mean()
    diffs = []
    idx = np.arange(n)
    for _ in range(nboot):
        bi = rng.choice(idx, n, replace=True)
        eb = exec_series[bi]; fb = fwd[bi]
        qh, ql = np.quantile(eb, [0.75, 0.25])
        mh = eb >= qh; ml = eb <= ql
        if mh.sum() and ml.sum():
            diffs.append(fb[mh].mean() - fb[ml].mean())
    diffs = np.array(diffs)
    return obs, np.percentile(diffs, 5), np.percentile(diffs, 95)

def spearman_ic(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    xm = rx.mean(); ym = ry.mean()
    num = ((rx-xm)*(ry-ym)).sum()
    den = np.sqrt(((rx-xm)**2).sum()*((ry-ym)**2).sum())
    return num/den if den else 0.0

def load_binance_cache():
    p = os.path.join(REPO, "data", "binance_vision_daily.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    snaps = load_snapshots()
    daily = daily_resample(snaps)
    print(f"snapshots={len(snaps)}  daily_points={len(daily)} "
          f"({daily[0]['ts'][:10]} -> {daily[-1]['ts'][:10]})")

    binance = load_binance_cache()
    if binance:
        print(f"binance cache loaded: {len(binance)} days "
              f"({min(binance)} -> {max(binance)})")
    else:
        print("WARNING: no binance cache -> funding fallback SFC only (short window).")

    def merged_row(s):
        """Attach Binance funding+close to an SFC daily snapshot (fallback SFC)."""
        day = s["ts"][:10]
        row = dict(s)
        if binance and day in binance:
            b = binance[day]
            if "funding_last" in b:
                # same convention as SFC: funding_imbalance = min(|fr|*10, 1)
                row["funding"] = min(abs(b["funding_last"]) * 10, 1.0)
            if "close" in b:
                row["btc"] = b["close"]  # forward returns from Binance close
        # if binance missing for this day, keep SFC funding_imbalance (real, if present)
        return row

    merged_daily = [merged_row(s) for s in daily]
    # usable window = daily points with liquidation L/S/T + density present
    usable = [s for s in merged_daily if None not in (s["L"], s["S"], s["T"], s["density"])]
    print(f"daily rows with liquidation L/S/T+density: {len(usable)} "
          f"({usable[0]['ts'][:10]} -> {usable[-1]['ts'][:10]})")
    # how many have real (nonzero) funding
    n_fund_nz = sum(1 for s in usable if s["funding"])
    print(f"  of which with funding != 0: {n_fund_nz}")

    def report(rows, btcs, label):
        n = len(btcs)
        horizons = [h for h in HORIZONS if h < n]
        print(f"\n=== {label}: {n} daily points ===")
        print(f"{'var':<4}{'std':>8}   " + "  ".join(f"{'fwd'+str(h)+'d':>30}" for h in horizons))
        for v in ["A", "B", "C", "D"]:
            ex = np.array([r[v] for r in rows])
            line = f"{v:<4}{ex.std():>8.4f}   "
            for h in horizons:
                fwd = np.array([btcs[i+h]/btcs[i]-1.0 for i in range(n-h)])
                em = ex[:len(fwd)]
                gap, lo, hi = bootstrap_ci_gap(em, fwd)
                ic = spearman_ic(em, fwd)
                sig = "***" if (lo < 0 and hi < 0) else ("+++" if (lo > 0 and hi > 0) else "")
                line += f"gap={gap:+.4f}[{lo:+.3f},{hi:+.3f}]{sig} IC={ic:+.3f}   "
            print(line)

    # PRIMARY — four-way A/B/C/D on the EXTENDED liquidation window
    rows = compute_variants(usable)
    btcs = np.array([s["btc"] for s in usable])
    report(rows, btcs, "FOUR-WAY A/B/C/D on extended liquidation window (Binance funding)")

    # A/B de-dup robustness on the same extended window
    print("\n=== A vs B de-dup robustness (extended window) ===")
    for v in ["A", "B"]:
        ex = np.array([r[v] for r in rows])
        detail = f"  {v}: std={ex.std():.4f}"
        for h in [7, 14, 30]:
            if h >= len(btcs): continue
            fwd = np.array([btcs[i+h]/btcs[i]-1.0 for i in range(len(btcs)-h)])
            em = ex[:len(fwd)]
            gap, lo, hi = bootstrap_ci_gap(em, fwd)
            ic = spearman_ic(em, fwd)
            sig = "***" if (lo < 0 and hi < 0) else ("+++" if (lo > 0 and hi > 0) else "")
            detail += f" | {h}d gap={gap:+.4f}[{lo:+.2f},{hi:+.2f}]{sig} IC={ic:+.3f}"
        print(detail)
    sa = np.array([r["A"] for r in rows]).std(); sb = np.array([r["B"] for r in rows]).std()
    print(f"std ratio A/B = {sa/sb:.3f}")

    exs = {v: np.array([r[v] for r in rows]) for v in ["A","B","C","D"]}
    print("\nPearson corr among variants (extended window):")
    print("        " + "  ".join(f"{v:>6}" for v in "ABCD"))
    for v1 in "ABCD":
        print(f"{v1:>6}   " + "  ".join(f"{np.corrcoef(exs[v1], exs[v2])[0,1]:>6.3f}" for v2 in "ABCD"))

    print("\nLegend: gap = mean(fwd | top25% exec) - mean(fwd | bottom25% exec).")
    print("Polarity benar = gap negatif. *** = 90% CI excl 0. Verdict = data-too-short jika CI lebar.")

if __name__ == "__main__":
    main()
