#!/usr/bin/env python3
"""
validate_orderflow_downside.py — DEFENSIVE / crisis-window validation of order-flow signals.

The mean-return predictive battery (validate_orderflow_predictive.py) returned NOT_BLEND
for every order-flow signal on 2017-2020. But a stress/defensive input does NOT need a
mean-return edge to be valuable — it is useful if it ELEVATES into real crashes AND
predicts forward DOWNSIDE. This follows Pitfall 28 (downside-defensive-validation) and
Section 9 crisis-window validation of the walk-forward skill.

Two tests per signal (taker_imbalance_qty, taker_buy_ratio, whale_qty_lo_count,
total_quote, n_trades), at 7d and 30d horizons:

  A. CRISIS-WINDOW ELEVATION — for each canonical crash (2018 bear bottom, Coincheck
     hack, COVID, Upbit hack, Luna, FTX, Bybit hack), does the signal RISE vs a 180-day
     prior control window? Report elevation in z-units (point-in-time z) and pp. A real
     defensive signal should be elevated in most/all crisis windows, not just ranked.

  B. FORWARD-DOWNSIDE DEFENSIVENESS — Spearman IC(signal_t, forward downside) where
     downside = max-drawdown-magnitude (neg) and worst-single-day-return over the
     window. Defensive polarity = IC < 0 (high signal -> worse forward downside),
     computed full-sample AND era-split (3 blocks) to check the polarity is STABLE
     (a crash-event relationship should not flip sign across eras).

Verdict: DEFENSIVE if the signal (a) elevates in a clear majority of crisis windows
and (b) has a stable defensive IC polarity across eras. Otherwise NOT_DEFENSIVE /
era-dependent — do NOT blend.

Output: analysis/.orderflow_downside_summary.json + human-readable printout.
"""
import json, os, random
import numpy as np
from datetime import datetime, timezone
from scipy import stats

REPO = "/home/ubuntu/sfc"
OF = os.path.join(REPO, "data", "binance_orderflow_daily.json")
SUM_OUT = os.path.join(REPO, "analysis", ".orderflow_downside_summary.json")
SEED = 42

SIGNALS = ["taker_imbalance_qty", "taker_buy_ratio",
           "whale_qty_lo_count", "total_quote", "n_trades"]
HORIZONS = [(7, "7d"), (30, "30d")]
CONTROL_DAYS = 180

# Canonical crashes (same as imbs_l1l2_calibration.py / historical_backtest_m1m6.py)
# plus notable exchange hacks with market impact within the 2017-2026 order-flow window.
CRISIS_WINDOWS = {
    "Coincheck hack (Jan 2018)": ("2018-01-26", "2018-01-30"),
    "2018 Bear Market Bottom": ("2018-11-01", "2018-12-31"),
    "Upbit hack (Nov 2019)": ("2019-11-27", "2019-11-28"),
    "COVID Crash (Mar 2020)": ("2020-03-08", "2020-03-20"),
    "Luna/UST Collapse (May 2022)": ("2022-05-07", "2022-05-16"),
    "FTX Collapse (Nov 2022)": ("2022-11-06", "2022-11-12"),
    "Bybit hack (Feb 2025)": ("2025-02-21", "2025-02-23"),
}


def ptz(sig, window=365):
    """Point-in-time z-score over trailing window (no lookahead)."""
    z = np.full(len(sig), np.nan)
    for i in range(len(sig)):
        if np.isnan(sig[i]):
            continue
        lo = max(0, i - window + 1)
        w = sig[lo:i]
        w = w[~np.isnan(w)]
        if len(w) >= 60:
            mu, sd = float(np.mean(w)), float(np.std(w))
            if sd > 0:
                z[i] = (sig[i] - mu) / sd
    return z


def max_drawdown(ret_sub):
    """Magnitude of max peak-to-trough drawdown over a return path (>=0, in frac)."""
    if len(ret_sub) == 0:
        return np.nan
    nav = np.cumprod(1 + ret_sub)
    peak = np.maximum.accumulate(nav)
    return float(np.max(peak - nav))   # magnitude (positive); deeper drawdown = bigger


def main():
    random.seed(SEED)
    of = json.load(open(OF))
    days = sorted(of)
    n = len(days)
    price = np.array([of[d]["price_close"] for d in days], dtype=float)
    day_idx = {d: i for i, d in enumerate(days)}

    # per-day returns for drawdown/worst-day
    dret = np.full(n, np.nan)
    for i in range(1, n):
        dret[i] = price[i] / price[i - 1] - 1

    # forward return + downside over h days
    fwd = {}
    for h, hl in HORIZONS:
        ret = np.full(n, np.nan)
        mdd = np.full(n, np.nan)      # forward max-drawdown magnitude
        worst = np.full(n, np.nan)    # forward worst single-day return
        for i in range(n - h):
            sub = dret[i + 1:i + h + 1]
            sub = sub[~np.isnan(sub)]
            if len(sub) == 0:
                continue
            ret[i] = price[i + h] / price[i] - 1
            mdd[i] = max_drawdown(sub)
            worst[i] = float(np.min(sub))
        fwd[hl] = {"ret": ret, "mdd": mdd, "worst": worst}

    # era split (3 contiguous)
    q = n // 3
    eras = {"era1": (0, q), "era2": (q, 2 * q), "era3": (2 * q, n)}
    era_labels = {k: f"{days[a]}..{days[b - 1]}" for k, (a, b) in eras.items()}

    out = {"generated_at": datetime.now(timezone.utc).isoformat(),
           "window": [days[0], days[-1]], "n_days": n,
           "crisis_windows": CRISIS_WINDOWS, "result": {}}

    print(f"Order-flow downside/crisis battery | {days[0]} .. {days[-1]} | n={n}")
    print("Eras: " + ", ".join(f"{k}: {v}" for k, v in era_labels.items()))

    for sname in SIGNALS:
        sig = np.array([(of[d][sname] if of[d].get(sname) is not None else np.nan)
                        for d in days], dtype=float)
        z = ptz(sig)

        # ---- A. crisis-window elevation (vs 180-day prior control) ----
        crisis_rows = []
        for cname, (cs, ce) in CRISIS_WINDOWS.items():
            if cs not in day_idx or ce not in day_idx:
                crisis_rows.append({"window": cname, "covered": False})
                continue
            a, b = day_idx[cs], day_idx[ce] + 1
            cz = z[a:b]
            cz = cz[~np.isnan(cz)]
            ctrl_lo = max(0, a - CONTROL_DAYS)
            ctrl = z[ctrl_lo:a]
            ctrl = ctrl[~np.isnan(ctrl)]
            if len(cz) == 0 or len(ctrl) < 30:
                crisis_rows.append({"window": cname, "covered": True,
                                    "n_crisis": int(len(cz)), "n_ctrl": int(len(ctrl))})
                continue
            elev = float(np.mean(cz) - np.mean(ctrl))
            frac_above = float(np.mean(cz > np.percentile(ctrl, 50)))
            crisis_rows.append({"window": cname, "covered": True,
                                "n_crisis": int(len(cz)),
                                "crisis_mean_z": round(float(np.mean(cz)), 2),
                                "ctrl_mean_z": round(float(np.mean(ctrl)), 2),
                                "elevation_z": round(elev, 2),
                                "frac_above_control_median": round(frac_above, 2)})

        # ---- B. forward-downside defensiveness IC, full + era-split ----
        def ic_def(sig_slice, fwd_slice):
            rows = []
            for measure, key in [("max_drawdown", "mdd"), ("worst_day", "worst")]:
                m = ~(np.isnan(sig_slice) | np.isnan(fwd_slice[key]))
                if m.sum() < 40 or np.var(fwd_slice[key][m]) == 0:
                    rows.append({"measure": measure, "ic": None, "n": int(m.sum())})
                    continue
                ic = float(stats.spearmanr(sig_slice[m], fwd_slice[key][m]).statistic)
                rows.append({"measure": measure, "ic": round(ic, 3), "n": int(m.sum())})
            return rows

        res_signal = {}
        for h, hl in HORIZONS:
            f = fwd[hl]
            full = ic_def(sig, f)
            era_res = {}
            for k, (a, b) in eras.items():
                era_res[k] = ic_def(sig[a:b], {kk: vv[a:b] for kk, vv in f.items()})
            res_signal[hl] = {"full": full, "eras": era_res}
        out["result"][sname] = {"crisis": crisis_rows, "downside": res_signal}

        # ---- print ----
        print(f"\n== {sname} ==")
        print("  CRISIS elevation (signal z, crisis vs 180d-prior control):")
        for c in crisis_rows:
            if not c["covered"]:
                print(f"    {c['window']:<28} not in panel")
                continue
            if "elevation_z" in c:
                print(f"    {c['window']:<28} crisis_z={c['crisis_mean_z']:+.2f} "
                      f"ctrl_z={c['ctrl_mean_z']:+.2f} elev={c['elevation_z']:+.2f} "
                      f"frac>ctrl_med={c['frac_above_control_median']:.2f}")
            else:
                print(f"    {c['window']:<28} n_crisis={c.get('n_crisis')} n_ctrl={c.get('n_ctrl')}")
        for h, hl in HORIZONS:
            print(f"  DOWNSIDE @ {hl} (IC < 0 = defensive: high signal -> worse forward downside):")
            f = res_signal[hl]
            for row in f["full"]:
                print(f"    FULL {row['measure']:<11} IC={row['ic']}  (n={row['n']})")
            for k in ["era1", "era2", "era3"]:
                parts = []
                for row in f["eras"].get(k, []):
                    parts.append(f"{row['measure']}={row['ic']}")
                print(f"    {k}  " + "  ".join(parts))

    json.dump(out, open(SUM_OUT, "w"), indent=1)
    print(f"\nSummary -> {SUM_OUT}")


if __name__ == "__main__":
    main()
