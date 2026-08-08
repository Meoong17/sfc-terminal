"""
Walk-forward test: does historical liquidation intensity / one-sidedness predict
adverse BTC moves?

DATA CONSTRAINT (honest): the OKX free endpoint (/api/v5/public/liquidation-orders)
returns only the most recent ~100 orders — NO historical API. So the only real
liquidation history available is what SFC stored in git snapshots of data.json,
which spans ~58 days (2026-06-10 .. 2026-08-09). That is a SHORT, single-regime
window: no era split, overlapping forward labels. Treat any result as
hypothesis-generating, NOT a deploy verdict (walk-forward-validation skill pitfall 22).

Question tested:
  Q1. Does high liquidation INTENSITY (daily mean liq_total) predict worse forward
      returns (1d/3d)?
  Q2. Does high ONE-SIDEDNESS (imbalance=|long_ratio-0.5|*2, the cascade direction
      component) predict worse forward returns?
  Q3. Interaction: does one-sidedness only matter when volume is ALSO high
      (justifying the magnitude-gating in the cascade_risk fix)?

Method: daily resample (kill 5-min cron autocorrelation), threshold-free quantile
tail comparison (top-25% vs bottom-25%), numpy-vectorized bootstrap of the direct
difference, 90% CI. Report n per tail + era/date range (confound).
"""
import json, subprocess, datetime
import numpy as np

RNG = np.random.default_rng(42)

def load_liquidation_daily():
    out = subprocess.run(['git','log','--format=%h %ci','--','data.json'],
                         capture_output=True, text=True).stdout.strip().splitlines()
    # day -> list of (total, ratio)
    per_day = {}
    for line in out:
        sha = line.split()[0]
        ci = line.split(None, 1)[1]
        try:
            d = json.loads(subprocess.run(['git','show',f'{sha}:data.json'],
                                          capture_output=True,text=True).stdout)
        except Exception:
            continue
        long_ = d.get('liq_long_vol'); short_ = d.get('liq_short_vol')
        total = d.get('liq_total_24h')
        if long_ is None or short_ is None or total is None or total <= 0:
            continue
        ratio = long_ / (long_ + short_) if (long_ + short_) > 0 else 0.5
        day = ci[:10]
        per_day.setdefault(day, []).append((float(total), ratio))
    rows = []
    for day in sorted(per_day):
        pairs = per_day[day]
        total = float(np.mean([p[0] for p in pairs]))
        ratio = float(np.mean([p[1] for p in pairs]))
        rows.append((day, total, abs(ratio-0.5)*2))
    return rows

def load_price_closes():
    d = json.load(open('data/binance_vision_daily.json'))
    return {k: float(v['close']) for k, v in d.items()}

def bootstrap_diff_gap(bottom, top, n_boot=20000):
    """Direct bootstrap of mean(top) - mean(bottom). CI excludes 0 => significant."""
    b = np.asarray(bottom); t = np.asarray(top)
    if len(b) < 3 or len(t) < 3:
        return None
    nb, nt = len(b), len(t)
    idx_b = RNG.integers(0, nb, (n_boot, nb))
    idx_t = RNG.integers(0, nt, (n_boot, nt))
    diff = t[idx_t].mean(axis=1) - b[idx_b].mean(axis=1)
    est = t.mean() - b.mean()
    lo, hi = np.percentile(diff, 5), np.percentile(diff, 95)
    return est, lo, hi

def run():
    liq = load_liquidation_daily()
    price = load_price_closes()
    print(f"liquidation days loaded: {len(liq)}  ({liq[0][0]} .. {liq[-1][0]})")
    print(f"price closes available: {len(price)} (.. {sorted(price)[-1]})")

    # total and imbalance series
    totals = np.array([r[1] for r in liq])
    imb = np.array([r[2] for r in liq])

    for h in (1, 3):
        rows = []
        for day, total, imbalance in liq:
            if day not in price:      # no same-day price -> drop
                continue
            # next close h days later
            nd = datetime.date.fromisoformat(day) + datetime.timedelta(days=h)
            key = nd.isoformat()
            if key not in price:
                continue
            fwd = price[key] / price[day] - 1.0
            rows.append((day, total, imbalance, fwd))
        if len(rows) < 20:
            print(f"\n[h={h}d] too few aligned rows ({len(rows)}) — skip")
            continue
        totals_r = np.array([r[1] for r in rows])
        imb_r = np.array([r[2] for r in rows])
        fwd = np.array([r[3] for r in rows])
        n = len(rows)
        k = max(2, int(n*0.25))

        print(f"\n===== h={h}d | n={n} | rows {rows[0] and ''}=====")
        print(f"  fwd mean {fwd.mean():+.3%} | range {fwd.min():+.3%}..{fwd.max():+.3%}")

        # Q1 intensity
        order = np.argsort(totals_r); last_q = "intensity"
        lo_i, hi_i = order[:k], order[-k:]
        g = bootstrap_diff_gap(fwd[lo_i], fwd[hi_i])
        if g:
            est, lo, hi = g
            sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
            print(f"  Q1 INTENSITY  low-vol {fwd[lo_i].mean():+.3%} vs high-vol {fwd[hi_i].mean():+.3%}"
                  f" | gap {est:+.3%} 90%CI [{lo:+.3%},{hi:+.3%}] {sig}")

        # Q2 one-sidedness
        order = np.argsort(imb_r); last_q = "one-sided"
        lo_i, hi_i = order[:k], order[-k:]
        g = bootstrap_diff_gap(fwd[lo_i], fwd[hi_i])
        if g:
            est, lo, hi = g
            sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
            print(f"  Q2 ONE-SIDED  low {fwd[lo_i].mean():+.3%} vs high {fwd[hi_i].mean():+.3%}"
                  f" | gap {est:+.3%} 90%CI [{lo:+.3%},{hi:+.3%}] {sig}")

        # Q3 interaction: one-sidedness effect conditional on volume
        vol_hi = totals_r >= np.median(totals_r)
        vol_lo = ~vol_hi
        imb_hi = imb_r >= np.median(imb_r)
        for label, mask in (("HIGH-volume", vol_hi), ("LOW-volume", vol_lo)):
            sel = np.where(mask)[0]
            if len(sel) < 8:
                continue
            o = np.argsort(imb_r[sel]); kk = max(2,int(len(sel)*0.4))
            lo_i, hi_i = sel[o[:kk]], sel[o[-kk:]]
            g = bootstrap_diff_gap(fwd[lo_i], fwd[hi_i])
            if g:
                est, lo, hi = g
                sig = "SIG" if (lo>0 or hi<0) else "n.s."
                print(f"  Q3 one-sided effect @{label}: gap {est:+.3%} [{lo:+.3%},{hi:+.3%}] {sig} (n={len(sel)})")

        # confound: date range + price level of the high-intensity tail
        hdays = [rows[i][0] for i in order[-k:]]
        htot = [rows[i][1] for i in order[-k:]]
        print(f"  confound: high-{last_q} tail dates {min(hdays)}..{max(hdays)}"
              f" | vol ${min(htot)/1e6:.0f}M..${max(htot)/1e6:.0f}M")

if __name__ == "__main__":
    run()
