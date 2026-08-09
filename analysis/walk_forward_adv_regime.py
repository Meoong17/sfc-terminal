"""
Walk-forward: does the advanced regime detector's CRISIS/severity predict adverse BTC moves?

Reconstruct the detector's output out-of-sample (no look-ahead): for each historical
5-min observation t, fit RegimeDetector on the PAST data [:t] and predict the regime
at t. Aggregate to daily severity. Compare severity (and crisis_prob) to forward BTC
returns (1d/3d/7d) via threshold-free quantile tails + bootstrap CI.

DATA: data_collection.json holds the raw m1-m5 method scores (0-1 scale) the detector
is fit on, with dates. Forward returns from binance_vision_daily.json.

Context: the live detector has a scale mismatch (predict-time feat_dict fed m1-m4 in
0-100 while fit data is 0-1), producing an unstable/arbitrary assignment. This test uses
correctly-scaled features, so it isolates the detector's LABELING quality. If severity
has no predictive edge, the CRISIS label is not a valid regime signal regardless of the
scale bug.
"""
import json, warnings, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
warnings.filterwarnings('ignore')
from ml.sfc_advanced import RegimeDetector

_ADV_SEV = {"BULL":20, "SIDEWAYS":45, "BEAR":60, "CRISIS":85, "NORMAL":20, "STRESS":55, "CAPITULATION":85}
RNG = np.random.default_rng(42)

def bootstrap_diff_gap(bottom, top, n_boot=20000):
    b, t = np.asarray(bottom), np.asarray(top)
    if len(b) < 3 or len(t) < 3: return None
    nb, nt = len(b), len(t)
    ib = RNG.integers(0, nb, (n_boot, nb)); it_ = RNG.integers(0, nt, (n_boot, nt))
    diff = t[it_].mean(axis=1) - b[ib].mean(axis=1)
    est = t.mean() - b.mean()
    return est, np.percentile(diff,5), np.percentile(diff,95)

def main():
    d = json.load(open('data_collection.json'))
    feats = d['features']; dates = d.get('dates', [])
    X = np.array([[float(o[i]) if i < len(o) and o[i] is not None else 0.5
                   for i in range(5)] for o in feats])
    n = len(X)
    price = json.load(open('data/binance_vision_daily.json'))
    price = {k: float(v['close']) for k, v in price.items()}

    # expanding walk-forward: fit on [:t], predict row t
    sev_by_date = {}   # date -> severity (keep LAST obs of day)
    crisis_by_date = {}
    start = 40
    for t in range(start, n):
        rd = RegimeDetector(n_regimes=4, random_state=42)
        rd.fit(X[:t])
        st = rd.get_regime_status(X[t:t+1])
        sev = _ADV_SEV.get(st['regime'], 45)
        date = (dates[t] or '')[:10]
        if not date: continue
        sev_by_date[date] = sev
        crisis_by_date[date] = st['crisis_probability']
    dates_sorted = sorted(sev_by_date)
    print(f"walk-forward severity over {len(dates_sorted)} daily points: "
          f"{dates_sorted[0]} .. {dates_sorted[-1]}")

    import datetime
    for h in (1,3,7):
        rows = []
        for date in dates_sorted:
            if date not in price: continue
            nd = datetime.date.fromisoformat(date) + datetime.timedelta(days=h)
            key = nd.isoformat()
            if key not in price: continue
            fwd = price[key]/price[date] - 1.0
            rows.append((date, sev_by_date[date], crisis_by_date[date], fwd))
        if len(rows) < 20:
            print(f"\n[h={h}d] too few rows ({len(rows)}), skip"); continue
        sev = np.array([r[1] for r in rows]); cr = np.array([r[2] for r in rows])
        fwd = np.array([r[3] for r in rows]); nrow = len(rows)
        k = max(2, int(nrow*0.25))
        # severity tail
        o = np.argsort(sev); lo, hi = o[:k], o[-k:]
        g = bootstrap_diff_gap(fwd[lo], fwd[hi])
        if g:
            est, a, b = g; sig = "SIG" if (a>0 or b<0) else "n.s."
            print(f"\n[h={h}d] n={nrow}")
            print(f"  SEV low {fwd[lo].mean():+.3%} vs high {fwd[hi].mean():+.3%} | gap {est:+.3%} 90%CI [{a:+.3%},{b:+.3%}] {sig}")
        # crisis_prob tail
        o = np.argsort(cr); lo, hi = o[:k], o[-k:]
        g = bootstrap_diff_gap(fwd[lo], fwd[hi])
        if g:
            est, a, b = g; sig = "SIG" if (a>0 or b<0) else "n.s."
            print(f"  CRISIS-P low {fwd[lo].mean():+.3%} vs high {fwd[hi].mean():+.3%} | gap {est:+.3%} 90%CI [{a:+.3%},{b:+.3%}] {sig}")
        # label distribution + confound
        from collections import Counter
        lbl = {20:'BULL',45:'SIDEWAYS',60:'BEAR',85:'CRISIS'}
        cnt = Counter(lbl.get(int(v),v) for v in sev)
        print(f"  sev distribution: {dict(cnt)}")
        hd = [r[0] for r in rows if r[1] >= 60]
        print(f"  high-sev(>=60) dates: {min(hd) if hd else '-'}..{max(hd) if hd else '-'} (n={len(hd)})")

if __name__ == "__main__":
    main()
