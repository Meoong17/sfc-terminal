#!/usr/bin/env python3
"""
sfc_evt_validate.py — Tahap 1: Uji empiris EVT tail risk (C1) pada return BTC nyata.

Backtest walk-forward (rolling, NO look-ahead): untuk tiap hari t setelah warm-up,
estimasi VaR_95 & VaR_99 dari jendela trailing (mis. 500 hari) dengan dua metode:
  (a) NORMAL  — VaR = mu + z_alpha * sigma
  (b) EVT-POT  — Generalized Pareto pada ekses di atas threshold (kuantil ~90%)
Lalu catat hit = return_t < -VaR_t. Ukur:

  - Unconditional Coverage (Kupiec POF): apakah rasio hit ≈ nominal alpha?
  - Independence (Christoffersen): apakah hit acak / tidak berkelompok?
  - Conditional Coverage gabungan (Kupiec+Christoffersen).
  - Mean realized excess vs ES: seberapa akurat prediksi besarnya kerugian.

Metode lebih baik = rasio hit mendekati nominal, p-value Kupiec/Christoffersen
tinggi (tidak ditolak), dan rasio excess (ES) mendekati 1 (tidak underestimate).

Jalankan:  cd ~/sfc && .venv/bin/python analysis/sfc_evt_validate.py
"""
import json
import sys
import numpy as np
from scipy import stats

sys.path.insert(0, "/home/ubuntu/sfc")
from analysis.sfc_methods_academic import _gpd_fit  # reuse EVT fitter


def load_returns():
    d = json.load(open("/home/ubuntu/sfc/historical_data.json"))
    closes = np.array([x["close"] for x in d], dtype=float)
    dates = [x["date"] for x in d]
    rets = np.diff(closes) / closes[:-1]  # simple daily returns
    return rets, dates[1:]


def normal_var(returns, q):
    mu, s = returns.mean(), returns.std(ddof=1)
    z = stats.norm.ppf(q)  # q=0.95 -> z=1.645, q=0.99 -> z=2.326
    return mu + z * s  # loss > 0


def evt_var(returns, q, threshold_q=0.90):
    """EVT-POT VaR (positive loss magnitude) dari jendela trailing."""
    r = returns[~np.isnan(returns)]
    n = len(r)
    if n < 30:
        return normal_var(returns, q)
    loss = -r
    u = np.quantile(loss, threshold_q)
    excess = loss[loss > u] - u
    if len(excess) < 5:
        return normal_var(returns, q)
    xi, beta = _gpd_fit(excess)
    prob = 1.0 - q
    if abs(xi) < 1e-9:
        var = u + beta * np.log((n / len(excess)) * prob)
    else:
        var = u + (beta / xi) * (((n / len(excess)) * prob) ** (-xi) - 1.0)
    return var


def kupiec_pof(hits, n, alpha):
    """Unconditional coverage: LR = -2 ln[ (1-a)^(n-x) a^x / (1-x/n)^(n-x) (x/n)^x ]."""
    x = int(hits)
    if x == 0:
        return 0.0
    pi = x / n
    lr = -2.0 * ((n - x) * np.log((1 - alpha) / (1 - pi)) + x * np.log(alpha / pi))
    return float(1 - stats.chi2.cdf(lr, df=1))


def christoffersen_indep(hits_vec):
    """Independence: transition matrix t-test / LR pada run hits."""
    h = np.asarray(hits_vec, dtype=int)
    if h.sum() == 0 or (1 - h).sum() == 0:
        return 1.0
    n01 = int(((h[:-1] == 0) & (h[1:] == 1)).sum())
    n00 = int(((h[:-1] == 0) & (h[1:] == 0)).sum())
    n11 = int(((h[:-1] == 1) & (h[1:] == 1)).sum())
    n10 = int(((h[:-1] == 1) & (h[1:] == 0)).sum())
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = h.sum() / len(h)
    if 0 < pi < 1 and 0 < pi01 < 1 and 0 < pi11 < 1:
        lr = -2.0 * ((n00 + n01) * np.log(1 - pi) + (n10 + n11) * np.log(pi)
                     - n00 * np.log(1 - pi01) - n01 * np.log(pi01)
                     - n10 * np.log(1 - pi11) - n11 * np.log(pi11))
        return float(1 - stats.chi2.cdf(lr, df=1))
    return 1.0


def backtest(returns, window=500, alphas=(0.95, 0.99)):
    n = len(returns)
    out = {}
    for a in alphas:
        hits_norm, hits_evt = 0, 0
        norm_vec, evt_vec = [], []
        es_norm_sum, es_evt_sum = 0.0, 0.0
        es_norm_cnt, es_evt_cnt = 0, 0
        for t in range(window, n):
            win = returns[t - window:t]
            r_t = returns[t]
            vn = normal_var(win, a)
            ve = evt_var(win, a)
            hit_n = float(r_t < -vn); hit_e = float(r_t < -ve)
            hits_norm += hit_n; hits_evt += hit_e
            norm_vec.append(hit_n); evt_vec.append(hit_e)
            if hit_n:
                es_norm_sum += -r_t; es_norm_cnt += 1
            if hit_e:
                es_evt_sum += -r_t; es_evt_cnt += 1
        m = n - window
        out[a] = {
            "n_test": m,
            "nominal": 1 - a,
            "norm_hit_rate": hits_norm / m,
            "evt_hit_rate": hits_evt / m,
            "norm_kupiec_p": kupiec_pof(hits_norm, m, 1 - a),
            "evt_kupiec_p": kupiec_pof(hits_evt, m, 1 - a),
            "norm_chris_p": christoffersen_indep(np.array(norm_vec)),
            "evt_chris_p": christoffersen_indep(np.array(evt_vec)),
            # ES precision: mean realized excess (for hits) vs normal-fit ES proxy
            "norm_mean_excess": es_norm_sum / es_norm_cnt if es_norm_cnt else float("nan"),
            "evt_mean_excess": es_evt_sum / es_evt_cnt if es_evt_cnt else float("nan"),
        }
    return out


def main():
    rets, dates = load_returns()
    print(f"Data: {len(rets)} return harian, {dates[0]} s/d {dates[-1]}")
    print(f"Warm-up window: 500 hari | OOS dari {dates[500]}")
    print(f"return harian: mean={rets.mean():+.4%} std={rets.std(ddof=1):.2%} "
          f"min={rets.min():+.1%} max={rets.max():+.1%}\n")

    res = backtest(rets)
    for a, r in res.items():
        print(f"=== VaR {int(a*100)}% (nominal exceedance {r['nominal']:.1%}) — OOS n={r['n_test']} ===")
        print(f"  NORMAL: hit-rate {r['norm_hit_rate']:.2%}  "
              f"(Kupiec p={r['norm_kupiec_p']:.3f}, Christoffersen p={r['norm_chris_p']:.3f})  "
              f"mean-excess {r['norm_mean_excess']:.2%}")
        print(f"  EVT   : hit-rate {r['evt_hit_rate']:.2%}  "
              f"(Kupiec p={r['evt_kupiec_p']:.3f}, Christoffersen p={r['evt_chris_p']:.3f})  "
              f"mean-excess {r['evt_mean_excess']:.2%}")

        # verdict per level
        nrate, erate = r["norm_hit_rate"], r["evt_hit_rate"]
        n_err = abs(nrate - r["nominal"]) / r["nominal"]
        e_err = abs(erate - r["nominal"]) / r["nominal"]
        better = "NORMAL" if n_err <= e_err else "EVT"
        well_cal = "YA (close to nominal)" if e_err < 0.5 else "TIDAK"
        print(f"  -> lebih dekat nominal: {better} | EVT kalibrasi-wajar: {well_cal} "
              f"(err normal {n_err:.2f} vs EVT {e_err:.2f})\n")


if __name__ == "__main__":
    main()
