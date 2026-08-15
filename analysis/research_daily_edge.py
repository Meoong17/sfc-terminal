#!/usr/bin/env python3
"""
research_daily_edge.py — uji edge prediktif variabel MAKRO HARIAN utk BTC
=====================================================================
Menjawab pertanyaan inti proposal Bisa.docx: "Apakah variabel makro daily
benar-benar menghasilkan alpha, atau hanya klasifikasi yang terlihat bagus
setelah hasil diketahui?" — dengan metode standar SFC (bootstrap P + BIC
posterior + era-split), TANPA sentuh scoring.

Sumber: data/merged/sfc_research_daily.json (macro FRED + BTC Binance Vision).
Menguji SETIAP kolom macro terhadap forward BTC return (7d, 30d).

Metode (sama dgn predictive-probability-validation skill):
  - Forward returns: BTC N hari kemudian, buang tail tanpa data masa depan.
  - Quantile gap: bottom-20% vs top-20% signal → gap fwd return.
    Polarity: signal tinggi → return lebih rendah (untuk variabel "stress").
  - Bootstrap P(predictive): fraksi dari 2000 draw dgn gap arah prediktif.
  - BIC posterior: regresi fwd return ~ signal vs intercept-only.
  - ERA-SPLIT (3 blok): edge BERTAHAN atau FLIP di era terbaru?

Output: .research_daily_edge.json + print tabel.
USAGE: .venv/bin/python analysis/research_daily_edge.py
"""
import json, os, random, sys, warnings
from datetime import datetime, timezone
import numpy as np
warnings.filterwarnings("ignore")

SFC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED = os.path.join(SFC_ROOT, "data", "merged", "sfc_research_daily.json")
OUTPUT = os.path.join(SFC_ROOT, ".research_daily_edge.json")

QUANTILE_TAIL = 0.20
HORIZONS = [7, 30]
N_BOOT = 2000
PRICE_COL = "BTC_close"

# Kolom macro yang diuji. Polaritas default: HIGH signal = stress/tight
# → fwd return LEBIH RENDAH (gap_high_minus_low < 0). Untuk seri risk-on
# (SPX/NDX/GOLD/COPPER/BRENT) high = kuat = BULLISH BTC → gap > 0.
# dict: kolom -> ("arah_polaritas", "resolusi")
MACRO_COLS = {
    # Rates / dollar (high = tight/expensive = bearish)
    "US2Y":  ("neg", "daily"), "US10Y": ("neg", "daily"),
    "US30Y": ("neg", "daily"), "US5Y": ("neg", "daily"),
    "BE10":  ("neg", "daily"), "REAL_Y10": ("neg", "daily"),
    "SPREAD_10_2": ("neg", "daily"), "SPREAD_5_30": ("neg", "daily"),
    "FEDFUNDS": ("neg", "daily"),
    # Vol / risk-off
    "VIX": ("neg", "daily"),
    # Equity (high = risk-on = bullish BTC)
    "SPX": ("pos", "daily"), "NDX": ("pos", "daily"),
    # Commodity (high = kuat/demand = mixed; kita uji dua arah di "pos")
    "BRENT": ("pos", "daily"), "WTI": ("pos", "daily"),
    "COPPER": ("pos", "monthly"),
    # Credit (high OAS = risk-off = bearish)
    "HY_OAS": ("neg", "daily"), "IG_OAS": ("neg", "daily"),
    # Liquidity (high FED_BS = mudah = bullish; high M2 = mudah; high TGA = drain)
    "FED_BS": ("pos", "weekly"), "ECB_BS": ("pos", "weekly"),
    "M2_US": ("pos", "monthly"), "BOJ_BS": ("pos", "monthly"),
    "TGA": ("neg", "weekly"), "RRP": ("neg", "daily"),
}


def add_forward_returns(rows):
    """Tambahkan kolom fwd_return_Nd (pct) per baris."""
    close = {r["date"]: r.get(PRICE_COL) for r in rows}
    dates = sorted(close)
    for i, r in enumerate(rows):
        d = r["date"]
        for h in HORIZONS:
            j = i + h
            if j < len(dates):
                c0, c1 = close[d], close.get(dates[j])
                if c0 and c1:
                    r[f"fwd_return_{h}d"] = (c1 - c0) / c0 * 100
                else:
                    r[f"fwd_return_{h}d"] = None
            else:
                r[f"fwd_return_{h}d"] = None
    return rows


def bootstrap_diff_ci(lo_g, hi_g, n_boot=N_BOOT):
    """CI90 dari perbedaan mean (high - low) via bootstrap."""
    if len(lo_g) < 2 or len(hi_g) < 2:
        return None, None, None
    nl, nh = len(lo_g), len(hi_g)
    diffs = []
    for _ in range(n_boot):
        sl = [lo_g[random.randrange(nl)] for _ in range(nl)]
        sh = [hi_g[random.randrange(nh)] for _ in range(nh)]
        diffs.append(sum(sh) / nh - sum(sl) / nl)
    diffs.sort()
    est = float(sum(diffs) / len(diffs))
    lo = diffs[int(0.05 * len(diffs))]
    hi = diffs[int(0.95 * len(diffs)) - 1]
    return est, lo, hi


def bootstrap_probability(lo_g, hi_g, polarity_neg):
    """Fraksi draw dgn gap arah prediktif."""
    if len(lo_g) < 2 or len(hi_g) < 2:
        return None
    nl, nh = len(lo_g), len(hi_g)
    n_ok = 0
    for _ in range(N_BOOT):
        sl = [lo_g[random.randrange(nl)] for _ in range(nl)]
        sh = [hi_g[random.randrange(nh)] for _ in range(nh)]
        gap = sum(sh) / nh - sum(sl) / nl
        if (gap < 0 and polarity_neg) or (gap > 0 and not polarity_neg):
            n_ok += 1
    return n_ok / N_BOOT


def bic_posterior(values, forward):
    from statsmodels.api import OLS
    from scipy import stats as sps
    x = np.array(values, float); y = np.array(forward, float); n = len(x)
    if n < 20:
        return {"error": "n too small"}
    b0 = float(np.sum((y - y.mean()) ** 2))
    bic0 = n * np.log(b0 / n) + 1 * np.log(n)
    X = np.column_stack([np.ones(n), x])
    m = OLS(y, X).fit()
    rss = float(np.sum(m.resid ** 2))
    bic1 = n * np.log(rss / n) + m.params.shape[0] * np.log(n)
    dbic = bic0 - bic1
    p_h1 = float(np.exp(dbic / 2) / (1 + np.exp(dbic / 2)))
    bf = float(np.exp(dbic / 2))
    ic = sps.spearmanr(x, y).correlation
    return {"n": n, "dBIC": round(dbic, 2), "bayes_factor_10": round(bf, 3),
            "posterior_prob_H1": round(p_h1, 4), "spearman_ic": round(float(ic), 4)}


def analyze_col(rows, col, polarity_neg, freq):
    """Analisis satu kolom macro utk semua horizon."""
    polarity = "neg" if polarity_neg else "pos"
    out = {"polarity": polarity, "freq": freq}
    for h in HORIZONS:
        fk = f"fwd_return_{h}d"
        paired = [(r[col], r.get(fk)) for r in rows
                  if r.get(col) is not None and r.get(fk) is not None]
        if len(paired) < 20:
            out[str(h)] = {"n": len(paired), "error": "insufficient"}
            continue
        paired.sort(key=lambda z: z[0])
        tn = max(1, int(len(paired) * QUANTILE_TAIL))
        lo_g = [v for _, v in paired[:tn]]
        hi_g = [v for _, v in paired[-tn:]]
        est, lo_, hi_ = bootstrap_diff_ci(lo_g, hi_g)
        prob = bootstrap_probability(lo_g, hi_g, polarity_neg)
        bic = bic_posterior([p[0] for p in paired], [p[1] for p in paired])
        pred_ok = (est < 0 and polarity_neg) or (est > 0 and not polarity_neg)
        out[str(h)] = {
            "n": len(paired),
            "quantile_gap_pp": round(est, 2) if est is not None else None,
            "ci90": [round(lo_, 2), round(hi_, 2)] if lo_ is not None else None,
            "prob_predictive": round(prob, 3) if prob is not None else None,
            "bic": bic,
            "predictive_dir_ok": bool(pred_ok),
        }
    # era-split (3 blok) — hanya utk horizon yang relevan, berbasis fwd_return_30d
    out["era_split"] = {}
    pts = [(r["date"], r[col], r.get("fwd_return_30d")) for r in rows
           if r.get(col) is not None and r.get("fwd_return_30d") is not None]
    pts.sort(key=lambda t: t[0])
    n = len(pts)
    if n >= 30:
        thirds = [pts[:n // 3], pts[n // 3: 2 * n // 3], pts[2 * n // 3:]]
        names = ["era1", "era2", "era3(latest)"]
        for name, block in zip(names, thirds):
            paired = [(p[1], p[2]) for p in block]
            if len(paired) < 10:
                continue
            paired.sort(key=lambda z: z[0])
            tn = max(1, int(len(paired) * QUANTILE_TAIL))
            lo_g = [v for _, v in paired[:tn]]
            hi_g = [v for _, v in paired[-tn:]]
            est, _, _ = bootstrap_diff_ci(lo_g, hi_g)
            prob = bootstrap_probability(lo_g, hi_g, polarity_neg)
            out["era_split"][name] = {
                "n": len(paired),
                "gap_30d_pp": round(est, 2) if est is not None else None,
                "prob_predictive": round(prob, 3) if prob is not None else None,
            }
    return out


def main():
    random.seed(42)
    if not os.path.exists(MERGED):
        print(f"No dataset: {MERGED}. Jalankan merge_macro_research.py dulu.")
        return
    rows = json.load(open(MERGED))
    rows = add_forward_returns(rows)
    print(f"Loaded {len(rows)} hari. Kolom macro diuji: {len(MACRO_COLS)}")
    print(f"Polaritas: neg = signal tinggi → return rendah (stress/risk-off); "
          f"pos = signal tinggi → return tinggi (risk-on).\n")

    result = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "n_days": len(rows), "columns": {}}
    for col, (pol, freq) in MACRO_COLS.items():
        polarity_neg = (pol == "neg")
        res = analyze_col(rows, col, polarity_neg, freq)
        result["columns"][col] = res
        # print ringkas
        line = f"{col:12s} [{freq:7s}] "
        for h in HORIZONS:
            r = res.get(str(h), {})
            if "error" in r:
                line += f"{h}d:n<20 "
            else:
                line += (f"{h}d:gap={r['quantile_gap_pp']:+.1f}pp "
                         f"P={r['prob_predictive']:.2f} "
                         f"(BF={r['bic']['bayes_factor_10']}) ")
        print(line)
        es = res.get("era_split", {})
        if es:
            esline = "   era30d: " + " | ".join(
                f"{k}:gap={v['gap_30d_pp']:+.1f}(P={v['prob_predictive']:.2f})"
                for k, v in es.items())
            print(esline)

    with open(OUTPUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {OUTPUT}")


if __name__ == "__main__":
    main()
