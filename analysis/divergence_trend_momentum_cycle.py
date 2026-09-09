#!/usr/bin/env python3
"""
Cycle / market-phase split of the divergence effect (final documentation).

Question: is the "bearish momentum within a strong structural uptrend"
(divergence) effect phase-conditional? Earlier era-split (divergence_trend_
momentum_deep.py) found the sign FLIPS across calendar eras (era1 negative,
era2 positive, interaction p=0.015). Calendar eras mix bull/bear/sideways
within one era, so here we condition on an OBJECTIVE market-cycle phase
instead, to see whether the effect is really bull-maturity-dependent rather
than merely time-conditional.

PHASE (per day, price-only, forward-looking-free):
    BULL = close > SMA200 AND SMA200_90d_slope > 0   (sustained uptrend cycle)
    BEAR = otherwise (downtrend / sideways / cycle break)

Reported, WITHIN strong-structure days (ts_struct >= 65):
    bear-momentum (divergence) vs non-bear fwd-return means, per phase and
    horizon, plus a pooled OLS bear coefficient controlling phase.
    Split further into "fresh/mid bull" vs "late bull" by drawdown from the
    trailing-180d high (>=15% off high = correction/weakness, else near-high).

NOT a blend; documentation only. Caveat: tiny event counts -> low power.
"""
import json, os
import numpy as np
import pandas as pd
from scipy import stats

from divergence_trend_momentum_test import (load_prices, build_features,
                                            classify, HORIZONS, MOM_BEAR,
                                            MOM_BULL, TS_STRONG)
from divergence_trend_momentum_deep import _par, ols

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(SFC_DIR, "analysis", ".divergence_trend_momentum_cycle.json")
BEAR_DD = 0.15  # >=15% below trailing-180d high => "correction / late-cycle weakness"


def label_phase(df):
    c = df["close"]
    ma200 = c.rolling(200).mean()
    slope = ma200.pct_change(90)
    df["ma200"] = ma200
    bull = (c > ma200) & (slope > 0)
    df["bull_cycle"] = bull.astype(float)
    # maturity: near-high vs off-high (correction) within any phase
    hi180 = c.rolling(180).max()
    dd = hi180 / c - 1.0  # drawdown from trailing 180d high (>=0)
    df["dd_hi180"] = dd
    df["near_high"] = (dd < BEAR_DD).astype(float)
    df["phase"] = np.where(bull, "BULL", "BEAR")
    df["maturity"] = np.where(
        (bull == 1) & (dd < BEAR_DD), "BULL_NEAR_HIGH",   # healthy bull, near high
        np.where((bull == 1) & (dd >= BEAR_DD), "BULL_OFF_HIGH",  # bull but correcting
                 "NOT_BULL"))
    return df


def stats_of(y):
    y = np.asarray(y, dtype=float); y = y[~np.isnan(y)]
    n = len(y)
    if n == 0:
        return {"n": 0}
    return {"n": int(n), "mean_pp": round(float(y.mean()) * 100, 3),
            "p_pos": round(float((y > 0).mean()), 3),
            "std_pp": round(float(y.std()) * 100, 3)}


def main():
    df = load_prices()
    df = build_features(df)
    df = classify(df)
    df = label_phase(df)
    df = df.dropna(subset=["trend_strength", "ts_struct", "ma200"]).reset_index(drop=True)
    c = df["close"]
    for h in HORIZONS:
        df[f"fwd_{h}d"] = c.shift(-h) / c - 1.0
    df["bear"] = (df["momentum_domain"] <= MOM_BEAR).astype(float)

    strong = df[df["ts_struct"] >= TS_STRONG].copy()
    res = {"meta": {
        "note": "Cycle-phase split (BULL = close>SMA200 & SMA200 90d slope>0). "
                "Within strong structural uptrend only. Documentation; tiny n -> low power.",
        "bear_dd_threshold": BEAR_DD, "n_strong": int(len(strong)),
        "date_range": [str(df["date"].iloc[0].date()), str(df["date"].iloc[-1].date())],
    }, "by_phase": {}, "by_maturity": {}}

    for ph in ("BULL", "BEAR"):
        sub = strong[strong["phase"] == ph]
        ent = {"n_total": int(len(sub)), "n_bear": int((sub["bear"] == 1).sum()),
               "pct_bear": round(float((sub["bear"] == 1).mean()) * 100, 2)}
        for h in HORIZONS:
            s = sub.dropna(subset=[f"fwd_{h}d"])
            bear = s[s["bear"] == 1][f"fwd_{h}d"]
            nonb = s[s["bear"] == 0][f"fwd_{h}d"]
            b = stats_of(bear.values); nb = stats_of(nonb.values)
            # pooled OLS bear coeff within phase (no era control), horizon
            Xp = sub[["bear"]].copy()
            yp = sub[f"fwd_{h}d"] * 100.0
            mm = ols(Xp, yp)
            ent[f"h{h}"] = {"bear": b, "nonbear": nb,
                            "gap_bear_minus_nonbear_pp": (round((b.get("mean_pp", np.nan) if "mean_pp" in b else np.nan) - (nb.get("mean_pp", np.nan) if "mean_pp" in nb else np.nan), 3)) if b.get("n") and nb.get("n") else None,
                            "bear_ols_pp": _par(mm, "bear")["coef_pp"],
                            "bear_ols_p": _par(mm, "bear")["p"]}
        res["by_phase"][ph] = ent

    for mt in ("BULL_NEAR_HIGH", "BULL_OFF_HIGH"):
        sub = strong[strong["maturity"] == mt]
        ent = {"n_total": int(len(sub)), "n_bear": int((sub["bear"] == 1).sum()),
               "pct_bear": round(float((sub["bear"] == 1).mean()) * 100, 2)}
        for h in HORIZONS:
            s = sub.dropna(subset=[f"fwd_{h}d"])
            bear = s[s["bear"] == 1][f"fwd_{h}d"]
            nonb = s[s["bear"] == 0][f"fwd_{h}d"]
            b = stats_of(bear.values); nb = stats_of(nonb.values)
            ent[f"h{h}"] = {"bear": b, "nonbear": nb,
                            "gap_pp": (round(b["mean_pp"] - nb["mean_pp"], 3)) if b.get("n") and nb.get("n") else None}
        res["by_maturity"][mt] = ent

    json.dump(res, open(OUT, "w"), indent=2, default=float)
    return res


if __name__ == "__main__":
    r = main()
    print(json.dumps(r, indent=2, default=float))
