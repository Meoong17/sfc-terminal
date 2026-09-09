#!/usr/bin/env python3
"""
Deep follow-up on the divergence screen (1.docx): within a STRONG STRUCTURAL
uptrend, does a bearish momentum read (the "divergence") predict forward BTC
returns — and is any effect era-consistent (not just an era2 artifact)?

WHY THIS FORM: the threshold screen (.divergence_trend_momentum_test.py) found
the divergence state is rare (n=66 over 9y) and the full-sample 7d gap
(+2.34pp) is driven almost entirely by era2 2021-23 (era1 NEGATIVE). Per the
project's IERF discipline, an era-flip alone is not proof of no effect — so we
(a) pool the regression with era dummies/controls, and (b) gain power by using
ALL strong-structure days and treating momentum continuously instead of
slicing off a tiny bearish-threshold tail.

Question answered by the momentum coefficient within strong structure:
    - coefficient < 0  => lower (bearish) momentum predicts HIGHER fwd return
                           => divergence = mean-reversion to the upside
    - coefficient > 0  => lower momentum predicts LOWER return
                           => divergence = continuation/downside (the bullish
                              reading of "MomentumOverlay bearish" as bearish)
    - era interaction tests whether the effect is stable or era2-only.

NOT a blend. Display-only research screen.
"""
import json, os
import numpy as np
import pandas as pd
from scipy import stats

from divergence_trend_momentum_test import (  # reuse the price reconstruction
    load_prices, build_features, classify, HORIZONS, MOM_BEAR, MOM_BULL,
    TS_STRONG)

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(SFC_DIR, "analysis", ".divergence_trend_momentum_deep.json")


def era_of(dts):
    """era label by date (same edges as the screen)."""
    d = pd.Series(pd.to_datetime(dts))
    e = pd.Series("era1", index=d.index)
    e[d >= "2021-01-01"] = "era2"
    e[d >= "2024-01-01"] = "era3"
    return e


def prep():
    df = load_prices()
    df = build_features(df)
    df = classify(df)
    df = df.dropna(subset=["trend_strength", "ts_struct"]).reset_index(drop=True)
    c = df["close"]
    for h in HORIZONS:
        df[f"fwd_{h}d"] = c.shift(-h) / c - 1.0
    df["era"] = era_of(df["date"])
    df["era2"] = (df["era"] == "era2").astype(float)
    df["era3"] = (df["era"] == "era3").astype(float)
    df["mom_bear"] = (df["momentum_domain"] <= MOM_BEAR).astype(float)
    df["mom_bull"] = (df["momentum_domain"] >= MOM_BULL).astype(float)
    return df


def ols(X, y, use_hc=True):
    """Manual OLS with HC1 (robust) standard errors; returns a dict of
    {name: [coef, se, t, p]} + nobs/r2. y is a pd.Series aligned to X index."""
    Xa = np.column_stack([np.ones(len(X))] + [X[c].to_numpy(float) for c in X.columns])
    names = ["const"] + list(X.columns)
    ya = y.to_numpy(float)
    n, k = Xa.shape
    XtX_inv = np.linalg.pinv(Xa.T @ Xa)
    beta = XtX_inv @ (Xa.T @ ya)
    resid = ya - Xa @ beta
    # HC1 robust covariance
    sigma2 = (resid ** 2)
    bread = XtX_inv @ (Xa.T @ (Xa * sigma2[:, None])) @ XtX_inv
    if use_hc:
        cov = bread * (n / (n - k))
    else:
        s2 = float((resid ** 2).sum() / (n - k))
        cov = XtX_inv * s2
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    t = beta / np.where(se == 0, np.nan, se)
    p = 2.0 * stats.t.sf(np.abs(t), df=n - k)
    return {"names": names, "coef": beta, "se": se, "t": t, "p": p,
            "nobs": int(n), "r2": float(1.0 - (resid ** 2).sum() /
                                        ((ya - ya.mean()) ** 2).sum())}


def _par(m, name):
    i = m["names"].index(name)
    return {"coef_pp": round(float(m["coef"][i]), 4),
            "se": round(float(m["se"][i]), 4),
            "t": round(float(m["t"][i]), 3),
            "p": round(float(m["p"][i]), 4)}


def run_within(df, h):
    """Regress fwd_h on momentum (cont + bear/bull bands) only among
    strong-structure days, with era controls + era x bear interactions."""
    strong = df[df["ts_struct"] >= TS_STRONG].dropna(subset=[f"fwd_{h}d"]).copy()
    strong["Y"] = strong[f"fwd_{h}d"] * 100.0
    out = {"h": h, "n_strong": int(len(strong))}

    # (1) pooled, era-controlled, continuous momentum
    X1 = strong[["momentum_domain", "era2", "era3"]].copy()
    m1 = ols(X1, strong["Y"])
    p1 = _par(m1, "momentum_domain")
    out["model1_cont_mom"] = {"coef_momentum_domain_pp_per_unit": p1["coef_pp"],
                              "se": p1["se"], "t": p1["t"], "p": p1["p"],
                              "nobs": m1["nobs"], "r2": m1["r2"]}

    # (2) bear/bull bands vs neutral, era-controlled (coef = band minus neutral)
    X2 = strong[["mom_bear", "mom_bull", "era2", "era3"]].copy()
    m2 = ols(X2, strong["Y"])
    pb = _par(m2, "mom_bear")
    pbu = _par(m2, "mom_bull")
    out["model2_bands"] = {"coef_mom_bear_vs_neutral_pp": pb["coef_pp"],
                           "se": pb["se"], "p": pb["p"],
                           "coef_mom_bull_vs_neutral_pp": pbu["coef_pp"],
                           "p_bull": pbu["p"], "nobs": m2["nobs"]}

    # (3) era x bear interactions (era1 = reference) — is the bear effect stable?
    X3 = pd.concat([strong[["mom_bear", "era2", "era3"]],
                    (strong["mom_bear"] * strong["era2"]).rename("bear_x_era2"),
                    (strong["mom_bear"] * strong["era3"]).rename("bear_x_era3")],
                   axis=1).copy()
    m3 = ols(X3, strong["Y"])
    b_e1 = m3["coef"][m3["names"].index("mom_bear")]
    b_e2 = m3["coef"][m3["names"].index("bear_x_era2")]
    b_e3 = m3["coef"][m3["names"].index("bear_x_era3")]
    out["model3_era_interact"] = {
        "coef_bear_era1_pp": round(float(b_e1), 4),
        "coef_bear_x_era2_pp": round(float(b_e2), 4),
        "p_x_era2": _par(m3, "bear_x_era2")["p"],
        "coef_bear_x_era3_pp": round(float(b_e3), 4),
        "p_x_era3": _par(m3, "bear_x_era3")["p"],
        "bear_effect_era1_pp": round(float(b_e1), 4),
        "bear_effect_era2_pp": round(float(b_e1 + b_e2), 4),
        "bear_effect_era3_pp": round(float(b_e1 + b_e3), 4),
        "nobs": m3["nobs"],
    }
    # per-era simple means for transparency
    out["era_means"] = {}
    for e in ("era1", "era2", "era3"):
        sub = strong[strong["era"] == e]
        bear = sub[sub["mom_bear"] == 1]
        nonbear = sub[sub["mom_bear"] == 0]
        out["era_means"][e] = {
            "n": int(len(sub)),
            "n_bear": int(len(bear)),
            "mean_fwd_pp": round(float(sub["Y"].mean()), 3),
            "bear_mean_pp": round(float(bear["Y"].mean()), 3) if len(bear) else None,
            "nonbear_mean_pp": round(float(nonbear["Y"].mean()), 3) if len(nonbear) else None,
        }
    return out


def main():
    df = prep()
    res = {"meta": {
        "note": "Deep follow-up: within STRONG structural uptrend (ts_struct>=65), "
                "does bearish momentum predict forward returns? Era-controlled pooled "
                "OLS (HC1). Overlapping forward labels bias SEs low — treat as screen.",
        "horizons": HORIZONS, "n_total": int(len(df)),
        "n_strong_ts": int((df["ts_struct"] >= TS_STRONG).sum()),
    }, "models": {}}
    for h in HORIZONS:
        res["models"][str(h)] = run_within(df, h)
    json.dump(res, open(OUT, "w"), indent=2, default=float)
    return res


if __name__ == "__main__":
    r = main()
    print(json.dumps(r, indent=2, default=float))
