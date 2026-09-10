#!/usr/bin/env python3
"""
audit_display_components_iaaft.py — Audit kejujuran komponen display-only
=========================================================================
Tujuan: untuk komponen yang DITAMPILKAN tapi TIDAK di-blend, uji apakah klaim
"informatif" — yang biasanya diwariskan dari IC-screen — benar-benar bertahan
terhadap tiga gate: purged-CV/embargo, era-split, dan **null surrogate IAAFT**
(uji yang belum pernah dipakai di repo ini sebelum 2026-09).

Kenapa null IAAFT relevan untuk sinyal (bukan hanya untuk harga):
    IAAFT pada suatu SERIAL SINYAL mempertahankan spektrum daya (jadi
    autokorelasi/persistensinya sendiri) dan distribusi amplitudonya, hanya
    mengacak fase. Kalau IC(sinyal, return ke depan) versi nyata TIDAK berada
    di luar distribusi surrogate, maka "edge" itu bisa direproduksi oleh
    persistensi sinyal itu sendiri → tidak ada informasi TIMING mandiri.
Contoh nyata yang gugur karena uji ini: te_asym (p=0.37), rho1 (p=0.11).

Data & sinyal (point-in-time; transformasi tren memakai window trailing):
  A. Orderflow Binance Vision (2017-08-17..2026-07-31, 3.241 hari) — panel
     LENGKAP, bukan lagi window 2017-2020 → era-split 3 era kini mungkin
     (ini alasan baru untuk menguji ulang; dokumen lama menyebut keterbatasan
     "reduced window 2017-2020").
  B. CoinMetrics BTC (409 hari 2025-07..2026-09) — MVRV/AdrActCnt/HashRate;
     level mentah adalah artefak tren → dipakai perubahan 30-hari (log),
     bukan level. Riwayat <3 tahun ⇒ TIDAK bisa era-split (dilaporkan apa
     adanya, bukan dipaksa).

Output: analysis/.audit_display_components.json  + ringkasan stdout.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from criticality_math import iaaft  # noqa: E402

SFC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(SFC, "analysis", ".audit_display_components.json")
SEED = 20260911
HORIZONS = (7, 30)
ERAS = (("era1", "2017-08-01", "2020-12-31"),
        ("era2", "2021-01-01", "2023-12-31"),
        ("era3", "2024-01-01", "2026-12-31"))
N_SURR = int(os.environ.get("AUDIT_N_SURR", "200"))


def load_prices():
    with open(os.path.join(SFC, "data/binance_vision_daily.json")) as f:
        raw = json.load(f)
    return {d: raw[d]["close"] for d in raw}


def forward_returns(close_by_date, dates, h):
    out = np.full(len(dates), np.nan)
    c = np.array([close_by_date.get(d, np.nan) for d in dates], dtype=float)
    for i in range(len(dates) - h):
        if np.isfinite(c[i]) and np.isfinite(c[i + h]) and c[i] > 0:
            out[i] = c[i + h] / c[i] - 1.0
    return out


def rank_ic(sig, fwd):
    m = np.isfinite(sig) & np.isfinite(fwd)
    if m.sum() < 100:
        return float("nan"), float("nan"), int(m.sum())
    rho, p = spearmanr(sig[m], fwd[m])
    return float(rho), float(p), int(m.sum())


def gap_and_signif(sig, fwd, n_boot=1000, block=20, rng=None, q=0.2):
    rng = rng or np.random.default_rng(SEED)
    m = np.isfinite(sig) & np.isfinite(fwd)
    s, f = sig[m], fwd[m]
    n = len(s)
    if n < 200:
        return float("nan"), float("nan")
    obs = float(f[s >= np.quantile(s, 1 - q)].mean() - f[s <= np.quantile(s, q)].mean())
    nb = int(np.ceil(n / block))
    vals = np.empty(n_boot)
    for b in range(n_boot):
        st = rng.integers(0, n - block + 1, nb)
        idx = np.concatenate([np.arange(x, x + block) for x in st])[:n]
        sb, fb = s[idx], f[idx]
        vals[b] = fb[sb >= np.quantile(sb, 1 - q)].mean() - fb[sb <= np.quantile(sb, q)].mean()
    p = float(np.mean(vals >= 0)) if obs > 0 else float(np.mean(vals <= 0))
    return obs, p


def purged_cv_auc(sig, y, embargo, n_folds=5):
    m = np.isfinite(sig) & np.isfinite(y)
    X = np.asarray(sig[m], dtype=float)[:, None]
    Y = np.asarray(y[m]).astype(int)
    n = len(Y)
    if n < 300 or len(np.unique(Y)) < 2:
        return float("nan"), 0
    bounds = np.linspace(0, n, n_folds + 1).astype(int)
    P, L = [], []
    for k in range(n_folds):
        a, b = bounds[k], bounds[k + 1]
        tr = np.ones(n, dtype=bool)
        tr[a:b] = False
        tr[max(0, a - embargo):min(n, b + embargo)] = False
        if tr.sum() < 100 or len(np.unique(Y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(), X[tr].std()
        sd = sd if sd > 0 else 1.0
        clf = LogisticRegression(max_iter=2000)
        clf.fit((X[tr] - mu) / sd, Y[tr])
        P.append(clf.predict_proba((X[a:b] - mu) / sd)[:, 1])
        L.append(Y[a:b])
    if not P:
        return float("nan"), 0
    P, L = np.concatenate(P), np.concatenate(L)
    if len(np.unique(L)) < 2:
        return float("nan"), len(L)
    return float(roc_auc_score(L, P)), len(L)


def iaaft_null_ic(sig, fwd, n_surr=N_SURR, rng=None):
    """p = fraksi |IC30 surrogate| ≥ |IC30 nyata|; surrogate = IAAFT sinyal."""
    rng = rng or np.random.default_rng(SEED + 5)
    m = np.isfinite(sig) & np.isfinite(fwd)
    s, f = sig[m], fwd[m]
    if len(s) < 300:
        return float("nan"), float("nan"), 0
    real, _, _ = rank_ic(s, f)
    vals = []
    for _ in range(n_surr):
        sur = iaaft(np.nan_to_num(s, nan=0.0), n_iter=40, rng=rng)
        r, _, _ = rank_ic(sur, f)
        if np.isfinite(r):
            vals.append(r)
    if not vals:
        return float("nan"), float("nan"), 0
    v = np.asarray(vals)
    return real, float(np.mean(np.abs(v) >= abs(real))), len(v)


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    n = len(pv)
    qv = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        prev = min(prev, pv[order[rank]] * n / (rank + 1))
        qv[order[rank]] = prev
    q[ok] = np.minimum(qv, 1.0)
    return q


def trailing_change(vals, dates, win=30):
    """Perubahan log 30 hari, point-in-time (hanya masa lalu)."""
    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        if i < win:
            continue
        a, b = vals[i - win], vals[i]
        if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
            out[i] = float(np.log(b / a))
    return out


def main():
    t0 = time.time()
    close_by_date = load_prices()
    result = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": SEED,
              "n_surrogates": N_SURR, "components": {}, "notes": {}}

    # ── A. ORDERFLOW (panel 2017-2026 penuh) ────────────────────
    with open(os.path.join(SFC, "data/binance_orderflow_daily.json")) as f:
        of = json.load(f)
    dates = sorted(of)
    result["notes"]["orderflow"] = {
        "n_days": len(dates), "span": [dates[0], dates[-1]],
        "catatan": ("panel LENGKAP (dulu hanya 2017-2020) → era-split 3 era "
                    "kini mungkin; ini alasan baru untuk uji ulang"),
    }
    def col(keys):
        return np.array([of[d].get(keys, np.nan) if of[d].get(keys) is not None else np.nan
                         for d in dates], dtype=float)
    of_signals = {
        "of_taker_imbalance_qty": col("taker_imbalance_qty"),
        "of_taker_buy_ratio": col("taker_buy_ratio"),
        "of_whale_lo_count": col("whale_lo_count"),
        "of_n_trades": col("n_trades"),
        "of_total_quote": col("total_quote"),
    }
    fwd = {h: forward_returns(close_by_date, dates, h) for h in HORIZONS}
    y30 = np.where(np.isfinite(fwd[30]), (fwd[30] < 0).astype(float), np.nan)
    era_of = np.array(["pre"] * len(dates), dtype=object)
    for name, a, b in ERAS:
        era_of[(np.array(dates) >= a) & (np.array(dates) <= b)] = name

    fam_p, fam_keys = [], []
    for name, sig in of_signals.items():
        entry = {}
        for h in HORIZONS:
            rho, p, n = rank_ic(sig, fwd[h])
            entry[f"ic{h}d"] = {"rho": rho, "p": p, "n": n}
            fam_p.append(p); fam_keys.append((name, f"ic{h}d"))
        gap, gp = gap_and_signif(sig, fwd[30])
        entry["gap30_pp"] = gap * 100 if np.isfinite(gap) else None
        entry["gap30_p_sign"] = gp
        auc, n_oos = purged_cv_auc(sig, y30, embargo=30)
        entry["purged_auc30"] = auc
        entry["n_oos"] = n_oos
        _, p_surr, n_surr_used = iaaft_null_ic(sig, fwd[30])
        entry["iaaft_p_ic30"] = p_surr
        entry["iaaft_n"] = n_surr_used
        entry["era30"] = {}
        for ename, _, _ in ERAS:
            mask = era_of == ename
            g, pv = gap_and_signif(np.where(mask, sig, np.nan), fwd[30], n_boot=500)
            entry["era30"][ename] = {"gap_pp": g * 100 if np.isfinite(g) else None,
                                     "p_sign": pv}
        e2, e3 = entry["era30"]["era2"], entry["era30"]["era3"]
        entry["era2_era3_sign_consistent"] = bool(
            e2["gap_pp"] is not None and e3["gap_pp"] is not None
            and np.sign(e2["gap_pp"]) == np.sign(e3["gap_pp"]))
        result["components"][name] = entry
    q = bh_fdr(fam_p)
    for (name, key), qv in zip(fam_keys, q):
        result["components"][name][key]["q_bh"] = float(qv)

    # ── B. COINMETRICS (409 hari; tanpa era-split) ──────────────
    with open(os.path.join(SFC, "data/coinmetrics_btc_daily.json")) as f:
        cm = json.load(f)
    cdates = sorted(cm)
    result["notes"]["coinmetrics"] = {
        "n_days": len(cdates), "span": [cdates[0], cdates[-1]],
        "catatan": ("<3 tahun → era-split TIDAK mungkin; level mentah = artefak "
                    "tren, dipakai perubahan 30-hari"),
    }
    cm_signals = {}
    for field, label in (("CapMVRVCur", "cm_mvrv_chg30"),
                         ("AdrActCnt", "cm_adract_chg30"),
                         ("HashRate", "cm_hashrate_chg30"),
                         ("SplyCur", "cm_supply_chg30")):
        raw = np.array([cm[d].get(field, np.nan) if cm[d].get(field) is not None
                        else np.nan for d in cdates], dtype=float)
        cm_signals[label] = trailing_change(raw, cdates, 30)
    fwd_cm = {h: forward_returns(close_by_date, cdates, h) for h in HORIZONS}
    yc30 = np.where(np.isfinite(fwd_cm[30]), (fwd_cm[30] < 0).astype(float), np.nan)
    for name, sig in cm_signals.items():
        entry = {}
        for h in HORIZONS:
            rho, p, n = rank_ic(sig, fwd_cm[h])
            entry[f"ic{h}d"] = {"rho": rho, "p": p, "n": n}
        gap, gp = gap_and_signif(sig, fwd_cm[30], n_boot=500)
        entry["gap30_pp"] = gap * 100 if np.isfinite(gap) else None
        entry["gap30_p_sign"] = gp
        auc, n_oos = purged_cv_auc(sig, yc30, embargo=30)
        entry["purged_auc30"] = auc
        entry["n_oos"] = n_oos
        _, p_surr, n_used = iaaft_null_ic(sig, fwd_cm[30], n_surr=max(50, N_SURR // 2))
        entry["iaaft_p_ic30"] = p_surr
        entry["iaaft_n"] = n_used
        entry["era30"] = {"catatan": "tidak dihitung (<3 tahun)"}
        entry["era2_era3_sign_consistent"] = None
        result["components"][name] = entry

    # ── Verdict per komponen ────────────────────────────────────
    for name, e in result["components"].items():
        auc = e.get("purged_auc30")
        p_surr = e.get("iaaft_p_ic30")
        cons = e.get("era2_era3_sign_consistent")
        c_auc = bool(np.isfinite(auc) and auc > 0.55)
        c_surr = bool(np.isfinite(p_surr) and p_surr < 0.05)
        c_era = bool(cons) if cons is not None else None
        if c_auc and c_surr and (c_era is not False):
            v = "HAS_EDGE (kandidat uji lanjut)"
        elif not c_surr:
            v = "NO_EDGE_LINEAR_ARTIFACT (gagal null IAAFT)"
        elif not c_auc:
            v = "NO_EDGE (purged-CV ≤ 0.55)"
        else:
            v = "ERA_UNSTABLE (edge tidak konsisten era2/era3)"
        e["verdict"] = v
        e["gates"] = {"purged_auc_gt_0.55": c_auc,
                      "iaaft_p_lt_0.05": c_surr,
                      "era2_era3_consistent": c_era}

    # ── Cetak ringkasan ─────────────────────────────────────────
    print(f"\n{'komponen':26s} {'IC7d':>7s} {'IC30d':>7s} {'gap30pp':>8s} "
          f"{'pAUC30':>7s} {'pIAAFT':>7s} {'era2/3':>7s}  VERDICT")
    for name, e in result["components"].items():
        ic7 = e["ic7d"]["rho"]; ic30 = e["ic30d"]["rho"]
        g = e["gap30_pp"]; a = e.get("purged_auc30"); ps = e.get("iaaft_p_ic30")
        cons = e.get("era2_era3_sign_consistent")
        cons_s = "-" if cons is None else ("OK" if cons else "FLIP")
        print(f"{name:26s} {ic7:+7.3f} {ic30:+7.3f} "
              f"{(g if g is not None else float('nan')):+8.2f} "
              f"{(a if a is not None and np.isfinite(a) else float('nan')):7.3f} "
              f"{(ps if ps is not None and np.isfinite(ps) else float('nan')):7.3f} "
              f"{cons_s:>7s}  {e['verdict']}")
    n_edge = sum(1 for e in result["components"].values() if e["verdict"].startswith("HAS_EDGE"))
    result["summary"] = {
        "n_components": len(result["components"]),
        "n_has_edge": n_edge,
        "n_no_edge": len(result["components"]) - n_edge,
    }
    print(f"\nRingkasan: {len(result['components'])} komponen diaudit, "
          f"{n_edge} lolos semua gate, {len(result['components'])-n_edge} tidak.")
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(f"Disimpan: {OUT}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
