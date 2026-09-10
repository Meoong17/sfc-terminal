#!/usr/bin/env python3
"""
validate_igc_criticality.py — Baterai uji untuk IGC (Information-Geometric
Criticality) terhadap harga BTC kanonik Binance Vision (2017-08 .. terbaru).
================================================================================
PRINSIP: tidak ada klaim sebelum lolos gate. Kerangka uji memakai standar repo
ini (purged-CV + embargo López de Prado, era-split, block bootstrap, BH-FDR,
kontrol permutasi/surrogate IAAFT), dan dibandingkan dengan INCUMBENT
(volatilitas realized 21 hari = inti harga/vol yang sudah era-stable).

TIGA TUGAS UJI
  A. Prediktif forward-return: rank-IC (7/30/90d) + BH-FDR; quantile gap
     top/bottom-20% + block-bootstrap P(sign); era-split 3 blok; purged-CV AUC.
  B. Early-warning (tugas yang secara teori memang cocok untuk besaran
     criticality): event study drawdown ≥15% — apakah sinyal naik di jendela
     pra-onset [onset−30, onset−5]? Uji blok-bootstrap vs unconditional, dan
     AUC deteksi "event dalam 30 hari ke depan".
  C. Inkremental atas incumbent: partial-IC setelah kontrol vol-21d +
     ΔAUC purged-CV (incumbent saja vs incumbent+IGC).
  N. NULL NONLINEAR: 100 surrogate IAAFT dari deret return (spektrum daya +
     distribusi amplitudo SAMA, urutan nonlinier dihancurkan) → distribusi
     statistik di bawah H0. p = fraksi |stat_surrogate| ≥ |stat_real|.

KRITERIA VERDICT (pre-spesifikasi, sebelum melihat hasil)
  LOLOS hanya jika SEMUA:
    (1) purged-CV pooled AUC > 0.55 DAN CI_lower(bootstrap) > 0.50
    (2) era2 & era3 tanda gap konsisten (keduanya negatif untuk sinyal "rapuh")
    (3) p_surrogate_IAAFT < 0.05 pada statistik utama (IC 30d)
    (4) inkremental: ΔAUC(incumbent+IGC − incumbent) > 0 dengan CI_lower > 0
  Selain itu: TIDAK di-blend (display-only / ditolak), dilaporkan apa adanya.

Output: analysis/.validate_igc_criticality.json
"""
import json
import os
import sys
import time
from datetime import date

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from criticality_math import (  # noqa: E402
    MEASURE_KEYS, iaaft, igc_composite, rolling_measures,
)

SFC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(SFC, "analysis", ".validate_igc_criticality.json")
SEED = 20260911
HORIZONS = (7, 30, 90)
ERAS = (("era1", "2017-08-01", "2020-12-31"),
        ("era2", "2021-01-01", "2023-12-31"),
        ("era3", "2024-01-01", "2026-12-31"))
WINDOW = 250
N_SURROGATES = int(os.environ.get("IGC_N_SURR", "100"))
SURR_STRIDE = 5
VOL_WIN = 21


# ────────────────────────────────────────────────────────────────
# Data
# ────────────────────────────────────────────────────────────────

def load_btc():
    with open(os.path.join(SFC, "data/binance_vision_daily.json")) as f:
        raw = json.load(f)
    dates = sorted(raw)
    close = np.array([raw[d]["close"] for d in dates], dtype=float)
    ret = np.diff(np.log(close))
    return dates[1:], close[1:], ret


# ────────────────────────────────────────────────────────────────
# Statistik uji
# ────────────────────────────────────────────────────────────────

def rank_ic(sig, fwd):
    m = np.isfinite(sig) & np.isfinite(fwd)
    if m.sum() < 60:
        return float("nan"), float("nan"), int(m.sum())
    rho, p = spearmanr(sig[m], fwd[m])
    return float(rho), float(p), int(m.sum())


def quantile_gap(sig, fwd, q=0.20):
    """mean fwd(top-q sinyal) − mean fwd(bottom-q sinyal)."""
    m = np.isfinite(sig) & np.isfinite(fwd)
    if m.sum() < 100:
        return float("nan"), float("nan"), int(m.sum())
    s, f = sig[m], fwd[m]
    hi = s >= np.quantile(s, 1 - q)
    lo = s <= np.quantile(s, q)
    return float(f[hi].mean() - f[lo].mean()) * 100.0, float(f[hi].mean() - f[lo].mean()), int(m.sum())


def block_bootstrap_p(sig, fwd, n_boot=2000, block=20, q=0.20, rng=None):
    """P(gap_boot < 0) dengan moving-block bootstrap (blok 20 hari)."""
    rng = rng or np.random.default_rng(SEED)
    m = np.isfinite(sig) & np.isfinite(fwd)
    s, f = sig[m], fwd[m]
    n = len(s)
    if n < 200:
        return float("nan"), float("nan")
    nb = int(np.ceil(n / block))
    gaps = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, nb)
        idx = np.concatenate([np.arange(st, st + block) for st in starts])[:n]
        sb, fb = s[idx], f[idx]
        hi = sb >= np.quantile(sb, 1 - q)
        lo = sb <= np.quantile(sb, q)
        gaps[b] = fb[hi].mean() - fb[lo].mean()
    obs = float(f[s >= np.quantile(s, 1 - q)].mean() - f[s <= np.quantile(s, q)].mean())
    p_neg = float(np.mean(gaps < 0)); p_pos = float(np.mean(gaps > 0))
    return obs, float(min(p_neg, p_pos) if obs < 0 else p_pos), float(np.std(gaps))


def purged_cv_auc(sig, y, embargo, n_folds=5, extra=None, mask=None):
    """Purged-CV/embargo (López de Prado), LogisticRegression 1-fitur (+extra).

    `mask` opsional: mask boolean eksplisit agar beberapa model dinilai pada
    SAMPEL OOS YANG SAMA PERSIS (syarat CI berpasangan ΔAUC yang sah).
    Mengembalikan (pooled_oos_auc, n_oos, oos_probs, oos_labels, n_train_used).
    """
    m = np.isfinite(sig) & np.isfinite(y)
    if mask is not None:
        m = m & mask
    X = sig[m][:, None]
    Y = y[m].astype(int)
    if extra is not None:
        X = np.hstack([X, extra[m][:, None]])
    X = np.asarray(X, dtype=float)
    Xs = X.copy()
    n = len(Y)
    if n < 300:
        return float("nan"), 0, np.array([]), np.array([]), 0
    bounds = np.linspace(0, n, n_folds + 1).astype(int)
    probs, labs, used = [], [], 0
    for k in range(n_folds):
        te0, te1 = bounds[k], bounds[k + 1]
        tr_mask = np.ones(n, dtype=bool)
        tr_mask[te0:te1] = False
        # embargo: buang sampel latih dalam `embargo` hari dari batas test
        lo = max(0, te0 - embargo); hi = min(n, te1 + embargo)
        tr_mask[lo:hi] = False
        if tr_mask.sum() < 100 or (te1 - te0) < 20:
            continue
        Xtr, Xte = Xs[tr_mask], Xs[te0:te1]
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
        sd[sd == 0] = 1.0
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        if len(np.unique(Y[tr_mask])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, Y[tr_mask])
        p = clf.predict_proba(Xte)[:, 1]
        probs.append(p); labs.append(Y[te0:te1]); used += int(tr_mask.sum())
    if not probs:
        return float("nan"), 0, np.array([]), np.array([]), 0
    P = np.concatenate(probs); L = np.concatenate(labs)
    if len(np.unique(L)) < 2:
        return float("nan"), len(L), P, L, used
    return float(roc_auc_score(L, P)), len(L), P, L, used


def bootstrap_auc_ci(P, L, n_boot=1000, block=20, rng=None):
    rng = rng or np.random.default_rng(SEED + 1)
    n = len(P)
    if n < 100 or len(np.unique(L)) < 2:
        return float("nan"), float("nan")
    nb = int(np.ceil(n / block))
    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        if len(np.unique(L[idx])) < 2:
            continue
        vals.append(roc_auc_score(L[idx], P[idx]))
    if len(vals) < 50:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_delta_auc(P1, L1, P2, L2, n_boot=1000, block=20, rng=None):
    """CI berpasangan untuk ΔAUC = AUC(model2) − AUC(model1) pada sampel OOS yang
    SAMA (moving-block bootstrap, indeks identik untuk kedua model)."""
    rng = rng or np.random.default_rng(SEED + 4)
    n = len(P1)
    if n < 100 or len(np.unique(L1)) < 2:
        return float("nan"), float("nan")
    nb = int(np.ceil(n / block))
    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        if len(np.unique(L1[idx])) < 2:
            continue
        vals.append(roc_auc_score(L1[idx], P2[idx]) - roc_auc_score(L1[idx], P1[idx]))
    if len(vals) < 50:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def partial_ic(sig, y, ctrl):
    """Spearman parsial: korelasi residual (sig & y di-regresi-kan pada ctrl)."""
    m = np.isfinite(sig) & np.isfinite(y) & np.isfinite(ctrl)
    if m.sum() < 100:
        return float("nan"), float("nan"), int(m.sum())
    s, yy, c = sig[m], y[m], ctrl[m]
    rs = s - np.polyval(np.polyfit(c, s, 1), c)
    ry = yy - np.polyval(np.polyfit(c, yy, 1), c)
    rho, p = spearmanr(rs, ry)
    return float(rho), float(p), int(m.sum())


def bh_fdr(pvals):
    """Benjamini–Hochberg: return (qvals, n_signif)."""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return q, 0
    pv = p[ok]
    order = np.argsort(pv)
    n = len(pv)
    qv = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        val = pv[order[rank]] * n / (rank + 1)
        prev = min(prev, val)
        qv[order[rank]] = prev
    q[ok] = np.minimum(qv, 1.0)
    return q, int(np.sum(q < 0.10))


# ────────────────────────────────────────────────────────────────
# Tugas B: event study drawdown
# ────────────────────────────────────────────────────────────────

def drawdown_events(close, dates, min_dd=0.15, min_gap=60):
    """Event drawdown ≥min_dd dari puncak lokal → (onset_idx per event)."""
    peak_i, peak_v = 0, close[0]
    events, in_dd = [], False
    trough_i, trough_v = 0, close[0]
    for i in range(1, len(close)):
        if close[i] > peak_v:
            if in_dd:
                dd = 1 - trough_v / peak_v
                if dd >= min_dd:
                    events.append({"peak": int(peak_i), "trough": int(trough_i),
                                   "dd": float(dd),
                                   "peak_date": dates[peak_i], "trough_date": dates[trough_i]})
                in_dd = False
            peak_i, peak_v = i, close[i]
            in_dd = False
        else:
            if close[i] < trough_v or not in_dd:
                trough_i, trough_v = i, close[i]
            dd_now = 1 - close[i] / peak_v
            if dd_now >= 0.05:
                in_dd = True
    if in_dd:
        dd = 1 - trough_v / peak_v
        if dd >= min_dd:
            events.append({"peak": int(peak_i), "trough": int(trough_i), "dd": float(dd),
                           "peak_date": dates[peak_i], "trough_date": dates[trough_i]})
    # buang event berdekatan (tetap yang terdalam)
    out = []
    for e in sorted(events, key=lambda x: x["peak"]):
        if out and e["peak"] - out[-1]["peak"] < min_gap:
            if e["dd"] > out[-1]["dd"]:
                out[-1] = e
            continue
        out.append(e)
    return out


def event_study(sig, dates, events, pre=(5, 30), n_boot=2000, block=20, rng=None):
    rng = rng or np.random.default_rng(SEED + 2)
    lo, hi = pre
    vals, det = [], []
    for e in events:
        a, b = e["peak"] - hi, e["peak"] - lo
        if a < 0:
            continue
        w = sig[a:b + 1]
        w = w[np.isfinite(w)]
        if len(w) == 0:
            continue
        vals.append(float(np.mean(w)))
        e["pre_onset_signal"] = float(np.mean(w))
    uncond = sig[np.isfinite(sig)]
    if not vals or len(uncond) < 200:
        return {"n_events": len(vals), "mean_pre_onset": None, "uncond_mean": None,
                "p_boot": None, "events": events}
    obs = float(np.mean(vals))
    obs_rank = float(np.mean([np.mean(uncond <= v) for v in vals]))
    n = len(uncond)
    nb = int(np.ceil(n / block))
    boot_rank = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        sample = uncond[idx]
        boot_rank[b] = float(np.mean([np.mean(sample <= v) for v in vals]))
    p = float(np.mean(boot_rank >= obs_rank))
    # AUC deteksi: label = "event mulai dalam 30 hari"
    ev_start = np.zeros(len(sig), dtype=float)
    for e in events:
        a = max(0, e["peak"] - 30)
        ev_start[a:e["peak"] + 1] = 1.0
    m = np.isfinite(sig)
    auc = float(roc_auc_score(ev_start[m], sig[m])) if len(np.unique(ev_start[m])) > 1 else float("nan")
    return {"n_events": len(vals), "pre_window": [lo, hi],
            "mean_pre_onset": obs, "uncond_mean": float(np.mean(uncond)),
            "pre_onset_pctile": obs_rank, "p_boot_block": p,
            "detect_auc_event_in_30d": auc, "base_rate": float(np.mean(ev_start[m])),
            "events": events}


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    dates, close, ret = load_btc()
    n = len(ret)
    print(f"data: {dates[0]} .. {dates[-1]}  n={n} hari | window={WINDOW}")

    fwd = {}
    for h in HORIZONS:
        f = np.full(n, np.nan)
        f[:-h] = close[h:] / close[:-h] - 1.0
        fwd[h] = f
    vol21 = np.full(n, np.nan)
    for i in range(VOL_WIN, n):
        vol21[i] = float(np.std(ret[i - VOL_WIN + 1: i + 1], ddof=1))
    era_of = np.array(["pre"] * n, dtype=object)
    for name, a, b in ERAS:
        era_of[(np.array(dates) >= a) & (np.array(dates) <= b)] = name

    print("menghitung besaran rolling (point-in-time) ...")
    meas = rolling_measures(ret, window=WINDOW)
    igc = igc_composite(meas)
    print(f"  selesai dalam {time.time()-t0:.1f}s")

    sigs = {k: meas[k] for k in MEASURE_KEYS}
    sigs["igc"] = igc
    sigs["vol21_incumbent"] = vol21
    # orientasi: sinyal "rapuh tinggi"; incumbent vol tinggi = rapuh
    NAMES = ["fim_excess", "fr_drift", "te_asym", "rho1", "skew", "lam1", "igc"]

    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data": {"first": dates[0], "last": dates[-1], "n_days": n,
                 "window": WINDOW, "source": "data/binance_vision_daily.json"},
        "seed": SEED, "n_surrogates": N_SURROGATES, "surrogate_stride": SURR_STRIDE,
        "descriptives": {}, "task_A_ic": {}, "task_A_gap": {}, "task_A_era": {},
        "task_A_purged_cv": {}, "task_B_event_study": {}, "task_C_incremental": {},
        "task_N_surrogate_null": {}, "verdict": {},
    }

    for k in NAMES + ["vol21_incumbent"]:
        v = sigs[k]
        ok = np.isfinite(v)
        result["descriptives"][k] = {
            "n_valid": int(ok.sum()),
            "mean": float(np.nanmean(v)) if ok.any() else None,
            "sd": float(np.nanstd(v)) if ok.any() else None,
            "min": float(np.nanmin(v)) if ok.any() else None,
            "max": float(np.nanmax(v)) if ok.any() else None,
        }

    # ── Task A: IC + BH-FDR ─────────────────────────────────────
    print("\n[TASK A] rank-IC vs forward return")
    fam_p, fam_keys = [], []
    for k in NAMES:
        result["task_A_ic"][k] = {}
        for h in HORIZONS:
            rho, p, nn = rank_ic(sigs[k], fwd[h])
            result["task_A_ic"][k][f"{h}d"] = {"rho": rho, "p": p, "n": nn}
            fam_p.append(p); fam_keys.append((k, h))
    q, n_sig = bh_fdr(fam_p)
    for (k, h), qv in zip(fam_keys, q):
        result["task_A_ic"][k][f"{h}d"]["q_bh"] = float(qv)
    result["task_A_ic_family"] = {"n_tests": len(fam_p), "n_q_lt_0.10": n_sig}
    for k in NAMES:
        print(f"  {k:16s} " + "  ".join(
            f"{h}d rho={result['task_A_ic'][k][f'{h}d']['rho']:+.3f}"
            f"(q={result['task_A_ic'][k][f'{h}d']['q_bh']:.2f})" for h in HORIZONS))

    # ── Task A: quantile gap + bootstrap ────────────────────────
    print("\n[TASK A] gap top/bottom-20% + block bootstrap")
    for k in NAMES:
        result["task_A_gap"][k] = {}
        for h in HORIZONS:
            obs, p, sd = block_bootstrap_p(sigs[k], fwd[h])
            result["task_A_gap"][k][f"{h}d"] = {"gap_pp": obs * 100, "p_sign": p,
                                                "boot_sd_pp": sd * 100}
        print(f"  {k:16s} " + "  ".join(
            f"{h}d gap={result['task_A_gap'][k][f'{h}d']['gap_pp']:+.2f}pp"
            f"(p={result['task_A_gap'][k][f'{h}d']['p_sign']:.3f})" for h in HORIZONS))

    # ── Task A: era-split ───────────────────────────────────────
    print("\n[TASK A] era-split (30d)")
    for k in NAMES:
        result["task_A_era"][k] = {}
        for ename, a, b in ERAS:
            mask = era_of == ename
            obs, p, sd = block_bootstrap_p(np.where(mask, sigs[k], np.nan), fwd[30],
                                           n_boot=800)
            result["task_A_era"][k][ename] = {"gap_pp": obs * 100, "p_sign": p}
        e = result["task_A_era"][k]
        result["task_A_era"][k]["era2_era3_sign_consistent"] = bool(
            np.isfinite(e["era2"]["gap_pp"]) and np.isfinite(e["era3"]["gap_pp"])
            and np.sign(e["era2"]["gap_pp"]) == np.sign(e["era3"]["gap_pp"]))
        print(f"  {k:16s} era1={e['era1']['gap_pp']:+.2f} era2={e['era2']['gap_pp']:+.2f} "
              f"era3={e['era3']['gap_pp']:+.2f} konsisten={e['era2_era3_sign_consistent']}")

    # ── Task A: purged-CV ───────────────────────────────────────
    print("\n[TASK A] purged-CV/embargo AUC (label = fwd<0)")
    for k in NAMES:
        result["task_A_purged_cv"][k] = {}
        for h in HORIZONS:
            y = np.where(np.isfinite(fwd[h]), (fwd[h] < 0).astype(float), np.nan)
            auc, n_oos, P, L, n_tr = purged_cv_auc(sigs[k], y, embargo=h)
            lo, hi = bootstrap_auc_ci(P, L) if n_oos else (float("nan"), float("nan"))
            result["task_A_purged_cv"][k][f"{h}d"] = {
                "pooled_auc": auc, "n_oos": n_oos, "n_train": n_tr,
                "ci_lower": lo, "ci_upper": hi}
        print(f"  {k:16s} " + "  ".join(
            f"{h}d AUC={result['task_A_purged_cv'][k][f'{h}d']['pooled_auc']:.3f}"
            f"[{result['task_A_purged_cv'][k][f'{h}d']['ci_lower']:.3f},"
            f"{result['task_A_purged_cv'][k][f'{h}d']['ci_upper']:.3f}]" for h in HORIZONS))

    # ── Task B: event study ─────────────────────────────────────
    print("\n[TASK B] event study drawdown ≥15%")
    ev = drawdown_events(close, dates)
    result["task_B_event_study"] = {k: event_study(sigs[k], dates, [dict(e) for e in ev])
                                    for k in ["igc", "fim_excess", "vol21_incumbent"]}
    for k, r in result["task_B_event_study"].items():
        print(f"  {k:16s} n_events={r['n_events']} pre-onset={r['mean_pre_onset']} "
              f"vs uncond={r['uncond_mean']} pctile={r['pre_onset_pctile']} "
              f"p={r['p_boot_block']} AUC30d={r['detect_auc_event_in_30d']}")
    result["task_B_event_study"]["_events_shared"] = ev

    # ── Task C: inkremental atas incumbent ──────────────────────
    print("\n[TASK C] inkremental atas volatilitas 21d")
    for k in ["igc", "fim_excess", "fr_drift", "te_asym", "rho1"]:
        result["task_C_incremental"][k] = {}
        for h in HORIZONS:
            y = np.where(np.isfinite(fwd[h]), (fwd[h] < 0).astype(float), np.nan)
            # mask bersama: sinyal, incumbent, dan label semuanya valid
            common = np.isfinite(sigs[k]) & np.isfinite(vol21) & np.isfinite(y)
            a1, n1, P1, L1, _ = purged_cv_auc(sigs[k], y, embargo=h, mask=common)
            a2, n2, P2, L2, _ = purged_cv_auc(sigs[k], y, embargo=h, extra=vol21, mask=common)
            a3, n3, P3, L3, _ = purged_cv_auc(vol21, y, embargo=h, mask=common)
            lo, hi = bootstrap_auc_ci(P2, L2) if n2 else (float("nan"), float("nan"))
            dlo, dhi = (bootstrap_delta_auc(P3, L3, P2, L2)
                        if (n2 and n3 and n2 == n3) else (float("nan"), float("nan")))
            rho, p, nn = partial_ic(sigs[k], fwd[h], vol21)
            result["task_C_incremental"][k][f"{h}d"] = {
                "auc_signal_only": a1, "auc_vol_only": a3, "auc_vol_plus_signal": a2,
                "delta_auc_vs_vol": (a2 - a3) if np.isfinite(a2) and np.isfinite(a3) else None,
                "delta_auc_ci_lower": dlo, "delta_auc_ci_upper": dhi,
                "ci_lower": lo, "ci_upper": hi,
                "partial_ic_vs_vol": rho, "partial_p": p, "n": nn}
        c = result["task_C_incremental"][k]
        print(f"  {k:12s} " + "  ".join(
            f"{h}d ΔAUC={c[f'{h}d']['delta_auc_vs_vol']:+.3f}"
            f"[{c[f'{h}d']['delta_auc_ci_lower']:+.3f}] "
            f"pIC={c[f'{h}d']['partial_ic_vs_vol']:+.3f}(p={c[f'{h}d']['partial_p']:.2f})"
            for h in HORIZONS))

    # ── Task N: null surrogate IAAFT ────────────────────────────
    SURR_COMPONENTS = ["fim_excess", "fr_drift", "te_asym", "rho1", "igc"]
    print(f"\n[TASK N] null IAAFT ({N_SURROGATES} surrogate, stride={SURR_STRIDE}) ...")

    def _stats(sig_map):
        out = {}
        for k, v in sig_map.items():
            rho, _, nn = rank_ic(v, fwd[30])
            m = np.isfinite(v) & np.isfinite(fwd[30])
            gap = float("nan")
            if m.sum() > 100:
                s, f = v[m], fwd[30][m]
                hi = s >= np.quantile(s, 0.8)
                lo = s <= np.quantile(s, 0.2)
                gap = float(f[hi].mean() - f[lo].mean())
            out[k] = {"ic30": rho, "gap30": gap, "n": nn}
        return out

    # deret NYATA pada grid stride yang SAMA (apples-to-apples)
    real_m = rolling_measures(ret, window=WINDOW,
                              include=("fim", "fr", "te", "csd"), stride=SURR_STRIDE)
    real_map = {k: real_m[k] for k in ("fim_excess", "fr_drift", "te_asym", "rho1")}
    real_map["igc"] = igc_composite(real_m)
    real_stats = _stats(real_map)

    rng = np.random.default_rng(SEED + 3)
    surr = {k: [] for k in SURR_COMPONENTS}
    t1 = time.time()
    for s in range(N_SURROGATES):
        sur = iaaft(ret, n_iter=50, rng=rng)
        m_s = rolling_measures(sur, window=WINDOW,
                               include=("fim", "fr", "te", "csd"), stride=SURR_STRIDE)
        smap = {k: m_s[k] for k in ("fim_excess", "fr_drift", "te_asym", "rho1")}
        smap["igc"] = igc_composite(m_s)
        st = _stats(smap)
        for k in SURR_COMPONENTS:
            surr[k].append((st[k]["ic30"], st[k]["gap30"]))
        if (s + 1) % 10 == 0:
            print(f"   surrogate {s+1}/{N_SURROGATES} ({time.time()-t1:.0f}s)")

    null_out = {"n_surrogates": N_SURROGATES, "stride": SURR_STRIDE, "per_component": {}}
    for k in SURR_COMPONENTS:
        arr = np.asarray(surr[k], dtype=float)
        ic_s, gap_s = arr[:, 0], arr[:, 1]
        ic_ok, gap_ok = ic_s[np.isfinite(ic_s)], gap_s[np.isfinite(gap_s)]
        p_ic = float(np.mean(np.abs(ic_ok) >= abs(real_stats[k]["ic30"]))) if len(ic_ok) else float("nan")
        p_gap = float(np.mean(np.abs(gap_ok) >= abs(real_stats[k]["gap30"]))) if len(gap_ok) else float("nan")
        null_out["per_component"][k] = {
            "real_ic30": real_stats[k]["ic30"], "real_gap30_pp": real_stats[k]["gap30"] * 100,
            "n_points": real_stats[k]["n"],
            "surrogate_ic30_mean": float(np.nanmean(ic_ok)) if len(ic_ok) else None,
            "surrogate_ic30_sd": float(np.nanstd(ic_ok)) if len(ic_ok) else None,
            "p_value_ic30": p_ic,
            "surrogate_gap30_mean_pp": float(np.nanmean(gap_ok)) * 100 if len(gap_ok) else None,
            "surrogate_gap30_sd_pp": float(np.nanstd(gap_ok)) * 100 if len(gap_ok) else None,
            "p_value_gap30": p_gap,
        }
        o = null_out["per_component"][k]
        print(f"  {k:12s} IC30 real={o['real_ic30']:+.4f} vs surrogate="
              f"{o['surrogate_ic30_mean']:+.4f}±{o['surrogate_ic30_sd']:.4f} → p={p_ic:.3f}"
              f" | gap30 real={o['real_gap30_pp']:+.2f}pp vs "
              f"{o['surrogate_gap30_mean_pp']:+.2f}pp → p={p_gap:.3f}")
    result["task_N_surrogate_null"] = null_out

    # ── Verdict ─────────────────────────────────────────────────
    def _evaluate(name: str) -> dict:
        """Terapkan 4 kriteria pre-spesifikasi untuk sinyal `name`."""
        pcx = result["task_A_purged_cv"][name]
        bh = max(HORIZONS, key=lambda h: (pcx[f"{h}d"]["pooled_auc"] or 0))
        bx = pcx[f"{bh}d"]
        c1 = bool(np.isfinite(bx["pooled_auc"]) and bx["pooled_auc"] > 0.55
                  and np.isfinite(bx["ci_lower"]) and bx["ci_lower"] > 0.50)
        era = result["task_A_era"][name]
        e2, e3 = era.get("era2", {}), era.get("era3", {})
        # era-stability KETAT: tanda sama DAN signifikan (p_sign<=0.05) di era2 & era3
        # (kriteria repo: era2 AND era3 harus signifikan; era1 bull-run tidak relevan)
        c2 = bool(e2.get("gap_pp") is not None and e3.get("gap_pp") is not None
                  and np.sign(e2["gap_pp"]) == np.sign(e3["gap_pp"])
                  and e2.get("p_sign", 1) <= 0.05 and e3.get("p_sign", 1) <= 0.05)
        nx = result["task_N_surrogate_null"]["per_component"].get(name)
        if nx is not None:
            p_ic = nx["p_value_ic30"]
        else:
            p_ic = result["task_N_surrogate_null"].get("p_value_ic30")
        c3 = bool(p_ic is not None and np.isfinite(p_ic) and p_ic < 0.05)
        inc = result["task_C_incremental"].get(name, {}).get(f"{bh}d")
        c4 = bool(inc and inc["delta_auc_vs_vol"] is not None
                  and inc["delta_auc_vs_vol"] > 0
                  and np.isfinite(inc["delta_auc_ci_lower"])
                  and inc["delta_auc_ci_lower"] > 0.0)
        return {
            "best_horizon": f"{bh}d",
            "purged_cv_auc": bx["pooled_auc"], "purged_cv_ci": [bx["ci_lower"], bx["ci_upper"]],
            "era2_era3_gap_pp": [era["era2"]["gap_pp"], era["era3"]["gap_pp"]],
            "surrogate_p_ic30": p_ic,
            "delta_auc_vs_vol": (inc or {}).get("delta_auc_vs_vol"),
            "delta_auc_ci": [(inc or {}).get("delta_auc_ci_lower"),
                             (inc or {}).get("delta_auc_ci_upper")],
            "cond1_auc": c1, "cond2_era_stable": c2, "cond3_surrogate": c3,
            "cond4_incremental": c4,
            "PASS_ALL": bool(c1 and c2 and c3 and c4),
        }

    v_igc = _evaluate("igc")
    v_fim = _evaluate("fim_excess")   # SEKUNDER/eksploratif (komponen komposit)
    result["verdict"] = {
        "primary_signal": "igc",
        "primary": v_igc,
        "primary_decision": ("BLEND_CANDIDATE" if v_igc["PASS_ALL"]
                             else "NOT_BLEND (display-only / rejected)"),
        "secondary_signal": "fim_excess",
        "secondary_note": ("Komponen pre-spesifikasi (bukan hasil tuning setelah "
                           "melihat data), tetapi uji sekunder → multiplicity: "
                           "jangan klaim sebagai temuan utama tanpa replikasi."),
        "secondary": v_fim,
        "secondary_decision": ("BLEND_CANDIDATE" if v_fim["PASS_ALL"]
                               else "NOT_BLEND (display-only / rejected)"),
    }
    print("\n" + "=" * 70)
    for k, v in result["verdict"].items():
        if isinstance(v, dict):
            print(f"  [{k}]")
            for kk, vv in v.items():
                print(f"     {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

    with open(OUT, "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(f"\nDisimpan: {OUT}  (total {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
