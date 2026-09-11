#!/usr/bin/env python3
"""
vol_stress_estimator_test.py — Uji apakah estimator stress price/vol ALTERNATIF
memperkuat inti era-stable SFC (stress-gauge harga/vol), dibanding estimator
yang sekarang dipakai (realized-vol 30d).

LATAR / PERTANYAAN
    Satu-satunya edge SFC yang lolos gate era-stable adalah stress-gauge
    harga/vol (lihat docs/PROJECT_STATUS.md §WFV Stress-Gap era-stability:
    7d era2 -1.34 sig / era3 -0.69 sig; 30d -4.59 / -4.76 sig). Semua kanal
    macro-liquidity sudah DITUTUP. Maka satu-satunya arah pengembangan yang
    tersisa adalah memperkuat inti ini sendiri.

KANDIDAT (semua dari OHLC harian, point-in-time, window trailing):
    rv30         realized vol (std log-return 30d)   <- INCUMBENT
    semidev30    downside semi-deviation (sqrt(mean(min(r,0)^2)))
    parkinson30  range vol Parkinson  sqrt(mean(ln(H/L)^2)/(4 ln2))
    gk30         Garman-Klass  sqrt(mean(0.5 ln(H/L)^2 - (2ln2-1) ln(C/O)^2))
    rs30         Rogers-Satchell sqrt(mean(ln(H/C)ln(H/O)+ln(L/C)ln(L/O)))
    jump_ratio30 jump share = clip(1 - BV/RV, 0, 1), BV=(pi/2)mean(|r_t||r_{t-1}|)
    vov30        vol-of-vol = std(rolling 7d RV) atas 30d
    tail30       max |log-return| dalam 30d
    range30      mean((H-L)/C) 30d (rentang intraday relatif)
    maxdd90      1 - C/max(C,90)  (kedalaman drawdown dari puncak 90d)

DESAIN VALIDASI — TIGA PILAR, DAN BAR PER-ERA YANG DIBEDAKAN
    1. TEST A (PRIMER) — measurement validity / stress->forward-return gap.
       Top-20% vs bottom-20% dari expanding-z DI DALAM tiap era (threshold-free,
       lihat skill walk-forward-validation §4: bucket absolut pincang lintas-era).
       Polaritas benar = gap NEGATIF (stress tinggi -> return forward lebih buruk).
       Bootstrap langsung atas selisih (bukan membandingkan dua CI terpisah).
    2. TEST B (KONFIRMASI) — purged-CV/embargo (Lopez de Prado), AUC OOS
       univariat + INKREMENTAL di atas baseline [rv30, mom30] DENGAN KONTROL
       PERMUTASI (feature-block shuffle >=100x). Wajib: menambah fitur apa pun
       (bahkan noise) menaikkan AUC OOS ~0.05 (skill pitfall 44).
    3. TEST C — redundansi: Spearman vs rv30 (kalau >0.9, kandidat kemungkinan
       besar hanya representasi ulang realized-vol).

BAR PER-ERA (bukan satu verdict pooled):
    - Era yang MENENTUKAN (gate) = era3b (2024-01.., struktur pasar sekarang,
      pasca spot-ETF) dan era3a (2022-01..2024-01). era2 sebagai penguat.
    - era0 (2012-2014) / era1 (2015-2018) = DIAGNOSTIK saja: mengukur seberapa
      era-spesifik efeknya. TIDAK dipakai sebagai gate kill, karena menuntut
      tanda sama di 2013 (bull satu arah) dan 2025 (terinstitusionalisasi)
      adalah bar yang salah (skill pitfall 35).
    - Konsekuensi: satu sinyal bisa sahih di struktur sekarang walau era0/era1
      berbeda; itu dilaporkan sebagai "valid di struktur X", bukan "gagal".

CAVEAT YANG HARUS DIBACA BERSAMA HASIL
    - Vol era3b teredam (vol harian ~2.46% vs ~3.39% era3a / 4.23% 2017-20)
      -> rentang dinamis stress-gauge mengecil; gap era3b bisa lebih tipis
      bukan karena sinyal lemah tapi karena rezimnya lebih tenang.
    - Harga: Kaggle Bitstamp (2012-01..2017-08) + Binance Vision kanonik
      (2017-08-17..). Overlap sudah divalidasi (mean rel.diff 0.190%).
      Return forward dihitung dari SATU seri gabungan supaya semua sinyal
      dibandingkan pada harga yang sama.
    - sfc_pct incumbent diambil dari cache .walk_forward_validation.json
      (replay faktor tereduksi, 2014-12+) -> dipakai sebagai TOLOK UKUR inti,
      bukan sebagai replay penuh sistem live (skill pitfall 12).

DETERMINISTIK: semua bootstrap/permutasi di-seed (SEED=42).

Output: .vol_stress_estimator_test.json
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SFC_DIR)

SEED = 42
HORIZONS = [7, 30]
TAIL_Q = 0.20
N_BOOT = 2000
N_PERM = 200
N_FOLDS = 5
MIN_Z_HIST = 180          # warmup expanding-z (point-in-time)
Z_YEARS_MIN = 180

BINANCE_DAILY = os.path.join(SFC_DIR, "data", "binance_vision_daily.json")
KAGGLE_CSV = os.path.join(SFC_DIR, "data", "kaggle_bitstamp", "btcusd_1-min_data.csv")
KAGGLE_DAILY_CACHE = os.path.join(SFC_DIR, "data", "kaggle_bitstamp", "bitstamp_daily.json")
WFV_CACHE = os.path.join(SFC_DIR, ".walk_forward_validation.json")
OUT_FILE = os.path.join(SFC_DIR, ".vol_stress_estimator_test.json")

# Era boundaries. era3 DIPECAH di structural break ETF 2024-01 (skill pitfall 35).
ERAS = [
    ("era0",  "2012-01-01", "2015-01-01"),   # diagnostik (Bitstamp)
    ("era1",  "2015-01-01", "2018-01-01"),   # diagnostik (bull pra-institusional)
    ("era2",  "2018-01-01", "2022-01-01"),   # penguat
    ("era3a", "2022-01-01", "2024-01-01"),   # GATE
    ("era3b", "2024-01-01", "2099-01-01"),   # GATE (struktur sekarang)
]
GATE_ERAS = ["era3b", "era3a"]
SUPPORT_ERAS = ["era2"]
DIAG_ERAS = ["era0", "era1"]


# ────────────────────────────────────────────────────────────────
# DATA
# ────────────────────────────────────────────────────────────────
def load_kaggle_daily():
    """CSV 1-menit Bitstamp -> OHLC harian (UTC). Chunked supaya tidak OOM."""
    if os.path.exists(KAGGLE_DAILY_CACHE):
        with open(KAGGLE_DAILY_CACHE) as f:
            return json.load(f)
    if not os.path.exists(KAGGLE_CSV):
        print("[Data] Kaggle Bitstamp CSV tidak ada — era0/era1 dilewati.", file=sys.stderr)
        return {}
    print("[Data] Agregasi 1-menit Bitstamp -> harian (chunked)...", file=sys.stderr)
    parts = []
    reader = pd.read_csv(
        KAGGLE_CSV, usecols=["Timestamp", "Open", "High", "Low", "Close"],
        chunksize=2_000_000,
    )
    for ch in reader:
        ch = ch.dropna(subset=["Close"])
        ch["date"] = pd.to_datetime(ch["Timestamp"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
        g = ch.groupby("date", sort=True).agg(
            o=("Open", "first"), h=("High", "max"), l=("Low", "min"), c=("Close", "last"),
        )
        parts.append(g)
    # Gabung per chunk: file terurut naik, jadi open = first-chunk non-null, close = last.
    acc = {}
    for g in parts:
        for d, row in g.iterrows():
            cur = acc.get(d)
            if cur is None:
                acc[d] = [row["o"], row["h"], row["l"], row["c"]]
            else:
                if pd.notna(row["o"]) and (cur[0] is None or pd.isna(cur[0])):
                    cur[0] = row["o"]
                cur[1] = np.nanmax([cur[1], row["h"]])
                cur[2] = np.nanmin([cur[2], row["l"]])
                if pd.notna(row["c"]):
                    cur[3] = row["c"]
    out = {d: {"open": float(v[0]), "high": float(v[1]), "low": float(v[2]), "close": float(v[3])}
           for d, v in acc.items() if not pd.isna(v[3])}
    os.makedirs(os.path.dirname(KAGGLE_DAILY_CACHE), exist_ok=True)
    with open(KAGGLE_DAILY_CACHE, "w") as f:
        json.dump(out, f)
    print(f"[Data] Bitstamp harian: {len(out)} hari", file=sys.stderr)
    return out


def load_prices():
    """Seri harga gabungan: Bitstamp (<=2017-08-16) + Binance kanonik (>=2017-08-17)."""
    with open(BINANCE_DAILY) as f:
        bnc = json.load(f)
    bit = load_kaggle_daily()
    rows = {}
    for d, v in bit.items():
        if d < "2017-08-17":
            rows[d] = v
    for d, v in bnc.items():
        rows[d] = {"open": v["open"], "high": v["high"], "low": v["low"], "close": v["close"]}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[(df["close"] > 0) & df["high"].notna() & df["low"].notna() & df["open"].notna()]
    print(f"[Data] Panel gabungan: {len(df)} hari {df.index[0].date()} .. {df.index[-1].date()}", file=sys.stderr)
    return df


# ────────────────────────────────────────────────────────────────
# ESTIMATOR
# ────────────────────────────────────────────────────────────────
def build_signals(df):
    """Semua estimator point-in-time (hanya memakai data <= t)."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    r = np.log(c / c.shift(1))
    ln_hl = np.log(h / l)
    ln_co = np.log(c / o)
    ln_hc = np.log(h / c)
    ln_ho = np.log(h / o)
    ln_lc = np.log(l / c)
    ln_lo = np.log(l / o)

    rv_var = (r ** 2).rolling(30).mean()
    rv = np.sqrt(rv_var)
    bv = (np.pi / 2.0) * (r.abs() * r.abs().shift(1)).rolling(30).mean()
    gk_var = (0.5 * ln_hl ** 2 - (2 * np.log(2) - 1) * ln_co ** 2).rolling(30).mean()
    rs_var = (ln_hc * ln_ho + ln_lc * ln_lo).rolling(30).mean()

    sig = pd.DataFrame(index=df.index)
    sig["rv30"] = rv
    sig["semidev30"] = np.sqrt((r.clip(upper=0) ** 2).rolling(30).mean())
    sig["parkinson30"] = np.sqrt((ln_hl ** 2).rolling(30).mean() / (4 * np.log(2)))
    sig["gk30"] = np.sqrt(gk_var.clip(lower=0))
    sig["rs30"] = np.sqrt(rs_var.clip(lower=0))
    sig["jump_ratio30"] = (1.0 - bv / rv_var).clip(0.0, 1.0)
    sig["vov30"] = r.rolling(7).std().rolling(30).std()
    sig["tail30"] = r.abs().rolling(30).max()
    sig["range30"] = ((h - l) / c).rolling(30).mean()
    sig["maxdd90"] = 1.0 - c / c.rolling(90).max()
    # Berbasis PERUBAHAN vol (bukan level) — sinyal level mis-spesifikasi saat
    # rezim vol teredam (era3b vol harian 2.46% vs 4.23% di 2017-20).
    sig["dvol30"] = rv - rv.shift(30)
    sig["dvol7"] = r.rolling(7).std() - r.rolling(7).std().shift(7)
    sig["mom30"] = np.log(c / c.shift(30))

    # Expanding z (point-in-time, tanpa look-ahead)
    s = sig.copy()
    for col in sig.columns:
        m = sig[col].expanding(MIN_Z_HIST).mean()
        sd = sig[col].expanding(MIN_Z_HIST).std()
        s[col + "_z"] = (sig[col] - m) / sd.replace(0.0, np.nan)

    out = df[["close"]].copy()
    for col in sig.columns:
        out[col] = sig[col]
        out[col + "_z"] = s[col + "_z"]
    return out


STRESS_CANDIDATES = ["semidev30", "parkinson30", "gk30", "rs30",
                     "jump_ratio30", "vov30", "tail30", "range30", "maxdd90",
                     "dvol30", "dvol7"]
INCUMBENT = "rv30"
BASE_FEATURES = ["rv30", "mom30"]     # baseline harga/vol (pola crypto_flow_stress_test)
ALL_SIGNALS = [INCUMBENT] + STRESS_CANDIDATES + ["mom30"]


# ────────────────────────────────────────────────────────────────
# TEST A — gap era (top-20% vs bottom-20% dalam era)
# ────────────────────────────────────────────────────────────────
def _block_resample(x, rng, n_boot, block):
    """Moving-block bootstrap (circular): resample blok panjang `block`."""
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return x[idx.reshape(n_boot, -1)[:, :n]]


def boot_diff_ci(a, b, n_boot=N_BOOT, seed=SEED, block=1):
    """Bootstrap langsung atas selisih mean(a) - mean(b).

    block > 1 = moving-block bootstrap, wajib saat observasi bergantung
    (label forward h-hari yang tumpang tindih: 30d berurutan berbagi 29 hari,
    effective n ~= n/(h/2+1)). Return (est, lo95, hi95).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 10 or len(b) < 10:
        return None
    est = a.mean() - b.mean()
    rng = np.random.default_rng(seed)
    if block <= 1:
        ia = rng.integers(0, len(a), size=(n_boot, len(a)))
        ib = rng.integers(0, len(b), size=(n_boot, len(b)))
        d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    else:
        d = (_block_resample(a, rng, n_boot, block).mean(axis=1)
             - _block_resample(b, rng, n_boot, block).mean(axis=1))
    return float(est), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def era_gap(df, zcol, fwdcol, q=TAIL_Q, h=1):
    """Top-q vs bottom-q dari zcol DI DALAM satu era (threshold-free).

    CI memakai moving-block bootstrap dengan panjang blok = h, karena label
    forward h-hari yang berturutan saling tumpang tindih (effective n jauh
    lebih kecil dari n nominal). n_eff dilaporkan apa adanya.
    """
    sub = df[[zcol, fwdcol]].dropna()
    n = len(sub)
    if n < 60:
        return {"n": n, "insufficient": True}
    lo = sub[zcol].quantile(q)
    hi = sub[zcol].quantile(1 - q)
    stress = sub.loc[sub[zcol] >= hi, fwdcol]     # z tinggi = stress
    calm = sub.loc[sub[zcol] <= lo, fwdcol]
    res = {"n": n, "n_stress": int(len(stress)), "n_calm": int(len(calm)),
           "n_eff": int(n / (h / 2.0 + 1)),
           "mean_stress": round(float(stress.mean()), 3), "mean_calm": round(float(calm.mean()), 3)}
    ci = boot_diff_ci(stress.values, calm.values, block=max(1, int(h)))
    ci_iid = boot_diff_ci(stress.values, calm.values, block=1)
    if ci is None:
        res["insufficient"] = True
        return res
    est, lo_, hi_ = ci
    res.update(gap=round(est, 3), ci_lo=round(lo_, 3), ci_hi=round(hi_, 3),
               significant=bool(hi_ < 0 or lo_ > 0),
               correct_polarity=bool(est < 0))
    if ci_iid is not None:
        res["ci_lo_iid"] = round(ci_iid[1], 3)
        res["ci_hi_iid"] = round(ci_iid[2], 3)
        res["significant_iid"] = bool(ci_iid[2] < 0 or ci_iid[1] > 0)
    return res


def test_a(panel):
    out = {}
    for sig in ALL_SIGNALS:
        zc = sig + "_z"
        if zc not in panel.columns:
            continue
        out[sig] = {}
        for h in HORIZONS:
            fwd = f"fwd_{h}d"
            per_era = {}
            for name, d0, d1 in ERAS:
                seg = panel.loc[(panel.index >= d0) & (panel.index < d1)]
                per_era[name] = era_gap(seg, zc, fwd, h=h)
            gate_ok, gate_txt = _gate_verdict(per_era, h)
            out[sig][f"{h}d"] = {"per_era": per_era, "gate_ok": gate_ok, "gate": gate_txt}
    return out


def _gate_verdict(per_era, h):
    """Gate: era3b wajib benar-polaritas + signifikan; era3a wajib benar arah."""
    e3b = per_era.get("era3b", {})
    e3a = per_era.get("era3a", {})
    if e3b.get("insufficient") or e3a.get("insufficient"):
        return None, "data tidak cukup di era gate"
    ok = bool(e3b.get("correct_polarity") and e3b.get("significant"))
    if e3a.get("correct_polarity") is False:
        ok = False
        return ok, "era3b lolos TAPI era3a berbalik arah"
    if not ok:
        return False, "era3b tidak signifikan / arah salah"
    return True, "era3b signifikan + era3a searah"


# ────────────────────────────────────────────────────────────────
# TEST B — purged-CV/embargo + kontrol permutasi
# ────────────────────────────────────────────────────────────────
def purged_folds(n, k, embargo):
    """K fold kontigu; purge label yang tumpang tindih + embargo."""
    bounds = np.linspace(0, n, k + 1).astype(int)
    for i in range(k):
        i0, i1 = bounds[i], bounds[i + 1]
        test = np.arange(i0, i1)
        train = np.array([j for j in range(n) if j < i0 - embargo or j >= i1 + embargo])
        yield train, test


def _fit_auc(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None, 0
    sc = StandardScaler().fit(Xtr)      # fit hanya di train (anti-leakage)
    mu = LogisticRegression(max_iter=1000, C=1.0)
    mu.fit(sc.transform(Xtr), ytr)
    p = mu.predict_proba(sc.transform(Xte))[:, 1]
    return float(roc_auc_score(yte, p)), len(yte)


def purged_auc(df, feat_cols, ycol, h):
    Xall = df[feat_cols].to_numpy(dtype=float)
    yall = df[ycol].to_numpy(dtype=float)
    accs, aucs = [], []
    for tr, te in purged_folds(len(df), N_FOLDS, h):
        a, n = _fit_auc(Xall[tr], yall[tr], Xall[te], yall[te])
        if a is None:
            continue
        aucs.append(a)
        accs.append((a, n))
    if not aucs:
        return {"pooled_auc": None}
    aucs = np.array(aucs)
    ns = np.array([n for _, n in accs], dtype=float)
    pooled = float(np.average(aucs, weights=ns))
    se = float(aucs.std(ddof=1) / np.sqrt(len(aucs))) if len(aucs) > 1 else float("nan")
    return {"pooled_auc": round(pooled, 4), "folds": [round(a, 3) for a in aucs],
            "se": round(se, 4), "ci95_lo": round(pooled - 1.96 * se, 4),
            "ci95_hi": round(pooled + 1.96 * se, 4)}


def test_b(panel, n_perm=N_PERM):
    """AUC univariat + inkremental di atas baseline [rv30, mom30], dgn null permutasi."""
    out = {}
    for h in HORIZONS:
        fwd = f"fwd_{h}d"
        d = panel.dropna(subset=[f + "_z" for f in BASE_FEATURES] + [fwd]).copy()
        # baseline: target biner return negatif
        d["y"] = (d[fwd] < 0).astype(int)
        base_cols = [f + "_z" for f in BASE_FEATURES]
        base = purged_auc(d, base_cols, "y", h)
        res = {"n": int(len(d)), "base_rate_neg": round(float(d["y"].mean()), 4),
               "baseline": base, "signals": {}}
        sig_list = [INCUMBENT] + STRESS_CANDIDATES
        if "sfc_pct_z" in panel.columns:
            sig_list = sig_list + ["sfc_pct"]
        for sig in sig_list:
            col = sig + "_z"
            if col not in d.columns:
                continue
            dd = d.dropna(subset=[col])
            uni = purged_auc(dd, [col], "y", h)
            aug = purged_auc(dd, base_cols + [col], "y", h)
            delta = None
            if aug["pooled_auc"] is not None and base["pooled_auc"] is not None:
                delta = round(aug["pooled_auc"] - base["pooled_auc"], 4)
            p_perm = None
            if delta is not None and n_perm > 0:
                rng = np.random.default_rng(SEED)
                y = dd["y"].to_numpy()
                Xb = dd[base_cols].to_numpy(dtype=float)
                Xc = dd[col].to_numpy(dtype=float)
                null = []
                for _ in range(n_perm):
                    Xc_s = rng.permutation(Xc)
                    X = np.column_stack([Xb, Xc_s])
                    aucs = []
                    for tr, te in purged_folds(len(dd), N_FOLDS, h):
                        a, _ = _fit_auc(X[tr], y[tr], X[te], y[te])
                        if a is not None:
                            aucs.append(a)
                    if aucs:
                        null.append(float(np.mean(aucs)))
                null = np.array(null)
                p_perm = round(float((null >= base["pooled_auc"] + delta).mean()), 4)
            res["signals"][sig] = {"univariate": uni, "augmented": aug,
                                   "delta_auc": delta, "perm_p": p_perm}
        out[f"{h}d"] = res
    return out


# ────────────────────────────────────────────────────────────────
# TEST C — redundansi
# ────────────────────────────────────────────────────────────────
def test_c(panel):
    from scipy.stats import spearmanr
    out = {}
    sub = panel.dropna(subset=[INCUMBENT])
    for sig in STRESS_CANDIDATES:
        s = sub[[INCUMBENT, sig]].dropna()
        if len(s) < 100:
            continue
        rho = float(spearmanr(s[INCUMBENT], s[sig]).statistic)
        out[sig] = round(rho, 4)
    return out


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    quick = "--quick" in sys.argv
    n_perm = 20 if quick else N_PERM

    df = load_prices()
    sig = build_signals(df)

    for h in HORIZONS:
        sig[f"fwd_{h}d"] = (sig["close"].shift(-h) / sig["close"] - 1.0) * 100.0

    # Era assignment
    sig["era"] = "na"
    for name, d0, d1 in ERAS:
        sig.loc[(sig.index >= d0) & (sig.index < d1), "era"] = name

    # Tolok ukur: sfc_pct incumbent dari cache WFV (replay), dilaporkan di Test A
    bench = None
    if os.path.exists(WFV_CACHE):
        with open(WFV_CACHE) as f:
            wfv = json.load(f)
        bs = pd.DataFrame(wfv)
        bs["date"] = pd.to_datetime(bs["date"])
        bs = bs.set_index("date")[["sfc_pct"]]
        sig = sig.join(bs, how="left")
        # sfc_pct sudah 0-100 "stress tinggi = tinggi"; buat z expanding sebanding
        zc = (sig["sfc_pct"] - sig["sfc_pct"].expanding(MIN_Z_HIST).mean()) / \
             sig["sfc_pct"].expanding(MIN_Z_HIST).std()
        sig["sfc_pct_z"] = zc
        bench = "sfc_pct"

    print(f"[Run] panel {len(sig)} hari, {len(ALL_SIGNALS)} sinyal, perm={n_perm}", file=sys.stderr)
    res = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "n_perm": n_perm,
        "panel": {"n_days": int(len(sig)),
                  "start": str(sig.index[0].date()), "end": str(sig.index[-1].date())},
        "eras": [{"name": n, "start": a, "end": b,
                  "role": "gate" if n in GATE_ERAS else "support" if n in SUPPORT_ERAS else "diagnostic",
                  "n": int((sig["era"] == n).sum())} for n, a, b in ERAS],
        "test_a_era_gaps": test_a(sig),
        "test_b_purged_cv": test_b(sig, n_perm=n_perm),
        "test_c_redundancy_vs_rv30": test_c(sig),
    }
    if bench:
        res["benchmark_incumbent_sfc_pct"] = {
            f"{h}d": {n: era_gap(sig.loc[sig["era"] == n], "sfc_pct_z", f"fwd_{h}d", h=h)
                      for n, _, _ in ERAS} for h in HORIZONS}

    with open(OUT_FILE, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[Run] selesai {time.time()-t0:.1f}s -> {OUT_FILE}", file=sys.stderr)
    _print_summary(res)


def _print_summary(res):
    print("\n=== TEST A — gap era (top20% vs bottom20%), polaritas benar = negatif ===")
    for sig, hs in res["test_a_era_gaps"].items():
        for h, d in hs.items():
            cells = []
            for name, _, _ in ERAS:
                g = d["per_era"].get(name, {})
                if g.get("insufficient"):
                    cells.append(f"{name}=n/a")
                else:
                    if g.get("significant"):
                        mark = "*"          # signifikan dgn block-bootstrap (blok=h)
                    elif g.get("significant_iid"):
                        mark = "~"          # hanya signifikan kalau iid (overlap diabaikan)
                    else:
                        mark = ""
                    cells.append(f"{name}={g['gap']:+.2f}{mark}")
            print(f"  {sig:14s} {h:>3s}  " + "  ".join(cells) + f"   [{d['gate']}]")
    print("\n=== TEST B — purged-CV AUC (univariat | delta atas baseline | perm p) ===")
    for h, d in res["test_b_purged_cv"].items():
        b = d["baseline"]
        print(f"  baseline({h}): AUC {b.get('pooled_auc')} n={d['n']} base_rate={d['base_rate_neg']}")
        for sig, v in d["signals"].items():
            u = v["univariate"].get("pooled_auc")
            print(f"    {sig:14s} uni={u}  dAUC={v['delta_auc']}  perm_p={v['perm_p']}")
    print("\n=== TEST C — Spearman vs rv30 ===")
    for k, v in res["test_c_redundancy_vs_rv30"].items():
        print(f"  {k:14s} {v:+.3f}")


if __name__ == "__main__":
    main()
