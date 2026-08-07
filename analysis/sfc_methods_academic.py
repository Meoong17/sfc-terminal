#!/usr/bin/env python3
"""
sfc_methods_academic.py — Sketsa implementasi metode akademik untuk SFC.

Referensi teori & sumber: docs/ACADEMIC_METHODS.md (2026-08-07).

Modul ini SELF-CONTAINED (hanya numpy) dan memberikan implementasi referensi yang
BISA diintegrasikan ke collect.py. Tiap fungsi mencatat di mana integrasinya
(baris/komponen collect.py). SEMUA fungsi sengaja dibuat murni & deterministik
agar mudah diuji. Ini SKETSA — belum diaktifkan di pipeline, kecuali dinyatakan.

Daftar:
  A1 ciss_composite_confidence(...)   -> CISS-style, ganti composite_confidence (collect.py ~3502)
  A2 ofr_zscore_weights(...)          -> OFR-style bobot co-movement untuk GLF/SLI/MPI
  C1 evt_var_es(...)                  -> EVT-POT VaR/ES, ganti m11_var / prob_crash
  D1 brier_decompose(...)             -> gate kalibrasi (Brier = REL + RES - UNC)
  D2 calibration_gate(...)            -> keputusan terima/tolak faktor baru
  F1 purged_kfold_split(...)          -> purged + embargo CV (ganti WFV single-path)

Jalankan test:  python3 analysis/sfc_methods_academic.py
"""
from __future__ import annotations

import numpy as np

# ============================================================================
# A1. CISS-style composite confidence (Hollo, Kremer, Lo Duca 2012)
# ----------------------------------------------------------------------------
# Sekarang (collect.py ~3502):
#   composite_confidence = macro_confidence * (1 - execution_risk)
# Itu perkalian linear: tidak menangkap bahwa stres yang MENYEBAR antar sub-faktor
# lebih berbahaya daripada stres di satu faktor saja.
#
# CISS menambahkan istilah korelasi: ketika macro_confidence TURUN dan
# execution_risk NAIK bersamaan (berkorelasi negatif/co-move), confidence harus
# diturunkan ekstra. Bobot korelasi time-varying.
# ============================================================================
def ciss_composite_confidence(
    macro_confidence: float,
    execution_risk: float,
    method_agreement: float,        # 0..1, seberapa method setuju
    recent_corr: float = 0.0,       # korelasi rolling (macro vs exec_risk), dari history
) -> float:
    """
    composite = macro * (1 - exec_risk)  DITAMBAH penalti korelasi ala CISS.

    CISS core: bobot agregasi naik saat sub-indeks co-move. Di sini diterapkan
    secara ringkas: jika macro lemah DAN exec_risk tinggi DAN korelasi
    (co-movement) positif, confidence dikoreksi turun; jika method_agreement
    tinggi, sedikit kompensasi (kesepakatan menambah keandalan).

    Parameters
    ----------
    recent_corr : korelasi rolling antara macro_confidence & (1-exec_risk);
        di pipeline dihitung dari history (lihat A2), default 0 = agnostik.
    """
    base = max(0.0, min(macro_confidence * (1.0 - execution_risk), 1.0))

    # Faktor penyebaran stres (CISS flavour): macro lemah & exec tinggi
    stress_spread = (1.0 - macro_confidence) * execution_risk

    # Koreksi korelasi: koefisien [0,1]; bila co-move kuat, spread lebih berbobot
    corr_penalty = 0.35 * (0.5 + 0.5 * max(0.0, min(recent_corr, 1.0)))

    # Kompensasi kesepakatan method (reliability): agreement tinggi -> +bobot
    agree_boost = 0.05 * method_agreement

    adjusted = base - corr_penalty * stress_spread + agree_boost * base
    return float(max(0.0, min(adjusted, 0.95)))


# ============================================================================
# A2. OFR-style bobot co-movement (Monin 2019) — bobot dari data, bukan tebak
# ----------------------------------------------------------------------------
# Sekarang bobot GLF/SLI/MPI manual (0.3/0.15/0.04..., 0.1-0.2...). OFR menurunkan
# bobot dari kovariansi: variabel yang bergerak bersama saat episode stress dapat
# bobot lebih. Implementasi ringkas: bobot = |kontribusi korelasi| dinormalisasi,
# dengan korelasi dihitung terhadap sinyal stres (mis. SFC) dari history.
# ============================================================================
def ofr_zscore_weights(
    X: np.ndarray,                 # (n_obs, n_vars) nilai mentah tiap komponen
    stress_ref: np.ndarray | None = None,  # (n_obs,) referensi stres (mis. SFC)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Z-score tiap variabel lalu hitung bobot co-movement terhadap stres.

    Returns
    -------
    (z, w) : z-score matrix (n_obs, n_vars) dan bobot (n_vars,) yg jumlahnya 1.
    w_i  ∝ |corr(z_i, stress_ref)| bila stress_ref diberikan; tanpa stress_ref,
    w_i  ∝ sum_j |corr(z_i, z_j)| (seberapa variabel-i ikut pola bersama).
    """
    X = np.asarray(X, dtype=float)
    n_vars = X.shape[1]
    if n_vars == 0:
        return X, np.array([])

    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    z = (X - mu) / sd

    if stress_ref is not None:
        s = np.asarray(stress_ref, dtype=float)
        s_sd = s.std()
        s_sd = s_sd if s_sd > 0 else 1.0
        refz = (s - s.mean()) / s_sd
        corr = np.clip(np.corrcoef(z.T, refz)[:-1, -1], -1, 1)
    else:
        c = np.corrcoef(z.T)
        corr = np.sum(np.abs(np.triu(c, 1)), axis=0) + \
               np.sum(np.abs(np.tril(c, -1)), axis=0)

    w = np.abs(corr)
    total = w.sum()
    if total <= 0:
        w = np.ones(n_vars) / n_vars
    else:
        w = w / total
    return z, w


# ============================================================================
# C1. EVT-POT Value-at-Risk / Expected Shortfall (Pickands 1975; Balkema-de Haan)
# ----------------------------------------------------------------------------
# Sekarang m11_var/es_95 berbasis normal (underestimate ekor). EVT-POT pasang
# Generalized Pareto pada ekses di atas threshold. Kembalikan juga skor stress
# untuk m11 (0-1, tinggi = ekor gemuk).
# ============================================================================
def _gpd_fit(excess: np.ndarray) -> tuple[float, float]:
    """MLE (sederhana) untuk GPD: P(X > u + x | X > u) = (1 + xi x / beta)^(-1/xi)."""
    e = np.asarray(excess, dtype=float)
    e = e[e > 0]
    if len(e) < 5:
        return 0.0, 1.0
    x = e
    # Hill-type init lalu 1 iterasi Newton untuk xi
    x_sorted = np.sort(x)
    k = max(1, int(0.05 * len(x)))
    xi = np.log(x_sorted[-k] / x_sorted[-1]) / max(k, 1)
    xi = max(-0.5, min(0.8, xi))
    beta = np.mean(x) * (1.0 - xi)
    beta = max(beta, 1e-9)
    return float(xi), float(beta)


def evt_var_es(
    returns: np.ndarray,            # (n,) return historis
    q: float = 0.95,
    threshold_q: float = 0.90,      # kuantil threshold u untuk POT
) -> tuple[float, float, float]:
    """
    VaR_q & ES_q berbasis EVT (Peaks-Over-Threshold).

    Returns
    -------
    (var, es, stress) : var/es berupa besar kerugian POSITIF (mis. 0.04 = rugi 4%),
    dan skor stress 0..1 (1 = ekor sangat gemuk / krisis). Dipakai untuk m11_var,
    m11_es, prob_crash.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 20:
        return 0.0, 0.0, 0.5
    # kerja pada return positif (kerugian) — ambil -r
    loss = -r
    u = np.quantile(loss, threshold_q)
    excess = loss[loss > u] - u
    n_u = len(excess)
    if n_u < 5:
        # fallback normal
        s = loss.std()
        z = {0.95: 1.645, 0.99: 2.326}.get(q, 1.645)
        var = u + z * s
        es = u + (loss[loss > u].mean() if (loss > u).any() else var)
        return float(var), float(es), 0.3

    xi, beta = _gpd_fit(excess)
    prob = 1.0 - q
    # VaR_q = u + beta/xi * (( (n/N_u)*prob )^(-xi) - 1),  xi != 0
    if abs(xi) < 1e-9:
        var = u + beta * np.log(n / n_u * prob)
    else:
        var = u + (beta / xi) * (((n / n_u) * prob) ** (-xi) - 1.0)
    # ES_q = (VaR_q + beta - xi*u)/(1-xi)
    es = (var + beta - xi * u) / max((1.0 - xi), 1e-9)

    # skor stress: ekor gemuk (xi besar) & rasio ekses (n_u/n) tinggi
    stress = float(max(0.0, min(0.05 + 0.5 * xi + 0.45 * (n_u / n), 1.0)))
    return float(var), float(es), stress


# ============================================================================
# D1. Brier decomposition (Murphy 1973) — gate kalibrasi
# ----------------------------------------------------------------------------
# Brier = Reliability + Resolution - Uncertainty. Untuk menilai kualitas
# probabilitas (mis. XGBoost drop-6h) secara proper, bukan confidence heuristic.
# ============================================================================
def brier_decompose(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10):
    """
    Returns dict {brier, reliability, resolution, uncertainty, n}.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float).clip(0, 1)
    n = len(y)
    brier = float(np.mean((p - y) ** 2))

    o_bar = y.mean()
    uncertainty = o_bar * (1.0 - o_bar)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(p, bins[1:-1])  # 0..n_bins-1
    reliability = 0.0
    resolution = 0.0
    for b in range(n_bins):
        m = bin_idx == b
        nk = m.sum()
        if nk == 0:
            continue
        ok = y[m].mean()
        pk = p[m].mean()
        reliability += nk * (pk - ok) ** 2
        resolution += nk * (ok - o_bar) ** 2
    reliability /= n
    resolution /= n
    return {
        "brier": brier,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "n": n,
    }


# ============================================================================
# D2. Gate kalibrasi — keputusan terima/tolak faktor baru (integra dgn walk-forward)
# ----------------------------------------------------------------------------
def calibration_gate(
    y_oos: np.ndarray, p_oos: np.ndarray,
    min_resolution: float = 0.0, max_reliability: float = 0.05,
) -> dict:
    """
    Terima faktor baru HANYA jika resolution > 0 (ada daya diskriminasi) dan
    reliability kecil (kalibrasi dekat diagonal). Ini mencegah faktor
    tak-terkalibrasi masuk ke sinyal efektif — kebijakan standing Meong.
    """
    d = brier_decompose(y_oos, p_oos)
    # resolution: variance antar bin yang dijelaskan; naive = 0 -> tak diskriminatif
    passed = d["resolution"] > min_resolution and d["reliability"] < max_reliability
    d["passed"] = bool(passed)
    return d


# ============================================================================
# F1. Purged k-fold split dengan embargo (Lopez de Prado 2018)
# ----------------------------------------------------------------------------
# Ganti WFV single-path: bikin banyak fold dengan purge + embargo agar label
# yang bergantung masa depan tidak bocor antar train/test. Tiap fold -> satu metrik
# (AUC/Sharpe); kumpulan fold -> CI, bukan satu angka.
# ============================================================================
def purged_kfold_split(
    n: int,
    n_folds: int = 5,
    purge: int = 1,      # baris yg dibuang dari train karena overlap label test
    embargo: int = 2,    # gap tambahan setelah test
):
    """
    Yield (train_idx, test_idx) untuk tiap fold. Test = blok kontigu; train =
    semua sebelum test MINUS purge terakhir, MINUS embargo. (Sederhana, urut waktu;
    untuk purged CV penuh dengan banyak path lihat Combinatorial Purged CV.)
    """
    bounds = np.linspace(0, n, n_folds + 1).astype(int)
    for i in range(n_folds):
        t0, t1 = bounds[i], bounds[i + 1]
        test = np.arange(t0, t1)
        # train: index sebelum test, buang purge baris terakhir & embargo setelah
        train_end = t0 - purge
        train = np.arange(0, train_end)
        # embargo: jangan pakai train yang posisinya 'dekat' setelah test sebelumnya
        if i > 0:
            train = train[train < (bounds[i - 1] + 1 + embargo)]
        if len(train) == 0 or len(test) == 0:
            continue
        yield train, test


# ============================================================================
# TEST
# ============================================================================
def _test():
    rng = np.random.default_rng(0)

    # A1
    c_hi = ciss_composite_confidence(0.8, 0.1, 0.7, 0.2)
    c_spread = ciss_composite_confidence(0.4, 0.6, 0.3, 0.9)
    assert 0 <= c_hi <= 0.95 and 0 <= c_spread <= 0.95
    # makro tenang+exec rendah harus lebih tinggi dari makro lemah+exec tinggi
    assert c_hi > c_spread, (c_hi, c_spread)

    # A2
    X = np.column_stack([np.sin(np.linspace(0, 20, 200)) + 2,
                         np.cos(np.linspace(0, 20, 200)) + 2,
                         np.random.default_rng(1).normal(size=200)])
    z, w = ofr_zscore_weights(X, stress_ref=X[:, 0])
    assert abs(w.sum() - 1) < 1e-9 and len(w) == 3

    # C1
    r = rng.normal(0, 0.02, 500)
    r = np.append(r, [-0.08, -0.11, -0.09, -0.13])  # ekor gemuk
    var, es, stress = evt_var_es(r, 0.95, 0.90)
    assert 0.0 <= var <= es, (var, es)   # ES selalu >= VaR (ekor lebih ekstrem)
    assert 0 <= stress <= 1

    # D1/D2
    y = rng.binomial(1, 0.1, 2000).astype(float)
    p_bad = np.clip(np.abs(rng.normal(size=2000)), 0, 1)  # tak terkalibrasi
    g_bad = calibration_gate(y, p_bad)
    assert g_bad["passed"] is False or g_bad["brier"] >= 0.05

    # F1
    folds = list(purged_kfold_split(100, 5, purge=2, embargo=3))
    assert len(folds) >= 2
    for tr, te in folds:
        assert max(tr, default=-1) < min(te)

    print("SEMUA TEST PASS (A1,A2,C1,D1,D2,F1)")


if __name__ == "__main__":
    _test()
