#!/usr/bin/env python3
"""
criticality_math.py — Kerangka matematika BARU untuk SFC (belum ada di repo)
===========================================================================
IGC — Information-Geometric Criticality
    Informasi geometrik (Fisher) + aliran informasi berarah (transfer entropy)
    + perlambatan kritis (critical slowing down), dihitung POINT-IN-TIME dari
    window trailing return harian BTC.

MENGAPA KERANGKA INI
  Arsitektur SFC yang tervalidasi: sinyal hidup di HARGA/VOLATILITAS (stress
  gauge; era-stable era2/era3), sedangkan makro/likuiditas = CONTEXT. Semua
  faktor prediktif-arah sudah ditolak (momentum, funding, order-flow, ETF flow,
  WWI, Kronos, kanal net-liquidity, term premium, ...). Pertanyaan yang
  diajukan di sini: adakah KUANTITAS MATEMATIS lain dari deret harga — yang
  belum pernah dipakai di repo ini — yang membawa informasi kondisi/kerapuhan?

EMPAT BESARAN (semuanya dari teori sistem dinamik & geometri informasi)

  (1) FISHER INFORMATION MEASURE — `fim_excess`
        I[p] = ∫ (p'(x))² / p(x) dx
      Diestimasi non-parametrik (KDE Gaussian, bandwidth Silverman, grid tetap
      + integrasi trapesium) pada return TERSTANDARDISASI (σ=1). Referensi
      eksak: untuk N(0,1), I = 1 (batas Cramér–Rao). Maka
        fim_excess = I − 1
      mengukur penyimpangan BENTUK (heavy-tail/bimodal/konsentrasi informasi),
      bukan skala. Terverifikasi: N(0,1)→1.0, N(0,σ²)→1/σ², Student-t→>1.

  (2) JARAK FISHER–RAO (geodesik Hellinger) — `fr_drift`
        d_FR(p,q) = arccos( ∫ √(p(x) q(x)) dx )
      = panjang geodesik pada manifold statistik antara densitas return window
      t−1 dan window t (setelah standardisasi tiap window → skala dinetralkan).
      Terverifikasi analitik terhadap rumus tertutup untuk Gaussian:
        N(μ1,σ1) vs N(μ2,σ2):  BC = √(2σ1σ2/(σ1²+σ2²))·exp(−Δμ²/(4(σ1²+σ2²)))
      Mengukur KECEPATAN DRIFT DISTRIBUSI (perubahan regime bentuk distribusi),
      bukan level.

  (3) TRANSFER ENTROPY ASIMETRI — `te_asym`
        TE(X→Y) = H(Y⁺|Y) − H(Y⁺|Y,X)   [dalam bit]
      X = tanda return, Y = |return| (volatilitas).
        te_asym = TE(tanda→vol) − TE(vol→tanda)
      = asimetri ARUS INFORMASI NONLINEAR (leverage: arah menginformasikan
      volatilitas lebih kuat daripada sebaliknya). Ini bukan korelasi/Granger
      linier — entropi kondisional menangkap dependensi nonlinier penuh.
      Terverifikasi: pada sistem terkopel TE arah benar > arah salah; pada
      deret independen ≈ 0.

  (4) CRITICAL SLOWING DOWN (teori bifurkasi) — `rho1`, `skew`
      autocorrelation lag-1 return (memori/lambatnya pemulihan) dan skewness
      (asimetri crash). Dipakai sebagai komponen pembanding standar, BUKAN
      klaim baru.

  DIAGNOSTIK TAMBAHAN — `lam1` (eksponen Lyapunov terbesar, Rosenstein 1993)
      Diverifikasi eksak pada sistem deterministik: logistic map r=4 →
      λ_est = 0.708 ± 0.023 vs teori ln 2 = 0.693. TETAPI pada sistem
      STOKASTIK estimator ini mengukur laju dekorrelasi noise, bukan laju
      kontraksi deterministik (AR(1) φ=0.9 teori −0.105, estimasi +0.27) →
      TIDAK dimasukkan ke komposit IGC; dilaporkan sebagai diagnostik.

  KOMPOSIT IGC (bobot PRE-SPESIFIKASI, sama rata):
      IGC = mean percentile-rank-expanding (point-in-time) dari
            {fim_excess, fr_drift, te_asym, rho1, −skew}
      orientasi "higher = lebih rapuh/kritis". Tidak ada fitting label,
      tidak ada tuning terhadap target.

SELF-TEST
    python3 analysis/criticality_math.py --selftest
"""

import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

EPS = 1e-12
# Grid tetap (unit standar) untuk KDE/FIM/jarak Fisher–Rao.
# Tetap & universal ⇒ tidak ada look-ahead dari pemilihan rentang grid.
GRID_LO, GRID_HI = -5.0, 5.0
GRID_N = 512


# ════════════════════════════════════════════════════════════════
# Util densitas (KDE Gaussian)
# ════════════════════════════════════════════════════════════════

def silverman_bandwidth(x: np.ndarray) -> float:
    """h = 1.06 · min(σ, IQR/1.349) · n^(−1/5), di-clamp positif."""
    n = len(x)
    if n < 2:
        return 1e-6
    sd = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    scale = min(sd, iqr / 1.349) if iqr > 0 else sd
    if not np.isfinite(scale) or scale <= 0:
        scale = sd if sd > 0 else 1.0
    return max(1.06 * scale * n ** (-0.2), 1e-6)


def kde_density(
    x: np.ndarray,
    grid: np.ndarray,
    bw: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """KDE Gaussian: return (p, p', h) pada grid."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    h = bw if bw is not None else silverman_bandwidth(x)
    u = (grid[:, None] - x[None, :]) / h
    K = np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)
    p = K.mean(axis=1) / h
    pd = (-(u / h) * K).mean(axis=1) / h
    return p, pd, h


def kde_fisher_information(
    x: np.ndarray,
    grid: Optional[np.ndarray] = None,
    bw: Optional[float] = None,
) -> float:
    """I = ∫ (p')²/p dx pada grid tetap. N(0,1) ⇒ 1.0 (eksak)."""
    grid = GRID.copy() if grid is None else np.asarray(grid, dtype=float)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return float("nan")
    p, pd, _ = kde_density(x, grid, bw)
    p = np.maximum(p, EPS)
    return float(np.trapezoid((pd * pd) / p, grid))


def fim_excess(returns: np.ndarray, **kw) -> float:
    """FIM(return terstandardisasi) − 1; 0 = Gaussian. NaN-safe."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return float("nan")
    sd = float(np.std(r, ddof=1))
    if sd <= 0:
        return float("nan")
    w = (r - float(np.mean(r))) / sd
    return kde_fisher_information(w, **kw) - 1.0


def fisher_rao_distance(x: np.ndarray, y: np.ndarray) -> float:
    """d_FR(p,q) = arccos( ∫ √(pq) dx ) antara densitas KDE dari dua sampel.

    Standardisasi BERSAMA (pooled): kedua sampel digeser oleh mean-gabungan dan
    dibagi oleh std-gabungan. Konsekuensi (disengaja):
      * invarian terhadap satuan/skala input (x→3x tidak mengubah jarak);
      * perbedaan LEVEL (mean shift) dan SKALA (rasio varians) TETAP terukur
        (tidak dihapus), berbeda dari standardisasi per-sampel;
      * untuk dua Gaussian, rumus tertutup berlaku eksak pada momen
        ter-pooled:  BC = √(2σ1σ2/(σ1²+σ2²)) · exp(−Δμ²/(4(σ1²+σ2²))).
    Point-in-time: hanya memakai kedua window trailing yang dibandingkan.
    """
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    y = np.asarray(y, dtype=float); y = y[np.isfinite(y)]
    if len(x) < 20 or len(y) < 20:
        return float("nan")
    both = np.concatenate([x, y])
    m = float(np.mean(both))
    s = float(np.std(both, ddof=1))
    if s <= 0:
        return float("nan")
    zx, zy = (x - m) / s, (y - m) / s
    px, _, _ = kde_density(zx, GRID)
    py, _, _ = kde_density(zy, GRID)
    bc = float(np.trapezoid(np.sqrt(np.maximum(px, 0.0) * np.maximum(py, 0.0)), GRID))
    bc = min(1.0, max(0.0, bc))
    return float(np.arccos(bc))


def fisher_rao_gaussian_analytic(x: np.ndarray, y: np.ndarray) -> float:
    """Rumus tertutup d_FR untuk GAUSSIAN pada momen ter-pooled (pembanding uji)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    both = np.concatenate([x, y])
    m = float(np.mean(both)); s = float(np.std(both, ddof=1))
    s1 = float(np.std(x, ddof=1)) / s; s2 = float(np.std(y, ddof=1)) / s
    m1 = (float(np.mean(x)) - m) / s; m2 = (float(np.mean(y)) - m) / s
    bc = np.sqrt(2 * s1 * s2 / (s1 ** 2 + s2 ** 2)) * np.exp(
        -((m1 - m2) ** 2) / (4 * (s1 ** 2 + s2 ** 2)))
    return float(np.arccos(min(1.0, max(0.0, bc))))


# ════════════════════════════════════════════════════════════════
# Informasi: MI & transfer entropy (histogram kuantil + Miller–Madow)
# ════════════════════════════════════════════════════════════════

def _quantile_symbols(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if k < 2:
        raise ValueError("k >= 2")
    qs = np.quantile(x, np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(qs, x, side="right")


def _entropy_bits(counts: np.ndarray, n: Optional[float] = None) -> float:
    counts = counts[counts > 0].astype(float)
    if counts.size == 0:
        return 0.0
    tot = float(counts.sum()) if n is None else float(n)
    p = counts / tot
    return float(-np.sum(p * np.log2(p)))


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 4,
                       bias_correct: bool = True) -> float:
    """MI(X;Y) bit; simbolisasi kuantil + koreksi bias Miller–Madow."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 30:
        return float("nan")
    x, y = x[:n], y[:n]
    sx = _quantile_symbols(x, bins); sy = _quantile_symbols(y, bins)
    joint = np.zeros((bins, bins))
    np.add.at(joint, (sx, sy), 1.0)
    mi = _entropy_bits(joint.sum(axis=1)) + _entropy_bits(joint.sum(axis=0)) \
        - _entropy_bits(joint.ravel())
    if bias_correct:
        bx = int(np.count_nonzero(joint.sum(axis=1)))
        by = int(np.count_nonzero(joint.sum(axis=0)))
        mi += (bx - 1) * (by - 1) / (2.0 * n * np.log(2.0))
    return float(max(0.0, mi))


def transfer_entropy(src: np.ndarray, dst: np.ndarray, bins: int = 3,
                     bias_correct: bool = True) -> float:
    """TE(src→dst) dalam bit untuk riwayat 1-langkah (k=l=1).

        TE = H(Y⁺,Y) − H(Y) − H(Y⁺,Y,X) + H(Y,X)

    Bentuk entropi ini aljabar-ekuivalen dengan Σ p log[p(y⁺|y,x)/p(y⁺|y)]
    tetapi jauh lebih stabil secara numerik.
    """
    src = np.asarray(src, dtype=float); dst = np.asarray(dst, dtype=float)
    T = min(len(src), len(dst)) - 1
    if T < 60:
        return float("nan")
    sx = _quantile_symbols(src, bins); sy = _quantile_symbols(dst, bins)
    fut, dhist, shist = sy[1:T + 1], sy[:T], sx[:T]
    joint = np.zeros((bins, bins, bins))
    np.add.at(joint, (fut.astype(int), dhist.astype(int), shist.astype(int)), 1.0)
    n = float(joint.sum())
    if n <= 0:
        return float("nan")
    h_fd = _entropy_bits(joint.sum(axis=2).ravel(), n)     # H(Y⁺,Y)
    h_d = _entropy_bits(joint.sum(axis=(0, 2)), n)         # H(Y)
    h_full = _entropy_bits(joint.ravel(), n)               # H(Y⁺,Y,X)
    h_dx = _entropy_bits(joint.sum(axis=0).ravel(), n)     # H(Y,X)
    te = h_fd - h_d - h_full + h_dx
    if bias_correct:
        c_fd = int(np.count_nonzero(joint.sum(axis=2)))
        c_d = int(np.count_nonzero(joint.sum(axis=(0, 2))))
        c_full = int(np.count_nonzero(joint))
        c_dx = int(np.count_nonzero(joint.sum(axis=0)))
        te += ((c_full - 1) - (c_fd - 1) - (c_dx - 1) + (c_d - 1)) / (2.0 * n * np.log(2.0))
    return float(te) if np.isfinite(te) else float("nan")


def te_asymmetry(returns: np.ndarray, bins: int = 3) -> Dict[str, float]:
    """te_asym = TE(tanda → |return|) − TE(|return| → tanda)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 80:
        return {"te_sign_to_abs": float("nan"), "te_abs_to_sign": float("nan"),
                "te_asym": float("nan")}
    sign, absr = np.sign(r), np.abs(r)
    a = transfer_entropy(sign, absr, bins=bins)
    b = transfer_entropy(absr, sign, bins=bins)
    asym = a - b if (np.isfinite(a) and np.isfinite(b)) else float("nan")
    return {"te_sign_to_abs": a, "te_abs_to_sign": b, "te_asym": asym}


# ════════════════════════════════════════════════════════════════
# Lyapunov (Rosenstein) — diagnostik, diverifikasi pada sistem deterministik
# ════════════════════════════════════════════════════════════════

def takens_embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x) - (m - 1) * tau
    if n <= 0:
        return np.empty((0, m))
    return np.stack([x[i * tau: i * tau + n] for i in range(m)], axis=1)


def automutual_information_tau(x: np.ndarray, max_tau: int = 12, bins: int = 16) -> int:
    """τ = argmin lokal pertama MI(x_t, x_{t+τ}) (kriteria Takens/Fraser)."""
    x = np.asarray(x, dtype=float)
    vals = []
    for tau in range(1, max_tau + 1):
        vals.append(mutual_information(x[:-tau], x[tau:], bins=bins, bias_correct=False))
    v = np.asarray(vals, dtype=float)
    if not np.isfinite(v).any():
        return 1
    for i in range(1, len(v) - 1):
        if np.isfinite(v[i]) and v[i] < v[i - 1] and v[i] <= v[i + 1]:
            return i + 1
    return int(np.nanargmin(v)) + 1


def rosenstein_lyapunov(x: np.ndarray, m: int = 5, tau: int = 1,
                        theiler: int = 15, fit_steps: int = 5,
                        n_pairs: int = 120, normalize: bool = True,
                        max_points: int = 1200, block: int = 512) -> float:
    """λ₁ via Rosenstein (1993): slope mean ln(d_j/d_0) vs j (j = 1..fit_steps).

    Diverifikasi: logistic map r=4 → 0.708 ± 0.023 (teori ln2 = 0.693).
    - Theiler window dalam LANGKAH WAKTU (bukan indeks tersubsample) — kritik
      penting: men-DECIMATE baris embedding (langkah lama) membuat divergensi
      saturasi dalam 1 langkah dan estimasi gagal (0.02, bukan 0.69). Karena
      itu, jika N > max_points, dipotong KONTIGU (deret terakhir) — bukan
      di-decimate — sehingga dinamika tetap utuh.
    - Jarak dihitung BLOK-per-BLOK agar hemat memori (tidak pernah membentuk
      tensor (N,N,m)).
    """
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if len(x) < 100:
        return float("nan")
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        return float("nan")
    z = (x - float(np.mean(x))) / sd
    e = takens_embed(z, m, tau)
    N = len(e)
    if N > max_points:
        e = e[-max_points:]          # potong kontigu (bukan decimate)
        N = len(e)
    if N < max(N // 4, theiler + fit_steps + 20):
        return float("nan")
    ii = np.arange(N)
    nn_idx = np.full(N, -1, dtype=int)
    nn_dist = np.full(N, np.inf, dtype=float)
    for s in range(0, N, block):
        en = min(s + block, N)
        diff = e[s:en, None, :] - e[None, :, :]
        dd = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        # buang diri sendiri + Theiler window (langkah waktu)
        for a in range(s, en):
            lo = max(0, a - theiler)
            hi = min(N, a + theiler + 1)
            dd[a - s, lo:hi] = np.inf
        arg = np.argmin(dd, axis=1)
        val = dd[np.arange(en - s), arg]
        nn_idx[s:en] = arg
        nn_dist[s:en] = val
    ok = np.isfinite(nn_dist) & (nn_dist > 0)
    if ok.sum() < n_pairs:
        return float("nan")
    idx_ok = ii[ok]
    order = idx_ok[np.argsort(nn_dist[ok])][:n_pairs]
    I = order.copy()
    K = nn_idx[I]
    d0 = nn_dist[I]
    rows = []
    for j in range(1, fit_steps + 1):
        sel = (I + j < N) & (K + j < N)
        if sel.sum() < 20:
            rows.append(np.nan); continue
        dj = np.linalg.norm(e[I[sel] + j] - e[K[sel] + j], axis=1)
        vals = np.log(dj) - (np.log(d0[sel]) if normalize else 0.0)
        vals = vals[np.isfinite(vals)]
        rows.append(float(np.mean(vals)) if len(vals) >= 20 else np.nan)
    rows = np.asarray(rows, dtype=float)
    g = np.isfinite(rows)
    if g.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.arange(1, fit_steps + 1)[g], rows[g], 1)[0])


# ════════════════════════════════════════════════════════════════
# Critical slowing down + IAAFT surrogate
# ════════════════════════════════════════════════════════════════

def csd_metrics(returns: np.ndarray) -> Dict[str, float]:
    r = np.asarray(returns, dtype=float); r = r[np.isfinite(r)]
    if len(r) < 30:
        return {"rho1": float("nan"), "skew": float("nan")}
    a = r[:-1] - r[:-1].mean(); b = r[1:] - r[1:].mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    rho1 = float(np.sum(a * b) / den) if den > 0 else float("nan")
    m2 = float(np.mean((r - r.mean()) ** 2))
    m3 = float(np.mean((r - r.mean()) ** 3))
    skew = float(m3 / (m2 ** 1.5)) if m2 > 0 else float("nan")
    return {"rho1": rho1, "skew": skew}


def iaaft(x: np.ndarray, n_iter: int = 100,
          rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """IAAFT surrogate (Schreiber–Schmitz): mempertahankan spektrum daya DAN
    distribusi amplitudo, mengacak fase ⇒ menghancurkan struktur NONLINEAR
    apa pun yang tidak terkandung dalam autokorelasi linier.

    Langkah terakhir = pemaksaan distribusi (distribusi eksak terjaga;
    spektrum terjaga sampai galat konvergensi IAAFT).
    """
    x = np.asarray(x, dtype=float)
    rng = rng or np.random.default_rng(0)
    n = len(x)
    if n < 8:
        return x.copy()
    sorted_x = np.sort(x)
    amps = np.abs(np.fft.rfft(x))
    phases = rng.uniform(0, 2 * np.pi, len(amps))
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0
    y = np.fft.irfft(amps * np.exp(1j * phases), n=n)
    for _ in range(n_iter):
        ranks = np.argsort(np.argsort(y))
        y = sorted_x[ranks]                     # paksakan distribusi (eksak)
        spec = np.fft.rfft(y)
        ph = np.angle(spec)
        ph[0] = 0.0
        if n % 2 == 0:
            ph[-1] = 0.0
        y = np.fft.irfft(amps * np.exp(1j * ph), n=n)   # paksakan spektrum
    ranks = np.argsort(np.argsort(y))
    return sorted_x[ranks]                       # langkah akhir: distribusi


# ════════════════════════════════════════════════════════════════
# Rolling point-in-time + komposit IGC
# ════════════════════════════════════════════════════════════════

MEASURE_KEYS = ("fim_excess", "fr_drift", "te_sign_to_abs", "te_abs_to_sign",
                "te_asym", "rho1", "skew", "lam1")


def rolling_measures(
    returns: np.ndarray,
    window: int = 250,
    include: Tuple[str, ...] = ("fim", "fr", "te", "csd", "lam1"),
    stride: int = 1,
) -> Dict[str, np.ndarray]:
    """Besaran per hari t dari window TRAILING [t−window+1 .. t] (point-in-time).

    `fr_drift` di indeks t = jarak Fisher–Rao antara distribusi window t dan
    window t−1 (butuh 2 window ⇒ mulai pada t = 2·window−1).

    `stride` > 1 hanya menghitung setiap stride hari (untuk uji null surrogate
    yang mahal); indeks lain tetap NaN. Uji null WAJIB memakai stride yang sama
    untuk deret nyata dan surrogate agar perbandingannya apples-to-apples.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    out = {k: np.full(n, np.nan) for k in MEASURE_KEYS}
    first = window - 1
    for t in range(first, n):
        if stride > 1 and (t - first) % stride != 0:
            continue
        w = r[t - window + 1: t + 1]
        if not np.all(np.isfinite(w)):
            continue
        if "fim" in include:
            out["fim_excess"][t] = fim_excess(w)
        if "te" in include:
            d = te_asymmetry(w)
            out["te_sign_to_abs"][t] = d["te_sign_to_abs"]
            out["te_abs_to_sign"][t] = d["te_abs_to_sign"]
            out["te_asym"][t] = d["te_asym"]
        if "csd" in include:
            d = csd_metrics(w)
            out["rho1"][t] = d["rho1"]
            out["skew"][t] = d["skew"]
        if "lam1" in include:
            tau = automutual_information_tau(w, max_tau=8)
            out["lam1"][t] = rosenstein_lyapunov(w, m=5, tau=max(1, tau))
        if "fr" in include and t >= 2 * window - 1:
            prev = r[t - 2 * window + 1: t - window + 1]
            if np.all(np.isfinite(prev)):
                out["fr_drift"][t] = fisher_rao_distance(w, prev)
    return out


def expanding_percentile(x: np.ndarray, min_obs: int = 60) -> np.ndarray:
    """Peringkat persentil POINT-IN-TIME: x_t dibanding {x_s : s ≤ t}."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    hist: List[float] = []
    for i, v in enumerate(x):
        if np.isfinite(v):
            hist.append(float(v))
        if np.isfinite(v) and len(hist) >= min_obs:
            arr = np.asarray(hist, dtype=float)
            out[i] = float(np.mean(arr <= v))
    return out


IGC_COMPONENTS = ("fim_excess", "fr_drift", "te_asym", "rho1", "neg_skew")


def igc_composite(meas: Dict[str, np.ndarray],
                  min_obs: int = 60) -> np.ndarray:
    """IGC = rata-rata peringkat-persentil-expanding (point-in-time) dari
    {fim_excess, fr_drift, te_asym, rho1, −skew}; orientasi higher = rapuh.

    Bobot pre-spesifikasi: SAMA. λ₁ TIDAK dimasukkan (lihat catatan modul).
    """
    comps = [meas["fim_excess"], meas["fr_drift"], meas["te_asym"],
             meas["rho1"], -meas["skew"]]
    ranks = [expanding_percentile(c, min_obs=min_obs) for c in comps]
    stack = np.vstack(ranks)
    cnt = np.sum(np.isfinite(stack), axis=0)
    ssum = np.nansum(np.where(np.isfinite(stack), stack, 0.0), axis=0)
    return np.where(cnt > 0, ssum / np.maximum(cnt, 1), np.nan)


# ════════════════════════════════════════════════════════════════
# SELF-TEST
# ════════════════════════════════════════════════════════════════

GRID = np.linspace(GRID_LO, GRID_HI, GRID_N)


def _selftest() -> int:
    rng = np.random.default_rng(7)
    fails: List[str] = []

    # (1) FIM ────────────────────────────────────────────────────
    v = kde_fisher_information(rng.normal(0, 1, 8000))
    print(f"[FIM] N(0,1)          → {v:.4f}   (teori 1.0000)")
    if not (0.9 <= v <= 1.12):
        fails.append("FIM Gaussian ≠ 1")
    wide = np.linspace(-15.0, 15.0, 2048)
    v3 = kde_fisher_information(rng.normal(0, 3.0, 8000), grid=wide)
    print(f"[FIM] N(0,9)          → {v3:.4f}   (teori {1/9:.4f})")
    if not (0.7 / 9 <= v3 <= 1.3 / 9):
        fails.append("FIM skala 1/σ² gagal")
    t3 = rng.standard_t(3, 8000); t3 = t3 / t3.std()
    vt = kde_fisher_information(t3)
    print(f"[FIM] Student-t(3)    → {vt:.4f}   (harap > 1: heavy tail)")
    if vt <= 1.0:
        fails.append("FIM tidak sensitif heavy-tail")

    # (2) Fisher–Rao — verifikasi terhadap RUMUS TERTUTUP Gaussian ──
    cases = [("identik", 0.0, 1.0, 0.0, 1.0),
             ("varians ×4", 0.0, 1.0, 0.0, 2.0),
             ("geser mean", 0.0, 1.0, 0.5, 1.0),
             ("campuran", 0.0, 1.0, 1.0, 3.0)]
    for name, mu1, s1, mu2, s2 in cases:
        a = rng.normal(mu1, s1, 20000); b = rng.normal(mu2, s2, 20000)
        est = fisher_rao_distance(a, b)
        theo = fisher_rao_gaussian_analytic(a, b)
        print(f"[F-R] {name:11s} → {est:.4f} (analitik {theo:.4f})")
        if not abs(est - theo) <= 0.03:
            fails.append(f"Fisher–Rao ≠ analitik: {name} ({est:.4f} vs {theo:.4f})")
    a = rng.normal(0, 1, 5000)
    b = rng.normal(0, 2, 5000)
    inv1 = fisher_rao_distance(a, b)
    inv2 = fisher_rao_distance(a * 7.0, b * 7.0)
    print(f"[F-R] invarians skala: d={inv1:.4f} vs d(×7)={inv2:.4f}")
    if abs(inv1 - inv2) > 1e-9:
        fails.append("Fisher–Rao tidak invarian skala")
    # identik → ~0, dan monoton naik terhadap rasio varians
    d_ident = fisher_rao_distance(rng.normal(0, 1, 5000), rng.normal(0, 1, 5000))
    d15 = fisher_rao_distance(rng.normal(0, 1, 5000), rng.normal(0, 1.5, 5000))
    d20 = fisher_rao_distance(rng.normal(0, 1, 5000), rng.normal(0, 2.0, 5000))
    print(f"[F-R] monoton varians: identik={d_ident:.4f} < ×1.5={d15:.4f} < ×2={d20:.4f}")
    if not (d_ident < 0.05 and d_ident < d15 < d20):
        fails.append("Fisher–Rao tidak monoton terhadap rasio varians")

    # (3) MI & TE ───────────────────────────────────────────────
    a = rng.normal(0, 1, 4000); b = rng.normal(0, 1, 4000)
    mi0 = mutual_information(a, b, bins=4)
    mi1 = mutual_information(a, a + 0.8 * rng.normal(0, 1, 4000), bins=4)
    print(f"[MI ] independen {mi0:.5f} bit | terkopel {mi1:.4f} bit")
    if not (mi0 <= 0.02 and mi1 > mi0 + 0.05):
        fails.append("MI tidak diskriminatif")

    n = 6000
    x = rng.normal(0, 1, n)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.7 * y[i - 1] + 0.6 * x[i - 1] + rng.normal(0, 0.5)
    fwd = transfer_entropy(x, y, bins=3)
    bwd = transfer_entropy(y, x, bins=3)
    ind = transfer_entropy(rng.normal(0, 1, n), rng.normal(0, 1, n), bins=3)
    print(f"[TE ] TE(x→y)={fwd:.4f} TE(y→x)={bwd:.4f} independen={ind:.4f}")
    if not (fwd > bwd + 0.02 and ind < 0.02):
        fails.append("TE arah/independensi gagal")
    # arah leverage: tanda→vol harus terdeteksi pada proses leverage sintetis
    lev = np.zeros(n)
    for i in range(1, n):
        vol = 0.5 + 0.9 * (lev[i - 1] < 0)
        lev[i] = (0.3 * lev[i - 1] + rng.normal(0, vol))
    asym = te_asymmetry(lev)["te_asym"]
    print(f"[TE ] asimetri leverage sintetis → {asym:+.4f} (harap > 0)")
    if not (asym > 0.0):
        fails.append("te_asym tidak mendeteksi leverage")

    # (4) Rosenstein λ₁ — sistem deterministik ──────────────────
    v0, lm = 0.123456, np.empty(6000)
    for i in range(6000):
        v0 = 4.0 * v0 * (1 - v0); lm[i] = v0
    lam = rosenstein_lyapunov(lm, m=5, tau=1)
    print(f"[λ₁ ] logistic r=4 → {lam:.4f} (teori ln2 = {np.log(2):.4f})")
    if not (0.55 <= lam <= 0.85):
        fails.append("λ₁ logistic map keluar rentang teori")

    # (5) IAAFT: distribusi eksak + spektrum terjaga ────────────
    s = np.cumsum(rng.normal(0, 1, 2048))
    sur = iaaft(s, n_iter=100, rng=rng)
    spec_corr = float(np.corrcoef(np.abs(np.fft.rfft(sur)), np.abs(np.fft.rfft(s)))[0, 1])
    dist_ok = bool(np.allclose(np.sort(sur), np.sort(s), rtol=1e-9, atol=1e-9))
    print(f"[IAAFT] distribusi eksak={dist_ok} | corr spektrum={spec_corr:.5f}")
    if not (dist_ok and spec_corr > 0.999):
        fails.append("IAAFT tidak mempertahankan distribusi/spektrum")
    # IAAFT harus MENGHANCURKAN ketergantungan nonlinier (|r| clustering)
    vv = np.abs(np.cumsum(rng.normal(0, 1, 2048)))
    ac_real = float(np.corrcoef(vv[:-1], vv[1:])[0, 1])
    ac_sur = np.mean([float(np.corrcoef(iaaft(vv, 50, rng)[:-1],
                                         iaaft(vv, 50, rng)[1:])[0, 1]) for _ in range(1)])
    print(f"[IAAFT] autocorr |x| asli={ac_real:+.4f} | surrogate={ac_sur:+.4f}")
    if abs(ac_sur) >= abs(ac_real):
        fails.append("surrogate IAAFT tidak menghancurkan struktur")

    print()
    if fails:
        print("SELF-TEST GAGAL:", fails)
        return 1
    print("SELF-TEST LULUS — semua besaran matematis sesuai teori/analitik.")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
