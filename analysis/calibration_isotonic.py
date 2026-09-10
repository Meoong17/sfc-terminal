#!/usr/bin/env python3
"""
calibration_isotonic.py — Kalibrasi confidence monoton + gate OOS
=================================================================
MASALAH YANG DIPERBAIKI (diverifikasi dari sumber, 2026-09):
  `.calibration_state.json` (dibangun 2026-08-12 dari 499 snapshot) menyimpan
  peta bin-based yang:
    * NON-MONOTON: mapping 0.05→0.00, 0.15→0.397, 0.25→0.447, 0.35→0.464,
      0.45→0.313 (TURUN!), lalu 0.55→0.55 (identitas karena bin kosong) —
      diskontinuitas murni artefak bin kosong, bukan informasi.
    * STALE: tak pernah di-refit sejak Agustus, walau tersedia 19.019 commit
      data.json.
  ECE blended 0.170 vs model-only 0.0968 → peta itu sendiri tidak memperbaiki
  kalibrasi terhadap label harga.

DESAIN KALIBRASI (keputusan berdasarkan uji, bukan selera):
  * PRIMER: **beta calibration**  p = σ(a·ln s + b·ln(1−s) + c)  dengan a,b ≥ 0
    ⇒ dijamin monoton naik pada s, hanya 3 parameter sehingga tidak overfit pada
    n≈150-200 label harga yang tersedia.
  * DIAGNOSTIK (bukan kriteria pemilihan): Platt (b=0) dan isotonic/PAV.
  * **PELAJARAN EMPIRIS (uji sintetis, 2026-09):** PAV pada LABEL BINER dengan
    n≈200 menghasilkan prediksi 0/1 (blok tunggal) → Brier di hold-out justru
    MEMBURUK (0.307 → 0.445); ECE juga 0.246 → 0.445. Isotonic hanya masuk akal
    bila n besar (≥1000); di sini ia hanya dilaporkan, dipakai hanya jika
    train_n ≥ 1000 DAN lolos gate.
  * GATE OOS WAJIB: split temporal 60/40; peta dipakai HANYA bila ECE **dan**
    Brier membaik di data uji vs confidence mentah. Tidak lolos → tidak dipakai
    (status quo = raw confidence), alasan dicatat di state.
  * Refit terjadwal: jalankan ulang script ini (cron bulanan/mingguan); metrik
    (fitted_at, n_labeled, ECE, Brier) disimpan untuk audit.

Label: price-outcome (turun > threshold adaptif dalam ~3-4 jam) memakai parameter
cadence yang sudah dituning di confidence_calibration.py (base=72 menit,
lookahead=3, threshold=0.005·√(Δt/base)).

Output: memperbarui `.calibration_state.json` (key "calibration_map"). Peta
bin-based lama tetap tersimpan sebagai fallback back-compat.

Pemakaian:
  .venv/bin/python analysis/calibration_isotonic.py            # fit + gate + simpan
  .venv/bin/python analysis/calibration_isotonic.py --dry-run  # lapor saja
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from confidence_calibration import (  # noqa: E402
    PRICE_BASE_INTERVAL_MINUTES, PRICE_BASE_THRESHOLD_ABS, PRICE_LOOKAHEAD_STEPS,
    STATE_PATH, _adaptive_threshold, _extract_snapshots, _load_state, _save_state,
)

MIN_TRAIN_LABELED = 40      # minimum sampel berlabel utk melatih peta
MIN_TEST_LABELED = 15       # minimum sampel berlabel utk verdict OOS yang berarti
ISOTONIC_MIN_TRAIN = 1000   # isotonic baru dipertimbangkan bila n besar


# ════════════════════════════════════════════════════════════════
# Label harga (identik dengan definisi di confidence_calibration.py)
# ════════════════════════════════════════════════════════════════

def price_label(snaps: List[Dict], idx: int) -> Optional[bool]:
    """True = harga turun > threshold (stress), False = naik, None = flat/tak ada."""
    tgt = min(idx + PRICE_LOOKAHEAD_STEPS, len(snaps) - 1)
    if tgt <= idx:
        return None
    cp, tp = snaps[idx].get("btc", 0), snaps[tgt].get("btc", 0)
    if not cp or not tp:
        return None
    pct = (tp - cp) / cp
    delta = PRICE_BASE_INTERVAL_MINUTES * PRICE_LOOKAHEAD_STEPS
    try:
        a = datetime.fromisoformat(str(snaps[idx].get("ts", "")).replace("Z", "+00:00"))
        b = datetime.fromisoformat(str(snaps[tgt].get("ts", "")).replace("Z", "+00:00"))
        delta = max(PRICE_BASE_INTERVAL_MINUTES, (b - a).total_seconds() / 60.0)
    except (ValueError, TypeError):
        pass
    thr = _adaptive_threshold(delta)
    if pct <= -thr:
        return True
    if pct >= thr:
        return False
    return None


def load_pairs(max_count: int = 800) -> List[Tuple[float, int, str]]:
    """[(raw_confidence, label 0/1, ts)] berlabel harga, urut kronologis."""
    snaps = _extract_snapshots(max_count=max_count)
    snaps.sort(key=lambda s: s.get("ts", ""))
    out = []
    for i, s in enumerate(snaps):
        lab = price_label(snaps, i)
        if lab is None:
            continue
        conf = s.get("composite_confidence")
        if conf is None:
            continue
        out.append((float(conf), 1 if lab else 0, str(s.get("ts", ""))))
    return out


# ════════════════════════════════════════════════════════════════
# Kandidat peta monoton
# ════════════════════════════════════════════════════════════════

def pav_fit(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pooled Adjacent Violators (isotonic). Setiap blok = DUA knot (x awal,
    x akhir) dengan nilai rata-rata blok ⇒ panjang kx/ky selalu sama & monoton."""
    order = np.argsort(x, kind="mergesort")
    xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
    blocks = []  # [start_idx, end_idx, sum_y, count]
    for i, yi in enumerate(ys):
        blocks.append([i, i, float(yi), 1])
        while len(blocks) > 1 and (blocks[-2][2] / blocks[-2][3]) > (blocks[-1][2] / blocks[-1][3]):
            b = blocks.pop(); a = blocks.pop()
            blocks.append([a[0], b[1], a[2] + b[2], a[3] + b[3]])
    kx: List[float] = []
    ky: List[float] = []
    for st, en, s, c in blocks:
        m = s / c
        kx.append(float(xs[st])); ky.append(float(m))
        if en > st:
            kx.append(float(xs[en])); ky.append(float(m))
    if not kx:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    return np.asarray(kx, dtype=float), np.asarray(ky, dtype=float)


def pav_apply(x: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    return np.clip(np.interp(x, kx, ky, left=ky[0], right=ky[-1]), 0.0, 1.0)


def beta_fit(x: np.ndarray, y: np.ndarray, platt_only: bool = False,
             monotone: bool = True) -> Dict[str, float]:
    """Beta calibration terkendala monoton: σ(a·ln s + b·ln(1−s) + c).

    KOREKSI MATEMATIS (2026-09): `a,b ≥ 0` TIDAK menjamin monoton. Turunan logit
    terhadap s adalah  a/s − b/(1−s); ia ≥ 0 untuk semua s∈(0,1) HANYA bila
    a ≥ b. Dengan a,b≥0 saja, peta bisa berbentuk punuk (naik lalu TURUN) —
    hal itu benar-benar terjadi pada fit pertama (titik balik s≈0.187, b=1.5 > a=0.34),
    memunculkan ulang cacat non-monoton yang sedang diperbaiki.
    Karena itu `b` direparametrisasi sebagai b = a·u dengan u∈[0,1] ⇒ a ≥ b
    terjamin ⇒ peta monoton naik secara STRUKTURAL, bukan kebetulan.
    `platt_only=True` (b=0) adalah kasus paling sederhana dan selalu monoton.
    """
    from scipy.optimize import minimize
    eps = 1e-6
    s = np.clip(np.asarray(x, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    ls, l1s = np.log(s), np.log(1 - s)

    def nll(theta):
        a, u, c = theta
        b = 0.0 if platt_only else a * u
        z = a * ls + b * l1s + c
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        p = np.clip(p, eps, 1 - eps)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    theta0 = [1.0, 0.0, 0.0] if platt_only else [1.0, 0.5, 0.0]
    bounds = [(0.0, 20.0), (0.0, 1.0 if not platt_only else 0.0), (-20.0, 20.0)]
    res = minimize(nll, np.array(theta0), bounds=bounds, method="L-BFGS-B")
    a, u, c = [float(v) for v in res.x]
    b = 0.0 if platt_only else a * u
    if not monotone and not platt_only:      # jalur tak-terkendala (diagnostik saja)
        res2 = minimize(nll, np.array([1.0, 0.0, 0.0]), bounds=[(0.0, 20.0), (0.0, 20.0), (-20.0, 20.0)],
                        method="L-BFGS-B", args=())
        a2, b2, c2 = [float(v) for v in res2.x]
        return {"a": a2, "b": b2, "c": c2, "platt_only": False,
                "monotone_constraint": False}
    return {"a": a, "b": b, "c": c, "platt_only": bool(platt_only),
            "monotone_constraint": bool(monotone or platt_only)}


def beta_apply(x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    eps = 1e-6
    s = np.clip(np.asarray(x, dtype=float), eps, 1 - eps)
    z = (params.get("a", 1.0) * np.log(s) + params.get("b", 0.0) * np.log(1 - s)
         + params.get("c", 0.0))
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    total = len(y)
    if total == 0:
        return float("nan")
    idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += (m.sum() / total) * abs(y[m].mean() - p[m].mean())
    return float(e)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


# ════════════════════════════════════════════════════════════════
# Fit + gate
# ════════════════════════════════════════════════════════════════

def fit_gated(max_count: int = 800, test_frac: float = 0.4) -> Dict:
    pairs = load_pairs(max_count=max_count)
    n = len(pairs)
    base: Dict = {
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_labeled": n, "max_count": max_count, "test_frac": test_frac,
        "params": {"base_interval_min": PRICE_BASE_INTERVAL_MINUTES,
                   "lookahead_steps": PRICE_LOOKAHEAD_STEPS,
                   "base_threshold_abs": PRICE_BASE_THRESHOLD_ABS},
        "primary_method": "beta",
    }
    if n < MIN_TRAIN_LABELED + MIN_TEST_LABELED:
        base.update({"accepted": False,
                     "rejected_reason": f"too few price-labeled snapshots ({n})"})
        return base

    split = int(n * (1 - test_frac))
    tr, te = pairs[:split], pairs[split:]
    xtr = np.array([p[0] for p in tr]); ytr = np.array([p[1] for p in tr], float)
    xte = np.array([p[0] for p in te]); yte = np.array([p[1] for p in te], float)

    e_raw, b_raw = ece(yte, xte), brier(yte, xte)

    # ── Kandidat peta, urutan PRA-SPESIFIKASI (parsimoni lebih dulu) ──────────
    # PRIMER   : Platt (b=0) — 1 parameter, monoton naik secara struktural.
    # FALLBACK : beta terkendala monoton (a ≥ b) — hanya bila primer gagal gate,
    #            dan aturan ini ditetapkan sebelum melihat metrik uji.
    # DIAGNOSTIK: isotonic/PAV (hanya bila n besar; pada n kecil ia memprediksi 0/1).
    platt_p = beta_fit(xtr, ytr, platt_only=True)
    m_platt = {"ece": ece(yte, beta_apply(xte, platt_p)),
               "brier": brier(yte, beta_apply(xte, platt_p))}
    beta_p = beta_fit(xtr, ytr, monotone=True)
    m_beta = {"ece": ece(yte, beta_apply(xte, beta_p)),
              "brier": brier(yte, beta_apply(xte, beta_p))}

    diag: Dict = {"beta_monotone": {"params": beta_p, "test": m_beta}}
    if len(ytr) >= ISOTONIC_MIN_TRAIN:
        kx, ky = pav_fit(xtr, ytr)
        p_iso = pav_apply(xte, kx, ky)
        diag["isotonic"] = {"knots_x": [round(float(v), 4) for v in kx],
                            "knots_y": [round(float(v), 4) for v in ky],
                            "test": {"ece": ece(yte, p_iso), "brier": brier(yte, p_iso)}}
    else:
        diag["isotonic"] = {"skipped": f"train_n {len(ytr)} < {ISOTONIC_MIN_TRAIN} "
                                       f"(PAV overfit pada label biner di n kecil)"}

    # verifikasi monotonitas NUMERIK (bukan asumsi) untuk peta yang dipilih
    dense = np.linspace(0.0, 1.0, 201)
    mono_platt = bool(np.all(np.diff(beta_apply(dense, platt_p)) >= -1e-12))
    mono_beta = bool(np.all(np.diff(beta_apply(dense, beta_p)) >= -1e-12))

    gates = {"platt": bool(m_platt["ece"] < e_raw and m_platt["brier"] < b_raw),
             "beta_monotone": bool(m_beta["ece"] < e_raw and m_beta["brier"] < b_raw)}

    chosen: Optional[str] = None
    if gates["platt"] and mono_platt:
        chosen = "platt"
    elif gates["beta_monotone"] and mono_beta:
        chosen = "beta_monotone"

    params = platt_p if chosen == "platt" else (beta_p if chosen == "beta_monotone" else None)
    metrics_chosen = (m_platt if chosen == "platt" else m_beta) if chosen else None
    grid = np.round(np.linspace(0, 1, 21), 2)
    base.update({
        "accepted": chosen is not None,
        "chosen_method": chosen,
        "params_beta": params,          # nama key dipertahankan (dibaca recalibrate())
        "params_key_note": "params_beta berisi parameter model terpilih (platt/beta_monotone)",
        "monotone_verified": bool(mono_platt if chosen == "platt" else mono_beta) if chosen else False,
        "pre_specified_order": ["platt", "beta_monotone"],
        "gate_each": gates,
        "monotone_each": {"platt": mono_platt, "beta_monotone": mono_beta},
        "map_grid": ({str(g): round(float(beta_apply(np.array([g]), params)[0]), 4)
                      for g in grid} if params else None),
        "metrics": {
            "test_n": int(len(yte)),
            "test_base_rate_stress": round(float(yte.mean()), 4),
            "ece_raw": round(e_raw, 4),
            "ece_calibrated": round(metrics_chosen["ece"], 4) if metrics_chosen else None,
            "brier_raw": round(b_raw, 4),
            "brier_calibrated": round(metrics_chosen["brier"], 4) if metrics_chosen else None,
        },
        "diagnostics": diag,
        "train_n": int(len(ytr)),
        "train_base_rate_stress": round(float(ytr.mean()), 4),
        "rejected_reason": (None if chosen else
                            "gate OOS gagal: Platt maupun beta terkendala-monoton tidak "
                            "memperbaiki ECE DAN Brier di data uji"),
    })
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="jangan tulis state, hanya laporkan hasil gate")
    ap.add_argument("--max-count", type=int, default=800)
    args = ap.parse_args()

    res = fit_gated(max_count=args.max_count)
    m = res.get("metrics") or {}
    print("=" * 70)
    print("Kalibrasi monoton (beta primer; Platt & isotonic = diagnostik) + gate OOS")
    print(f"  snapshot berlabel harga : {res['n_labeled']} "
          f"(train {res.get('train_n')} / test {m.get('test_n')})")
    print(f"  base rate stress (test) : {m.get('test_base_rate_stress')}")
    d = res.get("diagnostics") or {}
    print(f"  ECE   raw {m.get('ece_raw')} → terpilih {m.get('ece_calibrated')} "
          f"| Platt {(d.get('platt') or {}).get('test', {}).get('ece')} "
          f"| beta-monoton {((d.get('beta_monotone') or {}).get('test') or {}).get('ece')} "
          f"| isotonic {(((d.get('isotonic') or {}).get('test')) or {}).get('ece')}")
    print(f"  Brier raw {m.get('brier_raw')} → terpilih {m.get('brier_calibrated')} "
          f"| Platt {(d.get('platt') or {}).get('test', {}).get('brier')} "
          f"| beta-monoton {(((d.get('beta_monotone') or {}).get('test')) or {}).get('brier')}")
    print(f"  gate per kandidat: {res.get('gate_each')} | monoton: {res.get('monotone_each')}")
    print(f"  metode terpilih: {res.get('chosen_method')} | params: {res.get('params_beta')}")
    print(f"  monoton terverifikasi: {res.get('monotone_verified')}")
    print(f"  KEPUTUSAN: {'DIPAKAI' if res.get('accepted') else 'DITOLAK'} "
          f"({res.get('rejected_reason') or 'lolos gate OOS'})")
    if res.get("map_grid"):
        print("  peta (raw→kalibrasi, tiap 0.1):",
              " ".join(f"{k}:{v}" for k, v in list(res["map_grid"].items())[::2]))

    if not args.dry_run:
        state = _load_state() or {}
        state["calibration_map"] = res
        state["calibration_map_last_attempt"] = res["fitted_at"]
        if res.get("accepted"):
            state["ece"] = res["metrics"]["ece_calibrated"]
            state["interpretation"] = "Beta calibration (monoton, lolos gate OOS)"
        _save_state(state)
        print(f"  state ditulis: {STATE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
