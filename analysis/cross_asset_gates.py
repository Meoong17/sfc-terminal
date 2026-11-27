#!/usr/bin/env python3
"""cross_asset_gates.py — gerbang lintas-aset (Model 1 & 3 dari cetak biru 4-lapis).

Sumber desain: dokumen A (Cross-Asset Liquidity & Structural Regime Engine).
Yang DIPAKAI dari dokumen: struktur gerbang bersyarat (ML-PT Gate, BIM 4 gerbang,
aturan alokasi) — karena bentuknya bisa gagal secara eksplisit, menggantikan narasi
kalender halving (n=3, tidak bisa dipakai untuk inferensi).

Yang SENGAJA TIDAK diimplementasikan:
  * Model 2 / ACBI — butuh CDS Tier-1, Oracle 5Y CDS, harga obligasi tunai 144A
    (CRWV/TRACTC). Tidak ada sumber gratis/tepercaya; memakai HY OAS/LQD sebagai
    pengganti lalu menamainya "ACBI" akan mengarang sinyal. Status dilaporkan
    `UNAVAILABLE (data)`.
  * Pemberian bobot skor: seluruh keluaran skrip ini adalah KONTEKS/DISPLAY.
    Tidak menyentuh `sfc_effective`, ambang zona, atau bobot faktor. Ambang di sini
    ditetapkan tangan dan BELUM walk-forward validated — lihat `meta.validation`.

Input yang dipakai (semua sumber yang benar-benar ada di mesin ini):
  FRED      : PCEPILFE (Core PCE), PCEC96 (Real PCE), DGS10, DGS20, DGS30,
              BAMLH0A0HYM2 (HY OAS), DEXJPUS (USDJPY)
  Lokal     : data_v2/btc_daily.csv (harga, 200W SMA), data/binance_oi_daily.json,
              data/binance_funding_daily.json, data/binance_orderflow_daily.json (CVD),
              .etf_cache.json (flow per-ticker), data.json (DXY, DVOL)
  Opsional  : stooq (XAUUSD/XAGUSD) — dilewati bila tidak terjangkau
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
import urllib.request
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "data", "cross_asset_state.json")
NOW = datetime.utcnow()


# ── util ──────────────────────────────────────────────────────────────────
def fred_key() -> str | None:
    k = os.getenv("FRED_API_KEY")
    if k:
        return k
    env = os.path.join(ROOT, ".env")
    if os.path.exists(env):
        for line in open(env):
            if line.startswith("FRED_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def fred_series(series_id: str, start: str = "2023-01-01") -> list[tuple[str, float]]:
    key = fred_key()
    if not key:
        return []
    url = ("https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={key}&file_type=json&observation_start={start}")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
    except Exception as e:
        print(f"[XAG] FRED {series_id}: {e}", file=sys.stderr)
        return []
    out = []
    for o in data.get("observations", []):
        v = o.get("value")
        if v in (None, ".", ""):
            continue
        try:
            out.append((o["date"], float(v)))
        except ValueError:
            continue
    return out


def pct_change(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    return (a - b) / abs(b) * 100.0


def diff_bps(series: list[tuple[str, float]], days: int) -> float | None:
    """Perubahan level dalam bps sepanjang ~`days` hari kalender."""
    if len(series) < 2:
        return None
    last_d, last_v = series[-1]
    target = (datetime.strptime(last_d, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
    prior = [v for d, v in series if d <= target]
    if not prior:
        prior = [series[0][1]]
    return round((last_v - prior[-1]) * 100.0, 1)


def stooq_series(symbol: str) -> list[tuple[str, float]]:
    """Harga komoditas harian. Saat ini TIDAK tersedia: stooq memasang
    proof-of-work anti-scraping ("This site requires JavaScript"), Yahoo Finance
    membalas "Edge: Too Many Requests", dan FRED sudah menghentikan seri
    GOLDAMGBD228NLBM/GOLDPMGBD228NLBM/SLVPRUSD (HTTP 400). Fungsi tetap ada
    supaya sumber bisa disambungkan tanpa mengubah pemanggil; kegagalan = [].
    """
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read().decode("utf-8", "ignore").strip().splitlines()
    except Exception:
        return []
    if not raw or raw[0].startswith("<"):
        return []
    out = []
    for line in raw[1:]:
        p = line.split(",")
        if len(p) < 5:
            continue
        try:
            out.append((p[0], float(p[4])))
        except ValueError:
            continue
    return out


def load_json(path: str, default=None):
    try:
        return json.load(open(path))
    except Exception:
        return default


# ── Model 1: ML-PT Gate ───────────────────────────────────────────────────
def model1_gate() -> dict:
    pce = fred_series("PCEPILFE", "2022-01-01")
    rpce = fred_series("PCEC96", "2022-01-01")
    y10 = fred_series("DGS10")
    d20 = fred_series("DGS20")
    d30 = fred_series("DGS30")
    oas = fred_series("BAMLH0A0HYM2", "1997-01-01")
    jpy = fred_series("DEXJPUS")

    out: dict = {"model": "ML-PT Gate", "status": "UNKNOWN", "inputs": {}, "flags": {}}

    # Core PCE: YoY + MoM annualized
    core_yoy = core_ann = None
    if len(pce) >= 13:
        last_d, last_v = pce[-1]
        y_ago = [v for d, v in pce if d[:7] == f"{int(last_d[:4]) - 1}-{last_d[5:7]}"]
        core_yoy = round(pct_change(last_v, y_ago[0]), 2) if y_ago else None
        mom = pct_change(last_v, pce[-2][1])
        core_ann = round(((1 + mom / 100.0) ** 12 - 1) * 100, 2) if mom is not None else None
    real_mom = None
    if len(rpce) >= 2:
        real_mom = round(pct_change(rpce[-1][1], rpce[-2][1]), 2)

    us10y = y10[-1][1] if y10 else None
    d_y10 = diff_bps(y10, 30)
    curve_20s30s = round((d30[-1][1] - d20[-1][1]) * 100, 1) if (d20 and d30) else None
    hy = oas[-1][1] if oas else None
    d_oas = diff_bps(oas, 30)

    dxy = (load_json(os.path.join(ROOT, "data.json"), {}) or {}).get("dxy")
    gold = stooq_series("xauusd")
    silver = stooq_series("xagusd")
    g30 = pct_change(gold[-1][1], gold[-31][1]) if len(gold) > 31 else None
    s30 = pct_change(silver[-1][1], silver[-31][1]) if len(silver) > 31 else None

    out["inputs"] = {
        "core_pce_yoy_pct": core_yoy,
        "core_pce_mom_annualized_pct": core_ann,
        "real_pce_mom_pct": real_mom,
        "us10y_pct": us10y,
        "us10y_chg_30d_bps": d_y10,
        "curve_20s30s_bps": curve_20s30s,
        "hy_oas_pct": hy,
        "hy_oas_chg_30d_bps": d_oas,
        "dxy": dxy,
        "usdjpy": jpy[-1][1] if jpy else None,
        "gold_usd": gold[-1][1] if gold else None,
        "gold_chg_30d_pct": round(g30, 2) if g30 is not None else None,
        "silver_usd": silver[-1][1] if silver else None,
        "silver_chg_30d_pct": round(s30, 2) if s30 is not None else None,
        "as_of": {"pce": pce[-1][0] if pce else None,
                  "us10y": y10[-1][0] if y10 else None,
                  "hy_oas": oas[-1][0] if oas else None},
    }

    # Policy trap: inflasi inti masih di atas target + konsumsi riil stagnan.
    # Ambang 2,5% / 0,5% dipilih dari definisi "di atas target" dan "stagnan",
    # BUKAN hasil optimasi — lihat meta.validation.
    trap = None
    if core_yoy is not None and real_mom is not None:
        trap = "BOXED_IN_HAWKISH" if (core_yoy > 2.5 and real_mom <= 0.5) else "NOT_TRAPPED"

    # Flight-to-quality: yield turun TAPI OAS melebar (bukan easing sehat)
    f2q = None
    if d_y10 is not None and d_oas is not None:
        if d_y10 < 0 and d_oas > 0:
            f2q = "DEFENSIVE_FLIGHT_TO_QUALITY"
        elif d_y10 < 0 and d_oas <= 0:
            f2q = "HEALTHY_EASING"
        elif d_y10 >= 0 and d_oas > 0:
            f2q = "RISK_OFF_RATES_UP"
        else:
            f2q = "RISK_ON"

    # Dash for cash: dolar kuat + logam jatuh bersamaan.
    # Bila data logam tidak tersedia, kondisi ini TIDAK boleh diklaim terkonfirmasi:
    # hanya level DXY yang terukur, separuh definisi tidak. Dilaporkan None + basis.
    drain = None
    drain_basis = "unavailable"
    if dxy is not None and g30 is not None and s30 is not None:
        drain = bool(dxy > 103 and g30 < 0 and s30 < 0)
        drain_basis = "dxy+gold+silver"
    elif dxy is not None:
        drain = None
        drain_basis = "dxy_only_partial"

    out["flags"] = {"policy_trap": trap, "yield_oas_regime": f2q,
                    "cross_asset_liquidity_drain": drain,
                    "drain_basis": drain_basis}

    # Trigger utama = policy trap + flight-to-quality (keduanya terukur penuh).
    # `drain` hanya penguat bila basis datanya lengkap (dxy+gold+silver).
    if trap == "BOXED_IN_HAWKISH" and (f2q == "DEFENSIVE_FLIGHT_TO_QUALITY"
                                       or drain is True):
        out["status"] = "CONTRACTION_LOCK"
    elif trap == "NOT_TRAPPED" and f2q in ("HEALTHY_EASING", "RISK_ON") and not drain:
        out["status"] = "EXPANSION"
    elif core_yoy is not None or us10y is not None:
        out["status"] = "NEUTRAL_CHOP"

    out["reason"] = (f"Core PCE {core_yoy}% YoY (annualized MoM {core_ann}%), "
                     f"real PCE {real_mom}% MoM; US10Y {us10y}% ({d_y10:+} bps/30d); "
                     f"HY OAS {hy}% ({d_oas:+} bps/30d); 20s30s {curve_20s30s} bps; "
                     f"DXY {dxy}; gold {out['inputs']['gold_chg_30d_pct']}% / "
                     f"silver {out['inputs']['silver_chg_30d_pct']}% (30d)"
                     if core_yoy is not None else "data makro tidak lengkap")
    return out


# ── Model 3: BIM 4 gerbang bottom ─────────────────────────────────────────
def model3_gates() -> dict:
    state = load_json(os.path.join(ROOT, "data.json"), {}) or {}
    btc_rows = []
    csv = os.path.join(ROOT, "data_v2", "btc_daily.csv")
    if os.path.exists(csv):
        with open(csv) as f:
            f.readline()
            for line in f:
                p = line.rstrip("\n").split(",")
                if len(p) > 4:
                    try:
                        btc_rows.append((p[0], float(p[4]), float(p[5]) if p[5] else 0.0))
                    except ValueError:
                        continue
    btc_rows.sort()
    closes = [c for _, c, _ in btc_rows]
    price = closes[-1] if closes else None
    sma200w = st.mean(closes[-1400:]) if len(closes) >= 200 else None

    # ETF: 15 hari bursa terakhir + IBIT harian (per-ticker dari cache)
    etf = load_json(os.path.join(ROOT, ".etf_cache.json"), {}) or {}
    flows = etf.get("flows") or []
    rows = []
    for f in flows[-40:]:
        try:
            etfs = f.get("etfs") or {}
            rows.append({"date": f["date"],
                         "total_btc": f.get("total_btc"),
                         "total_usd": f.get("total_usd") or 0.0,
                         "ibit_usd_m": float(etfs.get("IBIT") or 0.0)})
        except Exception:
            continue
    # .etf_cache.json menyimpan total_usd + flow per-ticker (juta USD); total_btc
    # biasanya None, jadi menjumlahkan total_btc menghasilkan 0 palsu. Pakai USD
    # sebagai angka utama dan estimasi BTC dari harga terakhir.
    net15_usd = sum((r["total_usd"] or 0) for r in rows[-15:])
    _px = closes[-1] if closes else None
    net15 = (net15_usd / _px) if _px else 0.0
    ibit_recent = [r["ibit_usd_m"] for r in rows[-5:]]
    ibit_pos_days = sum(1 for v in ibit_recent if v >= 0)

    # OI: baseline 90 hari vs sekarang
    oi = load_json(os.path.join(ROOT, "data", "binance_oi_daily.json"), {}) or {}
    oi_days = sorted(k for k in oi if not k.startswith("_"))
    oi_now = oi[oi_days[-1]].get("sum_oi") if oi_days else None
    oi_hi = max((oi[d].get("sum_oi") or 0) for d in oi_days[-90:]) if oi_days else None
    oi_dd = round((oi_now - oi_hi) / oi_hi * 100, 2) if (oi_now and oi_hi) else None

    fund = load_json(os.path.join(ROOT, "data", "binance_funding_daily.json"), {}) or {}
    fdays = sorted(k for k in fund if not k.startswith("_"))[-3:]
    fund_mean = round(st.mean([fund[d]["rate_mean"] for d in fdays]), 6) if fdays else None

    # CVD: kumulatif taker buy - taker sell (quote), 7 hari terakhir
    of = load_json(os.path.join(ROOT, "data", "binance_orderflow_daily.json"), {}) or {}
    odays = sorted(k for k in of if not k.startswith("_"))[-14:]
    cvd_series = []
    run = 0.0
    for d in odays:
        row = of[d]
        run += (row.get("taker_buy_quote") or 0) - (row.get("taker_sell_quote") or 0)
        cvd_series.append((d, run, row.get("price_close")))
    cvd_low7 = min(c[1] for c in cvd_series[-7:]) if len(cvd_series) >= 7 else None
    cvd_low_first = min(c[1] for c in cvd_series[:7]) if len(cvd_series) >= 14 else None
    px_min7 = min(c[2] for c in cvd_series[-7:] if c[2]) if len(cvd_series) >= 7 else None
    px_min_first = min(c[2] for c in cvd_series[:7] if c[2]) if len(cvd_series) >= 14 else None

    gates = []
    # Gate 1 — ETF exhaustion
    g1_pass = bool(ibit_pos_days >= 3)
    gates.append({
        "gate": 1, "name": "ETF Exhaustion", "pass": g1_pass,
        "detail": (f"15D net flow {net15_usd/1e9:+.2f}B USD (~{net15:,.0f} BTC); IBIT ≥0 pada {ibit_pos_days}/5 hari terakhir "
                   f"({', '.join(f'{v:+.1f}M' for v in ibit_recent)})"),
        "fail_reason": None if g1_pass else "IBIT belum 3 hari berturut net inflow",
    })
    # Gate 2 — Leverage flush
    g2_pass = bool(oi_dd is not None and oi_dd <= -15 and (fund_mean is not None and fund_mean <= 0))
    gates.append({
        "gate": 2, "name": "Leverage Flush", "pass": g2_pass,
        "detail": (f"OI {oi_now:,.0f} BTC vs puncak 90D {oi_hi:,.0f} ({oi_dd:+.1f}%); "
                   f"funding 3D mean {fund_mean*100:+.4f}%/8h" if oi_now and oi_hi and fund_mean is not None
                   else "data OI/funding tidak lengkap"),
        "fail_reason": None if g2_pass else "OI belum ter-flush ≤ -15% dari puncak 90D atau funding masih positif",
    })
    # Gate 3 — Spot absorption
    g3_pass = bool(cvd_low7 is not None and cvd_low_first is not None and px_min7 is not None
                   and px_min_first is not None
                   and px_min7 < px_min_first and cvd_low7 > cvd_low_first)
    gates.append({
        "gate": 3, "name": "Spot Absorption", "pass": g3_pass,
        "detail": (f"harga 7D min {px_min7:,.0f} vs 7D sebelumnya {px_min_first:,.0f}; "
                   f"CVD min {cvd_low7:,.0f} vs {cvd_low_first:,.0f} (USD kumulatif 14D)"
                   if None not in (cvd_low7, cvd_low_first, px_min7, px_min_first)
                   else "data orderflow tidak lengkap"),
        "fail_reason": None if g3_pass else "belum ada higher low CVD saat harga lower low",
    })
    # Gate 4 — Structure reclaim
    g4_pass = bool(price is not None and sma200w is not None and price > sma200w)
    gates.append({
        "gate": 4, "name": "Structure Reclaim", "pass": g4_pass,
        "detail": (f"harga {price:,.0f} vs 200W SMA {sma200w:,.0f} "
                   f"({(price/sma200w-1)*100:+.1f}%)" if price and sma200w else "data harga tidak lengkap"),
        "fail_reason": None if g4_pass else "harga masih di bawah 200W SMA",
    })
    passed = sum(1 for g in gates if g["pass"])
    return {
        "model": "BIM-ETF Bottom Gates",
        "gates": gates,
        "gates_passed": passed,
        "gates_total": 4,
        "status": "BOTTOM_CONFIRMED" if passed == 4 else f"{passed}/4 GATES",
        "inputs": {"etf_net_15d_usd": round(net15_usd, 0) if rows else None,
                   "etf_net_15d_btc_est": round(net15, 0) if rows else None,
                   "ibit_last5_usd_m": ibit_recent,
                   "oi_now_btc": oi_now, "oi_peak_90d_btc": oi_hi, "oi_drawdown_pct": oi_dd,
                   "funding_3d_mean_pct_8h": round(fund_mean * 100, 5) if fund_mean is not None else None,
                   "cvd_low7_usd": round(cvd_low7, 0) if cvd_low7 is not None else None,
                   "price": price, "sma200w": round(sma200w, 0) if sma200w else None,
                   "dvol": state.get("dvol")},
    }


# ── Model 4: aturan alokasi (display, bukan eksekusi order) ───────────────
def model4_allocation(m1: dict, m3: dict) -> dict:
    passed = m3.get("gates_passed", 0)
    if m1.get("status") == "CONTRACTION_LOCK" and passed < 3:
        mode = "CAPITAL_PRESERVATION_&_RENTALS"
        band = {"cash_like_pct": [80, 90], "tactical_rental_pct": [10, 20]}
        note = ("Rezim kontraksi + gerbang bottom belum terpenuhi: prioritaskan likuiditas. "
                "Posisi rental hanya transaksi pendek (mean-reversion ke resistensi), "
                "dilarang dikonversi jadi posisi panjang selama CVD agregat belum positif.")
    elif passed >= 3:
        mode = "GRADUAL_RE_ENTRY"
        band = {"cash_like_pct": [40, 60], "tactical_rental_pct": [40, 60]}
        note = "Gerbang bottom ≥3/4; alokasi dapat dinaikkan bertahap, bukan all-in."
    else:
        mode = "DEFENSIVE_CORE"
        band = {"cash_like_pct": [60, 80], "tactical_rental_pct": [20, 40]}
        note = "Sinyal campuran; kurangi ukuran posisi, tunggu konfirmasi gerbang."
    return {
        "model": "TR-SSA Allocator",
        "mode": mode,
        "allocation_band_pct": band,
        "note": note,
        "sector_rules": {
            "avoid": ["token AI/DePIN GPU bersubsidi emisi atau narasi kelangkaan komputasi",
                      "memecoin tanpa dukungan likuiditas makro"],
            "priority_when_gates_ge_3": ["BTC sebagai penyerap likuiditas pertama",
                                         "protokol dengan pendapatan biaya organik non-subsidi"],
        },
        "disclaimer": "Aturan display, bukan saran investasi; tidak masuk sfc_effective.",
    }


def main() -> int:
    m1 = model1_gate()
    m3 = model3_gates()
    m4 = model4_allocation(m1, m3)
    acbi = {
        "model": "ACBI (AI Credit Bifurcation)",
        "status": "UNAVAILABLE",
        "reason": ("Butuh CDS Tier-1 (MSFT/GOOGL/AMZN/META/NVDA), Oracle 5Y CDS, dan harga "
                   "obligasi tunai 144A (CRWV/TRACTC). Tidak ada sumber gratis/tepercaya; "
                   "menggantinya dengan HY OAS/LQD lalu menamainya ACBI = sinyal karangan."),
    }
    state = {
        "generated_at": NOW.isoformat() + "Z",
        "source_design": "A.docx — Cross-Asset Liquidity & Structural Regime Engine",
        "models": {"ml_pt_gate": m1, "acbi": acbi, "bim_etf": m3, "tr_ssa": m4},
        "meta": {
            "scoring_impact": "NONE — konteks/display only, tidak dipakai di sfc_effective",
            "validation": ("Ambang gerbang ditetapkan tangan dari definisi dokumen dan BELUM "
                           "walk-forward validated. Jangan dipakai sebagai klaim prediktif "
                           "sampai diuji pada riwayat (HY OAS 1997+, ETF 2024+, OI/CVD 2026+)."),
            "gates_summary": f"{m3['gates_passed']}/4",
        },
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1, allow_nan=False)
    os.replace(tmp, OUT_PATH)

    print(f"ML-PT Gate : {m1['status']}  | trap={m1['flags']['policy_trap']} "
          f"yield/oas={m1['flags']['yield_oas_regime']} drain={m1['flags']['cross_asset_liquidity_drain']}")
    print(f"  {m1['reason']}")
    print(f"ACBI       : {acbi['status']} (data tidak tersedia)")
    print(f"BIM gates  : {m3['gates_passed']}/4")
    for g in m3["gates"]:
        print(f"  [{('PASS' if g['pass'] else 'FAIL')}] Gate {g['gate']} {g['name']}: {g['detail']}")
    print(f"TR-SSA     : {m4['mode']} cash {m4['allocation_band_pct']['cash_like_pct']}% "
          f"rental {m4['allocation_band_pct']['tactical_rental_pct']}%")
    print(f"→ {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
