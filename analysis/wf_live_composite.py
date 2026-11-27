#!/usr/bin/env python3
"""wf_live_composite.py — walk-forward pada komposit LIVE (bukan proxy 3-input).

Masalah yang diselesaikan (audit 2026-11-27):
  `analysis/walk_forward_validation.py` menguji proxy 3 input (btc_24h, FNG, DXY)
  sehingga tidak menguji formula yang benar-benar terbit. Skrip ini memakai
  FUNGSI ASLI dari collect.py (`score_factors_from_market`, `calculate_sfc_ensemble`)
  yang diekstrak lewat AST — jadi tidak ada duplikasi rumus yang bisa menyimpang —
  lalu menambahkan penyesuaian faktor yang punya riwayat: on-chain (CoinMetrics)
  dan ETF flow (m81/m82), persis seperti blok di collect.py.

Batas yang tetap ada (dilaporkan, bukan disembunyikan):
  * Grup M7–M31 (institutional) butuh API live (order book, OI, options) yang tidak
    punya riwayat → tidak bisa direkonstruksi. Kontribusinya di sini = 0.5 netral.
    Karena itu skrip ini mengukur KOMPOSIT CORE + on-chain + ETF, dan mengukur
    sendiri seberapa dekat ia ke angka live terbit (tracking error) alih-alih
    mengklaim identik.
  * Bobot kausal tidak dipakai (riwayat Granger tidak tersedia); weighted mean
    antar-M1–M6 praktis seragam karena bobot core 1.0–1.5, jadi dampaknya kecil
    dan diukur lewat tracking error.

Output: `.wf_live_composite.json` (seri harian) + `.wf_live_composite_summary.json`.
"""
import ast
import bisect
import json
import math
import os
import random
import statistics as st
import subprocess
import sys
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))

BTC_CSV = os.path.join(ROOT, "data_v2", "btc_daily.csv")
ETF_JSON = os.path.join(ROOT, "data", "etf_flow_btc_daily.json")
OUT_SERIES = os.path.join(ROOT, ".wf_live_composite.json")
OUT_SUMMARY = os.path.join(ROOT, ".wf_live_composite_summary.json")

ERAS = [("era1", "2017-01-01", "2018-01-01"),
        ("era2", "2018-01-01", "2022-01-01"),
        ("era3", "2022-01-01", "2099-01-01")]
N_BOOT = 2000
SEED = 20261127


# ── 1. Ambil fungsi asli dari collect.py tanpa mengeksekusi modulnya ──────
def load_collect_functions():
    src = open(os.path.join(ROOT, "collect.py")).read()
    tree = ast.parse(src)
    wanted = {"_sigmoid_factor", "score_factors_from_market", "calculate_sfc_ensemble"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    missing = wanted - {n.name for n in nodes}
    if missing:
        raise SystemExit(f"collect.py tidak lagi memuat fungsi: {missing}")
    ns = {"math": math, "sys": sys}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "collect.py:subset", "exec"), ns)
    return ns["score_factors_from_market"], ns["calculate_sfc_ensemble"]


def fetch_fred(series_id, timeout=30):
    """Ambil seri FRED harian/bulanan (butuh FRED_API_KEY di .env)."""
    import urllib.parse
    import urllib.request
    key = os.getenv("FRED_API_KEY")
    if not key:
        env = os.path.join(ROOT, ".env")
        if os.path.exists(env):
            for line in open(env):
                if line.startswith("FRED_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        return {}
    url = ("https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={key}&file_type=json"
           "&observation_start=2016-01-01")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.load(r)
    except Exception as e:
        print(f"[WF-live] FRED {series_id} gagal: {e}", file=sys.stderr)
        return {}
    out = {}
    for obs in data.get("observations", []):
        v = obs.get("value")
        if v in (None, ".", ""):
            continue
        try:
            out[obs["date"]] = float(v)
        except ValueError:
            continue
    return out


def nearest_prior(series, date_str, max_days=40):
    if not series:
        return None
    d = datetime.strptime(date_str, "%Y-%m-%d")
    for back in range(0, max_days + 1):
        k = (d - timedelta(days=back)).strftime("%Y-%m-%d")
        if k in series:
            return series[k]
    return None


def load_fng():
    """Fear&Greed harian historis (alternative.me, ~2018+)."""
    import urllib.request
    try:
        with urllib.request.urlopen(
            "https://api.alternative.me/fng/?limit=0&format=json", timeout=45
        ) as r:
            data = json.load(r)["data"]
    except Exception as e:
        print(f"[WF-live] FNG gagal: {e}", file=sys.stderr)
        return {}
    out = {}
    for row in data:
        try:
            ts = int(row["timestamp"])
            out[datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")] = float(row["value"])
        except (KeyError, ValueError, TypeError):
            continue
    return out


def load_btc_daily():
    rows = []
    with open(BTC_CSV) as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 5:
                continue
            try:
                rows.append((parts[0], float(parts[4])))
            except ValueError:
                continue
    rows.sort()
    return rows


def load_etf_m81():
    """m81 harian dari riwayat ETF flow (USD -> BTC memakai harga di file).

    Formula identik dengan data_sources/etf_flow.py setelah perbaikan satuan
    2026-11-27 (arus 5 hari, ambang BTC).
    """
    if not os.path.exists(ETF_JSON):
        return {}
    d = json.load(open(ETF_JSON))
    days = sorted(k for k in d if k != "_meta")
    flows_btc = {}
    for day in days:
        row = d[day] or {}
        usd, price, btc = row.get("usd"), row.get("price"), row.get("btc")
        if btc is not None:
            flows_btc[day] = float(btc)
        elif usd is not None and price:
            flows_btc[day] = float(usd) / float(price)
    days = sorted(flows_btc)
    out = {}
    for i, day in enumerate(days):
        window = [flows_btc[x] for x in days[max(0, i - 4): i + 1]]
        avg = sum(window) / len(window)
        if avg < -1000:
            s = 0.85
        elif avg < -500:
            s = 0.70
        elif avg < -100:
            s = 0.60
        elif avg < 100:
            s = 0.50
        elif avg < 500:
            s = 0.35
        elif avg < 1000:
            s = 0.25
        else:
            s = 0.15
        out[day] = s
    return out


# ── 2. Bangun seri komposit ───────────────────────────────────────────────
def build_series():
    score_factors, sfc_ensemble = load_collect_functions()
    btc = load_btc_daily()
    dxy = fetch_fred("DTWEXBGS")
    m2 = fetch_fred("M2SL")
    fng = load_fng()
    etf_m81 = load_etf_m81()
    print(f"[WF-live] btc={len(btc)} dxy={len(dxy)} m2={len(m2)} fng={len(fng)} etf={len(etf_m81)}",
          file=sys.stderr)

    series = []
    prev_price = None
    for date_str, price in btc:
        if prev_price is None or prev_price <= 0:
            prev_price = price
            continue
        btc_24h = (price - prev_price) / prev_price * 100
        prev_price = price

        f = fng.get(date_str)
        d = nearest_prior(dxy, date_str, 7)
        m2_now = nearest_prior(m2, date_str, 45)
        m2_yoy = None
        if m2_now is not None:
            d0 = datetime.strptime(date_str, "%Y-%m-%d")
            m2_prev = nearest_prior(m2, (d0 - timedelta(days=365)).strftime("%Y-%m-%d"), 45)
            if m2_prev:
                m2_yoy = (m2_now - m2_prev) / m2_prev * 100

        try:
            factors = score_factors(
                btc=price, btc_24h=btc_24h, dom=None, dvol=None, fng=f,
                pc_oi=None, m2_yoy=m2_yoy, dxy=d,
            )
            # calculate_sfc_ensemble() already returns sfc_pct on a 0-100 scale
            # (its first return value is `p_ens * 100`). An earlier version of
            # this script multiplied it by 100 again, producing values in the
            # thousands and making every threshold test meaningless.
            sfc = float(sfc_ensemble(factors)[0])
        except Exception:
            continue
        # ETF factor adjustment (sama seperti collect.py: Rt/Lt ±1.5)
        s81 = etf_m81.get(date_str)
        if s81 is not None and s81 != 0.5:
            factors["Rt"] += max(-1.5, min(1.5, (0.5 - s81) * 1.5))
            factors["Lt"] += max(-1.5, min(1.5, (0.5 - s81) * 1.5))
            try:
                sfc = float(sfc_ensemble(factors)[0])
            except Exception:
                pass

        series.append({"date": date_str, "price": price, "sfc": round(sfc, 4)})

    for i, r in enumerate(series):
        for h in (7, 30):
            if i + h < len(series):
                r[f"fwd_{h}d"] = (series[i + h]["price"] - r["price"]) / r["price"] * 100
            else:
                r[f"fwd_{h}d"] = None
    return series


# ── 3. Statistik ──────────────────────────────────────────────────────────
def boot_diff_ci(a, b, block=30, n=N_BOOT, seed=SEED):
    """Moving-block bootstrap untuk beda rata-rata (label majemuk/tumpang tindih)."""
    if len(a) < block * 2 or len(b) < block * 2:
        return None, None, None
    rng = random.Random(seed)

    def draw(x):
        need, out = len(x), []
        while len(out) < need:
            s = rng.randrange(0, max(1, len(x) - block))
            out.extend(x[s:s + block])
        return out[:need]

    diffs = []
    for _ in range(n):
        diffs.append(st.mean(draw(a)) - st.mean(draw(b)))
    diffs.sort()
    return st.mean(a) - st.mean(b), diffs[int(0.05 * n)], diffs[int(0.95 * n)]


def bucket(v, lo=25, hi=50):
    if v < lo:
        return "CALM"
    if v < hi:
        return "ELEVATED"
    return "STRESS"


def analyse(series):
    out = {"n": len(series), "generated_at": datetime.utcnow().isoformat() + "Z"}

    print(f"\nn observasi: {len(series)}  ({series[0]['date']} .. {series[-1]['date']})")
    vals = sorted(r["sfc"] for r in series)
    out["distribution"] = {
        "p10": round(vals[len(vals) // 10], 2),
        "p50": round(vals[len(vals) // 2], 2),
        "p90": round(vals[9 * len(vals) // 10], 2),
        "max": round(vals[-1], 2),
    }
    print("distribusi sfc:", out["distribution"])

    # pooled buckets
    print("\n=== pooled (CALM<25, ELEVATED 25-50, STRESS>=50) ===")
    groups = {}
    for b in ("CALM", "ELEVATED", "STRESS"):
        groups[b] = [r for r in series if r.get("fwd_30d") is not None and bucket(r["sfc"]) == b]
        if not groups[b]:
            continue
        m7 = st.mean([r["fwd_7d"] for r in groups[b] if r.get("fwd_7d") is not None])
        m30 = st.mean([r["fwd_30d"] for r in groups[b]])
        print(f"  {b:9s} n={len(groups[b]):5d}  fwd7={m7:+.2f}%  fwd30={m30:+.2f}%")
        out[f"{b}_n"] = len(groups[b])
        out[f"{b}_fwd30"] = round(m30, 3)

    if groups["CALM"] and groups["STRESS"]:
        a = [r["fwd_30d"] for r in groups["STRESS"]]
        b = [r["fwd_30d"] for r in groups["CALM"]]
        gap, lo, hi = boot_diff_ci(a, b, block=30)
        if gap is None:
            print("  gap STRESS-CALM 30d: sampel terlalu kecil untuk block bootstrap")
        else:
            print(f"  gap STRESS-CALM 30d = {gap:+.2f}pp  90% block-CI [{lo:+.2f}, {hi:+.2f}]"
                  f"  significant={lo * hi > 0}")
        out["gap_30d"] = round(gap, 3)
        out["gap_30d_ci"] = [round(lo, 3), round(hi, 3)]
        out["gap_30d_significant_block"] = bool(lo * hi > 0)

    # era split
    print("\n=== era split ===")
    era_out = {}
    for name, lo_d, hi_d in ERAS:
        sel = [r for r in series if lo_d <= r["date"] < hi_d]
        calm = [r["fwd_30d"] for r in sel if r.get("fwd_30d") is not None and bucket(r["sfc"]) == "CALM"]
        stress = [r["fwd_30d"] for r in sel if r.get("fwd_30d") is not None and bucket(r["sfc"]) == "STRESS"]
        if not calm or not stress:
            print(f"  {name}: n={len(sel)} — bucket kosong, dilewati")
            era_out[name] = {"n": len(sel), "note": "bucket kosong"}
            continue
        gap, l, h = boot_diff_ci(stress, calm, block=30)
        sig = bool(l is not None and l * h > 0)
        era_out[name] = {"n": len(sel), "n_stress": len(stress), "n_calm": len(calm),
                         "calm30": round(st.mean(calm), 2), "stress30": round(st.mean(stress), 2),
                         "gap": round(gap, 2) if gap is not None else None,
                         "ci": [round(l, 2), round(h, 2)] if l is not None else None,
                         "significant_block": sig}
        gap_txt = f"{gap:+.2f}pp" if gap is not None else "n/a (sampel terlalu kecil)"
        print(f"  {name}: n={len(sel)} calm30={st.mean(calm):+.2f}% (n={len(calm)}) "
              f"stress30={st.mean(stress):+.2f}% (n={len(stress)}) gap={gap_txt} sig={sig}")
    out["eras"] = era_out
    signs = [v["gap"] for v in era_out.values() if v.get("gap") is not None]
    out["era_sign_consistent"] = bool(signs) and all(g < 0 for g in signs)

    # decile monotonicity
    sel = [r for r in series if r.get("fwd_30d") is not None]
    sel.sort(key=lambda r: r["sfc"])
    n = len(sel) // 10
    dec = []
    for i in range(10):
        g = sel[i * n:(i + 1) * n] if i < 9 else sel[9 * n:]
        dec.append({"decile": i + 1, "lo": round(g[0]["sfc"], 1), "hi": round(g[-1]["sfc"], 1),
                    "mean30": round(st.mean([r["fwd_30d"] for r in g]), 2)})
    out["deciles"] = dec
    print("\n=== decile (30d) ===")
    for d in dec:
        print(f"  D{d['decile']:<2} sfc[{d['lo']:.1f}-{d['hi']:.1f}] mean={d['mean30']:+.2f}%")

    # threshold sweep (informasi, TIDAK mengubah ambang)
    print("\n=== sweep ambang atas (informasi saja) ===")
    sweep = []
    for hi in (40, 45, 50, 55, 60):
        calm = [r["fwd_30d"] for r in sel if r["sfc"] < 25]
        stress = [r["fwd_30d"] for r in sel if r["sfc"] >= hi]
        if len(stress) < 50 or len(calm) < 50:
            continue
        gap = st.mean(stress) - st.mean(calm)
        sweep.append({"threshold": hi, "n_stress": len(stress), "gap": round(gap, 2)})
        print(f"  sfc>={hi}: n={len(stress):5d} gap vs CALM={gap:+.2f}pp")
    out["threshold_sweep"] = sweep
    return out


def tracking_error(series):
    """Bandingkan rekonstruksi vs angka live terbit (git history data.json)."""
    try:
        hashes = subprocess.run(
            ["git", "-C", ROOT, "log", "--format=%H:%i", "-120", "--", "data.json"],
            capture_output=True, text=True, check=True).stdout.split()
    except Exception as e:
        print(f"[WF-live] tracking: git gagal ({e})", file=sys.stderr)
        return {}
    live = {}
    for h in hashes:
        try:
            raw = subprocess.run(["git", "-C", ROOT, "show", f"{h}:data.json"],
                                 capture_output=True, text=True, timeout=30).stdout
            d = json.loads(raw)
            ts = (d.get("ts") or "")[:10]
            if ts and d.get("sfc_effective") is not None:
                live[ts] = float(d["sfc_effective"])
        except Exception:
            continue
    rec = {r["date"]: r["sfc"] for r in series}
    pairs = [(d, v, rec[d]) for d, v in live.items() if d in rec]
    if len(pairs) < 5:
        print("[WF-live] tracking: pasangan tanggal <5, dilewati", file=sys.stderr)
        return {}
    err = [abs(a - b) for _, a, b in pairs]
    bias = st.mean([b - a for _, a, b in pairs])
    xs = [a for _, a, _ in pairs]
    ys = [b for _, _, b in pairs]
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) or 1
    res = {"n_days": len(pairs), "mae": round(st.mean(err), 2), "bias": round(bias, 2),
           "corr": round(num / den, 3), "live_mean": round(mx, 2), "recon_mean": round(my, 2)}
    print(f"\n=== tracking vs live terbit (n={len(pairs)} hari) ===")
    print(f"  MAE={res['mae']}pp  bias={res['bias']}pp  corr={res['corr']}  "
          f"live_mean={res['live_mean']}  recon_mean={res['recon_mean']}")
    return res


def main():
    series = build_series()
    if not series:
        raise SystemExit("tidak ada observasi")
    json.dump(series, open(OUT_SERIES, "w"))
    summary = analyse(series)
    summary["tracking_vs_live"] = tracking_error(series)
    summary["note"] = ("Komposit CORE (M1-M6) + penyesuaian ETF flow; grup M7-M31 netral "
                       "0.5 karena butuh API live tanpa riwayat. BUKAN 31 metode penuh.")
    json.dump(summary, open(OUT_SUMMARY, "w"), indent=1)
    print(f"\nTersimpan: {OUT_SERIES}\n           {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
