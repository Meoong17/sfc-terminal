#!/usr/bin/env python3
"""L0 — Data Integrity & State Engine.

Implements layer L0 of the SFC V2 blueprint: every input series carries a
definition, a source, a frequency and a maximum tolerated lag, and the pipeline
publishes a machine-readable health report instead of silently serving old data.

Why this exists: on 2026-11-12..14 the dashboard served btc 71,914 while spot
traded 65,4xx, with a fresh timestamp on a frozen payload, and the canonical daily
series had a 34-day hole (Oct 1 - Nov 3) that no component noticed. Deception came
from the plumbing, not the model.

Output: analysis/data_integrity_results.json

  {
    "generated_at": "...", "health": "ok|degraded|broken",
    "counts": {"ok": N, "stale": N, "gap": N, "missing": N},
    "series": [{label, layer, path, last, age_days, max_lag_days, gaps_90d, status, note}],
    "live_price": {"data_json_btc": ..., "feed_btc": ..., "divergence_pct": ..., "status": ...},
    "findings": ["..."]
  }

Status rules (per series):
  ok      -> age <= max_lag_days and no gap in the last 90 calendar days
  stale   -> age >  max_lag_days
  gap     -> missing >= 1 calendar day in the last 90 days
  missing -> file absent or unreadable

Exit code is always 0: the pipeline must never fail because of this report. The
health verdict is published, and collect.py turns it into a dashboard banner.
"""
import json
import os
import sys
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "data_integrity_results.json"

# (path, label, layer, max_lag_days, date_source)
# date_source: "keys" = dict keyed by YYYY-MM-DD, "as_of" = payload carries as_of,
#              "list_date" = list of dicts with a 'date' field, "mtime" = file age only
REGISTRY = [
    ("data/binance_vision_daily.json",  "BTC daily OHLCV (Binance Vision, canonical)", "L0", 3,  "keys"),
    ("data/binance_vision_monthly.json", "BTC monthly OHLCV",                           "L0", 45, "keys",   False),
    ("data/binance_vision_weekly.json",  "BTC weekly OHLCV",                            "L0", 21, "keys",   False),
    # Binance Vision publishes aggTrades monthly, so this input is inherently up
    # to ~31 days behind; the allowance says so instead of crying "stale" daily.
    ("data/binance_orderflow_daily.json", "Orderflow: taker imbalance / trade size (monthly Vision cadence)", "L2", 40, "keys"),
    ("data/binance_oi_daily.json",       "Open interest (Binance)",                     "L2", 2,  "keys"),
    ("data/okx_oi_daily.json",           "Open interest (OKX)",                          "L2", 3,  "keys"),
    ("data/deribit_options_daily.json",  "Options OI / put-call / IV (Deribit)",         "L2", 2,  "keys"),
    ("data/bitmex_funding_daily.json",   "Funding rate history (BitMEX, 2016+; endpoint 403 since 2026-09-16)", "L1", 3, "keys", True, "data/binance_funding_daily.json"),
    ("data/binance_funding_daily.json",  "Funding rate history (Binance perp, 2019+) — spliced tail", "L1", 3, "keys"),
    ("data/binance_futures_basis_daily.json", "Futures basis (Binance quarterly vs spot) — exchange proxy for CME", "L2", 4, "keys"),
    ("data/coinmetrics_btc_daily.json",  "On-chain: active addr / tx / MVRV / hash",     "L3", 3,  "keys"),
    (".etf_cache.json",                  "Spot ETF net flows (Farside)",                 "L2", 4,  "list_date"),
    ("data/tradingview_btc1_daily.csv",  "CME front-month futures (manual TV export)",   "L1", 5,  "csv",   True, "data/binance_futures_basis_daily.json"),
    ("analysis/flqd_engine_results.json", "FLQD v2 engine output",                       "L4", 3,  "as_of"),
    ("analysis/structure_engine_results.json", "Structure engine output (PFS/FQ/NDS)",  "L2", 10, "as_of"),
    ("models/hmm_regime.pkl",            "HMM regime model (fitted)",                   "L4", 90, "mtime"),
]

GAP_WINDOW_DAYS = 90


def _is_date_key(k):
    if not isinstance(k, str):
        return False
    if len(k) == 10 and k[4] == "-" and k[7] == "-":
        return True
    # monthly series are keyed YYYY-MM
    return len(k) == 7 and k[4] == "-"


def _norm_date_key(k):
    return k if len(k) == 10 else f"{k}-01"


def _last_from_keys(obj):
    if not isinstance(obj, dict):
        return None
    dates = [_norm_date_key(k) for k in obj if _is_date_key(k)]
    if not dates:
        return None
    return max(dates)


def _gaps_from_keys(obj, window_days=GAP_WINDOW_DAYS):
    if not isinstance(obj, dict):
        return None
    have = {_norm_date_key(k) for k in obj if _is_date_key(k)}
    if not have:
        return None
    end = date.fromisoformat(max(have))
    start = max(date.fromisoformat(min(have)), end - timedelta(days=window_days))
    missing = []
    cur = start
    while cur <= end:
        if cur.isoformat() not in have:
            missing.append(cur.isoformat())
        cur += timedelta(days=1)
    return missing


def _last_from_list(obj):
    # .etf_cache.json nests the rows under "flows"; accept both shapes.
    if isinstance(obj, dict):
        obj = obj.get("flows")
    if not isinstance(obj, list) or not obj:
        return None
    dates = [r.get("date") for r in obj if isinstance(r, dict) and isinstance(r.get("date"), str)]
    return max(dates) if dates else None


def _gaps_from_list(obj, window_days=GAP_WINDOW_DAYS):
    if isinstance(obj, dict):
        obj = obj.get("flows")
    if not isinstance(obj, list):
        return None
    have = {r.get("date") for r in obj if isinstance(r, dict) and isinstance(r.get("date"), str)}
    if not have:
        return None
    end = date.fromisoformat(max(have))
    start = max(date.fromisoformat(min(have)), end - timedelta(days=window_days))
    missing, cur = [], start
    while cur <= end:
        # ETF flows are business days only — a weekend is not a data gap.
        if cur.isoformat() not in have and cur.weekday() < 5:
            missing.append(cur.isoformat())
        cur += timedelta(days=1)
    return missing


def _csv_last(path):
    try:
        with open(path) as f:
            lines = [ln for ln in f.read().strip().split("\n") if ln.strip()]
        if len(lines) < 2:
            return None
        stamp = lines[-1].split(",")[0].strip()[:10]
        return stamp
    except Exception:
        return None


def check_series(path_str, label, layer, max_lag, date_source, gap_check=True, fallback=None):
    path = ROOT / path_str
    row = {"path": path_str, "label": label, "layer": layer, "max_lag_days": max_lag,
           "fallback": fallback}
    if not path.exists():
        row.update(status="missing", last=None, age_days=None, gaps_90d=None,
                   note="file not found")
        return row

    last, gaps, note = None, None, ""
    if date_source == "csv":
        last = _csv_last(path)
        gaps = None
        note = "TradingView CSV (manual export)"
    elif date_source == "mtime":
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        last = mtime.date().isoformat()
        note = "file modification time"
    elif True:
        try:
            payload = json.load(open(path))
        except Exception as exc:
            row.update(status="missing", last=None, age_days=None, gaps_90d=None,
                       note=f"unreadable: {exc}")
            return row
        if date_source == "keys":
            last = _last_from_keys(payload)
            gaps = _gaps_from_keys(payload) if gap_check else None
        elif date_source == "list_date":
            last = _last_from_list(payload)
            gaps = _gaps_from_list(payload) if gap_check else None
        elif date_source == "as_of":
            last = payload.get("as_of") if isinstance(payload, dict) else None
        elif date_source == "csv":
            last = _csv_last(path)

    if not last:
        row.update(status="missing", last=None, age_days=None, gaps_90d=None,
                   note="no parseable dates" + (f" ({note})" if note else ""))
        return row

    age = (date.today() - date.fromisoformat(last)).days
    if age > max_lag:
        status = "stale"
    elif gaps:
        status = "gap"
    else:
        status = "ok"
    # A dead-but-superseded input is not a degradation of the model: the engine
    # already switched to the replacement series. Say "superseded", not "stale",
    # so the banner keeps pointing at real holes.
    if status != "ok" and fallback:
        try:
            fb = json.load(open(ROOT / fallback))
            fb_last = _last_from_keys(fb)
            if fb and fb_last:
                fb_age = (date.today() - date.fromisoformat(fb_last)).days
                if fb_age <= max_lag:
                    status = "superseded"
                    note = f"replaced by {fallback} (last {fb_last})"
        except Exception:
            pass
    row.update(status=status, last=last, age_days=age,
               gaps_90d=len(gaps) if gaps is not None else None)
    if gaps:
        shown = ", ".join(gaps[:4]) + ("..." if len(gaps) > 4 else "")
        note = f"{len(gaps)} missing day(s) in {GAP_WINDOW_DAYS}d: {shown}"
    row["note"] = note
    return row


def live_price_check():
    """Compare the published dashboard price with the live WS feed."""
    out = {"data_json_btc": None, "feed_btc": None, "divergence_pct": None,
           "feed_age_s": None, "status": "unknown", "note": ""}
    try:
        d = json.load(open(ROOT / "data.json"))
        out["data_json_btc"] = d.get("btc")
        ts = d.get("ts")
        if ts:
            out["data_json_age_min"] = round(
                (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds() / 60, 1)
    except Exception as exc:
        out["note"] = f"data.json unreadable: {exc}"
        return out
    try:
        ws = json.load(open(ROOT / "btc_ws.json"))
        out["feed_btc"] = ws.get("btc")
        if ws.get("ts"):
            out["feed_age_s"] = round(
                (datetime.now(timezone.utc) - datetime.fromisoformat(ws["ts"])).total_seconds())
    except Exception:
        pass
    if out["data_json_btc"] and out["feed_btc"]:
        div = (out["data_json_btc"] / out["feed_btc"] - 1) * 100
        out["divergence_pct"] = round(div, 2)
        out["status"] = "ok" if abs(div) <= 1.0 else "divergent"
        if abs(div) > 1.0:
            out["note"] = (f"dashboard price is {div:+.2f}% away from the live feed — "
                           f"a frozen or stale payload, not a market move")
    return out


def main():
    series = [check_series(*row) for row in REGISTRY]
    counts = {"ok": 0, "stale": 0, "gap": 0, "missing": 0, "superseded": 0}
    for row in series:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    live = live_price_check()
    findings = []
    for row in series:
        if row["status"] == "superseded":
            continue
        if row["status"] == "stale":
            findings.append(f"{row['label']}: stale — last {row['last']} "
                            f"({row['age_days']}d > {row['max_lag_days']}d allowance)")
        elif row["status"] == "gap":
            findings.append(f"{row['label']}: gap — {row['note']}")
        elif row["status"] == "missing":
            findings.append(f"{row['label']}: missing — {row['note']}")
    if live.get("status") == "divergent":
        findings.insert(0, f"live price: {live['note']}")

    critical = [r for r in series
                if r["layer"] == "L0" and r["status"] not in ("ok", "superseded")]
    if critical or live.get("status") == "divergent":
        health = "broken"
    elif findings:
        health = "degraded"
    else:
        health = "ok"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "health": health,
        "counts": counts,
        "series": series,
        "live_price": live,
        "findings": findings,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    print(f"[L0] health={health} ok={counts['ok']} stale={counts['stale']} "
          f"gap={counts['gap']} missing={counts['missing']}", file=sys.stderr)
    for f_ in findings:
        print(f"[L0] {f_}", file=sys.stderr)
    print(json.dumps({"health": health, "counts": counts, "findings": findings[:8]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
