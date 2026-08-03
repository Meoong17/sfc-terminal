#!/usr/bin/env python3
"""
sse_server.py — SFC Terminal SSE Real-time Server
Streams BTC price (from binance_ws.py WebSocket daemon) + SFC data pipeline updates
to connected browser clients via Server-Sent Events.

Usage:
  cd /home/ubuntu/sfc && .venv/bin/python sse_server.py
  # Server runs on http://127.0.0.1:8765 (loopback only)
"""
import asyncio, json, os, secrets, sys, time, signal
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
import uvicorn

BASE_DIR = Path(__file__).parent
BTC_WS_PATH = BASE_DIR / "btc_ws.json"
DATA_JSON_PATH = BASE_DIR / "data.json"

# ── Origin auth token (defense in depth) ─────────────────────────
# The Cloudflare Worker is the ONLY legitimate client of this origin.
# Set SFC_ORIGIN_TOKEN in the systemd unit (or env) to a strong random
# value; the Worker sends it as `Authorization: Bearer <token>`. Without
# a matching token every protected route returns 401. If the env var is
# empty the server stays OPEN (backward-compat) — but production MUST set it.
_ORIGIN_TOKEN = os.environ.get("SFC_ORIGIN_TOKEN", "").strip()


def _auth(request: Request):
    """Require Bearer token unless SFC_ORIGIN_TOKEN is unset (open mode)."""
    if not _ORIGIN_TOKEN:
        return  # no token configured -> open (dev/backward-compat)
    supplied = request.headers.get("Authorization", "")
    if not secrets.compare_digest(supplied, "Bearer " + _ORIGIN_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")


app = FastAPI(title="SFC Terminal SSE")

# CORS — allow the Cloudflare Worker domain and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sfc-terminal.meoong17.workers.dev",
        "http://localhost:*",
        "http://127.0.0.1:*",
        "https://meoong17.github.io",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory cache for latest values ──
_latest_btc = None
_latest_sfc = None
_btc_mtime = 0
_sfc_mtime = 0
_last_broadcast_btc = None  # avoid duplicate events
_last_broadcast_sfc = None

def _read_json(path, last_mtime):
    """Read JSON file if modified, return (data, new_mtime)."""
    try:
        mtime = path.stat().st_mtime
        if mtime <= last_mtime:
            return None, last_mtime
        with open(path) as f:
            data = json.load(f)
        return data, mtime
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None, last_mtime


async def event_generator(request: Request):
    """SSE event stream — pushes BTC price tick + SFC data updates."""
    global _latest_btc, _latest_sfc, _btc_mtime, _sfc_mtime
    global _last_broadcast_btc, _last_broadcast_sfc

    async def _publish_btc(data):
        """Emit BTC ticker event (every ~1s from ws, but only when price changes meaningfully)."""
        nonlocal request
        price = data.get("btc")
        if price is None:
            return
        payload = {
            "price": price,
            "change_pct": data.get("btc_24h", 0),
            "high_24h": data.get("high_24h", 0),
            "low_24h": data.get("low_24h", 0),
            "volume_24h": data.get("volume_24h", 0),
            "ts": data.get("ts", datetime.now(timezone.utc).isoformat()),
        }
        return {"event": "btc_ticker", "data": json.dumps(payload)}

    async def _publish_sfc(data):
        """Emit full SFC dashboard update."""
        nonlocal request
        payload = {k: data[k] for k in (
            "btc", "btc_24h", "btc_mcap", "fng", "fng_cls",
            "dom", "dvol", "sfc_base", "sfc_effective",
            "zone", "regime", "signal", "news_stress",
            "news_headlines", "news_sentiment", "m2_yoy",
            "liq_mod", "liq_total_24h", "cascade_risk", "rsi_14",
            "ath", "ath_date",
            "kelly_fraction", "composite_confidence", "signal_type",
        ) if k in data}
        payload["ts"] = data.get("ts", datetime.now(timezone.utc).isoformat())
        return {"event": "sfc_update", "data": json.dumps(payload)}

    try:
        while True:
            if await request.is_disconnected():
                break

            # ── Read BTC WS file (written by binance_ws.py daemon, real-time) ──
            btc_data, btc_mtime = _read_json(BTC_WS_PATH, _btc_mtime)
            if btc_data and btc_mtime > _btc_mtime:
                _btc_mtime = btc_mtime
                _latest_btc = btc_data
                event = await _publish_btc(btc_data)
                if event:
                    # Deduplicate identical BTC events (avoid flood)
                    price_key = round(btc_data.get("btc", 0), 2)
                    if price_key != _last_broadcast_btc:
                        yield event
                        _last_broadcast_btc = price_key

            # ── Read data.json (written by collect.py every 5 min) ──
            sfc_data, sfc_mtime = _read_json(DATA_JSON_PATH, _sfc_mtime)
            if sfc_data and sfc_mtime > _sfc_mtime:
                _sfc_mtime = sfc_mtime
                _latest_sfc = sfc_data
                # Only emit if data actually changed
                ts_key = sfc_data.get("ts", "")
                if ts_key != _last_broadcast_sfc:
                    event = await _publish_sfc(sfc_data)
                    if event:
                        yield event
                        _last_broadcast_sfc = ts_key

            # Heartbeat every 15s (keeps connection alive, also delivers initial data)
            yield {"event": "heartbeat", "data": json.dumps({"ts": datetime.now(timezone.utc).isoformat()})}
            await asyncio.sleep(1)

    except asyncio.CancelledError:
        pass


# ── Routes ──

@app.get("/events", dependencies=[Depends(_auth)])
async def sse_events(request: Request):
    """SSE endpoint — client connects here with EventSource."""
    return EventSourceResponse(event_generator(request))

@app.get("/health")
async def health():
    """Health check endpoint."""
    btc_ok = BTC_WS_PATH.exists()
    sfc_ok = DATA_JSON_PATH.exists()
    btc_age = time.time() - BTC_WS_PATH.stat().st_mtime if btc_ok else -1
    sfc_age = time.time() - DATA_JSON_PATH.stat().st_mtime if sfc_ok else -1
    return {
        "status": "ok",
        "btc_ws": {"exists": btc_ok, "age_s": round(btc_age, 1)},
        "data_json": {"exists": sfc_ok, "age_s": round(sfc_age, 1)},
        "ts": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/snapshot", dependencies=[Depends(_auth)])
async def snapshot():
    """Return latest cached data (for initial page load or sync)."""
    btc = None
    sfc = None
    try:
        with open(BTC_WS_PATH) as f:
            btc = json.load(f)
    except: pass
    try:
        with open(DATA_JSON_PATH) as f:
            sfc = json.load(f)
    except: pass
    return {"btc": btc, "sfc": sfc, "ts": datetime.now(timezone.utc).isoformat()}


# ── Serve static frontend files (WHITELIST only) ──
#
# SECURITY FIX: previously this mounted the ENTIRE repo at "/", which exposed
# .env, .git/, all *.py, logs, and trained models over the public tunnel.
# Now only explicitly-whitelisted public assets/data are served; everything
# else returns 404. Keep this list minimal.
_PUBLIC_FILES = {
    "index.html",   # SFC dashboard
    "app.js",       # dashboard JS
    "sw.js",        # service worker
    "data.json",    # pipeline output (public dashboard data)
    "btc_ws.json",  # live BTC ws data (public dashboard data)
    "ai_analysis.json",  # weekly Hermes LLM analyst brief (public dashboard data)
    "sitemap.xml",  # search-engine sitemap (indexing)
    "robots.txt",   # search-engine crawl directives (indexing)
    "static/chart.umd.min.js",  # self-hosted Chart.js (no CDN dependency)
}


@app.get("/", dependencies=[Depends(_auth)])
async def index():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/{path:path}", dependencies=[Depends(_auth)])
async def static_file(path: str):
    # Whitelist + normalization guard: only explicit public filenames are served.
    # resolve() + is_file() blocks path traversal even if the list grows.
    # `fonts/` prefix is allowed for the self-hosted Inter webfonts only.
    is_font = path.startswith("fonts/") and path.endswith(".woff2")
    if path in _PUBLIC_FILES or is_font:
        fpath = (BASE_DIR / path).resolve()
        try:
            fpath.relative_to(BASE_DIR.resolve())
        except ValueError:
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        if fpath.is_file():
            # SEO files (sitemap/robots) must never be cached at the edge —
            # a stale cached 404 (e.g. before a file existed) sticks for the
            # whole cache TTL and makes Google Search Console report
            # "Couldn't fetch". Force revalidation every time.
            if path in ("sitemap.xml", "robots.txt"):
                return FileResponse(fpath, headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                })
            return FileResponse(fpath)
    return JSONResponse({"detail": "Not Found"}, status_code=404)


if __name__ == "__main__":
    PORT = int(os.environ.get("SSE_PORT", 8765))
    # Bind to loopback only. The Cloudflare named tunnel (cloudflared, running
    # on this same host) reaches us via localhost; binding 127.0.0.1 keeps the
    # origin unreachable directly from the internet (no public IP:port).
    HOST = os.environ.get("SSE_HOST", "127.0.0.1")
    print(f"[SSE] Starting SFC SSE server on {HOST}:{PORT}", file=sys.stderr)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
