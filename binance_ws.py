#!/usr/bin/env python3
"""
binance_ws.py — Binance WebSocket BTC ticker daemon
Writes latest BTC price to btc_ws.json for fast local reads.
Auto-reconnects on disconnect. Runs as background process.
"""
import json, os, sys, time, signal
from datetime import datetime, timezone

# Use stdlib ssl/select as fallback when websocket-client unavailable
try:
    import websocket
    HAVE_WS = True
except ImportError:
    HAVE_WS = False

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "btc_ws.json")
PING_INTERVAL = 30  # seconds

running = True

def signal_handler(sig, frame):
    global running
    running = False
    print("[WS] Shutting down...", file=sys.stderr)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def write_price(data):
    """Write ticker data to JSON file atomically."""
    tmp = OUTPUT + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, OUTPUT)
    except Exception as e:
        print(f"[WS] Write error: {e}", file=sys.stderr)

def on_message(ws, message):
    try:
        d = json.loads(message)
        if "e" in d and d["e"] == "24hrTicker":
            price = float(d["c"])
            chg = float(d["P"])  # 24h change %
            high = float(d["h"])
            low = float(d["l"])
            vol = float(d["v"])  # volume
            ts = datetime.now(timezone.utc).isoformat()
            data = {
                "source": "binance_ws",
                "btc": price,
                "btc_24h": chg,
                "high_24h": high,
                "low_24h": low,
                "volume_24h": vol,
                "ts": ts
            }
            write_price(data)
    except (json.JSONDecodeError, KeyError, ValueError):
        pass  # ignore malformed messages

def on_error(ws, error):
    print(f"[WS] Error: {error}", file=sys.stderr)

def on_close(ws, close_status_code, close_msg):
    print(f"[WS] Closed (code={close_status_code})", file=sys.stderr)

def on_open(ws):
    print("[WS] Connected to Binance stream", file=sys.stderr)
    # Subscribe to ticker
    ws.send(json.dumps({
        "method": "SUBSCRIBE",
        "params": ["btcusdt@ticker"],
        "id": 1
    }))

def run():
    if not HAVE_WS:
        print("[WS] websocket-client not installed. Install: pip install websocket-client", file=sys.stderr)
        sys.exit(1)

    url = "wss://stream.binance.com:9443/ws"
    
    while running:
        try:
            ws = websocket.WebSocketApp(
                url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            ws.run_forever(
                ping_interval=PING_INTERVAL,
                ping_timeout=10,
                reconnect=5  # reconnect delay
            )
        except Exception as e:
            print(f"[WS] Connection error: {e}", file=sys.stderr)
        
        if running:
            print("[WS] Reconnecting in 5s...", file=sys.stderr)
            time.sleep(5)

if __name__ == "__main__":
    # Write initial placeholder
    write_price({"source": "binance_ws", "btc": None, "btc_24h": None, "ts": datetime.now(timezone.utc).isoformat(), "status": "connecting"})
    run()
