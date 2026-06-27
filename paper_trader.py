#!/usr/bin/env python3
"""
Paper Trading Engine — Server-side execution for SFC Terminal
=============================================================
Reads data.json, evaluates signal, executes simulated trades,
saves track record to paper_trades.json.

Run: python3 paper_trader.py
Called by: cron every 5 minutes (same cycle as data collection)
"""
import json, os, math, sys, time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR / "data.json"
TRADES_FILE = SCRIPT_DIR / "paper_trades.json"
HISTORY_FILE = SCRIPT_DIR / "paper_history.json"  # daily snapshots

# ── Market Impact Model (optional — silent fallback) ──
_MI_AVAILABLE = False
_MI_CALC = None
try:
    from market_impact import calculate_entry_cost, calculate_exit_cost
    _MI_AVAILABLE = True
except ImportError:
    pass

# ── Paper Trading Engine ──

class PaperTrader:
    """Server-side paper trader with persistent state."""

    INITIAL_CAPITAL = 50000.0
    MAX_POSITION_PCT = 0.25  # max 25% of capital per position

    def __init__(self):
        self.capital = self.INITIAL_CAPITAL
        self.peak_capital = self.INITIAL_CAPITAL
        self.positions = []       # open positions
        self.trades = []          # all closed trades
        self.equity_history = []  # [(timestamp, equity), ...]
        self.daily_snapshots = {} # date -> {equity, return, sharpe, win_rate, max_dd}
        self.load()

    # ── persistence ──

    def load(self):
        if TRADES_FILE.exists():
            try:
                data = json.loads(TRADES_FILE.read_text())
                self.capital = data.get("capital", self.INITIAL_CAPITAL)
                self.peak_capital = data.get("peak_capital", self.capital)
                self.positions = data.get("positions", [])
                self.trades = data.get("trades", [])
                self.equity_history = data.get("equity_history", [])
                self.daily_snapshots = data.get("daily_snapshots", {})
            except (json.JSONDecodeError, KeyError):
                pass

    def save(self):
        TRADES_FILE.write_text(json.dumps({
            "capital": round(self.capital, 2),
            "peak_capital": round(self.peak_capital, 2),
            "positions": self.positions,
            "trades": self.trades[-500:],  # keep last 500
            "equity_history": self.equity_history[-2000:],  # keep last 2000
            "daily_snapshots": self.daily_snapshots,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))

    # ── signal processing ──

    @staticmethod
    def _safe_float(v, default=0.0):
        try: return float(v)
        except (TypeError, ValueError): return default

    def evaluate_signal(self, data: dict) -> dict:
        """Evaluate SFC data and return trading decision."""
        sfc = PaperTrader._safe_float(data.get("sfc_effective", 0), 0) / 100.0
        conf = PaperTrader._safe_float(data.get("composite_confidence", 0.3), 0.3)
        kelly = PaperTrader._safe_float(data.get("kelly_fraction", 0), 0)
        fng = PaperTrader._safe_float(data.get("fng", 50), 50)
        cascade = PaperTrader._safe_float(data.get("cascade_risk", 0), 0)
        regime = data.get("regime", "NORMAL")
        zone = data.get("zone", "NORMAL")
        btc = data.get("btc", 0)
        signal_type = data.get("signal_type", "CALM")
        dvol = data.get("dvol", 0)

        # Decision logic (mirrors frontend PaperTrader.decide())
        is_extreme_fear = fng < 15
        has_cascade = cascade > 0.5

        if kelly <= 0:
            action = "CASH"
            reason = "No edge"
            if data.get("kelly_override_reason") == "TRANSITION_RISK_OVER_60":
                reason = "Transition Risk >60% · Forced CASH"
            elif is_extreme_fear:
                reason = f"FNG {fng} · Extreme Fear"
            elif has_cascade:
                reason = f"Cascade Risk {cascade*100:.0f}%"
        elif sfc < 0.25 and conf > 0.15:
            action = "BUY"
            reason = f"SFC {sfc*100:.0f}% · Conf {conf*100:.0f}%"
        elif sfc >= 0.45:
            action = "CASH"
            reason = f"SFC {sfc*100:.0f}% · Stress too high"
        else:
            action = "HOLD"
            reason = f"SFC {sfc*100:.0f}% · Neutral zone"

        # Position sizing
        size_pct = 0
        if action == "BUY":
            # Use Kelly fraction capped at MAX_POSITION_PCT
            size_pct = min(kelly, self.MAX_POSITION_PCT)
            # Scale down with confidence
            size_pct *= min(conf * 2, 1.0)

        size = round(self.capital * size_pct, 2)

        return {
            "action": action,
            "size": size if action == "BUY" else 0,
            "reason": reason,
            "btc_price": btc,
            "sfc_pct": round(sfc * 100, 1),
            "confidence": round(conf * 100, 0),
            "kelly_pct": round(kelly * 100, 1),
            "regime": regime,
            "zone": zone,
            "signal_type": signal_type,
            "daily_volume": dvol,
        }

    def execute(self, decision: dict):
        """Execute a trading decision against current state."""
        now = datetime.now(timezone.utc).isoformat()
        price = decision.get("btc_price", 0)
        if not price or price <= 0:
            return

        # Estimate daily volume from data.json
        daily_volume = float(decision.get("daily_volume", 0) or 0)

        # Close positions if signal says CASH and we have positions
        if decision["action"] == "CASH" and self.positions:
            self._close_all_positions(price, "Signal CASH", daily_volume)

        # Open new position if signal says BUY and we're flat
        if decision["action"] == "BUY" and not self.positions and decision["size"] >= 10:
            self._open_position(decision["size"], price, decision, daily_volume)

        # Update PnL for reporting (no action needed on HOLD)
        self._update_pnl(price)
        self._update_peak()

        # Take daily snapshot
        self._daily_snapshot()

        self.save()

    def _open_position(self, size: float, price: float, decision: dict, daily_volume: float = 0):
        # Compute market impact cost
        entry_cost = 0.0
        if _MI_AVAILABLE and daily_volume > 0:
            entry_cost = calculate_entry_cost(size, price, daily_volume)
            if entry_cost > size * 0.5:  # sanity: never lose >50% to slippage
                entry_cost = size * 0.5
        net_size = size - entry_cost
        if net_size <= 0:
            return  # slippage ate the whole position — skip

        self.positions.append({
            "type": "LONG",
            "entry_price": price,
            "size": round(net_size, 2),
            "entry_date": datetime.now(timezone.utc).isoformat(),
            "reason": decision.get("reason", ""),
            "sfc_at_entry": decision.get("sfc_pct"),
            "confidence_at_entry": decision.get("confidence"),
            "slippage_entry": round(entry_cost, 2),
        })
        self.capital -= size  # commit full allocation, but position is smaller
        self.trades.append({
            "id": len(self.trades) + 1,
            "type": "OPEN",
            "date": datetime.now(timezone.utc).isoformat(),
            "price": price,
            "size": round(net_size, 2),
            "slippage": round(entry_cost, 2),
            "reason": decision.get("reason", ""),
        })

    def _close_all_positions(self, price: float, reason: str, daily_volume: float = 0):
        for pos in self.positions:
            exit_cost = 0.0
            if _MI_AVAILABLE and daily_volume > 0:
                exit_cost = calculate_exit_cost(pos["size"], price, daily_volume)
                if exit_cost > pos["size"] * 0.5:
                    exit_cost = pos["size"] * 0.5
            gross_proceeds = pos["size"]
            pnl = (price - pos["entry_price"]) / pos["entry_price"] * pos["size"]
            net_pnl = pnl - exit_cost
            self.capital += gross_proceeds + net_pnl
            self.trades.append({
                "id": len(self.trades) + 1,
                "type": "CLOSE",
                "date": datetime.now(timezone.utc).isoformat(),
                "price": price,
                "size": pos["size"],
                "pnl": round(net_pnl, 2),
                "pnl_pct": round((price - pos["entry_price"]) / pos["entry_price"] * 100, 2),
                "slippage_exit": round(exit_cost, 2),
                "reason": reason,
            })
        self.positions = []

    def _update_pnl(self, price: float):
        unrealized = 0
        for pos in self.positions:
            if pos["type"] == "LONG":
                unrealized += (price - pos["entry_price"]) / pos["entry_price"] * pos["size"]
        self._unrealized = unrealized
        equity = self.capital + sum(pos["size"] for pos in self.positions) + unrealized
        self.equity_history.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "btc": price,
            "equity": round(equity, 2),
            "capital": round(self.capital, 2),
            "unrealized": round(unrealized, 2),
        })
        return equity

    def _update_peak(self):
        current_equity = self.get_equity()
        if current_equity > self.peak_capital:
            self.peak_capital = current_equity

    def _daily_snapshot(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        perf = self.get_performance()
        self.daily_snapshots[today] = {
            "date": today,
            "equity": round(perf["equity"], 2),
            "return_pct": round(perf["total_return"] * 100, 2),
            "sharpe": round(perf["sharpe"], 2),
            "win_rate": round(perf["win_rate"], 4),
            "max_dd_pct": round(perf["max_dd"] * 100, 2),
            "total_trades": perf["total_trades"],
            "open_positions": len(self.positions),
            "status": "IN_MARKET" if self.positions else "CASH",
        }
        # Write daily snapshot separately for dashboard
        self._write_history_json()

    def _write_history_json(self):
        """Write daily snapshot history for frontend dashboard."""
        history = {
            "daily": sorted(self.daily_snapshots.values(), key=lambda x: x["date"]),
            "current": self.get_performance(),
            "equity_curve": self.equity_history[-500:],
        }
        HISTORY_FILE.write_text(json.dumps(history, indent=2))

    # ── performance metrics ──

    def get_equity(self) -> float:
        if not self.positions:
            return self.capital
        return self.equity_history[-1]["equity"] if self.equity_history else self.capital

    def get_performance(self) -> dict:
        equity = self.get_equity()
        total_return = (equity - self.INITIAL_CAPITAL) / self.INITIAL_CAPITAL

        closed = [t for t in self.trades if t.get("pnl") is not None]
        wins = [t for t in closed if t.get("pnl", 0) > 0]
        win_rate = len(wins) / len(closed) if closed else 0

        # Sharpe from equity history
        rets = []
        for i in range(1, len(self.equity_history)):
            prev = self.equity_history[i-1]["equity"]
            cur = self.equity_history[i]["equity"]
            if prev > 0:
                rets.append((cur - prev) / prev)
        mean_ret = sum(rets) / len(rets) if rets else 0
        std_ret = math.sqrt(sum((r - mean_ret)**2 for r in rets) / len(rets)) if rets else 1
        sharpe = mean_ret / std_ret * math.sqrt(252) if std_ret > 0 else 0

        # Max drawdown
        mdd = 0
        peak = self.INITIAL_CAPITAL
        for h in self.equity_history:
            eq = h["equity"]
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > mdd:
                mdd = dd

        return {
            "initial_capital": self.INITIAL_CAPITAL,
            "capital": round(self.capital, 2),
            "equity": round(equity, 2),
            "peak_equity": round(self.peak_capital, 2),
            "total_return": round(total_return, 4),
            "total_return_pct": round(total_return * 100, 2),
            "sharpe": round(sharpe, 2),
            "win_rate": round(win_rate, 4),
            "win_rate_pct": round(win_rate * 100, 1),
            "max_dd": round(mdd, 4),
            "max_dd_pct": round(mdd * 100, 2),
            "total_trades": len(self.trades),
            "closed_trades": len(closed),
            "open_positions": len(self.positions),
            "unrealized_pnl": round(getattr(self, '_unrealized', 0), 2),
            "status": "IN_MARKET" if self.positions else "CASH",
        }


# ── Main ──

def main():
    if not DATA_FILE.exists():
        print("[PaperTrader] No data.json found")
        return

    data = json.loads(DATA_FILE.read_text())
    if not data:
        print("[PaperTrader] data.json is empty — skipping this cycle")
        return
    btc = data.get("btc", 0)
    if btc is None or btc == 0:
        print("[PaperTrader] No BTC price in data")
        return

    trader = PaperTrader()
    decision = trader.evaluate_signal(data)
    trader.execute(decision)

    perf = trader.get_performance()

    # Print summary
    print(json.dumps({
        "action": decision["action"],
        "price": decision["btc_price"],
        "size": decision["size"],
        "reason": decision["reason"],
        "equity": perf["equity"],
        "return_pct": perf["total_return_pct"],
        "sharpe": perf["sharpe"],
        "win_rate": perf["win_rate_pct"],
        "max_dd": perf["max_dd_pct"],
        "trades": perf["total_trades"],
        "status": perf["status"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, indent=2))


if __name__ == "__main__":
    main()
