#!/usr/bin/env python3
"""
Paper Trading Engine — Server-side execution for SFC Terminal
=============================================================
Reads data.json, evaluates signal, executes simulated trades (LONG & SHORT),
saves track record to paper_trades.json.

Run: python3 paper_trader.py
Called by: cron every 5 minutes (same cycle as data collection)

Supports:
  - LONG positions (BUY signal, sfc low)
  - SHORT positions (SELL signal, sfc high)
  - Time slippage simulation (execution price != signal price)
  - Market impact cost (Almgren-Chriss via market_impact.py)
"""
import json, os, math, sys, time, random
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent
DATA_FILE = SCRIPT_DIR / "data.json"
TRADES_FILE = SCRIPT_DIR / "paper_trades.json"
HISTORY_FILE = SCRIPT_DIR / "paper_history.json"  # daily snapshots

# ── Execution Delay Config ──
# Simulates real-world delay between signal generation and order execution.
# During EXECUTION_DELAY_MINUTES, BTC price drifts by a random walk calibrated
# to typical BTC volatility (~1% hourly = ~0.0167% per minute std).
# This replaces the old TIME_SLIPPAGE_STD (simple random jitter) with a
# more realistic delay-based drift model.
EXECUTION_DELAY_MINUTES = 5       # Simulated delay between signal and execution
BTC_MINUTE_VOLATILITY = 0.000167  # ~1% hourly / 60 = 0.0167% per minute
MIN_SLIPPAGE = 0.0001             # Minimum slippage floor (0.01%)
_MAX_SLIPPAGE = 0.05              # Cap at 5% adverse move

# ── Risk Management Config (NEW) ──
# Previously there was NO price-based exit mechanism at all — positions
# only closed when the SFC signal itself flipped to CASH or the opposite
# direction. This meant a position could stay open indefinitely
# accumulating losses if the signal simply didn't change (e.g. stuck in
# a regime that doesn't flip), which is a real gap given the model's
# actual prediction horizon is 6 hours (LABEL_LOOKAHEAD_MINUTES=360 in
# ml_ensemble.py) — holding a position for days on a signal designed to
# mean something over 6 hours isn't what the model was built to support.
STOP_LOSS_PCT = 0.08           # Force-close if unrealized loss exceeds 8%
MAX_HOLDING_HOURS = 24          # Force-close after 24h regardless of signal —
                                 # ~4x the model's own 6h prediction horizon,
                                 # not an arbitrary round number; a position
                                 # the signal hasn't already exited by then is
                                 # outside what the model was validated to say
                                 # anything about.

# ── Position Sizing Config (NEW) ──
# Previously used data.json's "kelly_fraction" directly — this is FULL
# Kelly (not kelly_half, despite kelly_half already being computed and
# available in the same data.json). Full Kelly is known to produce high
# variance in realized returns even when the edge estimate is exactly
# correct, and confidence_calibration.py's own measurement found
# ECE=0.216 ("poorly calibrated") for the confidence feeding into this
# edge estimate — betting FULL Kelly on top of a poorly-calibrated edge
# compounds two sources of risk rather than hedging either one. Half
# Kelly is the standard practitioner adjustment for exactly this
# situation (edge estimation uncertainty), trading some expected growth
# for meaningfully lower variance.
USE_HALF_KELLY = True
DRAWDOWN_SIZE_REDUCTION_THRESHOLD = 0.15  # if current drawdown exceeds 15%...
DRAWDOWN_SIZE_REDUCTION_FACTOR = 0.5      # ...cut position size in half
LOSING_STREAK_THRESHOLD = 3               # if last N closed trades were all losses...
LOSING_STREAK_SIZE_REDUCTION_FACTOR = 0.5 # ...cut position size in half

# ── Execution Cost Config (NEW) ──
# Previously only modeled time-delay slippage + Almgren-Chriss market
# impact — missing the bid-ask spread itself, which is a real, distinct
# cost component on every trade (you cross the spread on entry AND
# exit) independent of order size or execution delay.
BID_ASK_SPREAD_PCT = 0.0005   # ~0.05% half-spread for BTC on major venues;
                                # applied once per side (entry, exit)

# RNG for execution slippage simulation.
# NOTE: this intentionally does NOT use a fixed seed (seed=42 was the
# previous behavior). A fixed seed in a module-level global is only
# "reproducible" within a single process lifetime — and because
# paper_trader.py is invoked as a fresh process by sfc-pipeline.sh on
# every cycle, a fixed seed means the slippage drift sequence resets
# to the exact same values every time. In practice this makes the
# "random walk during execution delay" deterministic and constant:
# every BUY or SELL always experiences the same drift direction and
# magnitude rather than sampling from the intended distribution of
# plausible market moves. This caused a systematic bias in paper
# trading results (consistently same-direction slippage), not the
# neutral-on-average distribution the delay model was designed for.
# os.urandom-seeded RNG varies per-process as intended.
_SLIPPAGE_RNG = random.Random()   # seeded from OS entropy at import time

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
    """Server-side paper trader with persistent state.

    Supports LONG and SHORT positions. Only one position at a time.
    Uses time slippage to simulate realistic execution delays.
    """

    INITIAL_CAPITAL = 50000.0
    MAX_POSITION_PCT = 0.25  # max 25% of capital per position

    def __init__(self):
        self.capital = self.INITIAL_CAPITAL
        self.peak_capital = self.INITIAL_CAPITAL
        self.positions = []       # open positions (max 1)
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

    def _current_drawdown_pct(self) -> float:
        """Current drawdown from peak_capital, as a positive fraction
        (0.15 = 15% drawdown). Used to scale down position size during
        a losing period — see DRAWDOWN_SIZE_REDUCTION_THRESHOLD."""
        equity = self.get_equity()
        if self.peak_capital <= 0:
            return 0.0
        dd = (self.peak_capital - equity) / self.peak_capital
        return max(0.0, dd)

    def _recent_losing_streak(self) -> int:
        """Count consecutive losing trades ending at the most recent
        CLOSED trade (0 if the most recent closed trade was a win, or
        if there are no closed trades yet)."""
        closed = [t for t in self.trades if t.get("pnl") is not None]
        streak = 0
        for t in reversed(closed):
            if t.get("pnl", 0) < 0:
                streak += 1
            else:
                break
        return streak

    def _get_execution_price(self, signal_price: float) -> float:
        """Simulate execution delay with realistic price drift.

        Instead of simple random jitter, simulates a random walk over
        EXECUTION_DELAY_MINUTES with BTC's typical per-minute volatility.
        This produces more realistic slippage: small moves most of the time
        with occasional larger gaps (fat tails), reflecting actual market
        behavior during the delay between signal and fill.

        Returns:
            Execution price after simulated delay.
        """
        steps = max(1, int(EXECUTION_DELAY_MINUTES))
        drift = 0.0
        for _ in range(steps):
            drift += _SLIPPAGE_RNG.gauss(0, BTC_MINUTE_VOLATILITY)

        # Clamp to avoid extreme outliers
        drift = max(-_MAX_SLIPPAGE, min(_MAX_SLIPPAGE, drift))

        # Enforce minimum slippage floor (always show SOME friction)
        if abs(drift) < MIN_SLIPPAGE:
            drift = MIN_SLIPPAGE if drift >= 0 else -MIN_SLIPPAGE

        exec_price = signal_price * (1.0 + drift)
        return round(max(exec_price, signal_price * 0.95), 2)

    def evaluate_signal(self, data: dict) -> dict:
        """Evaluate SFC data and return trading decision.

        Supports both LONG (BUY) and SHORT (SELL) signals.
        LONG: sfc low, kelly > 0
        SHORT: sfc high, kelly > 0 (bearish confidence)
        """
        sfc = PaperTrader._safe_float(data.get("sfc_effective", 0), 0) / 100.0
        conf = PaperTrader._safe_float(data.get("composite_confidence", 0.3), 0.3)
        # Switched from "kelly_fraction" (Full Kelly) to "kelly_half" — see
        # USE_HALF_KELLY config comment above for the full reasoning.
        # Falls back to computing half of kelly_fraction manually if
        # kelly_half isn't present in an older data.json (defensive, not
        # expected to trigger against a current pipeline).
        if USE_HALF_KELLY:
            kelly = PaperTrader._safe_float(data.get("kelly_half"), None)
            if kelly is None:
                kelly = PaperTrader._safe_float(data.get("kelly_fraction", 0), 0) / 2.0
        else:
            kelly = PaperTrader._safe_float(data.get("kelly_fraction", 0), 0)
        fng = PaperTrader._safe_float(data.get("fng", 50), 50)
        cascade = PaperTrader._safe_float(data.get("cascade_risk", 0), 0)
        regime = data.get("regime", "NORMAL")
        zone = data.get("zone", "NORMAL")
        btc = data.get("btc", 0)
        signal_type = data.get("signal_type", "CALM")
        dvol = data.get("dvol", 0)
        bear_conf = PaperTrader._safe_float(data.get("prob_stress", 0), 0)

        is_extreme_fear = fng < 15
        has_cascade = cascade > 0.5

        # Determine action
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
        elif sfc >= 0.45 and bear_conf > 0.4:
            # Strong bearish signal + bearish confidence → SHORT
            action = "SELL"
            reason = f"SFC {sfc*100:.0f}% · Bear {bear_conf*100:.0f}%"
        elif sfc >= 0.45:
            action = "CASH"
            reason = f"SFC {sfc*100:.0f}% · Stress too high"
        else:
            action = "HOLD"
            reason = f"SFC {sfc*100:.0f}% · Neutral zone"

        # Position sizing
        size_pct = 0
        size_reduction_reason = None
        if action in ("BUY", "SELL"):
            # Use Kelly fraction (half-Kelly, see USE_HALF_KELLY) capped at MAX_POSITION_PCT
            size_pct = min(kelly, self.MAX_POSITION_PCT)
            # Scale down with confidence
            size_pct *= min(conf * 2, 1.0)

            # ── Drawdown-based size reduction (NEW) ──
            # Standard risk management: reduce size while recovering from
            # a drawdown, rather than sizing purely off the current
            # signal's own Kelly/confidence — a losing period doesn't
            # change what the SIGNAL says, but it's a real reason to bet
            # smaller until the strategy demonstrates it's back on track.
            current_dd = self._current_drawdown_pct()
            if current_dd >= DRAWDOWN_SIZE_REDUCTION_THRESHOLD:
                size_pct *= DRAWDOWN_SIZE_REDUCTION_FACTOR
                size_reduction_reason = f"Drawdown {current_dd*100:.0f}% → size halved"

            # ── Losing-streak-based size reduction (NEW) ──
            elif self._recent_losing_streak() >= LOSING_STREAK_THRESHOLD:
                size_pct *= LOSING_STREAK_SIZE_REDUCTION_FACTOR
                size_reduction_reason = f"{LOSING_STREAK_THRESHOLD}+ losses in a row → size halved"

        size = round(self.capital * size_pct, 2)

        # Apply time slippage to execution price
        signal_price = btc
        exec_price = self._get_execution_price(signal_price)

        if size_reduction_reason:
            reason = f"{reason} · {size_reduction_reason}"

        return {
            "action": action,
            "size": size if action in ("BUY", "SELL") else 0,
            "reason": reason,
            "btc_price": btc,
            "execution_price": round(exec_price, 2),
            "time_slippage_pct": round((exec_price - signal_price) / signal_price * 100, 3),
            "sfc_pct": round(sfc * 100, 1),
            "confidence": round(conf * 100, 0),
            "kelly_pct": round(kelly * 100, 1),
            "regime": regime,
            "zone": zone,
            "signal_type": signal_type,
            "daily_volume": dvol,
            "is_short": action == "SELL",
        }

    def _check_risk_management_exit(self, price: float, daily_volume: float) -> bool:
        """
        Force-close the open position if either:
        (a) unrealized loss exceeds STOP_LOSS_PCT, or
        (b) the position has been held longer than MAX_HOLDING_HOURS.

        This runs BEFORE signal-based close logic and takes priority over
        it — previously there was NO price-based or time-based exit at
        all, meaning a position could accumulate unbounded losses or sit
        open indefinitely if the SFC signal simply never flipped. Given
        this system's actual prediction horizon is 6 hours (see
        ml_ensemble.py's LABEL_LOOKAHEAD_MINUTES), a position still open
        well beyond that (MAX_HOLDING_HOURS=24, ~4x the horizon) is
        outside anything the model was validated to say something about,
        regardless of what the signal currently reads.

        Returns True if a forced exit happened (caller should skip normal
        signal-based open/close logic for this cycle, since we already
        closed).
        """
        if not self.positions:
            return False

        pos = self.positions[0]
        entry_price = pos["entry_price"]

        if pos["type"] == "LONG":
            unrealized_pct = (price - entry_price) / entry_price
        else:  # SHORT
            unrealized_pct = (entry_price - price) / entry_price

        if unrealized_pct <= -STOP_LOSS_PCT:
            self._close_all_positions(
                price, f"Stop-loss triggered ({unrealized_pct*100:.1f}%)", daily_volume
            )
            return True

        try:
            entry_dt = datetime.fromisoformat(pos["entry_date"])
            held_hours = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600.0
        except (ValueError, TypeError, KeyError):
            held_hours = 0

        if held_hours >= MAX_HOLDING_HOURS:
            self._close_all_positions(
                price, f"Max holding period reached ({held_hours:.0f}h)", daily_volume
            )
            return True

        return False

    def execute(self, decision: dict):
        """Execute a trading decision against current state."""
        now = datetime.now(timezone.utc).isoformat()
        price = decision.get("execution_price", decision.get("btc_price", 0))
        if not price or price <= 0:
            return

        is_short = decision.get("is_short", False)

        # Estimate daily volume from data.json
        daily_volume = float(decision.get("daily_volume", 0) or 0)

        # ── Risk management exit takes priority over signal-based logic ──
        forced_exit = self._check_risk_management_exit(price, daily_volume)

        # Close positions if signal says CASH or opposite direction
        # (skipped if we already force-closed above this cycle)
        if not forced_exit and self.positions and (
            decision["action"] == "CASH"
            or (decision["action"] == "BUY" and self.positions[0]["type"] == "SHORT")
            or (decision["action"] == "SELL" and self.positions[0]["type"] == "LONG")
        ):
            self._close_all_positions(price, decision["reason"], daily_volume)

        # Open new position if signal says BUY/SELL and we're flat
        if decision["action"] in ("BUY", "SELL") and not self.positions and decision["size"] >= 10:
            pos_type = "SHORT" if is_short else "LONG"
            self._open_position(pos_type, decision["size"], price, decision, daily_volume)

        # Update PnL for reporting
        self._update_pnl(price)
        self._update_peak()

        # Take daily snapshot
        self._daily_snapshot()

        self.save()

    def _open_position(self, pos_type: str, size: float, price: float, decision: dict, daily_volume: float = 0):
        # Compute market impact cost
        entry_cost = 0.0
        if _MI_AVAILABLE and daily_volume > 0:
            entry_cost = calculate_entry_cost(size, price, daily_volume)
            if entry_cost > size * 0.5:  # sanity: never lose >50% to slippage
                entry_cost = size * 0.5
        # ── Bid-ask spread cost (NEW) ──
        # Previously only market impact + time-delay slippage were
        # modeled — the spread itself (a cost on every trade regardless
        # of size or delay) was missing entirely.
        spread_cost = size * BID_ASK_SPREAD_PCT
        entry_cost += spread_cost
        net_size = size - entry_cost
        if net_size <= 0:
            return  # slippage ate the whole position — skip

        self.positions.append({
            "type": pos_type,
            "entry_price": price,
            "size": round(net_size, 2),
            "entry_date": datetime.now(timezone.utc).isoformat(),
            "reason": decision.get("reason", ""),
            "sfc_at_entry": decision.get("sfc_pct"),
            "confidence_at_entry": decision.get("confidence"),
            "slippage_entry": round(entry_cost, 2),
            "spread_cost_entry": round(spread_cost, 2),
            "time_slippage_pct": decision.get("time_slippage_pct", 0),
        })
        self.capital -= size  # commit full allocation
        self.trades.append({
            "id": len(self.trades) + 1,
            "type": "OPEN",
            "direction": pos_type,
            "date": datetime.now(timezone.utc).isoformat(),
            "price": price,
            "size": round(net_size, 2),
            "slippage": round(entry_cost, 2),
            "time_slippage_pct": decision.get("time_slippage_pct", 0),
            "reason": decision.get("reason", ""),
        })

    def _close_all_positions(self, price: float, reason: str, daily_volume: float = 0):
        for pos in self.positions:
            exit_cost = 0.0
            if _MI_AVAILABLE and daily_volume > 0:
                exit_cost = calculate_exit_cost(pos["size"], price, daily_volume)
                if exit_cost > pos["size"] * 0.5:
                    exit_cost = pos["size"] * 0.5
            # Bid-ask spread cost on exit too (see BID_ASK_SPREAD_PCT config)
            exit_cost += pos["size"] * BID_ASK_SPREAD_PCT

            gross_proceeds = pos["size"]
            if pos["type"] == "LONG":
                pnl = (price - pos["entry_price"]) / pos["entry_price"] * pos["size"]
                pnl_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100
            else:  # SHORT
                pnl = (pos["entry_price"] - price) / pos["entry_price"] * pos["size"]
                pnl_pct = (pos["entry_price"] - price) / pos["entry_price"] * 100

            net_pnl = pnl - exit_cost
            self.capital += gross_proceeds + net_pnl
            self.trades.append({
                "id": len(self.trades) + 1,
                "type": "CLOSE",
                "direction": pos["type"],
                "date": datetime.now(timezone.utc).isoformat(),
                "price": price,
                "size": pos["size"],
                "pnl": round(net_pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "slippage_exit": round(exit_cost, 2),
                "reason": reason,
            })
        self.positions = []

    def _update_pnl(self, price: float):
        unrealized = 0
        for pos in self.positions:
            if pos["type"] == "LONG":
                unrealized += (price - pos["entry_price"]) / pos["entry_price"] * pos["size"]
            else:  # SHORT
                unrealized += (pos["entry_price"] - price) / pos["entry_price"] * pos["size"]
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

        # Sharpe from DAILY snapshots (not per-trade equity_history).
        # Using equity_history would compute inter-trade returns which cannot
        # be annualized with √252 (that assumes daily returns) — if trades
        # happen every 5 minutes, √252 understates the factor by ~20x.
        # Crypto markets run 24/7 so annualization uses √365, not √252.
        daily_equities = sorted(self.daily_snapshots.values(), key=lambda x: x["date"])
        daily_rets = []
        for i in range(1, len(daily_equities)):
            prev_eq = daily_equities[i-1]["equity"]
            cur_eq = daily_equities[i]["equity"]
            if prev_eq > 0:
                daily_rets.append((cur_eq - prev_eq) / prev_eq)
        if daily_rets:
            mean_ret = sum(daily_rets) / len(daily_rets)
            std_ret = math.sqrt(sum((r - mean_ret)**2 for r in daily_rets) / len(daily_rets))
            sharpe = mean_ret / std_ret * math.sqrt(365) if std_ret > 0 else 0
        else:
            # Fall back to inter-trade returns if no daily data yet (early
            # cycles before first daily snapshot exists); label it clearly.
            rets = []
            for i in range(1, len(self.equity_history)):
                prev = self.equity_history[i-1]["equity"]
                cur = self.equity_history[i]["equity"]
                if prev > 0:
                    rets.append((cur - prev) / prev)
            mean_ret = sum(rets) / len(rets) if rets else 0
            std_ret = math.sqrt(sum((r - mean_ret)**2 for r in rets) / len(rets)) if len(rets) > 1 else 1
            sharpe = mean_ret / std_ret * math.sqrt(365) if std_ret > 0 else 0

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
        "execution_price": decision.get("execution_price"),
        "time_slippage_pct": decision.get("time_slippage_pct", 0),
        "is_short": decision.get("is_short", False),
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
