#!/usr/bin/env python3
"""
market_impact.py — Almgren-Chriss Market Impact Model for SFC
==============================================================
Calculates realistic execution costs for paper trading.

Formula (simplified Almgren-Chriss):
    slippage_pct = spread_pct * 0.5 + impact_const * (|Q| / V)^0.5

Where:
    Q = trade size (USD)
    V = 24h market volume (USD)
    σ = daily volatility
    spread_pct = estimated bid-ask spread

Usage:
    from market_impact import calc_slippage
    cost_pct = calc_slippage(trade_size=12500, daily_volume=25e9, volatility=0.02)
"""

import math
from typing import Optional

# ── Market Impact Parameters ──
# Almgren-Chriss constants calibrated for crypto
PERMANENT_IMPACT_FACTOR = 0.1    # γ: permanent impact coefficient
TEMPORARY_IMPACT_FACTOR = 0.05   # η: temporary impact coefficient
EXECUTION_HORIZON_FRAC = 0.1     # Execute over 10% of the day
BASE_SPREAD_PCT = 0.0005         # 0.05% base spread (BTC on Binance)
MIN_SLIPPAGE_PCT = 0.0001        # 0.01% minimum
MAX_SLIPPAGE_PCT = 0.05          # 5% cap

# Volume thresholds (which volume source to use)
DEFAULT_BTC_VOLUME_24H = 25e9    # $25B default BTC daily volume


def calc_slippage(
    trade_size_usd: float,
    daily_volume_usd: Optional[float] = None,
    volatility: float = 0.02,
    spread_pct: float = BASE_SPREAD_PCT,
) -> float:
    """Calculate Almgren-Chriss slippage as a fraction of trade value.

    Args:
        trade_size_usd: Size of the trade in USD.
        daily_volume_usd: 24h market volume in USD (default: $25B).
        volatility: Daily volatility as decimal (default: 0.02 = 2%).
        spread_pct: Bid-ask spread as decimal (default: 0.0005 = 0.05%).

    Returns:
        Slippage cost as a fraction of trade value (0.001 = 0.1%).
    """
    if trade_size_usd <= 0 or daily_volume_usd is not None and daily_volume_usd <= 0:
        return MIN_SLIPPAGE_PCT

    if daily_volume_usd is None:
        daily_volume_usd = DEFAULT_BTC_VOLUME_24H

    # Participation rate: how much of daily volume our trade represents
    participation = min(1.0, trade_size_usd / daily_volume_usd)  # cap at 100%

    # Execution horizon: assume T days to execute (partial fill)
    T = max(0.01, EXECUTION_HORIZON_FRAC)

    # Permanent impact (price moves permanently due to information leakage)
    # γ * σ * (Q/V)^0.5
    perm_impact = PERMANENT_IMPACT_FACTOR * volatility * math.sqrt(participation)

    # Temporary impact (price moves due to our order flow, reverses after)
    # η * σ * (Q/(V*T))^0.5
    temp_impact = TEMPORARY_IMPACT_FACTOR * volatility * math.sqrt(participation / T)

    # Total slippage = half-spread + permanent + temporary
    half_spread = spread_pct * 0.5
    total = half_spread + perm_impact + temp_impact

    # Clamp
    return max(MIN_SLIPPAGE_PCT, min(MAX_SLIPPAGE_PCT, total))


def calc_effective_entry_exit(
    trade_size_usd: float,
    entry_price: float,
    exit_price: float,
    daily_volume_usd: Optional[float] = None,
    volatility: float = 0.02,
) -> dict:
    """Calculate effective entry/exit prices after slippage.

    Args:
        trade_size_usd: Size of position in USD.
        entry_price: Intended entry price.
        exit_price: Intended exit price.
        daily_volume_usd: 24h volume (optional).
        volatility: Daily volatility.

    Returns:
        Dict with effective_entry, effective_exit, slippage_pct,
        entry_cost, exit_cost, total_cost.
    """
    entry_slippage = calc_slippage(trade_size_usd, daily_volume_usd, volatility)
    exit_slippage = calc_slippage(trade_size_usd, daily_volume_usd, volatility)

    # For LONG: entry = ask (higher), exit = bid (lower)
    effective_entry = entry_price * (1.0 + entry_slippage)
    effective_exit = exit_price * (1.0 - exit_slippage)

    entry_cost = trade_size_usd * entry_slippage
    exit_cost = trade_size_usd * exit_slippage

    return {
        "effective_entry": round(effective_entry, 2),
        "effective_exit": round(effective_exit, 2),
        "entry_slippage_pct": round(entry_slippage * 100, 4),
        "exit_slippage_pct": round(exit_slippage * 100, 4),
        "entry_cost_usd": round(entry_cost, 2),
        "exit_cost_usd": round(exit_cost, 2),
        "total_cost_usd": round(entry_cost + exit_cost, 2),
        "total_cost_pct": round((entry_slippage + exit_slippage) * 100, 4),
    }


def estimate_from_data(data: dict, trade_size_usd: float) -> dict:
    """Estimate slippage from live data.json values.

    Uses dvol (daily volume index) and btc_24h to estimate volume and vol.
    """
    dvol = None
    if "dvol" in data:
        try:
            dvol = float(data["dvol"])
        except (TypeError, ValueError):
            pass

    btc_24h = None
    if "btc_24h" in data:
        try:
            btc_24h = abs(float(data["btc_24h"])) / 100.0  # convert to decimal
        except (TypeError, ValueError):
            pass

    # Estimate daily volume from dvol (if available, scale from $25B baseline)
    if dvol is not None and dvol > 0:
        # dvol is a liquidity index 0-100, 50 = normal
        daily_vol_est = DEFAULT_BTC_VOLUME_24H * (0.3 + 0.7 * min(1.0, dvol / 50.0))
    else:
        daily_vol_est = DEFAULT_BTC_VOLUME_24H

    volatility = btc_24h if btc_24h is not None else 0.02

    return calc_effective_entry_exit(
        trade_size_usd=trade_size_usd,
        entry_price=float(data.get("btc", 60000)),
        exit_price=float(data.get("btc", 60000)) * 1.01,  # placeholder
        daily_volume_usd=daily_vol_est,
        volatility=volatility,
    )


# ════════════════════════════════════════════════════════════════
# Standalone Test
# ════════════════════════════════════════════════════════════════


def main() -> None:
    """Test market impact model with various trade sizes."""
    print("=" * 60)
    print("Almgren-Chriss Market Impact — Test")
    print("=" * 60)

    # Scenario: different trade sizes
    test_cases = [
        ("Retail $1K", 1_000),
        ("Medium $12.5K (max position)", 12_500),
        ("Large $100K", 100_000),
        ("Whale $1M", 1_000_000),
    ]

    print(f"\n{'Label':<25} {'Size':>10} {'Slippage':>10} {'Cost':>10}")
    print("-" * 55)

    for label, size in test_cases:
        slip = calc_slippage(size, daily_volume_usd=DEFAULT_BTC_VOLUME_24H)
        cost = size * slip
        print(f"{label:<25} {size:>10,.0f} {slip*100:>9.4f}% ${cost:>7,.2f}")

    # Scenario: different volume conditions
    print("\n── Volume Sensitivity (trade=$12.5K) ──")
    for label, vol in [("Low vol 10%", 2.5e9), ("Normal 100%", 25e9), ("High vol 500%", 125e9)]:
        slip = calc_slippage(12_500, daily_volume_usd=vol)
        print(f"  {label:<20} → slippage {slip*100:.4f}%")

    # Scenario: with dvol from data.json
    print("\n── From data.json ──")
    import json, os
    data_path = os.path.join(os.path.dirname(__file__), "data.json")
    try:
        with open(data_path) as f:
            data = json.load(f)
        result = estimate_from_data(data, 12_500)
        for k, v in result.items():
            print(f"  {k}: {v}")
    except FileNotFoundError:
        print("  (data.json not found)")

    print("\n✓ Market impact model test complete")


if __name__ == "__main__":
    main()
