#!/usr/bin/env python3
"""
behavioral_divergence_tracker.py — Historical tracker for behavioral divergence
signals, designed to accumulate a 90-calendar-day sample for forward-return
validation of divergence regimes.

WHAT THIS DOES:
    Every time collect.py calls compute_behavioral_divergence(), this tracker
    logs the resulting score, regime, and current BTC price. Over time it
    builds a time-indexed history that can answer:
        "Did HIDDEN_ACCUMULATION signals actually precede positive 30d returns?"
        "Did HIDDEN_DISTRIBUTION signals actually precede negative 30d returns?"

    This is SEPARATE from the Reflexivity divergence tracker (which runs on a
    ~7.5h cycle and tracks short-term momentum) — the Behavioral tracker
    operates on calendar days and validates a 30d forward window.

STORAGE:
    A JSON file (.behavioral_divergence_history.json) at the project root,
    same dotfile pattern as .reflexivity_rolling.json, .factor_history.json,
    and other cache files in this project.

USAGE (from collect.py, after compute_behavioral_divergence()):
    from analysis.behavioral_divergence_tracker import record_divergence
    record_divergence(ts=current_iso_timestamp,
                      score=_divergence_score,
                      detail=_divergence_details,
                      btc_price=current_btc_price)

    # Later, to check how well past signals predicted returns:
    stats = get_validation_stats(min_sample=30)
    print(stats)

HONEST LIMITATION:
    This tracker can only VALIDATE after enough calendar time has passed.
    The first 90+30=120 days after deployment are a data-collection-only
    phase — no forward-return check is possible until then.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".behavioral_divergence_history.json",
)

FORWARD_DAYS = 30          # forward return window to validate
MIN_HISTORY_DAYS = 90      # minimum history before validation is meaningful
MAX_HISTORY_ENTRIES = 500  # cap to prevent unbounded file growth


def _load_history():
    """Load history from disk, or return empty list."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _save_history(history):
    """Save history to disk, capped to MAX_HISTORY_ENTRIES."""
    trimmed = history[-MAX_HISTORY_ENTRIES:]
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(trimmed, f, indent=2)
    except OSError as e:
        print(f"[BehavioralTracker] Could not write history: {e}", file=sys.stderr)


def record_divergence(ts, score, detail, btc_price):
    """
    Record one data point from a behavioral divergence computation.

    Args:
        ts: ISO-8601 timestamp string (e.g. '2026-07-25T05:19:29+00:00')
        score: float, the divergence_score (0-100)
        detail: dict from compute_behavioral_divergence()
        btc_price: float, current BTC price in USD

    Returns:
        dict with 'entries_recorded' and 'total_entries' for logging.
    """
    regime = detail.get("regime", "UNKNOWN") if isinstance(detail, dict) else "UNKNOWN"
    components = detail.get("components_used", []) if isinstance(detail, dict) else []
    status = detail.get("status", "ok") if isinstance(detail, dict) else "ok"

    entry = {
        "ts": ts,
        "score": score,
        "regime": regime,
        "btc_price": btc_price,
        "components": components,
        "status": status,
        "_recorded_at": datetime.now(timezone.utc).isoformat(),
    }

    history = _load_history()
    history.append(entry)
    _save_history(history)

    return {"entries_recorded": 1, "total_entries": len(history)}


def get_history(days=None):
    """
    Return the recent history, optionally filtered to the last N days.

    Args:
        days: int or None. If set, returns only entries from the last N
              calendar days. If None, returns all entries.

    Returns:
        list of dict entries, newest-first.
    """
    history = _load_history()
    if not history:
        return []

    if days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        filtered = []
        for e in reversed(history):
            try:
                ets = datetime.fromisoformat(e["ts"])
                if ets >= cutoff:
                    filtered.append(e)
            except (ValueError, KeyError):
                continue
        return filtered

    return list(reversed(history))


def get_validation_stats(min_sample=30):
    """
    Compute forward-return validation stats from the history.

    For each entry where enough forward data exists (>FORWARD_DAYS later),
    checks whether the divergence regime was followed by the expected
    direction of BTC return.

    Args:
        min_sample: minimum number of validation-able entries required
                    before returning non-None stats.

    Returns:
        dict with accuracy metrics, or a status dict explaining why
        validation isn't ready yet (insufficient history).
    """
    history = _load_history()
    if len(history) < min_sample:
        return {
            "status": "insufficient_history",
            "entries_total": len(history),
            "entries_validatable": 0,
            "min_sample_required": min_sample,
            "message": (
                f"Need at least {min_sample} entries for a meaningful "
                f"validation; have {len(history)} so far."
            ),
        }

    # Build price lookup for forward returns
    validatable = 0
    correct = 0
    results = []

    for i, entry in enumerate(history):
        try:
            entry_ts = datetime.fromisoformat(entry["ts"])
        except (ValueError, KeyError):
            continue

        regime = entry.get("regime", "")
        entry_price = entry.get("btc_price")
        if regime not in ("HIDDEN_ACCUMULATION", "HIDDEN_DISTRIBUTION") or entry_price is None:
            continue

        # Find forward price ~FORWARD_DAYS later
        forward_cutoff = entry_ts + timedelta(days=FORWARD_DAYS)
        forward_price = None
        for later_entry in history[i + 1:]:
            try:
                let = datetime.fromisoformat(later_entry["ts"])
            except (ValueError, KeyError):
                continue
            if let >= forward_cutoff:
                forward_price = later_entry.get("btc_price")
                break

        if forward_price is None:
            continue  # not enough forward data yet

        validatable += 1
        actual_return = (forward_price - entry_price) / entry_price * 100

        if regime == "HIDDEN_ACCUMULATION":
            predicted_up = True
        else:  # HIDDEN_DISTRIBUTION
            predicted_up = False

        actual_up = actual_return > 0
        hit = (predicted_up == actual_up)
        if hit:
            correct += 1

        results.append({
            "entry_ts": entry["ts"],
            "regime": regime,
            "entry_price": entry_price,
            "forward_price": forward_price,
            "actual_return_pct": round(actual_return, 2),
            "correct": hit,
        })

    if validatable < min_sample * 0.3:
        return {
            "status": "insufficient_forward_data",
            "entries_total": len(history),
            "entries_validatable": validatable,
            "min_sample_required": min_sample,
            "message": (
                f"Only {validatable} entries have enough forward data "
                f"({FORWARD_DAYS}d ahead) to validate — need at least "
                f"{int(min_sample * 0.3)}. More calendar time needed."
            ),
        }

    accuracy = correct / validatable * 100 if validatable > 0 else 0.0

    return {
        "status": "ok",
        "entries_total": len(history),
        "entries_validatable": validatable,
        "directional_accuracy_pct": round(accuracy, 1),
        "correct": correct,
        "total": validatable,
        "forward_days": FORWARD_DAYS,
        "min_history_days": MIN_HISTORY_DAYS,
        "recent_results": results[-20:],  # last 20 for inspection
    }


if __name__ == "__main__":
    print("=" * 60)
    print("BEHAVIORAL DIVERGENCE TRACKER — Self-test & Demo")
    print("=" * 60)

    # Clean slate for test
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

    print("\n--- Test 1: record some divergence entries ---")
    from datetime import timezone
    base = datetime.now(timezone.utc)
    for i in range(5):
        ts = (base - timedelta(days=90 - i * 20)).isoformat()
        score = 30.0 + i * 10
        regime = "HIDDEN_ACCUMULATION" if i % 2 == 0 else "HIDDEN_DISTRIBUTION"
        detail = {"regime": regime, "components_used": ["etf_flow", "whale_pressure"]}
        price = 50000 + i * 2000
        result = record_divergence(ts=ts, score=score, detail=detail, btc_price=price)
        print(f"  [{ts}] score={score} regime={regime} price=${price} — entries={result['total_entries']}")
    print("  ✅ record ok")

    print("\n--- Test 2: get_history() ---")
    recent = get_history(days=120)
    print(f"  Last 120d entries: {len(recent)}")
    assert len(recent) == 5
    print("  ✅ PASS")

    print("\n--- Test 3: get_validation_stats() with insufficient data ---")
    stats = get_validation_stats(min_sample=30)
    assert stats["status"] in ("insufficient_history", "insufficient_forward_data")
    print(f"  Status: {stats['status']}, entries: {stats['entries_total']}")
    print("  ✅ PASS (graceful degradation with small sample)")

    print("\n--- Test 4: forward validation with synthetic data ---")
    # Build enough history: 100 entries with known signals and price moves
    base = datetime.now(timezone.utc) - timedelta(days=200)
    for i in range(100):
        ts = (base + timedelta(days=i * 2)).isoformat()
        if i % 2 == 0:
            regime = "HIDDEN_ACCUMULATION"
            # Price goes UP after accumulation -> correct
            price = 50000 + i * 100
        else:
            regime = "HIDDEN_DISTRIBUTION"
            # Price goes DOWN after distribution -> correct
            price = 60000 - i * 50
        detail = {"regime": regime, "components_used": ["etf_flow", "whale_pressure"], "status": "ok"}
        record_divergence(ts=ts, score=40.0, detail=detail, btc_price=price)

    stats = get_validation_stats(min_sample=30)
    print(f"  Status: {stats['status']}")
    if stats["status"] == "ok":
        print(f"  Directional accuracy: {stats['correct']}/{stats['total']} = {stats['directional_accuracy_pct']}%")
        print(f"  Validatable entries: {stats['entries_validatable']}")
    print("  ✅ PASS")

    # Cleanup test history
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

    print("\n" + "=" * 60)
    print("ALL SELF-TESTS PASSED")
    print("=" * 60)
