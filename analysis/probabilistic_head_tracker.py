#!/usr/bin/env python3
"""
probabilistic_head_tracker.py — Live-forward calibration tracking for
ProbabilisticHead (models/probabilistic_output.py), designed to be run
PERIODICALLY (e.g. weekly via cron) rather than as a one-off test.

WHY A SEPARATE TRACKER (not data_collection.json):
    data_collection.json's "features"/"dates" arrays are capped at 2000
    observations — at collect.py's 5-minute cycle, that's only ~7 days
    of retention (its cap is tuned for the CNN/LSTM's own short-term
    training needs, not for a 30-day-forward calibration check).
    price_log goes back 30 days but doesn't store effective_sfc itself.
    Neither is sufficient on its own for checking whether
    ProbabilisticHead's PREDICTED sfc_score distribution actually
    matches what REALIZED sfc_score does 7/30 days later. This tracker
    is a lightweight, purpose-built log (just ts + effective_sfc +
    confidence + regime — not full feature vectors) that can retain
    MONTHS of history cheaply, specifically for this calibration check.

WHAT "PERIODIC" MEANS HERE:
    1. log_cycle() gets called from collect.py every live cycle (~5 min)
       — same lightweight, non-blocking, try/except-wrapped pattern as
       the other trackers built this session.
    2. run_periodic_check() is meant to be invoked on a SCHEDULE (e.g.
       a weekly cron entry) — NOT every cycle, since checking
       calibration only makes sense once meaningful forward-return time
       has passed, and running it every 5 minutes would be wasted work
       recomputing the same insufficient-data message repeatedly. See
       "SUGGESTED CRON SETUP" below.

WHAT THIS CHECKS (mirrors the Schrödinger model calibration check, but
for ProbabilisticHead's own OUTPUT — same STANDARD, so results are
directly comparable to that earlier finding, e.g. "60%" or "75%"):
    For each historical point (with only data available at that time),
    reconstruct what ProbabilisticHead WOULD have predicted, then check
    whether the REALIZED effective_sfc N days later actually fell
    within the predicted 90% interval (q_05 to q_95) roughly 90% of the
    time, as a well-calibrated model's intervals should.

SUGGESTED CRON SETUP (run weekly, not every cycle):
    0 6 * * 1 cd ~/S && python3 analysis/probabilistic_head_tracker.py --check >> logs/prob_head_calibration.log 2>&1

USAGE:
    python3 analysis/probabilistic_head_tracker.py --check   # run the periodic check now
    (log_cycle() is called automatically from collect.py — see wiring instructions)
"""
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".probabilistic_head_history.json")
MIN_TRACKING_DAYS = 120  # matches the 120-day standard already adopted
                         # for behavioral_divergence_tracker this session
                         # — kept consistent rather than reintroducing a
                         # different number for a similar kind of check
FORWARD_HORIZONS_DAYS = [7, 30]
N_BOOTSTRAP = 2000
MAX_RETAINED_POINTS = 60000  # ~208 days at 5-min cycles — generous
                              # headroom beyond the 120-day minimum,
                              # cheap to store (just 5 small fields/entry,
                              # not full method-score vectors)


def log_cycle(effective_sfc, composite_confidence, regime, method_scores):
    """Called from collect.py every live cycle. Appends one point — does
    NOT run any analysis (that's run_periodic_check()'s job, on its own
    schedule). Fails silently — logging failures must never break the
    live pipeline."""
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE) as f:
                log = json.load(f)
        else:
            log = []

        log.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "effective_sfc": effective_sfc,
            "composite_confidence": composite_confidence,
            "regime": regime,
            "method_scores": method_scores,
        })

        if len(log) > MAX_RETAINED_POINTS:
            log = log[-MAX_RETAINED_POINTS:]

        with open(LOG_FILE, "w") as f:
            json.dump(log, f)
    except Exception as e:
        print(f"[ProbHeadTracker] Logging failed (non-fatal): {e}", file=sys.stderr)


def _bootstrap_diff_ci(group_a, group_b, n_bootstrap=N_BOOTSTRAP, ci=0.90):
    import random
    if len(group_a) < 2 or len(group_b) < 2:
        return None, None, None
    n_a, n_b = len(group_a), len(group_b)
    diffs = []
    for _ in range(n_bootstrap):
        sample_a = [group_a[random.randrange(n_a)] for _ in range(n_a)]
        sample_b = [group_b[random.randrange(n_b)] for _ in range(n_b)]
        diffs.append(sum(sample_b) / n_b - sum(sample_a) / n_a)
    diffs.sort()
    lo_idx = int((1 - ci) / 2 * n_bootstrap)
    hi_idx = int((1 + ci) / 2 * n_bootstrap) - 1
    return sum(group_b) / n_b - sum(group_a) / n_a, diffs[lo_idx], diffs[hi_idx]


def _find_forward_value(log, index, days_ahead, field):
    target_ts = datetime.fromisoformat(log[index]["ts"])
    target_seconds = days_ahead * 86400
    best, best_diff = None, float("inf")
    for j in range(index + 1, len(log)):
        pt_ts = datetime.fromisoformat(log[j]["ts"])
        diff_seconds = (pt_ts - target_ts).total_seconds()
        if diff_seconds < 0:
            continue
        gap = abs(diff_seconds - target_seconds)
        if gap < best_diff:
            best_diff = gap
            best = log[j][field]
        if diff_seconds > target_seconds + 86400:
            break
    return best


def run_periodic_check():
    """Meant to be invoked on a SCHEDULE (see module docstring's cron
    suggestion), not every collect.py cycle."""
    if not os.path.exists(LOG_FILE):
        print("[ProbHeadTracker] No log file yet — log_cycle() hasn't run.")
        return

    with open(LOG_FILE) as f:
        log = json.load(f)

    if len(log) < 20:
        print(f"[ProbHeadTracker] Only {len(log)} points logged — too early.")
        return

    first_ts = datetime.fromisoformat(log[0]["ts"])
    last_ts = datetime.fromisoformat(log[-1]["ts"])
    days_tracked = (last_ts - first_ts).total_seconds() / 86400
    print(f"Tracking span: {days_tracked:.1f} days ({len(log)} points)")
    if days_tracked < MIN_TRACKING_DAYS:
        print(f"[ProbHeadTracker] Need {MIN_TRACKING_DAYS} days — only "
              f"{days_tracked:.1f} so far. Re-run this check later "
              f"(e.g. via the weekly cron entry — see module docstring).")
        return

    try:
        from models.probabilistic_output import ProbabilisticHead
    except ImportError as e:
        print(f"[ProbHeadTracker] Could not import ProbabilisticHead: {e}")
        return

    head = ProbabilisticHead()

    print("\n" + "=" * 60)
    print(f"PERIODIC CALIBRATION CHECK — run at {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    for horizon in FORWARD_HORIZONS_DAYS:
        hits, total = 0, 0
        for i, entry in enumerate(log):
            fwd_sfc = _find_forward_value(log, i, horizon, "effective_sfc")
            if fwd_sfc is None:
                continue
            try:
                result = head.compute(
                    sfc_score=entry["effective_sfc"], method_scores=entry["method_scores"],
                    composite_confidence=entry["composite_confidence"], regime=entry["regime"],
                )
            except Exception:
                continue
            q05, q95 = result["quantiles"]["q_05"], result["quantiles"]["q_95"]
            if q05 <= fwd_sfc <= q95:
                hits += 1
            total += 1

        if total > 0:
            pct = hits / total * 100
            print(f"  {horizon}d 90% interval calibration: {hits}/{total} = {pct:.1f}% "
                  f"(ideal ~90%; well below = overconfident, well above = underconfident)")
        else:
            print(f"  {horizon}d: insufficient data")

    print(f"\n[ProbHeadTracker] Check complete. Run again per the cron "
          f"schedule to monitor calibration DRIFT over time — a single "
          f"check is a snapshot, not a trend.")


if __name__ == "__main__":
    print("=== Self-test ===\n")

    print("--- Test 1: log_cycle() appends without crashing ---")
    test_file = "/tmp/test_prob_head_log.json"
    LOG_FILE_BACKUP = LOG_FILE
    globals()["LOG_FILE"] = test_file
    if os.path.exists(test_file):
        os.remove(test_file)
    log_cycle(20.0, 0.15, "NORMAL", [10.0] * 6)
    log_cycle(55.0, 0.08, "CRISIS", [50.0] * 6)
    with open(test_file) as f:
        loaded = json.load(f)
    assert len(loaded) == 2, f"FAIL: {len(loaded)}"
    print(f"✅ PASS: {len(loaded)} entries logged\n")

    print("--- Test 2: retention cap trims oldest entries ---")
    globals()["MAX_RETAINED_POINTS"] = 3
    for _ in range(5):
        log_cycle(20.0, 0.15, "NORMAL", [10.0] * 6)
    with open(test_file) as f:
        loaded = json.load(f)
    assert len(loaded) == 3, f"FAIL: expected 3, got {len(loaded)}"
    print(f"✅ PASS: retained {len(loaded)} (cap enforced)\n")
    globals()["MAX_RETAINED_POINTS"] = 60000

    print("--- Test 3: _find_forward_value finds closest future point ---")
    fake_log = [
        {"ts": "2026-01-01T00:00:00+00:00", "effective_sfc": 20.0},
        {"ts": "2026-01-08T00:00:00+00:00", "effective_sfc": 35.0},
    ]
    fwd = _find_forward_value(fake_log, 0, 7, "effective_sfc")
    assert fwd == 35.0, f"FAIL: {fwd}"
    print("✅ PASS\n")

    print("--- Test 4: _bootstrap_diff_ci detects clear difference ---")
    import random
    random.seed(1)
    a = [random.gauss(2.0, 1.0) for _ in range(50)]
    b = [random.gauss(-3.0, 1.0) for _ in range(50)]
    diff, lo, hi = _bootstrap_diff_ci(a, b)
    assert hi < 0, "FAIL"
    print(f"✅ PASS: diff={diff:.2f}\n")

    os.remove(test_file)
    globals()["LOG_FILE"] = LOG_FILE_BACKUP
    print("ALL SELF-TESTS PASSED")

    if "--check" in sys.argv:
        print("\n" + "=" * 60)
        print("Running periodic check against real log file...")
        print("=" * 60)
        run_periodic_check()
