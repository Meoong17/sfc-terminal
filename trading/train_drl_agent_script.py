#!/usr/bin/env python3
"""
train_drl_agent_script.py — M68 DRL training pipeline
==========================================================

Previously: trading/drl_agent.py had a complete Q-learning implementation
(CryptoPortfolioEnv, QLearningAgent, train_drl_agent(), save()/load()) —
but NOTHING in collect.py ever called train_drl_agent() or loaded a saved
agent. collect.py's get_trading_signal(_drl_market_state) was always
called WITHOUT an agent argument, meaning the `if agent is not None:`
branch inside get_trading_signal() was permanently unreachable in
production, and the simple 4-branch rule-based fallback ran every single
cycle instead — verified against live data.json (m68_drl_signal
predictions matched the manual rule-based calculation exactly). "M68
DRL" was a misleading name for a rule-based heuristic, not a bug that
crashed anything, but a genuine "sounds sophisticated, isn't" gap.

This script closes that gap: extracts historical market states from git
history of data.json (same field mapping collect.py itself already uses
for _drl_market_state), trains a QLearningAgent on it, and saves the
result so collect.py can load and actually use it.

Usage:
    cd ~/sfc
    python3 train_drl_agent_script.py
    (run periodically, e.g. weekly alongside the other model retrains —
     see scripts/weekly-model-train.sh for the pattern this should be added to)
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drl_agent import train_drl_agent, QLearningAgent

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(SFC_DIR, "models", "drl_agent.pkl")
MIN_HISTORY_POINTS = 30  # below this, training is too noisy to trust


def extract_market_state_history():
    """
    Extract (stress, rsi, price, momentum) history from git log of
    data.json — same pattern used by hmm_regime.py's _extract_snapshots()
    and ensemble_meta.py's extract_historical_snapshots(), and the SAME
    field mapping collect.py itself uses to build _drl_market_state:
        stress = sfc_effective / 100.0
        rsi = rsi_14
        price = btc
        momentum = btc_24h / 100.0

    Returns:
        list of dicts with keys stress/rsi/price/momentum, oldest-first,
        or [] if git history is unavailable / too short.
    """
    try:
        # Limit to last 500 commits to avoid timeout with 6k+ data.json changes
        result = subprocess.check_output(
            ["git", "log", "--oneline", "--all", "--diff-filter=M",
             "--reverse", "-n", "500", "--", "data.json"],
            text=True, timeout=30, cwd=SFC_DIR,
        ).strip().split("\n")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[DRLTrain] git log failed: {e} — are you running this from "
              f"the actual repo (with .git), not an extracted zip?", file=sys.stderr)
        return []

    result = [r for r in result if r.strip()]
    history = []
    for line in result:
        commit_hash = line.split()[0] if line.split() else None
        if not commit_hash:
            continue
        try:
            content = subprocess.check_output(
                ["git", "show", f"{commit_hash}:data.json"],
                text=True, timeout=10, cwd=SFC_DIR,
            )
            snap = json.loads(content)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                json.JSONDecodeError):
            continue

        effective_sfc = snap.get("sfc_effective")
        rsi = snap.get("rsi_14")
        btc = snap.get("btc")
        chg = snap.get("btc_24h")
        if effective_sfc is None or btc is None:
            continue  # skip incomplete snapshots rather than fabricating defaults

        history.append({
            "stress": effective_sfc / 100.0,
            "rsi": rsi if rsi is not None else 50.0,
            "price": btc,
            "momentum": (chg or 0) / 100.0,
        })

    return history


def main():
    print("=" * 60)
    print("M68 DRL AGENT TRAINING")
    print("=" * 60)

    history = extract_market_state_history()
    print(f"\nExtracted {len(history)} historical market states from git")

    if len(history) < MIN_HISTORY_POINTS:
        print(f"⚠ Only {len(history)} points (need >= {MIN_HISTORY_POINTS}) — "
              f"skipping training. Let the pipeline accumulate more git "
              f"history first, then re-run this script.")
        sys.exit(1)

    print(f"\nTraining Q-learning agent on {len(history)} states...")
    agent = train_drl_agent(history, episodes=500, state_bins=10)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    agent.save(MODEL_PATH)

    print(f"\n✅ Trained agent saved to {MODEL_PATH}")
    print(f"   Q-table size: {len(agent.q_table)} discrete states learned")
    print(f"   Final epsilon: {agent.epsilon:.4f}")
    print(f"\nNext cycle of collect.py will automatically load this agent")
    print(f"(see the updated M68 section — falls back to rule-based signal")
    print(f"if this file is missing or fails to load, so this is safe to")
    print(f"run without risk of breaking the pipeline if something's wrong).")


if __name__ == "__main__":
    main()
