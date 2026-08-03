#!/usr/bin/env python3
"""
sfc_weekly_ai_analysis.py — Weekly Hermes LLM analyst brief for SFC Terminal
---------------------------------------------------------------------------
Reads the latest data.json, extracts the key dashboard metrics, invokes Hermes
(the same LLM that powers this agent) to write a structured market analysis,
and writes it to ai_analysis.json in the SFC repo.

The dashboard fetches /ai_analysis.json and renders it in the "AI ANALYST"
panel. This script is meant to run ONCE A WEEK (via cron) so the analysis is
refreshed weekly, not on every 5-minute data cycle.

Writes ai_analysis.json:
{
  "ts": <ISO timestamp>,
  "model": "deepseek-v4-flash (Hermes)",
  "bias": "Bullish"|"Neutral"|"Bearish",
  "summary": "<2-3 sentence overview>",
  "risks": [["Risk name", "detail"], ...],
  "action": "<recommended action>"
}

Usage:
    python sfc_weekly_ai_analysis.py [--data path/to/data.json] [--out path/ai_analysis.json]
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/ubuntu/sfc")
HERMES = "hermes"  # hermes CLI in PATH


def pick_metrics(d: dict) -> dict:
    """Extract the fields the analyst needs from data.json."""
    f = d.get("factors", {}) or {}
    return {
        "ts": d.get("ts"),
        "btc": d.get("btc"),
        "btc_24h": d.get("btc_24h"),
        "sfc_effective": d.get("sfc_effective"),
        "sfc_base": d.get("sfc_base"),
        "regime": d.get("regime"),
        "zone": d.get("zone"),
        "regime_prob": d.get("regime_prob"),
        "transition_risk": d.get("transition_risk"),
        "factors": {k: f.get(k) for k in ("Lt", "St", "Rt", "Ft", "Sc")},
        "composite_confidence": d.get("composite_confidence"),
        "method_agreement": d.get("method_agreement"),
        "rsi_14": d.get("rsi_14"),
        "sopr_proxy": d.get("sopr_proxy"),
        "cascade_risk": d.get("cascade_risk"),
        "fng": d.get("fng"),
        "dvol": d.get("dvol"),
        "glf_score": d.get("glf_score"),
        "sli_score": d.get("sli_score"),
        "mpi_score": d.get("mpi_score"),
        "wfv_gap_7d": d.get("wfv_gap_7d"),
        "wfv_gap_30d": d.get("wfv_gap_30d"),
        "cont_prob_90d": d.get("cont_prob_90d"),
        "tail_risk_score": d.get("tail_risk_score"),
        "kelly_fraction": d.get("kelly_fraction"),
        "readiness_score": d.get("readiness_score"),
        "signal": d.get("signal"),
    }


def run_hermes(prompt: str) -> str:
    """Invoke Hermes CLI in one-shot mode and return its text reply."""
    try:
        res = subprocess.run(
            [HERMES, "chat", "-q", prompt, "-Q"],
            capture_output=True, text=True, timeout=600,
        )
        return (res.stdout or "").strip()
    except Exception as e:
        print(f"⚠ Hermes call failed: {e}", file=sys.stderr)
        return ""


def extract_json(text: str) -> dict:
    """Pull a JSON object out of Hermes' reply (it may wrap it in prose)."""
    if not text:
        return {}
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Strip markdown code fences
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass
    # Find first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return {}


def build_prompt(m: dict) -> str:
    return f"""You are the SFC Terminal AI Analyst. Analyze the following live quantitative stress metrics for Bitcoin and produce a concise, honest weekly market brief.

STRICT OUTPUT FORMAT: Return ONLY a single JSON object with EXACTLY these keys (no markdown, no prose around it):
{{
  "bias": "Bullish" or "Neutral" or "Bearish",
  "summary": "2-3 sentence overview of current market conditions citing the real numbers above",
  "risks": [["Risk Name", "1 sentence detail"], ...]  (2-4 items, from the data, not generic),
  "action": "Recommended action in one short sentence"
}}

MANDATORY: Reply ENTIRELY in ENGLISH. All keys' values (summary, risks, action) must be written in English only. Do not use any other language.

Do NOT invent data not present. If a metric is missing/null, do not fabricate it. Keep the tone measured and data-driven.

Live metrics:
{json.dumps(m, indent=2, ensure_ascii=False)}"""


def main():
    ap = argparse.ArgumentParser(description="Generate weekly Hermes LLM analyst brief")
    ap.add_argument("--data", default=str(REPO / "data.json"))
    ap.add_argument("--out", default=str(REPO / "ai_analysis.json"))
    args = ap.parse_args()

    dp = Path(args.data)
    if not dp.exists():
        print(f"⚠ data.json not found: {dp}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(dp.read_text(encoding="utf-8"))
    m = pick_metrics(data)

    print("Asking Hermes to analyze...", file=sys.stderr)
    reply = run_hermes(build_prompt(m))
    if not reply:
        print("❌ Hermes returned empty — leaving ai_analysis.json untouched.", file=sys.stderr)
        sys.exit(1)

    parsed = extract_json(reply)
    # Normalize
    bias = str(parsed.get("bias") or "Neutral").capitalize()
    if bias not in ("Bullish", "Neutral", "Bearish"):
        bias = "Neutral"
    out = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": "deepseek-v4-flash (Hermes)",
        "bias": bias,
        "summary": parsed.get("summary") or reply[:600],
        "risks": parsed.get("risks") or [],
        "action": parsed.get("action") or "Monitor.",
        "source_ts": m.get("ts"),
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ ai_analysis.json written to {out_path}")
    print(f"   bias={out['bias']} · summary={out['summary'][:80]}...")
    print(f"   risks={len(out['risks'])} · action={out['action']}")


if __name__ == "__main__":
    main()
