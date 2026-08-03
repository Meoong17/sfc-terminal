#!/usr/bin/env python3
"""
SFC Regime Consolidation (P0) — single source of truth for regime label.
=======================================================================
The pipeline currently emits FOUR regime labels that can disagree for the
same day (see analysis/gap_analysis_dokumen_vs_model.md §4.1):
    regime        : main ensemble (NORMAL/STRESS/CAPITULATION) + prob
    hmm_regime    : HMM (BULL/BEAR/SIDEWAYS/CRISIS) + crisis_prob
    adv_regime    : advanced (NORMAL/STRESS/CAPITULATION) + crisis_prob
    behavior_state: IMBS L5 overlay (ACCUMULATION/EXPANSION/EUPHORIA/
                    DISTRIBUTION/PANIC)

This module consolidates them into ONE consensus regime label for
institutional display, plus a conflict flag + agreement score so the
dashboard can honestly show "these subsystems disagree" instead of four
confusing labels. It is DISPLAY-ONLY — it does NOT change sfc_effective /
signal / kelly_fraction.

CONSENSUS RULE (defensible, not arbitrary):
    Stress-domain labels (main regime, hmm, adv) are mapped onto a common
    severity axis. The most-stressed source takes precedence (conservative
    for a family-office / risk context). When behavior_state is PANIC or
    DISTRIBUTION it reinforces stress; EXPANSION/ACCUMULATION reinforce calm.

    Mapping to stress severity (0-100):
        main regime : NORMAL=20, STRESS=55, CAPITULATION=85
        hmm regime  : BULL=20, SIDEWAYS=45, BEAR=60, CRISIS=85
        adv regime  : NORMAL=20, STRESS=55, CAPITULATION=85

    Consensus severity = max of available. Label buckets:
        0-35   BULLISH     (calm / expansion / accumulation)
        35-60  ELEVATED    (sideways / distribution / moderate stress)
        60+    STRESSED    (stress / bear / crisis / capitulation)

    Conflict = count of sources whose bucket differs from the consensus
    bucket. Agreement = fraction of sources matching consensus bucket.

Usage:
    from data_sources.regime_consolidation import consolidate_regime
    out = consolidate_regime(
        regime='STRESS', regime_prob=0.434,
        hmm_regime='SIDEWAYS', hmm_crisis_prob=0.05,
        adv_regime='CRISIS', adv_crisis_prob=0.2,
        behavior_state='EXPANSION',
    )
"""
import os, json, time

SFC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_FILE = os.path.join(SFC_DIR, ".regime_consolidation_cache.json")
CACHE_TTL = 300  # 5 min — regime inputs update intraday


# Stress-severity mapping per subsystem label.
_MAIN_MAP = {"NORMAL": 20, "STRESS": 55, "CAPITULATION": 85}
_HMM_MAP = {"BULL": 20, "SIDEWAYS": 45, "BEAR": 60, "CRISIS": 85}
# The advanced detector (ml/sfc_advanced.py RegimeDetector) outputs in the
# SAME 4-regime label space as the main HMM (BULL/BEAR/SIDEWAYS/CRISIS).
# It previously listed only NORMAL/STRESS/CAPITULATION here, so adv_regime
# labels (e.g. CRISIS) were silently DROPPED from the consensus — a hidden
# source of "inconsistent assessment". Align it with the actual output space.
_ADV_MAP = {"BULL": 20, "SIDEWAYS": 45, "BEAR": 60, "CRISIS": 85,
            "NORMAL": 20, "STRESS": 55, "CAPITULATION": 85}
# Behavior-state reinforces direction but is NOT a stress bucket by itself.
_BEHAVIOR_STRESS = {"PANIC": 85, "DISTRIBUTION": 65, "EUPHORIA": 45,
                    "EXPANSION": 20, "ACCUMULATION": 25}


def _bucket(severity):
    if severity >= 60:
        return "STRESSED"
    if severity >= 35:
        return "ELEVATED"
    return "BULLISH"


def _load_cache():
    try:
        with open(_CACHE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(state):
    try:
        with open(_CACHE_FILE, "w") as f:
            json.dump(state, f)
    except OSError:
        pass


def consolidate_regime(regime=None, regime_prob=None, hmm_regime=None,
                       hmm_crisis_prob=None, adv_regime=None,
                       adv_crisis_prob=None, behavior_state=None):
    """Produce a single consensus regime label + conflict/agreement metrics.

    All args optional; a missing source is excluded from consensus (does not
    force a label). Returns (label, details dict).
    """
    cached = _load_cache()
    now = time.time()
    # Cache keyed by the full input tuple so different live inputs never
    # collide (same pattern as tail_risk_engine).
    _inputs = (str(regime), str(hmm_regime), str(adv_regime),
               str(behavior_state), str(regime_prob), str(hmm_crisis_prob),
               str(adv_crisis_prob))
    _key = "|".join(_inputs)
    if (cached.get("key") == _key and cached.get("ts")
            and now - cached.get("ts", 0) < CACHE_TTL):
        return cached.get("label", "UNKNOWN"), cached.get("details", {"status": "cached"})

    sources = {}   # name -> severity
    if regime is not None and str(regime).upper() in _MAIN_MAP:
        sources["regime"] = _MAIN_MAP[str(regime).upper()]
    if hmm_regime is not None and str(hmm_regime).upper() in _HMM_MAP:
        sources["hmm"] = _HMM_MAP[str(hmm_regime).upper()]
    if adv_regime is not None and str(adv_regime).upper() in _ADV_MAP:
        sources["adv"] = _ADV_MAP[str(adv_regime).upper()]

    # Behavior-state is direction evidence, blended at lower weight into the
    # consensus so a lone PANIC cannot override three calm subsystems, but a
    # PANIC alongside STRESS/BEAR reinforces it.
    behavior_sev = None
    if behavior_state is not None and str(behavior_state).upper() in _BEHAVIOR_STRESS:
        behavior_sev = _BEHAVIOR_STRESS[str(behavior_state).upper()]
        sources["behavior"] = behavior_sev

    if not sources:
        label, status = "UNKNOWN", "unavailable"
        details = {"status": status, "available": False,
                   "reason": "No regime source supplied."}
        _save_cache({"label": label, "details": details, "ts": now, "key": _key})
        return label, details

    # Consensus severity: max of structural stress sources (regime/hmm/adv),
    # with behavior as a modifier that can only pull it toward the extremes
    # when it already agrees (never flips a calm structural reading to panic).
    structural = [s for name, s in sources.items() if name != "behavior"]
    sev = max(structural) if structural else 0
    if behavior_sev is not None and structural:
        # If behavior agrees with the structural direction, edge toward it.
        if behavior_sev >= 60 and sev >= 45:
            sev = max(sev, 70)
        elif behavior_sev <= 30 and sev <= 30:
            sev = min(sev, 25)
    elif not structural:
        sev = behavior_sev

    label = _bucket(sev)

    # Conflict: how many sources disagree with the consensus bucket.
    consensus_bucket = label
    votes = []
    for name, sev_src in sources.items():
        votes.append({"source": name, "label": str(regime if name == "regime" else
                       hmm_regime if name == "hmm" else adv_regime if name == "adv" else
                       behavior_state), "severity": sev_src, "bucket": _bucket(sev_src)})
    matching = sum(1 for v in votes if v["bucket"] == consensus_bucket)
    conflict_sources = [v["source"] for v in votes if v["bucket"] != consensus_bucket]
    agreement = round(matching / len(votes), 3) if votes else None

    details = {
        "status": "ok",
        "available": True,
        "label": label,
        "severity": round(sev, 1),
        "consensus_bucket": consensus_bucket,
        "sources": votes,
        "n_sources": len(votes),
        "agreement": agreement,
        "conflict": bool(conflict_sources),
        "conflict_sources": conflict_sources,
        "source_labels": {
            "regime": regime, "hmm_regime": hmm_regime,
            "adv_regime": adv_regime, "behavior_state": behavior_state,
        },
        "rule": "Consensus = max stress of structural sources (regime/hmm/adv); "
                "behavior_state is a reinforcing modifier. Display-only.",
        "ts": now,
    }
    _save_cache({"label": label, "details": details, "ts": now, "key": _key})
    return label, details


if __name__ == "__main__":
    # Self-test with the live-observed contradiction (regime=STRESS, hmm=SIDEWAYS,
    # adv=CRISIS, behavior=EXPANSION) plus calm and crisis cases. Note: adv now
    # uses the 4-regime space (CRISIS=85), so it participates in the consensus.
    import sys
    live = dict(regime="STRESS", regime_prob=0.434, hmm_regime="SIDEWAYS",
                hmm_crisis_prob=0.05, adv_regime="CRISIS", adv_crisis_prob=0.2,
                behavior_state="EXPANSION")
    calm = dict(regime="NORMAL", hmm_regime="BULL", adv_regime="NORMAL",
                behavior_state="EXPANSION")
    crisis = dict(regime="CAPITULATION", hmm_regime="CRISIS", adv_regime="CAPITULATION",
                  behavior_state="PANIC")
    for name, kw in (("LIVE", live), ("CALM", calm), ("CRISIS", crisis)):
        lab, det = consolidate_regime(**kw)
        print(f"{name:7s} label={lab:9s} sev={det.get('severity')} "
              f"agreement={det.get('agreement')} conflict={det.get('conflict')} "
              f"conflict_sources={det.get('conflict_sources')}")
