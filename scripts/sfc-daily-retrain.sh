#!/bin/bash
# ============================================================
# sfc-daily-retrain.sh — Auto-training for SFC ML Models
# Runs daily at 2AM via cron. Trains fast models only.
# Mamba step removed 2026-09-04: mamba compute is DISABLED in
# collect.py (SSM collapsed to zeros, inert) — train_mamba.py
# wrote mamba_weights.pth nothing reads. Kept reversible.
# ============================================================
cd /home/ubuntu/sfc || exit 1
# ML modules live in models/ (moved from repo root ~2026-07-16); put them on path
export PYTHONPATH="/home/ubuntu/sfc/models${PYTHONPATH:+:$PYTHONPATH}"
LOG="/home/ubuntu/sfc/logs/train_$(date +%Y%m%d).log"
mkdir -p "$(dirname "$LOG")"

echo "╔═══════════════════════════════════════════════" | tee -a "$LOG"
echo "║ SFC DAILY RETRAIN — $(date)" | tee -a "$LOG"
echo "╚═══════════════════════════════════════════════" | tee -a "$LOG"

# ── 1. ML Ensemble (existing, fast) ──
echo "" | tee -a "$LOG"
echo "[1/4] ML Ensemble retrain..." | tee -a "$LOG"
/usr/bin/python3 /home/ubuntu/sfc/models/ml_ensemble.py --retrain 2>&1 | tee -a "$LOG"
/usr/bin/python3 /home/ubuntu/sfc/models/ml_ensemble.py --evaluate 2>&1 | tee -a "$LOG"

# ── 2. HMM Regime Detector (from git history) ──
echo "" | tee -a "$LOG"
echo "[2/4] HMM Regime retrain..." | tee -a "$LOG"
/usr/bin/python3 -c "
from hmm_regime import HMMRegimeDetector
hmm = HMMRegimeDetector()
result = hmm.fit_from_git()
if result:
    result.save('/home/ubuntu/sfc/models/hmm_regime.pkl')
    print(f'[HMM] Retrained: {result.n_components} states, saved')
else:
    print('[HMM] Retrain failed or skipped')
" 2>&1 | tee -a "$LOG"

# ── 3. XGBoost Meta-Ensemble ──
echo "" | tee -a "$LOG"
echo "[3/4] XGBoost Meta retrain..." | tee -a "$LOG"
/usr/bin/python3 -c "
from ensemble_meta import train_from_git_history
model = train_from_git_history(verbose=False)
if model:
    model.save('/home/ubuntu/sfc/models/xgboost_meta.json')
    print(f'[XGB] Retrained: OK')
else:
    print('[XGB] Retrain failed or skipped')
" 2>&1 | tee -a "$LOG"

# ── 4. DRL Agent (if available) ──
echo "" | tee -a "$LOG"
echo "[4/4] DRL Agent retrain..." | tee -a "$LOG"
if [ -f "/home/ubuntu/sfc/train_drl_agent_script.py" ]; then
    /usr/bin/python3 /home/ubuntu/sfc/train_drl_agent_script.py 2>&1 | tail -5 | tee -a "$LOG"
    echo "[DRL] Done" | tee -a "$LOG"
else
    echo "[DRL] Skipped (no training script)" | tee -a "$LOG"
fi

# ── Summary ──
echo "" | tee -a "$LOG"
echo "✅ Daily retrain complete — $(date)" | tee -a "$LOG"
