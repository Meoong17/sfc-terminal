#!/usr/bin/env python3
"""
check_retrain_readiness.py — Cek kesiapan data SEBELUM menjalankan
weekly-model-train.sh, supaya tidak buang waktu retrain kalau data
belum cukup.

Usage:
    cd ~/sfc
    python3 check_retrain_readiness.py
"""
import json
import os
import sys

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_collection.json")

# Minimum thresholds found in each training script:
#   ensemble_meta.py (XGBoost): len(X) < 50 -> skip training
#   train_mamba.py / qlstm_model.py: seq_len(8) + ~20 buffer -> need ~28+
THRESHOLDS = {
    "xgboost (ensemble_meta.py)": 50,
    "mamba (train_mamba.py)": 28,
    "qlstm (qlstm_model.py, in sfc2/)": 28,
}


def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ {DATA_PATH} tidak ditemukan.")
        print("   Ini file yang diisi bertahap oleh collect.py setiap cycle —")
        print("   kalau belum ada sama sekali, pipeline mungkin belum pernah")
        print("   sempat menulis histori observasi.")
        sys.exit(1)

    with open(DATA_PATH) as f:
        data = json.load(f)

    labels = data.get("labels", [])
    features = data.get("features", [])

    total_obs = len(features)
    resolved = sum(1 for l in labels if l is not None)
    pending = sum(1 for l in labels if l is None)

    print("=" * 60)
    print("KESIAPAN DATA UNTUK RETRAIN")
    print("=" * 60)
    print(f"\nTotal observasi tercatat: {total_obs}")
    print(f"Label sudah resolved (siap dipakai training): {resolved}")
    print(f"Label masih pending (menunggu ~6 jam price-outcome): {pending}")

    print(f"\n{'-'*60}")
    print("Status per model:")
    print(f"{'-'*60}")
    all_ready = True
    for model_name, threshold in THRESHOLDS.items():
        ready = resolved >= threshold
        status = "✅ SIAP" if ready else f"⏳ BELUM ({resolved}/{threshold})"
        print(f"  {model_name}: {status}")
        if not ready:
            all_ready = False

    print(f"\n{'-'*60}")
    if all_ready:
        print("✅ Semua model punya data cukup — aman jalankan weekly-model-train.sh")
        print("\n   cd ~/sfc && bash weekly-model-train.sh")
    else:
        needed = max(t - resolved for t in THRESHOLDS.values() if resolved < t)
        # Label resolve ~6 jam setelah observasi dicatat (LABEL_LOOKAHEAD_MINUTES=360
        # di ml_ensemble.py), dan collect.py jalan tiap ~5 menit — jadi kira-kira
        # 1 observasi baru "matang" labelnya tiap ~5 menit setelah 6 jam awal.
        print(f"⏳ Belum semua model siap — masih butuh {needed} observasi resolved lagi.")
        print(f"   Karena label butuh ~6 jam untuk 'matang' (price-outcome window),")
        print(f"   dan collect.py jalan tiap ~5 menit, perkiraan kasar: tunggu")
        print(f"   pipeline jalan terus tanpa gangguan selama beberapa hari lagi.")
        print(f"   Jalankan script ini lagi nanti untuk cek progress.")


if __name__ == "__main__":
    main()
