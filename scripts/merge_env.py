#!/usr/bin/env python3
"""
merge_env.py — Gabungkan .env lama dengan key baru tanpa menghilangkan nilai existing.

Aturan penggabungan:
  - Key yang SUDAH ADA nilainya di .env lama -> nilai lama DIPERTAHANKAN
  - Key yang BELUM ADA atau KOSONG di .env lama -> diisi dari NEW_KEYS di bawah
  - Key lain yang ada di .env lama tapi tidak dikenal script ini -> tetap dipertahankan apa adanya
  - Backup otomatis dibuat sebelum menimpa .env

Cara pakai:
    cd ~/sfc
    python3 merge_env.py

Aman dijalankan berkali-kali (idempotent) — kalau key baru sudah pernah
digabungkan sebelumnya, run kedua tidak akan mengubah apa-apa.
"""
import os
import shutil
import sys
from datetime import datetime

ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")

# Key baru yang perlu ditambahkan jika belum ada / masih kosong di .env lama.
# Hanya dipakai sebagai fallback -- kalau key ini SUDAH punya nilai di .env
# lama, nilai lama itu yang menang, bukan nilai di sini.
#
# SECURITY: dulu ada nilai key ASLI di sini yang ke-commit ke repo PUBLIK.
# Sekarang dikosongkan supaya tidak ada secret hidup di source code. Script
# ini cuma bikin placeholder `KEY=` kosong yang diisi manual di .env. Nilai
# yang sudah terisi di .env TIDAK akan tertimpa. (Key yang pernah ter-expose
# tetap idealnya di-rotate — lihat README/security docs.)
NEW_KEYS = {
    "GOLDAPI_KEY": "",
    "TWELVEDATA_KEY": "",
    "ALPHAVANTAGE_KEY": "",
}


def parse_env_lines(path):
    """Parse .env preserving original lines, and a dict of key->value for lookups."""
    if not os.path.exists(path):
        return [], {}
    with open(path) as f:
        lines = f.readlines()
    values = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, val = stripped.partition("=")
        values[key.strip()] = val.strip()
    return lines, values


def main():
    if not os.path.exists(ENV_PATH):
        print(f"⚠ Tidak ada .env di {ENV_PATH} — membuat baru dengan placeholder key inti + key baru.")
        core_placeholder = (
            "# SFC Terminal Pro — Environment Variables\n\n"
            "CMC_API_KEY=\n"
            "FRED_API_KEY=\n"
            "CRYPTOPANIC_KEY=\n"
            "COINGLASS_API_KEY=\n"
        )
        with open(ENV_PATH, "w") as f:
            f.write(core_placeholder)
        lines, existing = parse_env_lines(ENV_PATH)
    else:
        lines, existing = parse_env_lines(ENV_PATH)
        # Backup dulu sebelum menyentuh apa pun
        backup_path = f"{ENV_PATH}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(ENV_PATH, backup_path)
        print(f"✅ Backup dibuat: {backup_path}")

    missing_keys = {k: v for k, v in NEW_KEYS.items() if k not in existing or not existing[k]}
    already_set = {k: existing[k] for k in NEW_KEYS if k in existing and existing[k]}

    if already_set:
        print("\n📌 Key berikut SUDAH ADA nilainya di .env lama — TIDAK diubah:")
        for k in already_set:
            print(f"   {k} (nilai lama dipertahankan)")

    if not missing_keys:
        print("\n✅ Semua key baru sudah lengkap di .env — tidak ada yang perlu ditambahkan.")
        return

    print("\n➕ Key berikut akan DITAMBAHKAN (belum ada / masih kosong):")
    for k in missing_keys:
        print(f"   {k}")

    # Kalau key sudah ADA sebagai baris kosong (mis. "GOLDAPI_KEY="), isi
    # nilainya di tempat itu juga, jangan duplikat baris baru.
    new_lines = []
    handled = set()
    for line in lines:
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith("#"):
            key = stripped.split("=", 1)[0].strip()
            if key in missing_keys:
                new_lines.append(f"{key}={missing_keys[key]}\n")
                handled.add(key)
                continue
        new_lines.append(line)

    # Key yang benar-benar tidak ada barisnya sama sekali -> tambahkan blok baru di akhir
    remaining = {k: v for k, v in missing_keys.items() if k not in handled}
    if remaining:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines.append("\n")

        # Build a fresh block containing ONLY the remaining keys, each with
        # its own comment line, rather than filtering the full template
        # (filtering left orphaned comment lines for keys handled in-place
        # above — confirmed by test: "# GoldAPI.io..." comment appeared
        # with no GOLDAPI_KEY line under it when that key was already
        # filled in-place higher up in the file).
        comments = {
            "GOLDAPI_KEY": "# GoldAPI.io — spot gold price (XAU/USD)",
            "TWELVEDATA_KEY": "# Twelve Data — SPX (S&P 500) daily time series (primary source)",
            "ALPHAVANTAGE_KEY": "# Alpha Vantage — SPX/SPY fallback if Twelve Data fails or hits rate limit",
        }
        block_lines = [
            "\n# ── Cross-asset data for M69 GNN Systemic Risk (market_data_fetcher.py) ──\n",
            "# ETH needs no key (Binance public REST API).\n",
        ]
        for k, v in remaining.items():
            block_lines.append("\n")
            block_lines.append(comments.get(k, f"# {k}") + "\n")
            block_lines.append(f"{k}={v}\n")
        new_lines.extend(block_lines)

    with open(ENV_PATH, "w") as f:
        f.writelines(new_lines)

    print(f"\n✅ .env berhasil diperbarui: {ENV_PATH}")
    print("   Jalankan lagi kapan saja — aman diulang (idempotent).")


if __name__ == "__main__":
    main()
