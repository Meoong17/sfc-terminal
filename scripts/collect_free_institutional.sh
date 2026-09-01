#!/usr/bin/env bash
# Daily accumulation of free institutional behaviour signals (OI, options, on-chain).
set -e
cd /home/ubuntu/sfc
.venv/bin/python scripts/collect_free_institutional.py >> logs/collect_free_institutional.log 2>&1
