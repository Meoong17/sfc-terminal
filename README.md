# SFC Terminal

Bitcoin macro stress dashboard, built around global liquidity as the
primary predictive factor for BTC price behavior — Fed/ECB/BOJ balance
sheets, stablecoin flows, repo market stress, ETF flows, and on-chain
signals feed a five-factor ensemble (`Lt`/`St`/`Rt`/`Ft`/`Sc`) that
produces a single stress score, `sfc_pct`.

## Quick links

- **[Architecture map](docs/ARCHITECTURE.md)** — what every module does and how they connect.
- **[Project status](docs/PROJECT_STATUS.md)** — audit history, known limitations, why the repo isn't fully reorganized into subfolders.

## Structure at a glance

```
collect.py              # orchestrator — run every pipeline cycle
sfc-pipeline.sh          # cron wrapper: retry, commit, push, index.html restore
ml/  analysis/  data_sources/   # modules confirmed safe to relocate (see docs/PROJECT_STATUS.md)
models/  risk/  trading/  optimization/  data_augmentation/   # pre-existing package folders
worker/                  # Cloudflare Worker (dashboard proxy + auth)
```

Most modules still live at repo root rather than in topic folders — the
majority resolve their own cache/model/data file paths relative to their
own location (`os.path.dirname(os.path.abspath(__file__))`), so moving
them requires updating that path logic in each file individually, not
just the import statement in `collect.py`. See
[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) for the full
verification and the 10 modules that were confirmed safe and relocated.

## Running the pipeline

```bash
cp .env.example .env   # fill in API keys
pip install -r requirements.txt   # if present, or see individual module imports
python3 collect.py > data.json
```

For scheduled operation, `sfc-pipeline.sh` is the entry point cron
should call — **note**: this repo's copy is the source of truth for
review, but the version actually invoked by cron on the VPS lives at
`~/.hermes/scripts/sfc-pipeline.sh` (see the comment at the top of
`sfc-pipeline.sh`). Keep both in sync manually when editing.

## Deployment

- **VPS**: runs `collect.py` on a schedule via `sfc-pipeline.sh`, plus background daemons (`binance_ws.py`, `qlstm_daemon.py`) kept alive by watchdog scripts (`ws-watchdog.sh`, `qlstm_watchdog.py`).
- **Cloudflare Worker** (`worker/`): proxies the dashboard, handles multi-user session auth. Secrets (`BACKUP_URL`, `SESSION_SECRET`) are set via `wrangler secret put`, never committed — see `.env.example` for the full list of required environment variables.

## Status

42+ audit findings addressed across five categories: circular model
labeling (5 modules), unit-scaling bugs, safety layers that were computed
but not wired to the value they should protect, security (session
forgery, exposed infrastructure), and formula/calibration errors. Full
history in [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).
