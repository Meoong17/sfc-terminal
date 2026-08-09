# SFC Terminal

Bitcoin macro stress dashboard, built around global liquidity as the
primary predictive factor for BTC price behavior — Fed/ECB/BOJ balance
sheets, stablecoin flows, repo market stress, ETF flows, and on-chain
signals feed a five-factor ensemble (`Lt`/`St`/`Rt`/`Ft`/`Sc`) that
produces a single stress score, `sfc_pct`.

## Quick links

- **[Architecture map](docs/ARCHITECTURE.md)** — what every module does and how they connect.
- **[Project status](docs/PROJECT_STATUS.md)** — audit history, known limitations, walk-forward validation verdicts.

## Structure

```
collect.py              # orchestrator — run every pipeline cycle
sse_server.py           # SSE push server for live dashboard updates
inject_data.py          # inject data.json into index.html (offline-first serving)
binance_ws.py           # Binance WebSocket price daemon
sfc-pipeline.sh         # cron wrapper: retry, commit, push, index.html restore
*.sh                    # training + watchdog cron entry points

data_sources/           # live data providers (on-chain, stablecoin, ETF, news, ...)
analysis/               # walk-forward validation, calibration, causal/regime analysis
models/                 # ML models (mamba, qlstm, hmm, ensemble, online learning)
ml/                     # feature engineering & dynamic feature weighting
risk/                   # systemic-risk GNN, XAI explainer
trading/                # paper trading engine, DRL agent
optimization/           # genetic algorithm optimizer
data_augmentation/      # time-series augmentation (timeGAN)
scripts/                # maintenance & audit utilities
docs/                   # architecture, validation, status reports
worker/                 # Cloudflare Worker (dashboard proxy + auth)
static/  fonts/         # static front-end assets
data/                   # canonical Binance Vision historical data
```

`index.html` is the compiled single-file dashboard; `app.js`, `sw.js`,
and `static/` are the front-end source and service worker.

## Running the pipeline

```bash
cp .env.example .env   # fill in API keys
pip install -r requirements.txt
python3 collect.py > data.json
```

For scheduled operation, `sfc-pipeline.sh` is the entry point cron
should call — **note**: this repo's copy is the source of truth for
review, but the version actually invoked by cron on the VPS lives at
`~/.hermes/scripts/sfc-pipeline.sh` (see the comment at the top of
`sfc-pipeline.sh`). Keep both in sync manually when editing.

## Deployment

- **VPS**: runs `collect.py` on a schedule via `sfc-pipeline.sh`, plus background daemons (`binance_ws.py`) kept alive by `ws-watchdog.sh`.
- **Cloudflare Worker** (`worker/`): proxies the dashboard, handles multi-user session auth. Secrets (`BACKUP_URL`, `SESSION_SECRET`) are set via `wrangler secret put`, never committed — see `.env.example` for the full list of required environment variables.

## Status

42+ audit findings addressed across five categories: circular model
labeling (5 modules), unit-scaling bugs, safety layers that were computed
but not wired to the value they should protect, security (session
forgery, exposed infrastructure), and formula/calibration errors. Full
history in [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).
