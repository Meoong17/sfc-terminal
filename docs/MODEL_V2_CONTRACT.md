# SFC Model Contract v2

Status: **partially implemented**. This document describes what the system
actually publishes today, in the layer vocabulary of the V2 blueprint
(`/home/ubuntu/C/SFC_Struktur_Model_Analisis_Meong_dan_Yves_V2.docx`), and
separates what is implemented from what is still a hypothesis. Nothing here
changes a score: the composite formula, its weights and its thresholds are
untouched by this contract.

Rule of the house, taken from the blueprint and enforced in code: **no new
weight enters the composite before it survives an era-split / walk-forward test
against a price-vol baseline.** Diagnostics may be published; they may not vote.

---

## 1. Layers and what each one may publish

| Layer | Name | Implementation | Can vote on the score? |
|---|---|---|---|
| L0 | Data Integrity & State | `analysis/data_integrity.py` | no — gates interpretation |
| L1 | Market / funding conditions | `data/binance_vision_*.json`, `binance_funding_daily.json`, `bitmex_funding_daily.json` | yes, through existing factors |
| L2 | Positioning & structure | `deribit_options_daily`, `okx_oi_daily`, `binance_orderflow_daily`, `analysis/structure_engine.py` | structure indices are display-only |
| L3 | On-chain & flows | `coinmetrics_btc_daily`, `.etf_cache.json` | partly (existing factors) |
| L4 | Model outputs | `collect.py`, `analysis/flqd_engine.py`, HMM, XGB/RF | composite is the only voter |

## 2. Published outputs (data.json)

| Field | Meaning | Provenance label |
|---|---|---|
| `sfc_effective` | composite stress, 0-100 | **Observed** formula, unchanged since 2026-08 |
| `zone` | NORMAL / ELEVATED / HIGH / CRITICAL band | **Observed** (derived from `sfc_effective`) |
| `signal`, `signal_type` | posture (BUY / WATCH / CASH) | **Observed** rule output |
| `composite_confidence` | confidence in the reading, 0-1 | **Observed**, but see §4 — it has no demonstrated resolution |
| `calibrated_confidence` | confidence mapped through a calibration curve | **Currently identical to raw**: the map is disabled (§4) |
| `flqd` | regime distribution R0-R5 + entropy | **Inferred** measurement; documented to lose to the price-vol baseline (Brier 0.260 vs 0.229) |
| `structure` | PFS / FQ / NDS, SFI, quadrant | **Display-only**; verdict in its own payload: "no index adds reliable information over the price/vol baseline" |
| `hmm_regime`, `hmm_crisis_prob` | 4-state HMM posterior | **Inferred**, refit 2026-11 on 20,397 snapshots |
| `data_health` | L0 verdict + per-series status | **Observed** (audited facts) |
| `calibration_walkforward` | walk-forward evidence for the calibration map | **Observed** (measured), verdict may be negative |

Direction, regime, conviction, confidence and permission are deliberately NOT
merged into one number by this contract: today the dashboard publishes them as
separate fields (stress, zone, signal, confidence, regime probabilities). The
blueprint's separate L4 "Direction / Conviction / Risk Permission" objects are
**not implemented** — inventing them before validation is exactly what the
blueprint forbids.

## 3. L0 contract (implemented, enforced)

Every input series is registered in `analysis/data_integrity.py` with:
path, label, layer, max lag allowance, gap check, and optional `fallback`.

Statuses: `ok`, `stale`, `gap` (only gaps inside the last 7 days), `missing`,
`superseded` (dead source replaced by a fresh one — not a degradation).

`health` is `broken` if an L0 series is stale/missing or if the dashboard price
diverges >1% from the live WS feed; `degraded` if any other series is stale/gap/
missing; `ok` otherwise. The dashboard banner renders only when health is not
`ok`. Full report: `analysis/data_integrity_results.json`.

Known data facts recorded by L0 as of 2026-11-14:
- `data/tradingview_btc1_daily.csv` (CME basis) is a manual export, frozen
  2026-09-01 → superseded by `binance_futures_basis_daily.json` (exchange basis
  proxy, not CME).
- `data/bitmex_funding_daily.json` stops 2026-09-16 (endpoint returns 403) →
  superseded after that date by `binance_funding_daily.json`.
- `data/binance_orderflow_daily.json` is monthly (Binance Vision aggTrades), so
  its newest complete day is the end of the previous month. Allowance is 40 days
  on purpose; PFS coverage is 0.75 because of it.

## 4. Calibration contract (implemented, negative result)

Two gates, both must pass:

1. `analysis/calibration_isotonic.py` — single temporal split (60/40), requires
   ECE and Brier to improve on hold-out, monotonicity verified numerically, and
   map spread ≥0.05 (a flat map is climatology, not calibration).
2. `analysis/calibration_walkforward.py` — expanding window, test block 15,
   min train 50, must beat raw confidence **and** the climatology base rate in a
   majority of folds.

Current evidence (138 price-outcome labeled snapshots, labels by design only on
decisive moves — 18% of snapshots, horizon 3 steps ≈ 3-4h):

```
Brier  raw 0.2885 | calibrated 0.2626 | climatology 0.2544
ECE    raw 0.2796 | calibrated 0.2074 | climatology 0.1581
beats raw 3/5 folds | beats climatology 1/5 folds
verdict: REJECT AS CLIMATOLOGY
```

Consequences, enforced in code: `calibration_enabled = false`,
`calibrated_confidence == composite_confidence`, the legacy bin map is gated off
(`legacy_accepted` must be set explicitly), and
`scripts/calib_refit_monthly.sh` runs the walk-forward gate after the isotonic
refit and downgrades any map it rejects.

## 5. Freshness contract (implemented)

`scripts/refresh_engine_inputs.sh` refreshes the automated engine inputs
(funding splice, basis proxy) and is called by `scripts/run_flqd_engine.sh`
before the engines run — daily at 03:15. Without it the structure engine
silently reverts to stale inputs, which is how it froze at 2026-10-06 for six
weeks.

## 6. Not implemented (honest list)

- L4 objects `Direction`, `Conviction`, `RiskPermission` as separate validated
  outputs (blueprint §2). Today only stress/zone/signal/confidence exist.
- Cross-layer structural candidates from the blueprint (ACEE, MDRE, APME, QCE,
  TSRE, IMIM, PSI/EPI/LSI/HBI/BDI/ROCM, reaction residual). None are coded; none
  may be coded as a score before the baseline test.
- Multi-horizon decomposition (1d / 7d / 30d) of the stress reading.
- Regime labels R0-R5 are the FLQD engine's own six classes, not yet mapped onto
  the blueprint's Expansion / Distribution / Contraction / Capitulation /
  Absorption / Recovery vocabulary with validation.

## 7. Change rules

1. A scoring change requires era-split or walk-forward evidence beating the
   price-vol baseline; otherwise the diagnostic is published, not scored.
2. A calibration curve is published only if it beats climatology walk-forward.
3. A data source may be replaced without validation only if the replacement is
   recorded in L0 with its own provenance field (`basis_source`,
   `funding_source`, `funding_splice_from`) and the discontinuity is disclosed.
4. Freshness must be machine-checkable: any input that can go stale gets a
   registry entry with a lag allowance.
