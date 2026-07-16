#!/usr/bin/env python3
"""Check all dashboard cards have valid data."""
import json, sys

d = json.load(open('/home/ubuntu/sfc/data.json'))

cards = {
    'BTC/USD':          ['btc', 'btc_24h', 'btc_mcap'],
    'SFC Stress':       ['sfc_base', 'sfc_effective', 'zone'],
    'DVOL':             ['dvol'],
    'Fear & Greed':     ['fng', 'fng_cls'],
    'Confidence':       ['composite_confidence'],
    'BTC Dom':          ['dom', 'm2_yoy'],
    'Factor Lt':        ['factors'],
    'RSI':              ['rsi_14', 'rsi_regime'],
    'SOPR':             ['sopr_proxy', 'sopr_signal', 'sopr_score'],
    'Liquidation':      ['liq_pressure', 'liq_density', 'cascade_risk', 'liq_total_24h'],
    'News':             ['news_stress', 'news_stats'],
    'Adv Regime':       ['adv_regime', 'adv_crisis_prob', 'adv_uncertainty'],
    'Floor / ATH':      ['floor_buffer', 'floor_total', 'ath'],
    'P/C OI/Vol':       ['pc_oi', 'pc_vol'],
    'Signal & Action':  ['signal_type', 'signal', 'timing_precision', 'alert_window_hours'],
    'Conf Components':  ['confidence_components'],
    'M1-M6 Ensemble':   ['m1_klr', 'm2_logit', 'm3_bayes', 'm4_ewc', 'm5_qreg', 'm6_regime_score', 'method_agreement'],
    'M7-M19 Extended':  ['m7_fisher', 'm8_yield', 'm9_liquidity', 'm10_garch', 'm11_var', 'm12_jump', 'm13_funding', 'm14_skew', 'm15_concentration', 'm16_regime_ml', 'm17_granger', 'm18_entropy', 'm19_mutual_info'],
    'M20-M31 Inst':     ['m20_obi', 'm21_trade_flow', 'm22_spread', 'm23_liquidity', 'm24_cape', 'm25_minsky', 'm26_kahneman', 'm27_taleb', 'm28_summers', 'm29_debt', 'm30_rajan', 'm31_altman'],
    'QLSTM M32':        ['m32_qlstm', 'm32_active', 'm32_garch_residual', 'm32_proadapt_weight'],
    'XAI Features':     ['xai_top_features'],
    'ML Ensemble':      ['ml_ensemble_score', 'ml_accuracy', 'ml_total_labeled'],
    'Kelly':            ['kelly_fraction', 'kelly_p_win', 'kelly_b_payoff'],
    'Backtest':         ['bt_sharpe', 'bt_win_rate', 'bt_max_dd', 'bt_stability', 'bt_periods'],
    'Paper Trading':    ['paper_trades'],
    'Regime/State':     ['regime', 'regime_prob', 'transition_risk', 'state', 'dv_sfc', 'phi'],
}

issues = []
for card, fields in cards.items():
    for f in fields:
        if f not in d:
            issues.append(f'MISSING [{card}] {f}')
        elif d[f] is None:
            issues.append(f'NULL [{card}] {f}')

# Check sub-fields
if 'factors' in d and d['factors']:
    for sub in ['Lt','St','Rt','Ft','Sc']:
        if sub not in d['factors']:
            issues.append(f'MISSING factor.{sub}')
if 'confidence_components' in d and d['confidence_components']:
    for sub in ['method_agree','rsi','sopr','dvol','cascade_penalty','fear_penalty']:
        if sub not in d['confidence_components']:
            issues.append(f'MISSING cc.{sub}')

if issues:
    print(f'❌ {len(issues)} ISSUES:')
    for i in issues:
        print(f'   {i}')
else:
    print('✅ ALL 28 CARD GROUPS — every field present and non-null')

print()
print('=== KEY DASHBOARD VALUES ===')
print(f'  BTC:         ${d.get("btc","?"):.1f}  (24h: {d.get("btc_24h","?")}%)')
print(f'  SFC eff:     {d.get("sfc_effective","?"):.1f}%  (base: {d.get("sfc_base","?"):.1f}%)')
print(f'  Zone:        {d.get("zone","?")}  |  State: {d.get("state","?")}')
print(f'  Regime:      {d.get("regime","?")} (prob {d.get("regime_prob","?"):.0%})  |  Adv: {d.get("adv_regime","?")}')
print(f'  Fear/Greed:  {d.get("fng","?")} ({d.get("fng_cls","?")})')
print(f'  RSI:         {d.get("rsi_14","?")} ({d.get("rsi_regime","?")})')
print(f'  SOPR:        {d.get("sopr_proxy","?")} ({d.get("sopr_signal","?")})  score={d.get("sopr_score","?"):.0%}')
print(f'  DVOL:        {d.get("dvol","?")}%')
print(f'  Confidence:  {d.get("composite_confidence","?"):.0%}  |  Method Agree: {d.get("method_agreement","?"):.0%}')
print(f'  Signal:      {d.get("signal","?")} ({d.get("signal_type","?")})')
print(f'  Kelly alloc: {d.get("kelly_fraction","?"):.0%} (half: {d.get("kelly_half","?"):.0%})')
print(f'  Backtest:    Sharpe={d.get("bt_sharpe","?")}  WR={d.get("bt_win_rate","?"):.0%}  DD={d.get("bt_max_dd","?"):.1%}  Stab={d.get("bt_stability","?"):.0%}')
print(f'  QLSTM:       {d.get("m32_qlstm","?")}  active={d.get("m32_active","?")}')
print(f'  Paper:       {len(d.get("paper_trades",[]))} trades')
print(f'  M1-M6 avg:   {d.get("m1_klr","?")}% | {d.get("m2_logit","?")}% | {d.get("m3_bayes","?")}% | {d.get("m4_ewc","?")}% | {d.get("m5_qreg","?")}% | {d.get("m6_regime_score","?")}%')
print(f'  Consensus:   {d.get("method_agreement","?"):.0%} ({d.get("causal_methods_active","?")} active methods)')

# Check XAI
xai = d.get('xai_top_features', [])
if xai:
    print(f'  XAI top:     {xai_summary}')
else:
    print('  XAI:         EMPTY')

# Check sync — BTC from WS vs data.json
import os, time
ws = json.load(open('/home/ubuntu/sfc/btc_ws.json'))
print()
print('=== REAL-TIME SYNC CHECK (WebSocket vs data.json) ===')
ws_ts = ws.get('ts','')
ws_btc = ws.get('btc',0)
dj_btc = d.get('btc',0)
diff_pct = abs(dj_btc - ws_btc) / ws_btc * 100 if ws_btc else 0
print(f'  WS BTC:      ${ws_btc:.1f}  at {ws_ts}')
print(f'  data.json:   ${dj_btc:.1f}')
print(f'  Diff:        ${abs(dj_btc-ws_btc):.1f} ({diff_pct:.2f}%)')
gap_status = "within 1% — sync OK" if diff_pct < 1 else "MORE than 1% — data may be stale"
print(f'  Gap:         {gap_status}')

# Check pipeline log for recent errors
print()
print('=== PIPELINE HEALTH (last 5 runs) ===')
with open('/home/ubuntu/sfc/sfc-pipeline.log') as f:
    lines = f.readlines()
pipelines = [l for l in lines if 'Pipeline done:' in l]
for p in pipelines[-5:]:
    print(f'  {p.strip()}')
