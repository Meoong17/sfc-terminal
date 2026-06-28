#!/usr/bin/env python3
"""Post-injection tab redesign. Called after inject_data.py in pipeline."""
import re, sys

path = '/home/ubuntu/sfc/index.html'
with open(path, 'rb') as f:
    raw = f.read()
c = raw.decode('utf-8')

# ─── ALWAYS-RUN PATCHES (idempotent, skip if already applied) ───

# 1. CSS — match on bytes to avoid unicode issues
raw_new_css = b'}\n\n/* TAB PANELS */\n.tab-panel { display: none; animation: tabFadeIn 0.25s ease; }\n.tab-panel.active { display: block; }\n@keyframes tabFadeIn {\n  from { opacity: 0; transform: translateY(6px); }\n  to   { opacity: 1; transform: translateY(0); }\n}\n\n/* \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80 Donation widget \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80 */\n.donate-card {\n  background: var(--bg-raise);'

anchor = b'}\n\n/* \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80 Donation widget \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80 */\n.donate-card {\n  background: var(--bg-raise);'
if b'.tab-panel { display: none;' in raw:
    print('tab: CSS skip (already has tab panels CSS)')
else:
    if anchor not in raw:
        print('tab: CSS anchor not found')
        sys.exit(1)
    raw = raw.replace(anchor, raw_new_css, 1)
    print('tab: CSS OK')
c = raw.decode('utf-8')

# 2. Nav tabs — skip if already has 5
nav_old = "onclick=\"switchTab('overview')\">Overview</div>\n      <div class=\"nav-tab\" onclick=\"switchTab('ensemble')\">Ensemble</div>\n      <div class=\"nav-tab\" onclick=\"switchTab('risk')\">Risk</div>\n      <div class=\"nav-tab\" onclick=\"switchTab('backtest')\">Backtest</div>"
nav_new = "onclick=\"switchTab('overview')\">Overview</div>\n      <div class=\"nav-tab\" onclick=\"switchTab('risk')\">Risk</div>\n      <div class=\"nav-tab\" onclick=\"switchTab('ensemble')\">Ensemble</div>\n      <div class=\"nav-tab\" onclick=\"switchTab('advanced')\">Advanced</div>\n      <div class=\"nav-tab\" onclick=\"switchTab('trading')\">Trading</div>"
if nav_old in c:
    c = c.replace(nav_old, nav_new, 1)
    print('tab: Nav OK')
else:
    print('tab: nav skip (already 5 tabs)')

# 3. switchTab — skip if already toggle-based
switch_old = "function switchTab(tab) {\n  activeTab = tab;\n  const el = document.getElementById(tab + 'Section');\n  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });\n  document.querySelectorAll('.nav-tab').forEach(t => {\n    t.classList.toggle('active', t.textContent.trim().toLowerCase() === tab);\n  });\n}"
switch_new = "function switchTab(tab) {\n  activeTab = tab;\n  document.querySelectorAll('.tab-panel').forEach(p => {\n    p.classList.toggle('active', p.dataset.tab === tab);\n  });\n  document.querySelectorAll('.nav-tab').forEach(t => {\n    t.classList.toggle('active', t.textContent.trim().toLowerCase() === tab);\n  });\n  if (currentData) {\n    setTimeout(function() {\n      if (tab === 'overview' || tab === 'ensemble') {\n        if (typeof buildFactorChart === 'function' && document.getElementById('factorChart')) buildFactorChart(currentData);\n        if (typeof buildMethodChart === 'function' && document.getElementById('methodChart')) buildMethodChart(currentData);\n      }\n      if (tab === 'overview' && chartVisible && typeof updatePriceChart === 'function') {\n        updatePriceChart();\n      }\n    }, 100);\n  }\n}"
if switch_old in c:
    c = c.replace(switch_old, switch_new, 1)
    print('tab: switchTab OK')
else:
    print('tab: switchTab skip (already updated)')

# 4. Mobile nav — always try to apply improved CSS (idempotent)
old_mob = '@media (max-width: 768px) {\n  .nav-center { display: none; }'
new_mob = '@media (max-width: 768px) {\n  nav.nav { flex-wrap: wrap; gap: 4px; padding: 4px 8px; }\n  .nav-center {\n    display: flex;\n    overflow-x: auto;\n    -webkit-overflow-scrolling: touch;\n    scrollbar-width: none;\n    width: 100%;\n    padding: 4px 0;\n    gap: 6px;\n    order: 3;\n    justify-content: flex-start;\n  }\n  .nav-center::-webkit-scrollbar { display: none; }\n  .nav-tab {\n    white-space: nowrap;\n    padding: 8px 16px !important;\n    font-size: 13px !important;\n    flex: 0 0 auto;\n    border-radius: 8px;\n    border: 1px solid rgba(255,255,255,0.08);\n  }\n  .nav-tab.active { border-color: var(--purple-dim); }\n  .tab-panel { padding: 8px 0; }'
if old_mob in c:
    c = c.replace(old_mob, new_mob, 1)
    print('tab: Mobile nav OK')
elif 'padding: 8px 16px' in c and 'font-size: 13px' in c:
    print('tab: Mobile nav skip (already has improved CSS)')
else:
    # Try to upgrade old mobile CSS — find existing mobile media query
    mob_start = c.find('@media (max-width: 768px)')
    if mob_start > 0:
        mob_end = c.find('@media', mob_start + 10)
        if mob_end < 0:
            mob_end = c.find('/* DATA */', mob_start)
        old_mob_block = c[mob_start:mob_end]
        # Replace padding/font-size within nav-tab
        old_mob_block = old_mob_block.replace('padding: 4px 10px', 'padding: 8px 16px')
        old_mob_block = old_mob_block.replace('font-size: 10px', 'font-size: 13px')
        old_mob_block = old_mob_block.replace('gap: 1px', 'gap: 6px')
        if 'border-radius: 8px' not in old_mob_block:
            old_mob_block = old_mob_block.replace('.nav-tab { white-space: nowrap;', '.nav-tab {\n    white-space: nowrap;\n    padding: 8px 16px !important;\n    font-size: 13px !important;\n    flex: 0 0 auto;\n    border-radius: 8px;\n    border: 1px solid rgba(255,255,255,0.08);\n  }')
        if 'tab-panel.active' not in old_mob_block:
            old_mob_block = old_mob_block.replace('}\n  .main', '}\n  .nav-tab.active { border-color: var(--purple-dim); }\n  .tab-panel { padding: 8px 0; }\n  .main')
        if 'grid-template-columns: repeat(2, 1fr)' not in old_mob_block:
            old_mob_block = old_mob_block.replace('.grid-4 { grid-template-columns: 1fr; }', '.grid-4 { grid-template-columns: repeat(2, 1fr); gap: 6px; }')
        c = c[:mob_start] + old_mob_block + c[mob_end:]
        print('tab: Mobile nav upgraded')
    else:
        print('tab: Mobile nav skip (no media query found)')

# ─── ONLY-ONCE: .main rewrite (guard: skip if already 5 tabs) ───

tpl_start = c.find('const html = `')
if tpl_start > 0:
    tpl_end = c.find('`;\n\n  document', tpl_start)
    tpl = c[tpl_start:tpl_end] if tpl_end > 0 else c[tpl_start:tpl_start+500]
    tpl_data_tabs = len(re.findall(r'data-tab="[^"]+"', tpl))
else:
    tpl_data_tabs = 0

# 4b. Inter font link (always run, independent of premium CSS)
if 'family=Inter' not in c:
    font_link = '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">\n'
    old_link = '<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk'
    if old_link in c:
        c = c.replace(old_link, font_link, 1)
        print('tab: Inter font link added')
    else:
        print('tab: Inter font link skip (anchor not found)')
else:
    print('tab: Inter font link skip (already present)')

# 4c. Premium CSS override (run every time, idempotent via marker check)
PREMIUM_MARKER = 'PREMIUM OVERRIDES'
if PREMIUM_MARKER not in c:
    premium_css = '''\n/* ══════════════════════════════════════════
   PREMIUM OVERRIDES — Linear-inspired
══════════════════════════════════════════ */

/* Font: use Inter for nav + UI, keep Space Grotesk for data */
body { font-family: 'Inter', 'Space Grotesk', system-ui, sans-serif; }

/* Nav: cleaner, premium */
.nav { padding: 0 20px; height: 52px; background: rgba(7,8,13,0.92); border-bottom: 1px solid rgba(255,255,255,0.05); }
.nav-brand { gap: 8px; }
.nav-logo { width: 26px; height: 26px; font-size: 9px; }

/* Nav tabs: Linear-style ghost buttons */
.nav-center { display: flex; align-items: center; gap: 2px; margin: 0 auto; }
.nav-tab {
  padding: 6px 14px; font-size: 13px; font-weight: 500;
  font-family: 'Inter', system-ui, sans-serif;
  color: var(--text-2); background: transparent;
  border: 1px solid transparent; border-radius: 6px;
  cursor: pointer; transition: all 0.15s ease; white-space: nowrap;
}
.nav-tab:hover { color: var(--text-1); background: rgba(255,255,255,0.04); border-color: rgba(255,255,255,0.08); }
.nav-tab.active { color: #fff; background: rgba(120,100,255,0.1); border-color: rgba(120,100,255,0.25); }

/* Cards: subtle translucent */
.kpi-card, .metric-card, .chart-card, .card {
  background: rgba(255,255,255,0.02) !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
  border-radius: 8px !important;
  transition: border-color 0.15s ease;
}
.kpi-card:hover, .metric-card:hover, .chart-card:hover { border-color: rgba(255,255,255,0.12) !important; }

/* KPI row: cleaner spacing */
.kpi-row { gap: 8px; }
.kpi-card { padding: 12px 16px !important; }
.kpi-num { font-family: 'JetBrains Mono', monospace; font-weight: 600; }
.kpi-label { font-weight: 400; color: var(--text-2); }

/* Status band: clean */
.status-band { background: rgba(255,255,255,0.02); border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 12px; padding: 6px 20px; letter-spacing: 0.01em; }

/* Tab panels: smooth transitions */
.tab-panel { animation: tabFadeIn 0.2s ease; }

/* Nav right: compact */
.nav-meta { gap: 12px; }
.nav-meta-item { gap: 4px; }
.nav-meta-label { font-size: 10px; color: var(--text-3); }
.nav-meta-val { font-size: 12px; font-weight: 500; }

/* Scrollbar: subtle dark */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }

/* Mobile: keep tabs scrollable */
@media (max-width: 768px) {
  .nav { padding: 0 12px; height: 48px; }
  .nav-center { gap: 1px; padding: 2px 0; }
  .nav-tab { padding: 6px 10px; font-size: 11px; flex: 0 0 auto; }
  .nav-right { display: none; }
  .kpi-card { padding: 8px 10px !important; }
  .status-band { padding: 4px 12px; font-size: 10px; }
}
@media (max-width: 480px) {
  .nav-tab { padding: 5px 8px; font-size: 10px; }
  .nav-brand .nav-logo { display: none; }
  .kpi-row { grid-template-columns: repeat(2, 1fr); gap: 4px; }
}

/* ══════════════════════════════════════════
   END PREMIUM OVERRIDES
══════════════════════════════════════════ */\n\n'''
    # Insert before INTER Font link or at end of head
    head_close = c.find('</head>')
    if head_close > 0:
        c = c[:head_close] + premium_css + c[head_close:]
        print('tab: Premium CSS injected')
    else:
        print('tab: Premium CSS skip (no </head>)')
else:
    print('tab: Premium CSS skip (already applied)')

if tpl_data_tabs >= 5:
    print('tab: .main already redesigned, skipping rewrite')
else:
    # 5. Rewrite .main into 5 tab panels
    im = c.find('\n  <!-- MAIN -->\n  <div class="main">')
    assert im > 0, 'main start not found'
    ie = c.rfind('</div><!-- /main -->', im, c.find("document.getElementById('app').innerHTML", im))
    assert ie > 0, 'main end not found'
    om = c[im:ie + len('</div><!-- /main -->')]

    om = om.replace('id="riskSection" class="grid-4 mb-16"', 'class="grid-4 mb-16"')
    om = om.replace('<!-- \u2550\u2550\u2550 TAB 1: OVERVIEW \u2550\u2550\u2550 -->\n    <div class="tab-panel active" data-tab="overview">\n\n', '', 1)
    om = om.replace('<!-- \u2550\u2550\u2550 TAB 1: OVERVIEW \u2550\u2550\u2550 -->\n    <div class="tab-panel active" data-tab="overview">\n', '', 1)

    def find(n, s=0):
        p = om.find(n, s); assert p >= 0, f'MISS: {n[:60]}'; return p

    a_kpi = find('<!-- ROW 1: SFC Stress Index + Signal -->')
    a_sfc_e = find('      </div>\n\n      <!-- ENSEMBLE -->')
    a_ens = find('      <!-- ENSEMBLE -->')
    a_ens_c = find('    </div>\n\n    <!-- PRICE CHART -->')
    a_chart = find('<!-- PRICE CHART -->')
    a_sig = find('      <!-- SIGNAL -->')
    a_q10 = find('<!-- Q10 ON-CHAIN PANEL')
    a_ms = find('<!-- MARKET STRUCTURE')
    a_liq = find('<!-- \u2550\u2550\u2550 NEW: LIQUIDITY INTELLIGENCE')
    a_mliq = find('<!-- MACRO LIQUIDITY')
    a_st = find('<!-- STABLECOIN LIQUIDITY')
    a_k = find('<!-- KELLY + MACRO row -->')
    a_bt = find('<!-- BACKTEST -->')
    a_x = find('<!-- XAI -->')
    a_p = find('<!-- PAPER TRADING -->')
    a_mg = find('    <div class="grid-4 mb-16">', a_p)
    a_h = find('<!-- HELP -->')
    sig_end = om.rfind('    </div>\n\n    <!-- Q10 ON-CHAIN', a_sig) + len('    </div>')
    macro_end = om.rfind('    </div>\n    </div>\n\n    <!-- HELP -->') + len('    </div>')

    P = [om[0:a_kpi]]

    def tab(name, content, active=False):
        act = ' active' if name == 'overview' else ''
        return f'    <!-- \u2550\u2550\u2550 TAB: {name.upper()} \u2550\u2550\u2550 -->\n    <div class="tab-panel{act}" data-tab="{name}">\n' + content + '\n    </div><!-- /{name} -->\n'

    P.append(tab('overview', om[a_kpi:a_sfc_e] + '      </div>\n    </div>\n' + om[a_chart:a_sig] + om[a_sig:sig_end], active=True))
    P.append(tab('risk', '    <div id="riskSection" class="grid-3 mb-16">\n' + om[find('      <!-- MOMENTUM -->'):a_sig] + '    </div>\n' + om[a_q10:a_ms] + om[a_ms:a_liq]))
    P.append(tab('ensemble', om[a_ens:a_ens_c+len('    </div>')] + om[a_k:a_bt] + om[a_bt:a_x] + om[a_x:a_p]))
    P.append(tab('advanced', om[a_liq:a_mliq] + om[a_mliq:a_st] + om[a_st:a_k] + om[a_mg:macro_end]))
    P.append(tab('trading', om[a_p:a_mg]))
    P.append(om[a_h:])

    c = c[:im] + ''.join(P) + c[ie + len('</div><!-- /main -->'):]
    print('tab: .main rewrite OK')

# ─── VERIFY + WRITE ───
checks = {'.tab-panel CSS': '.tab-panel { display: none;' in c,
          'nav trading': "switchTab('trading')" in c,
          'switchTab toggle': "querySelectorAll('.tab-panel')" in c,
          'factorChart': 'id="factorChart"' in c, 'methodChart': 'id="methodChart"' in c,
          'paperSection': 'id="paperSection"' in c, 'helpPanel': 'id="helpPanel"' in c}
missing = [k for k, v in checks.items() if not v]
if missing:
    print(f'tab: FAILED: {missing}')
    sys.exit(1)

with open(path, 'w') as f:
    f.write(c)
tabs = re.findall(r'data-tab="([^"]+)"', c)
print(f'tab: DONE! {len(tabs)} panels: {tabs} ({len(c)} bytes)')
