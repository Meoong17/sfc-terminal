#!/usr/bin/env python3
"""Post-injection tab redesign. Called after inject_data.py in pipeline."""
import re, sys

path = '/home/ubuntu/sfc/index.html'
with open(path, 'rb') as f:
    raw = f.read()
c = raw.decode('utf-8')

# Skip if already redesigned — check INSIDE template only, not injected JSON data
tpl_start = c.find('const html = `')
if tpl_start > 0:
    tpl = c[tpl_start:c.find('`;', tpl_start)]
    tpl_data_tabs = len(re.findall(r'data-tab="[^"]+"', tpl))
else:
    tpl_data_tabs = 0
if tpl_data_tabs >= 5:
    print('tab: already redesigned, skipping')
    sys.exit(0)

# 1. CSS — match on bytes to avoid unicode issues
raw_new_css = b'}\n\n/* TAB PANELS */\n.tab-panel { display: none; animation: tabFadeIn 0.25s ease; }\n.tab-panel.active { display: block; }\n@keyframes tabFadeIn {\n  from { opacity: 0; transform: translateY(6px); }\n  to   { opacity: 1; transform: translateY(0); }\n}\n\n/* \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80 Donation widget \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80 */\n.donate-card {\n  background: var(--bg-raise);'

# Find the CSS anchor in bytes
anchor = b'}\n\n/* \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80 Donation widget \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80 */\n.donate-card {\n  background: var(--bg-raise);'
if anchor not in raw:
    print('tab: CSS anchor not found')
    sys.exit(1)
raw = raw.replace(anchor, raw_new_css, 1)
print('tab: CSS OK')
c = raw.decode('utf-8')

# 2. Nav tabs
old = "onclick=\"switchTab('overview')\">Overview</div>\n      <div class=\"nav-tab\" onclick=\"switchTab('ensemble')\">Ensemble</div>\n      <div class=\"nav-tab\" onclick=\"switchTab('risk')\">Risk</div>\n      <div class=\"nav-tab\" onclick=\"switchTab('backtest')\">Backtest</div>"
new = "onclick=\"switchTab('overview')\">Overview</div>\n      <div class=\"nav-tab\" onclick=\"switchTab('risk')\">Risk</div>\n      <div class=\"nav-tab\" onclick=\"switchTab('ensemble')\">Ensemble</div>\n      <div class=\"nav-tab\" onclick=\"switchTab('advanced')\">Advanced</div>\n      <div class=\"nav-tab\" onclick=\"switchTab('trading')\">Trading</div>"
if old not in c:
    print('tab: nav anchor not found')
    sys.exit(1)
c = c.replace(old, new, 1)
print('tab: Nav OK')

# 3. switchTab
old = "function switchTab(tab) {\n  activeTab = tab;\n  const el = document.getElementById(tab + 'Section');\n  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });\n  document.querySelectorAll('.nav-tab').forEach(t => {\n    t.classList.toggle('active', t.textContent.trim().toLowerCase() === tab);\n  });\n}"
new = "function switchTab(tab) {\n  activeTab = tab;\n  document.querySelectorAll('.tab-panel').forEach(p => {\n    p.classList.toggle('active', p.dataset.tab === tab);\n  });\n  document.querySelectorAll('.nav-tab').forEach(t => {\n    t.classList.toggle('active', t.textContent.trim().toLowerCase() === tab);\n  });\n  if (currentData) {\n    setTimeout(function() {\n      if (tab === 'overview' || tab === 'ensemble') {\n        if (typeof buildFactorChart === 'function' && document.getElementById('factorChart')) buildFactorChart(currentData);\n        if (typeof buildMethodChart === 'function' && document.getElementById('methodChart')) buildMethodChart(currentData);\n      }\n      if (tab === 'overview' && chartVisible && typeof updatePriceChart === 'function') {\n        updatePriceChart();\n      }\n    }, 100);\n  }\n}"
if old not in c:
    print('tab: switchTab anchor not found')
    sys.exit(1)
c = c.replace(old, new, 1)
print('tab: switchTab OK')

# 4. Rewrite .main into 5 tab panels
im = c.find('\n  <!-- MAIN -->\n  <div class="main">')
assert im > 0, 'main start not found'
ie = c.rfind('</div><!-- /main -->', im, c.find("document.getElementById('app').innerHTML", im))
assert ie > 0, 'main end not found'
om = c[im:ie + len('</div><!-- /main -->')]

# Sanitize: remove duplicate id=riskSection from original grid-4 (now in Overview tab)
om = om.replace('id="riskSection" class="grid-4 mb-16"', 'class="grid-4 mb-16"')

# Remove stale overview wrapper (from previous partial runs)
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
sig_end = om.rfind('    </div>\n\n    <!-- Q10 ON-CHAIN', a_sig, a_q10) + len('    </div>')
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

# Verify
checks = {'.tab-panel CSS': '.tab-panel { display: none;', '5 tabs': len(re.findall(r'data-tab="[^"]+"', c)) == 5,
          'overview active': 'class="tab-panel active"', 'nav trading': "switchTab('trading')" in c,
          'switchTab toggle': "querySelectorAll('.tab-panel')" in c,
          'factorChart': 'id="factorChart"', 'methodChart': 'id="methodChart"',
          'paperSection': 'id="paperSection"', 'helpPanel': 'id="helpPanel"'}
missing = [k for k, v in checks.items() if not v]
if missing:
    print(f'tab: FAILED: {missing}')
    sys.exit(1)

# 5. Mobile nav fix: show scrollable tabs instead of hide
old_mob = '@media (max-width: 768px) {\n  .nav-center { display: none; }'
new_mob = '@media (max-width: 768px) {\n  nav.nav { flex-wrap: wrap; gap: 4px; }\n  .nav-center {\n    display: flex;\n    overflow-x: auto;\n    -webkit-overflow-scrolling: touch;\n    scrollbar-width: none;\n    width: 100%;\n    padding: 2px 0;\n    gap: 1px;\n    order: 3;\n  }\n  .nav-center::-webkit-scrollbar { display: none; }\n  .nav-tab { white-space: nowrap; padding: 4px 10px; font-size: 10px; }'
if old_mob in c:
    c = c.replace(old_mob, new_mob, 1)
    print('tab: Mobile nav OK')
else:
    print('tab: Mobile nav anchor not found')
    sys.exit(1)

with open(path, 'w') as f:
    f.write(c)
tabs = re.findall(r'data-tab="([^"]+)"', c)
print(f'tab: DONE! {len(tabs)} panels: {tabs} ({len(c)} bytes)')
