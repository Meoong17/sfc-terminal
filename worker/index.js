// sfc-terminal Cloudflare Worker
// Proxies frontend + SSE to VPS via Cloudflare Tunnel
// Multi-user paper trading via KV storage
//
// Worker tries multiple backends in order:
// 1. Cloudflare Tunnel (set via env TUNNEL_URL or hardcoded below)
// 2. Direct VPS IP (blocked by some Tencent Cloud security groups)

const TUNNEL = 'https://galleries-actively-grew-require.trycloudflare.com';
const BACKUP = 'http://43.134.89.23:8765';

async function fetchAny(urls, path, accept) {
  for (const base of urls) {
    try {
      const resp = await fetch(base + path, {
        headers: { 'Accept': accept || '*/*' },
        signal: AbortSignal.timeout(5000),
      });
      if (resp.ok) return resp;
    } catch (_) {}
  }
  return null;
}

// Default state for new users
function defaultUserState(username) {
  return {
    user_id: username,
    capital: 50000,
    initial_capital: 50000,
    peak_capital: 50000,
    positions: [],
    trades: [],
    equity_history: [],
    daily_snapshots: {},
    config: {
      max_allocation_pct: 25,
      take_profit_pct: 0,
      trailing_stop_pct: 15,
      stop_loss_pct: 0,
      risk_per_trade: 2,
      kelly_enabled: true,
    },
    last_update: null,
    created_at: new Date().toISOString(),
  };
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;
    const urls = [TUNNEL, BACKUP];

    // CORS headers for all responses
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': '*',
    };

    // CORS preflight
    if (method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    // ========== MULTI-USER KV ENDPOINTS ==========

    // GET /user/:username/state — load user state, create default if not exists
    const userStateMatch = path.match(/^\/user\/([^\/]+)\/state$/);
    if (userStateMatch && method === 'GET') {
      const username = decodeURIComponent(userStateMatch[1]);
      const key = `user:${username}:state`;
      let raw = await env.SFC_USER_STATE.get(key);
      if (!raw) {
        // Create default state for new user
        const state = defaultUserState(username);
        raw = JSON.stringify(state);
        await env.SFC_USER_STATE.put(key, raw);
      }
      return new Response(raw, {
        headers: { 'Content-Type': 'application/json', ...corsHeaders },
      });
    }

    // POST /user/:username/state — save full user state
    if (userStateMatch && method === 'POST') {
      const username = decodeURIComponent(userStateMatch[1]);
      const key = `user:${username}:state`;
      let newState;
      try {
        newState = await request.json();
      } catch (e) {
        return new Response('Invalid JSON', { status: 400, headers: corsHeaders });
      }
      newState.user_id = username;
      newState.last_update = new Date().toISOString();
      // Trim large arrays to stay within KV limits (25MB per value, but keep it lean)
      if (newState.trades && newState.trades.length > 500) {
        newState.trades = newState.trades.slice(-500);
      }
      if (newState.equity_history && newState.equity_history.length > 1000) {
        newState.equity_history = newState.equity_history.slice(-1000);
      }
      await env.SFC_USER_STATE.put(key, JSON.stringify(newState));
      return new Response(JSON.stringify({ status: 'ok' }), {
        headers: { 'Content-Type': 'application/json', ...corsHeaders },
      });
    }

    // POST /user/:username/config — update config only
    const userConfigMatch = path.match(/^\/user\/([^\/]+)\/config$/);
    if (userConfigMatch && method === 'POST') {
      const username = decodeURIComponent(userConfigMatch[1]);
      const key = `user:${username}:state`;
      let existing = await env.SFC_USER_STATE.get(key);
      if (!existing) {
        return new Response('User not found', { status: 404, headers: corsHeaders });
      }
      const state = JSON.parse(existing);
      const configUpdate = await request.json();
      state.config = { ...state.config, ...configUpdate };
      state.last_update = new Date().toISOString();
      await env.SFC_USER_STATE.put(key, JSON.stringify(state));
      return new Response(JSON.stringify({ status: 'ok' }), {
        headers: { 'Content-Type': 'application/json', ...corsHeaders },
      });
    }

    // GET /user/:username/status — quick summary (no full state transfer)
    const userStatusMatch = path.match(/^\/user\/([^\/]+)\/status$/);
    if (userStatusMatch && method === 'GET') {
      const username = decodeURIComponent(userStatusMatch[1]);
      const key = `user:${username}:state`;
      let raw = await env.SFC_USER_STATE.get(key);
      if (!raw) {
        return new Response(JSON.stringify({ exists: false, username }), {
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      }
      const state = JSON.parse(raw);
      const summary = {
        username: state.user_id,
        exists: true,
        capital: state.capital,
        initial_capital: state.initial_capital,
        positions_count: (state.positions || []).length,
        trades_count: (state.trades || []).length,
        last_update: state.last_update,
      };
      return new Response(JSON.stringify(summary), {
        headers: { 'Content-Type': 'application/json', ...corsHeaders },
      });
    }

    // GET /login or /login.html — serve login page directly from worker
    if ((path === '/login' || path === '/login.html') && method === 'GET') {
      const loginHtml = `<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SFC Terminal | Paper Trading Login</title>
    <style>
        *{margin:0;padding:0;box-sizing:border-box}
        body{background:#07080d;font-family:'Space Grotesk',system-ui,-apple-system,sans-serif;display:flex;align-items:center;justify-content:center;min-height:100vh;padding:20px}
        .login-card{background:#0e111a;border-radius:28px;padding:40px;width:100%;max-width:440px;border:1px solid rgba(255,255,255,0.08);box-shadow:0 20px 40px rgba(0,0,0,0.5)}
        .login-card h1{font-size:28px;font-weight:700;background:linear-gradient(135deg,#fff,#7864ff);-webkit-background-clip:text;background-clip:text;color:transparent;margin-bottom:8px}
        .login-card p{color:#8892a8;font-size:14px;margin-bottom:32px;line-height:1.5}
        input{width:100%;padding:14px 18px;background:#1a1f2e;border:1px solid #2a3347;border-radius:16px;color:#edf1f7;font-size:16px;margin-bottom:24px;outline:none}
        input:focus{border-color:#7864ff;box-shadow:0 0 0 2px rgba(120,100,255,0.2)}
        button{width:100%;padding:14px;background:#7864ff;border:none;border-radius:16px;color:white;font-weight:600;font-size:16px;cursor:pointer}
        button:hover{background:#5d4ae0}
        .note{font-size:12px;color:#5a6478;text-align:center;margin-top:24px}
    </style>
</head>
<body>
<div class="login-card">
    <h1>SFC TERMINAL</h1>
    <p>Masukkan username untuk memulai paper trading pribadi.<br>Data Anda disimpan secara privat di Cloudflare KV.</p>
    <input type="text" id="username" placeholder="Contoh: alice, bob, trader1" autocomplete="off" autofocus>
    <button onclick="login()">Mulai Paper Trading \u2192</button>
    <div class="note">\u26a1 Tidak perlu password. Gunakan username yang mudah diingat.</div>
</div>
<script>
function login(){var u=document.getElementById('username').value.trim();if(!u)return alert('Masukkan username');localStorage.setItem('sfc_username',u);window.location.href='/?user='+encodeURIComponent(u)}
document.getElementById('username').addEventListener('keydown',function(e){if(e.key==='Enter')login()});
<\/script>
</body>
</html>`;
      return new Response(loginHtml, {
        status: 200,
        headers: { 'Content-Type': 'text/html; charset=utf-8', ...corsHeaders },
      });
    }

    // ========== EXISTING PROXY ENDPOINTS ==========

    // /events — SSE stream generated by worker polling /snapshot
    if (path === '/events') {
      const preCheck = await fetchAny(urls, '/snapshot', 'application/json');
      if (!preCheck) return new Response('Backend unreachable', { status: 502 });

      const { readable, writable } = new TransformStream();
      const writer = writable.getWriter();
      const encoder = new TextEncoder();

      async function pollAndStream() {
        let lastBtcKey = null;
        let lastSfcTs = null;
        let failCount = 0;
        const MAX_FAIL = 5;
        try {
          for (;;) {
            const resp = await fetchAny(urls, '/snapshot', 'application/json');
            if (resp) {
              failCount = 0;
              const data = await resp.json();
              const now = new Date().toISOString();

              if (data.btc && data.btc.btc != null) {
                const key = Math.round(data.btc.btc * 100);
                if (key !== lastBtcKey) {
                  lastBtcKey = key;
                  const payload = {
                    price: data.btc.btc,
                    change_pct: data.btc.change_pct || 0,
                    high_24h: data.btc.high_24h || 0,
                    low_24h: data.btc.low_24h || 0,
                    volume_24h: data.btc.volume_24h || 0,
                    ts: data.btc.ts || now,
                  };
                  await writer.write(encoder.encode(
                    'event: btc_ticker\ndata: ' + JSON.stringify(payload) + '\n\n'
                  ));
                }
              }

              if (data.sfc && data.sfc.ts !== lastSfcTs) {
                lastSfcTs = data.sfc.ts;
                await writer.write(encoder.encode(
                  'event: sfc_update\ndata: ' + JSON.stringify(data.sfc) + '\n\n'
                ));
              }
            }

            await writer.write(encoder.encode(
              'event: heartbeat\ndata: {"ts":"' + new Date().toISOString() + '"}\n\n'
            ));

            if (!resp) {
              failCount++;
              if (failCount >= MAX_FAIL) {
                await writer.close();
                break;
              }
            }

            await new Promise(r => setTimeout(r, 1000));
          }
        } catch (_) {}
      }

      ctx.waitUntil(pollAndStream());

      return new Response(readable, {
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          ...corsHeaders,
        },
      });
    }

    // /snapshot — initial data
    if (path === '/snapshot') {
      const resp = await fetchAny(urls, '/snapshot', 'application/json');
      if (!resp) return new Response('Backend unreachable', { status: 502 });
      const data = await resp.json();
      return new Response(JSON.stringify(data), {
        status: 200,
        headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-cache', ...corsHeaders },
      });
    }

    // /paper_history.json — paper trading track record
    if (path === '/paper_history.json') {
      const resp = await fetchAny(urls, '/paper_history.json', 'application/json');
      if (!resp) return new Response('{"daily":[],"current":{}}', {
        status: 200,
        headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-cache', ...corsHeaders },
      });
      const data = await resp.json();
      return new Response(JSON.stringify(data), {
        status: 200,
        headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-cache', ...corsHeaders },
      });
    }

    // /paper_trades.json — paper trading server state (for client-side init)
    if (path === '/paper_trades.json') {
      const resp = await fetchAny(urls, '/paper_trades.json', 'application/json');
      if (!resp) return new Response('{"capital":50000,"positions":[],"trades":[],"equity_history":[],"daily_snapshots":{}}', {
        status: 200,
        headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-cache', ...corsHeaders },
      });
      const data = await resp.json();
      return new Response(JSON.stringify(data), {
        status: 200,
        headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-cache', ...corsHeaders },
      });
    }

    // /health
    if (path === '/health') {
      const resp = await fetchAny(urls, '/health', 'application/json');
      if (!resp) return new Response('Backend unreachable', { status: 502 });
      const data = await resp.json();
      return new Response(JSON.stringify(data), {
        status: 200,
        headers: { 'Content-Type': 'application/json', ...corsHeaders },
      });
    }

    // / — index.html
    if (path === '/' || path === '') {
      const resp = await fetchAny(urls, '/', 'text/html');
      if (!resp) return new Response('Backend unreachable', { status: 502 });
      const html = await resp.text();
      return new Response(html, {
        status: 200,
        headers: {
          'Content-Type': 'text/html; charset=utf-8',
          'Cache-Control': 'public, max-age=0, must-revalidate',
        },
      });
    }

    return new Response('Not Found', { status: 404, headers: corsHeaders });
  },
};
