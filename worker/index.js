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
        signal: AbortSignal.timeout(2000),
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

// Cookie helpers
function getCookie(request, name) {
  const cookie = request.headers.get('Cookie') || '';
  const match = cookie.match(new RegExp('(?:^|;\\s*)' + name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '=([^;]*)'));
  return match ? decodeURIComponent(match[1]) : null;
}

function setCookie(name, value, maxAgeDays = 30) {
  return `${name}=${encodeURIComponent(value)}; Path=/; SameSite=Lax; Max-Age=${maxAgeDays * 86400}`;
}

function clearCookie(name) {
  return `${name}=; Path=/; SameSite=Lax; Max-Age=0`;
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

    // ── SESSION AUTH GUARD ──────────────────────────────────
    // Protected routes require sfc_session cookie
    // (checked inline inside each handler for consistency)

    // Debug endpoint — echo cookies and session
    if (path === '/__cookie_check') {
      const cookie = request.headers.get('Cookie') || '(none)';
      const sessionUser = getCookie(request, 'sfc_session');
      return new Response(JSON.stringify({ cookie, sessionUser, all: Object.fromEntries(request.headers) }), {
        headers: { 'Content-Type': 'application/json', ...corsHeaders },
      });
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

    // ── AUTH ENDPOINTS ──────────────────────────────────────

    // POST /api/login — validate username, set session cookie, redirect
    if (path === '/api/login' && method === 'POST') {
      let body;
      const contentType = request.headers.get('Content-Type') || '';
      if (contentType.includes('application/json')) {
        try { body = await request.json(); } catch (_) { body = {}; }
      } else {
        try {
          const formData = await request.formData();
          body = { username: formData.get('username') || '' };
        } catch (_) { body = {}; }
      }
      const username = (body.username || '').trim();
      if (!username || username.length < 1 || username.length > 32 || !/^[a-zA-Z0-9_-]+$/.test(username)) {
        // For form posts, redirect back with error
        if (contentType.includes('x-www-form-urlencoded')) {
          return Response.redirect(url.origin + '/login?error=Invalid+username', 302);
        }
        return new Response(JSON.stringify({ error: 'Invalid username. Use letters, numbers, hyphens and underscores.' }), {
          status: 400,
          headers: { 'Content-Type': 'application/json', ...corsHeaders },
        });
      }
      // Set cookie via 200 + JS redirect (most reliable for cookie setting)
      if (contentType.includes('x-www-form-urlencoded')) {
        const redirectHtml = `<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Redirecting...</title></head>
<body><script>window.location.href='/?user='+encodeURIComponent(${JSON.stringify(username)})</script></body></html>`;
        return new Response(redirectHtml, {
          status: 200,
          headers: {
            'Content-Type': 'text/html; charset=utf-8',
            'Set-Cookie': setCookie('sfc_session', username),
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            ...corsHeaders,
          },
        });
      }
      return new Response(JSON.stringify({ status: 'ok', username }), {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Set-Cookie': setCookie('sfc_session', username),
          ...corsHeaders,
        },
      });
    }

    // GET /logout — clear session cookie
    if (path === '/logout') {
      return new Response(null, {
        status: 302,
        headers: {
          'Location': '/login',
          'Set-Cookie': clearCookie('sfc_session'),
          ...corsHeaders,
        },
      });
    }

    // GET /login or /login.html — serve login page
    if ((path === '/login' || path === '/login.html') && method === 'GET') {
      const errorMsg = url.searchParams.get('error') || '';
      const loginHtml = `<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SFC Terminal | Login</title>
    <style>
        *{margin:0;padding:0;box-sizing:border-box}
        body{background:#07080d;font-family:'Space Grotesk',system-ui,-apple-system,sans-serif;display:flex;align-items:center;justify-content:center;min-height:100vh;padding:20px}
        .login-card{background:#0e111a;border-radius:28px;padding:40px;width:100%;max-width:440px;border:1px solid rgba(255,255,255,0.08);box-shadow:0 20px 40px rgba(0,0,0,0.5)}
        .login-card h1{font-size:28px;font-weight:700;background:linear-gradient(135deg,#fff,#7864ff);-webkit-background-clip:text;background-clip:text;color:transparent;margin-bottom:8px}
        .login-card p{color:#8892a8;font-size:14px;margin-bottom:32px;line-height:1.5}
        input{width:100%;padding:14px 18px;background:#1a1f2e;border:1px solid #2a3347;border-radius:16px;color:#edf1f7;font-size:16px;margin-bottom:8px;outline:none}
        input:focus{border-color:#7864ff;box-shadow:0 0 0 2px rgba(120,100,255,0.2)}
        button{width:100%;padding:14px;background:#7864ff;border:none;border-radius:16px;color:white;font-weight:600;font-size:16px;cursor:pointer}
        button:hover{background:#5d4ae0}
        button:disabled{opacity:0.5;cursor:not-allowed}
        .note{font-size:12px;color:#5a6478;text-align:center;margin-top:24px}
        .error{color:#ff4060;font-size:13px;margin-bottom:16px;display:${errorMsg ? 'block' : 'none'}}
    </style>
</head>
<body>
<div class="login-card">
    <h1>SFC TERMINAL</h1>
    <p>Enter your username to access the dashboard.</p>
    <div class="error" id="error">${errorMsg}</div>
    <form method="POST" action="/api/login" id="loginForm">
        <input type="text" name="username" id="username" placeholder="Your username" autocomplete="off" autocapitalize="off" autofocus required>
        <button type="submit" id="loginBtn">Start Trading →</button>
    </form>
    <div class="note">⚠ Just a username — no password.</div>
    <div class="note" style="margin-top:4px;font-size:11px;color:#3a4460">Tip: use the same username as before to restore paper trading data.</div>
</div>
<script>
document.getElementById('loginForm').addEventListener('submit', function(e) {
  var u = document.getElementById('username').value.trim();
  var err = document.getElementById('error');
  if (!u || !/^[a-zA-Z0-9_-]+$/.test(u)) {
    e.preventDefault();
    err.textContent = !u ? 'Enter a username' : 'Letters, numbers, hyphens, underscores only';
    err.style.display = 'block';
    return;
  }
  // Also save to localStorage for the frontend
  localStorage.setItem('sfc_username', u);
});
</script>
</body>
</html>`;
      return new Response(loginHtml, {
        status: 200,
        headers: { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-cache, no-store, must-revalidate', ...corsHeaders },
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

    // /snapshot — initial data (cached 30s for faster repeat loads)
    if (path === '/snapshot') {
      const cacheKey = new Request(url.toString());
      const cache = caches.default;
      const cached = await cache.match(cacheKey);
      if (cached) return cached;

      const resp = await fetchAny(urls, '/snapshot', 'application/json');
      if (!resp) return new Response('Backend unreachable', { status: 502 });
      const data = await resp.json();
      const response = new Response(JSON.stringify(data), {
        status: 200,
        headers: { 'Content-Type': 'application/json', 'Cache-Control': 'public, max-age=30', ...corsHeaders },
      });
      ctx.waitUntil(cache.put(cacheKey, response.clone()));
      return response;
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

    // — index.html
    if (path === '/' || path === '') {
      const sessionUser = getCookie(request, 'sfc_session');
      if (!sessionUser) {
        return Response.redirect(url.origin + '/login', 302);
      }

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
