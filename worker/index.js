// sfc-terminal Cloudflare Worker
// Proxies frontend + SSE to VPS via Cloudflare Tunnel
// Multi-user paper trading via KV storage
//
// Worker tries multiple backends in order:
// 1. Cloudflare Tunnel (set via env TUNNEL_URL, falls back to hardcoded default below)
// 2. Direct VPS IP (set via env BACKUP_URL secret — see wrangler.toml / `wrangler secret put`)
//
// IMPORTANT: BACKUP_URL must be set as a Cloudflare secret, NOT committed to source.
// This repo is public — a hardcoded VPS IP here would expose your server address
// to anyone reading the code. Set it once via:
//   wrangler secret put BACKUP_URL
// and paste your VPS URL (e.g. http://YOUR_VPS_IP:8765) when prompted.

const TUNNEL_DEFAULT = 'https://notify-directories-blanket-antibody.trycloudflare.com';

// ── PAPER TRADING REMOVED (2026-07) ─────────────────────────
// SFC Terminal is analysis-only. Multi-user paper trading is gone and all
// /user/* endpoints are disabled (see the 410 guard in the fetch handler).
// The dashboard is fully open — no app login and no Cloudflare Access.

async function fetchAny(env, path, accept) {
  // Try tunnel first, then VPS direct IP (from secret env, if configured)
  const tunnel = (env && env.TUNNEL_URL) || TUNNEL_DEFAULT;
  const backup = env && env.BACKUP_URL; // intentionally no fallback default — see note above
  const ordered = backup ? [tunnel, backup] : [tunnel];
  for (const base of ordered) {
    try {
      const resp = await fetch(base + path, {
        headers: { 'Accept': accept || '*/*', 'Accept-Encoding': 'identity' },
        signal: AbortSignal.timeout(5000),
      });
      if (resp.ok) return resp;
    } catch (_) {}
  }
  return null;
}

// Compression helper — gzip JSON responses for ~80% size reduction
async function gzip(data) {
  if (typeof data === 'string') data = new TextEncoder().encode(data);
  const cs = new CompressionStream('gzip');
  const writer = cs.writable.getWriter();
  writer.write(data);
  writer.close();
  const chunks = [];
  const reader = cs.readable.getReader();
  let total = 0;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    chunks.push(value);
    total += value.length;
  }
  const out = new Uint8Array(total);
  let offset = 0;
  for (const c of chunks) { out.set(c, offset); offset += c.length; }
  return out;
}

// REMOVED (2026-07): cookie helpers, the app's /api/login + signed-session
// cookie, AND Cloudflare Access identity — the dashboard is fully open and
// analysis-only. No per-user state: paper trading (and all /user/* routes)
// is disabled entirely; see the 410 guard in the fetch handler.

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;

    // Allowed origins for CORS (GitHub Pages frontend that calls worker API)
    const ALLOWED_ORIGINS = [
      'https://meoong17.github.io',
      'https://sfc-terminal.meoong17.workers.dev',
      'https://sfcterminal.xyz',
    ];

    // Dynamic CORS — only respond with ACAO when Origin matches allowlist
    function getCorsHeaders(request) {
      const origin = request.headers.get('Origin') || '';
      if (ALLOWED_ORIGINS.includes(origin)) {
        return {
          'Access-Control-Allow-Origin': origin,
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': '*',
          'Vary': 'Origin',
        };
      }
      // Same-origin requests don't need CORS
      return {};
    }

    // Security headers for HTML pages (clickjacking, MIME sniffing, HSTS, referrer)
    const securityHeaders = {
      'X-Frame-Options': 'DENY',
      'X-Content-Type-Options': 'nosniff',
      'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',
      'Referrer-Policy': 'no-referrer',
      // ADDED (2026-07, security hardening pass): CSP restricts which
      // external resources can load — built from the ACTUAL domains this
      // dashboard uses (verified via grep against index.html, not guessed):
      // cdn.jsdelivr.net (JS libs), fonts.googleapis.com/fonts.gstatic.com
      // (Google Fonts). 'unsafe-inline' is needed for script-src/style-src
      // since this dashboard embeds inline <script>/<style> blocks directly
      // in index.html (a nonce-based CSP would be stronger but requires
      // restructuring how the page is served — a bigger change than this
      // pass covers). frame-ancestors 'none' blocks this site from being
      // embedded in an iframe elsewhere (clickjacking protection,
      // reinforcing X-Frame-Options above for browsers that prefer CSP).
      'Content-Security-Policy': [
        "default-src 'self'",
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
        "font-src 'self' https://fonts.gstatic.com",
        "img-src 'self' data:",
        "connect-src 'self' https://sfc-terminal.meoong17.workers.dev https://sfcterminal.xyz",
        "frame-ancestors 'none'",
        "base-uri 'self'",
        "form-action 'self'",
      ].join('; '),
      // Restricts browser features this dashboard has no legitimate need
      // for — reduces attack surface even if a script injection ever
      // succeeded despite CSP above.
      'Permissions-Policy': 'geolocation=(), camera=(), microphone=(), payment=(), usb=()',
    };

    // CORS preflight
    if (method === 'OPTIONS') {
      return new Response(null, { headers: getCorsHeaders(request) });
    }

    // ── ACCESS GUARD REMOVED (2026-07) ───────────────────────
    // Cloudflare Access is no longer in use — the dashboard is fully open
    // and analysis-only. The __cookie_check debug endpoint (which echoed the
    // Access identity header) has been removed along with the Access layer.

    // ========== PAPER TRADING: REMOVED ==========
    // SFC Terminal is analysis-only (2026-07). All /user/* endpoints —
    // multi-user paper-trading state — are disabled. Any stale frontend
    // call gets a clear 410 Gone instead of silently writing to KV.
    if (/^\/user\//.test(path)) {
      return new Response(JSON.stringify({
        status: 'disabled',
        reason: 'Paper Trading removed — SFC Terminal is analysis-only',
      }), {
        status: 410,
        headers: { 'Content-Type': 'application/json', ...getCorsHeaders(request) },
      });
    }

    // POST /api/push-subscribe — store a browser push subscription.
    // Deliberately NOT gated behind the username login system — push
    // notifications are a device/browser-level feature (anyone visiting
    // the dashboard can opt in), independent of the multi-user paper
    // trading state. Subscriptions are keyed by a hash of the push
    // endpoint URL (unique per browser+device registration), so the same
    // browser subscribing twice just overwrites its own entry rather than
    // creating duplicates.
    if (path === '/api/push-subscribe' && method === 'POST') {
      let body;
      try {
        body = await request.json();
      } catch (e) {
        return new Response('Invalid JSON', { status: 400, headers: getCorsHeaders(request) });
      }
      const subscription = body.subscription;
      if (!subscription || !subscription.endpoint) {
        return new Response('Missing subscription', { status: 400, headers: getCorsHeaders(request) });
      }
      // Hash the endpoint URL to get a stable, KV-safe key (endpoint URLs
      // can be long and contain characters not ideal for direct key use).
      try {
        const encoder = new TextEncoder();
        const digest = await crypto.subtle.digest('SHA-256', encoder.encode(subscription.endpoint));
        const hashHex = [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, '0')).join('').slice(0, 32);
        const key = `push:subscription:${hashHex}`;
        await env.SFC_USER_STATE.put(key, JSON.stringify({
          subscription,
          subscribed_at: new Date().toISOString(),
        }));
        console.log(`[Push] Subscribed ${subscription.endpoint.substring(0,40)}... as ${key}`);
      } catch (err) {
        console.log(`[Push] KV write failed: ${err.message}`);
        return new Response(JSON.stringify({ status: 'error', error: err.message }), {
          status: 500,
          headers: { 'Content-Type': 'application/json', ...getCorsHeaders(request) },
        });
      }
      return new Response(JSON.stringify({ status: 'ok' }), {
        headers: { 'Content-Type': 'application/json', ...getCorsHeaders(request) },
      });
    }

    // POST /api/push-unsubscribe — remove a stored subscription (called
    // when a user disables notifications, or the browser reports the
    // subscription as expired).
    if (path === '/api/push-unsubscribe' && method === 'POST') {
      let body;
      try {
        body = await request.json();
      } catch (e) {
        return new Response('Invalid JSON', { status: 400, headers: getCorsHeaders(request) });
      }
      const endpoint = body.endpoint;
      if (!endpoint) {
        return new Response('Missing endpoint', { status: 400, headers: getCorsHeaders(request) });
      }
      const encoder = new TextEncoder();
      const digest = await crypto.subtle.digest('SHA-256', encoder.encode(endpoint));
      const hashHex = [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, '0')).join('').slice(0, 32);
      await env.SFC_USER_STATE.delete(`push:subscription:${hashHex}`);
      return new Response(JSON.stringify({ status: 'ok' }), {
        headers: { 'Content-Type': 'application/json', ...getCorsHeaders(request) },
      });
    }

    // ── AUTH ENDPOINTS ──────────────────────────────────────
    // REMOVED (2026-07): /api/login, /logout, and the /login page, along
    // with Cloudflare Access. The dashboard is fully open (analysis-only).

    // ========== EXISTING PROXY ENDPOINTS ==========

    // /events — SSE stream generated by worker polling /snapshot
    if (path === '/events') {
      const preCheck = await fetchAny(env, '/snapshot', 'application/json');
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
            const resp = await fetchAny(env, '/snapshot', 'application/json');
            if (resp) {
              failCount = 0;
              let data;
              try {
                data = await resp.json();
              } catch (_) {
                // JSON parse error — skip this poll cycle
                await writer.write(encoder.encode(
                  'event: heartbeat\ndata: {"ts":"' + new Date().toISOString() + '"}\n\n'
                ));
                await new Promise(r => setTimeout(r, 1000));
                continue;
              }
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
          ...getCorsHeaders(request),
        },
      });
    }

    // /snapshot — initial data (cached 30s for faster repeat loads)
    if (path === '/snapshot') {
      const cacheKey = new Request(url.toString());
      const cache = caches.default;
      const cached = await cache.match(cacheKey);
      if (cached) return cached;

      const resp = await fetchAny(env, '/snapshot', 'application/json');
      if (!resp) return new Response('Backend unreachable', { status: 502 });
      const data = await resp.json();
      const response = new Response(JSON.stringify(data), {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'public, max-age=30',
          ...getCorsHeaders(request),
        },
      });
      ctx.waitUntil(cache.put(cacheKey, response.clone()));
      return response;
    }

    // /data.json — SFC live data (passthrough, no gzip — saves Worker CPU)
    if (path === '/data.json') {
      const resp = await fetchAny(env, '/data.json', 'application/json');
      if (!resp) return new Response('{}', {
        status: 200,
        headers: { 'Content-Type': 'application/json', ...getCorsHeaders(request) },
      });
      const data = await resp.json();
      return new Response(JSON.stringify(data), {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-cache',
          ...getCorsHeaders(request),
        },
      });
    }

    // /paper_history.json & /paper_trades.json — paper trading is removed
    // (SFC Terminal is analysis-only). Return 410 Gone so any stale frontend
    // gets a clear signal instead of silently reading stale track-record data.
    if (path === '/paper_history.json' || path === '/paper_trades.json') {
      return new Response(JSON.stringify({
        status: 'disabled',
        reason: 'Paper Trading removed — SFC Terminal is analysis-only',
      }), {
        status: 410,
        headers: { 'Content-Type': 'application/json', ...getCorsHeaders(request) },
      });
    }

    // /health
    if (path === '/health') {
      const resp = await fetchAny(env, '/health', 'application/json');
      if (!resp) return new Response('Backend unreachable', { status: 502 });
      const data = await resp.json();
      return new Response(JSON.stringify(data), {
        status: 200,
        headers: { 'Content-Type': 'application/json', ...getCorsHeaders(request) },
      });
    }

    // — index.html — serve dashboard (Cloudflare Access already verified
    // the request before it got here; no app-level auth needed)
    if (path === '/' || path === '') {
      const resp = await fetchAny(env, '/', 'text/html');
      if (!resp) return new Response('Backend unreachable', { status: 502 });
      let html = await resp.text();

      return new Response(html, {
        status: 200,
        headers: {
          'Content-Type': 'text/html; charset=utf-8',
          'Cache-Control': 'public, max-age=0, must-revalidate',
          ...securityHeaders,
          ...getCorsHeaders(request),
        },
      });
    }

    // Static assets — proxy to actual path on backend
    if (path === '/app.js' || path === '/sw.js' || path === '/manifest.json') {
      const resp = await fetchAny(env, path, 'application/javascript');
      if (!resp) return new Response('Not Found', { status: 404 });
      const data = await resp.text();
      const contentType = path.endsWith('.js') ? 'application/javascript' : path.endsWith('.json') ? 'application/json' : 'text/html';
      return new Response(data, {
        status: 200,
        headers: {
          'Content-Type': contentType + '; charset=utf-8',
          'Cache-Control': 'public, max-age=3600',
        },
      });
    }

    // Catch-all: serve dashboard for any unknown path (SPA fallback)
    const resp = await fetchAny(env, '/', 'text/html');
    if (!resp) return new Response('Not Found', { status: 404, headers: getCorsHeaders(request) });
    const fallbackHtml = await resp.text();
    return new Response(fallbackHtml, {
      status: 200,
      headers: {
        'Content-Type': 'text/html; charset=utf-8',
        'Cache-Control': 'public, max-age=0, must-revalidate',
        ...securityHeaders,
        ...getCorsHeaders(request),
      },
    });
  },

  // ── Cron Trigger: BUY signal push notifications ──
  // Configured via wrangler.toml's [triggers] crons — runs on a schedule
  // INDEPENDENT of any HTTP request, which is what makes "notify even
  // when no tab is open" possible. Each run: fetch current data.json,
  // check if the signal just became BUY (edge-triggered, compared
  // against the last-known signal stored in KV — NOT a level check,
  // to avoid re-notifying every run while the signal stays BUY), and if
  // so, send a push message to every stored subscription.
  //
  // ⚠️ IMPORTANT — NOT TESTED END-TO-END: the Web Push protocol
  // implementation below (VAPID JWT signing + RFC8291 payload
  // encryption) was written carefully following the relevant RFCs, but
  // could not be verified against a real push service from the sandbox
  // this was developed in (no network egress available). Test this
  // thoroughly against your own subscribed device before relying on it —
  // if notifications don't arrive, check wrangler tail logs for errors
  // from sendWebPush() below first.
  async scheduled(event, env, ctx) {
    ctx.waitUntil(checkSignalAndNotify(env));
  },
};

async function checkSignalAndNotify(env) {
  try {
    const resp = await fetchAny(env, '/data.json', 'application/json');
    if (!resp) {
      console.log('[Cron] Could not fetch data.json — skipping this cycle');
      return;
    }
    const data = await resp.json();
    const currentSignal = data.signal_type || data.action || null;
    if (!currentSignal) return;

    const lastSignalRaw = await env.SFC_USER_STATE.get('push:last_signal');
    const lastSignal = lastSignalRaw ? JSON.parse(lastSignalRaw).signal : null;

    // Edge-triggered: only notify when the signal TRANSITIONS into BUY,
    // not on every cycle it happens to still be BUY (which would spam a
    // notification every ~5 minutes for as long as the signal holds).
    const justBecameBuy = currentSignal === 'BUY' && lastSignal !== 'BUY';

    await env.SFC_USER_STATE.put('push:last_signal', JSON.stringify({
      signal: currentSignal,
      updated_at: new Date().toISOString(),
    }));

    if (!justBecameBuy) return;

    if (!env.VAPID_PRIVATE_KEY || !env.VAPID_PUBLIC_KEY || !env.VAPID_SUBJECT) {
      console.log('[Cron] VAPID secrets not configured — cannot send push. '
        + 'Set via: wrangler secret put VAPID_PRIVATE_KEY / VAPID_PUBLIC_KEY, '
        + 'and VAPID_SUBJECT (e.g. mailto:you@example.com)');
      return;
    }

    const list = await env.SFC_USER_STATE.list({ prefix: 'push:subscription:' });
    const payload = JSON.stringify({
      title: 'SFC Terminal — BUY Signal',
      body: `SFC ${data.sfc_effective != null ? data.sfc_effective.toFixed(1) + '%' : ''} · BTC $${Math.round(data.btc || 0).toLocaleString()}`,
      url: '/',
    });

    let sent = 0, failed = 0;
    for (const item of list.keys) {
      const raw = await env.SFC_USER_STATE.get(item.name);
      if (!raw) continue;
      const { subscription } = JSON.parse(raw);
      try {
        await sendWebPush(subscription, payload, env);
        sent++;
      } catch (err) {
        failed++;
        console.log(`[Cron] Push failed for ${item.name}: ${err.message}`);
        // 410 Gone / 404 means the subscription is no longer valid (user
        // uninstalled, cleared browser data, etc.) — clean it up so future
        // cycles don't keep trying a dead endpoint.
        if (err.message.includes('410') || err.message.includes('404')) {
          await env.SFC_USER_STATE.delete(item.name);
        }
      }
    }
    console.log(`[Cron] BUY signal notification: ${sent} sent, ${failed} failed`);
  } catch (err) {
    console.log('[Cron] checkSignalAndNotify error:', err.message);
  }
}

// ── Web Push protocol implementation ──
// Sends a single push message per RFC8030 (Web Push protocol) with
// RFC8291 (message encryption, aes128gcm) and RFC8292 (VAPID auth).
// This is a from-scratch implementation using only the Web Crypto API
// available in the Workers runtime (no external npm packages, since
// this worker is deployed as a single file with no build step) —
// see the scheduled() handler's warning above about live-testing this.

async function sendWebPush(subscription, payloadString, env) {
  const endpoint = subscription.endpoint;
  const p256dhKey = subscription.keys.p256dh;
  const authSecret = subscription.keys.auth;

  const audience = new URL(endpoint).origin;
  const vapidHeaders = await buildVapidHeaders(audience, env.VAPID_SUBJECT, env.VAPID_PUBLIC_KEY, env.VAPID_PRIVATE_KEY);
  const encrypted = await encryptPayload(payloadString, p256dhKey, authSecret);

  const resp = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'TTL': '86400', // push service may drop the message after this many seconds if undelivered
      'Content-Encoding': 'aes128gcm',
      'Content-Type': 'application/octet-stream',
      ...vapidHeaders,
    },
    body: encrypted,
  });

  if (!resp.ok) {
    throw new Error(`Push service responded ${resp.status}: ${await resp.text()}`);
  }
}

// Base64url helpers (Web Push uses unpadded base64url throughout)
function base64urlToBytes(base64url) {
  const padding = '='.repeat((4 - base64url.length % 4) % 4);
  const base64 = (base64url + padding).replace(/-/g, '+').replace(/_/g, '/');
  const binary = atob(base64);
  return Uint8Array.from([...binary].map(c => c.charCodeAt(0)));
}

function bytesToBase64url(bytes) {
  let binary = '';
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

// VAPID (RFC8292): a JWT signed with the VAPID private key (ES256/ECDSA
// P-256), proving to the push service that this server is authorized to
// send to this subscription.
async function buildVapidHeaders(audience, subject, publicKeyB64, privateKeyB64) {
  const header = { typ: 'JWT', alg: 'ES256' };
  const exp = Math.floor(Date.now() / 1000) + 12 * 3600; // 12h, well under the spec's 24h max
  const claims = { aud: audience, exp, sub: subject };

  const encoder = new TextEncoder();
  const headerB64 = bytesToBase64url(encoder.encode(JSON.stringify(header)));
  const claimsB64 = bytesToBase64url(encoder.encode(JSON.stringify(claims)));
  const unsignedToken = `${headerB64}.${claimsB64}`;

  const privateKeyBytes = base64urlToBytes(privateKeyB64);
  const publicKeyBytes = base64urlToBytes(publicKeyB64); // 65 bytes, uncompressed point (0x04 + X + Y)
  const x = publicKeyBytes.slice(1, 33);
  const y = publicKeyBytes.slice(33, 65);

  const jwk = {
    kty: 'EC',
    crv: 'P-256',
    x: bytesToBase64url(x),
    y: bytesToBase64url(y),
    d: bytesToBase64url(privateKeyBytes),
    ext: true,
  };

  const cryptoKey = await crypto.subtle.importKey(
    'jwk', jwk, { name: 'ECDSA', namedCurve: 'P-256' }, false, ['sign']
  );

  const signature = await crypto.subtle.sign(
    { name: 'ECDSA', hash: 'SHA-256' }, cryptoKey, encoder.encode(unsignedToken)
  );

  const jwt = `${unsignedToken}.${bytesToBase64url(new Uint8Array(signature))}`;

  return {
    'Authorization': `vapid t=${jwt}, k=${publicKeyB64}`,
  };
}

// RFC8291 message encryption (aes128gcm content-coding). This derives a
// shared secret via ECDH between an ephemeral server keypair and the
// subscription's p256dh public key, combines it with the subscription's
// auth secret via HKDF, and encrypts the payload with AES-128-GCM.
async function encryptPayload(payloadString, p256dhKeyB64, authSecretB64) {
  const encoder = new TextEncoder();
  const plaintext = encoder.encode(payloadString);

  const clientPublicKeyBytes = base64urlToBytes(p256dhKeyB64);
  const authSecret = base64urlToBytes(authSecretB64);

  // Import the CLIENT's public key (from the subscription) for ECDH
  const clientX = clientPublicKeyBytes.slice(1, 33);
  const clientY = clientPublicKeyBytes.slice(33, 65);
  const clientPublicKey = await crypto.subtle.importKey(
    'jwk',
    { kty: 'EC', crv: 'P-256', x: bytesToBase64url(clientX), y: bytesToBase64url(clientY), ext: true },
    { name: 'ECDH', namedCurve: 'P-256' }, false, []
  );

  // Generate an ephemeral SERVER keypair for this one message
  const serverKeyPair = await crypto.subtle.generateKey(
    { name: 'ECDH', namedCurve: 'P-256' }, true, ['deriveBits']
  );
  const serverPublicKeyRaw = new Uint8Array(
    await crypto.subtle.exportKey('raw', serverKeyPair.publicKey)
  );

  // ECDH shared secret
  const sharedSecretBits = await crypto.subtle.deriveBits(
    { name: 'ECDH', public: clientPublicKey }, serverKeyPair.privateKey, 256
  );
  const sharedSecret = new Uint8Array(sharedSecretBits);

  // salt: 16 random bytes, sent alongside the ciphertext so the client
  // can reverse the key derivation
  const salt = crypto.getRandomValues(new Uint8Array(16));

  // HKDF per RFC8291 section 3.3-3.4
  const prkKey = await crypto.subtle.importKey('raw', authSecret, { name: 'HKDF' }, false, ['deriveBits']);
  const keyInfo = concatBytes(
    encoder.encode('WebPush: info\0'), clientPublicKeyBytes, serverPublicKeyRaw
  );
  const ikm = new Uint8Array(await crypto.subtle.deriveBits(
    { name: 'HKDF', hash: 'SHA-256', salt: sharedSecret.buffer, info: keyInfo }, prkKey, 256
  ));
  // NOTE: this uses sharedSecret as the HKDF "salt" input for the PRK
  // derivation step (per RFC8291's two-stage HKDF: auth_secret extracts
  // a PRK from the ECDH shared secret, keyed by "WebPush: info"+keys) —
  // this is the step most likely to have a subtle bug if something's
  // wrong, since it's the least commonly hand-implemented part of the spec.

  const prk2Key = await crypto.subtle.importKey('raw', ikm, { name: 'HKDF' }, false, ['deriveBits']);
  const cekInfo = encoder.encode('Content-Encoding: aes128gcm\0');
  const cekBits = await crypto.subtle.deriveBits(
    { name: 'HKDF', hash: 'SHA-256', salt: salt.buffer, info: cekInfo }, prk2Key, 128
  );
  const contentEncryptionKey = new Uint8Array(cekBits);

  const nonceInfo = encoder.encode('Content-Encoding: nonce\0');
  const nonceBits = await crypto.subtle.deriveBits(
    { name: 'HKDF', hash: 'SHA-256', salt: salt.buffer, info: nonceInfo }, prk2Key, 96
  );
  const nonce = new Uint8Array(nonceBits);

  // Padding: a single 0x02 delimiter byte + no extra padding (minimal,
  // valid per spec — padding is optional, used here for simplicity).
  const paddedPlaintext = concatBytes(plaintext, new Uint8Array([2]));

  const aesKey = await crypto.subtle.importKey('raw', contentEncryptionKey, { name: 'AES-GCM' }, false, ['encrypt']);
  const ciphertext = new Uint8Array(
    await crypto.subtle.encrypt({ name: 'AES-GCM', iv: nonce }, aesKey, paddedPlaintext)
  );

  // aes128gcm content-coding header: salt(16) + record size(4, big-endian)
  // + key id length(1) + key id (server's ephemeral public key, 65 bytes)
  const recordSize = new Uint8Array(4);
  new DataView(recordSize.buffer).setUint32(0, 4096, false);
  const header = concatBytes(
    salt, recordSize, new Uint8Array([serverPublicKeyRaw.length]), serverPublicKeyRaw
  );

  return concatBytes(header, ciphertext);
}

function concatBytes(...arrays) {
  const total = arrays.reduce((sum, a) => sum + a.length, 0);
  const result = new Uint8Array(total);
  let offset = 0;
  for (const arr of arrays) {
    result.set(arr, offset);
    offset += arr.length;
  }
  return result;
}

