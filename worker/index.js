// sfc-terminal Cloudflare Worker
// Proxies frontend + SSE to VPS via Cloudflare Tunnel
//
// Worker tries multiple backends in order:
// 1. Cloudflare Tunnel (set via env TUNNEL_URL or hardcoded below)
// 2. Direct VPS IP (blocked by some Tencent Cloud security groups)

const TUNNEL = 'https://excitement-baghdad-remark-colored.trycloudflare.com';
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

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;
    const urls = [TUNNEL, BACKUP];

    // CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, OPTIONS',
          'Access-Control-Allow-Headers': '*',
        },
      });
    }

    // /events — SSE stream
    if (path === '/events') {
      const resp = await fetchAny(urls, '/events', 'text/event-stream');
      if (!resp) return new Response('Backend unreachable', { status: 502 });
      return new Response(resp.body, {
        status: resp.status,
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          'Access-Control-Allow-Origin': '*',
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
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-cache',
          'Access-Control-Allow-Origin': '*',
        },
      });
    }

    // /health
    if (path === '/health') {
      const resp = await fetchAny(urls, '/health', 'application/json');
      if (!resp) return new Response('Backend unreachable', { status: 502 });
      const data = await resp.json();
      return new Response(JSON.stringify(data), {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
        },
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

    return new Response('Not Found', { status: 404 });
  },
};
