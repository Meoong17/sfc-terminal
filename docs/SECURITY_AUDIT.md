# SFC Dashboard — Security Audit (2026-09-10)

Scope: `https://sfcterminal.xyz` (Cloudflare Worker SPA + named tunnel + nginx + `sse_server.py`).
Metode: probe HTTP langsung, uji header, uji auth origin, scan rahasia pada aset publik, telaah
konfigurasi (worker/index.js, nginx vhost, sse_server.py, tunnel, git).

## Kuat (terverifikasi)

| Kontrol | Bukti |
|---|---|
| Auth origin aktif | `SFC_ORIGIN_TOKEN` (64 char) di env sse_server; `8765/`, `/snapshot`, `/events` → 401 tanpa token; hanya `/health` terbuka. Lewat nginx 8090 juga 401. |
| Origin tidak terekspos | 8765 & nginx 8090 loopback-only; tunnel hanya 2 hostname (`sfc-origin.` → 8090, `terminal.altcointrendterminal.` → 8081). |
| Tanpa file disclosure | `/.env`, `/collect.py`, `/git/config`, `/analysis/`, rute acak → semua mengembalikan index.html (SPA catch-all), bukan file asli. nginx deny: `/.env` → 403 (SFC & altcoin). |
| Tanpa rahasia di aset publik | Scan index.html + data.json → 0 hit (sk-, Bearer, api_key, FRED/CF token, private key). data.json tak punya key sensitif. |
| Rahasia tak masuk git | `.env` (perm 600) & kredensial tunnel tidak dilacak; remote SSH tanpa token; `.gitignore` menutup `.env`, `.venv`. |
| Header keamanan | HSTS preload 1 thn; CSP (`frame-ancestors 'none'`, `base-uri 'self'`, `form-action 'self'`); `no-referrer`; `nosniff`; `X-Frame-Options: DENY`; Permissions-Policy ketat. TLS valid (GTS, s/d 2026-10-21); HTTP→HTTPS 301. |

## Temuan

| # | Severitas | Temuan | Status |
|---|---|---|---|
| 1 | SEDANG | **XSS berita**: `index.html` menyisipkan headline RSS mentah (`${body}`, `title="${src}"`) ke `innerHTML`; tidak ada escaping di seluruh file. Judul dari feed pihak ketiga = JS jalan di browser tiap pengunjung. | **FIXED** — `esc()` ditambah; `esc(body/src/label)` diterapkan di kartu berita (index.html ~1919 & ~3898). Live. |
| 2 | RENDAH | CSP `script-src 'unsafe-inline'` + `cdn.jsdelivr.net` melemahkan mitigasi XSS (konsekuensi desain single-file). | Terbuka (by design) — dampak diperkecil oleh fix #1. |
| 3 | RENDAH | `robots.txt` mengiklankan `/sitemap.xml` tetapi publik **404**. Akar: regex anti-scanner nginx vhost `sfc-origin` memuat `sitemap\.xml` → 404. | **FIXED** — `sitemap\.xml` dihapus dari regex nginx; reload. `/sitemap.xml` → 200 `application/xml`. |
| 4 | RENDAH | `/events` (SSE) dapat dibuka publik via Worker (token disuntik otomatis) → stream persisten = permukaan abuse/DoS. | Terbuka — pertimbangkan rate-limit / Cloudflare rule. |
| 5 | RENDAH | Port 80 VPS publik menyajikan halaman default nginx ("Welcome to nginx"); 22/80 terbuka. | Terbuka — tunnel tidak butuh inbound; tutup 80 di firewall. |
| 6 | INFO | CORS allowlist memuat `http://localhost:*` & `http://127.0.0.1:*` + `allow_credentials=True`; robots.txt punya blok Cloudflare terduplikasi. | Risiko rendah / kosmetik. |

## Verifikasi perbaikan

- JS: 2 blok skrip nyata lolos `node --check` (blok ke-3 = JSON-LD, bukan JS).
- `esc()`: `"<img src=x onerror=alert(1)>"` → `&lt;img src=x onerror=alert(1)&gt;`; `null`/`undefined` → `''`.
- Renderer berita (node, logika asli): judul jahat → output tanpa `<img`/`<script>` mentah; 3 headline nyata tetap terender utuh.
- Halaman LIVE mengandung `function esc(s)`, `${esc(body)}`, `title="${esc(src)}"`.
- `/sitemap.xml` publik → 200 `application/xml`.

Catatan: `index.html` ber-flag git `skip-worktree` → perbaikan berlaku live (origin menyajikan file dari disk) tetapi tidak ikut commit, konsisten dengan alur yang ada. Perubahan nginx ada di `/etc/nginx/sites-available/sfc-origin` (backup: `/tmp/sfc-origin.conf.bak`).
