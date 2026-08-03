# SFC Terminal — Security Hardening: Deploy Steps (run AFTER token rotation)

Origin sudah diamankan di sisi VPS (localhost-only + Bearer token + nginx deny layer).
Yang tersisa adalah langkah Cloudflare agar dashboard feed jalan lagi dengan origin
yang aman. JALANKAN INI HANYA SETELAH `CLOUDFLARE_API_TOKEN` DIROTASI (lihat bagian
terakhir) — jangan pakai token lama yang sudah bocor.

## 0) Prasyarat: token Cloudflare baru
Ganti token di panel: https://dash.cloudflare.com/136819e77d5874e68b1c1ca6588c8a0b/api-tokens
Lalu update /home/ubuntu/sfc/.env dan ekspor ke env shell sesi ini:
    export CLOUDFLARE_API_TOKEN="<TOKEN_BARU>"

## 1) Tambah ingress named-tunnel untuk origin SFC -> nginx:8090
Edit /home/ubuntu/.cloudflared/config.yml, sisipkan hostname SFC di ATAS catch-all:
```yaml
ingress:
  - hostname: terminal.altcointrendterminal.site
    service: http://localhost:8081
  - hostname: sfc-origin.altcointrendterminal.site
    service: http://127.0.0.1:8090        # <- baru (nginx deny layer di depan sse_server)
  - service: http_status:404
```

## 2) Buat DNS CNAME (zone altcointrendterminal.site)
ZONE_ID=$(curl -s -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  "https://api.cloudflare.com/client/v4/zones?name=altcointrendterminal.site" | jq -r '.result[0].id')
curl -s -X POST "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records" \
  -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" -H "Content-Type: application/json" \
  --data '{"type":"CNAME","name":"sfc-origin","content":"6842682a-38f5-40fc-88ca-b78656f9a77b.cfargotunnel.com","proxied":true}'

## 3) Restart tunnel bernama
sudo systemctl restart cloudflared-altcoin
curl -s -o /dev/null -w 'origin HTTP %{http_code}\n' https://sfc-origin.altcointrendterminal.site/data.json

## 4) Set secret worker + deploy
cd /home/ubuntu/sfc
npx wrangler secret put SFC_ORIGIN --env production          # https://sfc-origin.altcointrendterminal.site
npx wrangler secret put SFC_ORIGIN_TOKEN --env production     # BTAdNNMyHSlnKTw7Sm3orbZ6ekK5kDSRsJl_pwxa3Iw
npx wrangler secret delete TUNNEL_URL --env production        # buang URL trycloudflare yang mati
npx wrangler deploy --env production

Verifikasi feed:
    curl -s -N --max-time 6 https://sfc-terminal.meoong17.workers.dev/events | head

## 5) Verifikasi publik: path sensitif harus TUTUP
    for p in .env .git/config sse_server.py sfc-pipeline.log; do
      echo "  /$p -> $(curl -s -o /dev/null -w '%{http_code}' https://sfc-origin.altcointrendterminal.site/$p)"
    done
Semua harus 403 (blocked nginx) atau 401/404 (token/whitelist). Tidak ada yang 200.

=====================================================================
## ROTASI 9 API KEY (WAJIB — .env sempat bocor ke publik)
=====================================================================
Ganti semua, lalu update /home/ubuntu/sfc/.env, restart service yang memakainya.

  CLOUDFLARE_API_TOKEN -> dash.cloudflare.com > My Profile > API Tokens   (paling kritis)
  CMC_API_KEY          -> coinmarketcap.com > dashboard
  COINGLASS_API_KEY    -> coinglass.com > API settings
  CRYPTOPANIC_KEY      -> cryptopanic.com > account
  GOLDAPI_KEY          -> goldapi.io > dashboard
  TWELVEDATA_KEY       -> twelvedata.com > account
  ALPHAVANTAGE_KEY     -> alphavantage.co > account
  FRED_API_KEY         -> fred.stlouisfed.org > my account
  SOPR_API_KEY         -> sesuai penyedia SOPR lo
  COINGECKO_API_KEY    -> coingecko.com > Developers > API Keys   [DONE 2026-08-03]

Catatan: nilai-nilai di atas TIDAK pernah dicetak ke terminal/laporan ini —
hanya nama key-nya. Rotasi = buat baru di panel, cabut yang lama setelah
pengganti aktif.

=====================================================================
## COINGECKO_API_KEY — refactor hardcode -> env (DONE 2026-08-03)
=====================================================================
Key CoinGecko demo (`CG-5jhQuf...`) pernah hardcode di source (committed secret).
Telah di-rotasi di dashboard dan semua call-site diubah baca dari env, supaya
rotasi berikutnya tinggal ganti .env tanpa edit source.

File yang sudah diubah ke `os.getenv("COINGECKO_API_KEY","")`:
  - collect.py                        (CG_API_KEY)
  - data_sources/methods_institutional.py  (CG_API_PARAM)
  - ml/sfc_advanced.py                (inline hardcode, baris ~788)
  - data_sources/stablecoin_liquidity.py   (sudah env; komentar dibersihkan)
  - data_sources/stablecoin_intelligence.py(sudah env; komentar dibersihkan)

Semua dibaca via `collect.py:117 load_dotenv()` -> ambil dari /home/ubuntu/sfc/.env.
Tanpa key di .env => request CG 401 => modul degrade ke netral (bukan crash).

Verifikasi: HTTP 200 pada simple/price dengan key .env; grep "CG-5jhQuf..." = 0 di source.

BONUS FIX (commit 1e097828): sfc-pipeline.sh auto-commit sekarang `git commit -- <data
pathspec>` saja — file code yang di-`git add` manual tidak lagi tersapu ke commit
"auto: SFC data" yang generic. Script di repo & ~/.hermes/scripts harus tetap sync.

Note: key lama tetap ada di riwayat git sebelum rotate. Rotasi membuatnya tidak
berguna. Rewrite history (filter-repo) untuk menghapusnya = destructive, opsional.

