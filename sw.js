/**
 * Service Worker v11 — Network-first for index.html + reload guard
 */

const CACHE_NAME = "sfc-terminal-v13";
const DATA_CACHE = "sfc-data-v6";
const MAX_DATA_AGE_MS = 5 * 60 * 1000; // 5 minutes (was 30 min — dashboard was serving stale data)

const STATIC_ASSETS = [
  "/",
  "index.html",
  "sw.js",
  "manifest.json",
];

self.addEventListener("install", e => {
  console.log("[SW] Installing...");
  e.waitUntil(
    caches.open(CACHE_NAME).then(c => {
      console.log("[SW] Caching static assets");
      return c.addAll(STATIC_ASSETS).catch(err => console.log("[SW] Some assets failed to cache:", err));
    }).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", e => {
  console.log("[SW] Activating...");
  e.waitUntil(
    caches.keys().then(keys => {
      return Promise.all(keys.filter(k => k !== CACHE_NAME && k !== DATA_CACHE).map(k => {
        console.log("[SW] Deleting old cache:", k);
        return caches.delete(k);
      }));
    }).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);

  if (url.pathname.includes("data.json")) {
    // Stale-while-revalidate: serve cached instantly, update in background
    e.respondWith(
      caches.open(DATA_CACHE).then(async cache => {
        const cachedResponse = await cache.match(e.request);
        const networkFetch = fetch(e.request).then(res => {
          if (res.ok) {
            cache.put(e.request, res.clone());
          }
          return res;
        }).catch(() => cachedResponse);

        if (!cachedResponse) {
          return networkFetch;
        }

        // Check if cached version is too old
        const cachedDate = cachedResponse.headers.get("date");
        if (cachedDate) {
          const age = Date.now() - new Date(cachedDate).getTime();
          if (age > MAX_DATA_AGE_MS) {
            console.log("[SW] Cached data is stale, forcing refresh");
            return networkFetch;
          }
        }

        // Serve cached, refresh in background
        // We still fire networkFetch for the next visit
        fetch(e.request).then(res => {
          if (res.ok) {
            cache.put(e.request, res.clone());
          }
        }).catch(() => {});

        return cachedResponse;
      })
    );
    return;
  }

  // Index HTML — network-first: serve fresh if possible, fallback to cache
  if (url.pathname.endsWith("index.html") || url.pathname === "/") {
    e.respondWith(
      fetch(e.request).then(res => {
        if (res.ok) {
          caches.open(CACHE_NAME).then(c => c.put(e.request, res.clone()));
        }
        return res;
      }).catch(() => caches.match(e.request).then(cached => cached || caches.match("/index.html")))
    );
    return;
  }

  e.respondWith(
    caches.match(e.request)
      .then(res => res || fetch(e.request))
      .catch(() => caches.match("/index.html"))
  );
});

self.addEventListener("sync", e => {
  if (e.tag === "sync-data") {
    e.waitUntil(
      fetch("data.json")
        .then(r => r.ok && caches.open(DATA_CACHE).then(c => c.put("data.json", r)))
        .catch(err => console.log("[SW] Sync failed:", err))
    );
  }
});

self.addEventListener("message", e => {
  if (e.data.type === "SYNC_DATA") {
    console.log("[SW] Manual sync requested");
    e.ports[0].postMessage({status: "syncing"});
    fetch("data.json")
      .then(r => r.json())
      .then(data => e.ports[0].postMessage({status: "done", data}))
      .catch(err => e.ports[0].postMessage({status: "error", error: err.message}));
  }
});

// ── Push notification support (NEW) ──
// The browser wakes this Service Worker when a push message arrives from
// the push service (e.g. Chrome/FCM, Firefox's push service) — this is
// purely event-driven, not a poll loop. The actual DECISION about when
// to send a push (i.e. "a BUY signal just appeared") happens server-side
// in the Cloudflare Worker's Cron Trigger (see worker/index.js), which
// calls the Web Push protocol to deliver the message that ends up here.
self.addEventListener("push", (event) => {
  let payload;
  try {
    payload = event.data ? event.data.json() : {};
  } catch (e) {
    payload = { title: "SFC Terminal", body: event.data ? event.data.text() : "New signal update" };
  }

  const title = payload.title || "SFC Terminal — BUY Signal";
  const options = {
    body: payload.body || "A new BUY signal has been detected.",
    icon: payload.icon || "/icon-192.png",
    badge: payload.badge || "/icon-96.png",
    tag: "sfc-buy-signal", // reuses the same notification slot instead of stacking duplicates
    renotify: true,        // but DOES alert again even if reusing the tag (vs silently updating)
    data: { url: payload.url || "/" },
    requireInteraction: false,
  };

  event.waitUntil(self.registration.showNotification(title, options));
});

// Clicking the notification focuses an existing tab if one is open, or
// opens a new one — standard PWA notification-click pattern.
self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const targetUrl = (event.notification.data && event.notification.data.url) || "/";

  event.waitUntil(
    clients.matchAll({ type: "window", includeUncontrolled: true }).then((windowClients) => {
      for (const client of windowClients) {
        if (client.url.includes(self.location.origin) && "focus" in client) {
          return client.focus();
        }
      }
      if (clients.openWindow) {
        return clients.openWindow(targetUrl);
      }
    })
  );
});

console.log("[SW] Service Worker v9 loaded");
