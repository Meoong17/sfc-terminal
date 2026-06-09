/**
 * Service Worker v7 — Offline-first + real-time data sync
 */

const CACHE_NAME = "sfc-terminal-v7";
const DATA_CACHE = "sfc-data-v1";
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
    e.respondWith(
      fetch(e.request)
        .then(res => {
          if (res.ok) {
            caches.open(DATA_CACHE).then(c => c.put(e.request, res.clone()));
          }
          return res;
        })
        .catch(() => {
          return caches.match(e.request) || caches.match("/index.html").then(r => r ? new Response("Data unavailable - using cached version", {status: 200}) : r);
        })
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

console.log("[SW] Service Worker loaded");
