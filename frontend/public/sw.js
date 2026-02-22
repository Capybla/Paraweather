const CACHE_NAME = 'paraweather-v2';
const APP_SHELL = ['/', '/index.html', '/manifest.json'];
const MAX_DYNAMIC_CACHE_ENTRIES = 80;

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL)));
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key)))),
  );
  self.clients.claim();
});

async function trimCache(cache) {
  const keys = await cache.keys();
  if (keys.length <= MAX_DYNAMIC_CACHE_ENTRIES) return;
  const removals = keys.slice(0, keys.length - MAX_DYNAMIC_CACHE_ENTRIES).map((key) => cache.delete(key));
  await Promise.all(removals);
}

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  const requestUrl = new URL(event.request.url);

  // Avoid caching API responses and third-party map tiles to reduce storage usage
  if (requestUrl.pathname.startsWith('/api/') || requestUrl.hostname !== self.location.hostname) {
    return;
  }

  event.respondWith(
    caches.match(event.request).then((cached) => {
      const networkFetch = fetch(event.request)
        .then(async (response) => {
          const responseClone = response.clone();
          const cache = await caches.open(CACHE_NAME);
          await cache.put(event.request, responseClone);
          await trimCache(cache);
          return response;
        })
        .catch(() => cached);

      return cached || networkFetch;
    }),
  );
});
