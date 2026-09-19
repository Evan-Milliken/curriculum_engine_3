const CACHE_NAME = "curriculum-shell-v1";
const SHELL_FILES = ["/index.html", "/manifest.json"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_FILES))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
});

// App-shell files: cache-first. API calls to the backend: always go to
// the network (never cache learner data/plans).
self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (SHELL_FILES.some((f) => url.pathname.endsWith(f))) {
    event.respondWith(
      caches.match(event.request).then((cached) => cached || fetch(event.request))
    );
  }
});
