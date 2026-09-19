"""
Resource link health-checker (addresses the "OER link rot" risk flagged
in external review). Sends a HEAD request to every resource URL in the
combined resource set and reports which ones are dead — it does NOT
automatically remove or modify anything; a human should review the
report before touching data files, since a false positive (a server
that blocks HEAD requests, or a temporary outage) could otherwise delete
a perfectly good resource.

IMPORTANT — sandbox limitation, stated honestly: this script could not
be executed against the real internet in the environment that built this
project (its outbound network allowlist covers package registries like
PyPI/npm/GitHub, not arbitrary sites like khanacademy.org or
en.wikipedia.org). It is correct, standard code, and reproducing this on
your own machine is exactly what it's for — it just hasn't been run
against your live resource URLs yet. Run it yourself and read the actual
output before trusting any of these links long-term.

Usage:
    python check_resource_links.py                 # checks all resources
    python check_resource_links.py --timeout 5      # custom timeout (seconds)
"""
from __future__ import annotations
import argparse
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = Path(__file__).parent.parent / "data"
RESOURCE_FILES = ["resources.json", "resources_lecturebank.json"]


def load_all_resources() -> list[dict]:
    resources = []
    for filename in RESOURCE_FILES:
        path = DATA_DIR / filename
        if path.exists():
            resources.extend(json.loads(path.read_text()))
    return resources


def check_url(url: str, timeout: float) -> tuple[bool, str]:
    """Returns (is_alive, detail). Falls back to GET if a server rejects
    HEAD (some do, e.g. certain CDN-fronted sites) before declaring dead."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status < 400, f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        if e.code in (403, 405):  # method not allowed / forbidden on HEAD — retry with GET
            try:
                req_get = urllib.request.Request(url, method="GET", headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req_get, timeout=timeout) as resp:
                    return resp.status < 400, f"HTTP {resp.status} (via GET fallback)"
            except Exception as e2:
                return False, f"GET fallback failed: {e2}"
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=6.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    resources = load_all_resources()
    print(f"Checking {len(resources)} resource URLs (timeout={args.timeout}s, "
          f"{args.workers} parallel workers)...\n")

    dead = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_url, r["url"], args.timeout): r for r in resources}
        for i, future in enumerate(as_completed(futures), 1):
            r = futures[future]
            alive, detail = future.result()
            status = "OK  " if alive else "DEAD"
            print(f"[{i}/{len(resources)}] {status} {detail:<30} {r['id']:<20} {r['url']}")
            if not alive:
                dead.append({**r, "check_detail": detail})

    elapsed = time.perf_counter() - start
    print(f"\nChecked {len(resources)} URLs in {elapsed:.1f}s. {len(dead)} appear dead.")

    if dead:
        report_path = DATA_DIR / "dead_links_report.json"
        report_path.write_text(json.dumps(dead, indent=2))
        print(f"Wrote details to {report_path} — review manually before removing "
              f"anything from resources.json / resources_lecturebank.json.")
        print("Common false positives: sites that block automated HEAD/GET requests")
        print("(some university servers, some CDNs) — verify a few by hand in a")
        print("real browser before trusting this report completely.")


if __name__ == "__main__":
    main()
