"""
Online Resource Retrieval — Methodology Section 5.

This is a STANDALONE ingestion script, run offline/periodically, NOT called
during a live recommendation request (calling an external API inside the
request path would make response latency depend on YouTube's uptime and
burn API quota per learner interaction — bad practice for a production app).

It fetches real video metadata via the official YouTube Data API v3 and
writes rows shaped exactly like backend/data/resources.json, so the output
can be merged into the seed file (or, in production, into a real database
such as SQLite/Postgres).

Requires a real API key: https://console.cloud.google.com/apis/credentials
(YouTube Data API v3, free tier: 10,000 quota units/day, a search.list call
costs 100 units -> ~100 searches/day on the free tier). Set it as an
environment variable, never hard-code it:

    export YOUTUBE_API_KEY="your-key-here"

Run:
    python ingestion.py --concept "gradient descent" --concept_id gradient_descent --max_results 5
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request

API_KEY_ENV = "YOUTUBE_API_KEY"
SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"


def _iso8601_duration_to_minutes(duration: str) -> int:
    """Parse ISO 8601 durations like 'PT15M33S' -> minutes (rounded up)."""
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
    if not match:
        return 0
    h, m, s = (int(x) if x else 0 for x in match.groups())
    total_seconds = h * 3600 + m * 60 + s
    return max(1, (total_seconds + 59) // 60)


def fetch_youtube_resources(query: str, concept_id: str, max_results: int = 5) -> list[dict]:
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"{API_KEY_ENV} is not set. This function makes a real network call "
            "to the YouTube Data API and requires a valid key — see module docstring."
        )

    search_params = urllib.parse.urlencode({
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "key": api_key,
        "safeSearch": "strict",
        "relevanceLanguage": "en",
    })
    with urllib.request.urlopen(f"{SEARCH_URL}?{search_params}") as resp:
        search_data = json.load(resp)

    video_ids = [item["id"]["videoId"] for item in search_data.get("items", [])]
    if not video_ids:
        return []

    videos_params = urllib.parse.urlencode({
        "part": "contentDetails,statistics,snippet",
        "id": ",".join(video_ids),
        "key": api_key,
    })
    with urllib.request.urlopen(f"{VIDEOS_URL}?{videos_params}") as resp:
        videos_data = json.load(resp)

    results = []
    for item in videos_data.get("items", []):
        duration_min = _iso8601_duration_to_minutes(item["contentDetails"]["duration"])
        stats = item.get("statistics", {})
        view_count = int(stats.get("viewCount", 0))
        like_count = int(stats.get("likeCount", 0))
        # Simple, transparent quality heuristic — NOT validated against
        # ground truth. Replace with a real model once you have labelled
        # quality/completion data (see RESEARCH_NOTE.md Action Item #2).
        quality = min(1.0, 0.5 + (like_count / max(view_count, 1)) * 10)
        results.append({
            "id": f"yt_{item['id']}",
            "title": item["snippet"]["title"],
            "concept": concept_id,
            "url": f"https://www.youtube.com/watch?v={item['id']}",
            "source": "youtube",
            "type": "video",
            "duration_min": duration_min,
            "difficulty": None,  # not derivable from API metadata — needs manual/LLM tagging
            "quality": round(quality, 3),
        })
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept", required=True, help="Search query, e.g. 'gradient descent tutorial'")
    parser.add_argument("--concept_id", required=True, help="Knowledge-graph concept id, e.g. 'gradient_descent'")
    parser.add_argument("--max_results", type=int, default=5)
    args = parser.parse_args()

    try:
        rows = fetch_youtube_resources(args.concept, args.concept_id, args.max_results)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(rows, indent=2))
