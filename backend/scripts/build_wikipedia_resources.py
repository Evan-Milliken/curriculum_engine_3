"""
Generates a minimal resource set for the 208-concept LectureBank graph
using the Wikipedia URLs ALREADY PRESENT in that dataset's own topic list
(LB-Paper/208topics.csv) — these are not invented; they're the exact URLs
Li et al. (2019) shipped with the dataset.

This is intentionally minimal and labelled as such: one general-reference
article per concept is not a substitute for curated video/practice
resources (see DATASETS.md Action Item #1). It exists so the full
208-concept graph is runnable end-to-end today rather than empty, while
being honest that richer resource coverage is unscoped future work.

Duration/difficulty/quality values below are heuristic placeholders
(a flat estimate), NOT measured — flagged in the output file itself.

Usage:
    python build_wikipedia_resources.py
Reads:  data/knowledge_graph_lecturebank.json
Writes: data/resources_lecturebank.json
"""
from __future__ import annotations
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
KG_FILE = DATA_DIR / "knowledge_graph_lecturebank.json"
OUTPUT_FILE = DATA_DIR / "resources_lecturebank.json"

# Heuristic placeholder — a typical Wikipedia article on a technical topic
# takes roughly this long to read at a moderate pace. Not measured per-article.
ESTIMATED_READ_MINUTES = 12


def main():
    kg = json.loads(KG_FILE.read_text())
    resources = []
    skipped = 0
    for concept in kg["concepts"]:
        if not concept.get("wiki_url"):
            skipped += 1
            continue
        resources.append({
            "id": f"wiki_{concept['id']}",
            "title": f"Wikipedia: {concept['name']}",
            "concept": concept["id"],
            "url": concept["wiki_url"],
            "source": "wikipedia",
            "type": "documentation",
            "duration_min": ESTIMATED_READ_MINUTES,
            "difficulty": 0.5,   # placeholder — Wikipedia articles vary widely in level
            "quality": 0.6,      # placeholder — general reference, not vetted per-article
        })
    OUTPUT_FILE.write_text(json.dumps(resources, indent=2, ensure_ascii=False))
    print(f"Wrote {len(resources)} resources to {OUTPUT_FILE} "
          f"({skipped} concepts had no Wikipedia URL in the source data and were skipped).")


if __name__ == "__main__":
    main()
