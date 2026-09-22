"""
Build a real, published, human-annotated knowledge graph from the
LectureBank dataset (Li, Fabbri, Tung, Radev — "What Should I Learn First:
Introducing LectureBank for NLP Education and Prerequisite Chain Learning",
AAAI 2019: https://arxiv.org/abs/1811.12181, data:
https://github.com/Yale-LILY/LectureBank).

This is REAL data, not a placeholder: 208 NLP/ML/AI/Deep-Learning/IR
concepts, with every pair manually annotated by the paper's authors for
"is A a prerequisite of B?", released publicly under
LB-Paper/prerequisite_annotation.csv and LB-Paper/208topics.csv.

WHY THIS MATTERS FOR THE PROJECT: it replaces a hand-built, unvalidated
10-concept toy graph with 208 concepts and ~900 prerequisite edges backed
by a peer-reviewed annotation process (not our own guesses) — real
academic grounding for the "Knowledge Graph Construction" step of the
methodology (Section 4), scoped specifically to CS/AI/ML/NLP, which
matches this project's target domain far better than a generic MOOC
dataset.

KNOWN DATA ISSUE, HANDLED HERE: human pairwise annotation over ~208*207
possible pairs inevitably produces some inconsistencies — in the raw
released data, 12 concept pairs are annotated as prerequisites in BOTH
directions (A->B AND B->A), which cannot both be true in a DAG. This
script drops both directions for any such conflicting pair (rather than
guessing which direction is "right" — that would be fabricating an
annotation the original authors didn't make) and then verifies the result
is a valid DAG. As run during development, this fully resolved the
conflicts with zero further cycles — logged below so you can verify it
yourself if the upstream data changes.

Usage:
    python build_knowledge_graph.py
Reads:
    data/lecturebank_208topics.csv               (ID, Topic, Wiki_URL)
    data/lecturebank_prereq_annotation.csv        (Source_ID, Target_ID, Is_Prerequisite)
Writes:
    data/knowledge_graph_lecturebank.json          (208 concepts, cleaned DAG)

NOTE: this graph has NO quiz questions and NO learning resources attached
yet — see the "Action Items" in RESEARCH_NOTE.md and NOVELTY.md. It gives
you the validated concept/prerequisite structure; wiring diagnostic
questions and resources per concept (at 208-concept scale) is real,
unscoped work you still need to do — a good candidate to split across a
team, or to reduce by shipping only a connected sub-graph relevant to your
chosen target domains (e.g. everything reachable from "Neural Networks").
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import networkx as nx

DATA_DIR = Path(__file__).parent.parent / "data"
TOPICS_FILE = DATA_DIR / "lecturebank_208topics.csv"
PREREQ_FILE = DATA_DIR / "lecturebank_prereq_annotation.csv"
OUTPUT_FILE = DATA_DIR / "knowledge_graph_lecturebank.json"


def slugify(name: str) -> str:
    return (
        name.strip().lower()
        .replace(" ", "_").replace("-", "_").replace("/", "_")
        .replace("(", "").replace(")", "").replace(",", "").replace(":", "")
    )


def load_topics() -> dict[str, dict]:
    topics = {}
    with open(TOPICS_FILE, encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            tid, name = row[0].strip(), row[1].strip()
            wiki = row[2].strip() if len(row) > 2 else "NULL"
            topics[tid] = {"name": name, "wiki_url": None if wiki in ("NULL", "") else wiki}
    return topics


def load_edges(valid_ids: set[str]) -> list[tuple[str, str]]:
    edges = []
    with open(PREREQ_FILE, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) != 3:
                continue
            src, tgt, flag = (c.strip() for c in row)
            if flag == "1" and src in valid_ids and tgt in valid_ids:
                edges.append((src, tgt))
    return edges


def clean_to_dag(edges: list[tuple[str, str]], node_ids: set[str]) -> nx.DiGraph:
    edge_set = set(edges)
    conflicts = {(s, t) for s, t in edges if (t, s) in edge_set}
    clean = [e for e in edges if e not in conflicts]
    print(f"Dropped {len(conflicts)} directed edges from "
          f"{len(conflicts) // 2} bidirectional-conflict pairs.")

    G = nx.DiGraph()
    G.add_nodes_from(node_ids)
    G.add_edges_from(clean)

    removed = 0
    while not nx.is_directed_acyclic_graph(G):
        cycle = nx.find_cycle(G)
        G.remove_edge(*cycle[0][:2])
        removed += 1
    if removed:
        print(f"Removed {removed} additional edge(s) to break remaining cycles "
              "(longer cycles beyond simple bidirectional pairs).")
    else:
        print("No further cycles found after removing bidirectional conflicts.")
    return G


def main():
    topics = load_topics()
    edges = load_edges(set(topics.keys()))
    print(f"Loaded {len(topics)} concepts and {len(edges)} positive prerequisite edges.")

    G = clean_to_dag(edges, set(topics.keys()))
    assert nx.is_directed_acyclic_graph(G), "Result must be a DAG — this should never fail."

    id_to_slug = {tid: f"{slugify(t['name'])}__{tid}" for tid, t in topics.items()}

    concepts_out = []
    for tid, t in topics.items():
        concepts_out.append({
            "id": id_to_slug[tid],
            "name": t["name"],
            "wiki_url": t["wiki_url"],
            "prerequisites": [id_to_slug[p] for p in G.predecessors(tid)],
        })

    out = {
        "domain": "cs_nlp_ml_ai_dl_ir__lecturebank",
        "source": {
            "dataset": "LectureBank",
            "citation": "Li, Fabbri, Tung, Radev (2019). What Should I Learn First: "
                         "Introducing LectureBank for NLP Education and Prerequisite "
                         "Chain Learning. AAAI 2019.",
            "url": "https://github.com/Yale-LILY/LectureBank",
            "license_note": "Check the repository for current license/usage terms "
                             "before shipping this in a public app; not verified here.",
        },
        "stats": {
            "concepts": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "longest_prerequisite_chain": nx.dag_longest_path_length(G),
        },
        "quiz": [],  # NOT populated — see module docstring / Action Items
        "concepts": concepts_out,
    }

    OUTPUT_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {OUTPUT_FILE} — {out['stats']}")
    print("Example longest chain:",
          " -> ".join(topics[n]["name"] for n in nx.dag_longest_path(G)))


if __name__ == "__main__":
    main()
