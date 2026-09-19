"""
Resource Relevance and Quality Scoring — Methodology Section 6.

Score(r) = w1*Rel(r) + w2*Qual(r) + w3*TimeFit(r) + w4*PrereqFit(r) + w5*Pref(r)

Implementation notes / honest scope limits:
- Rel(r): the methodology doc suggests sentence embeddings + cosine similarity.
  Full sentence-transformer models (e.g. all-MiniLM-L6-v2) are a ~90MB download
  and add real latency/memory cost on a free-tier server. This module uses
  TF-IDF + cosine similarity as the default (scikit-learn, no network/model
  download required) and isolates it behind `relevance_fn` so swapping in
  sentence-transformers later is a one-line change. Flagged as Action Item #4.
- Qual(r): this MVP uses the static `quality` field seeded in resources.json
  (manually estimated 0-1). A real system needs an actual quality signal
  (view counts, ratings, completion rates) — flagged as Action Item #2.
- PrereqFit(r): 1.0 if the resource's own difficulty is achievable given the
  learner's current mastery of the resource's concept's prerequisites, else 0.
"""
from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_WEIGHTS = {
    "relevance": 0.35,
    "quality": 0.2,
    "time_fit": 0.2,
    "prereq_fit": 0.15,
    "preference": 0.1,
}


def relevance_fn(query_text: str, resource_titles: list[str]) -> list[float]:
    """TF-IDF cosine similarity between the learner's current-gap concept
    name (as a proxy query) and each candidate resource's title. Returns
    a similarity score in [0,1] per resource, aligned to resource_titles."""
    if not resource_titles:
        return []
    corpus = [query_text] + resource_titles
    vec = TfidfVectorizer(stop_words="english")
    try:
        tfidf = vec.fit_transform(corpus)
    except ValueError:
        # empty vocabulary (e.g. query/titles share no non-stopword tokens)
        return [0.5 for _ in resource_titles]
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    return [float(s) for s in sims]


def time_fit(duration_min: int, available_minutes: int) -> float:
    """1.0 if it fits comfortably, decaying to 0 as it exceeds the budget."""
    if available_minutes <= 0:
        return 0.0
    ratio = duration_min / available_minutes
    if ratio <= 1.0:
        return 1.0
    if ratio >= 2.0:
        return 0.0
    return max(0.0, 1.0 - (ratio - 1.0))


def preference_fit(resource, preferred_types: list[str], preferred_sources: list[str]) -> float:
    score = 0.5  # neutral baseline when learner gave no preferences
    hits, checks = 0, 0
    if preferred_types:
        checks += 1
        if resource.type in preferred_types:
            hits += 1
    if preferred_sources:
        checks += 1
        if resource.source in preferred_sources:
            hits += 1
    if checks == 0:
        return score
    return hits / checks


def score_resources(
    resources: list,
    gap_concept: str,
    concept_display_name: str,
    available_minutes: int,
    prereq_ok: bool,
    preferred_types: list[str],
    preferred_sources: list[str],
    weights: dict[str, float] = DEFAULT_WEIGHTS,
) -> list[dict]:
    """Score a candidate list of Resource objects already filtered to
    gap_concept. Returns list of dicts with per-criterion breakdown."""
    titles = [r.title for r in resources]
    rels = relevance_fn(concept_display_name, titles)
    out = []
    for r, rel in zip(resources, rels):
        tf = time_fit(r.duration_min, available_minutes)
        pf = 1.0 if prereq_ok else 0.0
        pref = preference_fit(r, preferred_types, preferred_sources)
        total = (
            weights["relevance"] * rel
            + weights["quality"] * r.quality
            + weights["time_fit"] * tf
            + weights["prereq_fit"] * pf
            + weights["preference"] * pref
        )
        out.append({
            "resource": r,
            "score": round(total, 4),
            "relevance": round(rel, 4),
            "time_fit": round(tf, 4),
            "prereq_fit": pf,
            "preference_fit": round(pref, 4),
        })
    out.sort(key=lambda d: d["score"], reverse=True)
    return out
