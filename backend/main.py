"""
Context-Aware Adaptive Curriculum Engine — API layer.

v0.3: state persistence. Everything that used to live in module-level
dicts (and therefore vanished on every server restart — flagged by
external review as making the PC-BwK bandit "a proof-of-concept
illusion") now lives in SQLite via db.py. Restart the server, and a
learner's mastery, bandit weights, and history are all still there.

Endpoints implement the full pipeline from the methodology doc, across
TWO selectable knowledge-graph domains (see DATASETS.md):
  - "demo_ml":        10 hand-built concepts, has a real MCQ diagnostic quiz.
  - "cs_lecturebank":  208 real, published concepts (Li et al., AAAI 2019),
                       no authored quiz yet — uses self-reported familiarity
                       instead (honestly labelled as lower-confidence).

  POST /learners                       -> Section 1 (profiling)
  GET  /domains                        -> list available knowledge-graph domains
  GET  /graph?domain=...               -> domain-level graph (no learner needed)
  GET  /quiz?domain=...                -> Section 2 (diagnostic quiz, demo_ml only)
  POST /quiz/submit                    -> Section 2 (knowledge-state estimation)
  POST /learners/{id}/self_assess      -> Section 2 fallback for domains with no quiz
  GET  /learners/{id}/gaps             -> Section 3
  GET  /learners/{id}/graph            -> knowledge graph + mastery, for visualization
  GET  /learners/{id}/plan             -> Sections 4-8 end to end
  POST /learners/{id}/complete         -> Section 9 (+ PC-BwK bandit reward)
  GET  /learners/{id}/replan           -> Section 10
  GET  /learners/{id}/personalization  -> bandit explainability (Section 11)
  GET  /learners/{id}/analytics        -> full instrumentation for a research dashboard

Run locally:
    uvicorn main:app --reload --port 8000
Then open http://localhost:8000/docs for interactive Swagger UI.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import LearnerProfile, QuizSubmission, Resource, DailyPlan, PlanItem
from knowledge_graph import get_knowledge_graph, list_domains
from scoring import score_resources
from optimizer import group_knapsack
from bandit import ARM_NAMES
import db

app = FastAPI(title="Context-Aware Adaptive Curriculum Engine", version="0.3.0")
db.init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).parent / "data"


def _load_resources(filename: str) -> list[Resource]:
    path = DATA_DIR / filename
    if not path.exists():
        return []
    with open(path) as f:
        return [Resource(**r) for r in json.load(f)]


ALL_RESOURCES: list[Resource] = (
    _load_resources("resources.json") + _load_resources("resources_lecturebank.json")
)
MASTERY_THRESHOLD = 0.6


def _require_learner(learner_id: str) -> LearnerProfile:
    profile = db.get_learner(learner_id)
    if profile is None:
        raise HTTPException(404, "Unknown learner_id")
    return profile


def _resources_for_concept(concept_id: str) -> list[Resource]:
    return [r for r in ALL_RESOURCES if r.concept == concept_id]


@app.get("/domains")
def get_domains():
    return list_domains()


@app.get("/graph")
def get_domain_graph(domain: str = "demo_ml"):
    kg = get_knowledge_graph(domain)
    return kg.to_viz_json()


@app.post("/learners", response_model=LearnerProfile)
def create_learner(profile: LearnerProfile):
    kg = get_knowledge_graph(profile.domain)
    if profile.goal_concept not in kg.concept_ids():
        raise HTTPException(
            400,
            f"Unknown goal_concept for domain '{profile.domain}'. "
            f"Call GET /graph?domain={profile.domain} to see valid ids.",
        )
    db.create_learner(profile)
    return profile


@app.get("/quiz")
def get_quiz(domain: str = "demo_ml"):
    kg = get_knowledge_graph(domain)
    return [
        {"id": q["id"], "concept": q["concept"], "prompt": q["prompt"], "options": q["options"]}
        for q in kg.quiz
    ]


@app.post("/quiz/submit")
def submit_quiz(submission: QuizSubmission):
    profile = _require_learner(submission.learner_id)
    kg = get_knowledge_graph(profile.domain)
    state = db.get_state(submission.learner_id)
    answer_map = {a.question_id: a.selected_option for a in submission.answers}
    state.apply_diagnostic(kg.quiz, answer_map)
    db.save_state(submission.learner_id, state)
    return {"learner_id": submission.learner_id, "mastery": state.mastery}


@app.post("/learners/{learner_id}/self_assess")
def self_assess(learner_id: str, concept: str, familiarity: int):
    profile = _require_learner(learner_id)
    kg = get_knowledge_graph(profile.domain)
    if concept not in kg.concept_ids():
        raise HTTPException(400, f"Unknown concept '{concept}' for this learner's domain")
    state = db.get_state(learner_id)
    new_mastery = state.apply_self_report(concept, familiarity)
    db.save_state(learner_id, state)
    return {"concept": concept, "new_mastery": new_mastery, "confidence": "self_reported"}


@app.get("/learners/{learner_id}/gaps")
def get_gaps(learner_id: str):
    profile = _require_learner(learner_id)
    kg = get_knowledge_graph(profile.domain)
    state = db.get_state(learner_id)
    gaps = kg.knowledge_gap(state.mastery, profile.goal_concept, MASTERY_THRESHOLD)
    return {
        "goal": profile.goal_concept,
        "gaps": [{"id": c, "name": kg.name_of(c), "mastery": state.get(c)} for c in gaps],
    }


@app.get("/learners/{learner_id}/graph")
def get_learner_graph(learner_id: str):
    profile = _require_learner(learner_id)
    kg = get_knowledge_graph(profile.domain)
    state = db.get_state(learner_id)
    viz = kg.to_viz_json(state.mastery)
    viz["goal_concept"] = profile.goal_concept
    viz["mastery_threshold"] = MASTERY_THRESHOLD
    return viz


@app.get("/learners/{learner_id}/plan", response_model=DailyPlan)
def get_daily_plan(learner_id: str):
    profile = _require_learner(learner_id)
    kg = get_knowledge_graph(profile.domain)
    state = db.get_state(learner_id)
    bandit = db.get_bandit(learner_id)
    elapsed_days = db.get_elapsed_days(learner_id)

    gaps = kg.knowledge_gap(state.mastery, profile.goal_concept, MASTERY_THRESHOLD)
    if not gaps:
        return DailyPlan(learner_id=learner_id, day=elapsed_days, total_minutes=0,
                          items=[], remaining_minutes=profile.available_minutes_per_day)

    context = bandit.build_context(
        current_mastery=sum(state.mastery.values()) / max(len(state.mastery), 1),
        days_remaining=max(profile.deadline_days - elapsed_days, 0),
        deadline_days=profile.deadline_days,
        minutes_available=profile.available_minutes_per_day,
    )
    chosen_arm, weights = bandit.select_weights(context)
    db.append_weight_log(learner_id, elapsed_days, chosen_arm, weights)

    groups = []
    for concept_id in gaps:
        candidates = _resources_for_concept(concept_id)
        if not candidates:
            continue
        prereq_ok = kg.prerequisites_satisfied(concept_id, state.mastery, MASTERY_THRESHOLD)
        if not prereq_ok:
            continue
        scored = score_resources(
            resources=candidates,
            gap_concept=concept_id,
            concept_display_name=kg.name_of(concept_id),
            available_minutes=profile.available_minutes_per_day,
            prereq_ok=True,
            preferred_types=profile.preferred_types,
            preferred_sources=profile.preferred_sources,
            weights=weights,
        )
        groups.append(scored[:3])

    selected = group_knapsack(groups, profile.available_minutes_per_day)

    items = []
    for sel in selected:
        r: Resource = sel["resource"]
        reason = (
            f"Your diagnostic shows {kg.name_of(r.concept)} mastery of "
            f"{state.get(r.concept):.2f} (below the {MASTERY_THRESHOLD} threshold), "
            f"and it sits on the prerequisite path to {kg.name_of(profile.goal_concept)}. "
            f"Recommended using your '{chosen_arm.replace('_', ' ')}' personalization profile."
        )
        items.append(PlanItem(resource=r, reason=reason))
        db.set_last_plan_choice(learner_id, r.id, chosen_arm, context.tolist())

    total = sum(i.resource.duration_min for i in items)
    return DailyPlan(
        learner_id=learner_id, day=elapsed_days, items=items,
        total_minutes=total, remaining_minutes=profile.available_minutes_per_day - total,
    )


@app.post("/learners/{learner_id}/complete")
def complete_resource(learner_id: str, resource_id: str, performance: float):
    _require_learner(learner_id)
    state = db.get_state(learner_id)
    bandit = db.get_bandit(learner_id)
    resource = next((r for r in ALL_RESOURCES if r.id == resource_id), None)
    if resource is None:
        raise HTTPException(404, "Unknown resource_id")

    mastery_before = state.get(resource.concept)
    day = db.get_elapsed_days(learner_id)
    new_mastery = state.update(resource.concept, performance, day=day)
    mastery_gain = max(0.0, new_mastery - mastery_before)
    db.save_state(learner_id, state)
    db.set_elapsed_days(learner_id, day + 1)

    choice = db.pop_last_plan_choice(learner_id, resource_id)
    if choice is not None:
        bandit.update(choice["arm"], np.array(choice["context"]), mastery_gain)
        db.save_bandit(learner_id, bandit)

    return {"concept": resource.concept, "new_mastery": new_mastery, "mastery_gain": round(mastery_gain, 4)}


@app.get("/learners/{learner_id}/personalization")
def get_personalization_summary(learner_id: str):
    _require_learner(learner_id)
    bandit = db.get_bandit(learner_id)
    return bandit.summary()


@app.get("/learners/{learner_id}/replan")
def replan_signal(learner_id: str):
    profile = _require_learner(learner_id)
    kg = get_knowledge_graph(profile.domain)
    state = db.get_state(learner_id)
    gaps = kg.knowledge_gap(state.mastery, profile.goal_concept, MASTERY_THRESHOLD)
    return {c: state.replanning_action(c) for c in gaps} if gaps else {"status": "goal_reached"}


@app.get("/learners/{learner_id}/analytics")
def get_analytics(learner_id: str):
    _require_learner(learner_id)
    state = db.get_state(learner_id)
    bandit = db.get_bandit(learner_id)
    return {
        "mastery_history": [
            {"day": d, "concept": c, "mastery": m} for d, c, m in state.history
        ],
        "weight_log": db.get_weight_log(learner_id),
        "bandit_summary": bandit.summary(),
        "arm_names": ARM_NAMES,
    }


@app.get("/")
def root():
    return {
        "name": "Context-Aware Adaptive Curriculum Engine",
        "docs": "/docs",
        "domains": list_domains(),
        "storage": str(db.DB_PATH),
    }
