"""
Data models for the Context-Aware Adaptive Curriculum Engine.
These map directly to the methodology sections:
  LearnerProfile      -> Section 1 (Learner and Goal Profiling)
  QuizSubmission      -> Section 2 (Diagnostic Assessment)
  KnowledgeState       -> Section 2/3 (Knowledge-State Estimation / Gap ID)
  Resource             -> Section 5 (Online Resource Retrieval)
  ScoredResource       -> Section 6 (Relevance & Quality Scoring)
  PlanItem / DailyPlan -> Section 8 (Personalized Daily Learning Plan)
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class LearnerProfile(BaseModel):
    learner_id: str
    domain: str = Field(default="demo_ml", description="Knowledge-graph domain id — see GET /domains")
    goal_concept: str = Field(..., description="Target concept id, e.g. 'neural_networks'")
    available_minutes_per_day: int = Field(..., gt=0)
    deadline_days: int = Field(..., gt=0)
    preferred_types: list[str] = Field(default_factory=list, description="e.g. ['video','documentation']")
    preferred_sources: list[str] = Field(default_factory=list, description="e.g. ['khan_academy','official_docs']")


class QuizAnswer(BaseModel):
    question_id: str
    selected_option: str


class QuizSubmission(BaseModel):
    learner_id: str
    answers: list[QuizAnswer]


class ConceptMastery(BaseModel):
    concept: str
    mastery: float = Field(..., ge=0.0, le=1.0)


class Resource(BaseModel):
    id: str
    title: str
    concept: str
    url: str
    source: str
    type: str
    duration_min: int
    difficulty: float
    quality: float


class ScoredResource(Resource):
    score: float
    relevance: float
    time_fit: float
    prereq_fit: float
    preference_fit: float


class PlanItem(BaseModel):
    resource: Resource
    reason: str


class DailyPlan(BaseModel):
    learner_id: str
    day: int
    total_minutes: int
    items: list[PlanItem]
    remaining_minutes: int
