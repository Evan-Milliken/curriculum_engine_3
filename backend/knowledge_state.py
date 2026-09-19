"""
Knowledge-State Estimation — Methodology Section 2 & 10.

EMA update rule (as specified in the methodology doc):
    M_(t+1) = alpha * M_t + (1 - alpha) * P_t
where M_t is prior mastery, P_t is observed performance (0-1), alpha in [0,1]
controls how much weight is given to history vs. the new observation.

This module is intentionally NOT a full Bayesian Knowledge Tracing (BKT)
implementation — BKT requires per-concept guess/slip/transition parameters
fit from real interaction data, which this project does not yet have
(see RESEARCH_NOTE.md, Action Item #3). EMA is the defensible lightweight
substitute the methodology doc explicitly allows.
"""
from __future__ import annotations
from dataclasses import dataclass, field

DEFAULT_ALPHA = 0.6  # weight given to prior mastery; tune once real data exists


@dataclass
class KnowledgeState:
    mastery: dict[str, float] = field(default_factory=dict)
    alpha: float = DEFAULT_ALPHA
    # (day_index, concept, mastery_after) log — purely for the analytics
    # dashboard (Section 11 / explainability); never read by the scoring
    # or optimization logic itself.
    history: list[tuple[int, str, float]] = field(default_factory=list)

    def get(self, concept: str) -> float:
        return self.mastery.get(concept, 0.0)

    def update(self, concept: str, performance: float, day: int = 0) -> float:
        """Apply the EMA update rule for a single concept observation."""
        performance = max(0.0, min(1.0, performance))
        prior = self.mastery.get(concept, 0.0)
        new_val = self.alpha * prior + (1 - self.alpha) * performance
        self.mastery[concept] = round(new_val, 4)
        self.history.append((day, concept, self.mastery[concept]))
        return self.mastery[concept]

    def apply_diagnostic(self, quiz_items: list[dict], answers: dict[str, str]) -> None:
        """
        Score a diagnostic quiz. Each quiz item nudges the mastery of its
        associated concept toward 1.0 (correct) or 0.0 (incorrect) via EMA,
        starting from a neutral prior of 0.0 so the diagnostic reflects
        demonstrated knowledge only (Methodology Section 2).
        """
        for item in quiz_items:
            qid = item["id"]
            concept = item["concept"]
            if qid not in answers:
                continue
            correct = answers[qid] == item["answer"]
            self.update(concept, 1.0 if correct else 0.0)

    def apply_self_report(self, concept: str, familiarity_0_to_5: int) -> float:
        """
        Fallback knowledge-state seeding for domains with NO diagnostic
        quiz authored yet (e.g. the 208-concept LectureBank graph — writing
        and validating real quiz questions for 208 topics is unscoped,
        real work; see DATASETS.md Action Items). This is a self-reported
        Likert-style familiarity rating (0 = never heard of it, 5 = could
        teach it), linearly mapped to an initial mastery estimate.

        This is explicitly weaker evidence than a graded diagnostic
        question — self-report is well known to be noisier than tested
        performance — so treat any concept seeded this way as lower-
        confidence than one seeded via `apply_diagnostic`. It is offered
        as an honest, clearly-labeled placeholder, not presented to the
        learner as equivalent to a real diagnostic.
        """
        familiarity_0_to_5 = max(0, min(5, familiarity_0_to_5))
        estimate = familiarity_0_to_5 / 5.0
        return self.update(concept, estimate)

    def replanning_action(self, concept: str) -> str:
        """
        Adaptive re-planning rule — Methodology Section 10:
            M > 0.80            -> "progress"
            0.50 <= M <= 0.80    -> "reinforce"
            M < 0.50             -> "introduce_prerequisite"
        """
        m = self.get(concept)
        if m > 0.80:
            return "progress"
        elif m >= 0.50:
            return "reinforce"
        else:
            return "introduce_prerequisite"

    # ---- persistence (see db.py) ----
    def to_dict(self) -> dict:
        return {"mastery": self.mastery, "alpha": self.alpha, "history": self.history}

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeState":
        state = cls(mastery=dict(d.get("mastery", {})), alpha=d.get("alpha", DEFAULT_ALPHA))
        state.history = [tuple(h) for h in d.get("history", [])]
        return state
