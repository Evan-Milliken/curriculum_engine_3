"""
Prerequisite-Constrained Bandits with Knapsacks (PC-BwK) — the actual
research contribution of this project. See NOVELTY.md for the full
positioning and citations; summary here:

Badanidiyuru, Kleinberg & Slivkins (2013, "Bandits with Knapsacks",
FOCS/JACM) introduced BwK: a multi-armed bandit that must respect one or
more consumable resource budgets (e.g. a time or money budget) while
learning which arms pay off, with regret bounds relative to the best
*dynamic* policy. Follow-up work (Agrawal & Devanur 2016 "Linear
Contextual Bandits with Knapsacks"; Sankararaman & Slivkins 2018
"Combinatorial Semi-Bandits with Knapsacks", which generalizes the
constraint set to matroids) extended BwK to contextual and combinatorial
settings — but, as far as the search conducted for this project found,
never to a constraint defined by an arbitrary prerequisite DAG over the
arms themselves (a concept can't be "selected" until its prerequisite
concepts have been mastered), and never applied to aggregating
heterogeneous open educational resources. That combination — knapsack
(time) + DAG-feasibility (prerequisites) + contextual learning
(per-learner weight personalization) applied to open-resource curriculum
recommendation — is this project's actual novel contribution, and it is
an incremental, honestly-scoped one: a domain-specific constraint
extension of an established framework, not a new theoretical result.

WHAT THIS MODULE DOES CONCRETELY:
Instead of the fixed DEFAULT_WEIGHTS in scoring.py, each learner gets
their own LinUCB bandit over a small discrete "arm" space: which of the
five scoring criteria (relevance / quality / time_fit / prereq_fit /
preference) should dominate for THIS learner. The reward after each
resource is completed is the learner's mastery *gain* on that concept
(Section 9's post-activity signal). Over a handful of interactions the
bandit shifts weight toward whichever criterion has actually predicted
this learner's learning gains — e.g. some learners do better with
shorter/lower-difficulty items regardless of "relevance", others do
better following raw topical relevance. This is the online-personalization
loop; the knapsack optimizer (optimizer.py) still enforces the hard time
budget and the knowledge graph still enforces the hard prerequisite
constraint — the bandit only ever adjusts which already-feasible resource
looks best.
"""
from __future__ import annotations
import numpy as np

# Each "arm" is a full weight vector emphasizing one criterion. This is a
# small, interpretable discrete action space (5 arms) rather than a
# continuous one — appropriate for the very small number of interactions
# a single learner will generate in a course-timeline deployment; a
# continuous LinUCB over weight-space would be more powerful but needs
# far more interaction data per learner to converge than this project
# will realistically collect (Action Item — see NOVELTY.md).
ARMS = {
    "relevance_heavy":  np.array([0.6, 0.1, 0.1, 0.1, 0.1]),
    "quality_heavy":    np.array([0.1, 0.6, 0.1, 0.1, 0.1]),
    "time_fit_heavy":   np.array([0.1, 0.1, 0.6, 0.1, 0.1]),
    "prereq_fit_heavy": np.array([0.1, 0.1, 0.1, 0.6, 0.1]),
    "balanced":         np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
}
ARM_NAMES = list(ARMS.keys())
CRITERIA_ORDER = ["relevance", "quality", "time_fit", "prereq_fit", "preference"]
CONTEXT_DIM = 4  # [current_mastery, deadline_pressure, minutes_available_norm, bias]


class LinUCBWeightBandit:
    """
    One LinUCB instance per learner. Context = a small feature vector
    describing the learner's current situation; arms = which scoring
    criterion to emphasize; reward = observed mastery gain in [0,1]
    after the chosen weighting produced a recommendation that was
    completed (Section 9).

    Reference: Li, Chu, Langford, Schapire (2010), "A Contextual-Bandit
    Approach to Personalized News Article Recommendation" — the standard
    LinUCB formulation, applied here to a different action space (scoring
    weight vectors) and a different domain (education) than the original.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.A = {arm: np.identity(CONTEXT_DIM) for arm in ARM_NAMES}
        self.b = {arm: np.zeros(CONTEXT_DIM) for arm in ARM_NAMES}
        self.pulls = {arm: 0 for arm in ARM_NAMES}
        self.history: list[dict] = []  # for transparency / debugging in the UI

    def _theta(self, arm: str) -> np.ndarray:
        return np.linalg.inv(self.A[arm]) @ self.b[arm]

    def select_weights(self, context: np.ndarray) -> tuple[str, dict[str, float]]:
        """Returns (chosen_arm_name, weight_dict) via the LinUCB upper
        confidence bound rule."""
        best_arm, best_score = None, -np.inf
        for arm in ARM_NAMES:
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            mean = float(theta @ context)
            ucb = self.alpha * float(np.sqrt(context @ A_inv @ context))
            score = mean + ucb
            if score > best_score:
                best_score, best_arm = score, arm
        weights = dict(zip(CRITERIA_ORDER, ARMS[best_arm].tolist()))
        return best_arm, weights

    def update(self, arm: str, context: np.ndarray, reward: float) -> None:
        reward = float(max(0.0, min(1.0, reward)))
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        self.pulls[arm] += 1
        self.history.append({"arm": arm, "reward": reward})

    def build_context(self, current_mastery: float, days_remaining: int,
                       deadline_days: int, minutes_available: int) -> np.ndarray:
        deadline_pressure = 1.0 - min(1.0, days_remaining / max(deadline_days, 1))
        minutes_norm = min(1.0, minutes_available / 120.0)
        return np.array([current_mastery, deadline_pressure, minutes_norm, 1.0])

    def summary(self) -> dict:
        """Human-readable state for the /explain endpoint / UI — this is
        what makes the bandit's personalization explainable rather than a
        black box, tying back into Section 11 of the methodology."""
        return {
            "pulls_per_arm": self.pulls,
            "estimated_best_arm": max(self.pulls, key=self.pulls.get) if any(self.pulls.values()) else None,
            "recent_rewards": self.history[-10:],
        }

    # ---- persistence (see db.py) ----
    # numpy arrays aren't JSON-serializable directly, so A/b are converted
    # to/from plain nested lists. This is the fix for the "bandit amnesia
    # on restart" gap: db.py calls these on every write/read instead of
    # keeping the bandit only in a process-memory dict.
    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "A": {arm: self.A[arm].tolist() for arm in ARM_NAMES},
            "b": {arm: self.b[arm].tolist() for arm in ARM_NAMES},
            "pulls": self.pulls,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LinUCBWeightBandit":
        bandit = cls(alpha=d.get("alpha", 1.0))
        for arm in ARM_NAMES:
            if arm in d.get("A", {}):
                bandit.A[arm] = np.array(d["A"][arm])
            if arm in d.get("b", {}):
                bandit.b[arm] = np.array(d["b"][arm])
        bandit.pulls = dict(d.get("pulls", {arm: 0 for arm in ARM_NAMES}))
        bandit.history = list(d.get("history", []))
        return bandit
