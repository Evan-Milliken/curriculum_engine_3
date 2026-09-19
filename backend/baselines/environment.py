"""
Shared simulation environment for the PC-BwK vs. Q-learning comparison
(RESEARCH_NOTE.md Action Item #1 / NOVELTY.md Action Item mentioning a
baseline comparison).

WHY THIS IS SIMPLIFIED, STATED UP FRONT: the real system (main.py) reasons
in minutes and resource-level detail. A tabular Q-learning baseline needs a
small discrete state/action space to be tractable at all, so this
environment abstracts one day == one concept studied, and mastery moves in
a single EMA step toward a stochastic "performance" outcome rather than
modeling resource selection. This is a fair simplification PRECISELY
because it is applied identically to both agents being compared — neither
gets an advantage from the abstraction. It would NOT be fair to report
these numbers as claims about the full system's real-world performance;
they are claims about the *decision policy* (which concept next) in
isolation from resource-selection, which is exactly the comparison
RESEARCH_NOTE.md calls for.

State: a tuple of 10 booleans (mastered / not-mastered per concept, using
the SAME 0.6 threshold as main.py).
Action: index of the concept to study today.
Transition: EMA update (same formula and alpha as knowledge_state.py)
    with a stochastic performance draw — Beta(6,2) when the concept's
    prerequisites are satisfied (models a learner in-range for the
    material, mean ~0.75, some variance), Beta(2,6) when they are NOT
    (models struggling with unprepared material, mean ~0.25) — the
    environment does not hard-block illegal actions the way main.py's
    knapsack does; it lets any agent attempt any action and pays the
    consequence, which is the whole point of the comparison: PC-BwK
    enforces the constraint structurally, Q-learning has to learn to
    avoid it from experience.
Reward: +10 on reaching the goal concept's mastery threshold, -1 per day
    (step cost, encourages efficiency), no additional penalty beyond the
    natural cost of wasted, low-yield studying on unprepared material.
Episode ends: goal reached, or `max_days` elapsed (counted as failure).
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge_graph import get_knowledge_graph  # noqa: E402

MASTERY_THRESHOLD = 0.6
ALPHA = 0.6  # same EMA alpha as knowledge_state.py DEFAULT_ALPHA


@dataclass
class CurriculumEnv:
    domain: str = "demo_ml"
    goal_concept: str = "neural_networks"
    max_days: int = 40
    seed: int | None = None

    def __post_init__(self):
        self.kg = get_knowledge_graph(self.domain)
        self.concepts = self.kg.concept_ids()
        self.n = len(self.concepts)
        self.idx = {c: i for i, c in enumerate(self.concepts)}
        self.rng = random.Random(self.seed)
        self.reset()

    def reset(self) -> tuple:
        self.mastery = {c: 0.0 for c in self.concepts}
        self.day = 0
        self.illegal_actions_taken = 0
        return self._state()

    def _state(self) -> tuple:
        return tuple(1 if self.mastery[c] >= MASTERY_THRESHOLD else 0 for c in self.concepts)

    def legal_actions(self) -> list[int]:
        """Actions whose prerequisites are currently satisfied — used by
        the PC-BwK-style baseline, which (like the real system) only ever
        considers legal actions. The Q-learning agent is NOT restricted to
        this list during training, by design (see module docstring)."""
        legal = []
        for i, c in enumerate(self.concepts):
            if self.mastery[c] >= MASTERY_THRESHOLD:
                continue  # already mastered, no point re-studying
            if self.kg.prerequisites_satisfied(c, self.mastery, MASTERY_THRESHOLD):
                legal.append(i)
        return legal

    def step(self, action: int) -> tuple[tuple, float, bool, dict]:
        concept = self.concepts[action]
        prereq_ok = self.kg.prerequisites_satisfied(concept, self.mastery, MASTERY_THRESHOLD)
        if not prereq_ok:
            self.illegal_actions_taken += 1
            performance = self.rng.betavariate(2, 6)  # struggling on unprepared material
        else:
            performance = self.rng.betavariate(6, 2)  # in-range material

        prior = self.mastery[concept]
        self.mastery[concept] = round(ALPHA * prior + (1 - ALPHA) * performance, 4)
        self.day += 1

        goal_reached = self.mastery[self.goal_concept] >= MASTERY_THRESHOLD
        done = goal_reached or self.day >= self.max_days
        reward = 10.0 if goal_reached else -1.0

        return self._state(), reward, done, {"prereq_ok": prereq_ok, "day": self.day}
