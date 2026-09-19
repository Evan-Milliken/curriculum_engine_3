"""
PC-BwK-style baseline policy: exactly what the real system's knowledge-
gap + hard-prerequisite-filter logic reduces to at this environment's
level of abstraction — always study the topologically-earliest unmastered
concept whose prerequisites are already satisfied. No learning, no
exploration, zero training time; the "policy" is just the knowledge graph
plus the hard constraint (knowledge_graph.py: knowledge_gap() +
prerequisites_satisfied(), the same functions main.py actually calls).
"""
from __future__ import annotations
import time
from environment import CurriculumEnv


def pcbwk_policy_action(env: CurriculumEnv) -> int:
    """Mirrors main.py's real logic: gaps = topologically-ordered
    ancestors of the goal below threshold; pick the first one whose
    prerequisites are satisfied."""
    gaps = env.kg.knowledge_gap(env.mastery, env.goal_concept, 0.6)
    for concept in gaps:
        if env.kg.prerequisites_satisfied(concept, env.mastery, 0.6):
            return env.idx[concept]
    # No legal action (shouldn't happen on a connected DAG with a
    # reachable goal) — fall back to the first gap concept.
    return env.idx[gaps[0]] if gaps else 0


def evaluate_pcbwk(env: CurriculumEnv, trials: int = 200, seed: int = 1) -> dict:
    rng_env = CurriculumEnv(domain=env.domain, goal_concept=env.goal_concept,
                             max_days=env.max_days, seed=seed)
    days_to_goal, successes, illegal_actions = [], 0, 0

    start = time.perf_counter()
    for _ in range(trials):
        rng_env.reset()
        done = False
        while not done:
            action = pcbwk_policy_action(rng_env)
            _, _, done, info = rng_env.step(action)
            if not info["prereq_ok"]:
                illegal_actions += 1  # should be structurally impossible — see assertion below
        days_to_goal.append(rng_env.day)
        if rng_env.mastery[rng_env.goal_concept] >= 0.6:
            successes += 1
    elapsed = time.perf_counter() - start

    assert illegal_actions == 0, (
        "PC-BwK policy took an illegal action — this would indicate a bug "
        "in the hard-constraint logic, not expected behavior."
    )
    return {
        "trials": trials,
        "success_rate": successes / trials,
        "mean_days_to_goal": sum(days_to_goal) / len(days_to_goal),
        "illegal_actions": illegal_actions,
        "decision_seconds_total": elapsed,
        "train_seconds": 0.0,  # the whole point: no training phase exists
    }


if __name__ == "__main__":
    env = CurriculumEnv(seed=42)
    result = evaluate_pcbwk(env)
    print(f"PC-BwK-style policy on '{env.domain}' (goal='{env.goal_concept}'):")
    for k, v in result.items():
        print(f"  {k} = {v}")
