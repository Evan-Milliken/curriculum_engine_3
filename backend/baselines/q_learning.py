"""
Tabular Q-learning baseline over the shared CurriculumEnv (environment.py).

This is deliberately the simplest possible RL baseline — tabular, not deep
— matching what RESEARCH_NOTE.md Action Item #1 asks for ("even a toy
Q-learning baseline"). The point is not to reproduce the sophisticated
architectures in the five reviewed papers (attention+RL, GraphRAG+RL,
DKT+RL — all deep-learning-based); it's to measure the *cost side* of
"RL vs. structural constraint" honestly: how much training/exploration
does even the simplest RL approach need to learn what PC-BwK gets for
free from the knowledge graph?
"""
from __future__ import annotations
import random
import time
from collections import defaultdict

from environment import CurriculumEnv


def train_q_learning(
    env: CurriculumEnv,
    episodes: int = 3000,
    alpha: float = 0.2,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    seed: int = 0,
) -> dict:
    rng = random.Random(seed)
    Q: dict[tuple, list[float]] = defaultdict(lambda: [0.0] * env.n)
    illegal_during_training = 0
    days_history = []

    start = time.perf_counter()
    for ep in range(episodes):
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * (ep / episodes)
        state = env.reset()
        done = False
        while not done:
            if rng.random() < epsilon:
                action = rng.randrange(env.n)
            else:
                action = max(range(env.n), key=lambda a: Q[state][a])
            next_state, reward, done, info = env.step(action)
            if not info["prereq_ok"]:
                illegal_during_training += 1
            best_next = max(Q[next_state]) if not done else 0.0
            Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])
            state = next_state
        days_history.append(env.day)
    train_seconds = time.perf_counter() - start

    return {
        "Q": Q,
        "train_seconds": train_seconds,
        "illegal_actions_during_training": illegal_during_training,
        "episodes": episodes,
        "days_history": days_history,
    }


def evaluate_policy(env: CurriculumEnv, Q: dict, trials: int = 200, seed: int = 1) -> dict:
    """Greedy rollout of the trained policy, on fresh stochastic episodes."""
    rng_env = CurriculumEnv(domain=env.domain, goal_concept=env.goal_concept,
                             max_days=env.max_days, seed=seed)
    days_to_goal, successes, illegal_at_eval = [], 0, 0
    for _ in range(trials):
        state = rng_env.reset()
        done = False
        while not done:
            action = max(range(rng_env.n), key=lambda a: Q[state][a]) if state in Q else rng_env.rng.randrange(rng_env.n)
            state, reward, done, info = rng_env.step(action)
            if not info["prereq_ok"]:
                illegal_at_eval += 1
        days_to_goal.append(rng_env.day)
        if rng_env.mastery[rng_env.goal_concept] >= 0.6:
            successes += 1
    return {
        "trials": trials,
        "success_rate": successes / trials,
        "mean_days_to_goal": sum(days_to_goal) / len(days_to_goal),
        "illegal_actions_at_eval": illegal_at_eval,
    }


if __name__ == "__main__":
    env = CurriculumEnv(seed=42)
    print(f"Training tabular Q-learning on '{env.domain}' domain "
          f"({env.n} concepts, goal='{env.goal_concept}')...")
    result = train_q_learning(env)
    print(f"  Trained for {result['episodes']} episodes in {result['train_seconds']:.2f}s")
    print(f"  Illegal (prerequisite-violating) actions taken DURING training: "
          f"{result['illegal_actions_during_training']}")

    eval_result = evaluate_policy(env, result["Q"])
    print(f"  Evaluation over {eval_result['trials']} fresh trials:")
    print(f"    success_rate = {eval_result['success_rate']:.2%}")
    print(f"    mean_days_to_goal = {eval_result['mean_days_to_goal']:.2f}")
    print(f"    illegal actions still taken at eval time = {eval_result['illegal_actions_at_eval']}")
