"""
Runs the Q-learning baseline and the PC-BwK-style policy on the identical
environment and prints a comparison table. Run this yourself and paste the
REAL output into your report — do not reuse numbers from this repo's
comments/docs verbatim, regenerate them, since results depend on the
random seed and episode count you choose and should be reproducible by
your professor.

Usage:
    cd backend/baselines
    python3 compare.py
"""
from __future__ import annotations
from environment import CurriculumEnv
from q_learning import train_q_learning, evaluate_policy
from pcbwk_baseline import evaluate_pcbwk


def run_condition(max_days: int, episodes: int = 3000):
    env = CurriculumEnv(goal_concept="neural_networks", seed=42, max_days=max_days)
    q_result = train_q_learning(env, episodes=episodes)
    q_eval = evaluate_policy(env, q_result["Q"], trials=200)
    pcbwk_eval = evaluate_pcbwk(CurriculumEnv(goal_concept="neural_networks", seed=42, max_days=max_days), trials=200)
    return q_result, q_eval, pcbwk_eval


def print_table(max_days, q_result, q_eval, pcbwk_eval):
    print(f"\n--- max_days = {max_days} ---")
    print(f"{'Metric':<45} {'Q-learning':>15} {'PC-BwK':>15}")
    print("-" * 77)
    rows = [
        ("Training time (seconds)", f"{q_result['train_seconds']:.3f}", f"{pcbwk_eval['train_seconds']:.3f}"),
        ("Training episodes needed", f"{q_result['episodes']}", "0"),
        ("Illegal actions DURING training/setup",
         f"{q_result['illegal_actions_during_training']}", "0 (structurally impossible)"),
        ("Illegal actions at evaluation time", f"{q_eval['illegal_actions_at_eval']}", f"{pcbwk_eval['illegal_actions']}"),
        ("Success rate (200 eval trials)", f"{q_eval['success_rate']:.1%}", f"{pcbwk_eval['success_rate']:.1%}"),
        ("Mean days to goal (200 eval trials)", f"{q_eval['mean_days_to_goal']:.2f}", f"{pcbwk_eval['mean_days_to_goal']:.2f}"),
    ]
    for label, a, b in rows:
        print(f"{label:<45} {a:>15} {b:>15}")


def main():
    print("Environment: domain='demo_ml', goal='neural_networks', 10 concepts\n")
    print("Running TWO conditions to separate 'can Q-learning learn the task at all'")
    print("from 'does it learn it fast enough to meet a realistic daily time budget':\n")

    print("=" * 77)
    print("CONDITION A: realistic tight budget (max_days=40)")
    print("=" * 77)
    q_result_a, q_eval_a, pcbwk_eval_a = run_condition(max_days=40)
    print_table(40, q_result_a, q_eval_a, pcbwk_eval_a)

    print("\n" + "=" * 77)
    print("CONDITION B: loose control budget (max_days=80) — sanity check that")
    print("Q-learning CAN reach the same optimum given enough room to explore")
    print("=" * 77)
    q_result_b, q_eval_b, pcbwk_eval_b = run_condition(max_days=80)
    print_table(80, q_result_b, q_eval_b, pcbwk_eval_b)

    print("\n" + "=" * 77)
    print("READ THIS BEFORE WRITING IT UP:")
    print("=" * 77)
    print("If Condition B shows both policies converging to near-identical")
    print("mean_days_to_goal with ~100% success, that CONFIRMS the Q-learning")
    print("implementation is correct (not a strawman) — it can learn the optimal")
    print("policy asymptotically. If Condition A then shows Q-learning's success")
    print("rate collapsing while PC-BwK's does not, that is a genuine, narrow,")
    print("defensible finding: under a REALISTIC tight time budget (the exact")
    print("constraint this project's thesis is about), an agent that must learn")
    print("the prerequisite structure through trial-and-error can miss the")
    print("deadline entirely, while a system that encodes that structure")
    print("directly does not pay that exploration cost. Report it exactly that")
    print("narrowly — this is NOT evidence that PC-BwK 'beats RL' in general.")


if __name__ == "__main__":
    main()
