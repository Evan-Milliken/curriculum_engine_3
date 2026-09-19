"""
Time-Constrained Learning Path Optimization — Methodology Section 7.

    max_P  sum_i Gain_i
    s.t.   sum_i Time_i <= T_available
           Prerequisites(P) = Satisfied

This is implemented as a GROUP KNAPSACK dynamic program:
  - Each knowledge-gap concept is a "group".
  - Each group offers its top-K scored resources as candidate items
    (already hard-filtered to prereq_fit == 1.0 upstream, satisfying the
    Prerequisites(P)=Satisfied constraint exactly, not just as a soft weight).
  - At most ONE resource is chosen per group (you don't need two resources
    for the same concept in one day's plan).
  - Value = the multi-criteria Score(r) from scoring.py (the "Gain" proxy).
  - Weight = duration_min.
  - Capacity = available_minutes_per_day.

This matches the methodology doc's explicit suggestion of "constrained
greedy algorithm, dynamic programming, or ILP" and is a legitimate,
cheap-to-run alternative to the RL-based approaches in the reviewed
literature (see RESEARCH_NOTE.md for why that trade-off is the actual
research contribution to defend, not "novelty" of the scoring formula).
"""
from __future__ import annotations


def group_knapsack(
    groups: list[list[dict]], capacity_minutes: int
) -> list[dict]:
    """
    groups: list of groups; each group is a list of scored-resource dicts
            (each dict must have ['resource'].duration_min and ['score']).
            A group may be empty (no eligible resource this round) — skipped.
    capacity_minutes: total time budget for the day.

    Returns the selected list of scored-resource dicts (at most one per
    group), maximizing total score subject to the time budget.
    """
    groups = [g for g in groups if g]  # drop empty groups
    n = len(groups)
    if n == 0 or capacity_minutes <= 0:
        return []

    # dp[i][c] = best total score achievable using the first i groups with
    # capacity c. choice[i][c] = which item index in group i-1 was chosen
    # (or None if the group was skipped entirely at this capacity).
    dp = [[0.0] * (capacity_minutes + 1) for _ in range(n + 1)]
    choice: list[list[int | None]] = [[None] * (capacity_minutes + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        group = groups[i - 1]
        for c in range(capacity_minutes + 1):
            best_val = dp[i - 1][c]  # option: take nothing from this group
            best_choice = None
            for idx, item in enumerate(group):
                dur = item["resource"].duration_min
                if dur <= c:
                    candidate = dp[i - 1][c - dur] + item["score"]
                    if candidate > best_val:
                        best_val = candidate
                        best_choice = idx
            dp[i][c] = best_val
            choice[i][c] = best_choice

    # backtrack
    selected = []
    c = capacity_minutes
    for i in range(n, 0, -1):
        idx = choice[i][c]
        if idx is not None:
            item = groups[i - 1][idx]
            selected.append(item)
            c -= item["resource"].duration_min
    selected.reverse()
    return selected
