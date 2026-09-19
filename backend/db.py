"""
Persistence layer — fixes the "bandit amnesia on restart" gap flagged in
external review (see the conversation this was added in response to).

Design choice, stated plainly: this is SQLite with each learner's state
stored as a JSON blob per table row, not a normalized relational schema.
That's a deliberate simplification appropriate for this project's scale
(single-server, course-project/demo traffic) — a real multi-server
deployment would want a proper schema and a real database (Postgres) with
row-level columns for anything you need to query across learners (e.g.
"average mastery gain across all learners this week" is awkward against
opaque JSON blobs). Treat that as the next persistence upgrade, not a
flaw in this one — see COMMERCIALIZATION.md Phase 2.

Every function here does a full read-modify-write per call rather than
caching in memory. For this project's scale that's the right trade:
correctness and simplicity over the write-throughput SQLite can't give
you anyway under concurrent access. Don't scale this past a few
concurrent users without moving to Postgres + connection pooling.
"""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from contextlib import contextmanager

from models import LearnerProfile
from knowledge_state import KnowledgeState
from bandit import LinUCBWeightBandit

DB_PATH = Path(__file__).parent / "data" / "curriculum_engine.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS learners (
    learner_id TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    elapsed_days INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS knowledge_state (
    learner_id TEXT PRIMARY KEY REFERENCES learners(learner_id),
    state_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bandit_state (
    learner_id TEXT PRIMARY KEY REFERENCES learners(learner_id),
    bandit_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS weight_log (
    learner_id TEXT NOT NULL REFERENCES learners(learner_id),
    day INTEGER NOT NULL,
    arm TEXT NOT NULL,
    weights_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS last_plan_choice (
    learner_id TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    arm TEXT NOT NULL,
    context_json TEXT NOT NULL,
    PRIMARY KEY (learner_id, resource_id)
);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as conn:
        conn.executescript(SCHEMA)


# ---------------- learners ----------------

def create_learner(profile: LearnerProfile) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO learners (learner_id, profile_json, elapsed_days) VALUES (?, ?, 0)",
            (profile.learner_id, profile.model_dump_json()),
        )
        conn.execute(
            "INSERT OR REPLACE INTO knowledge_state (learner_id, state_json) VALUES (?, ?)",
            (profile.learner_id, json.dumps(KnowledgeState().to_dict())),
        )
        conn.execute(
            "INSERT OR REPLACE INTO bandit_state (learner_id, bandit_json) VALUES (?, ?)",
            (profile.learner_id, json.dumps(LinUCBWeightBandit().to_dict())),
        )


def get_learner(learner_id: str) -> LearnerProfile | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT profile_json FROM learners WHERE learner_id = ?", (learner_id,)
        ).fetchone()
    return LearnerProfile.model_validate_json(row[0]) if row else None


def get_elapsed_days(learner_id: str) -> int:
    with _conn() as conn:
        row = conn.execute(
            "SELECT elapsed_days FROM learners WHERE learner_id = ?", (learner_id,)
        ).fetchone()
    return row[0] if row else 0


def set_elapsed_days(learner_id: str, day: int) -> None:
    with _conn() as conn:
        conn.execute("UPDATE learners SET elapsed_days = ? WHERE learner_id = ?", (day, learner_id))


# ---------------- knowledge state ----------------

def get_state(learner_id: str) -> KnowledgeState | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT state_json FROM knowledge_state WHERE learner_id = ?", (learner_id,)
        ).fetchone()
    return KnowledgeState.from_dict(json.loads(row[0])) if row else None


def save_state(learner_id: str, state: KnowledgeState) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE knowledge_state SET state_json = ? WHERE learner_id = ?",
            (json.dumps(state.to_dict()), learner_id),
        )


# ---------------- bandit ----------------

def get_bandit(learner_id: str) -> LinUCBWeightBandit | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT bandit_json FROM bandit_state WHERE learner_id = ?", (learner_id,)
        ).fetchone()
    return LinUCBWeightBandit.from_dict(json.loads(row[0])) if row else None


def save_bandit(learner_id: str, bandit: LinUCBWeightBandit) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE bandit_state SET bandit_json = ? WHERE learner_id = ?",
            (json.dumps(bandit.to_dict()), learner_id),
        )


# ---------------- weight log (append-only, for the analytics dashboard) ----------------

def append_weight_log(learner_id: str, day: int, arm: str, weights: dict) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO weight_log (learner_id, day, arm, weights_json) VALUES (?, ?, ?, ?)",
            (learner_id, day, arm, json.dumps(weights)),
        )


def get_weight_log(learner_id: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT day, arm, weights_json FROM weight_log WHERE learner_id = ? ORDER BY rowid",
            (learner_id,),
        ).fetchall()
    return [{"day": d, "arm": a, "weights": json.loads(w)} for d, a, w in rows]


# ---------------- last plan choice (short-lived, for crediting bandit reward) ----------------

def set_last_plan_choice(learner_id: str, resource_id: str, arm: str, context: list[float]) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO last_plan_choice (learner_id, resource_id, arm, context_json) "
            "VALUES (?, ?, ?, ?)",
            (learner_id, resource_id, arm, json.dumps(context)),
        )


def pop_last_plan_choice(learner_id: str, resource_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT arm, context_json FROM last_plan_choice WHERE learner_id = ? AND resource_id = ?",
            (learner_id, resource_id),
        ).fetchone()
        if row:
            conn.execute(
                "DELETE FROM last_plan_choice WHERE learner_id = ? AND resource_id = ?",
                (learner_id, resource_id),
            )
    return {"arm": row[0], "context": json.loads(row[1])} if row else None
