# Context-Aware Adaptive Curriculum Engine

Two documents, that's it:

- **`GUIDE.md`** - everything from installing Python for the first time
  through publishing on Google Play, in order.
- **`RESEARCH.md`** - the novelty claim (Prerequisite-Constrained Bandits
  with Knapsacks), the literature comparison, the datasets used, and the
  real baseline experiment results. Read before writing your report.

## What changed since the last build

- UI overhaul: a winding "journey path" (Duolingo-style) replacing
  the plain list, a one-question-at-a-time diagnostic flow (Kinnu-style)
  instead of a long form, and a swipeable card stack for the daily plan
  with a completion celebration. Same backend, new frontend
  (`frontend/index.html`).
- Real app icons (`frontend/icon-192.png`, `icon-512.png`) - fixes
  the 404s you saw in the terminal.
- Persistence: every learner's mastery and bandit state now survives
  a server restart (SQLite via `backend/db.py`) - verified with a real
  process kill and restart, not just a graceful shutdown.
- A real baseline experiment: `backend/baselines/` runs tabular
  Q-learning against a PC-BwK-style policy on identical simulated
  conditions. Real, captured numbers are in `RESEARCH.md` Section 4.
- Docs consolidated from seven scattered files down to two.
  Commercialization content has been dropped per your last message -
  the focus is research + UI + shipping to Play Store.

## Quick start

```bash
cd backend
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
```bash
cd frontend
python3 -m http.server 5500
# open http://localhost:5500
```

Everything else - deployment, Play Store packaging, the research
write-up - is in `GUIDE.md` and `RESEARCH.md`.
