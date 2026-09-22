# Context-Aware Adaptive Curriculum Engine

Two documents:

- **GUIDE.md**: everything from installing Python for the first time
  through publishing on Google Play, in order.
- **RESEARCH.md**: the novelty claim (Prerequisite-Constrained Bandits
  with Knapsacks), the hybrid recommender and local neural network, the
  literature comparison, the datasets used, and the real baseline
  experiment results. Read before writing your report.

## What changed in this build

- A real local AI model, no API key, ever. backend/ml/ trains and
  runs a genuine neural network (scikit-learn MLPRegressor, trained with
  backpropagation) that predicts expected mastery gain per resource. It
  auto-trains on first server start and never makes an external network
  call at inference time. This is the "Gain_i" term the original
  methodology always referenced but never defined. See RESEARCH.md
  Section 5 for exactly what it does and does not prove.
- Every element is clickable and goes somewhere real. Journey-path
  nodes and knowledge-map nodes open a detail view with that concept's
  mastery, prerequisites, and live AI-scored resources. The "Next up"
  banner jumps to today's plan. The streak, XP, and level chips open
  Analytics.
- Strict three-color palette: red, white, and purple only (with
  lighter and darker shades of each), audited hex by hex. No emoji, no
  stray accent colors.
- No em dashes anywhere in the interface or the documentation.

## Quick start

```bash
cd backend
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
First run trains the local neural network automatically (a few seconds,
one time). Then:
```bash
cd frontend
python3 -m http.server 5500
# open http://localhost:5500
```

Everything else, deployment, Play Store packaging, the research
write-up, is in GUIDE.md and RESEARCH.md.
