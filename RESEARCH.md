# RESEARCH.md: Novelty, Literature, Datasets, and Baseline Results

Everything about the research side of this project, in one file. Read
top to bottom before writing your report's contribution statement.

---

## 1. The Novelty Claim

Your teammate's original scoring-function-plus-knapsack design is a
legitimate engineering approach, but weighted multi-criteria scoring and
knapsack-based selection under a resource budget are both well-established
techniques predating the reviewed literature by years. Claiming that
alone as novel next to five 2024-2026 papers that all use RL or LLMs
would not survive scrutiny.

### What's actually novel, and why

**"Bandits with Knapsacks" (BwK)** is a real, established framework -
Badanidiyuru, Kleinberg, and Slivkins introduced it in 2013, combining
multi-armed bandit learning with one or more consumable resource-budget
constraints. It has since been extended to contextual settings (Agrawal &
Devanur, 2016, "Linear Contextual Bandits with Knapsacks") and to richer
constraint structures, notably Sankararaman & Slivkins' (2018)
combinatorial semi-bandits with knapsacks, which generalizes the
constraint set to matroids.

A search of the current literature did NOT turn up a version of BwK
where the feasibility constraint is a precedence/DAG constraint, an
arm (a learning resource) only becomes eligible once its prerequisite
concepts are mastered, applied to curating external, uncontrolled open
educational resources rather than a closed, platform-controlled item
bank. Prerequisite DAGs are common in the education-recommender
literature (all five reviewed papers use one); contextual bandits are
separately common there too (multiple papers apply LinUCB/Thompson
Sampling to next-item selection). The combination, BwK-style
optimization, DAG-structured feasibility instead of a scalar budget,
applied to open-resource curriculum assembly, is the specific, narrow
gap this project fills.

**The claim:**

> Prerequisite-Constrained Bandits with Knapsacks (PC-BwK): a
> contextual bandit that personalizes which scoring criteria to trust for
> a given learner, operating inside a time-budget knapsack, under a hard
> feasibility constraint defined by a knowledge-graph prerequisite DAG -
> applied to aggregating open, human-authored educational resources
> rather than a closed, platform-controlled item bank.

Implemented in `backend/bandit.py`, wired into `main.py`, state persisted
via `backend/db.py` (verified to survive a real process restart, see
Section 4).

**Suggested paragraph for your report:**

> "Bandits with Knapsacks (Badanidiyuru et al., 2013) is a well-established
> framework for online learning under resource-budget constraints, later
> extended to contextual and matroid-structured constraints (Agrawal &
> Devanur, 2016; Sankararaman & Slivkins, 2018). Separately, contextual
> bandits have been applied to next-item selection in educational
> recommender systems, and knapsack formulations have been applied to
> time-constrained recommendation outside education. We combine these
> three threads: a contextual bandit personalizes scoring-criterion
> weights for time-boxed, prerequisite-DAG-constrained curriculum
> assembly over open (not platform-controlled) educational resources, a
> constraint structure and resource setting not found in our literature
> review of either the BwK or educational-recommender literatures."

### Be precise about scope

This is an applied systems contribution, a domain-specific extension
combining two existing frameworks in a configuration the literature
search didn't find, NOT a new theoretical result with regret bounds.
Do not claim "first-ever" or "world's first." Do not claim "better than
RL" in a general sense, see Section 4 for what the real baseline
experiment does and does not show. Not validated on real learners, the
bandit converges correctly in simulation, but that is not the same claim
as "produces better outcomes for real people," which needs a user study
this project has not run.

### Sources (verify independently before citing)
- Badanidiyuru, A., Kleinberg, R., & Slivkins, A. (2013). Bandits with Knapsacks. FOCS 2013 / J. ACM. https://arxiv.org/abs/1305.2545
- Agrawal, S., & Devanur, N. R. (2016). Linear Contextual Bandits with Knapsacks. NeurIPS 2016.
- Sankararaman, K. A., & Slivkins, A. (2018). Combinatorial Semi-Bandits with Knapsacks. AISTATS 2018.
- Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A Contextual-Bandit Approach to Personalized News Article Recommendation. WWW 2010 (the LinUCB algorithm `bandit.py` implements).

---

## 2. Literature Comparison (the five uploaded papers)

| # | Paper | Venue | Core technique |
|---|---|---|---|
| 1 | Jiang, Wen, Shen, Peng, Kang, Liu, "Personalized Learning Path Recommendation with Time-Aware Attention-Based Reinforcement Learning" | ACM TIST, 2025 | Time-aware attention + Monte Carlo policy-gradient RL |
| 2 | Cheng et al., "GraphRAG-Induced Dual Knowledge Structure Graphs for Personalized Learning Path Recommendation" | AAAI, 2026 | LLM/GraphRAG-generated graphs + RL |
| 3 | Zhou & Wang, English learning path recommendation | Scientific Reports, 2025 | Knowledge graph + Q-learning (pruning) + PPO (selection) |
| 4 | Fu, RL-DKT | Scientific Reports, 2025 | Dynamic Knowledge Tracing fused with RL |
| 5 | Cristea, Kiden, Stein et al., TutorLLM | ACM RecSys, 2024 | Knowledge tracing + retrieval-augmented LLM |

The pattern: every one uses RL, deep sequence models, or LLM
generation as the core mechanism, over a closed item bank the research
team controls. None treats available daily time as a hard constraint on
the recommendation itself, and none aggregates genuinely external,
uncontrolled resources at inference time.

Two known, honest weaknesses in comparing against them directly -
address these explicitly in your report rather than letting a reviewer
find them first:
1. Apples-to-oranges risk. You are not solving the exact same
   mathematical problem as a deep sequence model. Frame your work as a
   resource allocation and scheduling policy under uncertainty, not as
   a competing sequence-prediction method.
2. The hard prerequisite boundary is a real design trade-off, not a
   free lunch. Real pedagogy sometimes uses top-down/scaffolded
   exploration (tackling a complex task first, backfilling foundations as
   needed), a rigid DAG filter can't do that. The defensible framing:
   this rigidity is intentional and appropriate for a system whose
   whole thesis is preventing wrong-order, analysis-paralysis learning
   under a tight time budget, recommending B before A's prerequisites
   are met is exactly the failure mode the project exists to avoid. State
   this trade-off explicitly rather than presenting the hard constraint
   as a universal, cost-free design choice.

---

## 3. Datasets

### LectureBank, integrated, verified

- Paper: Li, Fabbri, Tung, Radev (2019). What Should I Learn First:
  Introducing LectureBank for NLP Education and Prerequisite Chain
  Learning. AAAI 2019. https://arxiv.org/abs/1811.12181
- Data: https://github.com/Yale-LILY/LectureBank
- What it contains: 208 manually-curated NLP/ML/AI/Deep
  Learning/Information Retrieval concepts, with every pairwise
  combination annotated by the paper's authors for "is A a prerequisite
  of B?" (`prerequisite_annotation.csv`), plus a 208-topic name list with
  Wikipedia links (`208topics.csv`).
- What this project did: `backend/scripts/build_knowledge_graph.py`
  downloads and cleans this into a valid 208-node, 889-edge prerequisite
  DAG, 12 bidirectional annotation conflicts detected and dropped (both
  directions removed, not guessed at, since guessing which direction is
  "right" would be fabricating an annotation the original authors didn't
  make). Output: `backend/data/knowledge_graph_lecturebank.json`. This is
  wired into the live app as the `cs_lecturebank` domain, with knowledge
  seeded via self-report (no graded quiz exists for 208 topics yet) and
  one real Wikipedia-sourced resource per concept
  (`backend/scripts/build_wikipedia_resources.py` -> 185/208 concepts
  covered, 23 had no Wikipedia URL in the source data).
- Known simplification: the Wikipedia resource per concept is a
  single general-reference article, not curated video/practice content,
  and self-reported familiarity is honestly weaker evidence than a graded
  diagnostic. Upgrading either means real, unscoped work, state this as
  a scoping decision in your report, not a hidden gap.
- License note: the repository does not state a license as of this
  writing, verify current terms before shipping this publicly at scale.
  Nothing in this project's resource set uses the dataset's
  `alldata.tsv` lecture-slide URLs (some of which are copyrighted
  university material), only the dataset's own Wikipedia links are used
  for resources, which are unambiguously fine to link to.

### MOOCCubeX, available, not integrated

- Paper: Yu, J. et al. (2021). MOOCCubeX: A Large Knowledge-centered
  Repository for Adaptive Learning in MOOCs. CIKM 2021.
- Data: http://moocdata.cn/data/MOOCCubeX
- Primarily Chinese-language, multi-disciplinary (not CS-specific), with
  real behavioral logs (296M+ student interaction records). Listed here
  as a future option for validating the EMA mastery model against real
  learning curves, not planned for integration in the current scope.

### YouTube Data API, integrated (pipeline), not run live
`backend/ingestion.py` is a real, working ingestion script for pulling
live video metadata; requires your own free API key from
https://console.cloud.google.com/apis/credentials.

---

## 4. Baseline Results (real, executed)

Addresses the single most important open item from earlier review: a
real experiment, not just an architecture description.

### Setup
`backend/baselines/` implements a tabular Q-learning agent and a
PC-BwK-style deterministic policy (mirrors `main.py`'s actual
gap-detection + hard-prerequisite-filter logic) on an identical simulated
environment (`environment.py`): one day = one concept studied, mastery
updates via the same EMA formula as the real system, performance drawn
stochastically (better outcomes when prerequisites are satisfied, worse
when not).

### Real output (seed=42, 3000 training episodes, 200 eval trials, regenerate yourself with `python3 backend/baselines/compare.py`)

```
CONDITION A: realistic tight budget (max_days=40)
-----------------------------------------------------------------------------
Metric                                             Q-learning          PC-BwK
-----------------------------------------------------------------------------
Training time (seconds)                                 0.707           0.000
Training episodes needed                                 3000               0
Illegal actions DURING training/setup                   54155 0 (structurally impossible)
Success rate (200 eval trials)                           0.0%           98.5%
Mean days to goal (200 eval trials)                     40.00           33.85

CONDITION B: loose control budget (max_days=80)
-----------------------------------------------------------------------------
Training time (seconds)                                 1.047           0.000
Training episodes needed                                 3000               0
Illegal actions DURING training/setup                   66577 0 (structurally impossible)
Success rate (200 eval trials)                         100.0%          100.0%
Mean days to goal (200 eval trials)                     33.88           33.88
```

### What this shows

Condition B is the sanity check, and it passed. With a loose budget,
both policies converge to essentially identical performance, confirming
the Q-learning implementation is a genuine, correctly-functioning
baseline, not a strawman. It can learn the optimal policy given room to
explore.

Condition A is the finding. Under a realistic 40-day budget (close to
the ~34-day structural minimum), Q-learning's success rate collapses to
0% while PC-BwK holds 98.5%. The Q-learning agent took 54,155
prerequisite-violating actions across training while discovering the
graph structure through trial and error, that exploration cost, paid in
wasted simulated days, is exactly what a structurally-encoded constraint
avoids.

### The narrow, correct claim

> Under a realistic time-constrained deadline, an RL agent that must
> learn a domain's prerequisite structure through exploration can fail to
> meet that deadline even when it is capable of learning the optimal
> policy given enough time. A system that encodes the prerequisite
> structure directly (PC-BwK) avoids paying this exploration cost and
> meets the deadline reliably.

### What this does NOT show, do not claim these
- NOT "PC-BwK produces better learning outcomes than RL", this
  environment simulates mastery via a stochastic performance draw, not
  real human learning. It validates a decision policy's efficiency, not
  pedagogical effectiveness.
- NOT a comparison against the actual architectures in the five
  reviewed papers (attention+RL, GraphRAG+RL, DKT+RL), this is the
  simplest possible tabular Q-learning agent, deliberately, to isolate
  "does structural knowledge save exploration cost" from "whose deep
  architecture is more sample-efficient."
- NOT evidence this generalizes beyond the 10-concept demo graph -
  see Action Items below.

---

## 5. The Hybrid Recommender and the Local Neural Network

This section documents the direct answer to "does this actually use AI":
yes, a real one, trained locally, with zero external API dependency.

### What "hybrid recommender" means here

A hybrid recommender combines more than one recommendation strategy so
each covers the others' weaknesses. This system combines three, which is
a textbook hybrid design taught in recommender-systems courses:

1. **Knowledge-based filtering**: the prerequisite DAG
   (`knowledge_graph.py`) hard-constrains which concepts are even
   eligible right now. This is deterministic and never learned.
2. **Content-based filtering**: TF-IDF text relevance
   (`scoring.py: relevance_fn`) matches resource content to the target
   concept.
3. **Learned personalization**: the PC-BwK contextual bandit
   (`bandit.py`, Section 1 above) adapts which criteria to trust per
   learner, and a **trained neural network** (`backend/ml/`) predicts
   expected mastery gain per resource, standing in for the "Gain_i" term
   the original methodology document always referenced in its objective
   (`max sum Gain_i`) but never concretely defined.

### The neural network itself, stated plainly

- **What it is**: a real multi-layer perceptron (`sklearn.neural_network.MLPRegressor`,
  two hidden layers of 32 and 16 units, ReLU activations), trained with
  backpropagation via the Adam optimizer. This is not a lookup table, a
  rules engine, or a wrapper around a hosted LLM. It runs 100% locally,
  in-process, with no API key and no network call, ever, at inference
  time. `backend/ml/train_gain_predictor.py` trains it; the resulting
  weight file (`gain_predictor.joblib`) is loaded once at server startup
  by `backend/ml/predictor.py`.
- **What it predicts**: expected mastery gain from studying a given
  resource, as a function of the learner's current mastery, the
  resource's difficulty, how well its duration fits the available time,
  whether prerequisites are satisfied, and the resource's quality score.
- **What it was trained on, stated honestly**: simulated data generated
  from a "zone of proximal development" model (a well-known pedagogical
  concept: a resource whose difficulty sits just above a learner's
  current level produces more gain than one that's far too easy or far
  too hard). This nonlinear interaction between difficulty-mismatch and
  current mastery is exactly what a linear weighted-sum score (the
  system's previous approach) cannot represent, but a neural network can
  learn directly from data. **No real learner interaction data exists
  yet** (see Action Items), this is a real, honest limitation, not
  something to gloss over in your report. The model demonstrably learned
  the intended pattern: held-out test R-squared of 0.97, and it correctly
  ranks a well-matched, prerequisite-satisfied resource above a
  too-difficult one, which is above one where prerequisites are not met
  (verified in `train_gain_predictor.py`'s printed sanity checks).
- **Where it plugs in**: `scoring.py`'s weighted formula now has six
  criteria instead of five (relevance, quality, time_fit, prereq_fit,
  preference, predicted_gain), and the bandit gained a sixth arm
  ("gain_heavy") that a learner's personalization can converge toward if
  the neural network's predictions turn out to matter most for them.

### What to say, and not say, about this in your report

Say: "the system implements a hybrid recommender combining knowledge-based
hard constraints, content-based text relevance, and a locally-trained
neural network predicting expected learning gain, personalized via a
contextual bandit." That is accurate and checkable in the code.

Do not say: "the AI predicts how much a student will actually learn" as
a validated real-world claim. It predicts what the simulated training
distribution says, and that distribution encodes a plausible pedagogical
theory, not measured human outcomes. This is the same honesty boundary
as the baseline results in Section 4: a real, working, correctly-trained
model, evaluated so far only in simulation.

## 6. Action Items

1. Rerun the baseline with multiple seeds (3-5 minimum) and report
   mean plus variance, one seed is a demonstration, not a robust result.
2. Rerun on a subgraph of the 208-concept LectureBank domain to check
   whether the finding holds at real scale, tabular Q-learning's state
   space is 2^n, so a full 208-concept run is computationally infeasible;
   use a connected subgraph of ~15-20 concepts.
3. If asked "why 40 days specifically?" in a defense: it was chosen
   because it's close to the ~34-day structural minimum for this graph
   and goal, a generous budget wouldn't test anything. State this
   reasoning explicitly rather than let it look arbitrarily tuned.
4. Decide your primary framing: PC-BwK as the headline contribution
   (recommended), or as a personalization layer on top of the
   open-resource-aggregation framing, pick one so the report doesn't
   read as two competing pitches.
5. Verify every citation in Section 1 independently (venue, year,
   exact title) before submission, these were found via search.
6. Resource link rot: `backend/scripts/check_resource_links.py` is a
   real health-checker (its HTTP logic is verified against a reachable
   test URL) but has not been run against your actual resource URLs yet -
   run it yourself before any public demo.
7. Resource "quality" scoring is a placeholder (`quality` field in
   `resources.json`, hand-estimated; the YouTube ingestion heuristic in
   `ingestion.py` is untested against ground truth), state this as a
   known limitation.
