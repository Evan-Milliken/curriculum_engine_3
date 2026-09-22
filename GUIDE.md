# GUIDE.md: From Zero to Google Play

One guide, start to finish. If you've already done part of this (you
mentioned finishing through running it locally), skip ahead, each part
says what to check before moving on.

---

## Part 0, What you're building

Two programs: a **backend** (Python/FastAPI, the knowledge graph,
scoring, bandit, and now a real SQLite database) and a **frontend** (one
HTML file, what the learner sees and taps). The frontend talks to the
backend over HTTP. Getting this onto Google Play means: (1) host the
backend somewhere with a public address, (2) host the frontend somewhere
with a public address, (3) wrap the frontend as an Android app. Three
deployments, done in order.

---

## Part 1, Install the tools (one-time)

### Python
**Mac:** `python3 --version` in Terminal, need 3.10+. If missing, install
from https://www.python.org/downloads/.
**Windows:** install from https://www.python.org/downloads/, **check
"Add python.exe to PATH"** on the first install screen, this is the most
common thing people miss. Then `python --version` in Command Prompt.

### Git
**Mac:** `git --version` in Terminal (usually prompts an install the first
time). **Windows:** https://git-scm.com/download/win, accept defaults.
Then, once, anywhere:
```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### GitHub account, code editor
GitHub: https://github.com/signup. VS Code: https://code.visualstudio.com/.

---

## Part 2, Get the project under Git

From inside your project folder (VS Code: File -> Open Folder, then View -> Terminal):
```bash
git init
git add .
git commit -m "Initial commit"
```
On github.com: "+" -> "New repository" -> name it (e.g. `curriculum-engine`)
-> Public -> do NOT add a README (you already have one) -> Create. Then:
```bash
git remote add origin https://github.com/YOUR_USERNAME/curriculum-engine.git
git branch -M main
git push -u origin main
```

---

## Part 3, Run the backend locally

```bash
cd backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Open **http://localhost:8000/docs** - you should see the Swagger UI
listing every endpoint. This also creates `backend/data/curriculum_engine.db`
(SQLite) on first run - that file now holds every learner's mastery and
bandit state persistently. Delete it any time to reset all data.

---

## Part 4, Run the frontend locally

Second terminal:
```bash
cd frontend
python3 -m http.server 5500
```
Open **http://localhost:5500**. You should see "Pathfinder" - pick a
domain, a goal, go through the diagnostic, and see a personalized plan.
**If you've already done this and it worked, you're caught up - continue
to Part 5.**

If you see 404s in the terminal for `icon-192.png` or `icon-512.png`:
those files now exist in `frontend/` - if you still see 404s, confirm
they're actually present in your local `frontend/` folder.

---

## Part 5, Understand the file layout

```
curriculum-engine/
├── backend/
│   ├── main.py              <- the API
│   ├── db.py                <- SQLite persistence (learner state, bandit state)
│   ├── models.py            <- data shapes
│   ├── knowledge_graph.py   <- the concept dependency graph
│   ├── knowledge_state.py   <- mastery tracking (EMA formula)
│   ├── scoring.py           <- resource ranking
│   ├── optimizer.py         <- the knapsack that builds the daily plan
│   ├── bandit.py            <- the PC-BwK personalization layer (the novelty)
│   ├── ingestion.py         <- pulls real YouTube video metadata (needs your own API key)
│   ├── ml/                   <- the local neural network (trained, no API key, ever)
│   ├── baselines/           <- the Q-learning vs. PC-BwK comparison experiment
│   ├── scripts/              <- dataset build/maintenance scripts
│   └── data/                 <- concept graphs, resources, the SQLite db file
├── frontend/
│   ├── index.html            <- the entire app UI
│   ├── manifest.json         <- makes it installable (PWA)
│   ├── sw.js                 <- offline caching (required for Play Store)
│   ├── icon-192.png           <- app icon
│   └── icon-512.png           <- app icon
├── README.md                  <- quick index
└── RESEARCH.md                <- the novelty claim, literature comparison, datasets, baseline results
```

---

## Part 6, Deploy the backend publicly

You need a host that keeps a Python process running continuously.
**Check current free-tier terms yourself** - they change.

### Render (recommended)
1. https://render.com -> sign up with GitHub.
2. "New +" -> "Web Service" -> connect your repo.
3. Set: **Root Directory** = `backend`, **Build Command** = `pip install -r requirements.txt`,
   **Start Command** = `uvicorn main:app --host 0.0.0.0 --port $PORT`.
4. Deploy. You get a URL like `https://your-app.onrender.com` - test
   `<that-url>/docs`.

**Important for persistence on a free tier:** most free hosting tiers use
an ephemeral filesystem - your SQLite file (`data/curriculum_engine.db`)
may be wiped on redeploy or after inactivity. That's fine for a course
demo; if you need data to survive redeploys, use the host's persistent
disk add-on (Render has one) or move to a hosted Postgres instance later.

### Railway (alternative)
Same flow at https://railway.app.

---

## Part 7, Deploy the frontend publicly

### 7.1 Point it at your live backend
In `frontend/index.html`, find:
```js
const API_BASE = window.CURRICULUM_API_BASE || "http://localhost:8000";
```
Change `"http://localhost:8000"` to your Part 6 URL.

### 7.2 Deploy to Netlify
1. https://app.netlify.com/signup -> sign up with GitHub.
2. "Add new site" -> "Import an existing project" -> your repo.
3. **Base directory** = `frontend`, build command empty.
4. Deploy -> you get `https://your-app.netlify.app`.
5. Open it, confirm the full flow works against your live backend.

Chrome/Edge desktop should now show an install icon in the address bar -
that's "installable on desktop," already satisfied, no extra work.

---

## Part 8, Wrap it for Google Play

The path here is a **Progressive Web App wrapped as a Trusted Web Activity
(TWA)** - a real, Play-listed, downloadable Android app that displays
your hosted site. This is a standard, Google-documented approach (via the
`bubblewrap` CLI from the Chrome team), not a shortcut or a hack.

### 8.1 Install Node.js
https://nodejs.org - download the LTS version, install with defaults.

### 8.2 Install and run Bubblewrap
```bash
npm install -g @bubblewrap/cli
bubblewrap init --manifest https://your-app.netlify.app/manifest.json
bubblewrap build
```
Answer the prompts (package name like `com.yourteam.curriculumengine`,
etc). The first run offers to install a JDK and Android SDK for you if
missing - let it. This produces a signed `.aab` file.

### 8.3 Host `assetlinks.json`
`bubblewrap init` generates an `assetlinks.json` file. Host it at:
```
https://your-app.netlify.app/.well-known/assetlinks.json
```
This proves you own both the site and the app - skipping it is the most
common reason a TWA opens with browser address-bar chrome visible instead
of looking like a native app.

### 8.4 Google Play Developer account
https://support.google.com/googleplay/android-developer - **check the
current fee and identity-verification requirements yourself**, these
change. Historically a one-time $25 fee plus identity verification that
can take a few days.

### 8.5 Upload and list
Play Console -> Create app -> upload the `.aab`. You'll need: an app icon
(512x512 - you have `icon-512.png`), a feature graphic, 2+ screenshots, a
short + full description, a content rating questionnaire, and a
**privacy policy URL** (required even for a course project if you collect
any data - even a one-page static page describing what you store is
enough, but it must exist and be linked).

### 8.6 Release track
- **Internal testing**: fastest (minutes), limited to testers you add by
  email - enough for your professor to install it on her phone. **Use
  this for your grading deadline** rather than betting on production
  review timing.
- **Production**: public listing, subject to Google's review queue and
  whatever the current testing-track policy is for new developer
  accounts - verify this yourself before committing to a timeline.

---

## Part 9, Common problems

- **"Failed to fetch" in the browser console**: `API_BASE` in
  `index.html` doesn't match your actual backend URL, or the backend
  isn't running/deployed yet.
- **CORS error**: shouldn't happen with the code as given
  (`CORSMiddleware` allows all origins in `main.py`) - if you see it after
  editing that file, you narrowed `allow_origins` without adding your
  real frontend URL.
- **`ModuleNotFoundError` running uvicorn**: you forgot to activate the
  virtual environment first.
- **Backend "loses" data after a while on the free host**: see Part 6's
  note on ephemeral filesystems - this is a hosting-tier limitation, not
  a bug in `db.py` (which is verified to persist correctly on a normal
  filesystem - see RESEARCH.md's persistence test).
- **TWA opens with the browser address bar visible**: you skipped hosting
  `assetlinks.json` (Part 8.3).

---

## Part 10, Order of operations, summarized

1. Part 3 + 4 working locally (you're likely already here).
2. Part 6: backend deployed, `/docs` loads publicly.
3. Part 7: frontend deployed, full flow works against the live backend.
4. Confirm desktop install works (Chrome/Edge install icon).
5. Part 8: Play Store packaging - last step, not first.
