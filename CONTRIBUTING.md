# Contributing to Deplyze Vision

First off — thank you. Deplyze Vision is built by and for the computer-vision community, and every issue, PR, or discussion moves it forward.

This document explains how to contribute effectively.

---

## Code of Conduct

By participating, you agree to uphold a respectful, harassment-free environment. Be kind, be curious, assume good intent. Report unacceptable behavior via a private GitHub issue to the maintainers.

---

## Ways to Contribute

- **Bug reports** — the more reproducible, the better
- **Feature proposals** — open a discussion issue before large PRs
- **Documentation** — typos, clarifications, new guides
- **New models / tasks** — add a task head or integrate a new TF.js-compatible backbone
- **Tests** — regression tests are always welcome
- **Triage** — help label / reproduce incoming issues

---

## Development Setup

### Prerequisites
- Node.js 18+
- Yarn (do **not** use npm — it breaks the lockfile)
- Python 3.11+
- MongoDB 6+ (local or Docker)

### 1. Fork & clone
```bash
git clone https://github.com/<your-username>/deplyze-vision.git
cd deplyze-vision
git remote add upstream https://github.com/dyglo/deplyze-vision.git
```

### 2. Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # set MONGO_URL=mongodb://localhost:27017 and DB_NAME=deplyze
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

### 3. Frontend
```bash
cd ../frontend
yarn install
cp .env.example .env    # set REACT_APP_BACKEND_URL=http://localhost:8001
yarn start
```

Visit `http://localhost:3000`.

---

## Branch & Commit Conventions

### Branches
- `main` — always deployable
- `feat/<short-description>` — new features
- `fix/<short-description>` — bug fixes
- `docs/<short-description>` — docs only
- `chore/<short-description>` — refactors, tooling, CI

### Commits (Conventional Commits)
```
feat(studio): add WebGPU backend toggle
fix(benchmark): prevent NaN when FPS=0
docs(readme): add export instructions for YOLO26-seg
chore(deps): bump tensorflow/tfjs to 4.22.1
```

Keep commits **atomic** — one logical change per commit.

---

## Pull Request Workflow

1. Create a feature branch off up-to-date `main`.
2. Make your changes. Keep them focused.
3. Run linters and tests (see below).
4. Push your branch and open a PR against `main`.
5. Fill out the PR template — **link the issue it closes**.
6. A maintainer will review within ~72 hours.
7. Squash & merge once approved.

### PR Checklist
- [ ] The PR title follows Conventional Commits
- [ ] I added/updated tests where relevant
- [ ] I ran `yarn lint` (frontend) and `ruff check backend/` (backend)
- [ ] I added `data-testid` attributes to new interactive elements
- [ ] I updated `README.md` / `memory/PRD.md` / `memory/CHANGELOG.md` if behavior changed
- [ ] The app still builds and runs (`yarn build` + `uvicorn server:app`)

---

## Coding Conventions

### Frontend (React + Tailwind)
- Functional components only. Hooks over classes.
- Prefer the existing **Shadcn/UI** primitives in `frontend/src/components/ui/` over new ad-hoc components.
- Use the Claude palette tokens from `index.css` — **no hardcoded hex outside the theme layer**.
- Every interactive element (button, input, nav link, modal, etc.) **must have a unique `data-testid`** in kebab-case, e.g. `data-testid="studio-run-inference-btn"`.
- Keep components small (< 150 lines). Split when they grow.

### Backend (FastAPI)
- All routes must be prefixed with `/api`.
- Use Pydantic response models. Never return raw MongoDB documents (`_id` is not JSON-serializable).
- Always exclude `_id` when querying: `db.col.find({}, {"_id": 0})`.
- Use `datetime.now(timezone.utc)` — never `datetime.utcnow()`.
- Read env vars via `os.environ.get("KEY")` — no defaults. Missing config should fail fast.

### General
- No secrets in code. Use `.env` files and `.env.example` placeholders.
- Prefer editing existing files over creating new ones.
- Keep imports sorted and unused ones removed.

---

## Testing

### Frontend
```bash
cd frontend
yarn lint
yarn test
```

### Backend
```bash
cd backend
ruff check .
pytest tests/
```

### End-to-End Smoke Test
1. Start both servers.
2. Open `/studio`, upload a sample image, run inference — confirm overlay renders.
3. Toggle compare mode, pick a second model — confirm both canvases update.
4. Switch task (e.g. detect → segment) — confirm canvas clears.
5. Open `/benchmark` — confirm the run appears.

---

## Adding a New Task or Model

1. Add the task metadata to `frontend/src/pages/Landing.jsx` (`TASKS` array).
2. Add an inference hook under `frontend/src/lib/inference/` following the existing `useYoloDetect.js` pattern.
3. Register the model in `backend/server.py` `/api/models` config.
4. Update `Studio.jsx` to dispatch to your hook when the task is selected.
5. Add a regression test in `tests/` that loads the model and runs a fixture image.

---

## Reporting Bugs

Open a GitHub issue with:
- **What happened** vs **what you expected**
- **Steps to reproduce** (the smaller the better)
- **Environment:** OS, browser, GPU, Node version
- Screenshots / console logs / network traces if relevant

Tag it `bug` — a maintainer will triage within ~72h.

---

## Proposing Features

Open a `Discussion` or an issue tagged `enhancement`. Include:
- The problem you're solving
- The proposed approach (UX + technical)
- Alternatives considered
- Whether you'd like to implement it yourself

Let's agree on scope **before** you write code — saves everyone time.

---

## Releasing (Maintainers Only)

1. Ensure `main` is green.
2. Bump version in `package.json` and `pyproject.toml`.
3. Update `memory/CHANGELOG.md`.
4. Tag: `git tag v0.x.y && git push --tags`.
5. GitHub Action publishes the release notes.

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](./LICENSE).

---

Questions? Open a GitHub Discussion or ping the maintainers. Happy hacking. 🎯
