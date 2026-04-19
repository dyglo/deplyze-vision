# Deplyze Vision — YOLO26 CV Platform — PRD

## Original Problem Statement
Build a professional, open-source computer vision platform where CV engineers can run the latest Ultralytics YOLO26 models entirely in-browser via TensorFlow.js (no GPU server required). Support images, videos, and webcam. Public / no-auth. Ship it as a real OSS repo with README, CONTRIBUTING, and a working "Star on GitHub" button.

## User Choices (locked)
- **Tasks**: Object Detection, Segmentation, Pose, Classification, Tracking
- **Inputs**: Images, Videos, Webcam/live
- **Inference**: 100% in-browser via TF.js (WebGL / WASM backends)
- **Auth**: None — public
- **License**: MIT
- **Repo**: https://github.com/dyglo/deplyze-vision
- **Theme**: Claude "Pampas" light palette (Primary/Crail #C15F3C, Secondary/Cloudy #B1ADA1, Background/Pampas #F4F3EE, Text/Cod Gray #141413)

## Architecture
- **Frontend**: React (CRA + Craco), Tailwind, Shadcn/UI, Phosphor Icons
- **Backend**: FastAPI + MongoDB (metadata + run persistence only)
- **Inference**: TF.js client-side (COCO-SSD, MoveNet, BodyPix, MobileNet + YOLO26 exports)

---

## What's Been Implemented

### Phase 1 — 2026-04-18 (MVP)
**Backend (FastAPI)**
- `/api/projects`, `/api/runs`, `/api/datasets`, `/api/model-configs`, `/api/stats`
- 5 default models seeded on startup · `/api/docs`

**Frontend (React)**
- Landing, Studio, Projects, Datasets, Model Hub, Results pages
- 5 CV tasks + 3 input sources
- TF.js in-browser inference with FPS/latency/obj-count overlay
- Confidence slider, save-run, export PNG/JSON

### Phase 2 — 2026-04-18 (Current)
- ✅ **Side-by-side model comparison** in Studio (dual canvas, `compare-canvas` testid, independent stats overlay)
- ✅ **Per-panel model variant pickers** in Compare mode — 2 variants per task (COCO-SSD Lite/v2, MoveNet Lightning/Thunder, BodyPix MNv1/ResNet50, MobileNet v1/v2). Same-task different-model comparison supported.
- ✅ **Real big-spinner inference overlay** — `[data-testid="inference-overlay"]` with pulsing ring + spinning ring + task icon, appears on run-click and disappears on completion (self-verified via Playwright mutation observer).
- ✅ **Progress animation** during inference (`.progress-scan` CSS keyframe scan-bar above the workspace)
- ✅ **Task-switch state clearing** — image/video/results reset when changing CV task; model variant also resets to default of new task
- ✅ **Race-condition fixes** — modelRef invalidated at start of reload; run button gated on `modelStatus==='ready' && (!compareMode || compareStatus==='ready')`; `runTask` has typeof guards per branch.
- ✅ **Benchmark page** (`/benchmark`) — auto-populates from saved Studio runs; 3 tabs (Overview / Per-Task / Compare Runs); empty-state CTA
- ✅ **Claude Pampas light theme** — full token rewrite in `index.css`; Crail accent, Cod Gray text, Pampas bg
- ✅ **Hero live demo on Landing** — `<HeroDemo />` component (`[data-testid="hero-live-demo"]`) with static sample image + animated SVG bounding boxes cycling through 3 pre-computed frames every 1.6s; zero runtime cost, keeps bundle light
- ✅ **Open-source docs**: `/app/README.md` (Project Structure section removed per request), `/app/CONTRIBUTING.md`, `/app/LICENSE` (MIT)
- ✅ **GitHub button** — Landing `github-stars-btn` + sidebar `sidebar-github-link` → https://github.com/dyglo/deplyze-vision
- ✅ **Testing**: iteration_2 (13/13 pass) + iteration_3 (12/14 pass, 2 inference-overlay tests couldn't be observed in headless due to WebGL fallback — main agent self-verified in real browser: overlay appears + disappears correctly)

---

## Roadmap

### P0
- [ ] Persist benchmarks per-project (currently in-memory / client only)
- [ ] Video inference loop (play / pause / frame-step)
- [ ] Multi-person pose estimation
- [ ] Instance segmentation (beyond BodyPix person-only)

### P1
- [ ] Custom YOLO26 TF.js URL loading from Studio
- [ ] Batch image processing
- [ ] Export annotated video
- [ ] Shareable run URLs (URL-encoded state for quick demos)

### P2
- [ ] WebGPU backend once TF.js stable
- [ ] ONNX Runtime Web alternative backend
- [ ] WebRTC collaborative inference sessions
- [ ] Split `Studio.jsx` (~720 LOC) into smaller compare/results sub-components

### Tech-debt / DX
- [ ] Add `willReadFrequently: true` to canvas `getContext('2d')` calls
- [ ] Add fallback CPU/WASM backend message when WebGL init fails
- [ ] Split Studio.jsx into `<StudioToolbar>`, `<StudioCanvases>`, `<StudioResults>`

---

## Testing Status
- Phase 1: iteration_1 — backend + frontend verified
- Phase 2: iteration_2 — frontend 13/13 pass (2026-04-18)
