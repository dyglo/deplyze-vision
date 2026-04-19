# Deplyze Vision — YOLO26 + TF.js Computer Vision Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.22-orange.svg)](https://www.tensorflow.org/js)
[![Ultralytics YOLO26](https://img.shields.io/badge/Ultralytics-YOLO26-%23C15F3C.svg)](https://docs.ultralytics.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](./CONTRIBUTING.md)
[![GitHub stars](https://img.shields.io/github/stars/dyglo/deplyze-vision?style=social)](https://github.com/dyglo/deplyze-vision)

> A professional, open-source computer vision platform where CV engineers can run the latest **Ultralytics YOLO26** models — entirely in the browser via **TensorFlow.js**. No GPU server. No vendor lock-in. No privacy trade-offs.

---

## Why Deplyze Vision?

Most CV playgrounds either (a) phone frames home to a rented GPU, (b) lock you into a SaaS tier, or (c) ship a demo that falls over the moment you leave the happy path. Deplyze Vision is different:

- **Everything runs locally.** Inference happens in your browser tab via `tfjs-webgl` / `wasm`. Your images never leave the device.
- **Production tasks, not toys.** Detection, pose, segmentation, classification, tracking — all with exportable results.
- **Side-by-side model comparison + auto-benchmarking.** Compare two YOLO26 variants head-to-head and watch the Benchmark page self-populate.
- **Open-source, self-hostable.** FastAPI + React + MongoDB. MIT-licensed. Fork it, extend it, ship it.

---

## Features

| Capability | What it does |
|---|---|
| **Inference Studio** | Upload images/video or stream from your webcam and run YOLO26 models live. |
| **Side-by-side Compare** | Run two models on the same input and diff their outputs in real time. |
| **Progress Animation** | Live scan-bar feedback while the model warms up and infers. |
| **Task State Isolation** | Switching task (detect → segment → pose …) clears stale state so you never see wrong overlays. |
| **Benchmark Page** | Auto-aggregates latency / FPS / detection counts from every compare run. |
| **Project Management** | Group runs, datasets, and exports into reusable projects. |
| **Model Hub** | Drop in any exported `model.json` + shards for YOLO26, MobileNet, BodyPix, MoveNet, etc. |
| **Export Anywhere** | JSON, CSV, or annotated PNG for every run. |
| **All Input Sources** | Images · MP4/WebM video · live webcam. |

---

## Supported Tasks

- **Object Detection** — 80 COCO classes out-of-the-box via YOLO26 or COCO-SSD
- **Pose Estimation** — 17 keypoints via MoveNet / YOLO26-pose
- **Segmentation** — Instance / person segmentation via BodyPix or YOLO26-seg
- **Classification** — 1000 ImageNet classes via MobileNet / YOLO26-cls
- **Object Tracking** — Multi-object centroid tracker on top of any detector

---

## Tech Stack

- **Frontend:** React 19 · React Router · Tailwind CSS · Shadcn/UI · Phosphor Icons
- **Inference Engine:** TensorFlow.js 4.22 (WebGL + WASM backends)
- **Backend:** FastAPI · Python 3.11 · Motor (async MongoDB)
- **Database:** MongoDB (project + run metadata only — no media stored server-side)
- **Models:** Ultralytics YOLO26, COCO-SSD, MoveNet, BodyPix, MobileNet

---

## Quickstart

### Prerequisites
- Node.js 18+ and npm
- Python 3.11+ and [uv](https://github.com/astral-sh/uv)
- Docker (optional, for local MongoDB)

### 1. Clone
```bash
git clone https://github.com/dyglo/deplyze-vision.git
cd deplyze-vision
```

### 2. Database (Docker)
If you don't have MongoDB running locally, start it via Docker:
```bash
docker run -d --name vision-mongo -p 27017:27017 mongo:latest
```

### 3. Backend
```bash
cd backend
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
cp .env.example .env       # then set MONGO_URL=mongodb://localhost:27017 and DB_NAME=deplyze_vision
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

### 4. Frontend
```bash
cd ../frontend
npm install --legacy-peer-deps
cp .env.example .env       # set REACT_APP_BACKEND_URL=http://localhost:8001
npm start
```

App is now live at **http://localhost:3000**. API docs at **http://localhost:8001/api/docs**.

---

## Exporting your own YOLO26 model

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="tfjs")   # produces model.json + .bin shards
```

Upload the resulting folder via **Model Hub → Import TF.js Model**, then pick it from the Studio dropdown.

---

## Design System

Deplyze Vision uses the **Claude "Pampas" palette** — calm, readable, engineering-friendly.

| Role | Name | Hex |
|---|---|---|
| Background | Pampas | `#F4F3EE` |
| Text / Primary | Cod Gray | `#141413` |
| Accent | Crail | `#C15F3C` |
| Muted | Cloudy | `#B1ADA1` |

Task accents (kept for brand/semantic clarity) live on task pills and overlays.

---

## Roadmap

- [x] In-browser inference for 5 CV tasks
- [x] Side-by-side model comparison
- [x] Auto-populating Benchmark page
- [x] Claude "Pampas" light theme
- [ ] Persisted benchmarks (per-project history)
- [ ] Shareable inference sessions (URL-encoded state)
- [ ] ONNX Runtime Web backend (alt to TF.js)
- [ ] WebGPU backend once TF.js hits stable

See [ROADMAP.md](./memory/ROADMAP.md) for the full backlog.

---

## Contributing

We love contributors. Start with **[CONTRIBUTING.md](./CONTRIBUTING.md)** — it covers:
- Dev setup & coding conventions
- Commit / PR workflow
- How to add a new task or model
- Running the test suite

Good first issues are tagged [`good-first-issue`](https://github.com/dyglo/deplyze-vision/labels/good-first-issue).

---

## License

Released under the [MIT License](./LICENSE). YOLO26 weights are distributed by Ultralytics under their own license — please review it before redistributing weights with a product.

---

## Acknowledgements

- [Ultralytics YOLO](https://docs.ultralytics.com/) — for the YOLO family
- [TensorFlow.js](https://www.tensorflow.org/js) — for making in-browser ML real
- [Shadcn/UI](https://ui.shadcn.com/) and [Phosphor Icons](https://phosphoricons.com/)

---

<p align="center">
  <b>⭐ If Deplyze Vision saves you time, please star the repo.</b><br/>
  <a href="https://github.com/dyglo/deplyze-vision">github.com/dyglo/deplyze-vision</a>
</p>
