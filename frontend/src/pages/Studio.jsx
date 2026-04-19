import { useState, useRef, useCallback, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import * as tf from "@tensorflow/tfjs";
import axios from "axios";
import {
  BoundingBox, PersonSimpleRun, Intersect, Tag, Path,
  Image as ImageIcon, VideoCamera, Camera,
  Play, Stop, FloppyDisk, Export, SpinnerGap,
  Warning, UploadSimple, Sliders, Columns, X, Trash,
} from "@phosphor-icons/react";
import {
  drawDetections, drawPose, drawClassification,
  drawSegmentationMask, drawTracking,
} from "../utils/drawUtils";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const TASKS = [
  { id: "detect",   label: "Detection",     icon: BoundingBox,      color: "#008B22", shortLabel: "DETECT" },
  { id: "pose",     label: "Pose",          icon: PersonSimpleRun,  color: "#CC1144", shortLabel: "POSE" },
  { id: "segment",  label: "Segment",       icon: Intersect,        color: "#0087B3", shortLabel: "SEGMENT" },
  { id: "classify", label: "Classify",      icon: Tag,              color: "#7B1CC4", shortLabel: "CLASSIFY" },
  { id: "track",    label: "Track",         icon: Path,             color: "#B08000", shortLabel: "TRACK" },
];

// Selectable model variants per task (all ship built-in; custom YOLO26 TF.js can be added later)
const MODELS_PER_TASK = {
  detect: [
    { id: "coco-ssd-lite", label: "COCO-SSD Lite",   sub: "lite MobileNet v2 · ~5 MB" },
    { id: "coco-ssd-v2",   label: "COCO-SSD v2",     sub: "MobileNet v2 · ~25 MB" },
  ],
  pose: [
    { id: "movenet-lightning", label: "MoveNet Lightning", sub: "Fast · 17 kpts" },
    { id: "movenet-thunder",   label: "MoveNet Thunder",   sub: "Accurate · 17 kpts" },
  ],
  segment: [
    { id: "bodypix-mnv1",    label: "BodyPix MobileNetV1", sub: "Fast · person seg" },
    { id: "bodypix-resnet50",label: "BodyPix ResNet50",    sub: "Accurate · person seg" },
  ],
  classify: [
    { id: "mobilenet-v1", label: "MobileNet v1", sub: "1000 ImageNet cls" },
    { id: "mobilenet-v2", label: "MobileNet v2", sub: "1000 ImageNet cls" },
  ],
  track: [
    { id: "coco-ssd-lite", label: "COCO-SSD Lite + Centroid", sub: "Fast tracker" },
    { id: "coco-ssd-v2",   label: "COCO-SSD v2 + Centroid",   sub: "Accurate tracker" },
  ],
};

const defaultVariantId = (task) => MODELS_PER_TASK[task]?.[0]?.id || "";

const modelCache = {};
let trackerNextId = 0;
let trackerObjects = {};

function updateTracker(detections) {
  const newObjects = {};
  const assigned = new Set();
  detections.forEach((det) => {
    const cx = det.bbox[0] + det.bbox[2] / 2;
    const cy = det.bbox[1] + det.bbox[3] / 2;
    let bestId = null, bestDist = 150;
    Object.entries(trackerObjects).forEach(([id, obj]) => {
      if (assigned.has(id)) return;
      const d = Math.hypot(cx - obj.cx, cy - obj.cy);
      if (d < bestDist) { bestDist = d; bestId = id; }
    });
    if (bestId !== null) { newObjects[bestId] = { cx, cy, det }; assigned.add(bestId); }
    else { newObjects[trackerNextId++] = { cx, cy, det }; }
  });
  trackerObjects = newObjects;
  return newObjects;
}

async function loadModelForTask(task, variantId) {
  const vid = variantId || defaultVariantId(task);
  const key = `${task}:${vid}`;
  if (modelCache[key]) return modelCache[key];
  await tf.ready();
  let model;
  switch (task) {
    case "detect": case "track": {
      const m = await import("@tensorflow-models/coco-ssd");
      const base = vid === "coco-ssd-v2" ? "mobilenet_v2" : "lite_mobilenet_v2";
      model = await m.load({ base });
      break;
    }
    case "pose": {
      const m = await import("@tensorflow-models/pose-detection");
      const modelType = vid === "movenet-thunder"
        ? m.movenet.modelType.SINGLEPOSE_THUNDER
        : m.movenet.modelType.SINGLEPOSE_LIGHTNING;
      model = await m.createDetector(m.SupportedModels.MoveNet, { modelType });
      break;
    }
    case "segment": {
      const m = await import("@tensorflow-models/body-pix");
      const cfg = vid === "bodypix-resnet50"
        ? { architecture: "ResNet50", outputStride: 16, quantBytes: 2 }
        : { architecture: "MobileNetV1", outputStride: 16, multiplier: 0.75, quantBytes: 2 };
      model = await m.load(cfg);
      break;
    }
    case "classify": {
      const m = await import("@tensorflow-models/mobilenet");
      const version = vid === "mobilenet-v1" ? 1 : 2;
      model = await m.load({ version, alpha: 1.0 });
      break;
    }
    default: throw new Error(`Unknown task: ${task}`);
  }
  modelCache[key] = model;
  return model;
}

async function runTask(taskKey, model, inputEl, confidence) {
  if (!model) return null;
  switch (taskKey) {
    case "segment": {
      if (typeof model.segmentPerson !== "function") return null;
      const seg = await model.segmentPerson(inputEl, { flipHorizontal: false, internalResolution: "medium", segmentationThreshold: confidence });
      return { type: "segment", raw: seg };
    }
    case "pose":
      if (typeof model.estimatePoses !== "function") return null;
      return { type: "pose", raw: await model.estimatePoses(inputEl) };
    case "classify":
      if (typeof model.classify !== "function") return null;
      return { type: "classify", raw: await model.classify(inputEl) };
    case "detect":
      if (typeof model.detect !== "function") return null;
      return { type: "detect", raw: await model.detect(inputEl, undefined, confidence) };
    case "track":
      if (typeof model.detect !== "function") return null;
      return { type: "track", raw: await model.detect(inputEl, undefined, confidence) };
    default: return null;
  }
}

function renderOnCanvas(canvas, inputEl, taskResult, taskKey) {
  if (!canvas || !inputEl || !taskResult) return [];
  const ctx = canvas.getContext("2d");
  if (inputEl.tagName === "VIDEO") {
    canvas.width = inputEl.videoWidth || 640;
    canvas.height = inputEl.videoHeight || 480;
  } else {
    canvas.width = inputEl.naturalWidth || 640;
    canvas.height = inputEl.naturalHeight || 480;
  }

  const { type, raw } = taskResult;
  let dets = [];

  if (type === "segment") {
    ctx.drawImage(inputEl, 0, 0, canvas.width, canvas.height);
    drawSegmentationMask(ctx, raw, canvas.width, canvas.height);
    dets = [{ type: "segmentation", score: raw.data.filter((v) => v === 1).length / raw.data.length }];
  } else if (type === "pose") {
    ctx.drawImage(inputEl, 0, 0, canvas.width, canvas.height);
    drawPose(ctx, raw);
    dets = raw.map((p) => ({ type: "pose", keypoints: p.keypoints.length, score: p.score || 1 }));
  } else if (type === "classify") {
    ctx.drawImage(inputEl, 0, 0, canvas.width, canvas.height);
    drawClassification(ctx, raw);
    dets = raw.slice(0, 5).map((p) => ({ class: p.className, score: p.probability, bbox: [0, 0, 0, 0] }));
  } else if (type === "detect") {
    ctx.drawImage(inputEl, 0, 0, canvas.width, canvas.height);
    drawDetections(ctx, raw);
    dets = raw.map((d) => ({ class: d.class, score: d.score, bbox: d.bbox }));
  } else if (type === "track") {
    ctx.drawImage(inputEl, 0, 0, canvas.width, canvas.height);
    const tracked = updateTracker(raw);
    drawTracking(ctx, tracked);
    dets = Object.entries(tracked).map(([id, obj]) => ({ id: parseInt(id), class: obj.det.class, score: obj.det.score, bbox: obj.det.bbox }));
  }
  return dets;
}

export default function Studio() {
  const [searchParams] = useSearchParams();
  const initialTask = searchParams.get("task") || "detect";

  const [task, setTask] = useState(initialTask);
  const [modelVariant, setModelVariant] = useState(defaultVariantId(initialTask));
  const [inputType, setInputType] = useState("image");
  const [modelStatus, setModelStatus] = useState("idle");
  const [isRunning, setIsRunning] = useState(false);
  const [isInferring, setIsInferring] = useState(false);
  const [confidence, setConfidence] = useState(0.5);
  const [detections, setDetections] = useState([]);
  const [stats, setStats] = useState({ fps: 0, latency: 0, count: 0 });
  const [hasInput, setHasInput] = useState(false);
  const [hasResult, setHasResult] = useState(false);
  const [projects, setProjects] = useState([]);
  const [selectedProject, setSelectedProject] = useState("");
  const [saveStatus, setSaveStatus] = useState("");
  // Compare mode
  const [compareMode, setCompareMode] = useState(false);
  const [compareTask, setCompareTask] = useState(initialTask);
  const [compareModelVariant, setCompareModelVariant] = useState(
    MODELS_PER_TASK[initialTask]?.[1]?.id || defaultVariantId(initialTask)
  );
  const [compareStatus, setCompareStatus] = useState("idle");
  const [compareDetections, setCompareDetections] = useState([]);
  const [compareStats, setCompareStats] = useState({ fps: 0, latency: 0, count: 0 });

  const canvasRef = useRef(null);
  const compareCanvasRef = useRef(null);
  const imgRef = useRef(null);
  const videoRef = useRef(null);
  const rafRef = useRef(null);
  const modelRef = useRef(null);
  const compareModelRef = useRef(null);
  const fpsRef = useRef({ frames: 0, last: 0 });

  useEffect(() => { axios.get(`${API}/projects`).then((r) => setProjects(r.data)).catch(() => {}); }, []);

  // Load model on task / variant change + clear canvas
  useEffect(() => {
    setModelStatus("loading");
    setDetections([]);
    setHasResult(false);
    clearCanvas();
    trackerObjects = {}; trackerNextId = 0;
    modelRef.current = null; // invalidate stale model so inference can't fire with wrong model
    loadModelForTask(task, modelVariant).then((m) => { modelRef.current = m; setModelStatus("ready"); })
      .catch(() => setModelStatus("error"));
  }, [task, modelVariant]);

  // Load compare model when compare task or variant changes
  useEffect(() => {
    if (!compareMode) return;
    setCompareStatus("loading");
    compareModelRef.current = null;
    loadModelForTask(compareTask, compareModelVariant).then((m) => { compareModelRef.current = m; setCompareStatus("ready"); })
      .catch(() => setCompareStatus("error"));
  }, [compareTask, compareModelVariant, compareMode]);

  const clearCanvas = () => {
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    if (compareCanvasRef.current) {
      const ctx = compareCanvasRef.current.getContext("2d");
      ctx.clearRect(0, 0, compareCanvasRef.current.width, compareCanvasRef.current.height);
    }
  };

  const clearInput = () => {
    stopInference();
    if (imgRef.current) imgRef.current.src = "";
    if (videoRef.current) { videoRef.current.src = ""; videoRef.current.srcObject = null; }
    clearCanvas();
    setHasInput(false);
    setHasResult(false);
    setDetections([]);
    setCompareDetections([]);
    setStats({ fps: 0, latency: 0, count: 0 });
  };

  const handleTaskChange = (newTask) => {
    if (isRunning) stopInference();
    setTask(newTask);
    setModelVariant(defaultVariantId(newTask));
    setHasResult(false);
    setDetections([]);
    // clear any left-over input so stale frames don't show
    clearInput();
  };

  const handleCompareTaskChange = (newTask) => {
    setCompareTask(newTask);
    // pick the OTHER variant if both panels end up on same task (nicer compare default)
    const variants = MODELS_PER_TASK[newTask] || [];
    if (newTask === task && variants.length > 1) {
      const other = variants.find((v) => v.id !== modelVariant);
      setCompareModelVariant(other ? other.id : variants[0].id);
    } else {
      setCompareModelVariant(defaultVariantId(newTask));
    }
  };

  // Core inference function
  const runInference = useCallback(async (inputEl) => {
    if (!modelRef.current) return null;
    if (compareMode && !compareModelRef.current) return null;
    setIsInferring(true);
    const t0 = performance.now();
    let dets = [];
    let compareDets = [];

    try {
      if (compareMode && compareModelRef.current) {
        const [left, right] = await Promise.all([
          runTask(task, modelRef.current, inputEl, confidence),
          runTask(compareTask, compareModelRef.current, inputEl, confidence),
        ]);
        dets = renderOnCanvas(canvasRef.current, inputEl, left, task);
        compareDets = renderOnCanvas(compareCanvasRef.current, inputEl, right, compareTask);
        const cLatency = performance.now() - t0;
        setCompareStats({ fps: Math.round(1000 / cLatency), latency: Math.round(cLatency), count: compareDets.length });
        setCompareDetections(compareDets);
      } else {
        const result = await runTask(task, modelRef.current, inputEl, confidence);
        dets = renderOnCanvas(canvasRef.current, inputEl, result, task);
      }
    } catch (err) {
      console.error("Inference error:", err);
    }

    const latency = performance.now() - t0;
    const fpsData = fpsRef.current;
    fpsData.frames++;
    const elapsed = performance.now() - fpsData.last;
    let fps = stats.fps;
    if (elapsed >= 1000) {
      fps = Math.round((fpsData.frames * 1000) / elapsed);
      fpsData.frames = 0;
      fpsData.last = performance.now();
    }
    setStats({ fps, latency: Math.round(latency), count: dets.length });
    setDetections(dets);
    setHasResult(true);
    setIsInferring(false);
    return { dets, latency };
  }, [task, compareTask, compareMode, confidence, stats.fps]);

  const runImageInference = useCallback(async () => {
    if (!imgRef.current?.src || modelStatus !== "ready") return;
    setIsRunning(true);
    await runInference(imgRef.current);
    setIsRunning(false);
  }, [runInference, modelStatus]);

  const inferenceLoop = useCallback(async () => {
    if (!videoRef.current) return;
    await runInference(videoRef.current);
    rafRef.current = requestAnimationFrame(inferenceLoop);
  }, [runInference]);

  const startVideoInference = useCallback(() => {
    if (isRunning) return;
    setIsRunning(true);
    fpsRef.current = { frames: 0, last: performance.now() };
    rafRef.current = requestAnimationFrame(inferenceLoop);
  }, [isRunning, inferenceLoop]);

  const stopInference = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    setIsRunning(false);
    setIsInferring(false);
    if (inputType === "webcam" && videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
  }, [inputType]);

  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => { videoRef.current.play(); startVideoInference(); };
    } catch (e) { alert("Cannot access webcam: " + e.message); }
  }, [startVideoInference]);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    imgRef.current.onload = () => {
      const canvas = canvasRef.current;
      canvas.width = imgRef.current.naturalWidth;
      canvas.height = imgRef.current.naturalHeight;
      canvas.getContext("2d").drawImage(imgRef.current, 0, 0);
      if (compareMode && compareCanvasRef.current) {
        compareCanvasRef.current.width = imgRef.current.naturalWidth;
        compareCanvasRef.current.height = imgRef.current.naturalHeight;
        compareCanvasRef.current.getContext("2d").drawImage(imgRef.current, 0, 0);
      }
      setHasInput(true);
      setHasResult(false);
      setDetections([]);
    };
    imgRef.current.src = url;
  };

  const handleVideoUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    videoRef.current.src = URL.createObjectURL(file);
    videoRef.current.onloadedmetadata = () => setHasInput(true);
  };

  const saveRun = useCallback(async () => {
    if (!hasResult) return;
    setSaveStatus("saving");
    try {
      const thumbnail = canvasRef.current?.toDataURL("image/jpeg", 0.4) || null;
      await axios.post(`${API}/runs`, {
        project_id: selectedProject || null, task,
        model_name: TASKS.find((t2) => t2.id === task)?.label + (compareMode ? ` vs ${TASKS.find((t2) => t2.id === compareTask)?.label}` : " (Built-in)"),
        source_type: inputType, results_count: detections.length, detections: detections.slice(0, 50), stats, thumbnail,
      });
      setSaveStatus("saved");
      setTimeout(() => setSaveStatus(""), 2000);
    } catch { setSaveStatus("error"); setTimeout(() => setSaveStatus(""), 2000); }
  }, [task, compareTask, compareMode, inputType, detections, stats, selectedProject, hasResult]);

  const exportImage = () => { const a = document.createElement("a"); a.download = `yolo26-${task}-${Date.now()}.png`; a.href = canvasRef.current?.toDataURL("image/png"); a.click(); };
  const exportJSON = () => {
    const data = JSON.stringify({ task, compareTask: compareMode ? compareTask : undefined, stats, compareStats: compareMode ? compareStats : undefined, detections, compareDetections: compareMode ? compareDetections : undefined, timestamp: new Date().toISOString() }, null, 2);
    const blob = new Blob([data], { type: "application/json" });
    const a = document.createElement("a"); a.download = `yolo26-${task}-${Date.now()}.json`; a.href = URL.createObjectURL(blob); a.click();
  };

  const currentTask = TASKS.find((t2) => t2.id === task);
  const compareTaskMeta = TASKS.find((t2) => t2.id === compareTask);
  const CompareIcon = compareTaskMeta?.icon;

  return (
    <div className="flex flex-col h-[calc(100vh-56px)] bg-[#F4F3EE]">
      {/* Studio top bar */}
      <div className="h-10 border-b border-[#DDD9D0] flex items-center px-4 gap-3 flex-shrink-0 bg-white">
        <span className="text-[10px] font-mono text-[#B1ADA1] tracking-[0.2em] uppercase">Studio</span>
        <div className="w-px h-4 bg-[#DDD9D0]" />
        <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-sm border" style={{ borderColor: currentTask?.color + "40", backgroundColor: currentTask?.color + "10" }}>
          <currentTask.icon size={11} style={{ color: currentTask?.color }} />
          <span className="text-[11px] font-mono tracking-wider" style={{ color: currentTask?.color }}>{currentTask?.shortLabel}</span>
        </div>
        {/* Compare mode toggle */}
        <button
          data-testid="compare-mode-btn"
          onClick={() => setCompareMode(!compareMode)}
          className={`flex items-center gap-1.5 px-2.5 py-1 text-[11px] rounded-sm border transition-colors ${
            compareMode ? "bg-[#C15F3C] text-white border-[#C15F3C]" : "border-[#DDD9D0] text-[#8A8580] hover:text-[#141413] hover:border-[#B1ADA1]"
          }`}
        >
          <Columns size={11} />
          Compare
        </button>
        <div className="ml-auto flex items-center gap-2">
          <select data-testid="project-selector" value={selectedProject} onChange={(e) => setSelectedProject(e.target.value)}
            className="bg-transparent border border-[#DDD9D0] rounded-sm text-[11px] text-[#8A8580] px-2 py-1 font-mono focus:outline-none focus:border-[#B1ADA1]">
            <option value="">No Project</option>
            {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
          <button data-testid="save-run-btn" onClick={saveRun} disabled={!hasResult}
            className="flex items-center gap-1.5 px-3 py-1 text-[11px] border border-[#DDD9D0] text-[#8A8580] hover:border-[#B1ADA1] hover:text-[#141413] rounded-sm transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
            <FloppyDisk size={12} />
            {saveStatus === "saving" ? "Saving…" : saveStatus === "saved" ? "Saved!" : "Save Run"}
          </button>
          <button data-testid="export-image-btn" onClick={exportImage} disabled={!hasResult}
            className="flex items-center gap-1.5 px-3 py-1 text-[11px] border border-[#DDD9D0] text-[#8A8580] hover:border-[#B1ADA1] hover:text-[#141413] rounded-sm transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
            <Export size={12} />Export
          </button>
        </div>
      </div>

      {/* Progress bar */}
      <div className="h-0.5 bg-[#EDE9E0] overflow-hidden relative flex-shrink-0">
        {isInferring && (
          <div className="absolute top-0 h-full w-1/3 bg-gradient-to-r from-transparent via-[#C15F3C] to-transparent progress-scan" />
        )}
        {modelStatus === "loading" && (
          <div className="absolute top-0 h-full w-1/2 bg-gradient-to-r from-transparent via-[#B1ADA1] to-transparent progress-scan" />
        )}
      </div>

      {/* Main workspace */}
      <div className="flex-1 grid lg:grid-cols-12 overflow-hidden">
        {/* Sidebar */}
        <aside className="lg:col-span-3 border-r border-[#DDD9D0] flex flex-col overflow-y-auto bg-[#F0EFE9]">
          {/* Task selector */}
          <div className="p-4 border-b border-[#DDD9D0]">
            <p className="text-[10px] font-mono text-[#B1ADA1] tracking-[0.2em] uppercase mb-3">CV Task</p>
            <div className="space-y-1">
              {TASKS.map((t2) => (
                <button key={t2.id} data-testid={`task-btn-${t2.id}`} onClick={() => handleTaskChange(t2.id)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-sm border transition-all duration-100 ${
                    task === t2.id ? "border-[#DDD9D0] bg-white shadow-sm" : "border-transparent hover:bg-white/60 hover:border-[#DDD9D0]"
                  }`}>
                  <t2.icon size={14} style={{ color: t2.color }} weight={task === t2.id ? "bold" : "regular"} />
                  <span className="text-sm text-[#141413]">{t2.label}</span>
                  {task === t2.id && <div className="ml-auto w-1 h-3 rounded-full" style={{ backgroundColor: t2.color }} />}
                </button>
              ))}
            </div>
          </div>

          {/* Model variant selector for LEFT panel */}
          <div className="p-4 border-b border-[#DDD9D0]">
            <p className="text-[10px] font-mono text-[#B1ADA1] tracking-[0.2em] uppercase mb-3">
              {compareMode ? "Left Model" : "Model"}
            </p>
            <div className="space-y-1">
              {(MODELS_PER_TASK[task] || []).map((mv) => (
                <button
                  key={mv.id}
                  data-testid={`model-variant-${mv.id}`}
                  onClick={() => setModelVariant(mv.id)}
                  className={`w-full flex items-start gap-2 px-3 py-2 rounded-sm border transition-all duration-100 text-left ${
                    modelVariant === mv.id ? "border-[#DDD9D0] bg-white shadow-sm" : "border-transparent hover:bg-white/60"
                  }`}
                >
                  <div className={`mt-1 w-1.5 h-1.5 rounded-full flex-shrink-0 ${modelVariant === mv.id ? "" : "opacity-40"}`} style={{ backgroundColor: currentTask?.color }} />
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-medium text-[#141413] truncate">{mv.label}</div>
                    <div className="text-[10px] text-[#B1ADA1] font-mono truncate">{mv.sub}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Compare task + model selector (shown when compare mode on) */}
          {compareMode && (
            <div className="p-4 border-b border-[#DDD9D0] bg-[#C15F3C]/5">
              <div className="flex items-center gap-2 mb-3">
                <Columns size={11} className="text-[#C15F3C]" />
                <p className="text-[10px] font-mono text-[#C15F3C] tracking-[0.2em] uppercase">Right Panel</p>
              </div>
              {/* Compare task picker (all 5 tasks — same task = same-task compare) */}
              <div className="space-y-1 mb-3">
                {TASKS.map((t2) => (
                  <button key={t2.id} data-testid={`compare-task-btn-${t2.id}`} onClick={() => handleCompareTaskChange(t2.id)}
                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-sm border transition-all duration-100 ${
                      compareTask === t2.id ? "border-[#DDD9D0] bg-white shadow-sm" : "border-transparent hover:bg-white/60"
                    }`}>
                    <t2.icon size={13} style={{ color: t2.color }} />
                    <span className="text-xs text-[#141413]">{t2.label}</span>
                    {compareTask === t2.id && <div className="ml-auto w-1 h-3 rounded-full" style={{ backgroundColor: t2.color }} />}
                  </button>
                ))}
              </div>
              {/* Compare model variant picker */}
              <p className="text-[10px] font-mono text-[#C15F3C]/70 tracking-[0.15em] uppercase mb-2">Right Model</p>
              <div className="space-y-1">
                {(MODELS_PER_TASK[compareTask] || []).map((mv) => (
                  <button
                    key={mv.id}
                    data-testid={`compare-model-variant-${mv.id}`}
                    onClick={() => setCompareModelVariant(mv.id)}
                    className={`w-full flex items-start gap-2 px-3 py-2 rounded-sm border transition-all duration-100 text-left ${
                      compareModelVariant === mv.id ? "border-[#DDD9D0] bg-white shadow-sm" : "border-transparent hover:bg-white/60"
                    }`}
                  >
                    <div className={`mt-1 w-1.5 h-1.5 rounded-full flex-shrink-0 ${compareModelVariant === mv.id ? "" : "opacity-40"}`} style={{ backgroundColor: compareTaskMeta?.color }} />
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-medium text-[#141413] truncate">{mv.label}</div>
                      <div className="text-[10px] text-[#B1ADA1] font-mono truncate">{mv.sub}</div>
                    </div>
                  </button>
                ))}
              </div>
              <div className="mt-3 flex items-center gap-1.5">
                {compareStatus === "loading" && <SpinnerGap size={11} className="animate-spin text-[#C15F3C]" />}
                {compareStatus === "ready" && <div className="w-1.5 h-1.5 rounded-full bg-[#008B22] stat-live" />}
                <span className="text-[10px] font-mono text-[#8A8580]">{compareStatus}</span>
              </div>
            </div>
          )}

          {/* Input source */}
          <div className="p-4 border-b border-[#DDD9D0]">
            <p className="text-[10px] font-mono text-[#B1ADA1] tracking-[0.2em] uppercase mb-3">Input Source</p>
            <div className="flex gap-1 mb-3">
              {[{ id: "image", icon: ImageIcon, label: "Image" }, { id: "video", icon: VideoCamera, label: "Video" }, { id: "webcam", icon: Camera, label: "Webcam" }].map(({ id, icon: Icon, label }) => (
                <button key={id} data-testid={`input-type-${id}`} onClick={() => { setInputType(id); clearInput(); }}
                  className={`flex-1 flex flex-col items-center gap-1 py-2 rounded-sm border text-xs transition-all duration-100 ${
                    inputType === id ? "border-[#B1ADA1] bg-white text-[#141413] shadow-sm" : "border-[#DDD9D0] text-[#8A8580] hover:text-[#141413] hover:border-[#B1ADA1]"
                  }`}>
                  <Icon size={14} />
                  {label}
                </button>
              ))}
            </div>

            {inputType === "image" && (
              <div className="space-y-2">
                <label data-testid="image-upload-zone"
                  className="flex flex-col items-center gap-2 p-4 border border-dashed border-[#C4BFB5] rounded-sm cursor-pointer hover:border-[#B1ADA1] hover:bg-white transition-colors">
                  <UploadSimple size={20} className="text-[#B1ADA1]" />
                  <span className="text-xs text-[#8A8580]">Upload Image</span>
                  <span className="text-[10px] text-[#B1ADA1]">JPG, PNG, WebP</span>
                  <input type="file" accept="image/*" className="hidden" onChange={handleImageUpload} />
                </label>
                {hasInput && (
                  <button onClick={clearInput} data-testid="clear-input-btn"
                    className="w-full flex items-center justify-center gap-1.5 py-1.5 text-xs text-[#CC1144] border border-[#CC1144]/20 rounded-sm hover:bg-[#CC1144]/5 transition-colors">
                    <Trash size={11} /> Remove Image
                  </button>
                )}
              </div>
            )}
            {inputType === "video" && (
              <div className="space-y-2">
                <label data-testid="video-upload-zone"
                  className="flex flex-col items-center gap-2 p-4 border border-dashed border-[#C4BFB5] rounded-sm cursor-pointer hover:border-[#B1ADA1] hover:bg-white transition-colors">
                  <UploadSimple size={20} className="text-[#B1ADA1]" />
                  <span className="text-xs text-[#8A8580]">Upload Video</span>
                  <span className="text-[10px] text-[#B1ADA1]">MP4, WebM, MOV</span>
                  <input type="file" accept="video/*" className="hidden" onChange={handleVideoUpload} />
                </label>
                {hasInput && (
                  <button onClick={clearInput} data-testid="clear-video-btn"
                    className="w-full flex items-center justify-center gap-1.5 py-1.5 text-xs text-[#CC1144] border border-[#CC1144]/20 rounded-sm hover:bg-[#CC1144]/5 transition-colors">
                    <Trash size={11} /> Remove Video
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Confidence */}
          <div className="p-4 border-b border-[#DDD9D0]">
            <div className="flex items-center gap-2 mb-3">
              <Sliders size={12} className="text-[#B1ADA1]" />
              <p className="text-[10px] font-mono text-[#B1ADA1] tracking-[0.2em] uppercase">Confidence</p>
              <span className="ml-auto font-mono text-xs text-[#141413]">{(confidence * 100).toFixed(0)}%</span>
            </div>
            <input type="range" min="0.1" max="0.95" step="0.05" value={confidence}
              onChange={(e) => setConfidence(parseFloat(e.target.value))} data-testid="confidence-slider"
              className="w-full h-1 bg-[#DDD9D0] rounded-full appearance-none cursor-pointer"
              style={{ accentColor: "#C15F3C" }} />
          </div>

          {/* Model status */}
          <div className="p-4 border-b border-[#DDD9D0]">
            <p className="text-[10px] font-mono text-[#B1ADA1] tracking-[0.2em] uppercase mb-2">Model Status</p>
            <div className="flex items-center gap-2">
              {modelStatus === "loading" && <SpinnerGap size={12} className="animate-spin text-[#B08000]" />}
              {modelStatus === "ready" && <div className="w-2 h-2 rounded-full bg-[#008B22] stat-live" />}
              {modelStatus === "error" && <Warning size={12} className="text-[#CC1144]" />}
              {modelStatus === "idle" && <div className="w-2 h-2 rounded-full bg-[#B1ADA1]" />}
              <span className="text-xs font-mono text-[#5C5751] capitalize">{modelStatus}</span>
            </div>
            <div className="mt-2 text-[11px] text-[#B1ADA1] font-mono truncate">
              {(MODELS_PER_TASK[task] || []).find((mv) => mv.id === modelVariant)?.label || task}
            </div>
          </div>

          {/* Run controls */}
          <div className="p-4 flex-1 flex flex-col gap-2">
            {inputType === "image" && (
              <button data-testid="run-inference-btn" onClick={runImageInference}
                disabled={modelStatus !== "ready" || isRunning || !hasInput || (compareMode && compareStatus !== "ready")}
                className="w-full flex items-center justify-center gap-2 py-2.5 bg-[#141413] text-[#F4F3EE] text-sm font-medium rounded-sm hover:bg-[#2A2925] transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
                {isRunning ? <SpinnerGap size={14} className="animate-spin" /> : <Play size={14} weight="fill" />}
                {isRunning ? "Running…" : compareMode ? "Compare" : "Run Inference"}
              </button>
            )}
            {inputType === "video" && (
              <button data-testid="run-video-btn" onClick={isRunning ? stopInference : startVideoInference}
                disabled={modelStatus !== "ready" || (compareMode && compareStatus !== "ready")}
                className={`w-full flex items-center justify-center gap-2 py-2.5 text-sm font-medium rounded-sm transition-colors disabled:opacity-50 ${
                  isRunning ? "bg-[#CC1144]/10 border border-[#CC1144]/30 text-[#CC1144] hover:bg-[#CC1144]/15" : "bg-[#141413] text-[#F4F3EE] hover:bg-[#2A2925]"
                }`}>
                {isRunning ? <Stop size={14} weight="fill" /> : <Play size={14} weight="fill" />}
                {isRunning ? "Stop" : "Start Inference"}
              </button>
            )}
            {inputType === "webcam" && (
              <button data-testid="run-webcam-btn" onClick={isRunning ? stopInference : startWebcam}
                disabled={modelStatus !== "ready" || (compareMode && compareStatus !== "ready")}
                className={`w-full flex items-center justify-center gap-2 py-2.5 text-sm font-medium rounded-sm transition-colors disabled:opacity-50 ${
                  isRunning ? "bg-[#CC1144]/10 border border-[#CC1144]/30 text-[#CC1144]" : "bg-[#141413] text-[#F4F3EE] hover:bg-[#2A2925]"
                }`}>
                {isRunning ? <Stop size={14} weight="fill" /> : <Camera size={14} />}
                {isRunning ? "Stop Webcam" : "Start Webcam"}
              </button>
            )}
            {hasResult && (
              <button data-testid="export-json-btn" onClick={exportJSON}
                className="w-full flex items-center justify-center gap-2 py-2 text-xs border border-[#DDD9D0] text-[#8A8580] rounded-sm hover:border-[#B1ADA1] hover:text-[#141413] transition-colors">
                <Export size={12} />Export JSON
              </button>
            )}
          </div>
        </aside>

        {/* Canvas area */}
        <div className="lg:col-span-9 flex flex-col overflow-hidden">
          <img ref={imgRef} className="hidden" alt="" />
          <video ref={videoRef} className="hidden" muted playsInline />

          {/* Canvas(es) */}
          <div className={`flex-1 overflow-hidden ${compareMode ? "grid grid-cols-2" : ""}`} data-testid="canvas-area">
            {/* Left canvas */}
            <div className={`canvas-container relative ${compareMode ? "border-r border-[#DDD9D0]" : ""}`}>
              {compareMode && (
                <div className="compare-panel-label">
                  <div className="flex items-center gap-1.5 px-2 py-1 bg-white/90 border border-[#DDD9D0] rounded-sm shadow-sm">
                    <currentTask.icon size={11} style={{ color: currentTask?.color }} weight="bold" />
                    <span className="text-[10px] font-mono font-medium" style={{ color: currentTask?.color }}>{currentTask?.label}</span>
                    <span className="text-[10px] font-mono text-[#B1ADA1]">·</span>
                    <span className="text-[10px] font-mono text-[#5C5751] truncate max-w-[160px]">
                      {(MODELS_PER_TASK[task] || []).find((mv) => mv.id === modelVariant)?.label || ""}
                    </span>
                  </div>
                </div>
              )}
              <canvas ref={canvasRef} data-testid="inference-canvas" className="inference-canvas" />
              {!hasResult && modelStatus === "ready" && (
                <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                  <currentTask.icon size={48} style={{ color: currentTask?.color + "25" }} />
                  <p className="mt-4 text-[#B1ADA1] text-sm font-mono">
                    {inputType === "webcam" ? "Click 'Start Webcam'" : `Upload ${inputType} to begin`}
                  </p>
                </div>
              )}
              {modelStatus === "loading" && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#F4F3EE]/80">
                  <SpinnerGap size={28} className="animate-spin text-[#B1ADA1]" />
                  <p className="mt-3 text-xs font-mono text-[#8A8580]">Loading {currentTask?.label} model…</p>
                </div>
              )}
              {/* BIG inference spinner overlay */}
              {isInferring && modelStatus === "ready" && (
                <div data-testid="inference-overlay" className="absolute inset-0 flex flex-col items-center justify-center bg-[#141413]/55 backdrop-blur-[2px] pointer-events-none z-20">
                  <div className="relative flex items-center justify-center">
                    <div className="absolute w-20 h-20 rounded-full border-2 animate-ping" style={{ borderColor: currentTask?.color + "55" }} />
                    <div className="relative w-14 h-14 rounded-full border-2 flex items-center justify-center" style={{ borderColor: currentTask?.color, borderTopColor: "transparent", animation: "spin 0.8s linear infinite" }}>
                      <currentTask.icon size={20} style={{ color: currentTask?.color }} weight="bold" />
                    </div>
                  </div>
                  <p className="mt-5 text-xs font-mono uppercase tracking-[0.25em] text-white/90">Running {currentTask?.label}</p>
                  <p className="mt-1 text-[10px] font-mono text-white/50">{(MODELS_PER_TASK[task] || []).find((mv) => mv.id === modelVariant)?.label || ""}</p>
                </div>
              )}
              {/* Stats overlay for left canvas */}
              {(isRunning || hasResult) && (
                <div data-testid="stats-overlay" className="absolute top-3 right-3 flex gap-2 pointer-events-none">
                  {[{ label: "FPS", val: stats.fps, color: "#008B22" }, { label: "ms", val: stats.latency, color: "#0087B3" }, { label: "Obj", val: stats.count, color: "#B08000" }].map((s) => (
                    <div key={s.label} className="bg-[#141413]/80 border border-[#141413]/20 px-2 py-1.5 rounded-sm">
                      <div className="text-[9px] text-white/50 font-mono uppercase tracking-wider">{s.label}</div>
                      <div className="text-sm font-mono font-medium tabular-nums" style={{ color: s.color }}>{s.val}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Right compare canvas */}
            {compareMode && (
              <div className="canvas-container relative">
                <div className="compare-panel-label">
                  <div className="flex items-center gap-1.5 px-2 py-1 bg-white/90 border border-[#DDD9D0] rounded-sm shadow-sm">
                    {CompareIcon && <CompareIcon size={11} style={{ color: compareTaskMeta?.color }} weight="bold" />}
                    <span className="text-[10px] font-mono font-medium" style={{ color: compareTaskMeta?.color }}>{compareTaskMeta?.label}</span>
                    <span className="text-[10px] font-mono text-[#B1ADA1]">·</span>
                    <span className="text-[10px] font-mono text-[#5C5751] truncate max-w-[160px]">
                      {(MODELS_PER_TASK[compareTask] || []).find((mv) => mv.id === compareModelVariant)?.label || ""}
                    </span>
                  </div>
                </div>
                <canvas ref={compareCanvasRef} data-testid="compare-canvas" className="inference-canvas" />
                {!hasResult && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                    {CompareIcon && <CompareIcon size={48} style={{ color: (compareTaskMeta?.color || "#B1ADA1") + "25" }} />}
                    <p className="mt-4 text-[#B1ADA1] text-sm font-mono">Compare panel</p>
                  </div>
                )}
                {compareStatus === "loading" && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#F4F3EE]/80">
                    <SpinnerGap size={24} className="animate-spin text-[#C15F3C]" />
                    <p className="mt-2 text-xs font-mono text-[#8A8580]">Loading {compareTaskMeta?.label}…</p>
                  </div>
                )}
                {/* BIG inference spinner overlay (compare) */}
                {isInferring && compareStatus === "ready" && (
                  <div data-testid="compare-inference-overlay" className="absolute inset-0 flex flex-col items-center justify-center bg-[#141413]/55 backdrop-blur-[2px] pointer-events-none z-20">
                    <div className="relative flex items-center justify-center">
                      <div className="absolute w-20 h-20 rounded-full border-2 animate-ping" style={{ borderColor: (compareTaskMeta?.color || "#C15F3C") + "55" }} />
                      <div className="relative w-14 h-14 rounded-full border-2 flex items-center justify-center" style={{ borderColor: compareTaskMeta?.color || "#C15F3C", borderTopColor: "transparent", animation: "spin 0.8s linear infinite" }}>
                        {CompareIcon && <CompareIcon size={20} style={{ color: compareTaskMeta?.color }} weight="bold" />}
                      </div>
                    </div>
                    <p className="mt-5 text-xs font-mono uppercase tracking-[0.25em] text-white/90">Running {compareTaskMeta?.label}</p>
                    <p className="mt-1 text-[10px] font-mono text-white/50">{(MODELS_PER_TASK[compareTask] || []).find((mv) => mv.id === compareModelVariant)?.label || ""}</p>
                  </div>
                )}
                {(isRunning || hasResult) && (
                  <div className="absolute top-3 right-3 flex gap-2 pointer-events-none">
                    {[{ label: "FPS", val: compareStats.fps, color: "#CC1144" }, { label: "ms", val: compareStats.latency, color: "#7B1CC4" }, { label: "Obj", val: compareStats.count, color: "#C15F3C" }].map((s) => (
                      <div key={s.label} className="bg-[#141413]/80 border border-[#141413]/20 px-2 py-1.5 rounded-sm">
                        <div className="text-[9px] text-white/50 font-mono uppercase tracking-wider">{s.label}</div>
                        <div className="text-sm font-mono font-medium tabular-nums" style={{ color: s.color }}>{s.val}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Results panel */}
          <div className={`border-t border-[#DDD9D0] flex flex-col ${compareMode ? "h-52" : "h-44"}`}>
            <div className="px-4 py-2 border-b border-[#DDD9D0] flex items-center gap-3 bg-white flex-shrink-0">
              <span className="text-[10px] font-mono text-[#B1ADA1] tracking-[0.2em] uppercase">Results</span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded-sm" style={{ color: currentTask?.color, backgroundColor: currentTask?.color + "15" }}>
                {detections.length} objects
              </span>
              {compareMode && (
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded-sm" style={{ color: compareTaskMeta?.color, backgroundColor: compareTaskMeta?.color + "15" }}>
                  {compareDetections.length} (compare)
                </span>
              )}
              {hasResult && !isRunning && <span className="text-[10px] font-mono text-[#B1ADA1]">Latency: {stats.latency}ms</span>}
              {compareMode && hasResult && !isRunning && <span className="text-[10px] font-mono" style={{ color: compareTaskMeta?.color }}>Compare: {compareStats.latency}ms</span>}
            </div>
            <div className="flex-1 overflow-auto">
              {detections.length > 0 || compareDetections.length > 0 ? (
                <div className={compareMode ? "grid grid-cols-2 divide-x divide-[#DDD9D0] h-full" : "h-full"}>
                  {/* Left detections */}
                  <table className="w-full text-xs" data-testid="detections-table">
                    <thead className="sticky top-0 bg-white">
                      <tr className="border-b border-[#DDD9D0]">
                        <th className="px-4 py-1.5 text-left font-mono text-[10px] text-[#B1ADA1] uppercase tracking-wider w-8">#</th>
                        <th className="px-4 py-1.5 text-left font-mono text-[10px] text-[#B1ADA1] uppercase tracking-wider">Class</th>
                        <th className="px-4 py-1.5 text-left font-mono text-[10px] text-[#B1ADA1] uppercase tracking-wider">Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detections.slice(0, 15).map((det, i) => (
                        <tr key={i} className="border-b border-[#F4F3EE] hover:bg-[#F4F3EE] transition-colors">
                          <td className="px-4 py-1.5 font-mono text-[#B1ADA1]">{i + 1}</td>
                          <td className="px-4 py-1.5 font-mono text-[#141413]">{det.class || det.type || "—"}</td>
                          <td className="px-4 py-1.5">
                            <div className="flex items-center gap-2">
                              <div className="w-12 h-1.5 bg-[#EDE9E0] rounded-full overflow-hidden">
                                <div className="h-full rounded-full" style={{ width: `${(det.score || 0) * 100}%`, backgroundColor: currentTask?.color }} />
                              </div>
                              <span className="font-mono text-[#5C5751] tabular-nums">{((det.score || 0) * 100).toFixed(1)}%</span>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {/* Right compare detections */}
                  {compareMode && (
                    <table className="w-full text-xs">
                      <thead className="sticky top-0 bg-white">
                        <tr className="border-b border-[#DDD9D0]">
                          <th className="px-4 py-1.5 text-left font-mono text-[10px] text-[#B1ADA1] uppercase tracking-wider w-8">#</th>
                          <th className="px-4 py-1.5 text-left font-mono text-[10px] text-[#B1ADA1] uppercase tracking-wider">Class</th>
                          <th className="px-4 py-1.5 text-left font-mono text-[10px] text-[#B1ADA1] uppercase tracking-wider">Score</th>
                        </tr>
                      </thead>
                      <tbody>
                        {compareDetections.slice(0, 15).map((det, i) => (
                          <tr key={i} className="border-b border-[#F4F3EE] hover:bg-[#F4F3EE] transition-colors">
                            <td className="px-4 py-1.5 font-mono text-[#B1ADA1]">{i + 1}</td>
                            <td className="px-4 py-1.5 font-mono text-[#141413]">{det.class || det.type || "—"}</td>
                            <td className="px-4 py-1.5">
                              <div className="flex items-center gap-2">
                                <div className="w-12 h-1.5 bg-[#EDE9E0] rounded-full overflow-hidden">
                                  <div className="h-full rounded-full" style={{ width: `${(det.score || 0) * 100}%`, backgroundColor: compareTaskMeta?.color }} />
                                </div>
                                <span className="font-mono text-[#5C5751] tabular-nums">{((det.score || 0) * 100).toFixed(1)}%</span>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              ) : (
                <div className="flex items-center justify-center h-full"><span className="text-xs text-[#B1ADA1] font-mono">No detections yet</span></div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
