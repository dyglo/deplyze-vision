/**
 * Deplyze Vision — Studio
 * Real YOLO TF.js inference: detection · segmentation · pose · OBB · tracking · classification
 */
import {
  useState, useEffect, useRef, useCallback, useReducer,
} from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import axios from "axios";
import {
  BoundingBox, PersonSimpleRun, Intersect, Tag, Path,
  Cube, UploadSimple, Camera, FilmSlate, Play, Stop, Columns,
  Faders, FloppyDisk, Warning, DownloadSimple, Spinner, ArrowsClockwise,
  Lightning, Timer, Crosshair,
} from "@phosphor-icons/react";

import { YOLO_MODELS, TASK_META, getModelById } from "../utils/yoloModels";
import { loadModel, runYOLOInference, initTFBackend, getTFBackendName } from "../utils/yoloInference";
import { drawResults, drawDetections, drawTracking } from "../utils/drawUtils";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// ─── Constants ────────────────────────────────────────────────────────────────
const SOURCE_TYPES = [
  { id: "image",  icon: UploadSimple, label: "Image"  },
  { id: "video",  icon: FilmSlate,   label: "Video"  },
  { id: "webcam", icon: Camera,      label: "Webcam" },
];

const TASK_ICONS = {
  detect:   BoundingBox,
  seg:      Intersect,
  pose:     PersonSimpleRun,
  obb:      Crosshair,
  classify: Tag,
  track:    Path,
};

// ─── Centroid tracker (pure-JS, YOLO output = normalised bbox) ───────────────
function createTracker() {
  let nextId = 1;
  let objects = {}; // id → { cx, cy, det, age }
  return {
    update(detections) {
      const active = {};
      const usedIds = new Set();
      for (const det of detections) {
        const [x1, y1, x2, y2] = det.bbox;
        const cx = (x1 + x2) / 2;
        const cy = (y1 + y2) / 2;
        let bestId = null;
        let bestDist = 0.05; // max distance threshold (normalised)
        for (const [id, obj] of Object.entries(objects)) {
          if (usedIds.has(id)) continue;
          const d = Math.hypot(cx - obj.cx, cy - obj.cy);
          if (d < bestDist) { bestDist = d; bestId = id; }
        }
        const assignedId = bestId ?? String(nextId++);
        usedIds.add(assignedId);
        active[assignedId] = { cx, cy, det, age: 0 };
      }
      objects = active;
      return objects;
    },
    reset() { objects = {}; nextId = 1; },
  };
}

// ─── State reducer ────────────────────────────────────────────────────────────
function studioReducer(state, action) {
  switch (action.type) {
    case "SET_SOURCE":    return { ...state, sourceType: action.v, mediaUrl: null, results: [], runStats: null, inferenceDebug: null, trackedObjects: {} };
    case "SET_TASK":      return { ...state, task: action.v, results: [], runStats: null, inferenceDebug: null, trackedObjects: {} };
    case "SET_MODEL":     return { ...state, modelId: action.v, loadedModel: null, results: [], runStats: null, inferenceDebug: null };
    case "SET_MODEL_OBJ": return { ...state, loadedModel: action.v };
    case "SET_MEDIA":     return { ...state, mediaUrl: action.v, results: [], runStats: null, inferenceDebug: null };
    case "SET_RESULTS":   return { ...state, results: action.v, runStats: action.stats, inferenceDebug: action.debug || null };
    case "SET_TRACKED":   return { ...state, trackedObjects: action.v };
    case "SET_LOADING":   return { ...state, loading: action.v };
    case "SET_RUNNING":   return { ...state, running: action.v };
    case "SET_CONF":      return { ...state, confidence: action.v };
    case "SET_IOU":       return { ...state, iouThreshold: action.v };
    case "SET_ERROR":     return { ...state, error: action.v };
    case "CLEAR_ERROR":   return { ...state, error: null };
    case "SET_COMPARE":   return { ...state, compareMode: action.v };
    case "SET_COMPARE_MODEL": return { ...state, compareModelId: action.v, compareLoadedModel: null };
    case "SET_COMPARE_OBJ":   return { ...state, compareLoadedModel: action.v };
    case "SET_COMPARE_RES":   return { ...state, compareResults: action.v, compareStats: action.stats };
    case "SET_SAVING":    return { ...state, saving: action.v };
    default: return state;
  }
}

const INITIAL_STATE = {
  sourceType:          "image",
  task:                "detect",
  modelId:             null,
  loadedModel:         null,
  mediaUrl:            null,
  results:             [],
  runStats:            null,
  inferenceDebug:      null,
  trackedObjects:      {},
  loading:             false,
  running:             false,
  confidence:          0.25,
  iouThreshold:        0.45,
  error:               null,
  compareMode:         false,
  compareModelId:      null,
  compareLoadedModel:  null,
  compareResults:      [],
  compareStats:        null,
  saving:              false,
};

// ─── Studio component ─────────────────────────────────────────────────────────
export default function Studio() {
  const [params]  = useSearchParams();
  const navigate  = useNavigate();
  const initialTask = TASK_META[params.get("task")] ? params.get("task") : INITIAL_STATE.task;
  const [st, dispatch] = useReducer(studioReducer, {
    ...INITIAL_STATE,
    task: initialTask,
    modelId: params.get("modelId") || INITIAL_STATE.modelId,
    compareMode: params.get("compare") === "true",
  });

  // DOM refs
  const imageRef     = useRef(null);
  const videoRef     = useRef(null);
  const webcamRef    = useRef(null);
  const canvasRef    = useRef(null);
  const cmpCanvasRef = useRef(null);
  const animRef      = useRef(null);
  const trackerRef   = useRef(createTracker());
  const fileInputRef = useRef(null);

  // Backend status
  const [backendName, setBackendName] = useState("—");
  const [customModels, setCustomModels] = useState([]);

  // ── Fetch custom models from backend ──────────────────────────────────────
  useEffect(() => {
    initTFBackend()
      .then((b) => setBackendName(b))
      .catch(() => {});

    axios.get(`${API}/model-configs`).then((r) => {
      setCustomModels(
        r.data.filter((m) => !m.is_builtin && m.url).map((m) => ({
          id: `custom-${m.id ?? m._id}`,
          name: m.name,
          family: "Custom",
          task: m.task || "detect",
          url: m.url,
          classes: m.labels?.length
            ? m.labels
            : YOLO_MODELS.find((x) => x.task === (m.task || "detect"))?.classes || [],
          numClasses: m.num_classes || 80,
          inputSize: m.input_size || 640,
          numKeypoints: m.num_keypoints || 17,
          metadataUrl: m.metadata_url || null,
          isCustom: true,
        }))
      );
    }).catch(() => {});
  }, []);

  // Pre-select first model for active task when task changes
  const allModels = [...YOLO_MODELS, ...customModels];
  const taskModels = allModels.filter(
    (m) => m.task === st.task || (st.task === "track" && m.task === "detect")
  );

  useEffect(() => {
    if (taskModels.length && !taskModels.find((m) => m.id === st.modelId)) {
      dispatch({ type: "SET_MODEL", v: taskModels[0].id });
    }
  }, [st.task, st.modelId, taskModels]);

  // ── Model loading ─────────────────────────────────────────────────────────
  const selectedMeta = allModels.find((m) => m.id === st.modelId) || null;
  const cmpMeta      = allModels.find((m) => m.id === st.compareModelId) || null;

  const ensureModel = useCallback(async (meta, which = "primary") => {
    if (!meta?.url) return null;
    dispatch({ type: "SET_LOADING", v: true });
    dispatch({ type: "CLEAR_ERROR" });
    try {
      const model = await loadModel(meta);
      if (which === "primary") dispatch({ type: "SET_MODEL_OBJ", v: model });
      else dispatch({ type: "SET_COMPARE_OBJ", v: model });
      return model;
    } catch (e) {
      dispatch({ type: "SET_ERROR", v: `Failed to load model: ${e.message || e}` });
      return null;
    } finally {
      dispatch({ type: "SET_LOADING", v: false });
    }
  }, []);

  // Auto-load when model selection changes and URL available
  useEffect(() => {
    if (selectedMeta?.url && !st.loadedModel) ensureModel(selectedMeta, "primary");
  }, [selectedMeta, st.loadedModel, ensureModel]);

  useEffect(() => {
    if (cmpMeta?.url && !st.compareLoadedModel) ensureModel(cmpMeta, "compare");
  }, [cmpMeta, st.compareLoadedModel, ensureModel]);

  // ── Canvas sizing ─────────────────────────────────────────────────────────
  function resizeCanvas(ref, src) {
    if (!ref.current || !src) return;
    const { naturalWidth: nw, naturalHeight: nh, videoWidth: vw, videoHeight: vh } = src;
    const w = nw || vw || 640;
    const h = nh || vh || 480;
    ref.current.width  = w;
    ref.current.height = h;
  }

  // ── Run inference (single frame) ──────────────────────────────────────────
  const runFrame = useCallback(async (source, model, meta, canvasEl, onResult) => {
    if (!model || !source || !canvasEl) return null;
    resizeCanvas({ current: canvasEl }, source);
    const cw = canvasEl.width;
    const ch = canvasEl.height;

    // Draw source underneath
    const ctx = canvasEl.getContext("2d");
    ctx.drawImage(source, 0, 0, cw, ch);

    const inference = await runYOLOInference(model, source, meta, {
      confidenceThreshold: st.confidence,
      iouThreshold: st.iouThreshold,
      canvasW: cw,
      canvasH: ch,
    });

    // Tracking: update centroid tracker
    if (meta.task === "track" || st.task === "track") {
      const tracked = trackerRef.current.update(inference.results);
      drawResults(ctx, "track", tracked, cw, ch, { clear: false });
      onResult && onResult(tracked, inference.fps, inference.latency, inference.debug);
    } else {
      drawResults(ctx, meta.task, inference.results, cw, ch, { clear: false });
      onResult && onResult(inference.results, inference.fps, inference.latency, inference.debug);
    }

    return inference;
  }, [st.confidence, st.iouThreshold, st.task]);

  // ── Image inference ───────────────────────────────────────────────────────
  const runImage = useCallback(async () => {
    if (!imageRef.current) return;
    const model = st.loadedModel || await ensureModel(selectedMeta, "primary");
    if (!model || !selectedMeta) return;

    dispatch({ type: "SET_LOADING", v: true });
    try {
      await runFrame(
        imageRef.current, model, selectedMeta, canvasRef.current,
        (results, fps, latency, debug) => dispatch({ type: "SET_RESULTS", v: results, stats: { fps, latency }, debug })
      );
      if (st.compareMode) {
        const cmpModel = st.compareLoadedModel || await ensureModel(cmpMeta, "compare");
        if (cmpModel && cmpMeta) {
          await runFrame(
            imageRef.current, cmpModel, cmpMeta, cmpCanvasRef.current,
            (results, fps, latency) =>
              dispatch({ type: "SET_COMPARE_RES", v: results, stats: { fps, latency } })
          );
        }
      }
    } finally {
      dispatch({ type: "SET_LOADING", v: false });
    }
  }, [st, selectedMeta, cmpMeta, ensureModel, runFrame]);

  // ── Video / Webcam loop ────────────────────────────────────────────────────
  const startLoop = useCallback(async (sourceRef) => {
    const model = st.loadedModel || await ensureModel(selectedMeta, "primary");
    if (!model || !selectedMeta) return;

    dispatch({ type: "SET_RUNNING", v: true });
    trackerRef.current.reset();

    const loop = async () => {
      const src = sourceRef.current;
      if (!src || src.paused || src.ended) {
        dispatch({ type: "SET_RUNNING", v: false });
        return;
      }
      await runFrame(
        src, model, selectedMeta, canvasRef.current,
        (results, fps, latency, debug) => dispatch({ type: "SET_RESULTS", v: results, stats: { fps, latency }, debug })
      );
      if (st.compareMode) {
        const cmpModel = st.compareLoadedModel || await ensureModel(cmpMeta, "compare");
        if (cmpModel && cmpMeta) {
          await runFrame(src, cmpModel, cmpMeta, cmpCanvasRef.current,
            (r, fps, latency) => dispatch({ type: "SET_COMPARE_RES", v: r, stats: { fps, latency } }));
        }
      }
      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);
  }, [st, selectedMeta, cmpMeta, ensureModel, runFrame]);

  const stopLoop = useCallback(() => {
    if (animRef.current) cancelAnimationFrame(animRef.current);
    animRef.current = null;
    dispatch({ type: "SET_RUNNING", v: false });
  }, []);

  // Cleanup on unmount
  useEffect(() => () => stopLoop(), [stopLoop]);

  // ── Webcam ────────────────────────────────────────────────────────────────
  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (webcamRef.current) {
        webcamRef.current.srcObject = stream;
        webcamRef.current.onloadedmetadata = () => {
          webcamRef.current.play();
          startLoop(webcamRef);
        };
      }
    } catch (e) {
      dispatch({ type: "SET_ERROR", v: `Webcam access denied: ${e.message}` });
    }
  }, [startLoop]);

  const stopWebcam = useCallback(() => {
    stopLoop();
    if (webcamRef.current?.srcObject) {
      webcamRef.current.srcObject.getTracks().forEach((t) => t.stop());
      webcamRef.current.srcObject = null;
    }
  }, [stopLoop]);

  // ── File upload handler ────────────────────────────────────────────────────
  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    dispatch({ type: "SET_MEDIA", v: url });
  };

  // ── Save run to backend ────────────────────────────────────────────────────
  const saveRun = async () => {
    if (!st.runStats) return;
    dispatch({ type: "SET_SAVING", v: true });
    const modelName = st.compareMode && cmpMeta
      ? `${selectedMeta?.name || "?"} vs ${cmpMeta?.name || "?"}`
      : selectedMeta?.name || "custom";
    try {
      await axios.post(`${API}/runs`, {
        task: st.task,
        source_type: st.sourceType,
        model_name: modelName,
        results_count: st.results.length,
        stats: { fps: st.runStats.fps, latency: st.runStats.latency },
      });
    } finally {
      dispatch({ type: "SET_SAVING", v: false });
    }
  };

  // ── Export canvas as PNG ───────────────────────────────────────────────────
  const exportPNG = () => {
    if (!canvasRef.current) return;
    const a = document.createElement("a");
    a.download = `deplyze-${st.task}-${Date.now()}.png`;
    a.href = canvasRef.current.toDataURL("image/png");
    a.click();
  };

  // ─── Render ───────────────────────────────────────────────────────────────
  const noModel = !selectedMeta?.url;
  const canRun = !noModel && (
    (st.sourceType === "image" && st.mediaUrl) ||
    (st.sourceType === "video" && st.mediaUrl) ||
    (st.sourceType === "webcam")
  );

  return (
    <div className="flex h-full relative" style={{ height: "calc(100vh - 56px)" }}>
      {/* Background grid + glow */}
      <div
        className="absolute inset-0 opacity-[0.03] pointer-events-none"
        style={{
          backgroundImage: "linear-gradient(#141413 1px, transparent 1px), linear-gradient(90deg, #141413 1px, transparent 1px)",
          backgroundSize: "32px 32px",
        }}
      />
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[600px] h-[300px] rounded-full bg-[#C15F3C]/5 blur-[100px] pointer-events-none" />

      {/* ── Left sidebar ─────────────────────────────────────────────── */}
      <aside className="w-72 flex-shrink-0 border-r border-[#DDD9D0] bg-[#F4F3EE]/80 backdrop-blur-md flex flex-col overflow-y-auto relative z-10">
        {/* Task selector */}
        <div className="p-4 border-b border-[#DDD9D0]">
          <p className="text-[10px] font-mono text-[#B1ADA1] uppercase tracking-wider mb-2">Task</p>
          <div className="grid grid-cols-3 gap-1">
            {Object.entries(TASK_META).map(([key, meta]) => {
              const Icon = TASK_ICONS[key];
              return (
                <button
                  key={key}
                  data-testid={`task-${key}`}
                  onClick={() => dispatch({ type: "SET_TASK", v: key })}
                  className={`flex flex-col items-center gap-1 py-2 px-1 rounded-sm text-[10px] font-mono transition-all border ${
                    st.task === key
                      ? "border-[#DDD9D0] bg-white shadow-sm"
                      : "border-transparent hover:bg-white/60"
                  }`}
                  style={{ color: st.task === key ? meta.color : "#8A8580" }}
                >
                  <Icon size={16} weight={st.task === key ? "bold" : "regular"} />
                  {meta.shortLabel}
                </button>
              );
            })}
          </div>
        </div>

        {/* Model selector */}
        <div className="p-4 border-b border-[#DDD9D0]">
          <div className="flex items-center justify-between mb-2">
            <p className="text-[10px] font-mono text-[#B1ADA1] uppercase tracking-wider">Model</p>
            <button
              onClick={() => navigate("/models")}
              className="text-[10px] font-mono text-[#C15F3C] hover:underline"
            >
              + Add model
            </button>
          </div>
          {taskModels.length === 0 ? (
            <p className="text-xs text-[#B1ADA1]">No models for this task yet.</p>
          ) : (
            <select
              data-testid="model-select"
              value={st.modelId || ""}
              onChange={(e) => dispatch({ type: "SET_MODEL", v: e.target.value })}
              className="w-full bg-white border border-[#DDD9D0] rounded-sm px-2 py-1.5 text-xs font-mono text-[#141413] focus:outline-none focus:border-[#B1ADA1]"
            >
              {taskModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}{m.isCustom ? " (custom)" : ""}
                </option>
              ))}
            </select>
          )}

          {/* Model status */}
          {selectedMeta && (
            <div className="mt-2 space-y-1">
              {noModel ? (
                <div className="flex items-start gap-1.5 p-2 bg-[#FFF8F0] border border-[#FFD9B5] rounded-sm">
                  <Warning size={12} className="text-[#C15F3C] mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="text-[10px] text-[#C15F3C] font-mono font-medium">No URL registered</p>
                    <p className="text-[10px] text-[#8A8580] mt-0.5 leading-snug">
                      Export this model to TF.js then register the URL in Model Hub.
                    </p>
                    <button
                      onClick={() => navigate(`/models?export=${selectedMeta.id}`)}
                      className="text-[10px] text-[#C15F3C] hover:underline mt-1"
                    >
                      View export guide →
                    </button>
                  </div>
                </div>
              ) : (
                <div className="flex items-center gap-1.5 text-[10px] font-mono text-[#008B22]">
                  <div className="w-1.5 h-1.5 rounded-full bg-[#008B22]" />
                  URL registered — ready to load
                </div>
              )}
              {selectedMeta.mapCOCO && (
                <div className="flex gap-3 text-[10px] font-mono text-[#8A8580]">
                  <span>mAP {selectedMeta.mapCOCO}</span>
                  <span>·</span>
                  <span>{selectedMeta.sizeApprox}</span>
                  <span>·</span>
                  <span>{selectedMeta.params}</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Source selector */}
        <div className="p-4 border-b border-[#DDD9D0]">
          <p className="text-[10px] font-mono text-[#B1ADA1] uppercase tracking-wider mb-2">Input Source</p>
          <div className="flex gap-1">
            {SOURCE_TYPES.map(({ id, icon: Icon, label }) => (
              <button
                key={id}
                data-testid={`source-${id}`}
                onClick={() => {
                  stopLoop();
                  stopWebcam();
                  dispatch({ type: "SET_SOURCE", v: id });
                }}
                className={`flex-1 flex flex-col items-center gap-1 py-2 rounded-sm text-[10px] font-mono border transition-all ${
                  st.sourceType === id
                    ? "bg-white border-[#DDD9D0] text-[#141413] shadow-sm"
                    : "border-transparent text-[#8A8580] hover:bg-white/60"
                }`}
              >
                <Icon size={16} />
                {label}
              </button>
            ))}
          </div>
          {/* File upload */}
          {(st.sourceType === "image" || st.sourceType === "video") && (
            <div className="mt-2">
              <input
                ref={fileInputRef}
                type="file"
                accept={st.sourceType === "image" ? "image/*" : "video/*"}
                onChange={handleFileUpload}
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="flex items-center justify-center gap-2 w-full py-2 border border-dashed border-[#DDD9D0] rounded-sm text-xs text-[#8A8580] cursor-pointer hover:border-[#B1ADA1] hover:text-[#141413] transition-colors"
              >
                <UploadSimple size={13} />
                {st.sourceType === "image" ? "Drop or click to upload image" : "Drop or click to upload video"}
              </label>
            </div>
          )}
        </div>

        {/* Inference controls */}
        <div className="p-4 border-b border-[#DDD9D0] space-y-3">
          <p className="text-[10px] font-mono text-[#B1ADA1] uppercase tracking-wider">Inference</p>

          {/* Confidence */}
          <div>
            <div className="flex justify-between mb-1">
              <span className="text-[10px] font-mono text-[#8A8580]">Confidence</span>
              <span className="text-[10px] font-mono text-[#141413]">{(st.confidence * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range" min="5" max="95" step="5"
              value={Math.round(st.confidence * 100)}
              onChange={(e) => dispatch({ type: "SET_CONF", v: +e.target.value / 100 })}
              className="w-full accent-[#008B22] h-1 cursor-pointer"
              data-testid="confidence-slider"
            />
          </div>

          {/* IoU threshold */}
          <div>
            <div className="flex justify-between mb-1">
              <span className="text-[10px] font-mono text-[#8A8580]">IoU (NMS)</span>
              <span className="text-[10px] font-mono text-[#141413]">{(st.iouThreshold * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range" min="10" max="90" step="5"
              value={Math.round(st.iouThreshold * 100)}
              onChange={(e) => dispatch({ type: "SET_IOU", v: +e.target.value / 100 })}
              className="w-full accent-[#0087B3] h-1 cursor-pointer"
              data-testid="iou-slider"
            />
          </div>

          {/* Run / Stop button */}
          {st.sourceType === "image" ? (
            <button
              data-testid="run-btn"
              onClick={runImage}
              disabled={!canRun || st.loading}
              className="w-full flex items-center justify-center gap-2 py-2 bg-[#141413] text-[#F4F3EE] text-xs font-medium rounded-sm hover:bg-[#2A2925] disabled:opacity-40 transition-colors"
            >
              {st.loading
                ? <><Spinner size={13} className="animate-spin" /> Loading model…</>
                : <><Play size={13} weight="fill" /> Run Inference</>}
            </button>
          ) : st.running ? (
            <button
              data-testid="stop-btn"
              onClick={st.sourceType === "webcam" ? stopWebcam : stopLoop}
              className="w-full flex items-center justify-center gap-2 py-2 bg-[#CC1144] text-white text-xs font-medium rounded-sm hover:bg-[#A50D35] transition-colors"
            >
              <Stop size={13} weight="fill" /> Stop
            </button>
          ) : (
            <button
              data-testid="start-btn"
              onClick={st.sourceType === "webcam"
                ? startWebcam
                : () => { if (videoRef.current) { videoRef.current.play(); startLoop(videoRef); } }
              }
              disabled={!canRun || st.loading}
              className="w-full flex items-center justify-center gap-2 py-2 bg-[#141413] text-[#F4F3EE] text-xs font-medium rounded-sm hover:bg-[#2A2925] disabled:opacity-40 transition-colors"
            >
              {st.loading
                ? <><Spinner size={13} className="animate-spin" /> Loading model…</>
                : <><Play size={13} weight="fill" />
                    {st.sourceType === "webcam" ? "Start Webcam" : "Start Video"}</>}
            </button>
          )}
        </div>

        {/* Compare mode */}
        <div className="p-4 border-b border-[#DDD9D0]">
          <div className="flex items-center justify-between mb-2">
            <p className="text-[10px] font-mono text-[#B1ADA1] uppercase tracking-wider">Compare</p>
            <button
              data-testid="compare-toggle"
              onClick={() => dispatch({ type: "SET_COMPARE", v: !st.compareMode })}
              className={`text-[10px] font-mono px-2 py-0.5 rounded-sm border transition-colors ${
                st.compareMode
                  ? "bg-[#C15F3C] border-[#C15F3C] text-white"
                  : "border-[#DDD9D0] text-[#8A8580] hover:border-[#B1ADA1]"
              }`}
            >
              {st.compareMode ? "ON" : "OFF"}
            </button>
          </div>
          {st.compareMode && (
            <select
              data-testid="compare-model-select"
              value={st.compareModelId || ""}
              onChange={(e) => dispatch({ type: "SET_COMPARE_MODEL", v: e.target.value })}
              className="w-full bg-white border border-[#DDD9D0] rounded-sm px-2 py-1.5 text-xs font-mono text-[#141413] focus:outline-none"
            >
              <option value="">Select model B…</option>
              {taskModels.map((m) => (
                <option key={m.id} value={m.id}>{m.name}</option>
              ))}
            </select>
          )}
        </div>

        {/* Stats */}
        {st.runStats && (
          <div className="p-4 border-b border-[#DDD9D0]">
            <p className="text-[10px] font-mono text-[#B1ADA1] uppercase tracking-wider mb-2">Stats</p>
            <div className="grid grid-cols-3 gap-2">
              {[
                { label: "FPS", value: st.runStats.fps, icon: Lightning, color: "#008B22" },
                { label: "ms", value: st.runStats.latency, icon: Timer, color: "#0087B3" },
                { label: "det", value: st.results.length, icon: BoundingBox, color: "#C15F3C" },
              ].map(({ label, value, icon: Icon, color }) => (
                <div key={label} className="bg-white border border-[#DDD9D0] rounded-sm p-2 text-center">
                  <Icon size={11} style={{ color }} className="mx-auto mb-0.5" />
                  <div className="font-mono text-sm font-medium" style={{ color }}>{value}</div>
                  <div className="text-[9px] font-mono text-[#B1ADA1] uppercase">{label}</div>
                </div>
              ))}
            </div>
            {st.inferenceDebug && (
              <div className="mt-2 rounded-sm border border-[#DDD9D0] bg-white p-2 text-[9px] font-mono text-[#5C5751] leading-relaxed">
                <div className="flex justify-between gap-2">
                  <span>raw</span>
                  <span className="text-[#141413]">
                    {(st.inferenceDebug.outputShapes || []).map((s) => `[${s.join(",")}]`).join(" ")}
                  </span>
                </div>
                <div className="flex justify-between gap-2">
                  <span>score max</span>
                  <span className="text-[#141413]">
                    {Number.isFinite(st.inferenceDebug.maxScore)
                      ? st.inferenceDebug.maxScore.toFixed(3)
                      : "n/a"}
                  </span>
                </div>
                <div className="flex justify-between gap-2">
                  <span>format</span>
                  <span className="text-[#141413]">{st.inferenceDebug.boxFormat || "n/a"}</span>
                </div>
                <div className="flex justify-between gap-2">
                  <span>kept / threshold / invalid</span>
                  <span className="text-[#141413]">
                    {st.inferenceDebug.parsedCount ?? 0} / {st.inferenceDebug.overThreshold ?? 0} / {st.inferenceDebug.invalidBoxes ?? 0}
                  </span>
                </div>
              </div>
            )}
            {/* Actions */}
            <div className="flex gap-2 mt-2">
              <button
                onClick={saveRun}
                disabled={st.saving}
                className="flex-1 flex items-center justify-center gap-1 py-1.5 text-[10px] font-mono border border-[#DDD9D0] text-[#8A8580] rounded-sm hover:border-[#B1ADA1] hover:text-[#141413] transition-colors disabled:opacity-50"
              >
                {st.saving ? <Spinner size={10} className="animate-spin" /> : <FloppyDisk size={10} />}
                Save
              </button>
              <button
                onClick={exportPNG}
                className="flex-1 flex items-center justify-center gap-1 py-1.5 text-[10px] font-mono border border-[#DDD9D0] text-[#8A8580] rounded-sm hover:border-[#B1ADA1] hover:text-[#141413] transition-colors"
              >
                <DownloadSimple size={10} /> Export PNG
              </button>
            </div>
          </div>
        )}

        {/* Backend badge */}
        <div className="p-4 mt-auto">
          <div className="flex items-center gap-1.5 text-[10px] font-mono text-[#8A8580]">
            <div className="w-1.5 h-1.5 rounded-full bg-[#008B22]" />
            TF.js · {String(backendName).toUpperCase()}
          </div>
        </div>
      </aside>

      <main className="flex-1 flex flex-col overflow-hidden bg-[#F4F3EE]">
        {/* Error banner */}
        {st.error && (
          <div className="flex items-center gap-2 px-4 py-2 bg-[#CC1144]/10 border-b border-[#CC1144]/30 text-xs text-[#CC1144] font-mono">
            <Warning size={13} />
            {st.error}
            <button onClick={() => dispatch({ type: "CLEAR_ERROR" })} className="ml-auto opacity-60">×</button>
          </div>
        )}

        {/* Canvas grid */}
        <div className={`flex-1 flex ${st.compareMode ? "divide-x divide-[#DDD9D0]" : ""} overflow-hidden`}>
          {/* Primary canvas */}
          <div className="relative flex-1 flex items-center justify-center overflow-hidden">
            {/* Source element (hidden behind canvas) */}
            {st.sourceType === "image" && st.mediaUrl && (
              <img
                ref={imageRef}
                src={st.mediaUrl}
                alt="input"
                className="absolute max-w-none opacity-0 pointer-events-none"
                style={{ maxWidth: "100%", maxHeight: "100%" }}
                onLoad={() => {
                  if (canvasRef.current && imageRef.current) {
                    resizeCanvas(canvasRef, imageRef.current);
                    // Draw source
                    const ctx = canvasRef.current.getContext("2d");
                    ctx.drawImage(imageRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
                  }
                }}
              />
            )}
            {st.sourceType === "video" && st.mediaUrl && (
              <video
                ref={videoRef}
                src={st.mediaUrl}
                muted
                preload="auto"
                className="absolute max-w-none opacity-0 pointer-events-none"
                style={{ maxWidth: "100%", maxHeight: "100%" }}
              />
            )}
            {st.sourceType === "webcam" && (
              <video
                ref={webcamRef}
                muted
                playsInline
                className="absolute max-w-none opacity-0 pointer-events-none"
              />
            )}

            {/* Annotation canvas */}
            <canvas
              ref={canvasRef}
              data-testid="inference-canvas"
              className="max-w-full max-h-full object-contain"
              style={{ imageRendering: "crisp-edges" }}
            />

            {/* Empty state */}
            {!st.mediaUrl && st.sourceType !== "webcam" && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-[#B1ADA1] select-none">
                <div className="w-20 h-20 bg-white border border-[#DDD9D0] rounded-sm flex items-center justify-center mb-4 shadow-sm">
                  <BoundingBox size={32} weight="thin" />
                </div>
                <p className="text-sm font-['Outfit'] text-[#141413] font-medium">Ready for Inference</p>
                <p className="text-xs text-[#8A8580] mt-1 max-w-[200px] text-center">
                  Select a model and upload an {st.sourceType} to begin.
                </p>
                <div className="flex gap-2 mt-6">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="px-4 py-1.5 bg-white border border-[#DDD9D0] text-[#141413] text-xs font-mono rounded-sm hover:border-[#B1ADA1] transition-all"
                  >
                    Upload File
                  </button>
                  <button
                    onClick={() => {
                      dispatch({ type: "SET_SOURCE", v: "image" });
                      dispatch({ type: "SET_MEDIA", v: "/demo.jpg" });
                    }}
                    className="px-4 py-1.5 bg-[#141413] text-[#F4F3EE] text-xs font-mono rounded-sm hover:bg-[#2A2925] transition-all"
                  >
                    Try Demo
                  </button>
                </div>
              </div>
            )}
            {st.sourceType === "webcam" && !st.running && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-[#B1ADA1] select-none">
                <Camera size={56} weight="thin" />
                <p className="mt-3 text-sm font-mono">Click "Start Webcam" to begin</p>
              </div>
            )}

            {/* Task/model badge */}
            <div className="absolute top-3 left-3 flex items-center gap-2">
              <span
                className="px-2 py-0.5 text-[10px] font-mono rounded-sm"
                style={{
                  background: (TASK_META[st.task]?.color || "#008B22") + "22",
                  color: TASK_META[st.task]?.color || "#008B22",
                  border: `1px solid ${(TASK_META[st.task]?.color || "#008B22")}44`,
                }}
              >
                {TASK_META[st.task]?.shortLabel}
              </span>
              {selectedMeta && (
                <span className="px-2 py-0.5 text-[10px] font-mono bg-white text-[#8A8580] border border-[#DDD9D0] rounded-sm shadow-sm">
                  {selectedMeta.name}
                </span>
              )}
              {st.running && (
                <span className="flex items-center gap-1 px-2 py-0.5 text-[10px] font-mono bg-[#008B22]/10 text-[#008B22] border border-[#008B22]/20 rounded-sm">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#008B22] animate-pulse" />
                  LIVE
                </span>
              )}
            </div>
          </div>

          {/* Compare canvas */}
          {st.compareMode && (
            <div className="relative flex-1 flex items-center justify-center overflow-hidden">
              <canvas
                ref={cmpCanvasRef}
                data-testid="compare-canvas"
                className="max-w-full max-h-full object-contain"
              />
              {!st.compareModelId && (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-[#B1ADA1]">
                  <Columns size={48} weight="thin" />
                  <p className="text-sm font-mono mt-3">Select model B in sidebar</p>
                </div>
              )}
              {cmpMeta && (
                <div className="absolute top-3 left-3">
                  <span className="px-2 py-0.5 text-[10px] font-mono bg-white text-[#C15F3C] border border-[#C15F3C]/30 rounded-sm shadow-sm">
                    {cmpMeta.name} (B)
                  </span>
                  {st.compareStats && (
                    <span className="ml-1 px-2 py-0.5 text-[10px] font-mono bg-white text-[#8A8580] border border-[#DDD9D0] rounded-sm shadow-sm">
                      {st.compareStats.fps}fps · {st.compareStats.latency}ms
                    </span>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Results strip */}
        {st.results.length > 0 && (
          <div className="h-20 flex-shrink-0 border-t border-[#DDD9D0] bg-[#F8F7F2] px-4 py-2 overflow-x-auto flex items-center gap-2">
            {st.results.slice(0, 20).map((r, i) => (
              <div
                key={i}
                className="flex-shrink-0 flex items-center gap-1.5 px-2 py-1 rounded-sm border text-[10px] font-mono bg-white"
                style={{
                  borderColor: (r.color || "#008B22") + "44",
                  color: r.color || "#008B22",
                }}
              >
                <div
                  className="w-1.5 h-1.5 rounded-full"
                  style={{ backgroundColor: r.color || "#008B22" }}
                />
                {r.class}
                <span className="opacity-60 text-[#8A8580]">{(r.score * 100).toFixed(0)}%</span>
              </div>
            ))}
            {st.results.length > 20 && (
              <span className="text-[10px] font-mono text-[#8A8580]">+{st.results.length - 20} more</span>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
