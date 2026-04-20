/**
 * Deplyze Vision - YOLO TF.js Inference Engine
 *
 * Pure browser inference for Ultralytics TF.js GraphModel exports.
 */

import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu";
import * as ort from "onnxruntime-web";
import { colorForClass } from "./cocoClasses";

let backendReady = false;

export async function initTFBackend() {
  if (backendReady) return tf.getBackend();

  for (const backend of ["webgpu", "webgl", "cpu"]) {
    try {
      await tf.setBackend(backend);
      await tf.ready();
      backendReady = true;
      console.info(`[TF.js] backend: ${backend}`);
      return backend;
    } catch (_) {
      // Try the next backend.
    }
  }

  throw new Error("No TensorFlow.js backend could be initialised");
}

export function getTFBackendName() {
  return tf.getBackend() || "unknown";
}

const MODEL_CACHE_MAX = 3;
const modelCache = new Map();
const metadataCache = new Map();

ort.env.wasm.wasmPaths = "/ort/";
ort.env.wasm.numThreads = 1;

function evictOldest() {
  if (modelCache.size < MODEL_CACHE_MAX) return;

  let oldestKey = null;
  let oldestTs = Infinity;
  for (const [key, value] of modelCache) {
    if (value.ts < oldestTs) {
      oldestTs = value.ts;
      oldestKey = key;
    }
  }

  if (oldestKey) {
    modelCache.get(oldestKey).model.dispose?.();
    modelCache.delete(oldestKey);
  }
}

export async function loadModel(urlOrMeta) {
  const url = typeof urlOrMeta === "string" ? urlOrMeta : urlOrMeta?.url;
  const runtime = typeof urlOrMeta === "string" ? "tfjs" : urlOrMeta?.runtime || "tfjs";
  if (!url) throw new Error("Model URL is required");

  const cacheKey = `${runtime}:${url}`;
  const cached = modelCache.get(cacheKey);
  if (cached) {
    cached.ts = Date.now();
    return cached.model;
  }

  evictOldest();

  let model;
  if (runtime === "onnx") {
    model = await ort.InferenceSession.create(url, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  } else {
    await initTFBackend();
    model = await tf.loadGraphModel(url);
  }

  modelCache.set(cacheKey, { model, ts: Date.now() });
  return model;
}

export function disposeAllModels() {
  for (const { model } of modelCache.values()) {
    model.dispose?.();
  }
  modelCache.clear();
}

export function preprocessSource(source, inputSize = 640) {
  return tf.tidy(() => {
    const pixels = tf.browser.fromPixels(source);
    const resized = tf.image.resizeBilinear(pixels, [inputSize, inputSize]);
    return resized.toFloat().expandDims(0);
  });
}

function preprocessSourceToCHW(source, inputSize = 640) {
  const canvas = document.createElement("canvas");
  canvas.width = inputSize;
  canvas.height = inputSize;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(source, 0, 0, inputSize, inputSize);

  const pixels = ctx.getImageData(0, 0, inputSize, inputSize).data;
  const chw = new Float32Array(3 * inputSize * inputSize);
  const plane = inputSize * inputSize;

  for (let i = 0, p = 0; i < pixels.length; i += 4, p += 1) {
    chw[p] = pixels[i] / 255;
    chw[plane + p] = pixels[i + 1] / 255;
    chw[plane * 2 + p] = pixels[i + 2] / 255;
  }

  return chw;
}

function disposeOutputs(outputs) {
  const seen = new Set();
  const tensors = Array.isArray(outputs) ? outputs : [outputs];
  tensors.forEach((tensor) => {
    if (!tensor || seen.has(tensor)) return;
    seen.add(tensor);
    try {
      tensor.dispose?.();
    } catch (_) {
      // Tensor may already be disposed by TF.js internals.
    }
  });
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const aArea = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const bArea = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  return inter / (aArea + bArea - inter + 1e-7);
}

function applyNMS(dets, iouThreshold = 0.45) {
  const kept = [];
  const used = new Array(dets.length).fill(false);

  for (let i = 0; i < dets.length; i += 1) {
    if (used[i]) continue;
    kept.push(dets[i]);

    for (let j = i + 1; j < dets.length; j += 1) {
      if (!used[j] && iou(dets[i].bbox, dets[j].bbox) > iouThreshold) {
        used[j] = true;
      }
    }
  }

  return kept;
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function normaliseCoord(value, inputSize) {
  if (!Number.isFinite(value)) return 0;
  return clamp01(Math.abs(value) <= 1.5 ? value : value / inputSize);
}

function normaliseLength(value, inputSize) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.abs(value) <= 1.5 ? Math.abs(value) : Math.abs(value) / inputSize);
}

function normaliseBoxFromCenter(cx, cy, width, height, inputSize) {
  const nx = normaliseCoord(cx, inputSize);
  const ny = normaliseCoord(cy, inputSize);
  const nw = normaliseLength(width, inputSize);
  const nh = normaliseLength(height, inputSize);
  return [
    clamp01(nx - nw / 2),
    clamp01(ny - nh / 2),
    clamp01(nx + nw / 2),
    clamp01(ny + nh / 2),
  ];
}

function normaliseBoxFromCorners(x1, y1, x2, y2, inputSize) {
  return [
    normaliseCoord(x1, inputSize),
    normaliseCoord(y1, inputSize),
    normaliseCoord(x2, inputSize),
    normaliseCoord(y2, inputSize),
  ];
}

function isProbablyEndToEndOutput(numChannels, numClasses) {
  return numChannels === 6 || numChannels === 7 || numChannels === numClasses + 5;
}

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function inferEndToEndBoxFormat(data, numAnchors, numChannels, inputSize) {
  const sampleCount = Math.min(numAnchors, 100);
  const cornerWidths = [];
  const cornerHeights = [];
  const centerWidths = [];
  const centerHeights = [];
  let validCornerRows = 0;

  for (let i = 0; i < sampleCount; i += 1) {
    const base = i * numChannels;
    const x1 = data[base + 0];
    const y1 = data[base + 1];
    const x2OrW = data[base + 2];
    const y2OrH = data[base + 3];
    if (![x1, y1, x2OrW, y2OrH].every(Number.isFinite)) continue;

    const cornerW = normaliseLength(x2OrW - x1, inputSize) * inputSize;
    const cornerH = normaliseLength(y2OrH - y1, inputSize) * inputSize;
    const centerW = normaliseLength(x2OrW, inputSize) * inputSize;
    const centerH = normaliseLength(y2OrH, inputSize) * inputSize;

    if (x2OrW > x1 && y2OrH > y1) validCornerRows += 1;
    if (cornerW > 0) cornerWidths.push(cornerW);
    if (cornerH > 0) cornerHeights.push(cornerH);
    if (centerW > 0) centerWidths.push(centerW);
    if (centerH > 0) centerHeights.push(centerH);
  }

  const validCornerRate = sampleCount ? validCornerRows / sampleCount : 0;
  const medianCornerW = median(cornerWidths);
  const medianCornerH = median(cornerHeights);
  const medianCenterW = median(centerWidths);
  const medianCenterH = median(centerHeights);
  const cornersCollapse =
    (medianCornerW < inputSize * 0.02 || medianCornerH < inputSize * 0.02) &&
    medianCenterW > inputSize * 0.05 &&
    medianCenterH > inputSize * 0.05;

  return validCornerRate < 0.85 || cornersCollapse ? "center" : "corners";
}

function normaliseEndToEndBox(data, base, inputSize, boxFormat) {
  return boxFormat === "center"
    ? normaliseBoxFromCenter(
      data[base + 0],
      data[base + 1],
      data[base + 2],
      data[base + 3],
      inputSize
    )
    : normaliseBoxFromCorners(
      data[base + 0],
      data[base + 1],
      data[base + 2],
      data[base + 3],
      inputSize
    );
}

function hasValidBox(bbox) {
  if (!bbox?.every(Number.isFinite)) return false;
  return bbox[2] - bbox[0] > 0.001 && bbox[3] - bbox[1] > 0.001;
}

function tensorAsAnchorRows(output) {
  if (output.shape.length === 2) {
    return {
      tensor: output,
      numAnchors: output.shape[0],
      numChannels: output.shape[1],
      ownsTensor: false,
    };
  }

  if (output.shape.length !== 3 || output.shape[0] !== 1) {
    throw new Error(`Unsupported YOLO output shape: [${output.shape.join(", ")}]`);
  }

  const [, dim1, dim2] = output.shape;
  const channelsFirst = dim1 < dim2;
  const tensor = channelsFirst
    ? tf.tidy(() => output.squeeze([0]).transpose([1, 0]))
    : tf.tidy(() => output.squeeze([0]));

  return {
    tensor,
    numAnchors: channelsFirst ? dim2 : dim1,
    numChannels: channelsFirst ? dim1 : dim2,
    ownsTensor: true,
  };
}

async function readRows(output) {
  const rows = tensorAsAnchorRows(output);
  const data = await rows.tensor.data();
  if (rows.ownsTensor) rows.tensor.dispose();
  return { ...rows, data };
}

function parseMetadataNames(yamlText) {
  const names = [];
  let inNames = false;

  for (const line of yamlText.split(/\r?\n/)) {
    if (/^names:\s*$/.test(line)) {
      inNames = true;
      continue;
    }
    if (!inNames) continue;
    if (/^\S/.test(line)) break;

    const match = line.match(/^\s*(\d+):\s*(.+?)\s*$/);
    if (match) {
      names[Number(match[1])] = match[2].replace(/^['"]|['"]$/g, "");
    }
  }

  return names;
}

async function resolveClasses(modelMeta) {
  if (modelMeta?.metadataUrl) {
    const cached = metadataCache.get(modelMeta.metadataUrl);
    if (cached) return cached.length ? cached : modelMeta.classes || modelMeta.labels || [];

    try {
      const res = await fetch(modelMeta.metadataUrl);
      if (res.ok) {
        const names = parseMetadataNames(await res.text());
        metadataCache.set(modelMeta.metadataUrl, names);
        if (names.length) return names;
      }
    } catch (_) {
      metadataCache.set(modelMeta.metadataUrl, []);
    }
  }

  return modelMeta?.classes || modelMeta?.labels || [];
}

export async function parseDetectionOutput(rawOutput, {
  classes = [],
  numClasses = classes.length,
  inputSize,
  confidenceThreshold = 0.25,
  iouThreshold = 0.45,
  debug = null,
}) {
  const output = Array.isArray(rawOutput) ? rawOutput[0] : rawOutput;
  const { data, numAnchors, numChannels } = await readRows(output);
  const detections = [];
  const endToEnd = isProbablyEndToEndOutput(numChannels, numClasses) && numChannels <= 7;
  const endToEndBoxFormat = endToEnd
    ? inferEndToEndBoxFormat(data, numAnchors, numChannels, inputSize)
    : "center";
  let finiteRows = 0;
  let overThreshold = 0;
  let invalidBoxes = 0;
  let maxScore = Number.NEGATIVE_INFINITY;
  let maxScoreRow = null;

  for (let i = 0; i < numAnchors; i += 1) {
    const base = i * numChannels;
    let bbox;
    let score;
    let classIndex;

    if (endToEnd) {
      bbox = normaliseEndToEndBox(data, base, inputSize, endToEndBoxFormat);
      score = data[base + 4];
      classIndex = Math.max(0, Math.round(data[base + 5] || 0));
    } else {
      bbox = normaliseBoxFromCenter(
        data[base + 0],
        data[base + 1],
        data[base + 2],
        data[base + 3],
        inputSize
      );

      score = 0;
      classIndex = 0;
      for (let c = 0; c < numClasses; c += 1) {
        const classScore = data[base + 4 + c];
        if (classScore > score) {
          score = classScore;
          classIndex = c;
        }
      }
    }

    const row = Array.from(data.slice(base, base + numChannels));
    if (row.every(Number.isFinite)) finiteRows += 1;
    if (Number.isFinite(score) && score > maxScore) {
      maxScore = score;
      maxScoreRow = row;
    }
    if (!Number.isFinite(score) || score < confidenceThreshold) continue;
    overThreshold += 1;
    if (!hasValidBox(bbox)) {
      invalidBoxes += 1;
      continue;
    }

    detections.push({
      bbox,
      score,
      class: classes[classIndex] ?? `class_${classIndex}`,
      classIndex,
      color: colorForClass(classIndex),
    });
  }

  detections.sort((a, b) => b.score - a.score);
  const results = endToEnd ? detections : applyNMS(detections, iouThreshold);
  if (debug) {
    Object.assign(debug, {
      outputShapes: [output.shape],
      numAnchors,
      numChannels,
      finiteRows,
      endToEnd,
      boxFormat: endToEndBoxFormat,
      confidenceThreshold,
      maxScore: Number.isFinite(maxScore) ? maxScore : null,
      maxScoreRow,
      overThreshold,
      invalidBoxes,
      parsedCount: results.length,
    });
  }
  return results;
}

export const POSE_KEYPOINT_NAMES = [
  "nose", "left_eye", "right_eye", "left_ear", "right_ear",
  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_hip", "right_hip",
  "left_knee", "right_knee", "left_ankle", "right_ankle",
];

export const POSE_SKELETON = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [5, 6],
  [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
];

export async function parsePoseOutput(rawOutput, {
  inputSize,
  numKeypoints = 17,
  confidenceThreshold = 0.25,
  iouThreshold = 0.45,
}) {
  const output = Array.isArray(rawOutput) ? rawOutput[0] : rawOutput;
  const { data, numAnchors, numChannels } = await readRows(output);
  const raw = [];
  const endToEnd = numChannels === 6 + numKeypoints * 3;

  for (let i = 0; i < numAnchors; i += 1) {
    const base = i * numChannels;
    const bbox = endToEnd
      ? normaliseBoxFromCorners(
        data[base + 0],
        data[base + 1],
        data[base + 2],
        data[base + 3],
        inputSize
      )
      : normaliseBoxFromCenter(
        data[base + 0],
        data[base + 1],
        data[base + 2],
        data[base + 3],
        inputSize
      );
    const conf = data[base + 4];
    if (!Number.isFinite(conf) || conf < confidenceThreshold) continue;

    const keypoints = [];
    const keypointStart = endToEnd ? 6 : 5;
    for (let k = 0; k < numKeypoints; k += 1) {
      const offset = base + keypointStart + k * 3;
      keypoints.push({
        x: clamp01(data[offset] / inputSize),
        y: clamp01(data[offset + 1] / inputSize),
        score: data[offset + 2],
        name: POSE_KEYPOINT_NAMES[k],
      });
    }

    raw.push({
      bbox,
      score: conf,
      class: "person",
      classIndex: 0,
      color: "#FF3366",
      keypoints,
    });
  }

  raw.sort((a, b) => b.score - a.score);
  return endToEnd ? raw : applyNMS(raw, iouThreshold);
}

export async function parseSegOutput(rawOutputs, {
  classes = [],
  numClasses = classes.length,
  inputSize,
  confidenceThreshold = 0.25,
  iouThreshold = 0.45,
  debug = null,
}) {
  if (!Array.isArray(rawOutputs) || rawOutputs.length < 2) {
    return parseDetectionOutput(rawOutputs, {
      classes,
      numClasses,
      inputSize,
      confidenceThreshold,
      iouThreshold,
      debug,
    });
  }

  const detTensor = rawOutputs.find((t) => t.shape.length === 3 && Math.min(t.shape[1], t.shape[2]) >= 6);
  const protoTensor = rawOutputs.find((t) => t !== detTensor);
  if (!detTensor || !protoTensor) {
    return parseDetectionOutput(rawOutputs[0], {
      classes,
      numClasses,
      inputSize,
      confidenceThreshold,
      iouThreshold,
      debug,
    });
  }

  const { data: detData, numAnchors, numChannels } = await readRows(detTensor);
  const endToEnd = numChannels > 6 && numChannels < 4 + numClasses;
  const endToEndBoxFormat = endToEnd
    ? inferEndToEndBoxFormat(detData, numAnchors, numChannels, inputSize)
    : "center";
  const numMaskCoeffs = endToEnd
    ? Math.max(0, numChannels - 6)
    : Math.max(0, numChannels - 4 - numClasses);
  const raw = [];
  let finiteRows = 0;
  let overThreshold = 0;
  let invalidBoxes = 0;
  let maxScore = Number.NEGATIVE_INFINITY;
  let maxScoreRow = null;

  for (let i = 0; i < numAnchors; i += 1) {
    const base = i * numChannels;
    let bbox;
    let score;
    let classIndex;
    let coeffStart;

    if (endToEnd) {
      bbox = normaliseEndToEndBox(detData, base, inputSize, endToEndBoxFormat);
      score = detData[base + 4];
      classIndex = Math.max(0, Math.round(detData[base + 5] || 0));
      coeffStart = base + 6;
    } else {
      bbox = normaliseBoxFromCenter(
        detData[base + 0],
        detData[base + 1],
        detData[base + 2],
        detData[base + 3],
        inputSize
      );

      score = 0;
      classIndex = 0;
      for (let c = 0; c < numClasses; c += 1) {
        const classScore = detData[base + 4 + c];
        if (classScore > score) {
          score = classScore;
          classIndex = c;
        }
      }
      coeffStart = base + 4 + numClasses;
    }
    const row = Array.from(detData.slice(base, base + numChannels));
    if (row.every(Number.isFinite)) finiteRows += 1;
    if (Number.isFinite(score) && score > maxScore) {
      maxScore = score;
      maxScoreRow = row;
    }
    if (!Number.isFinite(score) || score < confidenceThreshold) continue;
    overThreshold += 1;
    if (!hasValidBox(bbox)) {
      invalidBoxes += 1;
      continue;
    }

    const coeffs = Array.from({ length: numMaskCoeffs }, (_, k) => detData[coeffStart + k]);
    raw.push({
      bbox,
      score,
      class: classes[classIndex] ?? `class_${classIndex}`,
      classIndex,
      color: colorForClass(classIndex),
      coeffs,
    });
  }

  raw.sort((a, b) => b.score - a.score);
  const kept = endToEnd ? raw : applyNMS(raw, iouThreshold);
  if (debug) {
    Object.assign(debug, {
      outputShapes: rawOutputs.map((t) => t.shape),
      numAnchors,
      numChannels,
      finiteRows,
      endToEnd,
      boxFormat: endToEndBoxFormat,
      confidenceThreshold,
      maxScore: Number.isFinite(maxScore) ? maxScore : null,
      maxScoreRow,
      overThreshold,
      invalidBoxes,
      parsedCount: kept.length,
      maskCoefficients: numMaskCoeffs,
      protoShape: protoTensor.shape,
    });
  }
  if (!kept.length || !numMaskCoeffs) return kept;

  const proto = tf.tidy(() => protoTensor.squeeze([0]));
  const protoData = await proto.data();
  let maskH = 0;
  let maskW = 0;
  let channelStride = 0;
  let pixelStride = 0;

  if (proto.shape.length === 3 && proto.shape[2] === numMaskCoeffs) {
    [maskH, maskW] = proto.shape;
    channelStride = 1;
    pixelStride = numMaskCoeffs;
  } else if (proto.shape.length === 3 && proto.shape[0] === numMaskCoeffs) {
    [, maskH, maskW] = proto.shape;
    channelStride = maskH * maskW;
    pixelStride = 1;
  }
  proto.dispose();
  if (!maskH || !maskW) return kept;

  return kept.map((det) => {
    const maskPixels = new Uint8ClampedArray(maskH * maskW);
    for (let px = 0; px < maskH * maskW; px += 1) {
      let dot = 0;
      for (let k = 0; k < numMaskCoeffs; k += 1) {
        dot += det.coeffs[k] * protoData[px * pixelStride + k * channelStride];
      }
      maskPixels[px] = 1 / (1 + Math.exp(-dot)) > 0.5 ? 1 : 0;
    }

    const { coeffs, ...result } = det;
    return { ...result, mask: { data: maskPixels, width: maskW, height: maskH } };
  });
}

export async function parseOBBOutput(rawOutput, {
  classes = [],
  numClasses = classes.length,
  inputSize,
  confidenceThreshold = 0.25,
  iouThreshold = 0.45,
}) {
  const output = Array.isArray(rawOutput) ? rawOutput[0] : rawOutput;
  const { data, numAnchors, numChannels } = await readRows(output);
  const raw = [];
  const endToEnd = numChannels === 7;

  for (let i = 0; i < numAnchors; i += 1) {
    const base = i * numChannels;
    let bbox;
    let score;
    let classIndex;
    let angle;

    if (endToEnd) {
      bbox = normaliseBoxFromCenter(
        data[base + 0],
        data[base + 1],
        data[base + 2],
        data[base + 3],
        inputSize
      );
      score = data[base + 4];
      classIndex = Math.max(0, Math.round(data[base + 5] || 0));
      angle = data[base + 6] || 0;
    } else {
      bbox = normaliseBoxFromCenter(
        data[base + 0],
        data[base + 1],
        data[base + 2],
        data[base + 3],
        inputSize
      );

      score = 0;
      classIndex = 0;
      for (let c = 0; c < numClasses; c += 1) {
        const classScore = data[base + 4 + c];
        if (classScore > score) {
          score = classScore;
          classIndex = c;
        }
      }
      angle = data[base + 4 + numClasses] || 0;
    }

    if (!Number.isFinite(score) || score < confidenceThreshold) continue;

    const [x1, y1, x2, y2] = bbox;
    const cx = (x1 + x2) / 2;
    const cy = (y1 + y2) / 2;
    const bw = x2 - x1;
    const bh = y2 - y1;

    raw.push({
      bbox,
      cx,
      cy,
      bw,
      bh,
      angle,
      score,
      class: classes[classIndex] ?? `class_${classIndex}`,
      classIndex,
      color: colorForClass(classIndex),
    });
  }

  raw.sort((a, b) => b.score - a.score);
  return endToEnd ? raw : applyNMS(raw, iouThreshold);
}

export async function parseClassifyOutput(rawOutput, { classes = [] }) {
  const output = Array.isArray(rawOutput) ? rawOutput[0] : rawOutput;
  const squeezed = tf.tidy(() => output.squeeze());
  const data = await squeezed.data();
  squeezed.dispose();

  return Array.from(data)
    .map((score, i) => ({
      class: classes[i] ?? `class_${i}`,
      score,
      classIndex: i,
      color: colorForClass(i),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 10);
}

function readYoloV8OnnxRows(output) {
  const { data, dims } = output;
  if (dims.length !== 3 || dims[0] !== 1) {
    throw new Error(`Unsupported YOLOv8 ONNX output shape: [${dims.join(", ")}]`);
  }

  const [, dim1, dim2] = dims;
  const channelsFirst = dim1 < dim2;
  return {
    data,
    numChannels: channelsFirst ? dim1 : dim2,
    numAnchors: channelsFirst ? dim2 : dim1,
    getValue: channelsFirst
      ? (anchor, channel) => data[channel * dim2 + anchor]
      : (anchor, channel) => data[anchor * dim2 + channel],
  };
}

function parseYoloV8OnnxOutput(output, {
  classes = [],
  numClasses = classes.length || 80,
  inputSize,
  confidenceThreshold = 0.25,
  iouThreshold = 0.45,
}) {
  const { numChannels, numAnchors, getValue } = readYoloV8OnnxRows(output);
  const hasObjectness = numChannels === numClasses + 5;
  const classStart = hasObjectness ? 5 : 4;
  const raw = [];
  let finiteRows = 0;
  let overThreshold = 0;
  let invalidBoxes = 0;
  let maxScore = Number.NEGATIVE_INFINITY;
  let maxScoreRow = null;

  for (let i = 0; i < numAnchors; i += 1) {
    const row = Array.from({ length: Math.min(numChannels, classStart + numClasses) }, (_, c) => getValue(i, c));
    if (row.every(Number.isFinite)) finiteRows += 1;

    let score = 0;
    let classIndex = 0;
    const objectness = hasObjectness ? getValue(i, 4) : 1;

    for (let c = 0; c < numClasses; c += 1) {
      const classScore = getValue(i, classStart + c) * objectness;
      if (classScore > score) {
        score = classScore;
        classIndex = c;
      }
    }

    if (Number.isFinite(score) && score > maxScore) {
      maxScore = score;
      maxScoreRow = row.slice(0, 12);
    }
    if (!Number.isFinite(score) || score < confidenceThreshold) continue;
    overThreshold += 1;

    const bbox = normaliseBoxFromCenter(
      getValue(i, 0),
      getValue(i, 1),
      getValue(i, 2),
      getValue(i, 3),
      inputSize
    );
    if (!hasValidBox(bbox)) {
      invalidBoxes += 1;
      continue;
    }

    raw.push({
      bbox,
      score,
      class: classes[classIndex] ?? `class_${classIndex}`,
      classIndex,
      color: colorForClass(classIndex),
    });
  }

  raw.sort((a, b) => b.score - a.score);
  const results = applyNMS(raw, iouThreshold);

  return {
    results,
    debug: {
      outputShapes: [output.dims],
      numAnchors,
      numChannels,
      finiteRows,
      endToEnd: false,
      boxFormat: "center",
      confidenceThreshold,
      maxScore: Number.isFinite(maxScore) ? maxScore : null,
      maxScoreRow,
      overThreshold,
      invalidBoxes,
      parsedCount: results.length,
      runtime: "onnxruntime-web",
    },
  };
}

export async function runYOLOOnnxInference(session, source, modelMeta, opts = {}) {
  const {
    confidenceThreshold = 0.25,
    iouThreshold = 0.45,
    canvasW = 640,
    canvasH = 640,
  } = opts;

  const inputSize = modelMeta.inputSize ?? 640;
  const classes = await resolveClasses(modelMeta);
  const numClasses = modelMeta.numClasses ?? modelMeta.num_classes ?? classes.length;
  const t0 = performance.now();
  const inputName = session.inputNames?.[0];
  const outputName = session.outputNames?.[0];
  if (!inputName) throw new Error("ONNX model has no input name");

  const input = preprocessSourceToCHW(source, inputSize);
  const feeds = {
    [inputName]: new ort.Tensor("float32", input, [1, 3, inputSize, inputSize]),
  };
  const rawOutputs = await session.run(feeds);
  const output = rawOutputs[outputName] || Object.values(rawOutputs)[0];
  const latency = Math.round(performance.now() - t0);
  const parsed = parseYoloV8OnnxOutput(output, {
    classes,
    numClasses,
    inputSize,
    confidenceThreshold,
    iouThreshold,
  });

  return {
    results: parsed.results,
    fps: latency > 0 ? Math.round(1000 / latency) : 0,
    latency,
    backend: "onnx-wasm",
    debug: {
      task: modelMeta.task,
      model: modelMeta.name || modelMeta.id || "unknown",
      backend: "onnx-wasm",
      inputSize,
      canvasW,
      canvasH,
      ...parsed.debug,
    },
  };
}

export async function runYOLOInference(model, source, modelMeta, opts = {}) {
  if (modelMeta?.runtime === "onnx") {
    return runYOLOOnnxInference(model, source, modelMeta, opts);
  }

  const {
    confidenceThreshold = 0.25,
    iouThreshold = 0.45,
    canvasW = 640,
    canvasH = 640,
  } = opts;

  const inputSize = modelMeta.inputSize ?? 640;
  const task = modelMeta.task;
  const classes = await resolveClasses(modelMeta);
  const numClasses = modelMeta.numClasses ?? modelMeta.num_classes ?? classes.length;
  const debug = {
    task,
    model: modelMeta.name || modelMeta.id || "unknown",
    backend: getTFBackendName(),
    inputSize,
    canvasW,
    canvasH,
  };
  const t0 = performance.now();
  const inputTensor = preprocessSource(source, inputSize);

  let rawOutput;
  try {
    try {
      rawOutput = await model.executeAsync(inputTensor);
    } catch (asyncError) {
      rawOutput = model.execute(inputTensor);
    }
  } finally {
    inputTensor.dispose();
  }

  const latency = Math.round(performance.now() - t0);
  let results = [];

  try {
    switch (task) {
      case "detect":
      case "track":
        results = await parseDetectionOutput(rawOutput, {
          classes,
          numClasses,
          inputSize,
          confidenceThreshold,
          iouThreshold,
          debug,
        });
        break;
      case "seg":
        results = await parseSegOutput(rawOutput, {
          classes,
          numClasses,
          inputSize,
          canvasW,
          canvasH,
          confidenceThreshold,
          iouThreshold,
          debug,
        });
        break;
      case "pose":
        results = await parsePoseOutput(rawOutput, {
          inputSize,
          numKeypoints: modelMeta.numKeypoints ?? modelMeta.num_keypoints ?? 17,
          confidenceThreshold,
          iouThreshold,
        });
        break;
      case "obb":
        results = await parseOBBOutput(rawOutput, {
          classes,
          numClasses,
          inputSize,
          confidenceThreshold,
          iouThreshold,
        });
        break;
      case "classify":
        results = await parseClassifyOutput(rawOutput, { classes });
        break;
      default:
        results = [];
        break;
    }
  } finally {
    disposeOutputs(rawOutput);
  }

  return {
    results,
    fps: latency > 0 ? Math.round(1000 / latency) : 0,
    latency,
    backend: getTFBackendName(),
    debug: { ...debug, parsedCount: results.length },
  };
}
