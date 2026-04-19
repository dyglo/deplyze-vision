export const CV_ACCENTS = {
  detect: "#00FF41",
  segment: "#00E5FF",
  pose: "#FF3366",
  track: "#FFD500",
  classify: "#B026FF",
};

const MONO_FONT = '"JetBrains Mono", "Fira Code", monospace';

// ── Detection ────────────────────────────────────────────────────────────────
export function drawDetections(ctx, detections) {
  if (!detections || !detections.length) return;
  const color = CV_ACCENTS.detect;
  detections.forEach((det) => {
    const [x, y, w, h] = det.bbox;
    // Box fill (10% opacity)
    ctx.fillStyle = color + "1A";
    ctx.fillRect(x, y, w, h);
    // Box border
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    // Label bg
    const label = `${det.class} ${Math.round(det.score * 100)}%`;
    ctx.font = `500 11px ${MONO_FONT}`;
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x, y - 18, tw + 8, 18);
    // Label text
    ctx.fillStyle = "#000";
    ctx.fillText(label, x + 4, y - 4);
  });
}

// ── Tracking ─────────────────────────────────────────────────────────────────
export function drawTracking(ctx, trackedObjects) {
  if (!trackedObjects) return;
  const color = CV_ACCENTS.track;
  Object.entries(trackedObjects).forEach(([id, obj]) => {
    if (!obj.det) return;
    const [x, y, w, h] = obj.det.bbox;
    ctx.fillStyle = color + "1A";
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    // Centroid dot
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(obj.cx, obj.cy, 4, 0, 2 * Math.PI);
    ctx.fill();
    // ID label
    const label = `#${id} ${obj.det.class}`;
    ctx.font = `500 11px ${MONO_FONT}`;
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x, y - 18, tw + 8, 18);
    ctx.fillStyle = "#000";
    ctx.fillText(label, x + 4, y - 4);
  });
}

// ── Pose skeleton connections (MoveNet COCO 17 keypoints) ────────────────────
const SKELETON = [
  ["nose", "left_eye"],
  ["nose", "right_eye"],
  ["left_eye", "left_ear"],
  ["right_eye", "right_ear"],
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_elbow"],
  ["right_shoulder", "right_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_elbow", "right_wrist"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"],
  ["right_hip", "right_knee"],
  ["left_knee", "left_ankle"],
  ["right_knee", "right_ankle"],
];

export function drawPose(ctx, poses) {
  if (!poses || !poses.length) return;
  const color = CV_ACCENTS.pose;
  poses.forEach((pose) => {
    const kps = {};
    pose.keypoints.forEach((kp) => { kps[kp.name] = kp; });
    // Skeleton lines
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    SKELETON.forEach(([a, b]) => {
      const kpA = kps[a], kpB = kps[b];
      if (kpA && kpB && (kpA.score || 0) > 0.25 && (kpB.score || 0) > 0.25) {
        ctx.beginPath();
        ctx.moveTo(kpA.x, kpA.y);
        ctx.lineTo(kpB.x, kpB.y);
        ctx.stroke();
      }
    });
    // Keypoints
    pose.keypoints.forEach((kp) => {
      if ((kp.score || 0) > 0.25) {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    });
  });
}

// ── Classification overlay ───────────────────────────────────────────────────
export function drawClassification(ctx, predictions) {
  if (!predictions || !predictions.length) return;
  const color = CV_ACCENTS.classify;
  const top3 = predictions.slice(0, 3);
  // semi-transparent bg panel
  const panelH = top3.length * 26 + 12;
  ctx.fillStyle = "rgba(0,0,0,0.65)";
  ctx.fillRect(8, 8, 260, panelH);
  ctx.strokeStyle = color + "66";
  ctx.lineWidth = 1;
  ctx.strokeRect(8, 8, 260, panelH);
  top3.forEach((pred, i) => {
    const pct = (pred.probability * 100).toFixed(1);
    const barW = Math.round(pred.probability * 220);
    // bar bg
    ctx.fillStyle = color + "22";
    ctx.fillRect(14, 18 + i * 26, 220, 14);
    // bar fill
    ctx.fillStyle = color + "88";
    ctx.fillRect(14, 18 + i * 26, barW, 14);
    // label
    ctx.fillStyle = "#fff";
    ctx.font = `500 11px ${MONO_FONT}`;
    const name = pred.className.split(",")[0].trim();
    ctx.fillText(`${name.substring(0, 24)} ${pct}%`, 16, 28 + i * 26);
  });
}

// ── Segmentation mask (for canvas-level drawing) ─────────────────────────────
export function drawSegmentationMask(ctx, segmentation, canvasW, canvasH) {
  if (!segmentation) return;
  const { data, width, height } = segmentation;
  const imageData = ctx.createImageData(width, height);
  const [r, g, b] = [0, 229, 255]; // cyan
  for (let i = 0; i < data.length; i++) {
    if (data[i] === 1) {
      imageData.data[i * 4] = r;
      imageData.data[i * 4 + 1] = g;
      imageData.data[i * 4 + 2] = b;
      imageData.data[i * 4 + 3] = 160; // semi-transparent
    }
  }
  // Put on offscreen then scale to canvas
  const offscreen = document.createElement("canvas");
  offscreen.width = width;
  offscreen.height = height;
  offscreen.getContext("2d").putImageData(imageData, 0, 0);
  ctx.save();
  ctx.globalAlpha = 0.7;
  ctx.drawImage(offscreen, 0, 0, canvasW, canvasH);
  ctx.restore();
}
