/**
 * Deplyze Vision — Canvas drawing utilities
 * All coordinates are normalised [0, 1] — call scaleBox() to get pixel coords.
 */

export const CV_ACCENTS = {
  detect:   "#00FF41",
  seg:      "#00E5FF",
  pose:     "#FF3366",
  track:    "#FFD500",
  classify: "#B026FF",
  obb:      "#FF8C00",
};

const MONO_FONT = '"JetBrains Mono", "Fira Code", monospace';

/** Convert normalised [x1,y1,x2,y2] → pixel [x,y,w,h] for a given canvas */
export function normToPixel([x1, y1, x2, y2], cw, ch) {
  return [x1 * cw, y1 * ch, (x2 - x1) * cw, (y2 - y1) * ch];
}

function labelBox(ctx, label, px, py, color) {
  const x = Math.max(0, px);
  const y = Math.max(20, py);
  ctx.font = `600 12px ${MONO_FONT}`;
  const tw = ctx.measureText(label).width;
  ctx.fillStyle = color;
  ctx.fillRect(x, y - 20, tw + 10, 20);
  ctx.fillStyle = "#000";
  ctx.fillText(label, x + 5, y - 5);
}

// ── Detection ────────────────────────────────────────────────────────────────
export function drawDetections(ctx, detections, canvasW, canvasH) {
  if (!detections?.length) return;
  detections.forEach((det) => {
    const color = det.color || CV_ACCENTS.detect;
    const [x, y, w, h] = normToPixel(det.bbox, canvasW, canvasH);
    ctx.fillStyle = color + "22";
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.strokeRect(x, y, w, h);
    labelBox(ctx, `${det.class} ${(det.score * 100).toFixed(0)}%`, x, y, color);
  });
}

// ── OBB (oriented bounding box) ──────────────────────────────────────────────
export function drawOBB(ctx, detections, canvasW, canvasH) {
  if (!detections?.length) return;
  detections.forEach((det) => {
    const color = det.color || CV_ACCENTS.obb;
    const cx = det.cx * canvasW;
    const cy = det.cy * canvasH;
    const hw = (det.bw / 2) * canvasW;
    const hh = (det.bh / 2) * canvasH;
    const angle = det.angle || 0;

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);
    ctx.fillStyle = color + "22";
    ctx.fillRect(-hw, -hh, hw * 2, hh * 2);
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.strokeRect(-hw, -hh, hw * 2, hh * 2);
    ctx.restore();

    // Label at top-left corner of bbox (approx)
    const [x, y] = [cx - hw, cy - hh];
    labelBox(ctx, `${det.class} ${(det.score * 100).toFixed(0)}%`, x, y, color);
  });
}

// ── Tracking ─────────────────────────────────────────────────────────────────
export function drawTracking(ctx, trackedObjects, canvasW, canvasH) {
  if (!trackedObjects) return;
  Object.entries(trackedObjects).forEach(([id, obj]) => {
    if (!obj.det) return;
    const color = obj.det.color || CV_ACCENTS.track;
    const [x, y, w, h] = normToPixel(obj.det.bbox, canvasW, canvasH);
    ctx.fillStyle = color + "1A";
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.strokeRect(x, y, w, h);
    // Centroid dot
    const cxPx = obj.cx * canvasW;
    const cyPx = obj.cy * canvasH;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(cxPx, cyPx, 4, 0, 2 * Math.PI);
    ctx.fill();
    labelBox(ctx, `#${id} ${obj.det.class}`, x, y, color);
  });
}

// ── Pose skeleton (COCO 17 keypoints) ────────────────────────────────────────
import { POSE_SKELETON } from "./yoloInference";

export function drawPose(ctx, poses, canvasW, canvasH) {
  if (!poses?.length) return;
  const color = CV_ACCENTS.pose;
  poses.forEach((pose) => {
    const kps = pose.keypoints || [];
    // Skeleton lines
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    POSE_SKELETON.forEach(([ai, bi]) => {
      const kpA = kps[ai];
      const kpB = kps[bi];
      if (kpA && kpB && (kpA.score ?? 0) > 0.25 && (kpB.score ?? 0) > 0.25) {
        ctx.beginPath();
        ctx.moveTo(kpA.x * canvasW, kpA.y * canvasH);
        ctx.lineTo(kpB.x * canvasW, kpB.y * canvasH);
        ctx.stroke();
      }
    });
    // Keypoint dots
    kps.forEach((kp) => {
      if ((kp.score ?? 0) > 0.25) {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(kp.x * canvasW, kp.y * canvasH, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    });
  });
}

// ── Segmentation masks ────────────────────────────────────────────────────────
export function drawSegmentation(ctx, detections, canvasW, canvasH) {
  if (!detections?.length) return;
  detections.forEach((det) => {
    if (!det.mask) {
      // Fallback: draw bounding box
      drawDetections(ctx, [det], canvasW, canvasH);
      return;
    }
    const { data, width, height } = det.mask;
    const color = det.color || CV_ACCENTS.seg;
    const [r, g, b] = hexToRgb(color);

    const imageData = ctx.createImageData(width, height);
    for (let i = 0; i < data.length; i++) {
      if (data[i]) {
        imageData.data[i * 4 + 0] = r;
        imageData.data[i * 4 + 1] = g;
        imageData.data[i * 4 + 2] = b;
        imageData.data[i * 4 + 3] = 150;
      }
    }
    const offscreen = document.createElement("canvas");
    offscreen.width = width;
    offscreen.height = height;
    offscreen.getContext("2d").putImageData(imageData, 0, 0);
    ctx.save();
    ctx.globalAlpha = 0.8;
    ctx.drawImage(offscreen, 0, 0, canvasW, canvasH);
    ctx.restore();

    // Also draw bounding box + label
    const [x, y, w, h] = normToPixel(det.bbox, canvasW, canvasH);
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.strokeRect(x, y, w, h);
    labelBox(ctx, `${det.class} ${(det.score * 100).toFixed(0)}%`, x, y, color);
  });
}

// ── Classification overlay ────────────────────────────────────────────────────
export function drawClassification(ctx, predictions) {
  if (!predictions?.length) return;
  const color = CV_ACCENTS.classify;
  const top5 = predictions.slice(0, 5);
  const panelH = top5.length * 26 + 16;
  ctx.fillStyle = "rgba(0,0,0,0.72)";
  ctx.fillRect(8, 8, 280, panelH);
  ctx.strokeStyle = color + "66";
  ctx.lineWidth = 1;
  ctx.strokeRect(8, 8, 280, panelH);

  top5.forEach((pred, i) => {
    const pct = (pred.score * 100).toFixed(1);
    const barW = Math.round(pred.score * 240);
    ctx.fillStyle = color + "22";
    ctx.fillRect(14, 20 + i * 26, 240, 14);
    ctx.fillStyle = pred.color || color + "88";
    ctx.fillRect(14, 20 + i * 26, barW, 14);
    ctx.fillStyle = "#fff";
    ctx.font = `500 11px ${MONO_FONT}`;
    const name = (pred.class || "").split(",")[0].trim().slice(0, 26);
    ctx.fillText(`${name}  ${pct}%`, 16, 31 + i * 26);
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function hexToRgb(hex) {
  // Handles #RRGGBB and hsl(...) strings
  if (hex.startsWith("hsl")) {
    const tmp = document.createElement("div");
    tmp.style.color = hex;
    document.body.appendChild(tmp);
    const c = getComputedStyle(tmp).color;
    document.body.removeChild(tmp);
    const m = c.match(/\d+/g);
    return m ? [+m[0], +m[1], +m[2]] : [0, 229, 255];
  }
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Master draw dispatcher */
export function drawResults(ctx, task, results, canvasW, canvasH, options = {}) {
  if (options.clear !== false) {
    ctx.clearRect(0, 0, canvasW, canvasH);
  }
  switch (task) {
    case "detect":    drawDetections(ctx, results, canvasW, canvasH); break;
    case "seg":       drawSegmentation(ctx, results, canvasW, canvasH); break;
    case "pose":      drawPose(ctx, results, canvasW, canvasH); break;
    case "obb":       drawOBB(ctx, results, canvasW, canvasH); break;
    case "classify":  drawClassification(ctx, results); break;
    case "track":     drawTracking(ctx, results, canvasW, canvasH); break;
    default: break;
  }
}
