/**
 * Deplyze Vision - YOLO26 TF.js Model Registry
 *
 * Every built-in entry points at a committed Ultralytics TF.js export folder
 * under public/models. These URLs are loaded with tf.loadGraphModel().
 */

import { COCO_CLASSES } from "./cocoClasses";

export const TASK_META = {
  detect: {
    label: "Detection",
    shortLabel: "DETECT",
    color: "#008B22",
    description: "Detect and localise objects with bounding boxes",
  },
  seg: {
    label: "Segmentation",
    shortLabel: "SEGMENT",
    color: "#0087B3",
    description: "Pixel-level instance segmentation masks",
  },
  pose: {
    label: "Pose",
    shortLabel: "POSE",
    color: "#CC1144",
    description: "Skeleton keypoint estimation (17 points, COCO format)",
  },
  obb: {
    label: "OBB",
    shortLabel: "OBB",
    color: "#B08000",
    description: "Oriented bounding boxes for rotated objects",
  },
  classify: {
    label: "Classify",
    shortLabel: "CLASSIFY",
    color: "#7B1CC4",
    description: "Whole-image classification (ImageNet / custom)",
  },
  track: {
    label: "Track",
    shortLabel: "TRACK",
    color: "#C15F3C",
    description: "Multi-object tracking using centroid association",
  },
};

export const DEFAULT_INPUT_SIZE = 640;

export const DOTA_CLASSES = [
  "plane",
  "ship",
  "storage tank",
  "baseball diamond",
  "tennis court",
  "basketball court",
  "ground track field",
  "harbor",
  "bridge",
  "large vehicle",
  "small vehicle",
  "helicopter",
  "swimming pool",
  "roundabout",
  "soccer ball field",
  "container crane",
];

const IMAGENET_CLASSES = Array.from({ length: 1000 }, (_, i) => `imagenet_${i}`);

export const YOLO_MODELS = [
  {
    id: "yolov8n-onnx-det",
    name: "YOLOv8n ONNX",
    family: "YOLOv8",
    variant: "nano",
    task: "detect",
    runtime: "onnx",
    inputSize: 640,
    classes: COCO_CLASSES,
    numClasses: 80,
    params: "3.2M",
    flops: "8.7G",
    mapCOCO: 37.3,
    speed: "browser WASM",
    sizeApprox: "12 MB",
    isBuiltin: true,
    requiresHosting: false,
    exportCmd: "yolo export model=yolov8n.pt format=onnx imgsz=640",
    description: "YOLOv8 nano object detection running in-browser with ONNX Runtime Web.",
    url: "/models/yolov8n_onnx/yolov8n.onnx",
  },
  {
    id: "yolo26n-det",
    name: "YOLO26n",
    family: "YOLO26",
    variant: "nano",
    task: "detect",
    inputSize: 640,
    classes: COCO_CLASSES,
    numClasses: 80,
    params: "2.8M",
    flops: "8.4G",
    mapCOCO: 39.2,
    speed: "~6ms",
    sizeApprox: "10 MB",
    isBuiltin: true,
    requiresHosting: false,
    exportCmd: "yolo export model=yolo26n.pt format=tfjs imgsz=640",
    description: "YOLO26 nano object detection exported to TensorFlow.js.",
    url: "/models/yolo26n_web_model/model.json",
    metadataUrl: "/models/yolo26n_web_model/metadata.yaml",
  },
  {
    id: "yolo26n-seg",
    name: "YOLO26n-seg",
    family: "YOLO26",
    variant: "nano",
    task: "seg",
    inputSize: 640,
    classes: COCO_CLASSES,
    numClasses: 80,
    maskCoefficients: 32,
    params: "3.0M",
    mapCOCO: 32.1,
    speed: "~8ms",
    sizeApprox: "11 MB",
    isBuiltin: true,
    requiresHosting: false,
    exportCmd: "yolo export model=yolo26n-seg.pt format=tfjs imgsz=640",
    description: "YOLO26 nano instance segmentation exported to TensorFlow.js.",
    url: "/models/yolo26n-seg_web_model/model.json",
    metadataUrl: "/models/yolo26n-seg_web_model/metadata.yaml",
  },
  {
    id: "yolo26n-pose",
    name: "YOLO26n-pose",
    family: "YOLO26",
    variant: "nano",
    task: "pose",
    inputSize: 640,
    classes: ["person"],
    numClasses: 1,
    numKeypoints: 17,
    params: "3.1M",
    mapCOCO: 51.2,
    speed: "~7ms",
    sizeApprox: "12 MB",
    isBuiltin: true,
    requiresHosting: false,
    exportCmd: "yolo export model=yolo26n-pose.pt format=tfjs imgsz=640",
    description: "YOLO26 nano human pose model with 17 COCO keypoints.",
    url: "/models/yolo26n-pose_web_model/model.json",
    metadataUrl: "/models/yolo26n-pose_web_model/metadata.yaml",
  },
  {
    id: "yolo26n-obb",
    name: "YOLO26n-obb",
    family: "YOLO26",
    variant: "nano",
    task: "obb",
    inputSize: 640,
    classes: DOTA_CLASSES,
    numClasses: 16,
    params: "3.1M",
    mapDOTA: 78.6,
    speed: "~7ms",
    sizeApprox: "10 MB",
    isBuiltin: true,
    requiresHosting: false,
    exportCmd: "yolo export model=yolo26n-obb.pt format=tfjs imgsz=640",
    description: "YOLO26 nano oriented bounding boxes for rotated targets.",
    url: "/models/yolo26n-obb_web_model/model.json",
    metadataUrl: "/models/yolo26n-obb_web_model/metadata.yaml",
  },
  {
    id: "yolo26n-cls",
    name: "YOLO26n-cls",
    family: "YOLO26",
    variant: "nano",
    task: "classify",
    inputSize: 224,
    classes: IMAGENET_CLASSES,
    numClasses: 1000,
    params: "2.8M",
    speed: "~4ms",
    sizeApprox: "11 MB",
    isBuiltin: true,
    requiresHosting: false,
    exportCmd: "yolo export model=yolo26n-cls.pt format=tfjs",
    description: "YOLO26 nano ImageNet classification exported to TensorFlow.js.",
    url: "/models/yolo26n-cls_web_model/model.json",
    metadataUrl: "/models/yolo26n-cls_web_model/metadata.yaml",
  },
];

export function getModelById(id) {
  return YOLO_MODELS.find((m) => m.id === id) || null;
}

export function getModelsForTask(task) {
  return YOLO_MODELS.filter((m) => m.task === task || (task === "track" && m.task === "detect"));
}

export const YOLO_FAMILIES = ["YOLOv8", "YOLO26"];

export const EXPORT_GUIDE = `# Export a YOLO26 model to TF.js

pip install ultralytics tensorflowjs

# Detection
yolo export model=yolo26n.pt format=tfjs imgsz=640

# Segmentation
yolo export model=yolo26n-seg.pt format=tfjs imgsz=640

# Pose
yolo export model=yolo26n-pose.pt format=tfjs imgsz=640

# OBB
yolo export model=yolo26n-obb.pt format=tfjs imgsz=640

# Classification
yolo export model=yolo26n-cls.pt format=tfjs

# Then host the produced *_web_model/ folder on any static host
# and register the model.json URL in Model Hub.`;
