/**
 * COCO dataset 80 class names — in index order.
 * Used for post-processing YOLO detection outputs.
 */
export const COCO_CLASSES = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
  "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
  "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
  "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
  "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
  "hair drier", "toothbrush",
];

/** Number of COCO classes */
export const COCO_NUM_CLASSES = COCO_CLASSES.length; // 80

/**
 * Per-class colour palette — deterministic, visually distinct.
 * Based on a stepped HSL wheel.
 */
export const CLASS_COLORS = COCO_CLASSES.map((_, i) => {
  const hue = (i * 137.508) % 360; // golden-angle step
  return `hsl(${hue.toFixed(1)}, 85%, 50%)`;
});

export function colorForClass(classIndex) {
  return CLASS_COLORS[classIndex % CLASS_COLORS.length];
}
