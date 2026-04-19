import { useEffect, useState } from "react";

/**
 * HeroDemo — zero-runtime "live" detection animation.
 * Uses a static sample image + pre-computed bounding boxes that cycle through
 * 3 "frames" on a loop to showcase the CV platform without actually running TF.js
 * on the landing page (keeps bundle + first-paint snappy).
 */

// Normalized bounding boxes (x, y, w, h in 0..1) for each frame.
// Image: a city street scene — coordinates hand-tuned for the sample image.
const FRAMES = [
  {
    t: 48,
    detections: [
      { label: "person", score: 0.94, color: "#00FF41", x: 0.06, y: 0.42, w: 0.12, h: 0.44 },
      { label: "person", score: 0.91, color: "#00FF41", x: 0.29, y: 0.38, w: 0.14, h: 0.5 },
      { label: "car",    score: 0.89, color: "#00E5FF", x: 0.52, y: 0.48, w: 0.26, h: 0.28 },
      { label: "bicycle",score: 0.78, color: "#FFD500", x: 0.78, y: 0.58, w: 0.14, h: 0.22 },
    ],
  },
  {
    t: 42,
    detections: [
      { label: "person", score: 0.96, color: "#00FF41", x: 0.08, y: 0.40, w: 0.13, h: 0.46 },
      { label: "person", score: 0.93, color: "#00FF41", x: 0.31, y: 0.36, w: 0.15, h: 0.52 },
      { label: "person", score: 0.81, color: "#00FF41", x: 0.46, y: 0.44, w: 0.08, h: 0.36 },
      { label: "car",    score: 0.92, color: "#00E5FF", x: 0.55, y: 0.47, w: 0.27, h: 0.29 },
      { label: "traffic light", score: 0.84, color: "#B026FF", x: 0.86, y: 0.10, w: 0.06, h: 0.18 },
    ],
  },
  {
    t: 55,
    detections: [
      { label: "person", score: 0.95, color: "#00FF41", x: 0.10, y: 0.41, w: 0.12, h: 0.45 },
      { label: "person", score: 0.88, color: "#00FF41", x: 0.33, y: 0.39, w: 0.14, h: 0.5 },
      { label: "car",    score: 0.90, color: "#00E5FF", x: 0.56, y: 0.48, w: 0.26, h: 0.28 },
      { label: "dog",    score: 0.76, color: "#FF3366", x: 0.21, y: 0.71, w: 0.08, h: 0.14 },
      { label: "bicycle",score: 0.82, color: "#FFD500", x: 0.80, y: 0.59, w: 0.13, h: 0.22 },
    ],
  },
];

const SAMPLE_IMG = "https://images.unsplash.com/photo-1519121785383-3229633bb75b?auto=format&fit=crop&w=1200&q=70";

export default function HeroDemo() {
  const [frameIdx, setFrameIdx] = useState(0);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setFrameIdx((i) => (i + 1) % FRAMES.length);
      setTick((t) => t + 1);
    }, 1600);
    return () => clearInterval(id);
  }, []);

  const frame = FRAMES[frameIdx];

  return (
    <div data-testid="hero-live-demo" className="relative w-full max-w-[560px] aspect-[4/3] rounded-md overflow-hidden border border-[#DDD9D0] shadow-[0_20px_60px_-20px_rgba(20,20,19,0.25)] bg-[#E8E5DF]">
      {/* image */}
      <img
        src={SAMPLE_IMG}
        alt="Live in-browser YOLO detection demo"
        className="w-full h-full object-cover"
        draggable={false}
      />

      {/* grayscale scan overlay */}
      <div
        className="absolute inset-0 pointer-events-none mix-blend-multiply opacity-20"
        style={{ backgroundImage: "linear-gradient(#141413 1px, transparent 1px), linear-gradient(90deg, #141413 1px, transparent 1px)", backgroundSize: "32px 32px" }}
      />

      {/* SVG bbox layer */}
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 w-full h-full pointer-events-none" key={frameIdx}>
        {frame.detections.map((d, i) => (
          <g key={`${frameIdx}-${i}`} style={{ animation: `bbox-in 0.35s ease-out ${i * 60}ms both` }}>
            {/* corner brackets for CV feel */}
            <rect
              x={d.x * 100}
              y={d.y * 100}
              width={d.w * 100}
              height={d.h * 100}
              fill="none"
              stroke={d.color}
              strokeWidth="0.35"
              vectorEffect="non-scaling-stroke"
              strokeDasharray="3 2"
            />
            {/* label chip */}
            <g transform={`translate(${d.x * 100}, ${Math.max(0, d.y * 100 - 3.2)})`}>
              <rect width={Math.max(13, d.label.length * 1.6 + 5)} height="2.8" rx="0.3" fill={d.color} />
              <text x="0.7" y="2" fontSize="1.6" fontFamily="'JetBrains Mono', monospace" fill="#0A0A0A" fontWeight="600">
                {d.label} {(d.score * 100).toFixed(0)}%
              </text>
            </g>
          </g>
        ))}
      </svg>

      {/* top-left LIVE badge */}
      <div className="absolute top-3 left-3 flex items-center gap-2 px-2 py-1 rounded-sm bg-[#141413]/85 backdrop-blur-sm">
        <span className="relative flex h-1.5 w-1.5">
          <span className="absolute inline-flex h-full w-full rounded-full bg-[#00FF41] opacity-75 animate-ping" />
          <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-[#00FF41]" />
        </span>
        <span className="text-[10px] font-mono tracking-[0.2em] text-white uppercase">Live · TF.js</span>
      </div>

      {/* top-right stats */}
      <div className="absolute top-3 right-3 flex gap-1.5">
        {[
          { l: "FPS", v: frame.t, c: "#00FF41" },
          { l: "ms", v: Math.round(1000 / frame.t), c: "#00E5FF" },
          { l: "obj", v: frame.detections.length, c: "#FFD500" },
        ].map((s) => (
          <div key={s.l} className="bg-[#141413]/85 backdrop-blur-sm px-2 py-1 rounded-sm">
            <div className="text-[8px] font-mono text-white/50 uppercase tracking-wider leading-none">{s.l}</div>
            <div className="text-xs font-mono font-medium tabular-nums leading-tight" style={{ color: s.c }}>{s.v}</div>
          </div>
        ))}
      </div>

      {/* scanning line */}
      <div
        key={`scan-${tick}`}
        className="absolute left-0 right-0 h-[2px] pointer-events-none"
        style={{
          background: "linear-gradient(90deg, transparent, rgba(193,95,60,0.0) 10%, #C15F3C 50%, rgba(193,95,60,0.0) 90%, transparent)",
          animation: "hero-scan 1.6s linear",
          top: "0%",
          boxShadow: "0 0 12px rgba(193,95,60,0.6)",
        }}
      />

      {/* bottom caption */}
      <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between text-[10px] font-mono">
        <div className="flex items-center gap-2 px-2 py-1 bg-[#141413]/85 backdrop-blur-sm rounded-sm">
          <span className="text-white/60">MODEL</span>
          <span className="text-white">COCO-SSD Lite</span>
        </div>
        <div className="flex items-center gap-2 px-2 py-1 bg-[#141413]/85 backdrop-blur-sm rounded-sm">
          <span className="text-white/60">TASK</span>
          <span style={{ color: "#00FF41" }}>DETECT</span>
        </div>
      </div>

      {/* keyframes */}
      <style>{`
        @keyframes bbox-in {
          0%   { opacity: 0; transform: scale(0.96); }
          100% { opacity: 1; transform: scale(1); }
        }
        @keyframes hero-scan {
          0%   { top: 0%; opacity: 0; }
          8%   { opacity: 1; }
          92%  { opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
      `}</style>
    </div>
  );
}
