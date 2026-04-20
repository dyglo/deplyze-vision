/**
 * Deplyze Vision — Benchmark
 * Compare Models by parameters, mAP, and historical speed data.
 */
import { useState, useEffect, useMemo } from "react";
import axios from "axios";
import {
  BoundingBox, PersonSimpleRun, Intersect, Tag, Path, Crosshair,
  ChartLineUp, Lightning, Timer, Target, Columns, Scales, Star,
  Database, HardDrives
} from "@phosphor-icons/react";

import { YOLO_MODELS, TASK_META } from "../utils/yoloModels";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const TASK_ICONS = {
  detect: BoundingBox,
  seg: Intersect,
  pose: PersonSimpleRun,
  obb: Crosshair,
  classify: Tag,
  track: Path,
};

function PerfBar({ value, max, color, suffix = "" }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  return (
    <div className="flex items-center gap-3">
      <div className="flex-1 h-1.5 bg-[#EDE9E0] rounded-full overflow-hidden flex items-center justify-end">
        {/* We can make it right-aligned or left-aligned depending. Let's just do left-aligned. */}
        <div className="h-full w-full rounded-full transition-all duration-500 origin-left" style={{ transform: `scaleX(${pct / 100})`, backgroundColor: color }} />
      </div>
      <span className="font-mono text-[10px] tabular-nums text-[#8A8580] w-12 text-right">
        {Number.isFinite(value) ? value.toFixed(1) : value}{suffix}
      </span>
    </div>
  );
}

export default function Benchmark() {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTask, setActiveTask] = useState("detect");

  useEffect(() => {
    // Fetch all runs to calculate average local FPS per model
    axios.get(`${API}/runs?limit=1000`)
      .then(res => setRuns(res.data))
      .catch(err => console.error(err))
      .finally(() => setLoading(false));
  }, []);

  // Compute aggregate stats per model
  const modelStats = useMemo(() => {
    const stats = {};
    for (const r of runs) {
      if (!r.stats?.fps || !r.model_name) continue;
      // Extract model name (handle "Model A vs Model B" cases from compare mode)
      const names = r.model_name.split(" vs ");
      for (const rawName of names) {
        const name = rawName.replace("(custom)", "").trim();
        if (!stats[name]) stats[name] = { sumFps: 0, sumLat: 0, count: 0 };
        stats[name].sumFps += r.stats.fps;
        stats[name].sumLat += r.stats.latency;
        stats[name].count += 1;
      }
    }
    const avg = {};
    for (const [name, d] of Object.entries(stats)) {
      avg[name] = {
        fps: d.sumFps / d.count,
        latency: d.sumLat / d.count,
        runs: d.count,
      };
    }
    return avg;
  }, [runs]);

  // Combine YOLO_MODELS static data with dynamic run data
  const comparisonData = useMemo(() => {
    const taskModels = YOLO_MODELS.filter(m => m.task === activeTask);
    return taskModels.map(m => {
      const ms = modelStats[m.name];
      return {
        ...m,
        avgFps: ms?.fps || 0,
        avgLatency: ms?.latency || 0,
        runCount: ms?.runs || 0,
        // Parsed params (e.g. "3.2M" -> 3.2)
        paramsParsed: parseFloat(m.params) || 0,
        mapParsed: parseFloat(m.mapCOCO) || parseFloat(m.mapDOTA) || 0,
      };
    }).sort((a, b) => b.mapParsed - a.mapParsed); // sort by accuracy by default
  }, [activeTask, modelStats]);

  const maxMap = Math.max(...comparisonData.map(m => m.mapParsed), 50);
  const maxParams = Math.max(...comparisonData.map(m => m.paramsParsed), 10);
  const maxFps = Math.max(...comparisonData.map(m => m.avgFps), 30);

  if (loading) {
    return (
      <div className="flex h-[calc(100vh-56px)] items-center justify-center bg-[#F8F7F2]">
        <div className="animate-pulse flex items-center gap-2 text-sm font-mono text-[#B1ADA1]">
          <Database className="animate-spin" /> Gathering benchmark telemetry...
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-6xl mx-auto min-h-[calc(100vh-56px)]">
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="font-['Outfit'] text-2xl font-medium text-[#141413] tracking-tight">Model Benchmark</h1>
          <p className="text-sm text-[#8A8580] mt-1">
            Compare model architectures by theoretical accuracy (mAP), size parameters, and real-world local performance.
          </p>
        </div>
        <div className="px-4 py-2 bg-[#FFF8F0] border border-[#FFD9B5] rounded-sm text-xs font-mono text-[#C15F3C] flex items-center gap-2">
          <Lightning weight="fill" /> Local execution metrics are averaged across all Studio runs.
        </div>
      </div>

      {/* Task Filters */}
      <div className="flex border-b border-[#DDD9D0] mb-8">
        {Object.entries(TASK_META).map(([task, meta]) => {
          const Icon = TASK_ICONS[task];
          const active = activeTask === task;
          return (
            <button
              key={task}
              onClick={() => setActiveTask(task)}
              className={`flex items-center gap-2 px-6 py-3 text-xs font-mono tracking-wider uppercase border-b-2 transition-colors ${
                active ? "border-[currentColor] bg-white" : "border-transparent text-[#8A8580] hover:bg-white/50 hover:text-[#141413]"
              }`}
              style={{ color: active ? meta.color : undefined }}
            >
              <Icon size={16} weight={active ? "bold" : "regular"} />
              {meta.shortLabel}
            </button>
          );
        })}
      </div>

      {comparisonData.length === 0 ? (
        <div className="text-center py-20 text-[#8A8580] font-mono text-sm border border-dashed border-[#DDD9D0] rounded-sm bg-white/50">
          No built-in models configured for this task yet.
        </div>
      ) : (
        <div className="bg-white border border-[#DDD9D0] rounded-sm overflow-hidden shadow-sm">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-[#F8F7F2] border-b border-[#DDD9D0]">
                <th className="px-4 py-3 text-[10px] font-mono text-[#8A8580] uppercase tracking-wider font-semibold w-1/5">Architecture</th>
                <th className="px-4 py-3 text-[10px] font-mono text-[#8A8580] uppercase tracking-wider font-semibold w-[15%]">
                  <div className="flex items-center gap-1"><HardDrives size={12}/> Size / Params</div>
                </th>
                <th className="px-4 py-3 text-[10px] font-mono text-[#8A8580] uppercase tracking-wider font-semibold w-[20%]">
                  <div className="flex items-center gap-1"><Target size={12}/> Theoretical mAP</div>
                </th>
                <th className="px-4 py-3 text-[10px] font-mono text-[#8A8580] uppercase tracking-wider font-semibold w-[20%]">
                  <div className="flex items-center gap-1"><Lightning size={12}/> Local Avg FPS</div>
                </th>
                <th className="px-4 py-3 text-[10px] font-mono text-[#8A8580] uppercase tracking-wider font-semibold w-1/4">Notes</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#F4F3EE]">
              {comparisonData.map((model, idx) => (
                <tr key={model.id} className="hover:bg-[#F8F7F2]/50 transition-colors group">
                  <td className="px-4 py-4">
                    <div className="flex gap-2 items-center">
                      {idx === 0 && <Star weight="fill" className="text-[#B08000]" size={14} title="Top Accuracy" />}
                      {idx !== 0 && <div className="w-3.5" />}
                      <div>
                        <div className="font-['Outfit'] font-semibold text-[#141413] flex items-center gap-2">
                          {model.name}
                          <span className="px-1.5 py-0.5 text-[9px] bg-[#141413] text-[#F4F3EE] rounded-sm font-mono opacity-80 group-hover:opacity-100 transition-opacity">
                            {model.family}
                          </span>
                        </div>
                        <div className="text-[10px] font-mono text-[#8A8580] mt-0.5">{model.sizeApprox} export</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-4 align-middle">
                    <PerfBar value={model.paramsParsed} max={maxParams} color="#4A4742" suffix="M" />
                  </td>
                  <td className="px-4 py-4 align-middle">
                    <PerfBar value={model.mapParsed} max={maxMap} color="#0087B3" />
                  </td>
                  <td className="px-4 py-4 align-middle">
                    {model.avgFps > 0 ? (
                      <PerfBar value={model.avgFps} max={maxFps} color="#008B22" />
                    ) : (
                      <span className="text-[10px] font-mono text-[#B1ADA1] block text-center bg-[#F4F3EE] rounded py-0.5">Untested</span>
                    )}
                  </td>
                  <td className="px-4 py-4 align-middle">
                    <p className="text-[10px] text-[#5C5751] leading-relaxed max-w-[200px]">
                      {model.description}
                    </p>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="px-4 py-3 bg-[#F8F7F2] border-t border-[#DDD9D0] text-[10px] font-mono text-[#8A8580] flex justify-between">
            <span>YOLO architecture performance varies greatly based on CPU, RAM, and WebGL backend limits.</span>
            <span>Based on {runs.length} recorded Studio runs.</span>
          </div>
        </div>
      )}

      {/* Feature matrix */}
      <div className="mt-12">
        <h2 className="font-['Outfit'] text-lg font-medium text-[#141413] mb-4">Architecture Matrix</h2>
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-[#141413] p-4 rounded-sm border border-[#2A2925] text-white">
            <h3 className="font-mono text-sm mb-2 text-[#00FF41]">YOLOv10</h3>
            <p className="text-[11px] leading-relaxed opacity-80">Introduced spatial-channel decoupled downsampling and NMS-free training. Superior inference speed as it requires absolutely zero post-processing Non-Maximum Suppression.</p>
          </div>
          <div className="bg-white p-4 rounded-sm border border-[#DDD9D0]">
            <h3 className="font-mono text-sm mb-2 text-[#6B21A8] font-semibold">YOLOv8</h3>
            <p className="text-[11px] leading-relaxed text-[#5C5751]">The current industry standard. Anchor-free detection with new loss functions. Perfect balance of accuracy vs parameters, but relies heavily on post-NMS.</p>
          </div>
          <div className="bg-[#FFF8F0] border-[#FFD9B5] p-4 rounded-sm border">
            <h3 className="font-mono text-sm mb-2 text-[#C15F3C] font-semibold">YOLO26 / Ultralytics</h3>
            <p className="text-[11px] leading-relaxed text-[#5C5751]">The latest internal iterations. Combines NMS-free paradigms with attention mechanisms while retaining extremely high zero-shot transfer capabilities.</p>
          </div>
          <div className="bg-[#F4F3EE] p-4 rounded-sm border border-[#DDD9D0]">
            <h3 className="font-mono text-sm mb-2 text-[#4A4742] font-semibold">YOLOv5</h3>
            <p className="text-[11px] leading-relaxed text-[#5C5751]">The legacy baseline. Anchor-based detection. Still retains the absolute smallest nano sizes and best fallback support for older TF.js WASM backends.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
