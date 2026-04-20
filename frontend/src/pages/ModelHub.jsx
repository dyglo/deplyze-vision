/**
 * Deplyze Vision — Model Hub
 * Browse all YOLO models, register custom model URLs, open export guide.
 */
import { useState, useEffect, useCallback } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import axios from "axios";
import {
  Cube, Plus, X, Check, Copy, ArrowRight, Warning,
  BoundingBox, Intersect, PersonSimpleRun, Crosshair, Tag, Path,
  ArrowSquareOut, Trash, Code, Spinner, Terminal,
} from "@phosphor-icons/react";

import { YOLO_MODELS, YOLO_FAMILIES, TASK_META, EXPORT_GUIDE } from "../utils/yoloModels";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const TASK_ICONS = {
  detect:   BoundingBox,
  seg:      Intersect,
  pose:     PersonSimpleRun,
  obb:      Crosshair,
  classify: Tag,
  track:    Path,
};

// ─── Small components ─────────────────────────────────────────────────────────

function TaskPill({ task }) {
  const meta = TASK_META[task] || { label: task, color: "#8A8580" };
  const Icon = TASK_ICONS[task] || Cube;
  return (
    <span
      className="inline-flex items-center gap-1 px-1.5 py-0.5 text-[10px] font-mono rounded-sm border"
      style={{
        color: meta.color,
        borderColor: meta.color + "40",
        backgroundColor: meta.color + "12",
      }}
    >
      <Icon size={10} />
      {meta.shortLabel || meta.label}
    </span>
  );
}

function FamilyBadge({ family }) {
  const colors = {
    YOLO26:  { bg: "#008B22", label: "YOLO26 ✦" },
    YOLOv10: { bg: "#007599", label: "v10" },
    YOLOv8:  { bg: "#6B21A8", label: "v8" },
    YOLOv5:  { bg: "#B08000", label: "v5" },
    Custom:  { bg: "#C15F3C", label: "Custom" },
  };
  const c = colors[family] || { bg: "#8A8580", label: family };
  return (
    <span
      className="inline-block px-1.5 py-0.5 text-[9px] font-mono rounded-sm text-white"
      style={{ backgroundColor: c.bg }}
    >
      {c.label}
    </span>
  );
}

function MetricPill({ label, value }) {
  if (!value) return null;
  return (
    <span className="text-[10px] font-mono text-[#8A8580]">
      <span className="text-[#B1ADA1]">{label} </span>{value}
    </span>
  );
}

function CodeBlock({ code }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <div className="relative group">
      <pre className="bg-[#F0EFE9] border border-[#DDD9D0] text-[#141413] text-[10px] font-mono p-3 rounded-sm overflow-x-auto leading-relaxed whitespace-pre-wrap">
        {code}
      </pre>
      <button
        onClick={copy}
        className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1 bg-white border border-[#DDD9D0] rounded-sm text-[#8A8580] hover:text-[#141413]"
      >
        {copied ? <Check size={11} /> : <Copy size={11} />}
      </button>
    </div>
  );
}

// ─── Model card ────────────────────────────────────────────────────────────────
function ModelCard({ model, onUseInStudio }) {
  const url = model.url || "";
  const hasUrl = !!url;

  return (
    <div className="border rounded-sm bg-white transition-all border-[#DDD9D0] hover:border-[#B1ADA1]">
      <div className="p-4 flex items-start gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <span className="font-['Outfit'] text-sm font-semibold text-[#141413]">{model.name}</span>
            <FamilyBadge family={model.family} />
            <TaskPill task={model.task} />
            {hasUrl && (
              <span className="flex items-center gap-1 text-[9px] font-mono text-[#008B22]">
                <div className="w-1.5 h-1.5 rounded-full bg-[#008B22]" /> Ready
              </span>
            )}
          </div>
          <p className="text-xs text-[#8A8580] leading-relaxed">{model.description}</p>
          <div className="flex flex-wrap gap-3 mt-2">
            {model.params && <MetricPill label="params" value={model.params} />}
            {model.mapCOCO && <MetricPill label="mAP" value={model.mapCOCO} />}
            {model.mapDOTA && <MetricPill label="mAP(DOTA)" value={model.mapDOTA} />}
            {model.sizeApprox && <MetricPill label="size" value={model.sizeApprox} />}
            {model.speed && <MetricPill label="speed" value={model.speed} />}
          </div>
        </div>
        <div className="flex flex-col items-end justify-center gap-2 flex-shrink-0">
          {hasUrl ? (
            <button
              data-testid={`use-studio-${model.id}`}
              onClick={(e) => { e.stopPropagation(); onUseInStudio(model); }}
              className="flex items-center gap-1 px-3 py-1.5 text-xs font-medium bg-[#141413] text-[#F4F3EE] rounded-sm hover:bg-[#2A2925] transition-colors"
            >
              Start in Studio <ArrowRight size={12} />
            </button>
          ) : (
            <span className="text-[10px] text-[#B1ADA1] font-mono italic">
              Model files pending...
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Main ModelHub ─────────────────────────────────────────────────────────────
export default function ModelHub() {
  const navigate = useNavigate();

  const [filterFamily, setFilterFamily] = useState("all");
  const [filterTask, setFilterTask] = useState("all");

  // Use in Studio
  const handleUseInStudio = (model) => {
    const taskParam = model.task === "track" ? "detect" : model.task;
    navigate(`/studio?task=${taskParam}&modelId=${model.id}`);
  };

  // Filter models
  const allModels = [...YOLO_MODELS];
  const filtered = allModels.filter((m) => {
    if (filterFamily !== "all" && m.family !== filterFamily) return false;
    if (filterTask !== "all" && m.task !== filterTask) return false;
    return true;
  });

  const readyCount = allModels.filter(m => !!m.url).length;

  return (
    <div className="p-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="font-['Outfit'] text-2xl font-medium text-[#141413] tracking-tight">Model Directory</h1>
          <p className="text-sm text-[#8A8580] mt-1">
            Browse available top-tier YOLO models ready to be used seamlessly in the Studio.
          </p>
        </div>
      </div>

      {/* Stats strip */}
      <div className="grid grid-cols-2 gap-px bg-[#DDD9D0] mb-6 border border-[#DDD9D0] rounded-sm overflow-hidden">
        {[
          { label: "Total Models", value: allModels.length },
          { label: "Ready to Run", value: readyCount },
        ].map((s) => (
          <div key={s.label} className="bg-[#F4F3EE] px-6 py-4">
            <div className="text-[10px] font-mono text-[#B1ADA1] uppercase tracking-wider mb-1">{s.label}</div>
            <div className="text-2xl font-['Outfit'] font-medium text-[#141413]">{s.value}</div>
          </div>
        ))}
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 mb-5">
        <div className="flex items-center gap-1">
          <span className="text-[10px] font-mono text-[#B1ADA1] mr-1">Family:</span>
          {["all", ...YOLO_FAMILIES].map((f) => (
            <button
              key={f}
              onClick={() => setFilterFamily(f)}
              className={`px-2 py-0.5 text-[10px] font-mono rounded-sm border transition-colors ${
                filterFamily === f
                  ? "bg-[#141413] text-[#F4F3EE] border-[#141413]"
                  : "border-[#DDD9D0] text-[#8A8580] hover:border-[#B1ADA1]"
              }`}
            >
              {f === "all" ? "All" : f}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1">
          <span className="text-[10px] font-mono text-[#B1ADA1] mr-1">Task:</span>
          {["all", ...Object.keys(TASK_META)].map((t) => (
            <button
              key={t}
              onClick={() => setFilterTask(t)}
              className={`px-2 py-0.5 text-[10px] font-mono rounded-sm border transition-colors ${
                filterTask === t
                  ? "bg-[#141413] text-[#F4F3EE] border-[#141413]"
                  : "border-[#DDD9D0] text-[#8A8580] hover:border-[#B1ADA1]"
              }`}
            >
              {t === "all" ? "All" : TASK_META[t]?.shortLabel || t}
            </button>
          ))}
        </div>
      </div>

      {/* Model cards */}
      <div className="space-y-4">
        {/* Group by family */}
        {(filterFamily === "all" ? YOLO_FAMILIES : [filterFamily]).map((family) => {
          const familyModels = filtered.filter((m) => m.family === family);
          if (!familyModels.length) return null;
          return (
            <div key={family}>
              <div className="flex items-center gap-2 mb-2">
                <h2 className="font-['Outfit'] text-sm font-semibold text-[#141413]">{family}</h2>
                <div className="flex-1 h-px bg-[#DDD9D0]" />
                <span className="text-[10px] font-mono text-[#B1ADA1]">{familyModels.length} models</span>
              </div>
              <div className="space-y-2">
                {familyModels.map((model) => (
                  <ModelCard
                    key={model.id}
                    model={model}
                    onUseInStudio={handleUseInStudio}
                  />
                ))}
              </div>
            </div>
          );
        })}
        {filtered.length === 0 && (
          <div className="text-center py-12 text-[#B1ADA1] text-sm font-mono">
            No models match the selected filters.
          </div>
        )}
      </div>
    </div>
  );
}
