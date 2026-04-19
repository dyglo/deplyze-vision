import { useState, useEffect } from "react";
import axios from "axios";
import { Cube, Plus, Trash, X, BoundingBox, PersonSimpleRun, Intersect, Tag, Path, Check, ArrowSquareOut, Lock, GlobeHemisphereWest } from "@phosphor-icons/react";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;
const TASK_META = {
  detect:   { icon: BoundingBox,     color: "#008B22", label: "Detection" },
  pose:     { icon: PersonSimpleRun, color: "#CC1144", label: "Pose" },
  segment:  { icon: Intersect,       color: "#0087B3", label: "Segment" },
  classify: { icon: Tag,             color: "#7B1CC4", label: "Classify" },
  track:    { icon: Path,            color: "#B08000", label: "Track" },
};

const EXPORT_CODE = `from ultralytics import YOLO
model = YOLO("yolo26n.pt")
model.export(format="tfjs")
# Load in CV Platform → Model Hub`;

export default function ModelHub() {
  const [models, setModels] = useState([]);
  const [filterTask, setFilterTask] = useState("all");
  const [loading, setLoading] = useState(true);
  const [showAdd, setShowAdd] = useState(false);
  const [form, setForm] = useState({ name: "", task: "detect", url: "", description: "", labels: [] });
  const [labelInput, setLabelInput] = useState("");
  const [selected, setSelected] = useState(null);

  const fetchModels = async () => {
    setLoading(true);
    const r = await axios.get(`${API}/model-configs${filterTask !== "all" ? `?task=${filterTask}` : ""}`);
    setModels(r.data); setLoading(false);
  };
  useEffect(() => { fetchModels(); }, [filterTask]);

  const addModel = async () => {
    if (!form.name.trim() || !form.url.trim()) return;
    await axios.post(`${API}/model-configs`, form);
    setForm({ name: "", task: "detect", url: "", description: "", labels: [] }); setLabelInput(""); setShowAdd(false); fetchModels();
  };
  const deleteModel = async (id) => { try { await axios.delete(`${API}/model-configs/${id}`); if (selected?.id === id) setSelected(null); fetchModels(); } catch (e) { alert(e.response?.data?.detail || "Cannot delete model"); } };
  const addLabel = () => { const lbl = labelInput.trim(); if (!lbl || form.labels.includes(lbl)) return; setForm((p) => ({ ...p, labels: [...p.labels, lbl] })); setLabelInput(""); };

  const builtinModels = models.filter((m) => m.is_builtin);
  const customModels = models.filter((m) => !m.is_builtin);

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="font-['Outfit'] text-2xl font-medium text-[#141413] tracking-tight">Model Hub</h1>
          <p className="text-sm text-[#8A8580] mt-1">Browse built-in models or load your custom YOLO26 TF.js models</p>
        </div>
        <button data-testid="add-model-btn" onClick={() => setShowAdd(true)}
          className="flex items-center gap-2 px-4 py-2 bg-[#141413] text-[#F4F3EE] text-sm font-medium rounded-sm hover:bg-[#2A2925] transition-colors">
          <Plus size={14} weight="bold" />Add Custom Model
        </button>
      </div>

      {/* Task filter */}
      <div className="flex gap-1 mb-6 border-b border-[#DDD9D0] pb-4">
        {["all", ...Object.keys(TASK_META)].map((t) => {
          const meta = TASK_META[t];
          return (
            <button key={t} data-testid={`filter-${t}`} onClick={() => setFilterTask(t)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-sm text-xs transition-all ${filterTask === t ? "bg-white border border-[#DDD9D0] shadow-sm text-[#141413] font-medium" : "text-[#8A8580] hover:text-[#141413]"}`}>
              {meta && <meta.icon size={11} style={{ color: filterTask === t ? meta.color : undefined }} />}
              {t === "all" ? "All Tasks" : meta.label}
            </button>
          );
        })}
      </div>

      {/* Add model modal */}
      {showAdd && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#141413]/30">
          <div data-testid="add-model-modal" className="w-full max-w-lg bg-white border border-[#DDD9D0] rounded-sm p-6 shadow-lg max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-5">
              <h2 className="font-['Outfit'] text-lg font-medium text-[#141413]">Add Custom YOLO26 Model</h2>
              <button onClick={() => setShowAdd(false)} className="text-[#B1ADA1] hover:text-[#141413]"><X size={16} /></button>
            </div>
            <div className="mb-5 bg-[#F4F3EE] border border-[#DDD9D0] rounded-sm p-4">
              <div className="flex items-center gap-2 mb-2"><ArrowSquareOut size={12} className="text-[#C15F3C]" /><span className="text-[10px] font-mono text-[#B1ADA1] uppercase tracking-wider">Export your model first</span></div>
              <pre className="text-[10px] font-mono text-[#5C5751] leading-relaxed">{EXPORT_CODE}</pre>
            </div>
            <div className="space-y-4">
              <div>
                <label className="text-xs font-mono text-[#B1ADA1] uppercase tracking-wider block mb-1.5">Model Name *</label>
                <input data-testid="model-name-input" value={form.name} onChange={(e) => setForm((p) => ({ ...p, name: e.target.value }))} placeholder="e.g. YOLOv26n Custom"
                  className="w-full bg-[#F4F3EE] border border-[#DDD9D0] rounded-sm px-3 py-2 text-sm text-[#141413] placeholder-[#B1ADA1] focus:outline-none focus:border-[#B1ADA1] transition-colors" />
              </div>
              <div>
                <label className="text-xs font-mono text-[#B1ADA1] uppercase tracking-wider block mb-1.5">Task</label>
                <select value={form.task} onChange={(e) => setForm((p) => ({ ...p, task: e.target.value }))}
                  className="w-full bg-[#F4F3EE] border border-[#DDD9D0] rounded-sm px-3 py-2 text-sm text-[#141413] focus:outline-none focus:border-[#B1ADA1] transition-colors">
                  {Object.entries(TASK_META).map(([id, meta]) => <option key={id} value={id}>{meta.label}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs font-mono text-[#B1ADA1] uppercase tracking-wider block mb-1.5">Model URL (model.json) *</label>
                <input data-testid="model-url-input" value={form.url} onChange={(e) => setForm((p) => ({ ...p, url: e.target.value }))}
                  placeholder="https://your-host.com/yolo26n_web_model/model.json"
                  className="w-full bg-[#F4F3EE] border border-[#DDD9D0] rounded-sm px-3 py-2 text-sm text-[#141413] placeholder-[#B1ADA1] focus:outline-none focus:border-[#B1ADA1] font-mono transition-colors" />
                <p className="text-[10px] text-[#B1ADA1] mt-1">Must be a CORS-enabled URL hosting the TF.js model.json</p>
              </div>
              <div>
                <label className="text-xs font-mono text-[#B1ADA1] uppercase tracking-wider block mb-1.5">Class Labels</label>
                <div className="flex gap-2 mb-2">
                  <input value={labelInput} onChange={(e) => setLabelInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && addLabel()} placeholder="person, car, dog…"
                    className="flex-1 bg-[#F4F3EE] border border-[#DDD9D0] rounded-sm px-3 py-1.5 text-sm text-[#141413] placeholder-[#B1ADA1] focus:outline-none focus:border-[#B1ADA1] transition-colors" />
                  <button onClick={addLabel} className="px-3 py-1.5 bg-[#141413] text-[#F4F3EE] text-xs rounded-sm hover:bg-[#2A2925] transition-colors">Add</button>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {form.labels.map((lbl) => (
                    <span key={lbl} className="flex items-center gap-1 text-[10px] font-mono px-1.5 py-0.5 bg-[#F4F3EE] border border-[#DDD9D0] text-[#5C5751] rounded-sm">
                      {lbl}<button onClick={() => setForm((p) => ({ ...p, labels: p.labels.filter((l) => l !== lbl) }))} className="opacity-60 hover:opacity-100"><X size={9} /></button>
                    </span>
                  ))}
                </div>
              </div>
            </div>
            <div className="flex gap-3 mt-6">
              <button onClick={() => setShowAdd(false)} className="flex-1 py-2 text-sm border border-[#DDD9D0] text-[#8A8580] rounded-sm hover:border-[#B1ADA1]">Cancel</button>
              <button data-testid="confirm-add-model-btn" onClick={addModel} disabled={!form.name.trim() || !form.url.trim()}
                className="flex-1 py-2 text-sm bg-[#141413] text-[#F4F3EE] rounded-sm hover:bg-[#2A2925] disabled:opacity-50">Add Model</button>
            </div>
          </div>
        </div>
      )}

      {loading ? <div className="flex items-center justify-center h-48 text-[#B1ADA1] text-sm font-mono">Loading…</div> : (
        <>
          {builtinModels.length > 0 && (
            <div className="mb-8">
              <div className="flex items-center gap-2 mb-4"><Lock size={12} className="text-[#B1ADA1]" /><p className="text-[10px] font-mono text-[#B1ADA1] uppercase tracking-[0.2em]">Built-in Models</p><span className="text-[10px] font-mono text-[#DDD9D0]">({builtinModels.length})</span></div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {builtinModels.map((m) => {
                  const meta = TASK_META[m.task];
                  return (
                    <div key={m.id} data-testid={`model-card-${m.id}`}
                      className={`border bg-white p-5 hover:border-[#B1ADA1] transition-colors cursor-pointer rounded-sm ${selected?.id === m.id ? "border-[#C15F3C] shadow-sm" : "border-[#DDD9D0]"}`}
                      onClick={() => setSelected(selected?.id === m.id ? null : m)}>
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center gap-2">
                          {meta && <meta.icon size={16} style={{ color: meta.color }} weight="bold" />}
                          <div><div className="text-sm font-medium text-[#141413]">{m.name}</div><div className="text-[10px] font-mono text-[#B1ADA1] mt-0.5">{meta?.label} · Built-in</div></div>
                        </div>
                        {selected?.id === m.id && <Check size={14} className="text-[#008B22]" />}
                      </div>
                      <p className="text-xs text-[#8A8580] line-clamp-2 mb-3">{m.description}</p>
                      <div className="pt-3 border-t border-[#EDE9E0] flex items-center justify-between">
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded-sm" style={{ color: meta?.color, backgroundColor: (meta?.color || "#141413") + "12" }}>{meta?.label || m.task}</span>
                        <span className="text-[10px] font-mono text-[#B1ADA1]">TF.js WebGL</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <div>
            <div className="flex items-center gap-2 mb-4"><GlobeHemisphereWest size={12} className="text-[#B1ADA1]" /><p className="text-[10px] font-mono text-[#B1ADA1] uppercase tracking-[0.2em]">Custom Models</p><span className="text-[10px] font-mono text-[#DDD9D0]">({customModels.length})</span></div>
            {customModels.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-40 border border-dashed border-[#DDD9D0] rounded-sm bg-white">
                <Cube size={32} className="text-[#DDD9D0] mb-3" />
                <p className="text-[#8A8580] text-sm">No custom models added</p>
                <p className="text-[#B1ADA1] text-xs mt-1">Export YOLO26 to TF.js and load here</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {customModels.map((m) => {
                  const meta = TASK_META[m.task];
                  return (
                    <div key={m.id} data-testid={`custom-model-card-${m.id}`} className="border border-[#DDD9D0] bg-white p-5 hover:border-[#B1ADA1] transition-colors rounded-sm">
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center gap-2">{meta && <meta.icon size={16} style={{ color: meta.color }} />}<div><div className="text-sm font-medium text-[#141413]">{m.name}</div><div className="text-[10px] font-mono text-[#B1ADA1] mt-0.5">Custom · {meta?.label}</div></div></div>
                        <button data-testid={`delete-model-${m.id}`} onClick={() => deleteModel(m.id)} className="text-[#DDD9D0] hover:text-[#CC1144] transition-colors p-1"><Trash size={13} /></button>
                      </div>
                      {m.url && <p className="text-[10px] font-mono text-[#B1ADA1] truncate mb-2">{m.url}</p>}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Export guide */}
          <div className="mt-8 bg-white border border-[#DDD9D0] rounded-sm p-6">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-5 h-5 bg-[#C15F3C] rounded-sm flex items-center justify-center"><BoundingBox size={11} weight="bold" color="#fff" /></div>
              <h3 className="font-['Outfit'] text-base font-medium text-[#141413]">Export YOLO26 to TF.js</h3>
            </div>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <p className="text-sm text-[#5C5751] mb-3 leading-relaxed">Export any YOLO26 model to TensorFlow.js for in-browser WebGL inference. No server or GPU required.</p>
                <pre className="text-[11px] font-mono text-[#5C5751] bg-[#F4F3EE] border border-[#DDD9D0] p-4 rounded-sm leading-relaxed overflow-x-auto">{`from ultralytics import YOLO\nmodel = YOLO("yolo26n.pt")\nmodel.export(format="tfjs")\n# Creates: ./yolo26n_web_model/model.json`}</pre>
              </div>
              <div className="space-y-3">
                {[{ opt: "format='tfjs'", desc: "Target TensorFlow.js format" }, { opt: "imgsz=640", desc: "Input image size (default 640px)" }, { opt: "half=True", desc: "FP16 quantization (smaller)" }, { opt: "int8=True", desc: "INT8 quantization (fastest)" }].map((o) => (
                  <div key={o.opt} className="flex gap-3 items-start">
                    <code className="text-[10px] font-mono text-[#C15F3C] bg-[#C15F3C]/8 px-2 py-0.5 rounded-sm shrink-0 border border-[#C15F3C]/20">{o.opt}</code>
                    <span className="text-xs text-[#8A8580]">{o.desc}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
