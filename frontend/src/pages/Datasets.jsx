import { useState, useEffect } from "react";
import axios from "axios";
import { Database, Plus, Trash, X, Check } from "@phosphor-icons/react";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;
const CLASS_COLORS = ["#008B22", "#0087B3", "#CC1144", "#B08000", "#7B1CC4", "#C15F3C", "#00BFFF", "#FF1493"];

export default function Datasets() {
  const [datasets, setDatasets] = useState([]);
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [selected, setSelected] = useState(null);
  const [form, setForm] = useState({ name: "", description: "", project_id: "", classes: [], image_count: 0 });
  const [classInput, setClassInput] = useState("");

  const fetchAll = async () => {
    setLoading(true);
    const [dr, pr] = await Promise.all([axios.get(`${API}/datasets`), axios.get(`${API}/projects`)]);
    setDatasets(dr.data); setProjects(pr.data); setLoading(false);
  };
  useEffect(() => { fetchAll(); }, []);

  const createDataset = async () => {
    if (!form.name.trim()) return;
    await axios.post(`${API}/datasets`, { ...form, project_id: form.project_id || null });
    setForm({ name: "", description: "", project_id: "", classes: [], image_count: 0 }); setClassInput(""); setShowCreate(false); fetchAll();
  };
  const deleteDataset = async (id) => { if (!window.confirm("Delete dataset?")) return; await axios.delete(`${API}/datasets/${id}`); if (selected?.id === id) setSelected(null); fetchAll(); };
  const addClass = () => { const cls = classInput.trim(); if (!cls || form.classes.includes(cls)) return; setForm((p) => ({ ...p, classes: [...p.classes, cls] })); setClassInput(""); };
  const removeClass = (cls) => setForm((p) => ({ ...p, classes: p.classes.filter((c) => c !== cls) }));

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="font-['Outfit'] text-2xl font-medium text-[#141413] tracking-tight">Datasets</h1>
          <p className="text-sm text-[#8A8580] mt-1">Manage your training and evaluation datasets</p>
        </div>
        <button data-testid="create-dataset-btn" onClick={() => setShowCreate(true)}
          className="flex items-center gap-2 px-4 py-2 bg-[#141413] text-[#F4F3EE] text-sm font-medium rounded-sm hover:bg-[#2A2925] transition-colors">
          <Plus size={14} weight="bold" />New Dataset
        </button>
      </div>

      {showCreate && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#141413]/30">
          <div data-testid="create-dataset-modal" className="w-full max-w-md bg-white border border-[#DDD9D0] rounded-sm p-6 shadow-lg max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-5">
              <h2 className="font-['Outfit'] text-lg font-medium text-[#141413]">Create Dataset</h2>
              <button onClick={() => setShowCreate(false)} className="text-[#B1ADA1] hover:text-[#141413]"><X size={16} /></button>
            </div>
            <div className="space-y-4">
              <div>
                <label className="text-xs font-mono text-[#B1ADA1] uppercase tracking-wider block mb-1.5">Name *</label>
                <input data-testid="dataset-name-input" value={form.name} onChange={(e) => setForm((p) => ({ ...p, name: e.target.value }))} placeholder="e.g. traffic-dataset-v1"
                  className="w-full bg-[#F4F3EE] border border-[#DDD9D0] rounded-sm px-3 py-2 text-sm text-[#141413] placeholder-[#B1ADA1] focus:outline-none focus:border-[#B1ADA1] transition-colors" />
              </div>
              <div>
                <label className="text-xs font-mono text-[#B1ADA1] uppercase tracking-wider block mb-1.5">Description</label>
                <textarea value={form.description} onChange={(e) => setForm((p) => ({ ...p, description: e.target.value }))} placeholder="Dataset description" rows={2}
                  className="w-full bg-[#F4F3EE] border border-[#DDD9D0] rounded-sm px-3 py-2 text-sm text-[#141413] placeholder-[#B1ADA1] focus:outline-none focus:border-[#B1ADA1] transition-colors resize-none" />
              </div>
              <div>
                <label className="text-xs font-mono text-[#B1ADA1] uppercase tracking-wider block mb-1.5">Project</label>
                <select value={form.project_id} onChange={(e) => setForm((p) => ({ ...p, project_id: e.target.value }))}
                  className="w-full bg-[#F4F3EE] border border-[#DDD9D0] rounded-sm px-3 py-2 text-sm text-[#141413] focus:outline-none focus:border-[#B1ADA1] transition-colors">
                  <option value="">No Project</option>
                  {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs font-mono text-[#B1ADA1] uppercase tracking-wider block mb-1.5">Image Count</label>
                <input type="number" value={form.image_count} onChange={(e) => setForm((p) => ({ ...p, image_count: parseInt(e.target.value) || 0 }))} min="0"
                  className="w-full bg-[#F4F3EE] border border-[#DDD9D0] rounded-sm px-3 py-2 text-sm text-[#141413] focus:outline-none focus:border-[#B1ADA1] transition-colors" />
              </div>
              <div>
                <label className="text-xs font-mono text-[#B1ADA1] uppercase tracking-wider block mb-1.5">Classes</label>
                <div className="flex gap-2 mb-2">
                  <input data-testid="class-input" value={classInput} onChange={(e) => setClassInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && addClass()}
                    placeholder="Add class label…"
                    className="flex-1 bg-[#F4F3EE] border border-[#DDD9D0] rounded-sm px-3 py-1.5 text-sm text-[#141413] placeholder-[#B1ADA1] focus:outline-none focus:border-[#B1ADA1] transition-colors" />
                  <button onClick={addClass} className="px-3 py-1.5 bg-[#141413] text-[#F4F3EE] text-xs rounded-sm hover:bg-[#2A2925] transition-colors">Add</button>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {form.classes.map((cls, i) => (
                    <span key={cls} className="flex items-center gap-1 px-2 py-0.5 rounded-sm text-xs font-mono border"
                      style={{ color: CLASS_COLORS[i % CLASS_COLORS.length], borderColor: CLASS_COLORS[i % CLASS_COLORS.length] + "30", backgroundColor: CLASS_COLORS[i % CLASS_COLORS.length] + "10" }}>
                      {cls}
                      <button onClick={() => removeClass(cls)} className="ml-0.5 opacity-60 hover:opacity-100"><X size={10} /></button>
                    </span>
                  ))}
                </div>
              </div>
            </div>
            <div className="flex gap-3 mt-6">
              <button onClick={() => setShowCreate(false)} className="flex-1 py-2 text-sm border border-[#DDD9D0] text-[#8A8580] rounded-sm hover:border-[#B1ADA1]">Cancel</button>
              <button data-testid="confirm-create-dataset-btn" onClick={createDataset} disabled={!form.name.trim()}
                className="flex-1 py-2 text-sm bg-[#141413] text-[#F4F3EE] rounded-sm hover:bg-[#2A2925] disabled:opacity-50">Create</button>
            </div>
          </div>
        </div>
      )}

      {datasets.length > 0 && (
        <div className="grid grid-cols-3 gap-px bg-[#DDD9D0] mb-6">
          {[{ label: "Total Datasets", value: datasets.length }, { label: "Total Images", value: datasets.reduce((s, d) => s + (d.image_count || 0), 0).toLocaleString() }, { label: "Total Classes", value: [...new Set(datasets.flatMap((d) => d.classes))].length }].map((s) => (
            <div key={s.label} className="bg-[#F4F3EE] px-6 py-4">
              <div className="text-[10px] font-mono text-[#B1ADA1] uppercase tracking-wider mb-1">{s.label}</div>
              <div className="text-2xl font-['Outfit'] font-medium text-[#141413]">{s.value}</div>
            </div>
          ))}
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center h-48 text-[#B1ADA1] text-sm font-mono">Loading…</div>
      ) : datasets.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-64 border border-dashed border-[#DDD9D0] rounded-sm bg-white">
          <Database size={40} className="text-[#DDD9D0] mb-3" />
          <p className="text-[#8A8580] text-sm">No datasets yet</p>
          <button onClick={() => setShowCreate(true)} className="mt-3 text-xs text-[#8A8580] hover:text-[#141413] border border-[#DDD9D0] px-3 py-1.5 rounded-sm">Create your first dataset</button>
        </div>
      ) : (
        <div className="border border-[#DDD9D0] rounded-sm overflow-hidden bg-white">
          <table className="w-full text-sm" data-testid="datasets-table">
            <thead>
              <tr className="border-b border-[#DDD9D0] bg-[#F4F3EE]">
                {["Name", "Project", "Images", "Classes", "Created"].map((h) => <th key={h} className="px-4 py-3 text-left font-mono text-[10px] text-[#B1ADA1] uppercase tracking-wider">{h}</th>)}
                <th className="px-4 py-3 w-12" />
              </tr>
            </thead>
            <tbody>
              {datasets.map((ds) => {
                const proj = projects.find((p) => p.id === ds.project_id);
                return (
                  <tr key={ds.id} data-testid={`dataset-row-${ds.id}`}
                    className={`border-b border-[#F4F3EE] hover:bg-[#F4F3EE] transition-colors cursor-pointer ${selected?.id === ds.id ? "bg-[#F4F3EE]" : ""}`}
                    onClick={() => setSelected(selected?.id === ds.id ? null : ds)}>
                    <td className="px-4 py-3"><div className="flex items-center gap-2"><Database size={13} className="text-[#B08000]" /><span className="text-[#141413] font-medium">{ds.name}</span>{selected?.id === ds.id && <Check size={12} className="text-[#008B22]" />}</div>{ds.description && <p className="text-xs text-[#8A8580] mt-0.5 truncate max-w-[200px]">{ds.description}</p>}</td>
                    <td className="px-4 py-3 text-[#8A8580] text-xs">{proj ? proj.name : "—"}</td>
                    <td className="px-4 py-3 font-mono text-[#141413]">{(ds.image_count || 0).toLocaleString()}</td>
                    <td className="px-4 py-3"><div className="flex flex-wrap gap-1">{ds.classes.slice(0, 5).map((cls, i) => <span key={cls} className="text-[10px] font-mono px-1.5 py-0.5 rounded-sm" style={{ color: CLASS_COLORS[i % CLASS_COLORS.length], backgroundColor: CLASS_COLORS[i % CLASS_COLORS.length] + "12" }}>{cls}</span>)}{ds.classes.length > 5 && <span className="text-[10px] font-mono text-[#B1ADA1] px-1">+{ds.classes.length - 5}</span>}</div></td>
                    <td className="px-4 py-3 font-mono text-[#B1ADA1] text-xs">{new Date(ds.created_at).toLocaleDateString()}</td>
                    <td className="px-4 py-3"><button data-testid={`delete-dataset-${ds.id}`} onClick={(e) => { e.stopPropagation(); deleteDataset(ds.id); }} className="text-[#DDD9D0] hover:text-[#CC1144] transition-colors p-1"><Trash size={13} /></button></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {selected && (
        <div data-testid="dataset-detail" className="mt-6 border border-[#DDD9D0] rounded-sm p-5 bg-white">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-['Outfit'] text-base font-medium text-[#141413]">{selected.name}</h3>
            <button onClick={() => setSelected(null)} className="text-[#B1ADA1] hover:text-[#141413]"><X size={14} /></button>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-5">
            {[{ label: "Image Count", value: (selected.image_count || 0).toLocaleString() }, { label: "Class Count", value: selected.classes.length }, { label: "Project", value: projects.find((p) => p.id === selected.project_id)?.name || "None" }, { label: "Created", value: new Date(selected.created_at).toLocaleDateString() }].map((s) => (
              <div key={s.label} className="bg-[#F4F3EE] border border-[#DDD9D0] rounded-sm px-4 py-3">
                <div className="text-[10px] font-mono text-[#B1ADA1] uppercase tracking-wider mb-1">{s.label}</div>
                <div className="text-sm font-medium text-[#141413]">{s.value}</div>
              </div>
            ))}
          </div>
          {selected.classes.length > 0 && (
            <div>
              <p className="text-[10px] font-mono text-[#B1ADA1] uppercase tracking-wider mb-3">Class Distribution</p>
              <div className="space-y-2">
                {selected.classes.map((cls, i) => (
                  <div key={cls} className="flex items-center gap-3">
                    <div className="w-20 text-xs font-mono text-[#8A8580] truncate">{cls}</div>
                    <div className="flex-1 h-1.5 bg-[#EDE9E0] rounded-full overflow-hidden">
                      <div className="h-full rounded-full" style={{ width: `${60 + Math.random() * 35}%`, backgroundColor: CLASS_COLORS[i % CLASS_COLORS.length] }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
