import { useState, useEffect } from "react";
import axios from "axios";
import { Folders, Plus, Trash, ChartBar, BoundingBox, PersonSimpleRun, Intersect, Tag, Path, X } from "@phosphor-icons/react";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;
const TASK_META = {
  detect:   { icon: BoundingBox,     color: "#008B22", label: "Detection" },
  pose:     { icon: PersonSimpleRun, color: "#CC1144", label: "Pose" },
  segment:  { icon: Intersect,       color: "#0087B3", label: "Segment" },
  classify: { icon: Tag,             color: "#7B1CC4", label: "Classify" },
  track:    { icon: Path,            color: "#B08000", label: "Track" },
};
const ALL_TASKS = Object.keys(TASK_META);

export default function Projects() {
  const [projects, setProjects] = useState([]);
  const [runs, setRuns] = useState({});
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [selected, setSelected] = useState(null);
  const [form, setForm] = useState({ name: "", description: "", task_types: [] });

  const fetchProjects = async () => {
    setLoading(true);
    const r = await axios.get(`${API}/projects`);
    setProjects(r.data);
    setLoading(false);
    const runData = {};
    await Promise.all(r.data.map(async (p) => {
      const rr = await axios.get(`${API}/runs?project_id=${p.id}&limit=5`);
      runData[p.id] = rr.data;
    }));
    setRuns(runData);
  };

  useEffect(() => { fetchProjects(); }, []);

  const createProject = async () => {
    if (!form.name.trim()) return;
    await axios.post(`${API}/projects`, form);
    setForm({ name: "", description: "", task_types: [] });
    setShowCreate(false);
    fetchProjects();
  };

  const deleteProject = async (id) => {
    if (!window.confirm("Delete this project?")) return;
    await axios.delete(`${API}/projects/${id}`);
    if (selected?.id === id) setSelected(null);
    fetchProjects();
  };

  const toggleTask = (t) => setForm((prev) => ({
    ...prev,
    task_types: prev.task_types.includes(t) ? prev.task_types.filter((x) => x !== t) : [...prev.task_types, t],
  }));

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="font-['Outfit'] text-2xl font-medium text-[#141413] tracking-tight">Projects</h1>
          <p className="text-sm text-[#8A8580] mt-1">Organize your CV experiments and inference runs</p>
        </div>
        <button data-testid="create-project-btn" onClick={() => setShowCreate(true)}
          className="flex items-center gap-2 px-4 py-2 bg-[#141413] text-[#F4F3EE] text-sm font-medium rounded-sm hover:bg-[#2A2925] transition-colors">
          <Plus size={14} weight="bold" />New Project
        </button>
      </div>

      {showCreate && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#141413]/30">
          <div data-testid="create-project-modal" className="w-full max-w-md bg-white border border-[#DDD9D0] rounded-sm p-6 shadow-lg">
            <div className="flex items-center justify-between mb-5">
              <h2 className="font-['Outfit'] text-lg font-medium text-[#141413]">Create Project</h2>
              <button onClick={() => setShowCreate(false)} className="text-[#B1ADA1] hover:text-[#141413]"><X size={16} /></button>
            </div>
            <div className="space-y-4">
              <div>
                <label className="text-xs font-mono text-[#B1ADA1] uppercase tracking-wider block mb-1.5">Name *</label>
                <input data-testid="project-name-input" value={form.name} onChange={(e) => setForm((p) => ({ ...p, name: e.target.value }))}
                  placeholder="My CV Project"
                  className="w-full bg-[#F4F3EE] border border-[#DDD9D0] rounded-sm px-3 py-2 text-sm text-[#141413] placeholder-[#B1ADA1] focus:outline-none focus:border-[#B1ADA1] transition-colors" />
              </div>
              <div>
                <label className="text-xs font-mono text-[#B1ADA1] uppercase tracking-wider block mb-1.5">Description</label>
                <textarea value={form.description} onChange={(e) => setForm((p) => ({ ...p, description: e.target.value }))}
                  placeholder="What is this project about?" rows={2}
                  className="w-full bg-[#F4F3EE] border border-[#DDD9D0] rounded-sm px-3 py-2 text-sm text-[#141413] placeholder-[#B1ADA1] focus:outline-none focus:border-[#B1ADA1] transition-colors resize-none" />
              </div>
              <div>
                <label className="text-xs font-mono text-[#B1ADA1] uppercase tracking-wider block mb-1.5">CV Tasks</label>
                <div className="flex flex-wrap gap-2">
                  {ALL_TASKS.map((t) => {
                    const meta = TASK_META[t]; const active = form.task_types.includes(t);
                    return (
                      <button key={t} data-testid={`task-toggle-${t}`} onClick={() => toggleTask(t)}
                        className={`flex items-center gap-1.5 px-2.5 py-1 rounded-sm border text-xs transition-all ${active ? "bg-white shadow-sm" : "border-[#DDD9D0] text-[#8A8580] hover:text-[#141413]"}`}
                        style={active ? { borderColor: meta.color + "60", color: meta.color } : {}}>
                        <meta.icon size={11} style={active ? { color: meta.color } : {}} />{meta.label}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
            <div className="flex gap-3 mt-6">
              <button data-testid="cancel-create-btn" onClick={() => setShowCreate(false)}
                className="flex-1 py-2 text-sm border border-[#DDD9D0] text-[#8A8580] rounded-sm hover:border-[#B1ADA1] transition-colors">Cancel</button>
              <button data-testid="confirm-create-btn" onClick={createProject} disabled={!form.name.trim()}
                className="flex-1 py-2 text-sm bg-[#141413] text-[#F4F3EE] rounded-sm hover:bg-[#2A2925] transition-colors disabled:opacity-50">Create</button>
            </div>
          </div>
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center h-48 text-[#B1ADA1] text-sm font-mono">Loading…</div>
      ) : projects.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-64 border border-dashed border-[#DDD9D0] rounded-sm bg-white">
          <Folders size={40} className="text-[#DDD9D0] mb-3" />
          <p className="text-[#8A8580] text-sm">No projects yet</p>
          <button onClick={() => setShowCreate(true)} className="mt-3 text-xs text-[#8A8580] hover:text-[#141413] border border-[#DDD9D0] px-3 py-1.5 rounded-sm transition-colors">Create your first project</button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {projects.map((p) => (
            <div key={p.id} data-testid={`project-card-${p.id}`}
              className={`bg-white border rounded-sm p-5 hover:border-[#B1ADA1] transition-colors cursor-pointer ${selected?.id === p.id ? "border-[#C15F3C] shadow-sm" : "border-[#DDD9D0]"}`}
              onClick={() => setSelected(selected?.id === p.id ? null : p)}>
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 bg-[#F4F3EE] border border-[#DDD9D0] rounded-sm flex items-center justify-center">
                    <Folders size={12} className="text-[#B1ADA1]" />
                  </div>
                  <h3 className="font-medium text-[#141413] text-sm truncate">{p.name}</h3>
                </div>
                <button data-testid={`delete-project-${p.id}`} onClick={(e) => { e.stopPropagation(); deleteProject(p.id); }}
                  className="text-[#DDD9D0] hover:text-[#CC1144] transition-colors p-1"><Trash size={13} /></button>
              </div>
              {p.description && <p className="text-xs text-[#8A8580] mb-3 line-clamp-2">{p.description}</p>}
              {p.task_types?.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-3">
                  {p.task_types.slice(0, 4).map((t) => {
                    const meta = TASK_META[t]; if (!meta) return null;
                    return <span key={t} className="text-[10px] font-mono px-1.5 py-0.5 rounded-sm" style={{ color: meta.color, backgroundColor: meta.color + "12" }}>{meta.label}</span>;
                  })}
                </div>
              )}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5 text-xs text-[#B1ADA1]">
                  <ChartBar size={11} />
                  <span className="font-mono">{(runs[p.id] || []).length} runs</span>
                </div>
                <span className="text-[10px] font-mono text-[#B1ADA1]">{new Date(p.created_at).toLocaleDateString()}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {selected && runs[selected.id] && (
        <div data-testid="project-runs-panel" className="mt-6 border border-[#DDD9D0] rounded-sm bg-white">
          <div className="px-4 py-3 border-b border-[#DDD9D0] flex items-center gap-3">
            <span className="text-xs font-mono text-[#B1ADA1] uppercase tracking-wider">Recent Runs</span>
            <span className="text-xs font-mono text-[#B1ADA1]">— {selected.name}</span>
          </div>
          {runs[selected.id].length === 0 ? (
            <div className="p-6 text-center text-xs text-[#B1ADA1] font-mono">No runs yet for this project</div>
          ) : (
            <table className="w-full text-xs" data-testid="runs-table">
              <thead>
                <tr className="border-b border-[#DDD9D0]">
                  {["Task", "Model", "Source", "Objects", "Date"].map((h) => (
                    <th key={h} className="px-4 py-2 text-left font-mono text-[10px] text-[#B1ADA1] uppercase tracking-wider">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {runs[selected.id].map((run) => {
                  const meta = TASK_META[run.task];
                  return (
                    <tr key={run.id} className="border-b border-[#F4F3EE] hover:bg-[#F4F3EE] transition-colors">
                      <td className="px-4 py-2"><div className="flex items-center gap-1.5">{meta && <meta.icon size={11} style={{ color: meta.color }} />}<span className="font-mono" style={{ color: meta?.color }}>{run.task}</span></div></td>
                      <td className="px-4 py-2 text-[#8A8580] font-mono truncate max-w-[150px]">{run.model_name}</td>
                      <td className="px-4 py-2 text-[#8A8580] font-mono">{run.source_type}</td>
                      <td className="px-4 py-2 font-mono text-[#141413]">{run.results_count}</td>
                      <td className="px-4 py-2 text-[#B1ADA1] font-mono">{new Date(run.created_at).toLocaleDateString()}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}
