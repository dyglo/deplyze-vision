import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import { ChartBar, Trash, Export, BoundingBox, PersonSimpleRun, Intersect, Tag, Path, Funnel, ArrowClockwise } from "@phosphor-icons/react";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;
const TASK_META = {
  detect:   { icon: BoundingBox,     color: "#008B22", label: "Detection" },
  pose:     { icon: PersonSimpleRun, color: "#CC1144", label: "Pose" },
  segment:  { icon: Intersect,       color: "#0087B3", label: "Segment" },
  classify: { icon: Tag,             color: "#7B1CC4", label: "Classify" },
  track:    { icon: Path,            color: "#B08000", label: "Track" },
};

export default function Results() {
  const [runs, setRuns] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);
  const [filterTask, setFilterTask] = useState("all");
  const [projects, setProjects] = useState([]);
  const [filterProject, setFilterProject] = useState("all");

  const fetchData = useCallback(async () => {
    setLoading(true);
    const params = new URLSearchParams();
    if (filterTask !== "all") params.set("task", filterTask);
    if (filterProject !== "all") params.set("project_id", filterProject);
    params.set("limit", "100");
    const [rr, sr, pr] = await Promise.all([axios.get(`${API}/runs?${params}`), axios.get(`${API}/stats`), axios.get(`${API}/projects`)]);
    setRuns(rr.data); setStats(sr.data); setProjects(pr.data); setLoading(false);
  }, [filterTask, filterProject]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const deleteRun = async (id) => { if (!window.confirm("Delete this run?")) return; await axios.delete(`${API}/runs/${id}`); if (selected?.id === id) setSelected(null); fetchData(); };

  const exportCSV = () => {
    if (!runs.length) return;
    const header = "id,task,model_name,source_type,results_count,fps,latency,created_at\n";
    const rows = runs.map((r) => `${r.id},${r.task},${r.model_name},${r.source_type},${r.results_count},${r.stats?.fps || 0},${r.stats?.latency || 0},${r.created_at}`).join("\n");
    const a = document.createElement("a"); a.download = `yolo26-runs-${Date.now()}.csv`; a.href = URL.createObjectURL(new Blob([header + rows], { type: "text/csv" })); a.click();
  };

  const exportSelectedJSON = () => { if (!selected) return; const a = document.createElement("a"); a.download = `run-${selected.id}.json`; a.href = URL.createObjectURL(new Blob([JSON.stringify(selected, null, 2)], { type: "application/json" })); a.click(); };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="font-['Outfit'] text-2xl font-medium text-[#141413] tracking-tight">Results</h1>
          <p className="text-sm text-[#8A8580] mt-1">Inference run history and experiment results</p>
        </div>
        <div className="flex items-center gap-2">
          <button data-testid="refresh-btn" onClick={fetchData} className="flex items-center gap-1.5 px-3 py-2 border border-[#DDD9D0] text-[#8A8580] text-sm rounded-sm hover:border-[#B1ADA1] hover:text-[#141413] transition-colors"><ArrowClockwise size={13} />Refresh</button>
          <button data-testid="export-csv-btn" onClick={exportCSV} disabled={!runs.length}
            className="flex items-center gap-2 px-4 py-2 bg-[#141413] text-[#F4F3EE] text-sm font-medium rounded-sm hover:bg-[#2A2925] transition-colors disabled:opacity-40">
            <Export size={14} />Export CSV
          </button>
        </div>
      </div>

      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-[#DDD9D0] mb-6">
          {[{ label: "Total Runs", value: stats.runs, color: "#141413" }, { label: "Projects", value: stats.projects, color: "#0087B3" }, { label: "Datasets", value: stats.datasets, color: "#B08000" }, { label: "Models", value: stats.models, color: "#7B1CC4" }].map((s) => (
            <div key={s.label} className="bg-[#F4F3EE] px-6 py-4">
              <div className="text-[10px] font-mono text-[#B1ADA1] uppercase tracking-wider mb-1">{s.label}</div>
              <div className="text-2xl font-['Outfit'] font-medium tabular-nums" style={{ color: s.color }}>{s.value}</div>
            </div>
          ))}
        </div>
      )}

      {stats?.task_distribution && Object.keys(stats.task_distribution).length > 0 && (
        <div className="mb-6 border border-[#DDD9D0] rounded-sm p-4 bg-white">
          <p className="text-[10px] font-mono text-[#B1ADA1] uppercase tracking-wider mb-3">Task Distribution</p>
          <div className="flex gap-4 flex-wrap">
            {Object.entries(stats.task_distribution).map(([task, count]) => {
              const meta = TASK_META[task]; const total = Object.values(stats.task_distribution).reduce((s, v) => s + v, 0);
              return (
                <div key={task} className="flex items-center gap-2">
                  {meta && <meta.icon size={12} style={{ color: meta.color }} />}
                  <span className="text-xs text-[#5C5751] font-mono">{meta?.label || task}</span>
                  <div className="w-16 h-1.5 bg-[#EDE9E0] rounded-full overflow-hidden"><div className="h-full rounded-full" style={{ width: `${(count / total) * 100}%`, backgroundColor: meta?.color || "#141413" }} /></div>
                  <span className="text-xs font-mono tabular-nums" style={{ color: meta?.color }}>{count}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap gap-3 mb-4">
        <div className="flex items-center gap-2"><Funnel size={12} className="text-[#B1ADA1]" /><span className="text-[10px] font-mono text-[#B1ADA1] uppercase tracking-wider">Filter</span></div>
        <div className="flex gap-1">
          {["all", ...Object.keys(TASK_META)].map((t) => {
            const meta = TASK_META[t];
            return (
              <button key={t} data-testid={`results-filter-${t}`} onClick={() => setFilterTask(t)}
                className={`flex items-center gap-1 px-2.5 py-1 rounded-sm text-xs transition-all ${filterTask === t ? "bg-white border border-[#DDD9D0] shadow-sm text-[#141413] font-medium" : "text-[#8A8580] hover:text-[#141413]"}`}>
                {meta && <meta.icon size={10} style={{ color: filterTask === t ? meta.color : undefined }} />}
                {t === "all" ? "All" : meta.label}
              </button>
            );
          })}
        </div>
        <select data-testid="project-filter" value={filterProject} onChange={(e) => setFilterProject(e.target.value)}
          className="bg-white border border-[#DDD9D0] rounded-sm text-xs text-[#8A8580] px-2 py-1 font-mono focus:outline-none focus:border-[#B1ADA1]">
          <option value="all">All Projects</option>
          {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
        </select>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-48 text-[#B1ADA1] text-sm font-mono">Loading…</div>
      ) : runs.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-64 border border-dashed border-[#DDD9D0] rounded-sm bg-white">
          <ChartBar size={40} className="text-[#DDD9D0] mb-3" />
          <p className="text-[#8A8580] text-sm">No inference runs yet</p>
          <p className="text-[#B1ADA1] text-xs mt-1">Run inference in the Studio and save results</p>
        </div>
      ) : (
        <div className="border border-[#DDD9D0] rounded-sm overflow-hidden bg-white">
          <table className="w-full text-xs" data-testid="runs-history-table">
            <thead>
              <tr className="border-b border-[#DDD9D0] bg-[#F4F3EE]">
                {["Task", "Model", "Source", "Objects", "FPS", "Latency", "Project", "Date", ""].map((h) => (
                  <th key={h} className="px-4 py-3 text-left font-mono text-[10px] text-[#B1ADA1] uppercase tracking-wider">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => {
                const meta = TASK_META[run.task]; const proj = projects.find((p) => p.id === run.project_id);
                return (
                  <tr key={run.id} data-testid={`run-row-${run.id}`}
                    className={`border-b border-[#F4F3EE] hover:bg-[#F4F3EE] transition-colors cursor-pointer ${selected?.id === run.id ? "bg-[#F4F3EE]" : ""}`}
                    onClick={() => setSelected(selected?.id === run.id ? null : run)}>
                    <td className="px-4 py-3"><div className="flex items-center gap-1.5">{meta && <meta.icon size={11} style={{ color: meta.color }} />}<span className="font-mono" style={{ color: meta?.color }}>{run.task}</span></div></td>
                    <td className="px-4 py-3 text-[#8A8580] font-mono truncate max-w-[140px]">{run.model_name}</td>
                    <td className="px-4 py-3 text-[#8A8580] font-mono">{run.source_type}</td>
                    <td className="px-4 py-3 font-mono text-[#141413] tabular-nums">{run.results_count}</td>
                    <td className="px-4 py-3 font-mono tabular-nums" style={{ color: "#008B22" }}>{run.stats?.fps || "—"}</td>
                    <td className="px-4 py-3 font-mono tabular-nums" style={{ color: "#0087B3" }}>{run.stats?.latency ? `${run.stats.latency}ms` : "—"}</td>
                    <td className="px-4 py-3 text-[#8A8580]">{proj ? proj.name : "—"}</td>
                    <td className="px-4 py-3 font-mono text-[#B1ADA1]">{new Date(run.created_at).toLocaleDateString()}</td>
                    <td className="px-4 py-3"><button data-testid={`delete-run-${run.id}`} onClick={(e) => { e.stopPropagation(); deleteRun(run.id); }} className="text-[#DDD9D0] hover:text-[#CC1144] transition-colors p-1"><Trash size={13} /></button></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {selected && (
        <div data-testid="run-detail-panel" className="mt-6 border border-[#DDD9D0] rounded-sm bg-white">
          <div className="px-5 py-3 border-b border-[#DDD9D0] flex items-center justify-between">
            <div className="flex items-center gap-3"><span className="text-xs font-mono text-[#B1ADA1] uppercase tracking-wider">Run Detail</span><code className="text-[10px] font-mono text-[#B1ADA1]">{selected.id}</code></div>
            <button data-testid="export-run-json-btn" onClick={exportSelectedJSON}
              className="flex items-center gap-1.5 px-3 py-1 text-xs border border-[#DDD9D0] text-[#8A8580] rounded-sm hover:border-[#B1ADA1] hover:text-[#141413] transition-colors">
              <Export size={11} />Export JSON
            </button>
          </div>
          <div className="p-5">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-5">
              {[{ label: "Task", value: selected.task, color: TASK_META[selected.task]?.color }, { label: "Objects", value: selected.results_count, color: "#141413" }, { label: "FPS", value: selected.stats?.fps || "—", color: "#008B22" }, { label: "Latency", value: selected.stats?.latency ? `${selected.stats.latency}ms` : "—", color: "#0087B3" }].map((s) => (
                <div key={s.label} className="bg-[#F4F3EE] border border-[#DDD9D0] rounded-sm px-4 py-3">
                  <div className="text-[10px] font-mono text-[#B1ADA1] uppercase tracking-wider mb-1">{s.label}</div>
                  <div className="text-sm font-mono font-medium tabular-nums" style={{ color: s.color }}>{s.value}</div>
                </div>
              ))}
            </div>
            {selected.detections?.length > 0 && (
              <div>
                <p className="text-[10px] font-mono text-[#B1ADA1] uppercase tracking-wider mb-3">Detections ({selected.detections.length})</p>
                <div className="border border-[#DDD9D0] rounded-sm overflow-hidden">
                  <table className="w-full text-xs">
                    <thead><tr className="border-b border-[#DDD9D0] bg-[#F4F3EE]">{["#", "Class", "Confidence", "BBox"].map((h) => <th key={h} className="px-4 py-2 text-left font-mono text-[10px] text-[#B1ADA1] uppercase tracking-wider">{h}</th>)}</tr></thead>
                    <tbody>
                      {selected.detections.slice(0, 20).map((det, i) => (
                        <tr key={i} className="border-b border-[#F4F3EE]">
                          <td className="px-4 py-1.5 font-mono text-[#B1ADA1]">{i + 1}</td>
                          <td className="px-4 py-1.5 font-mono text-[#141413]">{det.class || det.type || "—"}</td>
                          <td className="px-4 py-1.5 font-mono text-[#8A8580] tabular-nums">{det.score ? `${(det.score * 100).toFixed(1)}%` : "—"}</td>
                          <td className="px-4 py-1.5 font-mono text-[#B1ADA1] text-[10px]">{det.bbox ? det.bbox.map((v) => Math.round(v)).join(", ") : "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
            {selected.thumbnail && <div className="mt-5"><p className="text-[10px] font-mono text-[#B1ADA1] uppercase tracking-wider mb-3">Thumbnail</p><img src={selected.thumbnail} alt="Run thumbnail" className="max-w-xs rounded-sm border border-[#DDD9D0]" /></div>}
          </div>
        </div>
      )}
    </div>
  );
}
