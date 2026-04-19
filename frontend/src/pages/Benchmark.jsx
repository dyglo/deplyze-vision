import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import {
  ChartLineUp, BoundingBox, PersonSimpleRun, Intersect, Tag, Path,
  ArrowRight, Columns, Lightning, Timer, Target, TrendUp,
} from "@phosphor-icons/react";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const TASK_META = {
  detect:   { icon: BoundingBox,     color: "#008B22", label: "Detection",     bg: "#008B22" },
  pose:     { icon: PersonSimpleRun, color: "#CC1144", label: "Pose",          bg: "#CC1144" },
  segment:  { icon: Intersect,       color: "#0087B3", label: "Segmentation",  bg: "#0087B3" },
  classify: { icon: Tag,             color: "#7B1CC4", label: "Classification", bg: "#7B1CC4" },
  track:    { icon: Path,            color: "#B08000", label: "Tracking",       bg: "#B08000" },
};

function MetricCard({ label, value, unit = "", color = "#141413", icon: Icon }) {
  return (
    <div className="bg-white border border-[#DDD9D0] rounded-sm p-4">
      <div className="flex items-center gap-2 mb-2">
        {Icon && <Icon size={14} style={{ color }} />}
        <span className="text-[10px] font-mono text-[#B1ADA1] uppercase tracking-wider">{label}</span>
      </div>
      <div className="font-['Outfit'] text-2xl font-medium tabular-nums" style={{ color }}>
        {value}<span className="text-sm text-[#B1ADA1] ml-1 font-normal">{unit}</span>
      </div>
    </div>
  );
}

function PerfBar({ value, max, color }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  return (
    <div className="flex items-center gap-3 flex-1">
      <div className="flex-1 h-2 bg-[#EDE9E0] rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="font-mono text-xs tabular-nums text-[#5C5751] w-10 text-right">{value}</span>
    </div>
  );
}

export default function Benchmark() {
  const navigate = useNavigate();
  const [runs, setRuns] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("overview");

  const fetchData = useCallback(async () => {
    setLoading(true);
    const [rr, sr] = await Promise.all([
      axios.get(`${API}/runs?limit=200`),
      axios.get(`${API}/stats`),
    ]);
    setRuns(rr.data);
    setStats(sr.data);
    setLoading(false);
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  // Aggregate metrics per task
  const taskMetrics = Object.keys(TASK_META).map((task) => {
    const taskRuns = runs.filter((r) => r.task === task);
    const fpsList = taskRuns.map((r) => r.stats?.fps || 0).filter((v) => v > 0);
    const latList = taskRuns.map((r) => r.stats?.latency || 0).filter((v) => v > 0);
    const objList = taskRuns.map((r) => r.results_count || 0);
    return {
      task,
      runs: taskRuns.length,
      avgFps: fpsList.length ? Math.round(fpsList.reduce((a, b) => a + b, 0) / fpsList.length) : 0,
      maxFps: fpsList.length ? Math.max(...fpsList) : 0,
      avgLatency: latList.length ? Math.round(latList.reduce((a, b) => a + b, 0) / latList.length) : 0,
      minLatency: latList.length ? Math.min(...latList) : 0,
      avgObjects: objList.length ? (objList.reduce((a, b) => a + b, 0) / objList.length).toFixed(1) : 0,
      totalObjects: objList.reduce((a, b) => a + b, 0),
    };
  });

  const maxFps = Math.max(...taskMetrics.map((t) => t.avgFps), 1);
  const totalRuns = runs.length;
  const allFps = runs.map((r) => r.stats?.fps || 0).filter((v) => v > 0);
  const allLat = runs.map((r) => r.stats?.latency || 0).filter((v) => v > 0);
  const overallAvgFps = allFps.length ? Math.round(allFps.reduce((a, b) => a + b, 0) / allFps.length) : 0;
  const overallAvgLat = allLat.length ? Math.round(allLat.reduce((a, b) => a + b, 0) / allLat.length) : 0;

  // Compare runs (runs that have compare data)
  const compareRuns = runs.filter((r) => r.model_name?.includes(" vs "));

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="font-['Outfit'] text-2xl font-medium text-[#141413] tracking-tight">Benchmark</h1>
          <p className="text-sm text-[#8A8580] mt-1">Auto-populated performance metrics from your inference runs</p>
        </div>
        <div className="flex items-center gap-3">
          <button onClick={fetchData} className="px-3 py-2 border border-[#DDD9D0] text-[#8A8580] text-sm rounded-sm hover:border-[#B1ADA1] hover:text-[#141413] transition-colors">
            Refresh
          </button>
          <button data-testid="go-to-studio-btn" onClick={() => navigate("/studio?compare=true")}
            className="flex items-center gap-2 px-4 py-2 bg-[#141413] text-[#F4F3EE] text-sm font-medium rounded-sm hover:bg-[#2A2925] transition-colors">
            <Columns size={14} />
            Compare in Studio
            <ArrowRight size={13} />
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-6 border-b border-[#DDD9D0]">
        {[{ id: "overview", label: "Overview" }, { id: "tasks", label: "Per-Task Metrics" }, { id: "compare", label: `Compare Runs (${compareRuns.length})` }].map((tab) => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm transition-all border-b-2 -mb-px ${
              activeTab === tab.id ? "border-[#C15F3C] text-[#141413] font-medium" : "border-transparent text-[#8A8580] hover:text-[#141413]"
            }`}>
            {tab.label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-48 text-[#B1ADA1] text-sm font-mono">Loading benchmark data…</div>
      ) : totalRuns === 0 ? (
        <div className="flex flex-col items-center justify-center h-64 border border-dashed border-[#DDD9D0] rounded-sm bg-white">
          <ChartLineUp size={48} className="text-[#DDD9D0] mb-4" />
          <h3 className="font-['Outfit'] text-lg font-medium text-[#141413] mb-2">No Benchmark Data Yet</h3>
          <p className="text-sm text-[#8A8580] text-center max-w-sm mb-6">
            Run inference in the Studio and save your results to start building your benchmark report.
          </p>
          <button onClick={() => navigate("/studio")}
            className="flex items-center gap-2 px-5 py-2 bg-[#141413] text-[#F4F3EE] text-sm rounded-sm hover:bg-[#2A2925] transition-colors">
            <BoundingBox size={14} />Open Studio
          </button>
        </div>
      ) : (
        <>
          {/* Overview Tab */}
          {activeTab === "overview" && (
            <div className="space-y-6">
              {/* Summary cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <MetricCard label="Total Runs" value={totalRuns} color="#141413" icon={Target} />
                <MetricCard label="Avg FPS" value={overallAvgFps} color="#008B22" icon={Lightning} />
                <MetricCard label="Avg Latency" value={overallAvgLat} unit="ms" color="#0087B3" icon={Timer} />
                <MetricCard label="Tasks Tested" value={taskMetrics.filter((t) => t.runs > 0).length} color="#C15F3C" icon={TrendUp} />
              </div>

              {/* FPS by task - bar chart */}
              <div className="bg-white border border-[#DDD9D0] rounded-sm p-5">
                <h3 className="font-['Outfit'] text-base font-medium text-[#141413] mb-4">Average FPS by Task</h3>
                <div className="space-y-3">
                  {taskMetrics.filter((t) => t.runs > 0).sort((a, b) => b.avgFps - a.avgFps).map((t) => {
                    const meta = TASK_META[t.task];
                    return (
                      <div key={t.task} className="flex items-center gap-3">
                        <div className="w-28 flex items-center gap-2">
                          <meta.icon size={12} style={{ color: meta.color }} />
                          <span className="text-xs text-[#5C5751] font-mono">{meta.label}</span>
                        </div>
                        <PerfBar value={t.avgFps} max={maxFps} color={meta.color} />
                        <span className="text-[10px] text-[#B1ADA1] font-mono w-12 text-right">{t.runs} runs</span>
                      </div>
                    );
                  })}
                  {taskMetrics.every((t) => t.runs === 0) && (
                    <p className="text-sm text-[#B1ADA1] font-mono text-center py-4">Run and save inference to populate chart</p>
                  )}
                </div>
              </div>

              {/* Latency by task */}
              <div className="bg-white border border-[#DDD9D0] rounded-sm p-5">
                <h3 className="font-['Outfit'] text-base font-medium text-[#141413] mb-4">Average Latency by Task (lower is better)</h3>
                <div className="space-y-3">
                  {taskMetrics.filter((t) => t.runs > 0).sort((a, b) => a.avgLatency - b.avgLatency).map((t) => {
                    const meta = TASK_META[t.task];
                    const maxLat = Math.max(...taskMetrics.map((x) => x.avgLatency), 1);
                    return (
                      <div key={t.task} className="flex items-center gap-3">
                        <div className="w-28 flex items-center gap-2">
                          <meta.icon size={12} style={{ color: meta.color }} />
                          <span className="text-xs text-[#5C5751] font-mono">{meta.label}</span>
                        </div>
                        <PerfBar value={t.avgLatency} max={maxLat} color={meta.color} />
                        <span className="text-[10px] text-[#B1ADA1] font-mono w-16 text-right">{t.avgLatency}ms avg</span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Task distribution pie (simple) */}
              {stats?.task_distribution && Object.keys(stats.task_distribution).length > 0 && (
                <div className="bg-white border border-[#DDD9D0] rounded-sm p-5">
                  <h3 className="font-['Outfit'] text-base font-medium text-[#141413] mb-4">Run Distribution</h3>
                  <div className="flex flex-wrap gap-3">
                    {Object.entries(stats.task_distribution).map(([task, count]) => {
                      const meta = TASK_META[task];
                      const pct = Math.round((count / totalRuns) * 100);
                      return (
                        <div key={task} className="flex items-center gap-2 bg-[#F4F3EE] border border-[#DDD9D0] rounded-sm px-3 py-2">
                          {meta && <meta.icon size={14} style={{ color: meta.color }} />}
                          <span className="text-sm text-[#141413] font-medium">{meta?.label || task}</span>
                          <span className="text-sm font-mono tabular-nums" style={{ color: meta?.color }}>{count}</span>
                          <span className="text-xs text-[#B1ADA1] font-mono">({pct}%)</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Per-Task Metrics Tab */}
          {activeTab === "tasks" && (
            <div className="space-y-4">
              <div className="border border-[#DDD9D0] rounded-sm overflow-hidden bg-white">
                <table className="w-full text-sm" data-testid="task-metrics-table">
                  <thead>
                    <tr className="border-b border-[#DDD9D0] bg-[#F4F3EE]">
                      {["Task", "Runs", "Avg FPS", "Max FPS", "Avg Latency", "Min Latency", "Avg Objects", "Total Objects"].map((h) => (
                        <th key={h} className="px-4 py-3 text-left font-mono text-[10px] text-[#B1ADA1] uppercase tracking-wider">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {taskMetrics.map((t) => {
                      const meta = TASK_META[t.task];
                      return (
                        <tr key={t.task} className={`border-b border-[#F4F3EE] hover:bg-[#F4F3EE] transition-colors ${t.runs === 0 ? "opacity-40" : ""}`}>
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-2">
                              <meta.icon size={13} style={{ color: meta.color }} />
                              <span className="font-medium text-[#141413]">{meta.label}</span>
                            </div>
                          </td>
                          <td className="px-4 py-3 font-mono text-[#141413] tabular-nums">{t.runs}</td>
                          <td className="px-4 py-3 font-mono tabular-nums" style={{ color: t.avgFps > 0 ? "#008B22" : "#B1ADA1" }}>{t.avgFps || "—"}</td>
                          <td className="px-4 py-3 font-mono tabular-nums text-[#5C5751]">{t.maxFps || "—"}</td>
                          <td className="px-4 py-3 font-mono tabular-nums" style={{ color: t.avgLatency > 0 ? "#0087B3" : "#B1ADA1" }}>{t.avgLatency ? `${t.avgLatency}ms` : "—"}</td>
                          <td className="px-4 py-3 font-mono tabular-nums text-[#5C5751]">{t.minLatency ? `${t.minLatency}ms` : "—"}</td>
                          <td className="px-4 py-3 font-mono tabular-nums text-[#5C5751]">{t.avgObjects}</td>
                          <td className="px-4 py-3 font-mono tabular-nums text-[#141413]">{t.totalObjects}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="text-xs text-[#B1ADA1] font-mono">
                Metrics are computed from all saved inference runs. Run inference in Studio and save runs to populate.
              </p>
            </div>
          )}

          {/* Compare Runs Tab */}
          {activeTab === "compare" && (
            <div>
              {compareRuns.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-48 border border-dashed border-[#DDD9D0] rounded-sm bg-white">
                  <Columns size={32} className="text-[#DDD9D0] mb-3" />
                  <p className="text-sm text-[#8A8580]">No compare runs yet</p>
                  <p className="text-xs text-[#B1ADA1] mt-1">Enable Compare Mode in Studio and save runs</p>
                  <button onClick={() => navigate("/studio")} className="mt-4 flex items-center gap-2 px-4 py-1.5 bg-[#141413] text-[#F4F3EE] text-xs rounded-sm hover:bg-[#2A2925]">
                    <Columns size={12} />Open Studio Compare
                  </button>
                </div>
              ) : (
                <div className="space-y-4">
                  <p className="text-xs text-[#8A8580] font-mono">{compareRuns.length} side-by-side comparison runs</p>
                  <div className="border border-[#DDD9D0] rounded-sm overflow-hidden bg-white">
                    <table className="w-full text-xs" data-testid="compare-runs-table">
                      <thead>
                        <tr className="border-b border-[#DDD9D0] bg-[#F4F3EE]">
                          {["Models Compared", "Source", "Objects", "FPS", "Latency", "Date"].map((h) => (
                            <th key={h} className="px-4 py-3 text-left font-mono text-[10px] text-[#B1ADA1] uppercase tracking-wider">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {compareRuns.map((run) => (
                          <tr key={run.id} className="border-b border-[#F4F3EE] hover:bg-[#F4F3EE] transition-colors">
                            <td className="px-4 py-3 font-mono text-[#141413]">{run.model_name}</td>
                            <td className="px-4 py-3 text-[#8A8580]">{run.source_type}</td>
                            <td className="px-4 py-3 font-mono text-[#141413] tabular-nums">{run.results_count}</td>
                            <td className="px-4 py-3 font-mono tabular-nums" style={{ color: "#008B22" }}>{run.stats?.fps || "—"}</td>
                            <td className="px-4 py-3 font-mono tabular-nums" style={{ color: "#0087B3" }}>{run.stats?.latency ? `${run.stats.latency}ms` : "—"}</td>
                            <td className="px-4 py-3 font-mono text-[#B1ADA1]">{new Date(run.created_at).toLocaleDateString()}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
