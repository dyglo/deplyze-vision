import { Outlet, NavLink, useLocation } from "react-router-dom";
import {
  BoundingBox,
  Folders,
  Database,
  Cube,
  ChartBar,
  Code,
  ArrowSquareOut,
  List,
  X,
  ChartLineUp,
} from "@phosphor-icons/react";
import { useState } from "react";

const NAV_ITEMS = [
  { path: "/studio", label: "Studio", icon: BoundingBox, accent: "#008B22", desc: "Inference" },
  { path: "/benchmark", label: "Benchmark", icon: ChartLineUp, accent: "#C15F3C", desc: "Compare" },
  { path: "/projects", label: "Projects", icon: Folders, accent: "#0087B3", desc: "Manage" },
  { path: "/datasets", label: "Datasets", icon: Database, accent: "#B08000", desc: "Data" },
  { path: "/models", label: "Model Hub", icon: Cube, accent: "#7B1CC4", desc: "Models" },
  { path: "/results", label: "Results", icon: ChartBar, accent: "#CC1144", desc: "History" },
];

export default function Layout() {
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="flex h-screen bg-[#F4F3EE] overflow-hidden">
      {/* Sidebar */}
      <aside
        data-testid="sidebar"
        className={`
          fixed inset-y-0 left-0 z-50 w-[220px] bg-[#F0EFE9] border-r border-[#DDD9D0] flex flex-col
          transition-transform duration-150 ease-out
          lg:relative lg:translate-x-0
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}
        `}
      >
        {/* Logo */}
        <div className="h-14 border-b border-[#DDD9D0] flex items-center px-4 gap-3">
          <div className="w-7 h-7 bg-[#C15F3C] rounded-sm flex items-center justify-center">
            <BoundingBox weight="bold" size={16} color="#fff" />
          </div>
          <div>
            <div className="font-['Outfit'] text-sm font-semibold text-[#141413] tracking-tight">YOLO26</div>
            <div className="text-[10px] text-[#B1ADA1] font-mono tracking-[0.15em] uppercase">CV Platform</div>
          </div>
          <button
            className="ml-auto lg:hidden text-[#8A8580] hover:text-[#141413]"
            onClick={() => setSidebarOpen(false)}
          >
            <X size={18} />
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 py-3 px-2 space-y-0.5">
          {NAV_ITEMS.map(({ path, label, icon: Icon, accent, desc }) => (
            <NavLink
              key={path}
              to={path}
              data-testid={`nav-${label.toLowerCase().replace(/\s+/g, "-")}`}
              onClick={() => setSidebarOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-sm group transition-all duration-100 ${
                  isActive
                    ? "bg-white border border-[#DDD9D0] shadow-sm"
                    : "hover:bg-white/60 border border-transparent"
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <Icon
                    size={16}
                    weight={isActive ? "bold" : "regular"}
                    style={{ color: isActive ? accent : "#B1ADA1" }}
                  />
                  <div className="flex-1 min-w-0">
                    <div
                      className="text-sm font-medium leading-none truncate"
                      style={{ color: isActive ? "#141413" : "#8A8580" }}
                    >
                      {label}
                    </div>
                    <div className="text-[10px] text-[#B1ADA1] mt-0.5 font-mono uppercase tracking-wider">{desc}</div>
                  </div>
                  {isActive && (
                    <div className="w-1 h-4 rounded-full" style={{ backgroundColor: accent }} />
                  )}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="p-3 border-t border-[#DDD9D0] space-y-1">
          <a
            href="https://github.com/dyglo/deplyze-vision"
            target="_blank"
            rel="noopener noreferrer"
            data-testid="sidebar-github-link"
            className="flex items-center gap-3 px-3 py-2 rounded-sm text-[#8A8580] hover:text-[#141413] hover:bg-white/60 transition-colors duration-100 text-sm"
          >
            <Code size={14} />
            <span>GitHub</span>
            <ArrowSquareOut size={12} className="ml-auto" />
          </a>
          <a
            href="/api/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 px-3 py-2 rounded-sm text-[#8A8580] hover:text-[#141413] hover:bg-white/60 transition-colors duration-100 text-sm"
          >
            <Code size={14} />
            <span>API Docs</span>
            <ArrowSquareOut size={12} className="ml-auto" />
          </a>
          <div className="px-3 py-2 text-[10px] text-[#B1ADA1] font-mono">
            YOLO26 · TF.js v4.22 · Open-Source
          </div>
        </div>
      </aside>

      {/* Overlay for mobile */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-[#141413]/30 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top bar */}
        <header className="h-14 border-b border-[#DDD9D0] flex items-center px-4 gap-4 bg-[#F4F3EE] flex-shrink-0 z-30">
          <button
            className="lg:hidden text-[#8A8580] hover:text-[#141413]"
            onClick={() => setSidebarOpen(true)}
            data-testid="sidebar-toggle"
          >
            <List size={20} />
          </button>
          <div className="text-xs font-mono text-[#B1ADA1] tracking-wider uppercase">
            {NAV_ITEMS.find((n) => location.pathname.startsWith(n.path))?.label || "Dashboard"}
          </div>
          <div className="ml-auto flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-[#008B22] stat-live" />
              <span className="text-[11px] font-mono text-[#8A8580]">TF.js WebGL</span>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-auto bg-[#F4F3EE]">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
