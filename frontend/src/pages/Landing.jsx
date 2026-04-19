import { useNavigate } from "react-router-dom";
import {
  BoundingBox,
  PersonSimpleRun,
  Intersect,
  Tag,
  Path,
  Camera,
  Export,
  Folders,
  ArrowRight,
  GithubLogo,
  Lightning,
  ShieldCheck,
  Globe,
  Star,
} from "@phosphor-icons/react";
import HeroDemo from "../components/HeroDemo";

// Official repository URL
const GITHUB_REPO_URL = "https://github.com/dyglo/deplyze-vision";

const TASKS = [
  { id: "detect", label: "Object Detection", icon: BoundingBox, color: "#008B22", desc: "80 COCO classes" },
  { id: "pose", label: "Pose Estimation", icon: PersonSimpleRun, color: "#CC1144", desc: "17 keypoints via MoveNet" },
  { id: "segment", label: "Segmentation", icon: Intersect, color: "#0087B3", desc: "Person seg with BodyPix" },
  { id: "classify", label: "Classification", icon: Tag, color: "#7B1CC4", desc: "1000 ImageNet classes" },
  { id: "track", label: "Object Tracking", icon: Path, color: "#B08000", desc: "Multi-object centroid tracking" },
];

const FEATURES = [
  { icon: Lightning, title: "In-Browser Inference", desc: "All inference runs locally via TF.js WebGL. Zero latency, zero privacy concerns. No GPU server.", color: "#008B22" },
  { icon: Camera, title: "All Input Sources", desc: "Images, videos, and live webcam. Real-time detection at up to 30 FPS.", color: "#0087B3" },
  { icon: Export, title: "Export Anywhere", desc: "Export JSON, CSV, or annotated PNG images. Full inference history in the database.", color: "#B08000" },
  { icon: Folders, title: "Project Management", desc: "Organize runs into projects. Track experiments and compare model performance.", color: "#7B1CC4" },
  { icon: ShieldCheck, title: "YOLO26 Ready", desc: "Export your Ultralytics YOLO26 models to TF.js format and load them in the Model Hub.", color: "#CC1144" },
  { icon: Globe, title: "Open Source", desc: "Fully open-source CV platform. Extend, self-host, contribute. Built for the community.", color: "#C15F3C" },
];

export default function Landing() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-[#F4F3EE] text-[#141413] overflow-x-hidden">
      {/* Navbar */}
      <header className="fixed top-0 w-full z-50 bg-[#F4F3EE]/90 backdrop-blur-xl border-b border-[#DDD9D0]">
        <div className="max-w-7xl mx-auto px-6 h-14 flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-[#C15F3C] rounded-sm flex items-center justify-center">
              <BoundingBox weight="bold" size={13} color="#fff" />
            </div>
            <span className="font-['Outfit'] font-semibold text-sm tracking-tight text-[#141413]">YOLO26 CV Platform</span>
          </div>
          <div className="ml-auto flex items-center gap-3">
            <a
              href={GITHUB_REPO_URL}
              target="_blank"
              rel="noopener noreferrer"
              data-testid="github-stars-btn"
              className="flex items-center gap-2 px-3 py-1.5 text-xs text-[#5C5751] hover:text-[#141413] border border-[#DDD9D0] hover:border-[#B1ADA1] rounded-sm transition-colors duration-100 bg-white"
            >
              <GithubLogo size={13} />
              <Star size={11} />
              Star on GitHub
            </a>
            <button
              data-testid="launch-studio-btn"
              onClick={() => navigate("/studio")}
              className="flex items-center gap-2 px-4 py-1.5 text-xs font-medium bg-[#141413] text-[#F4F3EE] hover:bg-[#2A2925] rounded-sm transition-colors duration-100"
            >
              Launch Studio
              <ArrowRight size={12} />
            </button>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="relative pt-32 pb-20 px-6 overflow-hidden">
        {/* Background grid */}
        <div
          className="absolute inset-0 opacity-30"
          style={{
            backgroundImage: "linear-gradient(#DDD9D0 1px, transparent 1px), linear-gradient(90deg, #DDD9D0 1px, transparent 1px)",
            backgroundSize: "40px 40px",
          }}
        />
        {/* Glow */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[600px] h-[300px] rounded-full bg-[#C15F3C]/8 blur-3xl pointer-events-none" />

        <div className="relative max-w-7xl mx-auto grid lg:grid-cols-12 gap-10 items-center">
          <div className="lg:col-span-7">
            <div className="flex items-center gap-2 mb-6">
              <div className="px-2 py-1 bg-[#C15F3C]/10 border border-[#C15F3C]/30 rounded-sm">
                <span className="text-[#C15F3C] text-xs font-mono tracking-wider">YOLO26 · TF.js · In-Browser</span>
              </div>
              <div className="px-2 py-1 bg-white border border-[#DDD9D0] rounded-sm">
                <span className="text-[#8A8580] text-xs font-mono">Open Source</span>
              </div>
            </div>

            <h1 className="font-['Outfit'] text-4xl sm:text-5xl lg:text-6xl font-medium tracking-tighter text-[#141413] leading-[1.05] mb-6 max-w-4xl">
              Professional Computer Vision
              <br />
              <span className="text-[#B1ADA1]">Powered by</span>{" "}
              <span style={{ color: "#C15F3C" }}>YOLO26 &amp; TF.js</span>
            </h1>
            <p className="text-base text-[#5C5751] max-w-2xl mb-10 leading-relaxed font-['IBM_Plex_Sans']">
              Run object detection, pose estimation, segmentation, classification, and tracking
              entirely in your browser — no GPU server required. Export results, manage projects,
              and deploy your own YOLO26 TF.js models.
            </p>

            <div className="flex flex-wrap gap-3">
              <button
                data-testid="hero-launch-btn"
                onClick={() => navigate("/studio")}
                className="flex items-center gap-2 px-6 py-3 bg-[#141413] text-[#F4F3EE] text-sm font-medium rounded-sm hover:bg-[#2A2925] transition-colors duration-100"
              >
                <BoundingBox size={16} weight="bold" />
                Open Inference Studio
              </button>
              <button
                data-testid="hero-benchmark-btn"
                onClick={() => navigate("/benchmark")}
                className="flex items-center gap-2 px-6 py-3 bg-white text-[#141413] text-sm font-medium rounded-sm border border-[#DDD9D0] hover:border-[#B1ADA1] transition-colors duration-100"
              >
                View Benchmark
                <ArrowRight size={14} />
              </button>
            </div>
          </div>

          {/* Live in-browser demo */}
          <div className="lg:col-span-5 flex justify-center lg:justify-end">
            <HeroDemo />
          </div>
        </div>
      </section>

      {/* CV Tasks strip */}
      <section className="border-y border-[#DDD9D0] py-4 overflow-hidden bg-white">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-wrap gap-0">
            {TASKS.map((task) => (
              <button
                key={task.id}
                data-testid={`task-pill-${task.id}`}
                onClick={() => navigate(`/studio?task=${task.id}`)}
                className="flex items-center gap-2 px-5 py-3 border-r border-[#DDD9D0] last:border-r-0 hover:bg-[#F4F3EE] transition-colors duration-100 group"
              >
                <task.icon size={16} style={{ color: task.color }} />
                <span className="text-sm text-[#5C5751] group-hover:text-[#141413] transition-colors">{task.label}</span>
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Features grid */}
      <section className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="mb-12">
            <p className="text-xs font-mono text-[#B1ADA1] tracking-[0.2em] uppercase mb-3">Platform Features</p>
            <h2 className="font-['Outfit'] text-2xl sm:text-3xl font-medium text-[#141413] tracking-tight">
              Everything a CV Engineer Needs
            </h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-px bg-[#DDD9D0]">
            {FEATURES.map((feat, i) => (
              <div
                key={i}
                className="bg-[#F4F3EE] p-6 hover:bg-white transition-colors duration-100 group"
              >
                <div className="mb-4">
                  <feat.icon size={20} style={{ color: feat.color }} />
                </div>
                <h3 className="font-['Outfit'] text-base font-medium text-[#141413] mb-2">{feat.title}</h3>
                <p className="text-sm text-[#8A8580] leading-relaxed">{feat.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* YOLO26 export section */}
      <section className="py-20 px-6 border-t border-[#DDD9D0] bg-white">
        <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-12 items-center">
          <div>
            <p className="text-xs font-mono text-[#B1ADA1] tracking-[0.2em] uppercase mb-3">YOLO26 Integration</p>
            <h2 className="font-['Outfit'] text-2xl sm:text-3xl font-medium text-[#141413] mb-4 tracking-tight">
              Export YOLO26 Models to TF.js
            </h2>
            <p className="text-sm text-[#5C5751] mb-6 leading-relaxed">
              Use Ultralytics to export your custom trained YOLO26 models to TensorFlow.js format,
              then load them directly in the Model Hub for in-browser inference.
            </p>
            <div className="bg-[#F4F3EE] border border-[#DDD9D0] rounded-sm p-4 font-mono text-xs">
              <div className="text-[#B1ADA1] mb-2"># Export YOLO26 model to TF.js</div>
              <div className="text-[#C15F3C]">from ultralytics import YOLO</div>
              <div className="text-[#141413] mt-1">model = YOLO(<span className="text-[#7B1CC4]">"yolo26n.pt"</span>)</div>
              <div className="text-[#141413]">model.export(format=<span className="text-[#7B1CC4]">"tfjs"</span>)</div>
              <div className="text-[#B1ADA1] mt-2"># Load in CV Platform → Model Hub</div>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: "Object Detection", color: "#008B22", classes: "80 COCO" },
              { label: "Pose Estimation", color: "#CC1144", classes: "17 keypoints" },
              { label: "Segmentation", color: "#0087B3", classes: "Instance masks" },
              { label: "Classification", color: "#7B1CC4", classes: "1000 classes" },
            ].map((item) => (
              <div key={item.label} className="bg-[#F4F3EE] border border-[#DDD9D0] rounded-sm p-4 hover:border-[#B1ADA1] transition-colors">
                <div className="w-2 h-2 rounded-full mb-3" style={{ backgroundColor: item.color }} />
                <div className="text-sm font-medium text-[#141413] mb-1">{item.label}</div>
                <div className="text-xs font-mono text-[#8A8580]">{item.classes}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-6 border-t border-[#DDD9D0]">
        <div className="max-w-7xl mx-auto text-center">
          <h2 className="font-['Outfit'] text-2xl sm:text-3xl font-medium text-[#141413] mb-4 tracking-tight">
            Start Detecting in Seconds
          </h2>
          <p className="text-sm text-[#8A8580] mb-8">
            No setup required. Your browser is the inference engine.
          </p>
          <div className="flex items-center justify-center gap-3">
            <button
              data-testid="cta-studio-btn"
              onClick={() => navigate("/studio")}
              className="inline-flex items-center gap-2 px-8 py-3 bg-[#141413] text-[#F4F3EE] text-sm font-medium rounded-sm hover:bg-[#2A2925] transition-colors duration-100"
            >
              <BoundingBox size={16} weight="bold" />
              Open Studio
              <ArrowRight size={14} />
            </button>
            <a
              href={GITHUB_REPO_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-8 py-3 bg-white text-[#141413] text-sm font-medium rounded-sm border border-[#DDD9D0] hover:border-[#B1ADA1] transition-colors"
            >
              <GithubLogo size={16} />
              View on GitHub
            </a>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-[#DDD9D0] py-6 px-6 bg-white">
        <div className="max-w-7xl mx-auto flex items-center justify-between text-xs text-[#B1ADA1] font-mono">
          <span>YOLO26 CV Platform · Open Source · MIT License</span>
          <span>Powered by TensorFlow.js v4.22 · Ultralytics YOLO26</span>
        </div>
      </footer>
    </div>
  );
}
