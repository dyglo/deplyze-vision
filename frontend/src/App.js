import "./App.css";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import Landing from "./pages/Landing";
import Studio from "./pages/Studio";
import Projects from "./pages/Projects";
import Datasets from "./pages/Datasets";
import ModelHub from "./pages/ModelHub";
import Results from "./pages/Results";
import Benchmark from "./pages/Benchmark";

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route element={<Layout />}>
            <Route path="/studio" element={<Studio />} />
            <Route path="/projects" element={<Projects />} />
            <Route path="/datasets" element={<Datasets />} />
            <Route path="/models" element={<ModelHub />} />
            <Route path="/results" element={<Results />} />
            <Route path="/benchmark" element={<Benchmark />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
