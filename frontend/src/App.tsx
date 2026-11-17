import { Activity, Cpu, Eye, ShieldCheck, Video } from 'lucide-react';

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-sky-500/10 border border-sky-500/30 rounded-lg text-sky-400">
            <Eye className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-sky-400 to-cyan-300 bg-clip-text text-transparent">
              AeroVision AI
            </h1>
            <p className="text-xs text-slate-400">Intelligent Drone Video Analytics & Agent Platform</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
            <span className="w-1.5 h-1.5 mr-1.5 bg-emerald-400 rounded-full animate-pulse" />
            Phase 1 Configured
          </span>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
        {/* Status Banner */}
        <div className="p-4 rounded-xl border border-sky-500/20 bg-sky-950/20 flex items-start space-x-4">
          <ShieldCheck className="w-6 h-6 text-sky-400 mt-1 flex-shrink-0" />
          <div>
            <h2 className="text-sm font-semibold text-sky-200">System Environment Initialized</h2>
            <p className="text-xs text-slate-400 mt-1">
              Python 3.11, PyTorch CPU, Ultralytics YOLO11, ByteTrack (Lapx), and React 19 Frontend are operational.
            </p>
          </div>
        </div>

        {/* Modules Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
            <div className="flex items-center space-x-2 text-sky-400">
              <Video className="w-5 h-5" />
              <h3 className="font-medium text-sm">Video Ingestion & Vision</h3>
            </div>
            <p className="text-xs text-slate-400">
              OpenCV streaming frame extraction paired with YOLO11n object detector.
            </p>
          </div>

          <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
            <div className="flex items-center space-x-2 text-indigo-400">
              <Activity className="w-5 h-5" />
              <h3 className="font-medium text-sm">Tracking & Analytics</h3>
            </div>
            <p className="text-xs text-slate-400">
              ByteTrack multi-object tracking with trajectory mapping and zone counting.
            </p>
          </div>

          <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
            <div className="flex items-center space-x-2 text-cyan-400">
              <Cpu className="w-5 h-5" />
              <h3 className="font-medium text-sm">LangGraph AI Agent</h3>
            </div>
            <p className="text-xs text-slate-400">
              RAG with Qdrant and Gemini API for grounded drone inspection reporting.
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 py-4 px-6 text-center text-xs text-slate-500">
        AeroVision AI &copy; 2026 &bull; Architecture and Phase 1 Environment Ready
      </footer>
    </div>
  );
}
