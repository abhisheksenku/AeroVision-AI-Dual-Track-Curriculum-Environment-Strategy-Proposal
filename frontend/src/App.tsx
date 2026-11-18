import { useState, useEffect, useCallback, useRef } from 'react';
import { Header } from './components/Header';
import { VideoUploader } from './components/VideoUploader';
import { VideoPlayer } from './components/VideoPlayer';
import { MissionSummaryCards } from './components/MissionSummaryCards';
import { AnalyticsCharts } from './components/AnalyticsCharts';
import { ViolationsTable } from './components/ViolationsTable';
import { JobHistorySelector } from './components/JobHistorySelector';
import { AiReportCard } from './components/AiReportCard';
import { RegulatoryCitationsCard } from './components/RegulatoryCitationsCard';
import { AgentCommanderConsole } from './components/AgentCommanderConsole';
import { api } from './services/api';
import type { JobResponse, MissionAnalytics, MissionComplianceReport } from './types';
import { ShieldCheck, Video, Activity, Sparkles, BarChart2 } from 'lucide-react';

export default function App() {
  const [jobs, setJobs] = useState<JobResponse[]>([]);
  const [selectedJob, setSelectedJob] = useState<JobResponse | null>(null);
  const [analytics, setAnalytics] = useState<MissionAnalytics | null>(null);
  const [aiReport, setAiReport] = useState<MissionComplianceReport | null>(null);
  const [isLoadingJobs, setIsLoadingJobs] = useState(false);
  const [, setIsLoadingAnalytics] = useState(false);

  // Polling ref to avoid concurrent timers
  const pollTimerRef = useRef<number | null>(null);

  // 1. Fetch all jobs
  const loadJobs = useCallback(async (autoSelectLatest = false) => {
    try {
      setIsLoadingJobs(true);
      const data = await api.getJobs();
      setJobs(data);

      if (autoSelectLatest && data.length > 0) {
        setSelectedJob(data[0]);
      }
    } catch (err) {
      console.warn('Failed to fetch jobs list:', err);
    } finally {
      setIsLoadingJobs(false);
    }
  }, []);

  // 2. Fetch analytics for a given job
  const loadAnalytics = useCallback(async (jobId: string) => {
    try {
      setIsLoadingAnalytics(true);
      const data = await api.getAnalytics(jobId);
      setAnalytics(data);
    } catch (err) {
      console.warn(`Analytics not ready or unavailable for job ${jobId}:`, err);
      setAnalytics(null);
    } finally {
      setIsLoadingAnalytics(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    loadJobs(true);
  }, [loadJobs]);

  // When selectedJob changes, fetch analytics and existing AI report if completed
  useEffect(() => {
    if (selectedJob?.status === 'COMPLETED') {
      loadAnalytics(selectedJob.job_id);
      api.getAiReport(selectedJob.job_id)
        .then((rep) => setAiReport(rep))
        .catch(() => setAiReport(null));
    } else {
      setAnalytics(null);
      setAiReport(null);
    }
  }, [selectedJob?.job_id, selectedJob?.status, loadAnalytics]);

  // Polling loop for active in-progress job
  useEffect(() => {
    if (selectedJob && (selectedJob.status === 'QUEUED' || selectedJob.status === 'PROCESSING')) {
      const pollInterval = window.setInterval(async () => {
        try {
          const updated = await api.getJob(selectedJob.job_id);
          setSelectedJob(updated);

          if (updated.status === 'COMPLETED' || updated.status === 'FAILED') {
            loadJobs(false);
          }
        } catch (err) {
          console.error('Error polling job status:', err);
        }
      }, 1500);

      pollTimerRef.current = pollInterval;
      return () => window.clearInterval(pollInterval);
    }
  }, [selectedJob?.job_id, selectedJob?.status, loadJobs]);

  const handleJobCreated = (newJob: JobResponse) => {
    setJobs((prev) => [newJob, ...prev]);
    setSelectedJob(newJob);
  };

  const handleSelectJob = (job: JobResponse) => {
    setSelectedJob(job);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-sky-500 selection:text-white">
      {/* Top Navigation */}
      <Header onRefreshJobs={() => loadJobs(false)} />

      {/* Main Mission Control Body */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 space-y-6">
        {/* Banner with Architecture Context */}
        <div className="p-4 rounded-2xl bg-gradient-to-r from-sky-950/40 via-indigo-950/20 to-slate-900 border border-sky-500/20 flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-lg">
          <div className="flex items-center space-x-3.5">
            <div className="p-2.5 bg-sky-500/10 border border-sky-500/30 rounded-xl text-sky-400">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-slate-100">
                AeroVision Autonomous Surveillance &amp; Compliance Hub
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Full-stack pipeline: YOLO11 inference, ByteTrack Kalman velocity estimation, and automated safety auditing.
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2 text-xs text-slate-400">
            <span className="flex items-center space-x-1 px-2.5 py-1 rounded-md bg-slate-900/80 border border-slate-800">
              <Video className="w-3.5 h-3.5 text-sky-400" />
              <span>YOLO11n</span>
            </span>
            <span className="flex items-center space-x-1 px-2.5 py-1 rounded-md bg-slate-900/80 border border-slate-800">
              <Activity className="w-3.5 h-3.5 text-indigo-400" />
              <span>ByteTrack</span>
            </span>
            <span className="flex items-center space-x-1 px-2.5 py-1 rounded-md bg-slate-900/80 border border-slate-800">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              <span>Audit Engine</span>
            </span>
          </div>
        </div>

        {/* Section 1: Ingestion & Mission Log */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <VideoUploader onJobCreated={handleJobCreated} activeJob={selectedJob} />
          </div>
          <div>
            <JobHistorySelector
              jobs={jobs}
              selectedJobId={selectedJob?.job_id || null}
              onSelectJob={handleSelectJob}
              onRefresh={() => loadJobs(false)}
              isLoading={isLoadingJobs}
            />
          </div>
        </div>

        {/* Section 2: Video Player & Mission Summary KPIs */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
          <div className="lg:col-span-2">
            <VideoPlayer job={selectedJob} />
          </div>
          <div className="space-y-4">
            <div className="flex items-center space-x-2 text-xs font-bold uppercase tracking-wider text-slate-400">
              <BarChart2 className="w-4 h-4 text-sky-400" />
              <span>Executive Telemetry</span>
            </div>
            <MissionSummaryCards analytics={analytics} />

            {selectedJob && (
              <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 text-xs space-y-2">
                <div className="text-slate-300 font-semibold">Active Mission Details</div>
                <div className="grid grid-cols-2 gap-2 text-slate-400 font-mono text-[11px]">
                  <div>File: <span className="text-slate-200">{selectedJob.filename}</span></div>
                  <div>Status: <span className="text-sky-400 font-bold">{selectedJob.status}</span></div>
                  <div>Frames: <span className="text-slate-200">{selectedJob.processed_frames} / {selectedJob.total_frames || 'N/A'}</span></div>
                  <div>Progress: <span className="text-emerald-400 font-bold">{selectedJob.progress_percent}%</span></div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Section 3: Visual Analytics Charts */}
        {analytics && (
          <div className="space-y-4 pt-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-xs font-bold uppercase tracking-wider text-slate-400">
                <Activity className="w-4 h-4 text-indigo-400" />
                <span>Kinematic &amp; Safety Visualizations</span>
              </div>
              <span className="text-[11px] font-mono text-slate-500">
                GSD Calibration: 0.05 m/px &bull; FPS: 30.0
              </span>
            </div>
            <AnalyticsCharts analytics={analytics} />
          </div>
        )}

        {/* Section 4: Gemini AI Safety & Compliance Briefing */}
        {analytics && (
          <div className="pt-2">
            <AiReportCard
              jobId={selectedJob?.job_id || null}
              report={aiReport}
              onReportGenerated={(rep) => setAiReport(rep)}
            />
          </div>
        )}

        {/* Section 5: RAG Statutory Legal Citations (Qdrant Vector Database) */}
        {analytics && (
          <div className="pt-2">
            <RegulatoryCitationsCard jobId={selectedJob?.job_id || null} />
          </div>
        )}

        {/* Section 6: Autonomous Incident Commander (LangGraph Multi-Agent) */}
        {analytics && (
          <div className="pt-2">
            <AgentCommanderConsole jobId={selectedJob?.job_id || null} />
          </div>
        )}

        {/* Section 7: Violations Audit Log */}
        {analytics && (
          <div className="pt-2">
            <ViolationsTable violations={analytics.safety_audit.violations_log || []} />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 py-4 px-6 text-center text-xs text-slate-500 bg-slate-950/80">
        AeroVision AI &copy; 2026 &bull; Production Video Intelligence Platform &bull; Phase 12 Autonomous Agents Verified
      </footer>
    </div>
  );
}
