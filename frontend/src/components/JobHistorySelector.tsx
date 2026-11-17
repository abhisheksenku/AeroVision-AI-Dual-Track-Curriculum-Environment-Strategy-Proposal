import type { FC } from 'react';
import { History, CheckCircle2, Clock, AlertCircle, RefreshCw } from 'lucide-react';
import type { JobResponse } from '../types';

interface JobHistorySelectorProps {
  jobs: JobResponse[];
  selectedJobId: string | null;
  onSelectJob: (job: JobResponse) => void;
  onRefresh: () => void;
  isLoading: boolean;
}

export const JobHistorySelector: FC<JobHistorySelectorProps> = ({
  jobs,
  selectedJobId,
  onSelectJob,
  onRefresh,
  isLoading,
}) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <History className="w-4 h-4 text-sky-400" />
          <h3 className="text-sm font-bold text-slate-200">Surveillance Missions Log</h3>
        </div>
        <button
          onClick={onRefresh}
          className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-200 transition"
          title="Refresh missions list"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-sky-400' : ''}`} />
        </button>
      </div>

      <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
        {jobs.length > 0 ? (
          jobs.map((job) => {
            const isSelected = job.job_id === selectedJobId;
            return (
              <div
                key={job.job_id}
                onClick={() => onSelectJob(job)}
                className={`p-3 rounded-xl border cursor-pointer transition flex items-center justify-between ${
                  isSelected
                    ? 'bg-sky-500/10 border-sky-500/50 text-slate-100 shadow-sm'
                    : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 text-slate-300'
                }`}
              >
                <div className="min-w-0 flex-1 mr-2">
                  <div className="flex items-center space-x-2">
                    <span className="font-mono text-xs font-semibold text-sky-300">
                      {job.job_id}
                    </span>
                    <span className="text-[10px] text-slate-400 truncate">
                      {job.filename}
                    </span>
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1 flex items-center space-x-2">
                    <span>{job.created_at.slice(0, 19).replace('T', ' ')}</span>
                    <span>&bull;</span>
                    <span>{job.total_frames || 0} frames</span>
                  </div>
                </div>

                <div className="flex-shrink-0">
                  {job.status === 'COMPLETED' ? (
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                      <CheckCircle2 className="w-3 h-3 mr-1" />
                      Ready
                    </span>
                  ) : job.status === 'PROCESSING' ? (
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 animate-pulse">
                      <Clock className="w-3 h-3 mr-1" />
                      {job.progress_percent}%
                    </span>
                  ) : job.status === 'FAILED' ? (
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-rose-500/10 text-rose-400 border border-rose-500/20">
                      <AlertCircle className="w-3 h-3 mr-1" />
                      Failed
                    </span>
                  ) : (
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-slate-800 text-slate-400 border border-slate-700">
                      Queued
                    </span>
                  )}
                </div>
              </div>
            );
          })
        ) : (
          <div className="text-center py-6 text-slate-500 text-xs">
            No missions recorded yet. Upload a video above to begin.
          </div>
        )}
      </div>
    </div>
  );
};
