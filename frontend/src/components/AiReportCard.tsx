import { useState, type FC } from 'react';
import { Sparkles, ShieldCheck, ShieldAlert, AlertTriangle, CheckCircle2, Loader2 } from 'lucide-react';
import { api } from '../services/api';
import type { MissionComplianceReport } from '../types';

interface AiReportCardProps {
  jobId: string | null;
  report: MissionComplianceReport | null;
  onReportGenerated: (report: MissionComplianceReport) => void;
}

export const AiReportCard: FC<AiReportCardProps> = ({ jobId, report, onReportGenerated }) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (!jobId) return;
    setIsGenerating(true);
    setErrorMessage(null);
    try {
      const data = await api.generateAiReport(jobId);
      onReportGenerated(data);
    } catch (err: unknown) {
      console.error('Failed to generate AI report:', err);
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        'Failed to generate Gemini AI report. Ensure the mission has completed analysis.';
      setErrorMessage(msg);
    } finally {
      setIsGenerating(false);
    }
  };

  const threatColor =
    report?.threat_level === 'CRITICAL'
      ? 'bg-rose-500/10 text-rose-400 border-rose-500/30'
      : report?.threat_level === 'ELEVATED'
      ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
      : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';

  const isGrounded = report?.grounding_audit.verdict === 'VERIFIED_GROUNDED';

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden flex flex-col space-y-5">
      {/* Background radial gradient accent */}
      <div className="absolute top-0 right-0 -mr-20 -mt-20 w-64 h-64 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-gradient-to-tr from-indigo-500/20 to-sky-500/20 border border-indigo-500/30 rounded-xl text-indigo-400 shadow-[0_0_15px_rgba(99,102,241,0.2)]">
            <Sparkles className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h3 className="text-base font-bold text-slate-100">
                Gemini Autonomous Safety Briefing
              </h3>
              <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-indigo-950 text-indigo-300 border border-indigo-800">
                GenAI 2.5
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Multimodal synthesis, spatio-temporal causal reasoning, and verified zero-hallucination auditing.
            </p>
          </div>
        </div>

        {/* Action Button */}
        <div>
          <button
            onClick={handleGenerate}
            disabled={!jobId || isGenerating}
            className={`px-4 py-2 rounded-xl text-xs font-semibold flex items-center space-x-2 transition ${
              isGenerating
                ? 'bg-indigo-600/50 text-indigo-200 cursor-wait'
                : 'bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white shadow-lg hover:shadow-indigo-500/25 active:scale-95'
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            {isGenerating ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Synthesizing Telemetry...</span>
              </>
            ) : report ? (
              <>
                <Sparkles className="w-4 h-4" />
                <span>Regenerate Briefing</span>
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                <span>Generate AI Compliance Briefing</span>
              </>
            )}
          </button>
        </div>
      </div>

      {errorMessage && (
        <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center space-x-2">
          <AlertTriangle className="w-4 h-4 flex-shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Body when Report Exists */}
      {report ? (
        <div className="space-y-5 pt-2">
          {/* Metadata Badges Bar */}
          <div className="flex flex-wrap items-center gap-2 text-xs">
            {/* Threat Level */}
            <span className={`inline-flex items-center px-3 py-1 rounded-lg font-bold border ${threatColor}`}>
              <AlertTriangle className="w-3.5 h-3.5 mr-1.5" />
              THREAT LEVEL: {report.threat_level}
            </span>

            {/* Grounding Verification */}
            <span
              className={`inline-flex items-center px-3 py-1 rounded-lg font-bold border ${
                isGrounded
                  ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                  : 'bg-amber-500/10 text-amber-400 border-amber-500/30'
              }`}
            >
              {isGrounded ? (
                <ShieldCheck className="w-3.5 h-3.5 mr-1.5" />
              ) : (
                <ShieldAlert className="w-3.5 h-3.5 mr-1.5" />
              )}
              {isGrounded ? '100% GROUNDED' : 'AUDIT DEVIATION'} (Score: {report.grounding_audit.grounding_score.toFixed(2)})
            </span>

            {/* Model Used */}
            <span className="px-3 py-1 rounded-lg font-mono text-[11px] bg-slate-950 border border-slate-800 text-slate-400">
              Model: {report.model_used}
            </span>

            {/* Peak Speed */}
            <span className="px-3 py-1 rounded-lg font-mono text-[11px] bg-slate-950 border border-slate-800 text-slate-300">
              Peak: {report.peak_speed_kmh} km/h
            </span>
          </div>

          {/* Executive Summary Narrative */}
          <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-1.5">
            <h4 className="text-xs font-bold text-sky-400 uppercase tracking-wider flex items-center space-x-1.5">
              <span>Executive Narrative Summary</span>
            </h4>
            <p className="text-xs leading-relaxed text-slate-200">
              {report.executive_summary}
            </p>
          </div>

          {/* Incident Breakdown */}
          {report.incidents.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                Causal Incident Analysis ({report.incidents.length})
              </h4>
              <div className="space-y-2">
                {report.incidents.map((inc, i) => (
                  <div
                    key={i}
                    className="p-3 rounded-xl bg-slate-950/50 border border-slate-800/80 flex items-start justify-between gap-3 text-xs"
                  >
                    <div className="space-y-1 flex-1">
                      <div className="flex items-center space-x-2">
                        <span className="font-mono text-xs font-semibold px-2 py-0.5 rounded bg-slate-800 text-sky-300 border border-slate-700">
                          #{inc.track_id}
                        </span>
                        <span className="font-bold text-slate-200 capitalize">
                          {inc.class_name}
                        </span>
                        <span className="text-slate-500">&bull;</span>
                        <span className="text-slate-300 font-medium">
                          {inc.rule_type.replace(/_/g, ' ')}
                        </span>
                      </div>
                      <p className="text-slate-400 text-xs leading-normal">
                        {inc.narrative}
                      </p>
                    </div>

                    <span
                      className={`flex-shrink-0 px-2 py-0.5 rounded text-[10px] font-bold ${
                        inc.severity === 'CRITICAL'
                          ? 'bg-rose-500/10 text-rose-400 border border-rose-500/30'
                          : 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
                      }`}
                    >
                      {inc.severity}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Operational Countermeasures */}
          {report.operational_recommendations.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                Actionable Remediation Countermeasures
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {report.operational_recommendations.map((rec, i) => (
                  <div
                    key={i}
                    className="p-3 rounded-xl bg-slate-950/40 border border-indigo-950/40 flex items-start space-x-2.5 text-xs"
                  >
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                    <span className="text-slate-300">{rec}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="py-8 text-center text-slate-500 flex flex-col items-center justify-center space-y-2 border border-dashed border-slate-800 rounded-xl">
          <Sparkles className="w-8 h-8 text-slate-600" />
          <p className="text-xs font-medium text-slate-400">
            No AI compliance briefing generated yet.
          </p>
          <p className="text-[11px] text-slate-600 max-w-sm">
            Click &ldquo;Generate AI Compliance Briefing&rdquo; to analyze tracking kinematics and generate an FAA-grade safety briefing with Gemini.
          </p>
        </div>
      )}
    </div>
  );
};
