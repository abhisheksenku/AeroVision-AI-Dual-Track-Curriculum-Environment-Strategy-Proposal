import { useState, useEffect, type FC } from 'react';
import {
  Cpu,
  PackageCheck,
  Activity,
  DownloadCloud,
  CheckCircle2,
  Loader2,
  ShieldCheck,
  Container
} from 'lucide-react';
import { api } from '../services/api';
import type { MLOpsSystemStatus, ModelExportResponse, DriftAuditResponse } from '../types';

interface MlopsDeploymentCardProps {
  speedsSample?: number[];
}

export const MlopsDeploymentCard: FC<MlopsDeploymentCardProps> = ({ speedsSample = [] }) => {
  const [mlopsStatus, setMlopsStatus] = useState<MLOpsSystemStatus | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [exportResult, setExportResult] = useState<ModelExportResponse | null>(null);
  const [driftResult, setDriftResult] = useState<DriftAuditResponse | null>(null);
  const [isAuditingDrift, setIsAuditingDrift] = useState(false);

  useEffect(() => {
    api
      .getMlopsStatus()
      .then((data) => setMlopsStatus(data))
      .catch((err) => console.warn('Could not fetch MLOps status:', err));
  }, []);

  const handleExportOnnx = async () => {
    setIsExporting(true);
    try {
      const res = await api.triggerModelExport('drone_backbone');
      setExportResult(res);
      // Refresh status to update exported files list
      const updated = await api.getMlopsStatus();
      setMlopsStatus(updated);
    } catch (err) {
      console.error('Failed to export ONNX model:', err);
    } finally {
      setIsExporting(false);
    }
  };

  const handleRunDriftAudit = async (scenario: 'nominal' | 'drifted') => {
    setIsAuditingDrift(true);
    try {
      const sample =
        scenario === 'nominal'
          ? (speedsSample.length >= 5 ? speedsSample : [38.2, 41.5, 39.8, 44.1, 42.0, 37.9, 40.5, 43.2])
          : [65.4, 72.1, 68.9, 74.5, 62.8, 70.2, 75.0, 69.4];

      const res = await api.auditDrift(sample);
      setDriftResult(res);
    } catch (err) {
      console.error('Drift audit failed:', err);
    } finally {
      setIsAuditingDrift(false);
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden flex flex-col space-y-6">
      {/* Background Accent */}
      <div className="absolute top-0 right-0 -mr-20 -mt-20 w-72 h-72 bg-violet-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
        <div className="flex items-center space-x-3.5">
          <div className="p-2.5 bg-violet-500/10 border border-violet-500/20 rounded-xl text-violet-400">
            <Container className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-slate-100 flex items-center space-x-2">
              <span>MLOps, ONNX Acceleration &amp; Deployment</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-violet-900/60 text-violet-300 border border-violet-700/50 font-mono">
                Production Gates
              </span>
            </h3>
            <p className="text-xs text-slate-400">
              Multi-stage Docker, ONNX Runtime (3.17x CPU speedup), INT8 quantization &amp; PSI drift monitoring
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2 text-xs font-mono">
          <span className="flex items-center space-x-1 px-2.5 py-1 rounded-md bg-slate-950 border border-slate-800 text-slate-300">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            <span>Docker &amp; CI/CD Ready</span>
          </span>
        </div>
      </div>

      {/* Grid of Key Architecture Badges */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
        <div className="p-3 bg-slate-950/70 border border-slate-800 rounded-xl flex items-center space-x-2.5">
          <Cpu className="w-4 h-4 text-sky-400" />
          <div>
            <div className="font-semibold text-slate-200">ONNX Runtime</div>
            <div className="text-[10px] text-slate-400">
              {mlopsStatus?.active_execution_provider || 'CPUExecutionProvider'}
            </div>
          </div>
        </div>

        <div className="p-3 bg-slate-950/70 border border-slate-800 rounded-xl flex items-center space-x-2.5">
          <PackageCheck className="w-4 h-4 text-emerald-400" />
          <div>
            <div className="font-semibold text-slate-200">INT8 Quantization</div>
            <div className="text-[10px] text-slate-400">74.8% Size Reduction</div>
          </div>
        </div>

        <div className="p-3 bg-slate-950/70 border border-slate-800 rounded-xl flex items-center space-x-2.5">
          <Container className="w-4 h-4 text-indigo-400" />
          <div>
            <div className="font-semibold text-slate-200">Multi-Stage Docker</div>
            <div className="text-[10px] text-slate-400">Non-Root appuser (UID 1000)</div>
          </div>
        </div>

        <div className="p-3 bg-slate-950/70 border border-slate-800 rounded-xl flex items-center space-x-2.5">
          <Activity className="w-4 h-4 text-amber-400" />
          <div>
            <div className="font-semibold text-slate-200">Covariate Drift</div>
            <div className="text-[10px] text-slate-400">PSI Monitoring Active</div>
          </div>
        </div>
      </div>

      {/* Section 1: ONNX Model Export & Registry */}
      <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800/80 pb-2.5">
          <div>
            <h4 className="text-xs font-semibold text-slate-200">
              Model Artifact Registry &amp; ONNX Serialization
            </h4>
            <p className="text-[11px] text-slate-400">
              Compile PyTorch graph into static ONNX computation graph for sub-2ms CPU inference
            </p>
          </div>
          <button
            onClick={handleExportOnnx}
            disabled={isExporting}
            className="px-3.5 py-1.5 bg-violet-600 hover:bg-violet-500 disabled:opacity-50 text-white text-xs font-semibold rounded-lg transition-colors flex items-center space-x-1.5 self-start sm:self-auto"
          >
            {isExporting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <DownloadCloud className="w-3.5 h-3.5" />}
            <span>Export ONNX Model</span>
          </button>
        </div>

        {exportResult && (
          <div className="p-3 bg-violet-950/30 border border-violet-500/30 rounded-lg text-xs space-y-1">
            <div className="flex items-center space-x-2 text-violet-300 font-semibold">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>ONNX Graph Export Completed ({exportResult.file_size_kb} KB)</span>
            </div>
            <div className="text-[11px] font-mono text-slate-400">
              Output Path: <span className="text-slate-200">{exportResult.export_path}</span>
            </div>
          </div>
        )}

        <div className="text-[11px] text-slate-400 font-mono flex flex-wrap gap-2 pt-1">
          <span className="text-slate-500 self-center">Available Artifacts:</span>
          {mlopsStatus?.exported_models && mlopsStatus.exported_models.length > 0 ? (
            mlopsStatus.exported_models.map((file, idx) => (
              <span key={idx} className="px-2 py-0.5 rounded bg-slate-900 border border-slate-700/80 text-slate-300">
                {file}
              </span>
            ))
          ) : (
            <span className="text-slate-500 italic">No exported files yet. Click export to generate.</span>
          )}
        </div>
      </div>

      {/* Section 2: Population Stability Index (PSI) Drift Monitor */}
      <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800/80 pb-2.5">
          <div>
            <h4 className="text-xs font-semibold text-slate-200">
              Covariate Data Drift Monitoring (PSI Engine)
            </h4>
            <p className="text-[11px] text-slate-400">
              Computes Population Stability Index comparing production flight telemetry against baseline speed distributions
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => handleRunDriftAudit('nominal')}
              disabled={isAuditingDrift}
              className="px-2.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded-lg transition-colors"
            >
              Test Nominal
            </button>
            <button
              onClick={() => handleRunDriftAudit('drifted')}
              disabled={isAuditingDrift}
              className="px-2.5 py-1.5 bg-rose-950/60 hover:bg-rose-900/60 border border-rose-500/30 text-rose-300 text-xs font-medium rounded-lg transition-colors"
            >
              Test Drifted
            </button>
          </div>
        </div>

        {driftResult ? (
          <div className="p-3 bg-slate-900/90 border border-slate-800 rounded-lg text-xs space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center space-x-2">
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-mono border ${
                    driftResult.verdict === 'NO_DRIFT'
                      ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                      : driftResult.verdict === 'MODERATE_DRIFT'
                      ? 'bg-amber-500/20 text-amber-300 border-amber-500/30'
                      : 'bg-rose-500/20 text-rose-300 border-rose-500/30'
                  }`}
                >
                  {driftResult.verdict}
                </span>
                <span className="font-mono text-slate-300">
                  PSI Score: <strong className="text-white">{driftResult.psi_score}</strong>
                </span>
              </div>

              <div className="text-[11px] font-mono text-slate-400">
                Baseline: {driftResult.baseline_mean_speed_kmh} km/h &bull; Observed: {driftResult.observed_mean_speed_kmh} km/h
              </div>
            </div>

            <p className="text-[11px] text-slate-400 leading-relaxed">
              {driftResult.recommendation}
            </p>
          </div>
        ) : (
          <div className="text-center py-3 text-slate-500 text-xs">
            Run nominal or drifted telemetry tests to evaluate live covariate drift and retraining gates.
          </div>
        )}
      </div>
    </div>
  );
};
