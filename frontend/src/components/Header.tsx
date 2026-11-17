import { useEffect, useState, type FC } from 'react';
import { Eye, Cpu, Database, Activity, RefreshCw } from 'lucide-react';
import { api } from '../services/api';
import type { SystemHealth } from '../types';

interface HeaderProps {
  onRefreshJobs?: () => void;
}

export const Header: FC<HeaderProps> = ({ onRefreshJobs }) => {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const checkHealth = async () => {
    try {
      const data = await api.getHealth();
      setHealth(data);
    } catch (err) {
      console.warn('Backend currently offline or unreachable', err);
      setHealth(null);
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleManualRefresh = async () => {
    setIsRefreshing(true);
    await checkHealth();
    if (onRefreshJobs) onRefreshJobs();
    setTimeout(() => setIsRefreshing(false), 500);
  };

  const isOnline = health?.status === 'HEALTHY';

  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50 px-6 py-3.5 flex items-center justify-between shadow-lg">
      <div className="flex items-center space-x-3">
        <div className="p-2.5 bg-sky-500/10 border border-sky-500/30 rounded-xl text-sky-400 shadow-[0_0_15px_rgba(56,189,248,0.2)]">
          <Eye className="w-6 h-6 animate-pulse" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-xl font-black tracking-tight bg-gradient-to-r from-sky-400 via-cyan-300 to-indigo-400 bg-clip-text text-transparent">
              AeroVision AI
            </h1>
            <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-slate-800 text-sky-400 border border-slate-700">
              v1.0 Pro
            </span>
          </div>
          <p className="text-xs text-slate-400">
            Autonomous Aerial Surveillance, YOLO11 Detection &amp; Safety Compliance
          </p>
        </div>
      </div>

      <div className="flex items-center space-x-3">
        {/* Hardware Status Pill */}
        {health ? (
          <div className="hidden md:flex items-center space-x-4 px-3.5 py-1.5 rounded-lg bg-slate-950/70 border border-slate-800 text-xs">
            <div className="flex items-center space-x-1.5 text-slate-300">
              <Cpu className="w-3.5 h-3.5 text-sky-400" />
              <span>CPU: {health.system_metrics.cpu_usage_percent}%</span>
            </div>
            <div className="w-px h-3.5 bg-slate-800" />
            <div className="flex items-center space-x-1.5 text-slate-300">
              <Activity className="w-3.5 h-3.5 text-indigo-400" />
              <span>RAM: {health.system_metrics.ram_used_gb} / {health.system_metrics.ram_total_gb} GB</span>
            </div>
            <div className="w-px h-3.5 bg-slate-800" />
            <div className="flex items-center space-x-1.5 text-slate-300">
              <Database className="w-3.5 h-3.5 text-emerald-400" />
              <span>SQLite</span>
            </div>
          </div>
        ) : null}

        {/* System Online / Offline Badge */}
        <div
          className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-semibold border ${
            isOnline
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
              : 'bg-rose-500/10 text-rose-400 border-rose-500/30'
          }`}
        >
          <span
            className={`w-2 h-2 rounded-full ${
              isOnline ? 'bg-emerald-400 animate-ping' : 'bg-rose-400'
            }`}
          />
          <span>{isOnline ? 'ENGINE READY' : 'OFFLINE'}</span>
        </div>

        {/* Refresh Button */}
        <button
          onClick={handleManualRefresh}
          className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition border border-slate-700/60"
          title="Refresh connection & jobs"
        >
          <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin text-sky-400' : ''}`} />
        </button>
      </div>
    </header>
  );
};
