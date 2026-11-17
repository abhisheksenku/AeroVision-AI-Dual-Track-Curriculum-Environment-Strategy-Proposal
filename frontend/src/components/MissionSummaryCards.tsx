import type { FC } from 'react';
import { ShieldCheck, ShieldAlert, Car, Clock, Zap } from 'lucide-react';
import type { MissionAnalytics } from '../types';

interface MissionSummaryCardsProps {
  analytics: MissionAnalytics | null;
}

export const MissionSummaryCards: FC<MissionSummaryCardsProps> = ({ analytics }) => {
  if (!analytics) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="p-4 rounded-xl bg-slate-900 border border-slate-800 animate-pulse h-24" />
        ))}
      </div>
    );
  }

  const isCompliant = analytics.safety_audit.compliance_status === 'COMPLIANT';

  // Calculate highest recorded speed in mission
  let maxFleetSpeed = 0;
  if (analytics.class_statistics) {
    Object.values(analytics.class_statistics).forEach((stat) => {
      if (stat.max_speed_kmh > maxFleetSpeed) {
        maxFleetSpeed = stat.max_speed_kmh;
      }
    });
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* 1. Tracked Objects */}
      <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-between shadow-md">
        <div>
          <p className="text-xs font-medium text-slate-400">Total Tracked Objects</p>
          <p className="text-2xl font-black text-slate-100 mt-1">
            {analytics.total_unique_objects}
          </p>
          <p className="text-[11px] text-sky-400 mt-0.5">
            {Object.keys(analytics.counts_by_class || {}).length} unique classes
          </p>
        </div>
        <div className="p-3 rounded-lg bg-sky-500/10 border border-sky-500/20 text-sky-400">
          <Car className="w-5 h-5" />
        </div>
      </div>

      {/* 2. Mission Duration */}
      <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-between shadow-md">
        <div>
          <p className="text-xs font-medium text-slate-400">Surveillance Duration</p>
          <p className="text-2xl font-black text-slate-100 mt-1">
            {analytics.duration_sec.toFixed(1)}s
          </p>
          <p className="text-[11px] text-indigo-400 mt-0.5">
            High-speed GSD calibrated
          </p>
        </div>
        <div className="p-3 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
          <Clock className="w-5 h-5" />
        </div>
      </div>

      {/* 3. Peak Velocity */}
      <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-between shadow-md">
        <div>
          <p className="text-xs font-medium text-slate-400">Peak Fleet Velocity</p>
          <p className="text-2xl font-black text-slate-100 mt-1">
            {maxFleetSpeed.toFixed(1)} <span className="text-xs font-normal text-slate-400">km/h</span>
          </p>
          <p className="text-[11px] text-amber-400 mt-0.5">
            EMA velocity smoothed
          </p>
        </div>
        <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-400">
          <Zap className="w-5 h-5" />
        </div>
      </div>

      {/* 4. Compliance & Safety Audit */}
      <div
        className={`p-4 rounded-xl border flex items-center justify-between shadow-md ${
          isCompliant
            ? 'bg-emerald-950/20 border-emerald-500/30'
            : 'bg-rose-950/20 border-rose-500/30'
        }`}
      >
        <div>
          <p className="text-xs font-medium text-slate-400">Compliance Audit</p>
          <p
            className={`text-xl font-black mt-1 ${
              isCompliant ? 'text-emerald-400' : 'text-rose-400'
            }`}
          >
            {isCompliant ? 'COMPLIANT' : 'FLAGGED'}
          </p>
          <p className="text-[11px] text-slate-400 mt-0.5">
            {analytics.safety_audit.total_violations} safety incident
            {analytics.safety_audit.total_violations === 1 ? '' : 's'}
          </p>
        </div>
        <div
          className={`p-3 rounded-lg border ${
            isCompliant
              ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
              : 'bg-rose-500/10 border-rose-500/30 text-rose-400'
          }`}
        >
          {isCompliant ? <ShieldCheck className="w-5 h-5" /> : <ShieldAlert className="w-5 h-5" />}
        </div>
      </div>
    </div>
  );
};
