import type { FC } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { BarChart3, Gauge, PieChart as PieIcon, Split } from 'lucide-react';
import type { MissionAnalytics } from '../types';

interface AnalyticsChartsProps {
  analytics: MissionAnalytics | null;
}

const COLORS = ['#38bdf8', '#818cf8', '#34d399', '#f59e0b', '#f43f5e', '#a855f7'];
const SEVERITY_COLORS: Record<string, string> = {
  CRITICAL: '#f43f5e',
  WARNING: '#f59e0b',
};

export const AnalyticsCharts: FC<AnalyticsChartsProps> = ({ analytics }) => {
  if (!analytics) return null;

  // 1. Class Distribution Data
  const classDistData = Object.entries(analytics.counts_by_class || {}).map(([name, count]) => ({
    name: name.toUpperCase(),
    count,
  }));

  // 2. Speed Profiles Data
  const speedData = Object.entries(analytics.class_statistics || {}).map(([name, stat]) => ({
    name: name.toUpperCase(),
    meanSpeed: parseFloat(stat.mean_speed_kmh.toFixed(1)),
    maxSpeed: parseFloat(stat.max_speed_kmh.toFixed(1)),
  }));

  // 3. Violation Severity Breakdown
  const violations = analytics.safety_audit.violations_log || [];
  const severityCounts: Record<string, number> = { CRITICAL: 0, WARNING: 0 };
  violations.forEach((v) => {
    severityCounts[v.severity] = (severityCounts[v.severity] || 0) + 1;
  });

  const severityPieData = Object.entries(severityCounts)
    .filter(([_, count]) => count > 0)
    .map(([severity, count]) => ({
      name: severity,
      value: count,
    }));

  // 4. Line Crossings
  const lineCrossingData = Object.entries(analytics.line_crossings || {}).map(([key, count]) => ({
    direction: key.replace('_', ' ').toUpperCase(),
    crossings: count,
  }));

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* 1. Fleet Composition */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col">
        <div className="flex items-center space-x-2 mb-4">
          <BarChart3 className="w-4 h-4 text-sky-400" />
          <h3 className="text-sm font-bold text-slate-200">Vehicle Classification Distribution</h3>
        </div>
        <div className="h-64 w-full">
          {classDistData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={classDistData} margin={{ top: 10, right: 10, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} tickLine={false} />
                <YAxis stroke="#94a3b8" fontSize={11} allowDecimals={false} tickLine={false} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    border: '1px solid #334155',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                  itemStyle={{ color: '#38bdf8' }}
                />
                <Bar dataKey="count" fill="#38bdf8" radius={[4, 4, 0, 0]} name="Vehicle Count" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-slate-500 text-xs">
              No vehicles detected in this mission.
            </div>
          )}
        </div>
      </div>

      {/* 2. Speed Analytics */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col">
        <div className="flex items-center space-x-2 mb-4">
          <Gauge className="w-4 h-4 text-indigo-400" />
          <h3 className="text-sm font-bold text-slate-200">Speed Profile (Mean vs Peak km/h)</h3>
        </div>
        <div className="h-64 w-full">
          {speedData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={speedData} margin={{ top: 10, right: 10, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} tickLine={false} />
                <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    border: '1px solid #334155',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                <Bar dataKey="meanSpeed" fill="#818cf8" radius={[4, 4, 0, 0]} name="Mean Speed (km/h)" />
                <Bar dataKey="maxSpeed" fill="#f43f5e" radius={[4, 4, 0, 0]} name="Peak Speed (km/h)" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-slate-500 text-xs">
              Speed telemetry not available.
            </div>
          )}
        </div>
      </div>

      {/* 3. Violation Severity Breakdown */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col">
        <div className="flex items-center space-x-2 mb-4">
          <PieIcon className="w-4 h-4 text-rose-400" />
          <h3 className="text-sm font-bold text-slate-200">Safety Infractions by Severity</h3>
        </div>
        <div className="h-64 w-full flex items-center justify-center">
          {severityPieData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={severityPieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={85}
                  paddingAngle={5}
                  dataKey="value"
                  label={({ name, percent }: { name?: string; percent?: number }) =>
                    `${name || ''}: ${((percent || 0) * 100).toFixed(0)}%`
                  }
                  labelLine={false}
                >
                  {severityPieData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={SEVERITY_COLORS[entry.name] || COLORS[index % COLORS.length]}
                    />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    border: '1px solid #334155',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-center text-slate-500 text-xs">
              <span className="text-emerald-400 font-semibold block mb-1">Zero Violations Recorded</span>
              All trajectories complied with speed and zone limits.
            </div>
          )}
        </div>
      </div>

      {/* 4. Directional Traffic Flows */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col">
        <div className="flex items-center space-x-2 mb-4">
          <Split className="w-4 h-4 text-emerald-400" />
          <h3 className="text-sm font-bold text-slate-200">Tripwire Directional Crossings</h3>
        </div>
        <div className="h-64 w-full">
          {lineCrossingData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={lineCrossingData} margin={{ top: 10, right: 10, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                <XAxis dataKey="direction" stroke="#94a3b8" fontSize={11} tickLine={false} />
                <YAxis stroke="#94a3b8" fontSize={11} allowDecimals={false} tickLine={false} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    border: '1px solid #334155',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
                <Bar dataKey="crossings" fill="#34d399" radius={[4, 4, 0, 0]} name="Crossings" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-slate-500 text-xs">
              No tripwire crossing events triggered.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
