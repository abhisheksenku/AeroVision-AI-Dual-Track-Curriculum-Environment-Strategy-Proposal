import { useState, type FC } from 'react';
import { ShieldAlert, AlertTriangle, Filter, Search, CheckCircle } from 'lucide-react';
import type { ViolationEvent } from '../types';

interface ViolationsTableProps {
  violations: ViolationEvent[];
}

export const ViolationsTable: FC<ViolationsTableProps> = ({ violations }) => {
  const [severityFilter, setSeverityFilter] = useState<'ALL' | 'CRITICAL' | 'WARNING'>('ALL');
  const [searchQuery, setSearchQuery] = useState('');

  const filteredViolations = violations.filter((v) => {
    const matchesSeverity = severityFilter === 'ALL' || v.severity === severityFilter;
    const matchesSearch =
      v.class_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      v.rule_type.toLowerCase().includes(searchQuery.toLowerCase()) ||
      v.details.toLowerCase().includes(searchQuery.toLowerCase()) ||
      String(v.track_id).includes(searchQuery);
    return matchesSeverity && matchesSearch;
  });

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl shadow-xl overflow-hidden flex flex-col">
      {/* Header and Controls */}
      <div className="p-5 border-b border-slate-800 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 bg-slate-950/60">
        <div>
          <h3 className="text-base font-bold text-slate-100 flex items-center space-x-2">
            <ShieldAlert className="w-5 h-5 text-rose-400" />
            <span>Safety &amp; Compliance Audit Log</span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Real-time algorithmic flags: speed limit overruns, restricted geofence penetrations, and loitering.
          </p>
        </div>

        {/* Filter Pills */}
        <div className="flex items-center space-x-2 self-stretch sm:self-auto">
          <div className="relative flex-1 sm:w-48">
            <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Filter by class, ID..."
              className="w-full bg-slate-900 border border-slate-700 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-sky-500"
            />
          </div>

          <div className="flex rounded-lg bg-slate-900 p-0.5 border border-slate-800 text-xs">
            {(['ALL', 'CRITICAL', 'WARNING'] as const).map((filter) => (
              <button
                key={filter}
                onClick={() => setSeverityFilter(filter)}
                className={`px-2.5 py-1 rounded-md font-medium transition ${
                  severityFilter === filter
                    ? 'bg-sky-500 text-white shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {filter}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Table Body */}
      <div className="overflow-x-auto">
        {filteredViolations.length > 0 ? (
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-950/80 border-b border-slate-800 text-slate-400 uppercase font-semibold text-[10px] tracking-wider">
                <th className="py-3 px-4">Time</th>
                <th className="py-3 px-4">Track ID</th>
                <th className="py-3 px-4">Entity</th>
                <th className="py-3 px-4">Rule Violation</th>
                <th className="py-3 px-4">Severity</th>
                <th className="py-3 px-4">Coordinates (X, Y)</th>
                <th className="py-3 px-4">Audit Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {filteredViolations.map((v) => (
                <tr key={v.event_id} className="hover:bg-slate-800/40 transition">
                  <td className="py-3 px-4 text-slate-300 font-medium">
                    {v.timestamp_sec.toFixed(2)}s
                  </td>
                  <td className="py-3 px-4">
                    <span className="px-2 py-0.5 rounded bg-slate-800 text-sky-300 border border-slate-700 text-[11px]">
                      #{v.track_id}
                    </span>
                  </td>
                  <td className="py-3 px-4 capitalize font-sans text-slate-200 font-medium">
                    {v.class_name}
                  </td>
                  <td className="py-3 px-4 font-sans text-slate-300 font-medium">
                    {v.rule_type.replace(/_/g, ' ')}
                  </td>
                  <td className="py-3 px-4">
                    <span
                      className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold ${
                        v.severity === 'CRITICAL'
                          ? 'bg-rose-500/10 text-rose-400 border border-rose-500/30'
                          : 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
                      }`}
                    >
                      {v.severity === 'CRITICAL' ? (
                        <AlertTriangle className="w-3 h-3 mr-1" />
                      ) : (
                        <Filter className="w-3 h-3 mr-1" />
                      )}
                      {v.severity}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-slate-400 text-[11px]">
                    ({v.location[0].toFixed(0)}, {v.location[1].toFixed(0)})
                  </td>
                  <td className="py-3 px-4 font-sans text-slate-300 text-xs">
                    {v.details}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="p-8 text-center text-slate-400 flex flex-col items-center justify-center">
            <CheckCircle className="w-8 h-8 text-emerald-400 mb-2" />
            <p className="text-sm font-semibold text-slate-200">No Violations Found</p>
            <p className="text-xs text-slate-500 mt-0.5">
              {violations.length === 0
                ? 'All detected trajectories adhered to security limits.'
                : 'No incidents match current filter conditions.'}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
