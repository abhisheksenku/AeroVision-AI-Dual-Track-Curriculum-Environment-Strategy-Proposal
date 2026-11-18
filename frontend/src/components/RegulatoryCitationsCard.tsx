import { useState, useEffect, type FC } from 'react';
import { BookOpen, Scale, Search, ShieldAlert, CheckCircle, Loader2 } from 'lucide-react';
import { api } from '../services/api';
import type { MissionRAGContextResponse, RegulatoryCitation } from '../types';

interface RegulatoryCitationsCardProps {
  jobId: string | null;
}

export const RegulatoryCitationsCard: FC<RegulatoryCitationsCardProps> = ({ jobId }) => {
  const [ragContext, setRagContext] = useState<MissionRAGContextResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<RegulatoryCitation[] | null>(null);

  useEffect(() => {
    if (!jobId) {
      setRagContext(null);
      return;
    }

    let isMounted = true;
    setLoading(true);

    api
      .getRAGContext(jobId)
      .then((data) => {
        if (isMounted) setRagContext(data);
      })
      .catch((err) => {
        console.warn('Could not load RAG context for job:', err);
        if (isMounted) setRagContext(null);
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [jobId]);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    try {
      const res = await api.searchRegulations(searchQuery, 3);
      setSearchResults(res.results);
    } catch (err) {
      console.error('Vector search failed:', err);
    } finally {
      setIsSearching(false);
    }
  };

  const getCategoryBadgeClass = (category: string) => {
    switch (category) {
      case 'FAA':
        return 'bg-blue-500/20 text-blue-300 border-blue-500/30';
      case 'OSHA':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/30';
      case 'PERIMETER_SECURITY':
        return 'bg-purple-500/20 text-purple-300 border-purple-500/30';
      default:
        return 'bg-slate-700 text-slate-300 border-slate-600';
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden flex flex-col space-y-5">
      {/* Background Accent */}
      <div className="absolute top-0 right-0 -mr-20 -mt-20 w-64 h-64 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800/80 pb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-cyan-500/10 border border-cyan-500/20 rounded-xl text-cyan-400">
            <Scale className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-slate-100 flex items-center space-x-2">
              <span>Statutory Legal Retrieval (RAG)</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-900/60 text-cyan-300 border border-cyan-700/50 font-mono">
                Qdrant DB
              </span>
            </h3>
            <p className="text-xs text-slate-400">
              Autonomous legal grounding via FAA Part 107, OSHA 1910, & NIST perimeter regulations
            </p>
          </div>
        </div>

        {ragContext && (
          <div className="flex items-center space-x-2 text-xs text-slate-400 font-mono">
            <CheckCircle className="w-4 h-4 text-emerald-400" />
            <span>{ragContext.total_infractions} Infractions Grounded</span>
          </div>
        )}
      </div>

      {/* Interactive Semantic Search Bar */}
      <form onSubmit={handleSearch} className="flex items-center gap-2">
        <div className="relative flex-1">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search regulations (e.g. 'speed limit in industrial runway' or 'line of sight')..."
            className="w-full bg-slate-950/80 border border-slate-700/80 rounded-xl pl-9 pr-4 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition-colors"
          />
        </div>
        <button
          type="submit"
          disabled={isSearching || !searchQuery.trim()}
          className="px-3.5 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white text-xs font-medium rounded-xl transition-colors flex items-center space-x-1.5"
        >
          {isSearching ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <BookOpen className="w-3.5 h-3.5" />}
          <span>Query</span>
        </button>
        {searchResults && (
          <button
            type="button"
            onClick={() => setSearchResults(null)}
            className="px-2.5 py-2 bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-200 text-xs rounded-xl"
          >
            Clear
          </button>
        )}
      </form>

      {/* Search Results Drawer */}
      {searchResults && (
        <div className="bg-slate-950/90 border border-cyan-500/30 rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between text-xs font-semibold text-cyan-300">
            <span>Vector Search Results ({searchResults.length} Matches)</span>
          </div>
          <div className="space-y-2.5">
            {searchResults.map((citation, idx) => (
              <div key={idx} className="bg-slate-900 border border-slate-800 rounded-lg p-3 text-xs space-y-1.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-mono border ${getCategoryBadgeClass(citation.category)}`}>
                      {citation.category}
                    </span>
                    <span className="font-semibold text-slate-200">{citation.statute_code}</span>
                  </div>
                  <span className="text-[11px] font-mono text-cyan-400 font-medium">
                    {(citation.relevance_score * 100).toFixed(1)}% match
                  </span>
                </div>
                <div className="text-slate-300 font-medium">{citation.section_title}</div>
                <p className="text-[11px] text-slate-400 leading-relaxed font-sans">{citation.text}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Mission Infractions Grounded Citations */}
      {loading ? (
        <div className="flex items-center justify-center py-8 text-slate-400 text-xs space-x-2">
          <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
          <span>Retrieving statutory citations from Qdrant vector database...</span>
        </div>
      ) : ragContext && ragContext.augmented_infractions.length > 0 ? (
        <div className="space-y-3">
          <div className="text-xs font-medium text-slate-400 uppercase tracking-wider">
            Audited Mission Infractions & Governing Codes
          </div>
          <div className="space-y-3">
            {ragContext.augmented_infractions.map((infraction, i) => (
              <div key={i} className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4 space-y-2.5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center space-x-2">
                    <ShieldAlert className="w-4 h-4 text-rose-400" />
                    <span className="text-xs font-semibold text-slate-200">
                      Track #{infraction.track_id} ({infraction.class_name})
                    </span>
                    <span className="text-[10px] px-2 py-0.5 rounded font-mono bg-rose-500/10 text-rose-400 border border-rose-500/20">
                      {infraction.rule_type}
                    </span>
                  </div>
                  <span className="text-[10px] text-slate-400 italic">
                    {infraction.details}
                  </span>
                </div>

                {/* Statutory Citations */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 pt-1">
                  {infraction.citations.map((cite, cIdx) => (
                    <div
                      key={cIdx}
                      className="bg-slate-900/90 border border-slate-800 rounded-lg p-2.5 text-xs flex flex-col justify-between hover:border-slate-700 transition-colors"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono border ${getCategoryBadgeClass(cite.category)}`}>
                          {cite.category}
                        </span>
                        <span className="text-[11px] font-mono text-cyan-400 font-semibold">
                          {(cite.relevance_score * 100).toFixed(1)}% match
                        </span>
                      </div>
                      <div className="font-semibold text-slate-200 text-xs mb-1">
                        {cite.statute_code} - {cite.section_title}
                      </div>
                      <p className="text-[11px] text-slate-400 line-clamp-3 leading-relaxed">
                        {cite.text}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : jobId ? (
        <div className="text-center py-6 text-slate-500 text-xs">
          <BookOpen className="w-6 h-6 mx-auto mb-2 opacity-40" />
          <span>No statutory infractions recorded for this mission flight. Clean safety audit.</span>
        </div>
      ) : (
        <div className="text-center py-6 text-slate-500 text-xs">
          Select or process a drone video to inspect automated RAG statutory citations.
        </div>
      )}
    </div>
  );
};
