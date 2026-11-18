import { useState, type FC } from 'react';
import {
  Bot,
  Zap,
  ShieldAlert,
  Send,
  Loader2,
  CheckCircle2,
  Terminal,
  Activity,
  Layers,
  Wrench
} from 'lucide-react';
import { api } from '../services/api';
import type { AutonomousResponseResult, AgentChatResponse } from '../types';

interface AgentCommanderConsoleProps {
  jobId: string | null;
}

export const AgentCommanderConsole: FC<AgentCommanderConsoleProps> = ({ jobId }) => {
  const [isRunningGraph, setIsRunningGraph] = useState(false);
  const [agentResult, setAgentResult] = useState<AutonomousResponseResult | null>(null);
  const [chatMessage, setChatMessage] = useState('');
  const [isSendingChat, setIsSendingChat] = useState(false);
  const [chatHistory, setChatHistory] = useState<
    Array<{ sender: 'OPERATOR' | 'AGENT'; text: string; tools?: string[] }>
  >([]);

  const handleTriggerResponse = async () => {
    if (!jobId) return;
    setIsRunningGraph(true);
    try {
      const res = await api.triggerIncidentResponse(jobId);
      setAgentResult(res);
    } catch (err) {
      console.error('Failed to trigger autonomous incident response:', err);
    } finally {
      setIsRunningGraph(false);
    }
  };

  const handleSendChat = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatMessage.trim()) return;

    const userText = chatMessage.trim();
    setChatMessage('');
    setChatHistory((prev) => [...prev, { sender: 'OPERATOR', text: userText }]);
    setIsSendingChat(true);

    try {
      const res: AgentChatResponse = await api.chatWithAgent(userText, jobId || undefined);
      setChatHistory((prev) => [
        ...prev,
        { sender: 'AGENT', text: res.reply, tools: res.tools_invoked }
      ]);
    } catch (err) {
      console.error('Operator chat failed:', err);
      setChatHistory((prev) => [
        ...prev,
        { sender: 'AGENT', text: 'Error executing agent reasoning loop. Please verify telemetry status.' }
      ]);
    } finally {
      setIsSendingChat(false);
    }
  };

  const priorityColor = (priority: string) => {
    switch (priority) {
      case 'CRITICAL':
        return 'bg-rose-500/20 text-rose-300 border-rose-500/30';
      case 'HIGH':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/30';
      case 'MEDIUM':
        return 'bg-blue-500/20 text-blue-300 border-blue-500/30';
      default:
        return 'bg-slate-700 text-slate-300 border-slate-600';
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden flex flex-col space-y-6">
      {/* Background Accent */}
      <div className="absolute top-0 right-0 -mr-20 -mt-20 w-72 h-72 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
        <div className="flex items-center space-x-3.5">
          <div className="p-2.5 bg-emerald-500/10 border border-emerald-500/20 rounded-xl text-emerald-400">
            <Bot className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-slate-100 flex items-center space-x-2">
              <span>Autonomous Incident Commander</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-900/60 text-emerald-300 border border-emerald-700/50 font-mono">
                LangGraph Multi-Agent
              </span>
            </h3>
            <p className="text-xs text-slate-400">
              StateGraph orchestration: Supervisor pattern, dynamic tool calling &amp; self-reflection
            </p>
          </div>
        </div>

        <button
          onClick={handleTriggerResponse}
          disabled={!jobId || isRunningGraph}
          className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-semibold rounded-xl shadow-lg shadow-emerald-900/30 transition-all flex items-center space-x-2 shrink-0"
        >
          {isRunningGraph ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Zap className="w-4 h-4 text-emerald-200" />
          )}
          <span>{isRunningGraph ? 'Running Multi-Agent Graph...' : 'Dispatch Incident Response'}</span>
        </button>
      </div>

      {/* Architecture Specialist Badges */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5 text-xs">
        <div className="p-3 bg-slate-950/70 border border-slate-800 rounded-xl flex items-center space-x-2.5">
          <Layers className="w-4 h-4 text-emerald-400" />
          <div>
            <div className="font-semibold text-slate-200">Incident Commander</div>
            <div className="text-[10px] text-slate-400">Supervisor Dispatcher</div>
          </div>
        </div>

        <div className="p-3 bg-slate-950/70 border border-slate-800 rounded-xl flex items-center space-x-2.5">
          <Activity className="w-4 h-4 text-sky-400" />
          <div>
            <div className="font-semibold text-slate-200">Kinematics Unit</div>
            <div className="text-[10px] text-slate-400">Velocity &amp; Trajectory</div>
          </div>
        </div>

        <div className="p-3 bg-slate-950/70 border border-slate-800 rounded-xl flex items-center space-x-2.5">
          <ShieldAlert className="w-4 h-4 text-purple-400" />
          <div>
            <div className="font-semibold text-slate-200">Compliance Officer</div>
            <div className="text-[10px] text-slate-400">Qdrant RAG Grounding</div>
          </div>
        </div>

        <div className="p-3 bg-slate-950/70 border border-slate-800 rounded-xl flex items-center space-x-2.5">
          <CheckCircle2 className="w-4 h-4 text-amber-400" />
          <div>
            <div className="font-semibold text-slate-200">Reflection Auditor</div>
            <div className="text-[10px] text-slate-400">Zero-Hallucination Gate</div>
          </div>
        </div>
      </div>

      {/* Agent Response Plan Display */}
      {agentResult && (
        <div className="bg-slate-950/90 border border-emerald-500/30 rounded-xl p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
            <div className="flex items-center space-x-3">
              <span className="text-xs font-bold text-slate-200 uppercase tracking-wide">
                Incident Action Plan (IAP)
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded font-mono bg-rose-500/10 text-rose-400 border border-rose-500/30">
                Threat: {agentResult.action_plan.threat_level}
              </span>
            </div>
            <div className="flex items-center space-x-3 text-xs font-mono">
              <span className="text-emerald-400 font-semibold">
                Containment: {agentResult.action_plan.containment_radius_m}m
              </span>
              <span className="text-slate-400">
                Audited: {agentResult.action_plan.grounding_verdict} (Score: {agentResult.action_plan.grounding_score.toFixed(2)})
              </span>
            </div>
          </div>

          <p className="text-xs text-slate-300 leading-relaxed font-sans">
            {agentResult.action_plan.kinematics_summary}
          </p>

          {/* Tactical Directives List */}
          <div className="space-y-2">
            <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
              Autonomous Containment Directives ({agentResult.action_plan.tactical_directives.length} Steps)
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
              {agentResult.action_plan.tactical_directives.map((dir) => (
                <div
                  key={dir.step_number}
                  className="bg-slate-900 border border-slate-800 rounded-lg p-3 text-xs space-y-1.5 flex flex-col justify-between"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-emerald-400 font-bold">
                      Step #{dir.step_number}
                    </span>
                    <span className={`px-2 py-0.5 rounded text-[10px] font-mono border ${priorityColor(dir.priority)}`}>
                      {dir.priority}
                    </span>
                  </div>
                  <div className="text-slate-200 leading-snug">{dir.action}</div>
                  <div className="text-[10px] font-mono text-slate-400 pt-1">
                    Target Unit: <span className="text-slate-300 font-semibold">{dir.target_unit}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Execution Trace Drawer */}
          <div className="border-t border-slate-800/80 pt-3">
            <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center space-x-1.5">
              <Terminal className="w-3.5 h-3.5 text-slate-400" />
              <span>Multi-Agent StateGraph Execution Trace ({agentResult.execution_trace.length} Transitions)</span>
            </div>
            <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800/80 font-mono text-[11px] space-y-1 text-slate-400 max-h-36 overflow-y-auto">
              {agentResult.execution_trace.map((step) => (
                <div key={step.step_index} className="flex items-start space-x-2">
                  <span className="text-emerald-400 shrink-0">[{step.step_index}]</span>
                  <span className="text-slate-300 font-semibold shrink-0">{step.node_name}:</span>
                  <span className="text-slate-400">{step.action_summary}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Interactive Operator Dialogue Console */}
      <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 space-y-3">
        <div className="flex items-center justify-between text-xs font-semibold text-slate-300 border-b border-slate-800/80 pb-2">
          <span className="flex items-center space-x-1.5">
            <Terminal className="w-3.5 h-3.5 text-emerald-400" />
            <span>Interactive Flight Operator Console (Tool-Enabled Dialogue)</span>
          </span>
          <span className="text-[11px] text-slate-500 font-mono">
            Tools: inspect_telemetry &bull; query_statutes &bull; calculate_radius
          </span>
        </div>

        {/* Chat Stream */}
        <div className="space-y-2.5 max-h-48 overflow-y-auto pr-1">
          {chatHistory.length === 0 ? (
            <div className="text-center py-4 text-slate-500 text-xs">
              Operator console online. Ask about speeds, FAA/OSHA rules, or containment radii.
            </div>
          ) : (
            chatHistory.map((msg, i) => (
              <div
                key={i}
                className={`p-2.5 rounded-lg text-xs leading-relaxed ${
                  msg.sender === 'OPERATOR'
                    ? 'bg-slate-900 border border-slate-700/60 text-slate-200 ml-8'
                    : 'bg-emerald-950/30 border border-emerald-500/20 text-emerald-100 mr-8'
                }`}
              >
                <div className="text-[10px] font-mono text-slate-400 mb-0.5">
                  {msg.sender === 'OPERATOR' ? 'Operator' : 'Incident Commander Agent'}
                </div>
                <div>{msg.text}</div>
                {msg.tools && msg.tools.length > 0 && (
                  <div className="mt-1.5 flex flex-wrap gap-1 text-[10px] font-mono text-emerald-300">
                    <Wrench className="w-3 h-3 self-center" />
                    <span>Tools Invoked:</span>
                    {msg.tools.map((t, idx) => (
                      <span key={idx} className="px-1.5 py-0.2 bg-emerald-900/60 rounded border border-emerald-700/40">
                        {t}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))
          )}
        </div>

        {/* Input Form */}
        <form onSubmit={handleSendChat} className="flex gap-2">
          <input
            type="text"
            value={chatMessage}
            onChange={(e) => setChatMessage(e.target.value)}
            placeholder="Type operator inquiry (e.g. 'What is the peak speed and governing regulation?')..."
            className="flex-1 bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500 transition-colors"
          />
          <button
            type="submit"
            disabled={isSendingChat || !chatMessage.trim()}
            className="px-3.5 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-semibold rounded-xl transition-colors flex items-center space-x-1"
          >
            {isSendingChat ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
            <span>Send</span>
          </button>
        </form>
      </div>
    </div>
  );
};
