/**
 * AeroVision AI - Frontend Data Types & TypeScript Contracts
 * Synchronized with FastAPI Backend Schemas and Models
 */

export interface SystemMetrics {
  cpu_usage_percent: number;
  ram_used_gb: number;
  ram_total_gb: number;
  ram_free_gb: number;
}

export interface SystemHealth {
  status: 'HEALTHY' | 'DEGRADED';
  database_connected: boolean;
  model_weights_available: boolean;
  system_metrics: SystemMetrics;
}

export type JobStatus = 'QUEUED' | 'PROCESSING' | 'COMPLETED' | 'FAILED';

export interface JobResponse {
  job_id: string;
  filename: string;
  status: JobStatus;
  progress_percent: number;
  total_frames: number;
  processed_frames: number;
  annotated_video_path?: string | null;
  error_message?: string | null;
  created_at: string;
}

export interface VehicleClassStat {
  count: number;
  mean_speed_kmh: number;
  max_speed_kmh: number;
  total_distance_meters: number;
}

export type ViolationSeverity = 'WARNING' | 'CRITICAL';
export type ViolationRuleType = 'SPEED_LIMIT' | 'RESTRICTED_ZONE_INTRUSION' | 'LOITERING_DWELL_TIME';

export interface ViolationEvent {
  event_id: string;
  rule_type: ViolationRuleType | string;
  track_id: number;
  class_name: string;
  timestamp_sec: number;
  severity: ViolationSeverity;
  details: string;
  location: [number, number];
}

export interface SafetyAudit {
  total_violations: number;
  compliance_status: 'COMPLIANT' | 'AUDIT_FLAGGED';
  violations_log: ViolationEvent[];
}

export interface MissionAnalytics {
  mission_id: string;
  job_id: string;
  duration_sec: number;
  total_unique_objects: number;
  counts_by_class: Record<string, number>;
  class_statistics: Record<string, VehicleClassStat>;
  line_crossings: Record<string, number>;
  safety_audit: SafetyAudit;
}

export interface IncidentDetail {
  track_id: number;
  class_name: string;
  rule_type: string;
  severity: 'WARNING' | 'CRITICAL';
  narrative: string;
  statutory_citations?: string[];
}

export interface GroundingAuditResult {
  grounding_score: number;
  verdict: 'VERIFIED_GROUNDED' | 'MINOR_DEVIATION' | 'UNRELIABLE_HALLUCINATION';
  entity_score: number;
  speed_score: number;
  incident_score: number;
  hallucinated_classes: string[];
  speed_delta_kmh: number;
}

export interface MissionComplianceReport {
  mission_id: string;
  job_id: string;
  executive_summary: string;
  threat_level: 'LOW' | 'ELEVATED' | 'CRITICAL';
  compliance_status: 'COMPLIANT' | 'AUDIT_FLAGGED';
  total_tracked_objects: number;
  peak_speed_kmh: number;
  incidents: IncidentDetail[];
  operational_recommendations: string[];
  grounding_audit: GroundingAuditResult;
  model_used: string;
  generated_at: string;
}

export interface RegulatoryCitation {
  statute_code: string;
  section_title: string;
  category: string;
  relevance_score: number;
  text: string;
}

export interface AugmentedInfraction {
  track_id: number;
  class_name: string;
  rule_type: string;
  severity: string;
  details: string;
  citations: RegulatoryCitation[];
}

export interface MissionRAGContextResponse {
  job_id: string;
  total_infractions: number;
  augmented_infractions: AugmentedInfraction[];
}

export interface RAGSearchResponse {
  query: string;
  total_results: number;
  results: RegulatoryCitation[];
}

export interface TacticalDirective {
  step_number: number;
  action: string;
  target_unit: string;
  priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
}

export interface IncidentActionPlan {
  mission_id: string;
  job_id: string;
  incident_type: string;
  threat_level: 'LOW' | 'ELEVATED' | 'CRITICAL';
  kinematics_summary: string;
  statutory_citations: string[];
  containment_radius_m: number;
  tactical_directives: TacticalDirective[];
  grounding_score: number;
  grounding_verdict: 'VERIFIED_GROUNDED' | 'MINOR_DEVIATION' | 'UNRELIABLE_HALLUCINATION';
  generated_at: string;
}

export interface AgentStepTrace {
  step_index: number;
  node_name: string;
  action_summary: string;
  details?: Record<string, unknown>;
}

export interface AutonomousResponseResult {
  job_id: string;
  status: 'SUCCESS' | 'FAILED';
  action_plan: IncidentActionPlan;
  execution_trace: AgentStepTrace[];
  iterations_completed: number;
}

export interface AgentChatResponse {
  reply: string;
  tools_invoked: string[];
  data_grounding: Record<string, unknown>;
}
