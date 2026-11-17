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
