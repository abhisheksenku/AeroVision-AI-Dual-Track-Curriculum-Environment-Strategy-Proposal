/**
 * AeroVision AI - API Client Service
 * Axios wrapper for FastAPI REST endpoints
 */

import axios from 'axios';
import type {
  SystemHealth,
  JobResponse,
  MissionAnalytics,
  MissionComplianceReport,
  MissionRAGContextResponse,
  RAGSearchResponse
} from '../types';

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const client = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  timeout: 30000,
});

export const api = {
  /**
   * Health status of backend, database, and inference hardware
   */
  async getHealth(): Promise<SystemHealth> {
    const res = await client.get<SystemHealth>('/health');
    return res.data;
  },

  /**
   * Upload drone video and initiate background CV pipeline
   */
  async uploadVideo(
    file: File,
    onProgress?: (percent: number) => void
  ): Promise<{ job_id: string; filename: string; status: string; message: string }> {
    const formData = new FormData();
    formData.append('video', file);

    const res = await client.post('/jobs/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total && onProgress) {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(percent);
        }
      },
    });
    return res.data;
  },

  /**
   * Retrieve list of all processed or queued jobs
   */
  async getJobs(): Promise<JobResponse[]> {
    const res = await client.get<JobResponse[]>('/jobs');
    return res.data;
  },

  /**
   * Poll status of an individual job
   */
  async getJob(jobId: string): Promise<JobResponse> {
    const res = await client.get<JobResponse>(`/jobs/${jobId}`);
    return res.data;
  },

  /**
   * Retrieve structured mission analytics for a completed job
   */
  async getAnalytics(jobId: string): Promise<MissionAnalytics> {
    const res = await client.get<MissionAnalytics>(`/analytics/${jobId}`);
    return res.data;
  },

  /**
   * Generate Gemini AI mission compliance report with grounding verification
   */
  async generateAiReport(jobId: string, modelName = 'gemini-2.5-flash'): Promise<MissionComplianceReport> {
    const res = await client.post<MissionComplianceReport>(`/reports/${jobId}/generate`, {
      model_name: modelName,
      temperature: 0.2,
    });
    return res.data;
  },

  /**
   * Retrieve cached Gemini AI mission report
   */
  async getAiReport(jobId: string): Promise<MissionComplianceReport> {
    const res = await client.get<MissionComplianceReport>(`/reports/${jobId}`);
    return res.data;
  },

  /**
   * Retrieve RAG regulatory context for a mission's infractions
   */
  async getRAGContext(jobId: string): Promise<MissionRAGContextResponse> {
    const res = await client.get<MissionRAGContextResponse>(`/rag/context/${jobId}`);
    return res.data;
  },

  /**
   * Perform direct semantic vector query against Qdrant regulatory database
   */
  async searchRegulations(query: string, topK = 3, category?: string): Promise<RAGSearchResponse> {
    const res = await client.post<RAGSearchResponse>('/rag/search', {
      query,
      top_k: topK,
      category,
    });
    return res.data;
  },

  /**
   * Resolve backend local file path to accessible HTTP streaming URL
   */
  resolveMediaUrl(rawPath?: string | null): string {
    if (!rawPath) return '';
    // Normalize path slashes
    const normalized = rawPath.replace(/\\/g, '/');
    const filename = normalized.split('/').pop();
    if (normalized.includes('processed')) {
      return `${API_BASE_URL}/media/processed/${filename}`;
    }
    if (normalized.includes('uploads')) {
      return `${API_BASE_URL}/media/uploads/${filename}`;
    }
    return `${API_BASE_URL}/${normalized}`;
  },
};
