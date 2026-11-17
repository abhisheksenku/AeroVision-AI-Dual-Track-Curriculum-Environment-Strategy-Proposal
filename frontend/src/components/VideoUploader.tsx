import { useState, useRef, type DragEvent, type ChangeEvent, type FC } from 'react';
import { UploadCloud, CheckCircle2, AlertCircle, Loader2, FileVideo } from 'lucide-react';
import { api } from '../services/api';
import type { JobResponse } from '../types';

interface VideoUploaderProps {
  onJobCreated: (job: JobResponse) => void;
  activeJob: JobResponse | null;
}

export const VideoUploader: FC<VideoUploaderProps> = ({ onJobCreated, activeJob }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = async (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      await processUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      await processUpload(e.target.files[0]);
    }
  };

  const processUpload = async (file: File) => {
    setErrorMessage(null);
    const validExtensions = ['.mp4', '.avi', '.mov', '.mkv'];
    const hasValidExt = validExtensions.some((ext) => file.name.toLowerCase().endsWith(ext));

    if (!hasValidExt) {
      setErrorMessage(`Invalid video format. Allowed formats: ${validExtensions.join(', ')}`);
      return;
    }

    try {
      setIsUploading(true);
      setUploadProgress(0);
      const res = await api.uploadVideo(file, (percent) => {
        setUploadProgress(percent);
      });

      // Poll initial state
      const initialJob = await api.getJob(res.job_id);
      onJobCreated(initialJob);
    } catch (err: unknown) {
      console.error('Upload failed:', err);
      const errorMsg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        'Video upload failed. Check connection with AeroVision backend.';
      setErrorMessage(errorMsg);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const isProcessing = activeJob?.status === 'PROCESSING' || activeJob?.status === 'QUEUED';

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden">
      {/* Glow decorative effect */}
      <div className="absolute top-0 right-0 -mr-16 -mt-16 w-48 h-48 bg-sky-500/10 rounded-full blur-3xl pointer-events-none" />

      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-slate-100 flex items-center space-x-2">
            <FileVideo className="w-5 h-5 text-sky-400" />
            <span>Drone Video Ingestion</span>
          </h2>
          <p className="text-xs text-slate-400">
            Submit aerial feeds for real-time YOLO11 detection, ByteTrack kinematics, and rule violation audits.
          </p>
        </div>

        {activeJob && (
          <div className="flex items-center space-x-2">
            <span className="text-xs text-slate-400">Active Job:</span>
            <span className="font-mono text-xs px-2 py-1 bg-slate-950 border border-slate-800 rounded text-sky-300">
              {activeJob.job_id}
            </span>
          </div>
        )}
      </div>

      {/* Drag & Drop Surface */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isUploading && !isProcessing && fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-8 text-center transition cursor-pointer flex flex-col items-center justify-center space-y-3 ${
          isDragging
            ? 'border-sky-400 bg-sky-500/10'
            : isProcessing
            ? 'border-slate-800 bg-slate-950/40 cursor-not-allowed opacity-80'
            : 'border-slate-700/80 hover:border-sky-500/50 bg-slate-950/50 hover:bg-slate-950/80'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/mp4,video/avi,video/quicktime,video/x-matroska"
          onChange={handleFileChange}
          className="hidden"
          disabled={isUploading || isProcessing}
        />

        {isUploading ? (
          <div className="w-full max-w-md space-y-3">
            <Loader2 className="w-8 h-8 text-sky-400 animate-spin mx-auto" />
            <p className="text-sm font-medium text-slate-200">Uploading Flight Video ({uploadProgress}%)</p>
            <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
              <div
                className="bg-gradient-to-r from-sky-500 to-indigo-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          </div>
        ) : isProcessing ? (
          <div className="w-full max-w-md space-y-3 py-2">
            <Loader2 className="w-8 h-8 text-indigo-400 animate-spin mx-auto" />
            <div>
              <p className="text-sm font-semibold text-slate-100">
                {activeJob?.status === 'QUEUED' ? 'Job Queued for Processing...' : 'Neural Pipeline Running...'}
              </p>
              <p className="text-xs text-slate-400 mt-1">
                Processed {activeJob?.processed_frames} / {activeJob?.total_frames || '...'} frames (
                {activeJob?.progress_percent}%)
              </p>
            </div>
            <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
              <div
                className="bg-gradient-to-r from-cyan-400 via-sky-500 to-indigo-500 h-2.5 rounded-full transition-all duration-500"
                style={{ width: `${activeJob?.progress_percent || 5}%` }}
              />
            </div>
          </div>
        ) : (
          <>
            <div className="p-3 bg-slate-800/80 border border-slate-700 rounded-full text-sky-400 shadow-inner">
              <UploadCloud className="w-7 h-7" />
            </div>
            <div>
              <p className="text-sm font-medium text-slate-200">
                Click to browse or drop aerial video files here
              </p>
              <p className="text-xs text-slate-400 mt-1">
                Supports MP4, AVI, MOV (H.264 / AVC codec recommended)
              </p>
            </div>
          </>
        )}
      </div>

      {/* Error Message */}
      {errorMessage && (
        <div className="mt-4 p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 flex items-center space-x-2 text-rose-400 text-xs">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Processing Stepper */}
      {activeJob && (
        <div className="mt-5 grid grid-cols-3 gap-2 text-center text-xs">
          <div
            className={`p-2.5 rounded-lg border transition ${
              activeJob.status === 'QUEUED' || activeJob.status === 'PROCESSING' || activeJob.status === 'COMPLETED'
                ? 'bg-sky-500/10 border-sky-500/30 text-sky-300'
                : 'bg-slate-950 border-slate-800 text-slate-400'
            }`}
          >
            <div className="font-semibold">1. Ingested</div>
            <div className="text-[10px] text-slate-400 truncate">{activeJob.filename}</div>
          </div>

          <div
            className={`p-2.5 rounded-lg border transition ${
              activeJob.status === 'PROCESSING'
                ? 'bg-indigo-500/10 border-indigo-500/30 text-indigo-300 animate-pulse'
                : activeJob.status === 'COMPLETED'
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                : 'bg-slate-950 border-slate-800 text-slate-400'
            }`}
          >
            <div className="font-semibold">2. YOLO11 &amp; ByteTrack</div>
            <div className="text-[10px] text-slate-400">
              {activeJob.status === 'COMPLETED' ? 'Completed' : `${activeJob.progress_percent}%`}
            </div>
          </div>

          <div
            className={`p-2.5 rounded-lg border transition ${
              activeJob.status === 'COMPLETED'
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                : activeJob.status === 'FAILED'
                ? 'bg-rose-500/10 border-rose-500/30 text-rose-300'
                : 'bg-slate-950 border-slate-800 text-slate-400'
            }`}
          >
            <div className="font-semibold flex items-center justify-center space-x-1">
              {activeJob.status === 'COMPLETED' ? (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  <span>3. Analytics Ready</span>
                </>
              ) : activeJob.status === 'FAILED' ? (
                <>
                  <AlertCircle className="w-3.5 h-3.5 text-rose-400" />
                  <span>Failed</span>
                </>
              ) : (
                <span>3. Verification</span>
              )}
            </div>
            <div className="text-[10px] text-slate-400">
              {activeJob.status === 'COMPLETED' ? 'Audit Synced' : 'Awaiting Completion'}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
