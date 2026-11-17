import { useRef, useState, useEffect, type FC, type ChangeEvent } from 'react';
import { Play, Pause, RotateCcw, Maximize, Volume2, VolumeX, Eye, FastForward } from 'lucide-react';
import { api } from '../services/api';
import type { JobResponse } from '../types';

interface VideoPlayerProps {
  job: JobResponse | null;
}

export const VideoPlayer: FC<VideoPlayerProps> = ({ job }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [isMuted, setIsMuted] = useState(true);
  const [hasError, setHasError] = useState(false);

  const videoUrl = job?.annotated_video_path ? api.resolveMediaUrl(job.annotated_video_path) : '';

  useEffect(() => {
    // Reset video state when job changes
    setIsPlaying(false);
    setCurrentTime(0);
    setHasError(false);
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
      videoRef.current.playbackRate = playbackRate;
    }
  }, [job?.job_id]);

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
      setIsPlaying(false);
    } else {
      videoRef.current.play().then(() => setIsPlaying(true)).catch((e) => console.warn('Autoplay blocked:', e));
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  const handleSeek = (e: ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  const cycleSpeed = () => {
    const speeds = [0.5, 1.0, 1.5, 2.0];
    const nextIdx = (speeds.indexOf(playbackRate) + 1) % speeds.length;
    const newSpeed = speeds[nextIdx];
    setPlaybackRate(newSpeed);
    if (videoRef.current) {
      videoRef.current.playbackRate = newSpeed;
    }
  };

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const toggleFullscreen = () => {
    if (videoRef.current) {
      if (videoRef.current.requestFullscreen) {
        videoRef.current.requestFullscreen();
      }
    }
  };

  const formatSeconds = (sec: number) => {
    const mins = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${mins}:${s < 10 ? '0' : ''}${s}`;
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl flex flex-col">
      {/* Header Bar */}
      <div className="px-5 py-3.5 bg-slate-950 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Eye className="w-4 h-4 text-sky-400" />
          <span className="font-semibold text-xs text-slate-200 tracking-wide uppercase">
            Surveillance Telemetry Feed
          </span>
        </div>
        <div className="flex items-center space-x-2">
          {job?.status === 'COMPLETED' ? (
            <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
              <span className="w-1.5 h-1.5 mr-1.5 bg-emerald-400 rounded-full animate-pulse" />
              ANNOTATED (YOLO11 + BYTETRACK)
            </span>
          ) : (
            <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-slate-800 text-slate-400 border border-slate-700">
              AWAITING INFERENCE
            </span>
          )}
        </div>
      </div>

      {/* Video Viewport */}
      <div className="relative aspect-video bg-black flex items-center justify-center overflow-hidden group">
        {videoUrl && job?.status === 'COMPLETED' ? (
          <>
            <video
              ref={videoRef}
              src={videoUrl}
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
              onEnded={() => setIsPlaying(false)}
              onError={() => setHasError(true)}
              muted={isMuted}
              playsInline
              className="w-full h-full object-contain"
              onClick={togglePlay}
            />

            {/* In-Video HUD Telemetry Badge */}
            <div className="absolute top-4 left-4 bg-slate-950/80 backdrop-blur border border-slate-700/60 rounded-lg px-3 py-1.5 text-[11px] font-mono text-slate-200 space-y-0.5 pointer-events-none shadow-lg">
              <div className="text-sky-400 font-bold flex items-center space-x-1">
                <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse" />
                <span>FLIGHT REC: {job.job_id.slice(0, 10)}</span>
              </div>
              <div className="text-slate-400">
                FRAME: {Math.floor((currentTime / (duration || 1)) * (job.total_frames || 120))} / {job.total_frames || 120}
              </div>
            </div>

            {hasError && (
              <div className="absolute inset-0 bg-slate-950/90 flex flex-col items-center justify-center p-6 text-center">
                <p className="text-rose-400 font-medium text-sm">Failed to stream video feed.</p>
                <p className="text-slate-400 text-xs mt-1">Ensure backend server is running and media file exists.</p>
              </div>
            )}
          </>
        ) : (
          <div className="flex flex-col items-center justify-center p-8 text-center text-slate-500">
            <Eye className="w-12 h-12 text-slate-700 mb-3" />
            <p className="text-sm font-medium text-slate-400">No Annotated Video Loaded</p>
            <p className="text-xs text-slate-600 mt-1">
              {job?.status === 'PROCESSING'
                ? 'Processing drone video feed... Player will activate automatically.'
                : 'Upload a flight video or select a completed mission to watch bounding boxes and trails.'}
            </p>
          </div>
        )}
      </div>

      {/* Control Bar */}
      {videoUrl && job?.status === 'COMPLETED' && (
        <div className="p-3 bg-slate-950/90 border-t border-slate-800 space-y-2">
          {/* Progress Slider */}
          <div className="flex items-center space-x-3">
            <span className="text-[11px] font-mono text-slate-400 w-10 text-right">
              {formatSeconds(currentTime)}
            </span>
            <input
              type="range"
              min="0"
              max={duration || 100}
              step="0.05"
              value={currentTime}
              onChange={handleSeek}
              className="flex-1 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-400"
            />
            <span className="text-[11px] font-mono text-slate-400 w-10">
              {formatSeconds(duration)}
            </span>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center justify-between pt-1">
            <div className="flex items-center space-x-2">
              <button
                onClick={togglePlay}
                className="p-1.5 rounded-lg bg-sky-500/20 hover:bg-sky-500/30 text-sky-400 transition"
              >
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              </button>
              <button
                onClick={() => {
                  if (videoRef.current) {
                    videoRef.current.currentTime = 0;
                    setCurrentTime(0);
                  }
                }}
                className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition"
                title="Restart"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
              <button
                onClick={toggleMute}
                className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition"
                title={isMuted ? 'Unmute' : 'Mute'}
              >
                {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
              </button>
            </div>

            <div className="flex items-center space-x-2">
              <button
                onClick={cycleSpeed}
                className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-mono flex items-center space-x-1"
                title="Change playback speed"
              >
                <FastForward className="w-3 h-3 text-sky-400" />
                <span>{playbackRate}x</span>
              </button>
              <button
                onClick={toggleFullscreen}
                className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition"
                title="Fullscreen"
              >
                <Maximize className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
