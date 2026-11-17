"""
=============================================================================
AeroVision AI - Computer Vision Engine: Video Reader & Frame Streamer
=============================================================================

Memory-Safe Sequential Frame Decoding with Constant O(1) Memory Footprint.
Wraps OpenCV cv2.VideoCapture with metadata extraction, corrupt frame resilience,
stride sampling, and aspect-ratio letterboxing.
"""

import os
from typing import Generator, Tuple, Optional, Dict, Any
import cv2
import numpy as np

from backend.app.schemas.cv import FrameMetadata

class VideoReader:
    """
    Sequential, generator-based video reader for high-resolution drone footage.
    Guarantees O(1) space complexity by streaming frames one-by-one without
    loading the entire video buffer into system RAM.
    """
    def __init__(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at path: {video_path}")
            
        self.video_path = video_path
        self._cap = cv2.VideoCapture(video_path)
        
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video file (corrupt or unsupported codec): {video_path}")
            
        # Extract intrinsic video metadata
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0.0 or np.isnan(self.fps):
            self.fps = 30.0  # Safe default fallback for variable framerate files
            
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0.0
        
        # Codec FourCC
        fourcc_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        self.fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    @property
    def metadata(self) -> Dict[str, Any]:
        """Returns structured dictionary of video metadata."""
        return {
            "video_path": self.video_path,
            "resolution": [self.width, self.height],
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "duration_sec": round(self.duration_sec, 2),
            "fourcc": self.fourcc
        }

    def stream_frames(
        self,
        stride: int = 1,
        max_frames: Optional[int] = None
    ) -> Generator[Tuple[FrameMetadata, np.ndarray], None, None]:
        """
        Memory-safe generator streaming frames sequentially.
        
        Args:
            stride: Process every k-th frame (e.g. stride=2 cuts inference compute by 50%).
            max_frames: Optional cap on total frames to yield.
            
        Yields:
            Tuple of (FrameMetadata, frame_bgr_numpy_array)
        """
        # Ensure pointer is at frame 0
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_idx = 0
        yielded_count = 0

        while self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                # End of stream or corrupt frame packet
                break
                
            if frame_idx % stride == 0:
                timestamp = frame_idx / self.fps
                meta = FrameMetadata(
                    frame_id=frame_idx,
                    timestamp_sec=timestamp,
                    width=frame.shape[1],
                    height=frame.shape[0]
                )
                yield meta, frame
                yielded_count += 1
                
                if max_frames is not None and yielded_count >= max_frames:
                    break
                    
            frame_idx += 1

    def release(self):
        """Releases the underlying OpenCV video capture handle."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

# =============================================================================
# SPATIAL PREPROCESSING: ASPECT-RATIO LETTERBOXING
# =============================================================================
def letterbox(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    fill_color: Tuple[int, int, int] = (114, 114, 114),
    auto_stride: int = 32
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resizes and pads an arbitrary rectangular image to square target size
    while preserving original aspect ratio (standard YOLO preprocessing).
    
    Returns:
        padded_image: Shape (target_h, target_w, 3)
        ratio: Scale ratio applied
        (dw, dh): Symmetric padding offsets applied to (left/right, top/bottom)
    """
    h_orig, w_orig = image.shape[:2]
    target_w, target_h = target_size

    # Scale ratio (new / old)
    r = min(target_w / w_orig, target_h / h_orig)

    # Compute unpadded resized dimensions
    new_w = int(round(w_orig * r))
    new_h = int(round(h_orig * r))

    # Compute padding required
    dw = (target_w - new_w) / 2.0
    dh = (target_h - new_h) / 2.0

    # Resize image if dimensions changed
    if (w_orig, h_orig) != (new_w, new_h):
        interp = cv2.INTER_AREA if r < 1.0 else cv2.INTER_LINEAR
        image = cv2.resize(image, (new_w, new_h), interpolation=interp)

    # Calculate integer padding borders
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    # Apply constant border
    padded = cv2.copyMakeBorder(
        image, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=fill_color
    )

    return padded, r, (dw, dh)

# =============================================================================
# VIDEO WRITER: ENCODING ANNOTATED OUTPUTS
# =============================================================================
class VideoWriterWrapper:
    """Encodes processed frames into an MP4 video container."""
    def __init__(self, output_path: str, fps: float, width: int, height: int):
        self.output_path = output_path
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # 'mp4v' codec works reliably on Windows across standard OpenCV distributions
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to initialize VideoWriter at: {output_path}")

    def write(self, frame: np.ndarray):
        self._writer.write(frame)

    def release(self):
        if self._writer is not None:
            self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
