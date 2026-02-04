"""
Performance optimization utilities for Glitch Mirror.
Provides frame skipping, caching, and profiling tools.
"""

import cv2
import numpy as np
import time
from functools import lru_cache


class FrameSkipper:
    """
    Manages frame skipping for expensive operations.
    Caches results and only recomputes every N frames.
    """

    def __init__(self):
        self.caches = {}
        self.frame_counts = {}
        self.skip_intervals = {
            'detection': 3,      # Run detection every 3 frames
            'body_detection': 5, # Bodies move slower, check less often
            'effects_heavy': 2,  # Heavy effects every 2 frames
        }

    def set_skip_interval(self, operation, interval):
        """Set how often an operation should run."""
        self.skip_intervals[operation] = max(1, interval)

    def should_run(self, operation, frame_number):
        """Check if operation should run this frame."""
        interval = self.skip_intervals.get(operation, 1)
        return frame_number % interval == 0

    def get_cached(self, operation):
        """Get cached result for operation."""
        return self.caches.get(operation)

    def set_cached(self, operation, result):
        """Cache result for operation."""
        self.caches[operation] = result

    def get_or_run(self, operation, frame_number, func, *args, **kwargs):
        """
        Run function only if needed, otherwise return cached result.
        """
        if self.should_run(operation, frame_number):
            result = func(*args, **kwargs)
            self.set_cached(operation, result)
            return result
        return self.get_cached(operation)


class PerformanceMonitor:
    """
    Tracks frame times and provides FPS statistics.
    """

    def __init__(self, window_size=30):
        self.frame_times = []
        self.window_size = window_size
        self.last_frame_time = time.time()
        self.section_times = {}
        self.current_section_start = None
        self.current_section_name = None

    def frame_start(self):
        """Call at start of frame processing."""
        self.last_frame_time = time.time()
        self.section_times.clear()

    def frame_end(self):
        """Call at end of frame processing."""
        frame_time = time.time() - self.last_frame_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)

    def section_start(self, name):
        """Start timing a section."""
        self.current_section_name = name
        self.current_section_start = time.time()

    def section_end(self):
        """End timing current section."""
        if self.current_section_name and self.current_section_start:
            elapsed = time.time() - self.current_section_start
            self.section_times[self.current_section_name] = elapsed
            self.current_section_name = None
            self.current_section_start = None

    def get_fps(self):
        """Get current FPS estimate."""
        if not self.frame_times:
            return 0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0

    def get_frame_time_ms(self):
        """Get average frame time in milliseconds."""
        if not self.frame_times:
            return 0
        return (sum(self.frame_times) / len(self.frame_times)) * 1000

    def get_slowest_section(self):
        """Get the slowest section this frame."""
        if not self.section_times:
            return None, 0
        slowest = max(self.section_times.items(), key=lambda x: x[1])
        return slowest[0], slowest[1] * 1000


class GrayscaleCache:
    """
    Caches grayscale conversions to avoid redundant cv2.cvtColor calls.
    """

    def __init__(self):
        self._frame_hash = None
        self._grayscale = None
        self._equalized = None
        self._clahe = None
        self._clahe_processor = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _compute_hash(self, frame):
        """Quick hash based on frame corners and center."""
        h, w = frame.shape[:2]
        # Sample a few pixels for quick comparison
        sample = np.array([
            frame[0, 0, 0], frame[0, w-1, 0],
            frame[h-1, 0, 0], frame[h-1, w-1, 0],
            frame[h//2, w//2, 0]
        ])
        return hash(sample.tobytes())

    def get_grayscale(self, frame):
        """Get grayscale version, using cache if frame unchanged."""
        frame_hash = self._compute_hash(frame)
        if frame_hash != self._frame_hash:
            self._frame_hash = frame_hash
            self._grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self._equalized = None
            self._clahe = None
        return self._grayscale

    def get_equalized(self, frame):
        """Get histogram-equalized grayscale."""
        gray = self.get_grayscale(frame)
        if self._equalized is None:
            self._equalized = cv2.equalizeHist(gray)
        return self._equalized

    def get_clahe(self, frame):
        """Get CLAHE-processed grayscale."""
        gray = self.get_grayscale(frame)
        if self._clahe is None:
            self._clahe = self._clahe_processor.apply(gray)
        return self._clahe

    def invalidate(self):
        """Force cache invalidation."""
        self._frame_hash = None
        self._grayscale = None
        self._equalized = None
        self._clahe = None


class ResizeCache:
    """
    Cache for downscaled frames used in detection.
    Detection can run on smaller frames for speed.
    """

    def __init__(self):
        self._cache = {}
        self._frame_hash = None

    def get_downscaled(self, frame, scale):
        """Get downscaled frame, using cache if available."""
        if scale >= 1.0:
            return frame, 1.0

        frame_hash = id(frame)
        cache_key = (frame_hash, scale)

        if cache_key in self._cache:
            return self._cache[cache_key], scale

        # Clear old cache entries
        if frame_hash != self._frame_hash:
            self._cache.clear()
            self._frame_hash = frame_hash

        h, w = frame.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self._cache[cache_key] = small

        return small, scale


def scale_detections(detections, scale):
    """Scale detection coordinates back to original size."""
    if scale >= 1.0 or not detections:
        return detections
    inv_scale = 1.0 / scale
    return [(int(x * inv_scale), int(y * inv_scale),
             int(w * inv_scale), int(h * inv_scale))
            for (x, y, w, h) in detections]


# Pre-allocated buffers for common operations
class BufferPool:
    """
    Pool of pre-allocated numpy arrays to reduce memory allocation overhead.
    """

    def __init__(self):
        self._buffers = {}

    def get_buffer(self, shape, dtype=np.uint8):
        """Get a buffer of specified shape, reusing if possible."""
        key = (shape, dtype)
        if key not in self._buffers:
            self._buffers[key] = np.empty(shape, dtype=dtype)
        return self._buffers[key]

    def get_like(self, array):
        """Get a buffer matching another array's shape and dtype."""
        return self.get_buffer(array.shape, array.dtype)


# Global instances
frame_skipper = FrameSkipper()
perf_monitor = PerformanceMonitor()
grayscale_cache = GrayscaleCache()
resize_cache = ResizeCache()
buffer_pool = BufferPool()


def optimize_frame_for_detection(frame, max_dimension=480):
    """
    Downscale frame for faster detection if needed.
    Returns (scaled_frame, scale_factor)
    """
    h, w = frame.shape[:2]
    max_dim = max(h, w)

    if max_dim <= max_dimension:
        return frame, 1.0

    scale = max_dimension / max_dim
    return resize_cache.get_downscaled(frame, scale)


def batch_color_convert(frames, conversion):
    """
    Convert multiple frames in a batch (more efficient than individual calls).
    """
    return [cv2.cvtColor(f, conversion) for f in frames]


# Optimized common operations
def fast_blend(img1, img2, alpha):
    """Fast alpha blending using numpy."""
    return cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)


def fast_mask_blend(base, overlay, mask):
    """
    Fast masked blending using pre-normalized mask.
    Mask should be float32 in range [0, 1].
    """
    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]
    return (base * (1 - mask) + overlay * mask).astype(np.uint8)


def create_normalized_mask(mask_uint8):
    """Convert uint8 mask to float32 normalized mask for blending."""
    return mask_uint8.astype(np.float32) / 255.0
