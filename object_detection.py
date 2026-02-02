"""
Object Detection Module for Glitch Mirror
Provides face, body, and object detection with masking capabilities
for applying effects selectively to detected regions.
"""

import cv2
import numpy as np
import os

# Performance optimization imports
try:
    from performance import (
        frame_skipper, grayscale_cache, resize_cache,
        optimize_frame_for_detection, scale_detections
    )
    PERF_AVAILABLE = True
except ImportError:
    PERF_AVAILABLE = False

# Store cascade classifiers globally for performance
_face_cascade = None
_eye_cascade = None
_body_cascade = None
_profile_cascade = None
_upper_body_cascade = None
_smile_cascade = None
_cat_face_cascade = None
_eye_glasses_cascade = None
_license_plate_cascade = None

# Try to import imageio for GIF export
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("Note: imageio not installed. GIF export will use fallback method.")
    print("Install with: pip install imageio")


def get_cascade_path(cascade_name):
    """Get the path to OpenCV's built-in cascade files."""
    cv2_data_path = cv2.data.haarcascades
    return os.path.join(cv2_data_path, cascade_name)


def initialize_detectors():
    """Initialize all cascade classifiers."""
    global _face_cascade, _eye_cascade, _body_cascade, _profile_cascade
    global _upper_body_cascade, _smile_cascade, _cat_face_cascade
    global _eye_glasses_cascade, _license_plate_cascade

    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(get_cascade_path('haarcascade_frontalface_default.xml'))
    if _eye_cascade is None:
        _eye_cascade = cv2.CascadeClassifier(get_cascade_path('haarcascade_eye.xml'))
    if _body_cascade is None:
        _body_cascade = cv2.CascadeClassifier(get_cascade_path('haarcascade_fullbody.xml'))
    if _profile_cascade is None:
        _profile_cascade = cv2.CascadeClassifier(get_cascade_path('haarcascade_profileface.xml'))
    if _upper_body_cascade is None:
        _upper_body_cascade = cv2.CascadeClassifier(get_cascade_path('haarcascade_upperbody.xml'))
    if _smile_cascade is None:
        _smile_cascade = cv2.CascadeClassifier(get_cascade_path('haarcascade_smile.xml'))
    if _cat_face_cascade is None:
        _cat_face_cascade = cv2.CascadeClassifier(get_cascade_path('haarcascade_frontalcatface.xml'))
    if _eye_glasses_cascade is None:
        _eye_glasses_cascade = cv2.CascadeClassifier(get_cascade_path('haarcascade_eye_tree_eyeglasses.xml'))
    if _license_plate_cascade is None:
        try:
            plate_path = get_cascade_path('haarcascade_russian_plate_number.xml')
            if os.path.exists(plate_path):
                _license_plate_cascade = cv2.CascadeClassifier(plate_path)
            else:
                # Try alternative name
                plate_path = get_cascade_path('haarcascade_licence_plate_rus_16stages.xml')
                if os.path.exists(plate_path):
                    _license_plate_cascade = cv2.CascadeClassifier(plate_path)
                else:
                    _license_plate_cascade = None
        except Exception:
            _license_plate_cascade = None


def detect_faces(frame, scale_factor=1.1, min_neighbors=5, min_size=(30, 30), use_optimization=True):
    """
    Detect faces in the frame using Haar cascades.
    Returns list of (x, y, w, h) tuples for each detected face.
    Optimized with downscaling for faster detection.
    """
    initialize_detectors()

    # Optimize: downscale large frames for faster detection
    if use_optimization and PERF_AVAILABLE:
        small_frame, scale = optimize_frame_for_detection(frame, max_dimension=480)
        if scale < 1.0:
            min_size = (max(20, int(min_size[0] * scale)), max(20, int(min_size[1] * scale)))
    else:
        small_frame = frame
        scale = 1.0

    # Use cached grayscale if available
    if PERF_AVAILABLE:
        gray = grayscale_cache.get_equalized(small_frame)
    else:
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Skip profile detection for better performance (optional)
    # Profile faces are less common and slow down detection
    all_faces = list(faces) if len(faces) > 0 else []

    # Scale detections back to original size
    if scale < 1.0 and PERF_AVAILABLE:
        all_faces = scale_detections(all_faces, scale)

    return all_faces


def detect_eyes(frame, faces=None, scale_factor=1.05, min_neighbors=4):
    """
    Detect eyes in the frame. If faces are provided, only search within face regions.
    Returns list of (x, y, w, h) tuples for each detected eye.
    Optimized to only search within face regions when faces are provided.
    """
    initialize_detectors()

    # Only detect eyes if we have faces - much faster
    if faces is None or len(faces) == 0:
        return []

    # Use cached CLAHE if available
    if PERF_AVAILABLE:
        gray = grayscale_cache.get_clahe(frame)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    eyes = []

    # Search for eyes within face regions (upper 60% for better coverage)
    for face in faces:
        # Handle both 4-element (x,y,w,h) and 5-element (x,y,w,h,id) tuples
        fx, fy, fw, fh = face[:4]

        # Expand search region slightly
        eye_region_top = fy + int(fh * 0.15)
        eye_region_bottom = fy + int(fh * 0.55)

        # Bounds check
        eye_region_top = max(0, eye_region_top)
        eye_region_bottom = min(gray.shape[0], eye_region_bottom)
        fx_end = min(gray.shape[1], fx + fw)
        fx = max(0, fx)

        roi_gray = gray[eye_region_top:eye_region_bottom, fx:fx_end]

        if roi_gray.size == 0:
            continue

        detected_eyes = _eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(max(10, int(fw * 0.08)), max(8, int(fh * 0.04))),
            maxSize=(int(fw * 0.4), int(fh * 0.3))
        )
        for (ex, ey, ew, eh) in detected_eyes:
            # Convert to global coordinates
            eyes.append((fx + ex, eye_region_top + ey, ew, eh))

    return eyes


def detect_bodies(frame, scale_factor=1.05, min_neighbors=2, min_size=(40, 80)):
    """
    Detect full bodies in the frame.
    Returns list of (x, y, w, h) tuples for each detected body.
    Optimized with downscaling for faster detection.
    """
    initialize_detectors()

    # Downscale for faster detection (body detection is expensive)
    if PERF_AVAILABLE:
        small_frame, scale = optimize_frame_for_detection(frame, max_dimension=320)
        gray = grayscale_cache.get_equalized(small_frame)
        min_size = (max(20, int(min_size[0] * scale)), max(40, int(min_size[1] * scale)))
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        scale = 1.0

    h, w = gray.shape[:2]

    bodies = _body_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        maxSize=(int(w * 0.9), int(h * 0.95)),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Filter out obvious false positives (aspect ratio check)
    filtered_bodies = []
    for (x, y, bw, bh) in bodies:
        aspect_ratio = bh / bw if bw > 0 else 0
        if 1.2 <= aspect_ratio <= 5.0:
            filtered_bodies.append((x, y, bw, bh))

    # Scale back to original size
    if scale < 1.0 and PERF_AVAILABLE:
        filtered_bodies = scale_detections(filtered_bodies, scale)

    return filtered_bodies


def detect_upper_body(frame, scale_factor=1.08, min_neighbors=2, min_size=(40, 40)):
    """
    Detect upper bodies (torso and head) in the frame.
    Returns list of (x, y, w, h) tuples for each detected upper body.
    Optimized with downscaling.
    """
    initialize_detectors()

    # Downscale for faster detection
    if PERF_AVAILABLE:
        small_frame, scale = optimize_frame_for_detection(frame, max_dimension=400)
        gray = grayscale_cache.get_clahe(small_frame)
        min_size = (max(20, int(min_size[0] * scale)), max(20, int(min_size[1] * scale)))
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        scale = 1.0

    h, w = gray.shape[:2]

    upper_bodies = _upper_body_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        maxSize=(int(w * 0.8), int(h * 0.8)),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Filter by aspect ratio
    filtered = []
    for (x, y, uw, uh) in upper_bodies:
        aspect_ratio = uh / uw if uw > 0 else 0
        if 0.5 <= aspect_ratio <= 2.5:
            filtered.append((x, y, uw, uh))

    # Scale back to original size
    if scale < 1.0 and PERF_AVAILABLE:
        filtered = scale_detections(filtered, scale)

    return filtered


def detect_smiles(frame, faces=None, scale_factor=1.7, min_neighbors=20):
    """
    Detect smiles in the frame. If faces are provided, only search within face regions.
    Returns list of (x, y, w, h) tuples for each detected smile.
    """
    initialize_detectors()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smiles = []

    if faces is not None and len(faces) > 0:
        # Search for smiles only within face regions (lower half)
        for face in faces:
            fx, fy, fw, fh = face[:4]  # Handle both 4 and 5 element tuples
            roi_gray = gray[fy + fh // 2:fy + fh, fx:fx + fw]
            detected_smiles = _smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(25, 25)
            )
            for (sx, sy, sw, sh) in detected_smiles:
                # Convert to global coordinates
                smiles.append((fx + sx, fy + fh // 2 + sy, sw, sh))
    else:
        # Search entire frame
        detected_smiles = _smile_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(25, 25)
        )
        smiles = list(detected_smiles)

    return smiles


def detect_cat_faces(frame, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    """
    Detect cat faces in the frame.
    Returns list of (x, y, w, h) tuples for each detected cat face.
    """
    initialize_detectors()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cat_faces = _cat_face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return list(cat_faces) if len(cat_faces) > 0 else []


def detect_eyes_with_glasses(frame, faces=None, scale_factor=1.1, min_neighbors=5):
    """
    Detect eyes with glasses in the frame.
    Returns list of (x, y, w, h) tuples for each detected eye with glasses.
    """
    initialize_detectors()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = []

    if faces is not None and len(faces) > 0:
        for face in faces:
            fx, fy, fw, fh = face[:4]  # Handle both 4 and 5 element tuples
            roi_gray = gray[fy:fy + fh // 2, fx:fx + fw]
            detected_eyes = _eye_glasses_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(20, 20)
            )
            for (ex, ey, ew, eh) in detected_eyes:
                eyes.append((fx + ex, fy + ey, ew, eh))
    else:
        detected_eyes = _eye_glasses_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(20, 20)
        )
        eyes = list(detected_eyes)

    return eyes


def detect_license_plates(frame, scale_factor=1.1, min_neighbors=3, min_size=(30, 10)):
    """
    Detect license plates in the frame.
    Returns list of (x, y, w, h) tuples for each detected license plate.
    """
    initialize_detectors()

    if _license_plate_cascade is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    plates = _license_plate_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return list(plates) if len(plates) > 0 else []


# ============================================
# GIF EXPORT FUNCTIONALITY
# ============================================

class GifExporter:
    """
    Renders frames and exports them as an animated GIF.
    Uses rendering instead of recording for consistent quality.
    """

    def __init__(self):
        self.frames = []
        self.is_rendering = False
        self.render_cancelled = False
        self.target_fps = 15
        self.max_frames = 600  # Max ~40 seconds at 15fps
        self.render_progress = 0
        self.render_status = "Ready"

    def get_frame_count(self):
        """Get number of rendered frames."""
        return len(self.frames)

    def get_duration(self):
        """Get estimated duration in seconds."""
        return len(self.frames) / self.target_fps if self.target_fps > 0 else 0

    def cancel_render(self):
        """Cancel ongoing render."""
        self.render_cancelled = True

    def render_gif_frames(self, source_frame, apply_effects_func, num_frames, fps=15,
                          effect_states=None, effect_params=None, detection_params=None,
                          region_effect_manager=None, progress_callback=None):
        """
        Render GIF frames by applying effects to a source frame.

        Args:
            source_frame: The base frame (image) to apply effects to
            apply_effects_func: Function to apply effects to a frame
            num_frames: Number of frames to render
            fps: Target FPS for the GIF
            effect_states: Dict of effect on/off states
            effect_params: Dict of effect parameters
            detection_params: Dict of detection parameters
            region_effect_manager: Manager for per-region effects
            progress_callback: Function(progress, status) to report progress

        Returns:
            True if successful, False otherwise
        """
        self.frames = []
        self.is_rendering = True
        self.render_cancelled = False
        self.target_fps = fps
        self.render_progress = 0
        self.render_status = "Rendering..."

        print(f"Rendering {num_frames} GIF frames at {fps} fps...")

        try:
            # Initialize buffers for effects that need them
            render_buffers = {}
            prev_gray = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)

            for frame_num in range(num_frames):
                if self.render_cancelled:
                    print("Render cancelled")
                    self.render_status = "Cancelled"
                    self.is_rendering = False
                    return False

                # Apply effects to the frame
                output_frame, prev_gray, render_buffers = apply_effects_func(
                    source_frame.copy(), prev_gray, frame_num, render_buffers
                )

                # Apply detection-based effects if enabled
                if detection_params and detection_params.get('detection_enabled', False):
                    output_frame = self._apply_detection_effects(
                        source_frame, output_frame, detection_params,
                        region_effect_manager, frame_num
                    )

                # Convert BGR to RGB for GIF
                rgb_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                self.frames.append(rgb_frame)

                # Update progress
                self.render_progress = ((frame_num + 1) / num_frames) * 100
                self.render_status = f"Rendering: {frame_num + 1}/{num_frames}"

                if progress_callback:
                    progress_callback(self.render_progress, self.render_status)

            self.render_status = f"Done: {len(self.frames)} frames"
            print(f"Rendered {len(self.frames)} frames")
            self.is_rendering = False
            return True

        except Exception as e:
            print(f"Error rendering GIF frames: {e}")
            self.render_status = f"Error: {e}"
            self.is_rendering = False
            return False

    def render_gif_from_video(self, video_path, apply_effects_func, start_frame, num_frames, fps=15,
                               effect_states=None, effect_params=None, detection_params=None,
                               region_effect_manager=None, progress_callback=None):
        """
        Render GIF frames from a video source.

        Args:
            video_path: Path to the video file
            apply_effects_func: Function to apply effects to a frame
            start_frame: Starting frame number in the video
            num_frames: Number of frames to render
            fps: Target FPS for the GIF
            effect_states: Dict of effect on/off states
            effect_params: Dict of effect parameters
            detection_params: Dict of detection parameters
            region_effect_manager: Manager for per-region effects
            progress_callback: Function(progress, status) to report progress

        Returns:
            True if successful, False otherwise
        """
        self.frames = []
        self.is_rendering = True
        self.render_cancelled = False
        self.target_fps = fps
        self.render_progress = 0
        self.render_status = "Rendering from video..."

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            self.render_status = "Error: Could not open video"
            self.is_rendering = False
            return False

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30

        # Calculate frame skip to match target FPS
        frame_skip = max(1, int(video_fps / fps))

        print(f"Rendering {num_frames} GIF frames from video at {fps} fps (skip: {frame_skip})...")

        try:
            render_buffers = {}
            prev_gray = None
            frames_rendered = 0
            current_frame = start_frame

            while frames_rendered < num_frames and current_frame < total_frames:
                if self.render_cancelled:
                    print("Render cancelled")
                    self.render_status = "Cancelled"
                    cap.release()
                    self.is_rendering = False
                    return False

                ret, frame = cap.read()
                if not ret:
                    break

                # Initialize prev_gray on first frame
                if prev_gray is None:
                    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Apply effects
                output_frame, prev_gray, render_buffers = apply_effects_func(
                    frame.copy(), prev_gray, frames_rendered, render_buffers
                )

                # Apply detection-based effects if enabled
                if detection_params and detection_params.get('detection_enabled', False):
                    output_frame = self._apply_detection_effects(
                        frame, output_frame, detection_params,
                        region_effect_manager, frames_rendered
                    )

                # Convert BGR to RGB for GIF
                rgb_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                self.frames.append(rgb_frame)

                frames_rendered += 1
                current_frame += frame_skip

                # Skip frames to match target FPS
                if frame_skip > 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

                # Update progress
                self.render_progress = (frames_rendered / num_frames) * 100
                self.render_status = f"Rendering: {frames_rendered}/{num_frames}"

                if progress_callback:
                    progress_callback(self.render_progress, self.render_status)

            cap.release()
            self.render_status = f"Done: {len(self.frames)} frames"
            print(f"Rendered {len(self.frames)} frames from video")
            self.is_rendering = False
            return True

        except Exception as e:
            print(f"Error rendering GIF from video: {e}")
            cap.release()
            self.render_status = f"Error: {e}"
            self.is_rendering = False
            return False

    def _apply_detection_effects(self, original_frame, effect_frame, detection_params,
                                   region_effect_manager, frame_number):
        """Apply detection-based effects to a frame."""
        output = effect_frame.copy()

        # Detect faces if enabled
        detected_faces = []
        detected_eyes = []
        detected_cats = []

        if detection_params.get('detect_faces', True):
            detected_faces = detect_faces(
                original_frame,
                scale_factor=detection_params.get('detection_sensitivity', 1.1),
                min_neighbors=detection_params.get('min_neighbors', 5)
            )

        if detection_params.get('detect_eyes', False) and detected_faces:
            detected_eyes = detect_eyes(original_frame, detected_faces)

        if detection_params.get('detect_cats', False):
            detected_cats = detect_cat_faces(original_frame)

        # Apply per-region effects if enabled
        if detection_params.get('per_region_mode', False) and region_effect_manager:
            detections_dict = {
                'faces': detected_faces,
                'eyes': detected_eyes,
                'cats': detected_cats,
                'bodies': [],
                'upper_bodies': [],
                'smiles': [],
                'plates': []
            }
            output = region_effect_manager.apply_all_region_effects(
                output, detections_dict, frame_number
            )

        return output

    def export_gif(self, output_path, fps=None, loop=0):
        """
        Export rendered frames as GIF.

        Args:
            output_path: Path to save the GIF
            fps: FPS for playback (default: render fps)
            loop: 0 = infinite loop, N = loop N times

        Returns:
            True if successful, False otherwise
        """
        if len(self.frames) == 0:
            print("No frames to export! Render frames first.")
            return False

        if fps is None:
            fps = self.target_fps

        frame_duration = 1.0 / fps

        try:
            if IMAGEIO_AVAILABLE:
                imageio.mimsave(
                    output_path,
                    self.frames,
                    duration=frame_duration,
                    loop=loop
                )
            else:
                self._export_gif_fallback(output_path, fps)

            print(f"GIF exported to: {output_path}")
            print(f"  Frames: {len(self.frames)}")
            print(f"  Duration: {len(self.frames) / fps:.2f} seconds")
            print(f"  FPS: {fps}")
            return True

        except Exception as e:
            print(f"Error exporting GIF: {e}")
            return False

    def _export_gif_fallback(self, output_path, fps):
        """Fallback GIF export without imageio (grayscale only)."""
        import struct

        height, width = self.frames[0].shape[:2]
        delay = max(1, int(100 / fps))

        with open(output_path, 'wb') as f:
            # GIF89a header
            f.write(b'GIF89a')
            f.write(struct.pack('<HH', width, height))
            f.write(struct.pack('BBB', 0xF7, 0, 0))

            # Global color table (256 grays)
            for i in range(256):
                f.write(struct.pack('BBB', i, i, i))

            # Netscape looping extension
            f.write(b'\x21\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00')

            for frame in self.frames:
                # Convert to grayscale
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else:
                    gray = frame

                # Graphic control extension
                f.write(b'\x21\xF9\x04\x00')
                f.write(struct.pack('<H', delay))
                f.write(b'\x00\x00')

                # Image descriptor
                f.write(b'\x2C')
                f.write(struct.pack('<HHHH', 0, 0, width, height))
                f.write(b'\x00')

                # LZW minimum code size
                f.write(b'\x08')

                # Simple uncompressed data (not true LZW, but works for basic GIFs)
                data = gray.flatten()
                block_size = 254
                for i in range(0, len(data), block_size):
                    block = data[i:i+block_size]
                    f.write(struct.pack('B', len(block)))
                    f.write(block.tobytes())
                f.write(b'\x00')

            f.write(b'\x3B')

    def clear(self):
        """Clear all rendered frames."""
        self.frames = []
        self.is_rendering = False
        self.render_cancelled = False
        self.render_progress = 0
        self.render_status = "Ready"


# Global GIF exporter instance
_gif_exporter = None


def get_gif_exporter():
    """Get or create the global GIF exporter."""
    global _gif_exporter
    if _gif_exporter is None:
        _gif_exporter = GifExporter()
    return _gif_exporter


def create_detection_mask(frame_shape, detections, padding=0, feather=0):
    """
    Create a binary mask from detection rectangles.
    Handles both 4-element (x,y,w,h) and 5-element (x,y,w,h,id) tuples.

    Args:
        frame_shape: (height, width, channels) of the frame
        detections: list of (x, y, w, h) or (x, y, w, h, id) rectangles
        padding: extra pixels to add around each detection
        feather: amount of edge blurring for smooth transitions

    Returns:
        Mask with same dimensions as frame (single channel, 0-255)
    """
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for det in detections:
        x, y, w, h = det[:4]  # Works for both 4 and 5 element tuples
        # Apply padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)

        # Draw filled rectangle
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # Apply feathering if requested
    if feather > 0:
        kernel_size = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    return mask


def create_elliptical_mask(frame_shape, detections, padding=0, feather=20):
    """
    Create an elliptical mask from detection rectangles (better for faces).
    Handles both 4-element (x,y,w,h) and 5-element (x,y,w,h,id) tuples.

    Args:
        frame_shape: (height, width, channels) of the frame
        detections: list of (x, y, w, h) or (x, y, w, h, id) rectangles
        padding: extra pixels to add around each detection
        feather: amount of edge blurring for smooth transitions

    Returns:
        Mask with same dimensions as frame (single channel, 0-255)
    """
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for det in detections:
        x, y, w, h = det[:4]  # Works for both 4 and 5 element tuples
        # Calculate ellipse center and axes
        center_x = x + w // 2
        center_y = y + h // 2
        axis_x = (w // 2) + padding
        axis_y = (h // 2) + padding

        # Draw filled ellipse
        cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 255, -1)

    # Apply feathering
    if feather > 0:
        kernel_size = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    return mask


def apply_effect_with_mask(original_frame, effect_frame, mask, invert=False):
    """
    Blend original and effect frames using a mask.

    Args:
        original_frame: the unprocessed frame
        effect_frame: the frame with effects applied
        mask: single channel mask (0-255)
        invert: if True, apply effect to area outside the mask

    Returns:
        Blended frame
    """
    if invert:
        mask = 255 - mask

    # Normalize mask to 0-1 range
    mask_normalized = mask.astype(np.float32) / 255.0

    # Expand mask to 3 channels
    mask_3d = np.stack([mask_normalized] * 3, axis=-1)

    # Blend: effect where mask is white, original where mask is black
    blended = (effect_frame.astype(np.float32) * mask_3d +
               original_frame.astype(np.float32) * (1 - mask_3d))

    return blended.astype(np.uint8)


def draw_detection_boxes(frame, detections, color=(0, 255, 0), thickness=2, label=None):
    """
    Draw rectangles around detected objects.
    Useful for debugging/visualization.
    Handles both 4-element (x,y,w,h) and 5-element (x,y,w,h,id) tuples.
    """
    output = frame.copy()

    for i, det in enumerate(detections):
        # Handle both 4 and 5 element tuples
        if len(det) >= 5:
            x, y, w, h, det_id = det[:5]
            display_label = f"{label} ID:{det_id}" if label else f"ID:{det_id}"
        else:
            x, y, w, h = det
            display_label = f"{label} {i+1}" if label else None

        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)

        if display_label:
            cv2.putText(output, display_label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return output


# ============================================
# SPECIAL DETECTION-BASED EFFECTS
# ============================================

def face_pixelate(frame, faces, pixel_size=10):
    """
    Pixelate detected faces for anonymization effect.
    Handles both 4-element and 5-element detection tuples.
    """
    output = frame.copy()

    for det in faces:
        x, y, w, h = det[:4]  # Works for both 4 and 5 element tuples
        # Extract face region
        face_roi = output[y:y+h, x:x+w]

        # Pixelate by downscaling and upscaling
        small = cv2.resize(face_roi, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        # Put back
        output[y:y+h, x:x+w] = pixelated

    return output


def face_blur(frame, faces, blur_amount=51):
    """
    Blur detected faces.
    """
    output = frame.copy()

    for face in faces:
        x, y, w, h = face[:4]  # Handle both 4 and 5 element tuples
        face_roi = output[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(face_roi, (blur_amount, blur_amount), 0)
        output[y:y+h, x:x+w] = blurred

    return output


def face_glitch(frame, faces, glitch_intensity=20):
    """
    Apply glitch effect specifically to faces.
    """
    output = frame.copy()

    for face in faces:
        x, y, w, h = face[:4]  # Handle both 4 and 5 element tuples
        face_roi = output[y:y+h, x:x+w].copy()

        # RGB channel shift
        b, g, r = cv2.split(face_roi)
        shift = np.random.randint(-glitch_intensity, glitch_intensity)
        r = np.roll(r, shift, axis=1)
        b = np.roll(b, -shift, axis=1)
        face_roi = cv2.merge([b, g, r])

        # Random horizontal line shifts
        for _ in range(5):
            row = np.random.randint(0, h - 1)
            row_shift = np.random.randint(-glitch_intensity, glitch_intensity)
            face_roi[row:row+2] = np.roll(face_roi[row:row+2], row_shift, axis=1)

        output[y:y+h, x:x+w] = face_roi

    return output


def face_thermal(frame, faces):
    """
    Apply thermal vision effect only to faces.
    """
    output = frame.copy()

    for face in faces:
        x, y, w, h = face[:4]  # Handle both 4 and 5 element tuples
        face_roi = output[y:y+h, x:x+w]
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        contrast = cv2.equalizeHist(gray)
        thermal = cv2.applyColorMap(contrast, cv2.COLORMAP_INFERNO)
        output[y:y+h, x:x+w] = thermal

    return output


def face_neon_outline(frame, faces, color=(0, 255, 255)):
    """
    Add neon glow outline around detected faces.
    """
    output = frame.copy()
    mask = create_elliptical_mask(frame.shape, faces, padding=5, feather=0)

    # Find edges of the mask
    edges = cv2.Canny(mask, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    # Create glow
    glow = cv2.GaussianBlur(edges, (21, 21), 0)

    # Color the glow
    glow_colored = np.zeros_like(frame)
    glow_colored[:, :, 0] = (glow * (color[0] / 255.0)).astype(np.uint8)
    glow_colored[:, :, 1] = (glow * (color[1] / 255.0)).astype(np.uint8)
    glow_colored[:, :, 2] = (glow * (color[2] / 255.0)).astype(np.uint8)

    # Add glow to frame
    output = cv2.add(output, glow_colored)

    return np.clip(output, 0, 255).astype(np.uint8)


def face_cartoon(frame, faces):
    """
    Apply cartoon effect to detected faces.
    """
    output = frame.copy()

    for face in faces:
        x, y, w, h = face[:4]  # Handle both 4 and 5 element tuples
        face_roi = output[y:y+h, x:x+w]

        # Bilateral filter for smooth colors
        smooth = cv2.bilateralFilter(face_roi, 9, 75, 75)

        # Edge detection
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 9, 9)

        # Color quantization
        quantized = (smooth // 32) * 32

        # Combine edges with quantized colors
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(quantized, edges_colored)

        output[y:y+h, x:x+w] = cartoon

    return output


def face_swap_color(frame, faces, hue_shift=90):
    """
    Shift the hue of detected faces for alien/colorful effect.
    """
    output = frame.copy()

    for face in faces:
        x, y, w, h = face[:4]  # Handle both 4 and 5 element tuples
        face_roi = output[y:y+h, x:x+w]

        # Convert to HSV and shift hue
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        hsv = hsv.astype(np.uint8)

        output[y:y+h, x:x+w] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return output


def face_edge_highlight(frame, faces):
    """
    Highlight edges/features of detected faces.
    """
    output = frame.copy()

    for face in faces:
        x, y, w, h = face[:4]  # Handle both 4 and 5 element tuples
        face_roi = output[y:y+h, x:x+w]

        # Edge detection
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Blend edges with original
        highlighted = cv2.addWeighted(face_roi, 0.7, edges_colored, 0.5, 0)
        output[y:y+h, x:x+w] = highlighted

    return output


def background_replace(frame, faces, background_color=(0, 0, 0), feather=30):
    """
    Replace everything except detected faces with a solid color.
    """
    # Create mask from face detections
    mask = create_elliptical_mask(frame.shape, faces, padding=20, feather=feather)

    # Create background
    background = np.full_like(frame, background_color)

    # Blend
    result = apply_effect_with_mask(background, frame, mask, invert=False)

    return result


def face_vignette(frame, faces, intensity=0.7):
    """
    Apply vignette effect centered on detected faces.
    """
    height, width = frame.shape[:2]
    output = frame.copy()

    if len(faces) == 0:
        return output

    # Create combined vignette from all face centers
    vignette_mask = np.zeros((height, width), dtype=np.float32)

    for face in faces:
        x, y, w, h = face[:4]  # Handle both 4 and 5 element tuples
        center_x = x + w // 2
        center_y = y + h // 2

        # Create radial gradient from this face
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        distance = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        max_distance = np.sqrt(width ** 2 + height ** 2) / 2

        face_vignette = 1 - (distance / max_distance) * intensity
        face_vignette = np.clip(face_vignette, 0, 1)

        # Combine with max (brightest wins)
        vignette_mask = np.maximum(vignette_mask, face_vignette)

    # Apply vignette
    for c in range(3):
        output[:, :, c] = (output[:, :, c] * vignette_mask).astype(np.uint8)

    return output


# ============================================
# TRACKING UTILITIES FOR SMOOTH DETECTION
# ============================================

class DetectionTracker:
    """
    Track detections across frames for smoother results.
    Helps reduce flickering/jitter in detection boxes.
    Now includes support for ignoring specific regions and persistent IDs.
    """

    def __init__(self, smoothing_frames=5, detection_threshold=0.5):
        self.history = []
        self.smoothing_frames = smoothing_frames
        self.detection_threshold = detection_threshold
        self.next_id = 1
        self.tracked_detections = {}  # id -> {rect, det_type, last_seen, ignored, ...}
        self.ignored_regions = set()  # Set of detection IDs to ignore
        self.kept_regions = set()  # Set of detection IDs to keep (won't be cleared on re-detect)
        self.max_missing_frames = 10  # Remove tracking after this many frames without detection
        self.static_mode = False  # When True, don't auto-cleanup old detections

    def update(self, detections, detection_type='face'):
        """
        Update tracker with new detections.
        Returns smoothed detections with persistent IDs.
        Each detection is now (x, y, w, h, id) tuple.
        """
        self.history.append(detections)

        if len(self.history) > self.smoothing_frames:
            self.history.pop(0)

        if len(self.history) < 2:
            # Assign IDs to first frame detections
            result = []
            for det in detections:
                det_id = self._assign_id(det, detection_type)
                if det_id not in self.ignored_regions:
                    result.append((*det, det_id))
            return result

        # Simple smoothing: average positions across history
        smoothed = []

        # Match detections across frames (simple IoU matching)
        if len(detections) > 0:
            for det in detections:
                x, y, w, h = det[:4]  # Handle both 4 and 5 element tuples

                # Find matching tracked detection
                det_id = self._match_or_create(det, detection_type)

                # Skip if this detection is ignored
                if det_id in self.ignored_regions:
                    continue

                # Find similar detections in history
                similar_positions = [(x, y, w, h)]

                for past_detections in self.history[:-1]:
                    for past_det in past_detections:
                        px, py, pw, ph = past_det[:4]  # Handle both 4 and 5 element tuples

                        # Check overlap (simple center distance check)
                        center_dist = np.sqrt((x + w/2 - px - pw/2)**2 + (y + h/2 - py - ph/2)**2)
                        max_dim = max(w, h, pw, ph)

                        if center_dist < max_dim:
                            similar_positions.append((px, py, pw, ph))
                            break

                # Average the similar positions
                if len(similar_positions) > 1:
                    avg_x = int(np.mean([p[0] for p in similar_positions]))
                    avg_y = int(np.mean([p[1] for p in similar_positions]))
                    avg_w = int(np.mean([p[2] for p in similar_positions]))
                    avg_h = int(np.mean([p[3] for p in similar_positions]))
                    smoothed.append((avg_x, avg_y, avg_w, avg_h, det_id))
                else:
                    smoothed.append((x, y, w, h, det_id))

        # Update last seen and clean up old detections (unless in static mode)
        if not self.static_mode:
            self._cleanup_old_detections()

        return smoothed

    def _assign_id(self, det, detection_type):
        """Assign a new ID to a detection."""
        det_id = self.next_id
        self.next_id += 1
        x, y, w, h = det
        self.tracked_detections[det_id] = {
            'rect': det,
            'type': detection_type,
            'last_seen': 0,
            'center': (x + w // 2, y + h // 2)
        }
        return det_id

    def _match_or_create(self, det, detection_type):
        """Match detection to existing tracked ID or create new one."""
        x, y, w, h = det
        center = (x + w // 2, y + h // 2)

        # Find best matching tracked detection
        best_match = None
        best_dist = float('inf')

        for det_id, tracked in self.tracked_detections.items():
            if tracked['type'] != detection_type:
                continue

            tx, ty = tracked['center']
            dist = np.sqrt((center[0] - tx) ** 2 + (center[1] - ty) ** 2)
            max_dim = max(w, h)

            if dist < max_dim * 1.5 and dist < best_dist:
                best_dist = dist
                best_match = det_id

        if best_match is not None:
            # Update existing detection
            self.tracked_detections[best_match]['rect'] = det
            self.tracked_detections[best_match]['center'] = center
            self.tracked_detections[best_match]['last_seen'] = 0
            return best_match
        else:
            # Create new detection
            return self._assign_id(det, detection_type)

    def _cleanup_old_detections(self):
        """Remove detections that haven't been seen recently (preserves kept detections)."""
        to_remove = []
        for det_id, tracked in self.tracked_detections.items():
            tracked['last_seen'] += 1
            # Don't remove kept detections
            if tracked['last_seen'] > self.max_missing_frames and det_id not in self.kept_regions:
                to_remove.append(det_id)

        for det_id in to_remove:
            del self.tracked_detections[det_id]
            self.ignored_regions.discard(det_id)

    def ignore_detection(self, det_id):
        """Mark a detection ID to be ignored."""
        self.ignored_regions.add(det_id)

    def unignore_detection(self, det_id):
        """Remove a detection ID from ignored list."""
        self.ignored_regions.discard(det_id)

    def toggle_ignore(self, det_id):
        """Toggle ignore status of a detection."""
        if det_id in self.ignored_regions:
            self.ignored_regions.discard(det_id)
            return False
        else:
            self.ignored_regions.add(det_id)
            return True

    def is_ignored(self, det_id):
        """Check if a detection is ignored."""
        return det_id in self.ignored_regions

    def get_all_tracked(self):
        """Get all currently tracked detections."""
        return dict(self.tracked_detections)

    def get_ignored_list(self):
        """Get list of ignored detection IDs."""
        return list(self.ignored_regions)

    def clear_ignored(self):
        """Clear all ignored detections."""
        self.ignored_regions.clear()

    # === Keep/Lock Detection Methods ===

    def keep_detection(self, det_id):
        """Mark a detection ID to be kept (persists through re-detection)."""
        if det_id in self.tracked_detections:
            self.kept_regions.add(det_id)

    def unkeep_detection(self, det_id):
        """Remove a detection ID from kept list."""
        self.kept_regions.discard(det_id)

    def is_kept(self, det_id):
        """Check if a detection is kept."""
        return det_id in self.kept_regions

    def get_kept_list(self):
        """Get list of kept detection IDs."""
        return list(self.kept_regions)

    def clear_kept(self):
        """Clear all kept detections."""
        self.kept_regions.clear()

    def clear_non_kept(self):
        """Clear all detections except kept ones."""
        to_remove = [det_id for det_id in self.tracked_detections if det_id not in self.kept_regions]
        for det_id in to_remove:
            del self.tracked_detections[det_id]
            self.ignored_regions.discard(det_id)
        self.history.clear()

    def clear(self):
        """Clear tracking history (preserves kept detections)."""
        self.history = []
        # Keep kept detections
        kept_detections = {k: v for k, v in self.tracked_detections.items() if k in self.kept_regions}
        self.tracked_detections = kept_detections
        # Only clear ignored that aren't kept
        self.ignored_regions = self.ignored_regions.intersection(self.kept_regions)
        # Don't reset next_id to avoid ID conflicts

    def clear_all(self):
        """Completely clear all tracking (including kept)."""
        self.history = []
        self.tracked_detections = {}
        self.ignored_regions.clear()
        self.kept_regions.clear()
        self.next_id = 1

    def set_static_mode(self, enabled):
        """
        Enable/disable static mode.
        In static mode, detections are not automatically cleaned up.
        Use this for static images or single-shot detection.
        """
        self.static_mode = enabled

    def refresh_all_last_seen(self):
        """Reset last_seen counter for all tracked detections (prevents cleanup)."""
        for det_id in self.tracked_detections:
            self.tracked_detections[det_id]['last_seen'] = 0


# ============================================
# REGION-BASED EFFECT SYSTEM
# Per-region independent effect stacks
# ============================================

class RegionEffectManager:
    """
    Manages independent effect stacks for individual detected regions.
    Each region (face, eye, body) can have its own effect configuration.
    """

    # Available effects that can be applied per-region
    AVAILABLE_EFFECTS = [
        'none',
        'restore_original',  # Special: removes all effects, shows base image
        'pixelate',
        'blur',
        'glitch',
        'thermal',
        'cartoon',
        'color_shift',
        'edge_highlight',
        'negative',
        'posterize',
        'emboss',
        'sketch',
        'neon_glow',
        'video_texture',  # Play video inside region
        'image_texture',  # Static image inside region
        'mirror',
        'wave_distort',
        'rgb_shift',
        'vhs',
        'kaleidoscope',
    ]

    def __init__(self):
        # Store effect configs per region ID
        # Region IDs: "face_0", "face_1", "eye_0", "eye_1", "body_0", etc.
        self.region_effects = {}

        # Effect stacks per region - list of effects to apply in order
        # Each entry: [{'effect_type': 'blur', 'blur_amount': 31, ...}, ...]
        self.effect_stacks = {}

        # Video textures for regions (region_id -> VideoCapture)
        self.video_textures = {}

        # Image textures for regions (region_id -> numpy array)
        self.image_textures = {}

        # Baked effect textures - stores rendered effect snapshots per region
        # Each entry: {'texture': numpy array, 'original_rect': (x,y,w,h)}
        self.baked_textures = {}

        # Default effect settings - includes all possible parameters
        self.default_effect = {
            'effect_type': 'none',
            # Universal
            'intensity': 50,
            'color_shift': 90,
            # Pixelate
            'pixelate_size': 10,
            # Blur
            'blur_amount': 31,
            # Posterize
            'posterize_levels': 6,
            # Wave/Distortion
            'wave_amplitude': 20,
            'wave_frequency': 5,
            # Chromatic
            'chromatic_offset': 5,
            # VHS/Noise
            'noise_intensity': 25,
            # Film Grain
            'grain_intensity': 30,
            # TV Static
            'static_blend': 30,
            # Scanlines
            'scanline_darkness': 40,
            # Ghost Trail
            'trail_decay': 85,
            # Kaleidoscope
            'segments': 6,
            # Emboss
            'emboss_strength': 50,
            # Radial Blur
            'radial_strength': 10,
            # Tunnel Vision
            'vignette_intensity': 70,
            # Double Vision
            'offset': 15,
            # Halftone
            'dot_size': 4,
            # Neon Glow
            'glow_size': 5,
            # Glitch
            'glitch_intensity': 20,
            'glitch_blocks': 8,
            # Heat Distort
            'heat_intensity': 8,
            # Drunk
            'wobble_intensity': 15,
            # Prism
            'prism_offset': 8,
            # Spiral
            'spiral_strength': 50,
            # Blocky Noise
            'block_chance': 10,
            # RGB Split
            'split_strength': 10,
            # Texture settings
            'video_path': None,
            'image_path': None,
            'video_loop': True,
            'blend_mode': 'replace',  # 'replace', 'overlay', 'multiply', 'screen'
            'blend_opacity': 1.0,
        }

    def get_region_effect(self, region_id):
        """Get effect configuration for a region, creating default if needed."""
        if region_id not in self.region_effects:
            self.region_effects[region_id] = self.default_effect.copy()
        return self.region_effects[region_id]

    def set_region_effect(self, region_id, effect_type, **kwargs):
        """Set effect type and parameters for a region (single effect, used for live preview)."""
        if region_id not in self.region_effects:
            self.region_effects[region_id] = self.default_effect.copy()

        self.region_effects[region_id]['effect_type'] = effect_type

        # Update any additional parameters (accept all, not just existing ones)
        for key, value in kwargs.items():
            self.region_effects[region_id][key] = value

    # === Effect Stack Methods ===

    def add_effect_to_stack(self, region_id, effect_type, **kwargs):
        """Add an effect to the region's effect stack."""
        if region_id not in self.effect_stacks:
            self.effect_stacks[region_id] = []

        # Create effect entry with all parameters
        effect_entry = {'effect_type': effect_type}
        effect_entry.update(kwargs)

        self.effect_stacks[region_id].append(effect_entry)
        return len(self.effect_stacks[region_id])

    def get_effect_stack(self, region_id):
        """Get the effect stack for a region."""
        return self.effect_stacks.get(region_id, [])

    def clear_effect_stack(self, region_id):
        """Clear all effects from a region's stack."""
        if region_id in self.effect_stacks:
            self.effect_stacks[region_id] = []
        # Also clear the single effect
        if region_id in self.region_effects:
            self.region_effects[region_id]['effect_type'] = 'none'

    def remove_last_effect_from_stack(self, region_id):
        """Remove the last effect from the stack (undo)."""
        if region_id in self.effect_stacks and self.effect_stacks[region_id]:
            return self.effect_stacks[region_id].pop()
        return None

    def has_effect_stack(self, region_id):
        """Check if region has any stacked effects."""
        return region_id in self.effect_stacks and len(self.effect_stacks[region_id]) > 0

    def load_video_texture(self, region_id, video_path):
        """Load a video file to use as texture for a region."""
        if region_id in self.video_textures:
            self.video_textures[region_id].release()

        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            self.video_textures[region_id] = cap
            self.region_effects[region_id] = self.region_effects.get(region_id, self.default_effect.copy())
            self.region_effects[region_id]['video_path'] = video_path
            self.region_effects[region_id]['effect_type'] = 'video_texture'
            return True
        return False

    def load_image_texture(self, region_id, image_path):
        """Load an image file to use as texture for a region."""
        img = cv2.imread(image_path)
        if img is not None:
            self.image_textures[region_id] = img
            self.region_effects[region_id] = self.region_effects.get(region_id, self.default_effect.copy())
            self.region_effects[region_id]['image_path'] = image_path
            self.region_effects[region_id]['effect_type'] = 'image_texture'
            return True
        return False

    def get_video_frame(self, region_id, target_size):
        """Get the next frame from a video texture, resized to target."""
        if region_id not in self.video_textures:
            return None

        cap = self.video_textures[region_id]
        ret, frame = cap.read()

        if not ret:
            # Loop video
            effect = self.region_effects.get(region_id, {})
            if effect.get('video_loop', True):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

        if ret and frame is not None:
            return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

        return None

    def get_image_texture(self, region_id, target_size):
        """Get image texture resized to target size."""
        if region_id not in self.image_textures:
            return None

        img = self.image_textures[region_id]
        return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    def bake_effect_texture(self, region_id, effected_frame, region_rect, padding=20):
        """
        Capture/bake the current effected region as a permanent texture.
        This stores a snapshot of the effect that persists independently of global effect changes.

        Args:
            region_id: ID like "face_1", "face_2", etc.
            effected_frame: The full frame WITH effects already applied
            region_rect: (x, y, w, h) or (x, y, w, h, id) of the region
            padding: Extra padding around the region
        """
        x, y, w, h = region_rect[:4]

        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(effected_frame.shape[1], x + w + padding)
        y2 = min(effected_frame.shape[0], y + h + padding)

        # Extract the effected region
        texture = effected_frame[y1:y2, x1:x2].copy()

        # Store it with metadata
        self.baked_textures[region_id] = {
            'texture': texture,
            'original_rect': (x, y, w, h),
            'padded_rect': (x1, y1, x2 - x1, y2 - y1),
            'padding': padding
        }

        # Set the effect type to use baked texture
        if region_id not in self.region_effects:
            self.region_effects[region_id] = self.default_effect.copy()
        self.region_effects[region_id]['effect_type'] = 'baked_texture'

        print(f"[BAKE] Stored texture for {region_id}: rect=({x},{y},{w},{h}), texture_shape={texture.shape}, total_baked={len(self.baked_textures)}")

        return True

    def clear_baked_texture(self, region_id):
        """Remove a baked texture for a region."""
        if region_id in self.baked_textures:
            del self.baked_textures[region_id]
        if region_id in self.region_effects:
            if self.region_effects[region_id].get('effect_type') == 'baked_texture':
                self.region_effects[region_id]['effect_type'] = 'none'

    def has_baked_texture(self, region_id):
        """Check if a region has a baked texture."""
        return region_id in self.baked_textures

    def get_baked_texture(self, region_id, target_rect):
        """Get baked texture resized/positioned for current detection rect."""
        if region_id not in self.baked_textures:
            return None

        baked = self.baked_textures[region_id]
        texture = baked['texture']
        padding = baked.get('padding', 20)

        # Target rect with padding
        x, y, w, h = target_rect[:4]
        target_w = w + 2 * padding
        target_h = h + 2 * padding

        # Resize texture to match current detection size
        return cv2.resize(texture, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def _apply_single_effect_to_roi(self, roi, effect, frame_number=0):
        """
        Apply a single effect to an ROI (region of interest).
        Returns the modified ROI.
        """
        effect_type = effect.get('effect_type', 'none')
        h, w = roi.shape[:2]
        h, w = int(h), int(w)  # Ensure integers for OpenCV

        if effect_type == 'none' or w <= 0 or h <= 0:
            return roi

        # Helper to safely get int values from effect dict
        def get_int(key, default):
            val = effect.get(key, default)
            return int(val) if val is not None else default

        def get_float(key, default):
            val = effect.get(key, default)
            return float(val) if val is not None else default

        # Apply the effect
        try:
            if effect_type == 'pixelate':
                pixel_size = max(2, get_int('pixelate_size', 10))
                small = cv2.resize(roi, (max(1, w // pixel_size), max(1, h // pixel_size)),
                                 interpolation=cv2.INTER_LINEAR)
                roi = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

            elif effect_type == 'blur':
                blur_amount = get_int('blur_amount', 31)
                blur_amount = blur_amount if blur_amount % 2 == 1 else blur_amount + 1
                roi = cv2.GaussianBlur(roi, (blur_amount, blur_amount), 0)

            elif effect_type == 'glitch':
                intensity = get_int('glitch_intensity', get_int('intensity', 50)) // 5
                if intensity > 0:
                    b, g, r = cv2.split(roi)
                    shift = np.random.randint(-intensity, intensity + 1)
                    r = np.roll(r, shift, axis=1)
                    b = np.roll(b, -shift, axis=1)
                    roi = cv2.merge([b, g, r])
                    for _ in range(3):
                        if h > 2:
                            row = np.random.randint(0, h - 1)
                            row_shift = np.random.randint(-intensity, intensity + 1)
                            roi[row:min(row+2, h)] = np.roll(roi[row:min(row+2, h)], row_shift, axis=1)

            elif effect_type == 'thermal':
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                contrast = cv2.equalizeHist(gray)
                roi = cv2.applyColorMap(contrast, cv2.COLORMAP_INFERNO)

            elif effect_type == 'cartoon':
                smooth = cv2.bilateralFilter(roi, 9, 75, 75)
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 5)
                edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY, 9, 9)
                quantized = (smooth // 32) * 32
                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                roi = cv2.bitwise_and(quantized, edges_colored)

            elif effect_type == 'color_shift':
                hue_shift = get_int('color_shift', 90)
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
                roi = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            elif effect_type == 'edge_highlight':
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                roi = cv2.addWeighted(roi, 0.7, edges_colored, 0.5, 0)

            elif effect_type == 'negative':
                roi = cv2.bitwise_not(roi)

            elif effect_type == 'posterize':
                levels = max(2, get_int('posterize_levels', 6))
                step = 256 // levels
                roi = (roi // step) * step

            elif effect_type == 'emboss':
                strength = get_float('emboss_strength', 50) / 50.0
                kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]) * strength
                roi = cv2.filter2D(roi, -1, kernel)
                roi = np.clip(roi + 128, 0, 255).astype(np.uint8)

            elif effect_type == 'sketch':
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                inverted = cv2.bitwise_not(gray)
                blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
                sketch = cv2.divide(gray, 255 - blurred, scale=256)
                roi = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

            elif effect_type == 'neon_glow':
                glow_size = get_int('glow_size', 5)
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
                blur_size = glow_size * 2 + 1
                glow = cv2.GaussianBlur(edges, (blur_size, blur_size), 0)
                glow_colored = np.zeros_like(roi)
                glow_colored[:, :, 1] = glow
                glow_colored[:, :, 0] = glow // 2
                roi = cv2.add(roi, glow_colored)
                roi = np.clip(roi, 0, 255).astype(np.uint8)

            elif effect_type == 'mirror':
                roi = cv2.flip(roi, 1)

            elif effect_type == 'wave_distort':
                amplitude = get_int('wave_amplitude', 20)
                frequency = get_int('wave_frequency', 5)
                for row_idx in range(h):
                    shift = int(np.sin(row_idx * 0.1 * frequency + frame_number * 0.1) * amplitude / 5)
                    roi[row_idx] = np.roll(roi[row_idx], shift, axis=0)

            elif effect_type == 'rgb_shift':
                offset = get_int('chromatic_offset', 5)
                b, g, r = cv2.split(roi)
                r = np.roll(r, offset, axis=1)
                b = np.roll(b, -offset, axis=1)
                roi = cv2.merge([b, g, r])

            elif effect_type == 'chromatic':
                offset = get_int('chromatic_offset', 5)
                b, g, r = cv2.split(roi)
                r = np.roll(r, offset, axis=1)
                b = np.roll(b, -offset, axis=1)
                roi = cv2.merge([b, g, r])

            elif effect_type == 'vhs':
                noise_intensity = get_int('noise_intensity', 25)
                noise = np.random.randn(h, w, 3) * noise_intensity
                roi = np.clip(roi.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                for row_idx in range(0, h, 2):
                    roi[row_idx] = (roi[row_idx] * 0.7).astype(np.uint8)

            elif effect_type == 'kaleidoscope':
                half_h, half_w = h // 2, w // 2
                if half_h > 0 and half_w > 0:
                    quadrant = roi[:half_h, :half_w]
                    roi[:half_h, half_w:half_w*2] = cv2.flip(quadrant, 1)
                    roi[half_h:half_h*2, :half_w] = cv2.flip(quadrant, 0)
                    roi[half_h:half_h*2, half_w:half_w*2] = cv2.flip(quadrant, -1)

            elif effect_type == 'oil_paint':
                roi = cv2.bilateralFilter(roi, 9, 75, 75)
                roi = cv2.bilateralFilter(roi, 9, 75, 75)

            elif effect_type == 'duotone':
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi[:, :, 0] = gray
                roi[:, :, 1] = (gray * 0.5).astype(np.uint8)
                roi[:, :, 2] = (255 - gray)

            elif effect_type == 'cross_process':
                roi = roi.astype(np.float32)
                roi[:, :, 0] = np.clip(roi[:, :, 0] * 1.2, 0, 255)
                roi[:, :, 1] = np.clip(roi[:, :, 1] * 0.9, 0, 255)
                roi[:, :, 2] = np.clip(roi[:, :, 2] * 1.1 + 20, 0, 255)
                roi = roi.astype(np.uint8)

            elif effect_type == 'heat_distort':
                intensity = get_int('heat_intensity', 8)
                for row_idx in range(h):
                    shift = int(np.sin(row_idx * 0.3 + frame_number * 0.2) * intensity)
                    roi[row_idx] = np.roll(roi[row_idx], shift, axis=0)

            elif effect_type == 'drunk':
                intensity = get_int('wobble_intensity', 15)
                for row_idx in range(h):
                    shift = int(np.sin(row_idx * 0.05 + frame_number * 0.1) * intensity)
                    roi[row_idx] = np.roll(roi[row_idx], shift, axis=0)

            elif effect_type == 'spiral_warp':
                strength = get_float('spiral_strength', 50) / 100.0
                center_x, center_y = w // 2, h // 2
                y_coords, x_coords = np.mgrid[0:h, 0:w]
                x_centered = x_coords - center_x
                y_centered = y_coords - center_y
                r = np.sqrt(x_centered**2 + y_centered**2)
                theta = np.arctan2(y_centered, x_centered)
                theta_new = theta + strength * r / max(w, h) * np.pi
                x_new = (center_x + r * np.cos(theta_new)).astype(np.float32)
                y_new = (center_y + r * np.sin(theta_new)).astype(np.float32)
                roi = cv2.remap(roi, x_new, y_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            elif effect_type == 'edge_glow':
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
                edges_colored = cv2.applyColorMap(edges, cv2.COLORMAP_HOT)
                roi = cv2.addWeighted(roi, 0.7, edges_colored, 0.3, 0)

            elif effect_type == 'halftone':
                dot_size = max(2, get_int('dot_size', 4))
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                result = np.zeros_like(roi)
                for row_idx in range(0, h, dot_size):
                    for col_idx in range(0, w, dot_size):
                        block = gray[row_idx:min(row_idx+dot_size, h), col_idx:min(col_idx+dot_size, w)]
                        if block.size > 0:
                            brightness = np.mean(block)
                            radius = int((brightness / 255) * (dot_size // 2))
                            if radius > 0:
                                cv2.circle(result, (col_idx + dot_size//2, row_idx + dot_size//2),
                                          radius, (int(brightness), int(brightness), int(brightness)), -1)
                roi = result

            elif effect_type == 'film_grain':
                intensity = get_int('grain_intensity', 30)
                noise = np.random.randn(h, w, 3) * intensity
                roi = np.clip(roi.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            elif effect_type == 'tv_static':
                blend = get_float('static_blend', 30) / 100.0
                static = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
                roi = cv2.addWeighted(roi, 1 - blend, static, blend, 0)

            elif effect_type == 'blocky_noise':
                chance = get_float('block_chance', 10) / 100.0
                block_size = 8
                for row_idx in range(0, h, block_size):
                    for col_idx in range(0, w, block_size):
                        if np.random.random() < chance:
                            color = np.random.randint(0, 256, 3)
                            roi[row_idx:min(row_idx+block_size, h),
                                col_idx:min(col_idx+block_size, w)] = color

            elif effect_type == 'scanlines':
                darkness = get_float('scanline_darkness', 40) / 100.0
                for row_idx in range(0, h, 2):
                    roi[row_idx] = (roi[row_idx] * (1 - darkness)).astype(np.uint8)

            elif effect_type == 'retro_crt':
                for row_idx in range(0, h, 2):
                    roi[row_idx] = (roi[row_idx] * 0.7).astype(np.uint8)
                b, g, r = cv2.split(roi)
                r = np.roll(r, 1, axis=1)
                b = np.roll(b, -1, axis=1)
                roi = cv2.merge([b, g, r])

            elif effect_type == 'ghost_trail':
                decay = get_float('trail_decay', 85) / 100.0
                kernel_size = int((1 - decay) * 30) + 1
                if kernel_size > 1:
                    kernel = np.zeros((kernel_size, kernel_size))
                    kernel[kernel_size//2, :] = 1.0 / kernel_size
                    roi = cv2.filter2D(roi, -1, kernel)

            elif effect_type == 'motion_blur':
                blur_amount = get_int('blur_amount', 15)
                kernel_size = max(3, blur_amount)
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[kernel_size//2, :] = 1.0 / kernel_size
                roi = cv2.filter2D(roi, -1, kernel)

            elif effect_type == 'radial_blur':
                strength = get_int('radial_strength', 10)
                center_x, center_y = w // 2, h // 2
                result = roi.copy().astype(np.float32)
                for i in range(1, min(strength, 10)):
                    scale = 1 - i * 0.01
                    M = cv2.getRotationMatrix2D((float(center_x), float(center_y)), 0, scale)
                    shifted = cv2.warpAffine(roi, M, (w, h))
                    result = cv2.addWeighted(result, 0.9, shifted.astype(np.float32), 0.1, 0)
                roi = result.astype(np.uint8)

            elif effect_type == 'zoom_blur':
                intensity = get_float('intensity', 50) / 100.0
                center_x, center_y = w // 2, h // 2
                result = roi.copy().astype(np.float32)
                for i in range(1, 10):
                    scale = 1 + i * intensity * 0.02
                    M = cv2.getRotationMatrix2D((float(center_x), float(center_y)), 0, scale)
                    zoomed = cv2.warpAffine(roi, M, (w, h))
                    result = cv2.addWeighted(result, 0.8, zoomed.astype(np.float32), 0.2, 0)
                roi = result.astype(np.uint8)

            elif effect_type == 'tunnel_vision':
                intensity = get_float('vignette_intensity', 70) / 100.0
                mask = np.zeros((h, w), dtype=np.float32)
                cv2.ellipse(mask, (w//2, h//2), (w//2, h//2), 0, 0, 360, 1.0, -1)
                mask = cv2.GaussianBlur(mask, (51, 51), 0)
                mask = (1 - intensity) + intensity * mask
                for c in range(3):
                    roi[:, :, c] = (roi[:, :, c] * mask).astype(np.uint8)

            elif effect_type == 'double_vision':
                offset = get_int('offset', 15)
                shifted = np.roll(roi, offset, axis=1)
                roi = cv2.addWeighted(roi, 0.5, shifted, 0.5, 0)

            elif effect_type == 'prism':
                offset = get_int('prism_offset', 8)
                b, g, r = cv2.split(roi)
                r = np.roll(r, offset, axis=1)
                r = np.roll(r, offset//2, axis=0)
                b = np.roll(b, -offset, axis=1)
                b = np.roll(b, -offset//2, axis=0)
                roi = cv2.merge([b, g, r])

            elif effect_type == 'rgb_split_radial':
                strength = get_int('split_strength', 10)
                center_x, center_y = w // 2, h // 2
                b, g, r = cv2.split(roi)
                y_coords, x_coords = np.mgrid[0:h, 0:w]
                dx = (x_coords - center_x).astype(np.float32) / w * strength
                dy = (y_coords - center_y).astype(np.float32) / h * strength
                map_x = (x_coords + dx).astype(np.float32)
                map_y = (y_coords + dy).astype(np.float32)
                r = cv2.remap(r, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                map_x = (x_coords - dx).astype(np.float32)
                map_y = (y_coords - dy).astype(np.float32)
                b = cv2.remap(b, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                roi = cv2.merge([b, g, r])

            elif effect_type == 'glitch_blocks':
                num_blocks = get_int('glitch_blocks', 8)
                intensity = get_int('glitch_intensity', 20)
                for _ in range(num_blocks):
                    if h > 10 and w > 10:
                        bh = np.random.randint(5, min(30, h//2))
                        bw = np.random.randint(10, min(50, w//2))
                        by = np.random.randint(0, h - bh)
                        bx = np.random.randint(0, w - bw)
                        shift = np.random.randint(-intensity, intensity + 1)
                        roi[by:by+bh, bx:bx+bw] = np.roll(roi[by:by+bh, bx:bx+bw], shift, axis=1)

            elif effect_type == 'glitch_shift':
                intensity = get_int('glitch_intensity', 20)
                num_lines = np.random.randint(3, 10)
                for _ in range(num_lines):
                    if h > 5:
                        line_y = np.random.randint(0, h - 3)
                        line_h = np.random.randint(1, min(5, h - line_y))
                        shift = np.random.randint(-intensity, intensity + 1)
                        roi[line_y:line_y+line_h] = np.roll(roi[line_y:line_y+line_h], shift, axis=1)

            elif effect_type == 'neon_edges':
                glow_size = get_int('glow_size', 5)
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
                blur_size = glow_size * 2 + 1
                glow = cv2.GaussianBlur(edges, (blur_size, blur_size), 0)
                glow_colored = cv2.applyColorMap(glow, cv2.COLORMAP_RAINBOW)
                roi = cv2.addWeighted(roi, 0.6, glow_colored, 0.4, 0)

            elif effect_type == 'ascii_art':
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                small_w, small_h = max(1, w//8), max(1, h//8)
                gray = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
                gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_NEAREST)
                roi = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            elif effect_type == 'slit_scan':
                result = roi.copy()
                for col in range(w):
                    shift = (col * 5) % h
                    result[:, col] = np.roll(roi[:, col], shift, axis=0)
                roi = result

            elif effect_type == 'color_swap':
                b, g, r = cv2.split(roi)
                roi = cv2.merge([r, b, g])

            elif effect_type == 'pulse_zoom':
                amount = get_float('pulse_zoom_amount', 3) / 100.0
                pulse = 1 + np.sin(frame_number * 0.2) * amount
                center_x, center_y = w // 2, h // 2
                M = cv2.getRotationMatrix2D((float(center_x), float(center_y)), 0, pulse)
                roi = cv2.warpAffine(roi, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            elif effect_type == 'time_echo':
                frames = get_int('time_echo_frames', 5)
                kernel_size = max(3, frames * 2 + 1)
                kernel = np.zeros((kernel_size, kernel_size))
                np.fill_diagonal(kernel, 1.0 / kernel_size)
                roi = cv2.filter2D(roi, -1, kernel)

            elif effect_type == 'digital_rain':
                overlay = roi.copy()
                num_lines = max(1, w // 20)
                for _ in range(num_lines):
                    x = np.random.randint(0, w)
                    length = np.random.randint(max(1, h // 4), h)
                    start_y = np.random.randint(0, max(1, h - length))
                    cv2.line(overlay, (x, start_y), (x, start_y + length), (0, 255, 0), 1)
                roi = cv2.addWeighted(roi, 0.7, overlay, 0.3, 0)

            elif effect_type == 'rotation':
                angle = frame_number * get_float('rotation_speed', 0.5)
                center = (float(w // 2), float(h // 2))
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                roi = cv2.warpAffine(roi, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            elif effect_type == 'zoom_punch':
                strength = get_float('zoom_punch_strength', 5) / 100.0
                center_x, center_y = w // 2, h // 2
                scale = 1 + strength
                M = cv2.getRotationMatrix2D((float(center_x), float(center_y)), 0, scale)
                roi = cv2.warpAffine(roi, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            elif effect_type == 'rgb_wave':
                intensity = get_int('rgb_wave_intensity', 12)
                b, g, r = cv2.split(roi)
                for row in range(h):
                    shift_r = int(np.sin(row * 0.1 + frame_number * 0.1) * intensity)
                    shift_b = int(np.sin(row * 0.1 + frame_number * 0.1 + np.pi) * intensity)
                    r[row] = np.roll(r[row], shift_r)
                    b[row] = np.roll(b[row], shift_b)
                roi = cv2.merge([b, g, r])

            elif effect_type == 'feedback_loop':
                decay = get_float('feedback_decay', 90) / 100.0
                center_x, center_y = w // 2, h // 2
                M = cv2.getRotationMatrix2D((float(center_x), float(center_y)), 0, 1.02)
                zoomed = cv2.warpAffine(roi, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                roi = cv2.addWeighted(roi, 1 - decay, zoomed, decay, 0)

            elif effect_type == 'color_drift':
                speed = get_float('color_drift_speed', 2) / 100.0
                hue_shift = int((frame_number * speed * 10) % 180)
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
                roi = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            elif effect_type == 'video_texture':
                # Apply video texture overlay
                texture_region_id = effect.get('texture_region_id', effect.get('region_id', 'face_0'))
                if texture_region_id in self.video_textures:
                    video_frame = self.get_video_frame(texture_region_id, (w, h))
                    if video_frame is not None:
                        blend_mode = effect.get('blend_mode', 'replace')
                        opacity = get_float('opacity', 1.0)
                        if blend_mode == 'replace':
                            roi = cv2.addWeighted(video_frame, opacity, roi, 1 - opacity, 0)
                        elif blend_mode == 'multiply':
                            blended = (roi.astype(np.float32) * video_frame.astype(np.float32) / 255).astype(np.uint8)
                            roi = cv2.addWeighted(blended, opacity, roi, 1 - opacity, 0)
                        elif blend_mode == 'screen':
                            inv_roi = 255 - roi
                            inv_video = 255 - video_frame
                            blended = 255 - (inv_roi.astype(np.float32) * inv_video.astype(np.float32) / 255).astype(np.uint8)
                            roi = cv2.addWeighted(blended, opacity, roi, 1 - opacity, 0)
                        elif blend_mode == 'overlay':
                            roi = cv2.addWeighted(roi, 0.5, video_frame, 0.5 * opacity, 0)
                        else:
                            roi = video_frame

            elif effect_type == 'image_texture':
                # Apply image texture overlay
                texture_region_id = effect.get('texture_region_id', effect.get('region_id', 'face_0'))
                if texture_region_id in self.image_textures:
                    image_texture = self.get_image_texture(texture_region_id, (w, h))
                    if image_texture is not None:
                        blend_mode = effect.get('blend_mode', 'replace')
                        opacity = get_float('opacity', 1.0)
                        if blend_mode == 'replace':
                            roi = cv2.addWeighted(image_texture, opacity, roi, 1 - opacity, 0)
                        elif blend_mode == 'multiply':
                            blended = (roi.astype(np.float32) * image_texture.astype(np.float32) / 255).astype(np.uint8)
                            roi = cv2.addWeighted(blended, opacity, roi, 1 - opacity, 0)
                        elif blend_mode == 'screen':
                            inv_roi = 255 - roi
                            inv_img = 255 - image_texture
                            blended = 255 - (inv_roi.astype(np.float32) * inv_img.astype(np.float32) / 255).astype(np.uint8)
                            roi = cv2.addWeighted(blended, opacity, roi, 1 - opacity, 0)
                        elif blend_mode == 'overlay':
                            roi = cv2.addWeighted(roi, 0.5, image_texture, 0.5 * opacity, 0)
                        else:
                            roi = image_texture

        except Exception as e:
            print(f"Error applying effect '{effect_type}': {e}")
            # Return original ROI on error
            pass

        return roi

    def apply_region_effect(self, frame, region_id, region_rect, frame_number=0, original_frame=None):
        """
        Apply all configured effects to a specific region (supports stacking).

        Args:
            frame: Full frame (will be modified in place)
            region_id: ID like "face_0", "eye_1", etc.
            region_rect: (x, y, w, h) of the region
            frame_number: Current frame number for animated effects
            original_frame: Original unmodified frame (for 'restore_original' effect)

        Returns:
            Modified frame
        """
        # Handle both 4 and 5 element tuples (5th element is detection ID)
        x, y, w, h = region_rect[:4]
        # Ensure bounds are valid
        x = max(0, x)
        y = max(0, y)
        h = min(h, frame.shape[0] - y)
        w = min(w, frame.shape[1] - x)

        if w <= 0 or h <= 0:
            return frame

        output = frame.copy()

        # Check for baked texture first - if baked, just apply that
        effect = self.get_region_effect(region_id)
        effect_type = effect.get('effect_type', 'none')

        # Special handling for restore_original - replaces region with original frame content
        if effect_type == 'restore_original' and original_frame is not None:
            # Get the original ROI and copy it to output, overriding any effects
            original_roi = original_frame[y:y+h, x:x+w].copy()

            # Optional: feathered edge for smoother blending
            feather = effect.get('feather', 10)
            if feather > 0 and min(w, h) > feather * 2:
                # Create feathered mask
                mask = np.ones((h, w), dtype=np.float32)
                for i in range(feather):
                    alpha = i / feather
                    mask[i, :] *= alpha
                    mask[-(i+1), :] *= alpha
                    mask[:, i] *= alpha
                    mask[:, -(i+1)] *= alpha
                mask_3ch = np.stack([mask, mask, mask], axis=2)

                # Blend original with current (feathered edges)
                current_roi = output[y:y+h, x:x+w].astype(np.float32)
                blended = (original_roi.astype(np.float32) * mask_3ch +
                          current_roi * (1 - mask_3ch))
                output[y:y+h, x:x+w] = blended.astype(np.uint8)
            else:
                # Hard replacement
                output[y:y+h, x:x+w] = original_roi
            return output

        if effect_type == 'baked_texture' and region_id in self.baked_textures:
            # Apply baked texture
            baked = self.baked_textures[region_id]
            texture = baked['texture']
            padding = baked.get('padding', 20)

            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            target_w = x2 - x1
            target_h = y2 - y1

            resized_texture = cv2.resize(texture, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # Feathered blending
            mask = np.ones((target_h, target_w), dtype=np.float32)
            feather = min(padding, 15)
            if feather > 0:
                for i in range(feather):
                    alpha = i / feather
                    mask[i, :] *= alpha
                    mask[-(i+1), :] *= alpha
                    mask[:, i] *= alpha
                    mask[:, -(i+1)] *= alpha

            mask_3ch = np.stack([mask, mask, mask], axis=2)
            region = output[y1:y2, x1:x2].astype(np.float32)
            blended = (resized_texture.astype(np.float32) * mask_3ch + region * (1 - mask_3ch))
            output[y1:y2, x1:x2] = blended.astype(np.uint8)
            return output

        # Get ROI
        roi = output[y:y+h, x:x+w].copy()

        # Apply effect stack first (previously applied effects)
        effect_stack = self.get_effect_stack(region_id)
        for stacked_effect in effect_stack:
            roi = self._apply_single_effect_to_roi(roi, stacked_effect, frame_number)

        # Then apply current live effect (if any)
        if effect_type != 'none' and effect_type != 'baked_texture':
            roi = self._apply_single_effect_to_roi(roi, effect, frame_number)

        # Put the processed ROI back
        output[y:y+h, x:x+w] = roi
        return output

    def _blend_texture(self, base, texture, opacity, blend_mode):
        """Blend texture onto base using specified mode."""
        if blend_mode == 'replace':
            return cv2.addWeighted(base, 1 - opacity, texture, opacity, 0)

        elif blend_mode == 'overlay':
            # Overlay blend mode
            base_f = base.astype(np.float32) / 255.0
            tex_f = texture.astype(np.float32) / 255.0
            mask = base_f < 0.5
            result = np.where(mask, 2 * base_f * tex_f, 1 - 2 * (1 - base_f) * (1 - tex_f))
            result = (result * 255).astype(np.uint8)
            return cv2.addWeighted(base, 1 - opacity, result, opacity, 0)

        elif blend_mode == 'multiply':
            result = (base.astype(np.float32) * texture.astype(np.float32) / 255).astype(np.uint8)
            return cv2.addWeighted(base, 1 - opacity, result, opacity, 0)

        elif blend_mode == 'screen':
            result = 255 - ((255 - base).astype(np.float32) * (255 - texture).astype(np.float32) / 255).astype(np.uint8)
            return cv2.addWeighted(base, 1 - opacity, result, opacity, 0)

        return texture if opacity >= 1.0 else cv2.addWeighted(base, 1 - opacity, texture, opacity, 0)

    def apply_all_region_effects(self, frame, detections_dict, frame_number=0, original_frame=None):
        """
        Apply effects to all detected regions.

        Args:
            frame: Full frame (with global effects already applied)
            detections_dict: Dict like {'faces': [...], 'eyes': [...], 'bodies': [...]}
            frame_number: Current frame for animated effects
            original_frame: Original unmodified frame (for 'restore_original' effect)

        Returns:
            Modified frame
        """
        output = frame.copy()

        # Helper to get region_id - use detection ID if available (5th element), else index
        def get_region_id(det_type, rect, index):
            if len(rect) >= 5:
                return f"{det_type}_{rect[4]}"  # Use tracking ID
            return f"{det_type}_{index}"

        # Process faces
        for i, rect in enumerate(detections_dict.get('faces', [])):
            region_id = get_region_id('face', rect, i)
            output = self.apply_region_effect(output, region_id, rect, frame_number, original_frame)

        # Process eyes
        for i, rect in enumerate(detections_dict.get('eyes', [])):
            region_id = get_region_id('eye', rect, i)
            output = self.apply_region_effect(output, region_id, rect, frame_number, original_frame)

        # Process bodies
        for i, rect in enumerate(detections_dict.get('bodies', [])):
            region_id = get_region_id('body', rect, i)
            output = self.apply_region_effect(output, region_id, rect, frame_number, original_frame)

        # Process upper bodies
        for i, rect in enumerate(detections_dict.get('upper_bodies', [])):
            region_id = get_region_id('upper_body', rect, i)
            output = self.apply_region_effect(output, region_id, rect, frame_number, original_frame)

        # Process smiles
        for i, rect in enumerate(detections_dict.get('smiles', [])):
            region_id = get_region_id('smile', rect, i)
            output = self.apply_region_effect(output, region_id, rect, frame_number, original_frame)

        # Process cat faces
        for i, rect in enumerate(detections_dict.get('cats', [])):
            region_id = get_region_id('cat', rect, i)
            output = self.apply_region_effect(output, region_id, rect, frame_number, original_frame)

        # Process license plates
        for i, rect in enumerate(detections_dict.get('plates', [])):
            region_id = get_region_id('plate', rect, i)
            output = self.apply_region_effect(output, region_id, rect, frame_number, original_frame)

        return output

    def apply_all_baked_textures(self, frame):
        """
        Apply ALL baked textures to the frame using their stored rect positions.
        This is used for rendering when we want to ensure all baked effects appear
        regardless of whether detection finds the same faces again.

        Args:
            frame: Input frame to apply baked textures to

        Returns:
            Frame with all baked textures applied
        """
        if not self.baked_textures:
            return frame

        output = frame.copy()

        for region_id, baked in self.baked_textures.items():
            try:
                texture = baked['texture']
                original_rect = baked.get('original_rect', (0, 0, 100, 100))
                padding = baked.get('padding', 20)

                x, y, w, h = original_rect

                # Calculate padded region
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                target_w = x2 - x1
                target_h = y2 - y1

                if target_w <= 0 or target_h <= 0:
                    continue

                # Resize baked texture to match target size
                resized_texture = cv2.resize(texture, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                # Create feathered mask for smooth blending
                mask = np.ones((target_h, target_w), dtype=np.float32)
                feather = min(padding, 15)
                if feather > 0:
                    for i in range(feather):
                        alpha = i / feather
                        mask[i, :] *= alpha
                        mask[-(i+1), :] *= alpha
                        mask[:, i] *= alpha
                        mask[:, -(i+1)] *= alpha

                mask_3ch = np.stack([mask, mask, mask], axis=2)

                # Blend baked texture with frame
                region = output[y1:y2, x1:x2].astype(np.float32)
                blended = (resized_texture.astype(np.float32) * mask_3ch + region * (1 - mask_3ch))
                output[y1:y2, x1:x2] = blended.astype(np.uint8)

            except Exception as e:
                print(f"Error applying baked texture for {region_id}: {e}")
                continue

        return output

    def cleanup(self):
        """Release all video captures."""
        for cap in self.video_textures.values():
            cap.release()
        self.video_textures.clear()
        self.image_textures.clear()

    def get_all_region_ids(self):
        """Get list of all configured region IDs."""
        return list(self.region_effects.keys())

    def reset_region(self, region_id):
        """Reset a region to default settings."""
        if region_id in self.region_effects:
            self.region_effects[region_id] = self.default_effect.copy()
        if region_id in self.video_textures:
            self.video_textures[region_id].release()
            del self.video_textures[region_id]
        if region_id in self.image_textures:
            del self.image_textures[region_id]


# Global region effect manager instance
_region_effect_manager = None


def get_region_effect_manager():
    """Get or create the global region effect manager."""
    global _region_effect_manager
    if _region_effect_manager is None:
        _region_effect_manager = RegionEffectManager()
    return _region_effect_manager


# ============================================
# CUSTOM MASK REGIONS (Draggable Effect Zones)
# ============================================

class CustomMaskRegion:
    """
    Represents a custom draggable mask region for selective effects.
    Can be rectangle, ellipse, or polygon shape.
    """

    def __init__(self, region_id, x, y, width, height, shape='rectangle'):
        self.region_id = region_id
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.shape = shape  # 'rectangle', 'ellipse', 'polygon'
        self.polygon_points = []  # For polygon shapes

        # Effect settings
        self.effect_type = 'none'
        self.effect_params = {
            'intensity': 50,
            'blur_amount': 31,
            'pixelate_size': 10,
            'color_shift': 90,
            'blend_opacity': 1.0,
        }

        # Visual settings
        self.border_color = (0, 255, 255)  # Cyan
        self.border_thickness = 2
        self.show_border = True
        self.invert_mask = False  # If True, effect applies OUTSIDE the region
        self.feather = 10

        # Dragging state
        self.is_selected = False
        self.drag_handle = None  # 'move', 'resize_tl', 'resize_br', etc.

    def get_rect(self):
        """Get bounding rectangle."""
        return (self.x, self.y, self.width, self.height)

    def set_rect(self, x, y, w, h):
        """Set bounding rectangle."""
        self.x = max(0, x)
        self.y = max(0, y)
        self.width = max(20, w)
        self.height = max(20, h)

    def contains_point(self, px, py):
        """Check if a point is inside this region."""
        if self.shape == 'rectangle':
            return (self.x <= px <= self.x + self.width and
                    self.y <= py <= self.y + self.height)
        elif self.shape == 'ellipse':
            cx = self.x + self.width / 2
            cy = self.y + self.height / 2
            rx = self.width / 2
            ry = self.height / 2
            if rx <= 0 or ry <= 0:
                return False
            return ((px - cx) ** 2 / rx ** 2 + (py - cy) ** 2 / ry ** 2) <= 1
        return False

    def get_resize_handle(self, px, py, handle_size=10):
        """Check if point is on a resize handle."""
        handles = {
            'resize_tl': (self.x, self.y),
            'resize_tr': (self.x + self.width, self.y),
            'resize_bl': (self.x, self.y + self.height),
            'resize_br': (self.x + self.width, self.y + self.height),
        }
        for handle_name, (hx, hy) in handles.items():
            if abs(px - hx) <= handle_size and abs(py - hy) <= handle_size:
                return handle_name
        return None

    def create_mask(self, frame_shape):
        """Create a mask for this region."""
        height, width = frame_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        if self.shape == 'rectangle':
            x1 = max(0, int(self.x))
            y1 = max(0, int(self.y))
            x2 = min(width, int(self.x + self.width))
            y2 = min(height, int(self.y + self.height))
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        elif self.shape == 'ellipse':
            cx = int(self.x + self.width / 2)
            cy = int(self.y + self.height / 2)
            rx = int(self.width / 2)
            ry = int(self.height / 2)
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

        elif self.shape == 'polygon' and len(self.polygon_points) >= 3:
            pts = np.array(self.polygon_points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

        # Apply feathering
        if self.feather > 0:
            kernel_size = self.feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

        if self.invert_mask:
            mask = 255 - mask

        return mask

    def draw_overlay(self, frame):
        """Draw the region border and handles on the frame."""
        if not self.show_border and not self.is_selected:
            return frame

        output = frame.copy()
        color = (0, 255, 0) if self.is_selected else self.border_color
        thickness = self.border_thickness + 1 if self.is_selected else self.border_thickness

        if self.shape == 'rectangle':
            cv2.rectangle(output,
                         (int(self.x), int(self.y)),
                         (int(self.x + self.width), int(self.y + self.height)),
                         color, thickness)
        elif self.shape == 'ellipse':
            cx = int(self.x + self.width / 2)
            cy = int(self.y + self.height / 2)
            cv2.ellipse(output, (cx, cy),
                       (int(self.width / 2), int(self.height / 2)),
                       0, 0, 360, color, thickness)

        # Draw resize handles if selected
        if self.is_selected:
            handle_size = 6
            handles = [
                (int(self.x), int(self.y)),
                (int(self.x + self.width), int(self.y)),
                (int(self.x), int(self.y + self.height)),
                (int(self.x + self.width), int(self.y + self.height)),
            ]
            for hx, hy in handles:
                cv2.rectangle(output,
                             (hx - handle_size, hy - handle_size),
                             (hx + handle_size, hy + handle_size),
                             (255, 255, 255), -1)
                cv2.rectangle(output,
                             (hx - handle_size, hy - handle_size),
                             (hx + handle_size, hy + handle_size),
                             color, 2)

        return output


class CustomMaskManager:
    """
    Manages multiple custom mask regions for selective effects.
    Provides interaction handling for dragging/resizing.
    """

    # Available effects for custom masks
    AVAILABLE_EFFECTS = [
        'none',
        'restore_original',  # Special: cuts through all effects, shows base image
        'blur', 'pixelate', 'glitch', 'thermal', 'negative',
        'cartoon', 'color_shift', 'edge_highlight', 'emboss', 'sketch',
        'neon_glow', 'posterize', 'vhs', 'wave_distort', 'kaleidoscope',
        'mirror', 'rgb_shift', 'grayscale', 'sepia', 'high_contrast',
    ]

    def __init__(self):
        self.regions = {}  # region_id -> CustomMaskRegion
        self.next_region_id = 1
        self.selected_region_id = None
        self.drag_start = None
        self.drag_offset = (0, 0)
        self.enabled = False

    def add_region(self, x, y, width=100, height=100, shape='rectangle'):
        """Add a new custom mask region."""
        region_id = f"custom_{self.next_region_id}"
        self.next_region_id += 1

        region = CustomMaskRegion(region_id, x, y, width, height, shape)
        self.regions[region_id] = region

        return region_id

    def remove_region(self, region_id):
        """Remove a region by ID."""
        if region_id in self.regions:
            del self.regions[region_id]
            if self.selected_region_id == region_id:
                self.selected_region_id = None

    def get_region(self, region_id):
        """Get a region by ID."""
        return self.regions.get(region_id)

    def get_all_regions(self):
        """Get all regions."""
        return list(self.regions.values())

    def select_region(self, region_id):
        """Select a region."""
        # Deselect current
        if self.selected_region_id and self.selected_region_id in self.regions:
            self.regions[self.selected_region_id].is_selected = False

        self.selected_region_id = region_id
        if region_id and region_id in self.regions:
            self.regions[region_id].is_selected = True

    def handle_mouse_down(self, x, y):
        """Handle mouse down event. Returns True if handled."""
        # Check if clicking on a resize handle of selected region
        if self.selected_region_id and self.selected_region_id in self.regions:
            region = self.regions[self.selected_region_id]
            handle = region.get_resize_handle(x, y)
            if handle:
                self.drag_start = (x, y)
                region.drag_handle = handle
                return True

        # Check if clicking inside any region
        for region_id, region in self.regions.items():
            if region.contains_point(x, y):
                self.select_region(region_id)
                self.drag_start = (x, y)
                self.drag_offset = (x - region.x, y - region.y)
                region.drag_handle = 'move'
                return True

        # Clicked outside all regions - deselect
        self.select_region(None)
        return False

    def handle_mouse_move(self, x, y):
        """Handle mouse move event during drag."""
        if not self.drag_start or not self.selected_region_id:
            return False

        region = self.regions.get(self.selected_region_id)
        if not region:
            return False

        if region.drag_handle == 'move':
            region.x = x - self.drag_offset[0]
            region.y = y - self.drag_offset[1]
            return True

        elif region.drag_handle == 'resize_br':
            region.width = max(20, x - region.x)
            region.height = max(20, y - region.y)
            return True

        elif region.drag_handle == 'resize_tl':
            new_x = min(x, region.x + region.width - 20)
            new_y = min(y, region.y + region.height - 20)
            region.width = region.width + (region.x - new_x)
            region.height = region.height + (region.y - new_y)
            region.x = new_x
            region.y = new_y
            return True

        elif region.drag_handle == 'resize_tr':
            new_y = min(y, region.y + region.height - 20)
            region.width = max(20, x - region.x)
            region.height = region.height + (region.y - new_y)
            region.y = new_y
            return True

        elif region.drag_handle == 'resize_bl':
            new_x = min(x, region.x + region.width - 20)
            region.width = region.width + (region.x - new_x)
            region.height = max(20, y - region.y)
            region.x = new_x
            return True

        return False

    def handle_mouse_up(self):
        """Handle mouse up event."""
        self.drag_start = None
        if self.selected_region_id and self.selected_region_id in self.regions:
            self.regions[self.selected_region_id].drag_handle = None

    def apply_all_masks(self, frame, apply_effect_func, frame_number=0, original_frame=None):
        """Apply all custom mask effects to the frame.

        Args:
            frame: Current frame (with global effects already applied)
            apply_effect_func: Function to apply effects (for compatibility)
            frame_number: Current frame number for animated effects
            original_frame: Original unmodified frame (for 'restore_original' effect)
        """
        if not self.enabled or not self.regions:
            return frame

        output = frame.copy()

        for region in self.regions.values():
            if region.effect_type == 'none':
                continue

            # Create mask for this region
            mask = region.create_mask(frame.shape)
            mask_3d = np.stack([mask / 255.0] * 3, axis=-1)

            # Special handling for restore_original - show the clean base image
            if region.effect_type == 'restore_original' and original_frame is not None:
                # Blend original frame through the mask, cutting through all effects
                output = (original_frame.astype(np.float32) * mask_3d +
                         output.astype(np.float32) * (1 - mask_3d)).astype(np.uint8)
            else:
                # Apply effect to entire frame
                effected_frame = self._apply_region_effect(
                    frame, region.effect_type, region.effect_params, frame_number
                )

                # Blend using mask
                output = (effected_frame * mask_3d + output * (1 - mask_3d)).astype(np.uint8)

        return output

    def _apply_region_effect(self, frame, effect_type, params, frame_number):
        """Apply a specific effect to a frame."""
        output = frame.copy()
        intensity = params.get('intensity', 50)

        if effect_type == 'blur':
            blur_amount = params.get('blur_amount', 31)
            blur_amount = blur_amount if blur_amount % 2 == 1 else blur_amount + 1
            output = cv2.GaussianBlur(output, (blur_amount, blur_amount), 0)

        elif effect_type == 'pixelate':
            pixel_size = max(2, params.get('pixelate_size', 10))
            h, w = output.shape[:2]
            small = cv2.resize(output, (max(1, w // pixel_size), max(1, h // pixel_size)),
                             interpolation=cv2.INTER_LINEAR)
            output = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        elif effect_type == 'glitch':
            b, g, r = cv2.split(output)
            shift = np.random.randint(-intensity // 5, intensity // 5 + 1)
            r = np.roll(r, shift, axis=1)
            b = np.roll(b, -shift, axis=1)
            output = cv2.merge([b, g, r])

        elif effect_type == 'thermal':
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            output = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

        elif effect_type == 'negative':
            output = cv2.bitwise_not(output)

        elif effect_type == 'cartoon':
            smooth = cv2.bilateralFilter(output, 9, 75, 75)
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 9)
            output = cv2.bitwise_and(smooth, smooth, mask=edges)

        elif effect_type == 'color_shift':
            hue_shift = params.get('color_shift', 90)
            hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        elif effect_type == 'edge_highlight':
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            output = cv2.addWeighted(output, 0.7, edges_colored, 0.5, 0)

        elif effect_type == 'emboss':
            kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            output = cv2.filter2D(output, -1, kernel)
            output = np.clip(output + 128, 0, 255).astype(np.uint8)

        elif effect_type == 'sketch':
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            inverted = cv2.bitwise_not(gray)
            blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
            sketch = cv2.divide(gray, 255 - blurred, scale=256)
            output = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

        elif effect_type == 'neon_glow':
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
            glow = cv2.GaussianBlur(edges, (15, 15), 0)
            glow_colored = np.zeros_like(output)
            glow_colored[:, :, 1] = glow
            glow_colored[:, :, 0] = glow // 2
            output = cv2.add(output, glow_colored)

        elif effect_type == 'posterize':
            levels = max(2, 8 - intensity // 15)
            step = 256 // levels
            output = (output // step) * step

        elif effect_type == 'vhs':
            noise = np.random.randn(*output.shape) * (intensity // 2)
            output = np.clip(output.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            for row_idx in range(0, output.shape[0], 2):
                output[row_idx] = (output[row_idx] * 0.7).astype(np.uint8)

        elif effect_type == 'wave_distort':
            h, w = output.shape[:2]
            for row_idx in range(h):
                shift = int(np.sin(row_idx * 0.1 + frame_number * 0.1) * (intensity // 5))
                output[row_idx] = np.roll(output[row_idx], shift, axis=0)

        elif effect_type == 'grayscale':
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        elif effect_type == 'sepia':
            kernel = np.array([[0.272, 0.534, 0.131],
                              [0.349, 0.686, 0.168],
                              [0.393, 0.769, 0.189]])
            output = cv2.transform(output, kernel)
            output = np.clip(output, 0, 255).astype(np.uint8)

        elif effect_type == 'high_contrast':
            output = cv2.convertScaleAbs(output, alpha=1.5, beta=0)

        elif effect_type == 'rgb_shift':
            b, g, r = cv2.split(output)
            shift = intensity // 10
            r = np.roll(r, shift, axis=1)
            b = np.roll(b, -shift, axis=1)
            output = cv2.merge([b, g, r])

        elif effect_type == 'mirror':
            output = cv2.flip(output, 1)

        elif effect_type == 'kaleidoscope':
            h, w = output.shape[:2]
            half_h, half_w = h // 2, w // 2
            if half_h > 0 and half_w > 0:
                quadrant = output[:half_h, :half_w]
                output[:half_h, half_w:half_w*2] = cv2.flip(quadrant, 1)
                output[half_h:half_h*2, :half_w] = cv2.flip(quadrant, 0)
                output[half_h:half_h*2, half_w:half_w*2] = cv2.flip(quadrant, -1)

        return output

    def draw_all_overlays(self, frame):
        """Draw all region overlays on the frame."""
        output = frame
        for region in self.regions.values():
            output = region.draw_overlay(output)
        return output

    def clear_all(self):
        """Remove all regions."""
        self.regions.clear()
        self.selected_region_id = None
        self.next_region_id = 1


# Global custom mask manager instance
_custom_mask_manager = None


def get_custom_mask_manager():
    """Get or create the global custom mask manager."""
    global _custom_mask_manager
    if _custom_mask_manager is None:
        _custom_mask_manager = CustomMaskManager()
    return _custom_mask_manager


# ============================================
# DATA VISUALIZATION EFFECTS
# (Detection overlays, connection lines, text, etc.)
# ============================================

class DataVisualization:
    """
    Advanced data visualization for detection overlays.
    Includes custom detection boxes, text, connection lines, and effects.
    """

    # Preset styles for detection boxes
    BOX_STYLES = {
        'default': {
            'color': (0, 255, 0),
            'thickness': 2,
            'corner_style': 'square',
            'fill_alpha': 0.0,
        },
        'cyberpunk': {
            'color': (255, 0, 255),
            'thickness': 2,
            'corner_style': 'tech',
            'fill_alpha': 0.1,
            'glow': True,
            'glow_color': (255, 100, 255),
        },
        'matrix': {
            'color': (0, 255, 0),
            'thickness': 1,
            'corner_style': 'bracket',
            'fill_alpha': 0.05,
            'scanline': True,
        },
        'hud': {
            'color': (0, 255, 255),
            'thickness': 2,
            'corner_style': 'tech',
            'fill_alpha': 0.0,
            'crosshair': True,
        },
        'thermal': {
            'color': (0, 100, 255),
            'thickness': 2,
            'corner_style': 'square',
            'fill_alpha': 0.15,
            'heat_glow': True,
        },
        'glitch': {
            'color': (255, 255, 0),
            'thickness': 2,
            'corner_style': 'glitch',
            'fill_alpha': 0.0,
            'rgb_shift': True,
        },
        'minimal': {
            'color': (255, 255, 255),
            'thickness': 1,
            'corner_style': 'corner_only',
            'fill_alpha': 0.0,
        },
        'neon': {
            'color': (0, 255, 255),
            'thickness': 2,
            'corner_style': 'rounded',
            'fill_alpha': 0.0,
            'glow': True,
            'glow_color': (0, 255, 255),
        },
    }

    def __init__(self):
        self.enabled = False
        self.box_style = 'default'
        self.custom_style = {}

        # Text settings
        self.show_labels = True
        self.label_font_scale = 0.6
        self.label_color = (255, 255, 255)
        self.label_bg_color = (0, 0, 0)
        self.label_bg_alpha = 0.7
        self.custom_labels = {}  # detection_type -> label text

        # Connection lines
        self.show_connections = False
        self.connection_style = 'line'  # 'line', 'dashed', 'dotted', 'curved', 'lightning'
        self.connection_color = (0, 255, 255)
        self.connection_thickness = 1
        self.connect_same_type = True  # Connect detections of same type
        self.connect_different_types = False  # Connect face to eyes, etc.

        # Data overlay
        self.show_coordinates = False
        self.show_confidence = False
        self.show_id = True
        self.show_center_point = False

        # Animation settings
        self.animate_boxes = False
        self.pulse_speed = 0.1
        self.scan_line_enabled = False
        self.scan_line_pos = 0

        # Effects
        self.corner_decoration = True
        self.info_panel_enabled = False
        self.info_panel_position = 'top_left'

        # === NEW: Per-detection customization ===
        # Stores individual settings per detection ID
        # Format: {det_id: {'enabled': True, 'style': 'cyberpunk', 'label': 'Custom', ...}}
        self.per_detection_settings = {}

        # Selected detections for operations (like connecting)
        self.selected_detections = set()  # Set of detection IDs

        # Manual connection pairs - list of (id1, id2) tuples
        self.manual_connections = []

        # Connection groups - each group has its own color
        # Format: [{'ids': [id1, id2, ...], 'color': (B, G, R), 'color_name': 'cyan'}, ...]
        self.connection_groups = []

        # Whether to use selective mode (only show viz for selected/configured detections)
        # Default to True so visualization only appears on selected faces
        self.selective_mode = True

        # Storage for selected detection rects - used for rendering
        # Format: {det_id: {'rect': (x, y, w, h), 'det_type': 'faces'}}
        self.selected_detection_rects = {}

    def get_detection_settings(self, det_id):
        """Get visualization settings for a specific detection ID."""
        if det_id not in self.per_detection_settings:
            self.per_detection_settings[det_id] = {
                'enabled': True,
                'style': None,  # None = use global style
                'custom_label': None,
                'show_label': None,  # None = use global setting
                'show_box': True,
                'custom_color': None,
            }
        return self.per_detection_settings[det_id]

    def set_detection_settings(self, det_id, **kwargs):
        """Set visualization settings for a specific detection."""
        settings = self.get_detection_settings(det_id)
        for key, value in kwargs.items():
            if key in settings:
                settings[key] = value

    def toggle_detection_viz(self, det_id):
        """Toggle visualization for a specific detection."""
        settings = self.get_detection_settings(det_id)
        settings['enabled'] = not settings['enabled']
        return settings['enabled']

    def is_detection_enabled(self, det_id):
        """Check if visualization is enabled for a detection."""
        if det_id not in self.per_detection_settings:
            return True  # Default to enabled
        return self.per_detection_settings[det_id].get('enabled', True)

    def select_detection(self, det_id, rect=None, det_type='faces'):
        """Add detection to selection and optionally store its rect for rendering."""
        self.selected_detections.add(det_id)
        if rect is not None:
            # Store the rect position for this detection (used in rendering)
            x, y, w, h = rect[:4]
            self.selected_detection_rects[det_id] = {
                'rect': (x, y, w, h),
                'det_type': det_type
            }

    def deselect_detection(self, det_id):
        """Remove detection from selection."""
        self.selected_detections.discard(det_id)
        # Also remove stored rect
        if det_id in self.selected_detection_rects:
            del self.selected_detection_rects[det_id]

    def toggle_selection(self, det_id, rect=None, det_type='faces'):
        """Toggle selection of a detection."""
        if det_id in self.selected_detections:
            self.selected_detections.discard(det_id)
            if det_id in self.selected_detection_rects:
                del self.selected_detection_rects[det_id]
            return False
        else:
            self.selected_detections.add(det_id)
            if rect is not None:
                x, y, w, h = rect[:4]
                self.selected_detection_rects[det_id] = {
                    'rect': (x, y, w, h),
                    'det_type': det_type
                }
            return True

    def update_detection_rect(self, det_id, rect, det_type='faces'):
        """Update the stored rect for a detection (call during live preview to keep positions current)."""
        if det_id in self.selected_detections:
            x, y, w, h = rect[:4]
            self.selected_detection_rects[det_id] = {
                'rect': (x, y, w, h),
                'det_type': det_type
            }

    def is_selected(self, det_id):
        """Check if a detection is selected."""
        return det_id in self.selected_detections

    def clear_selection(self):
        """Clear all selections."""
        self.selected_detections.clear()
        self.selected_detection_rects.clear()

    def get_selected_list(self):
        """Get list of selected detection IDs."""
        return list(self.selected_detections)

    def add_connection(self, id1, id2):
        """Add a manual connection between two detections."""
        if id1 != id2 and (id1, id2) not in self.manual_connections and (id2, id1) not in self.manual_connections:
            self.manual_connections.append((id1, id2))

    def remove_connection(self, id1, id2):
        """Remove a manual connection."""
        if (id1, id2) in self.manual_connections:
            self.manual_connections.remove((id1, id2))
        elif (id2, id1) in self.manual_connections:
            self.manual_connections.remove((id2, id1))

    def connect_selected(self):
        """Create connections between all selected detections."""
        selected_list = list(self.selected_detections)
        for i in range(len(selected_list)):
            for j in range(i + 1, len(selected_list)):
                self.add_connection(selected_list[i], selected_list[j])

    def clear_connections(self):
        """Clear all manual connections."""
        self.manual_connections.clear()

    def get_style_for_detection(self, det_id):
        """Get the appropriate style for a detection."""
        settings = self.per_detection_settings.get(det_id, {})
        custom_style = settings.get('style')
        if custom_style and custom_style in self.BOX_STYLES:
            return self.BOX_STYLES[custom_style]
        return self.BOX_STYLES.get(self.box_style, self.BOX_STYLES['default'])

    def draw_detection_box(self, frame, rect, detection_type, index, frame_number=0, det_id=None):
        """Draw a styled detection box with per-detection customization."""
        # Handle both 4-element (x,y,w,h) and 5-element (x,y,w,h,id) tuples
        if len(rect) >= 5:
            x, y, w, h = rect[:4]
            det_id = rect[4] if det_id is None else det_id
        else:
            x, y, w, h = rect

        # Check if this detection should be visualized
        if det_id is not None:
            if not self.is_detection_enabled(det_id):
                return frame
            if self.selective_mode and det_id not in self.selected_detections:
                if det_id not in self.per_detection_settings:
                    return frame

        # Get style - check per-detection settings first
        if det_id is not None and det_id in self.per_detection_settings:
            per_det = self.per_detection_settings[det_id]
            if per_det.get('style') and per_det['style'] in self.BOX_STYLES:
                style = self.BOX_STYLES[per_det['style']].copy()
            else:
                style = self.BOX_STYLES.get(self.box_style, self.BOX_STYLES['default']).copy()

            # Apply custom color if set
            if per_det.get('custom_color'):
                style['color'] = per_det['custom_color']
        else:
            style = self.BOX_STYLES.get(self.box_style, self.BOX_STYLES['default']).copy()

        style = {**style, **self.custom_style}  # Merge with global custom overrides

        color = style.get('color', (0, 255, 0))
        thickness = style.get('thickness', 2)
        corner_style = style.get('corner_style', 'square')
        fill_alpha = style.get('fill_alpha', 0.0)

        output = frame.copy()

        # Check if detection has a custom color set - this takes priority
        has_custom_color = False
        if det_id is not None and det_id in self.per_detection_settings:
            per_det = self.per_detection_settings[det_id]
            if per_det.get('custom_color'):
                color = per_det['custom_color']
                has_custom_color = True

        # Highlight selected detections (only if no custom color)
        if det_id is not None and det_id in self.selected_detections:
            if not has_custom_color:
                color = (0, 255, 255)  # Yellow for selected (default)
            thickness = thickness + 1

        # Animated pulse effect
        if self.animate_boxes:
            pulse = abs(np.sin(frame_number * self.pulse_speed))
            color = tuple(int(c * (0.7 + 0.3 * pulse)) for c in color)

        # Fill with alpha
        if fill_alpha > 0:
            overlay = output.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            output = cv2.addWeighted(overlay, fill_alpha, output, 1 - fill_alpha, 0)

        # Draw box based on corner style
        if corner_style == 'square':
            cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)

        elif corner_style == 'tech':
            # Tech corners with small lines
            corner_len = min(w, h) // 4
            # Top-left
            cv2.line(output, (x, y), (x + corner_len, y), color, thickness)
            cv2.line(output, (x, y), (x, y + corner_len), color, thickness)
            # Top-right
            cv2.line(output, (x + w, y), (x + w - corner_len, y), color, thickness)
            cv2.line(output, (x + w, y), (x + w, y + corner_len), color, thickness)
            # Bottom-left
            cv2.line(output, (x, y + h), (x + corner_len, y + h), color, thickness)
            cv2.line(output, (x, y + h), (x, y + h - corner_len), color, thickness)
            # Bottom-right
            cv2.line(output, (x + w, y + h), (x + w - corner_len, y + h), color, thickness)
            cv2.line(output, (x + w, y + h), (x + w, y + h - corner_len), color, thickness)

        elif corner_style == 'bracket':
            # Bracket style [ ]
            bracket_len = min(w, h) // 5
            # Left bracket
            cv2.line(output, (x, y), (x + bracket_len, y), color, thickness)
            cv2.line(output, (x, y), (x, y + h), color, thickness)
            cv2.line(output, (x, y + h), (x + bracket_len, y + h), color, thickness)
            # Right bracket
            cv2.line(output, (x + w - bracket_len, y), (x + w, y), color, thickness)
            cv2.line(output, (x + w, y), (x + w, y + h), color, thickness)
            cv2.line(output, (x + w - bracket_len, y + h), (x + w, y + h), color, thickness)

        elif corner_style == 'corner_only':
            # Only corners, no full edges
            corner_len = min(w, h) // 4
            corners = [
                ((x, y), (x + corner_len, y), (x, y + corner_len)),
                ((x + w, y), (x + w - corner_len, y), (x + w, y + corner_len)),
                ((x, y + h), (x + corner_len, y + h), (x, y + h - corner_len)),
                ((x + w, y + h), (x + w - corner_len, y + h), (x + w, y + h - corner_len)),
            ]
            for corner, h_end, v_end in corners:
                cv2.line(output, corner, h_end, color, thickness)
                cv2.line(output, corner, v_end, color, thickness)

        elif corner_style == 'glitch':
            # Glitchy box with random offsets
            offsets = [np.random.randint(-3, 4) for _ in range(4)]
            cv2.rectangle(output, (x + offsets[0], y), (x + w + offsets[1], y + h), color, thickness)
            # Add RGB shifted copies
            cv2.rectangle(output, (x + 2, y), (x + w + 2, y + h), (color[0], 0, 0), 1)
            cv2.rectangle(output, (x - 2, y), (x + w - 2, y + h), (0, 0, color[2]), 1)

        elif corner_style == 'rounded':
            # Rounded corners using ellipse arcs
            r = min(w, h) // 6
            # Draw edges
            cv2.line(output, (x + r, y), (x + w - r, y), color, thickness)
            cv2.line(output, (x + r, y + h), (x + w - r, y + h), color, thickness)
            cv2.line(output, (x, y + r), (x, y + h - r), color, thickness)
            cv2.line(output, (x + w, y + r), (x + w, y + h - r), color, thickness)
            # Draw rounded corners
            cv2.ellipse(output, (x + r, y + r), (r, r), 180, 0, 90, color, thickness)
            cv2.ellipse(output, (x + w - r, y + r), (r, r), 270, 0, 90, color, thickness)
            cv2.ellipse(output, (x + r, y + h - r), (r, r), 90, 0, 90, color, thickness)
            cv2.ellipse(output, (x + w - r, y + h - r), (r, r), 0, 0, 90, color, thickness)

        # Glow effect
        if style.get('glow', False):
            glow_color = style.get('glow_color', color)
            glow_mask = np.zeros(output.shape[:2], dtype=np.uint8)
            cv2.rectangle(glow_mask, (x, y), (x + w, y + h), 255, thickness + 4)
            glow_mask = cv2.GaussianBlur(glow_mask, (15, 15), 0)
            glow_overlay = np.zeros_like(output)
            glow_overlay[:, :, 0] = (glow_mask * glow_color[0] / 255).astype(np.uint8)
            glow_overlay[:, :, 1] = (glow_mask * glow_color[1] / 255).astype(np.uint8)
            glow_overlay[:, :, 2] = (glow_mask * glow_color[2] / 255).astype(np.uint8)
            output = cv2.add(output, glow_overlay)

        # Crosshair
        if style.get('crosshair', False):
            cx, cy = x + w // 2, y + h // 2
            cross_size = min(w, h) // 4
            cv2.line(output, (cx - cross_size, cy), (cx + cross_size, cy), color, 1)
            cv2.line(output, (cx, cy - cross_size), (cx, cy + cross_size), color, 1)
            cv2.circle(output, (cx, cy), 3, color, -1)

        # Center point
        if self.show_center_point:
            cx, cy = x + w // 2, y + h // 2
            cv2.circle(output, (cx, cy), 4, color, -1)

        return output

    def draw_label(self, frame, rect, detection_type, index, det_id=None):
        """Draw a label for a detection with per-detection customization."""
        # Handle both 4-element and 5-element tuples
        if len(rect) >= 5:
            x, y, w, h = rect[:4]
            det_id = rect[4] if det_id is None else det_id
        else:
            x, y, w, h = rect

        # Check per-detection settings for label visibility
        if det_id is not None:
            per_det = self.per_detection_settings.get(det_id, {})
            show_label = per_det.get('show_label')
            if show_label is False:
                return frame
            if show_label is None and not self.show_labels:
                return frame
            if not self.is_detection_enabled(det_id):
                return frame
        elif not self.show_labels:
            return frame

        output = frame.copy()

        # Get label text - check per-detection custom label first
        if det_id is not None:
            per_det = self.per_detection_settings.get(det_id, {})
            custom_label = per_det.get('custom_label')
            if custom_label:
                label = custom_label
            else:
                label = self.custom_labels.get(detection_type, detection_type.capitalize())
        else:
            label = self.custom_labels.get(detection_type, detection_type.capitalize())

        if self.show_id:
            if det_id is not None:
                label = f"{label} [ID:{det_id}]"
            else:
                label = f"{label} #{index + 1}"
        if self.show_coordinates:
            label += f" ({x},{y})"

        # Mark selected detections
        if det_id is not None and det_id in self.selected_detections:
            label = f"[SEL] {label}"

        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(label, font, self.label_font_scale, 1)

        # Draw background
        padding = 4
        bg_x1 = x
        bg_y1 = y - text_h - padding * 2 - 2
        bg_x2 = x + text_w + padding * 2
        bg_y2 = y - 2

        if bg_y1 < 0:  # If above frame, put below box
            bg_y1 = y + h + 2
            bg_y2 = y + h + text_h + padding * 2 + 2

        if self.label_bg_alpha > 0:
            overlay = output.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), self.label_bg_color, -1)
            output = cv2.addWeighted(overlay, self.label_bg_alpha, output, 1 - self.label_bg_alpha, 0)

        # Draw text
        text_y = bg_y2 - padding - baseline // 2
        cv2.putText(output, label, (bg_x1 + padding, text_y), font,
                   self.label_font_scale, self.label_color, 1, cv2.LINE_AA)

        return output

    def draw_connection_lines(self, frame, detections_dict, frame_number=0):
        """Draw connection lines between detections with manual connection support."""
        output = frame.copy()
        all_centers = {}  # det_id -> (cx, cy, det_type)

        # Collect all detection centers with their IDs
        for det_type, detections in detections_dict.items():
            for i, det in enumerate(detections):
                if len(det) >= 5:
                    x, y, w, h, det_id = det[:5]
                else:
                    x, y, w, h = det
                    det_id = f"{det_type}_{i}"

                cx, cy = x + w // 2, y + h // 2
                all_centers[det_id] = (cx, cy, det_type)

        # Track which connections have been drawn
        drawn_connections = set()

        # Draw connection groups first (with their specific colors)
        if hasattr(self, 'connection_groups'):
            for group in self.connection_groups:
                group_ids = group.get('ids', [])
                group_color = group.get('color', self.connection_color)

                # Connect all IDs in this group with the group's color
                for i, id1 in enumerate(group_ids):
                    for id2 in group_ids[i + 1:]:
                        if id1 in all_centers and id2 in all_centers:
                            conn_key = (min(id1, id2), max(id1, id2)) if isinstance(id1, int) and isinstance(id2, int) else (id1, id2)
                            if conn_key in drawn_connections:
                                continue

                            pt1 = (all_centers[id1][0], all_centers[id1][1])
                            pt2 = (all_centers[id2][0], all_centers[id2][1])
                            output = self._draw_connection(output, pt1, pt2, frame_number, color_override=group_color)
                            drawn_connections.add(conn_key)

        # Draw manual connections (with default connection color)
        for id1, id2 in self.manual_connections:
            if id1 in all_centers and id2 in all_centers:
                conn_key = (min(id1, id2), max(id1, id2)) if isinstance(id1, int) and isinstance(id2, int) else (id1, id2)
                if conn_key in drawn_connections:
                    continue
                pt1 = (all_centers[id1][0], all_centers[id1][1])
                pt2 = (all_centers[id2][0], all_centers[id2][1])
                output = self._draw_connection(output, pt1, pt2, frame_number)
                drawn_connections.add(conn_key)

        # Draw connections between SELECTED faces (when in selective mode or when faces are selected)
        # Only if they're not already part of a connection group
        # Uses the global connection_color setting
        if len(self.selected_detections) >= 2:
            selected_list = [det_id for det_id in self.selected_detections if det_id in all_centers]
            for i, id1 in enumerate(selected_list):
                for id2 in selected_list[i + 1:]:
                    # Skip if already drawn
                    conn_key = (min(id1, id2), max(id1, id2)) if isinstance(id1, int) and isinstance(id2, int) else (id1, id2)
                    if conn_key in drawn_connections:
                        continue

                    pt1 = (all_centers[id1][0], all_centers[id1][1])
                    pt2 = (all_centers[id2][0], all_centers[id2][1])
                    # Use the global connection color (can be changed via Apply Line Settings)
                    output = self._draw_connection(output, pt1, pt2, frame_number)
                    drawn_connections.add(conn_key)

        # Draw automatic connections if enabled (for non-selective mode or all faces)
        if self.show_connections and not self.selective_mode:
            center_list = list(all_centers.items())

            for i, (id1, (cx1, cy1, type1)) in enumerate(center_list):
                for j, (id2, (cx2, cy2, type2)) in enumerate(center_list[i + 1:], i + 1):
                    # Skip if already connected
                    conn_key = (min(id1, id2), max(id1, id2)) if isinstance(id1, int) and isinstance(id2, int) else (id1, id2)
                    if conn_key in drawn_connections:
                        continue

                    should_connect = False

                    if self.connect_same_type and type1 == type2:
                        should_connect = True
                    elif self.connect_different_types and type1 != type2:
                        should_connect = True

                    if should_connect:
                        output = self._draw_connection(output, (cx1, cy1), (cx2, cy2), frame_number)
                        drawn_connections.add(conn_key)

        return output

    def _draw_connection(self, frame, pt1, pt2, frame_number, color_override=None):
        """Draw a single connection line with style."""
        output = frame.copy()
        color = color_override if color_override else self.connection_color
        thickness = self.connection_thickness

        if self.connection_style == 'line':
            cv2.line(output, pt1, pt2, color, thickness)

        elif self.connection_style == 'dashed':
            # Dashed line
            dist = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
            if dist > 0:
                num_dashes = int(dist / 10)
                for k in range(0, num_dashes, 2):
                    t1 = k / num_dashes
                    t2 = min((k + 1) / num_dashes, 1)
                    p1 = (int(pt1[0] + t1 * (pt2[0] - pt1[0])),
                          int(pt1[1] + t1 * (pt2[1] - pt1[1])))
                    p2 = (int(pt1[0] + t2 * (pt2[0] - pt1[0])),
                          int(pt1[1] + t2 * (pt2[1] - pt1[1])))
                    cv2.line(output, p1, p2, color, thickness)

        elif self.connection_style == 'dotted':
            # Dotted line
            dist = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
            if dist > 0:
                num_dots = int(dist / 5)
                for k in range(num_dots):
                    t = k / num_dots
                    px = int(pt1[0] + t * (pt2[0] - pt1[0]))
                    py = int(pt1[1] + t * (pt2[1] - pt1[1]))
                    cv2.circle(output, (px, py), thickness, color, -1)

        elif self.connection_style == 'curved':
            # Curved bezier-like line
            mid_x = (pt1[0] + pt2[0]) // 2
            mid_y = (pt1[1] + pt2[1]) // 2 - 30  # Curve up
            pts = np.array([[pt1[0], pt1[1]], [mid_x, mid_y], [pt2[0], pt2[1]]], np.int32)
            cv2.polylines(output, [pts], False, color, thickness)

        elif self.connection_style == 'lightning':
            # Lightning bolt style
            num_segments = 5
            points = [pt1]
            for k in range(1, num_segments):
                t = k / num_segments
                px = int(pt1[0] + t * (pt2[0] - pt1[0]) + np.random.randint(-10, 11))
                py = int(pt1[1] + t * (pt2[1] - pt1[1]) + np.random.randint(-10, 11))
                points.append((px, py))
            points.append(pt2)
            for k in range(len(points) - 1):
                cv2.line(output, points[k], points[k + 1], color, thickness)

        return output

    def draw_info_panel(self, frame, detections_dict, frame_number=0):
        """Draw an info panel with detection statistics."""
        if not self.info_panel_enabled:
            return frame

        output = frame.copy()
        h, w = output.shape[:2]

        # Panel content
        lines = ["=== DETECTION DATA ==="]
        total = 0
        for det_type, detections in detections_dict.items():
            count = len(detections)
            total += count
            if count > 0:
                lines.append(f"{det_type.upper()}: {count}")
        lines.append(f"TOTAL: {total}")
        lines.append(f"FRAME: {frame_number}")

        # Calculate panel size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        padding = 10
        line_height = 20

        panel_w = 180
        panel_h = len(lines) * line_height + padding * 2

        # Position
        if self.info_panel_position == 'top_left':
            px, py = 10, 10
        elif self.info_panel_position == 'top_right':
            px, py = w - panel_w - 10, 10
        elif self.info_panel_position == 'bottom_left':
            px, py = 10, h - panel_h - 10
        else:  # bottom_right
            px, py = w - panel_w - 10, h - panel_h - 10

        # Draw panel background
        overlay = output.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (0, 0, 0), -1)
        output = cv2.addWeighted(overlay, 0.7, output, 0.3, 0)

        # Draw border
        cv2.rectangle(output, (px, py), (px + panel_w, py + panel_h), (0, 255, 255), 1)

        # Draw text
        for i, line in enumerate(lines):
            text_y = py + padding + (i + 1) * line_height - 5
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(output, line, (px + padding, text_y), font, font_scale, color, 1, cv2.LINE_AA)

        return output

    def apply_all(self, frame, detections_dict, frame_number=0):
        """Apply all visualization effects with per-detection customization."""
        if not self.enabled:
            return frame

        output = frame.copy()

        # Draw detection boxes and labels
        for det_type, detections in detections_dict.items():
            for i, det in enumerate(detections):
                # Extract detection ID if present
                if len(det) >= 5:
                    det_id = det[4]
                else:
                    det_id = None

                # In selective mode, ONLY draw for selected detections
                if self.selective_mode:
                    if det_id is None or det_id not in self.selected_detections:
                        continue  # Skip non-selected detections

                output = self.draw_detection_box(output, det, det_type, i, frame_number, det_id)
                output = self.draw_label(output, det, det_type, i, det_id)

        # Draw connection lines (includes both auto and manual connections)
        output = self.draw_connection_lines(output, detections_dict, frame_number)

        # Draw info panel (only show for selected if in selective mode)
        output = self.draw_info_panel(output, detections_dict, frame_number)

        # Scan line effect
        if self.scan_line_enabled:
            h = output.shape[0]
            self.scan_line_pos = (self.scan_line_pos + 3) % h
            cv2.line(output, (0, self.scan_line_pos), (output.shape[1], self.scan_line_pos),
                    (0, 255, 255), 1)

        return output

    def apply_for_render(self, frame, frame_number=0):
        """
        Apply visualization effects for rendering using stored detection positions.
        This bypasses the need to re-detect faces by using the rects stored when
        detections were selected during live preview.

        Args:
            frame: Frame to render visualization on
            frame_number: Current frame number

        Returns:
            Frame with visualization applied
        """
        if not self.enabled:
            return frame

        if not self.selected_detection_rects:
            # No stored rects - fall back to drawing nothing in selective mode
            # or everything if selective mode is off
            if self.selective_mode:
                return frame
            # If not in selective mode but no stored rects, nothing to draw
            return frame

        output = frame.copy()

        # Build a fake detections_dict from stored rects
        detections_dict_from_stored = {}

        for det_id, data in self.selected_detection_rects.items():
            rect = data['rect']
            det_type = data.get('det_type', 'faces')

            # Create detection tuple with tracking ID as 5th element
            det_with_id = (rect[0], rect[1], rect[2], rect[3], det_id)

            if det_type not in detections_dict_from_stored:
                detections_dict_from_stored[det_type] = []
            detections_dict_from_stored[det_type].append(det_with_id)

        # Now draw using the stored detections
        for det_type, detections in detections_dict_from_stored.items():
            for i, det in enumerate(detections):
                det_id = det[4] if len(det) >= 5 else None
                output = self.draw_detection_box(output, det, det_type, i, frame_number, det_id)
                output = self.draw_label(output, det, det_type, i, det_id)

        # Draw connection lines using stored positions
        output = self.draw_connection_lines(output, detections_dict_from_stored, frame_number)

        # Draw info panel
        output = self.draw_info_panel(output, detections_dict_from_stored, frame_number)

        # Scan line effect
        if self.scan_line_enabled:
            h = output.shape[0]
            self.scan_line_pos = (self.scan_line_pos + 3) % h
            cv2.line(output, (0, self.scan_line_pos), (output.shape[1], self.scan_line_pos),
                    (0, 255, 255), 1)

        return output


# Global data visualization instance
_data_viz = None


def get_data_visualization():
    """Get or create the global data visualization instance."""
    global _data_viz
    if _data_viz is None:
        _data_viz = DataVisualization()
    return _data_viz
