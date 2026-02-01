"""
Object Detection Module for Glitch Mirror
Provides face, body, and object detection with masking capabilities
for applying effects selectively to detected regions.
"""

import cv2
import numpy as np
import os

# Store cascade classifiers globally for performance
_face_cascade = None
_eye_cascade = None
_body_cascade = None
_profile_cascade = None


def get_cascade_path(cascade_name):
    """Get the path to OpenCV's built-in cascade files."""
    cv2_data_path = cv2.data.haarcascades
    return os.path.join(cv2_data_path, cascade_name)


def initialize_detectors():
    """Initialize all cascade classifiers."""
    global _face_cascade, _eye_cascade, _body_cascade, _profile_cascade

    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(get_cascade_path('haarcascade_frontalface_default.xml'))
    if _eye_cascade is None:
        _eye_cascade = cv2.CascadeClassifier(get_cascade_path('haarcascade_eye.xml'))
    if _body_cascade is None:
        _body_cascade = cv2.CascadeClassifier(get_cascade_path('haarcascade_fullbody.xml'))
    if _profile_cascade is None:
        _profile_cascade = cv2.CascadeClassifier(get_cascade_path('haarcascade_profileface.xml'))


def detect_faces(frame, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    """
    Detect faces in the frame using Haar cascades.
    Returns list of (x, y, w, h) tuples for each detected face.
    """
    initialize_detectors()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast for better detection

    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Also try to detect profile faces
    profile_faces = _profile_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Combine detections
    all_faces = list(faces) if len(faces) > 0 else []
    if len(profile_faces) > 0:
        all_faces.extend(list(profile_faces))

    return all_faces


def detect_eyes(frame, faces=None, scale_factor=1.1, min_neighbors=10):
    """
    Detect eyes in the frame. If faces are provided, only search within face regions.
    Returns list of (x, y, w, h) tuples for each detected eye.
    """
    initialize_detectors()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = []

    if faces is not None and len(faces) > 0:
        # Search for eyes only within face regions (upper half)
        for (fx, fy, fw, fh) in faces:
            roi_gray = gray[fy:fy + fh // 2, fx:fx + fw]
            detected_eyes = _eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(20, 20)
            )
            for (ex, ey, ew, eh) in detected_eyes:
                # Convert to global coordinates
                eyes.append((fx + ex, fy + ey, ew, eh))
    else:
        # Search entire frame
        detected_eyes = _eye_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(20, 20)
        )
        eyes = list(detected_eyes)

    return eyes


def detect_bodies(frame, scale_factor=1.05, min_neighbors=3, min_size=(50, 100)):
    """
    Detect full bodies in the frame.
    Returns list of (x, y, w, h) tuples for each detected body.
    """
    initialize_detectors()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bodies = _body_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return list(bodies) if len(bodies) > 0 else []


def create_detection_mask(frame_shape, detections, padding=0, feather=0):
    """
    Create a binary mask from detection rectangles.

    Args:
        frame_shape: (height, width, channels) of the frame
        detections: list of (x, y, w, h) rectangles
        padding: extra pixels to add around each detection
        feather: amount of edge blurring for smooth transitions

    Returns:
        Mask with same dimensions as frame (single channel, 0-255)
    """
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for (x, y, w, h) in detections:
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

    Args:
        frame_shape: (height, width, channels) of the frame
        detections: list of (x, y, w, h) rectangles
        padding: extra pixels to add around each detection
        feather: amount of edge blurring for smooth transitions

    Returns:
        Mask with same dimensions as frame (single channel, 0-255)
    """
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for (x, y, w, h) in detections:
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
    """
    output = frame.copy()

    for i, (x, y, w, h) in enumerate(detections):
        cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)

        if label:
            cv2.putText(output, f"{label} {i+1}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return output


# ============================================
# SPECIAL DETECTION-BASED EFFECTS
# ============================================

def face_pixelate(frame, faces, pixel_size=10):
    """
    Pixelate detected faces for anonymization effect.
    """
    output = frame.copy()

    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = output[y:y+h, x:x+w]

        # Pixelate by downscaling and upscaling
        small = cv2.resize(face_roi, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        # Put back
        output[y:y+h, x:x+w] = pixelated

    return output


def face_blur(frame, faces, blur_amount=51):
    """
    Blur detected faces.
    """
    output = frame.copy()

    for (x, y, w, h) in faces:
        face_roi = output[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(face_roi, (blur_amount, blur_amount), 0)
        output[y:y+h, x:x+w] = blurred

    return output


def face_glitch(frame, faces, glitch_intensity=20):
    """
    Apply glitch effect specifically to faces.
    """
    output = frame.copy()

    for (x, y, w, h) in faces:
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

    for (x, y, w, h) in faces:
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

    for (x, y, w, h) in faces:
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

    for (x, y, w, h) in faces:
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

    for (x, y, w, h) in faces:
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

    for (x, y, w, h) in faces:
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
    """

    def __init__(self, smoothing_frames=5, detection_threshold=0.5):
        self.history = []
        self.smoothing_frames = smoothing_frames
        self.detection_threshold = detection_threshold

    def update(self, detections):
        """
        Update tracker with new detections.
        Returns smoothed detections.
        """
        self.history.append(detections)

        if len(self.history) > self.smoothing_frames:
            self.history.pop(0)

        if len(self.history) < 2:
            return detections

        # Simple smoothing: average positions across history
        smoothed = []

        # Match detections across frames (simple IoU matching)
        if len(detections) > 0:
            for det in detections:
                x, y, w, h = det

                # Find similar detections in history
                similar_positions = [(x, y, w, h)]

                for past_detections in self.history[:-1]:
                    for past_det in past_detections:
                        px, py, pw, ph = past_det

                        # Check overlap (simple center distance check)
                        center_dist = np.sqrt((x + w/2 - px - pw/2)**2 + (y + h/2 - py - ph/2)**2)
                        max_dim = max(w, h, pw, ph)

                        if center_dist < max_dim:
                            similar_positions.append(past_det)
                            break

                # Average the similar positions
                if len(similar_positions) > 1:
                    avg_x = int(np.mean([p[0] for p in similar_positions]))
                    avg_y = int(np.mean([p[1] for p in similar_positions]))
                    avg_w = int(np.mean([p[2] for p in similar_positions]))
                    avg_h = int(np.mean([p[3] for p in similar_positions]))
                    smoothed.append((avg_x, avg_y, avg_w, avg_h))
                else:
                    smoothed.append(det)

        return smoothed

    def clear(self):
        """Clear tracking history."""
        self.history = []
