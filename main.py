import cv2
import numpy as np
import keyboard
import effects
import time
import tkinter as tk
from tkinter import filedialog, ttk, simpledialog, messagebox
import threading
import subprocess
import os
import shutil
import tempfile
import json
import copy
import sys

# Import panel modules
from effect_control_panel import EffectControlPanel
from audio_effects_panel import AudioEffectsPanel
from timeline_panel import TimelinePanel
from panel_utils import PRESETS_DIR, VIDEO_PRESETS_FILE, AUDIO_PRESETS_FILE, TIMELINE_FILE, PYGAME_AVAILABLE

print("ffmpeg path:", shutil.which("ffmpeg"))

# Track key states to detect single taps vs held keys
keyboard_state = {}

# Effect parameters dictionary - stores all adjustable parameters
effect_params = {
    'rgb_wave_intensity': 12,
    'posterize_levels': 6,
    'motion_smear_amount': 60,
    'feedback_decay_rate': 0.9,
    'glitch_slice_height': 12,
    'glitch_max_shift': 40,
    'zoom_punch_strength': 0.05,
    'chromatic_offset': 5,
    'vhs_noise': 25,
    'pixelate_size': 8,
    'kaleidoscope_segments': 6,
    'emboss_strength': 1.0,
    'radial_blur_strength': 0.02,
    'glitch_blocks_count': 8,
    'color_drift_speed': 0.02,
    'drunk_intensity': 15,
    'film_grain_intensity': 30,
    'tv_static_blend': 0.3,
    'wave_amplitude': 20,
    'ghost_trail_decay': 0.85,
    'tunnel_vignette': 0.7,
    'double_vision_offset': 15,
    'scanline_darkness': 0.4,
    'rgb_split_strength': 10,
    'halftone_dot_size': 4,
    'neon_glow_size': 5,
    'glitch_shift_intensity': 20,
    'heat_distort_intensity': 8,
    'pulse_zoom_amount': 0.03,
    'blocky_noise_chance': 0.1,
    'time_echo_frames': 5,
    'prism_offset': 8,
    'spiral_warp_strength': 0.5,
    'digital_rain_drops': 200,
    'rotation_speed': 0.5,  # Degrees per frame
}


def select_video_source():
    """
    Display a dialog to choose between webcam, video file, or image file.
    Returns a tuple: (source_type, source_path_or_index)
    source_type can be: 'webcam', 'file' (video), or 'image'
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Create a simple dialog
    dialog = tk.Toplevel(root)
    dialog.title("Select Media Source")
    dialog.geometry("400x180")
    dialog.resizable(False, False)

    # Center the dialog
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() - 400) // 2
    y = (dialog.winfo_screenheight() - 180) // 2
    dialog.geometry(f"400x180+{x}+{y}")

    result = {'source': None, 'path': None}

    def use_webcam():
        result['source'] = 'webcam'
        result['path'] = 0
        dialog.destroy()
        root.destroy()

    def use_video_file():
        dialog.destroy()
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            result['source'] = 'file'
            result['path'] = file_path
        else:
            result['source'] = 'webcam'
            result['path'] = 0
        root.destroy()

    def use_image_file():
        dialog.destroy()
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.webp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            result['source'] = 'image'
            result['path'] = file_path
        else:
            result['source'] = 'webcam'
            result['path'] = 0
        root.destroy()

    label = tk.Label(dialog, text="Choose media source:", font=("Arial", 12))
    label.pack(pady=15)

    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=10)

    webcam_btn = tk.Button(btn_frame, text="Webcam", command=use_webcam, width=12, height=2, bg="#4CAF50", fg="white")
    webcam_btn.pack(side=tk.LEFT, padx=8)

    file_btn = tk.Button(btn_frame, text="Video File", command=use_video_file, width=12, height=2, bg="#2196F3", fg="white")
    file_btn.pack(side=tk.LEFT, padx=8)

    image_btn = tk.Button(btn_frame, text="Image File", command=use_image_file, width=12, height=2, bg="#FF9800", fg="white")
    image_btn.pack(side=tk.LEFT, padx=8)

    hint_label = tk.Label(dialog, text="Select Video or Image to apply effects to static or moving content",
                          font=("Arial", 8), fg="gray")
    hint_label.pack(pady=10)

    dialog.protocol("WM_DELETE_WINDOW", use_webcam)  # Default to webcam if closed
    dialog.grab_set()
    root.wait_window(dialog)

    return result['source'], result['path']


def reset_all_effects():
    """Reset all effect flags to their default (off) state."""
    global alternate_sort_direction, current_mirror_mode, current_color_swap_mode
    global feedback_buffer, slit_scan_buffer, ghost_trail_buffer, time_echo_buffer
    global digital_rain_drops_state

    # Reset all effect states in dictionary
    for key in effect_states:
        effect_states[key] = False

    # Reset additional state variables
    alternate_sort_direction = False
    current_mirror_mode = 'horizontal'
    current_color_swap_mode = 'rgb_to_bgr'
    feedback_buffer = None
    slit_scan_buffer = None
    ghost_trail_buffer = None
    time_echo_buffer = None
    digital_rain_drops_state = None
    print(">>> ALL EFFECTS RESET <<<")


def was_key_just_pressed(key_name):
    """
    Check if a key was just pressed (rising edge detection).
    Returns True only on the first frame the key is pressed.
    """
    is_currently_pressed = keyboard.is_pressed(key_name)

    if key_name not in keyboard_state:
        keyboard_state[key_name] = is_currently_pressed
        return False

    was_pressed_last_frame = keyboard_state[key_name]

    if is_currently_pressed and not was_pressed_last_frame:
        keyboard_state[key_name] = is_currently_pressed
        return True

    keyboard_state[key_name] = is_currently_pressed
    return False


def apply_effects_to_frame(frame, prev_gray, frame_number, render_buffers):
    """
    Apply all enabled effects to a single frame.
    Used for both live preview and rendering.
    Returns: (output_frame, current_gray, updated_buffers)
    """
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output_frame = frame.copy()

    if prev_gray is None:
        prev_gray = current_gray.copy()

    # Compute motion mask
    motion_mask = effects.compute_motion_mask(current_gray, prev_gray, threshold=25, blur_kernel_size=5)

    # Background scanline effect
    blurred_motion_mask = cv2.GaussianBlur(motion_mask, (9, 9), 0)
    static_background_mask = blurred_motion_mask < 5

    if effect_states['background_effect_enabled']:
        background_copy = output_frame.copy()
        scanlined_background = background_copy.copy()
        frame_height = scanlined_background.shape[0]
        for row_index in range(0, frame_height, 3):
            scanlined_background[row_index:row_index + 1] = (
                scanlined_background[row_index:row_index + 1] * 0.6
            ).astype(np.uint8)
        output_frame[static_background_mask] = scanlined_background[static_background_mask]

    if effect_states['motion_visualization_enabled']:
        output_frame = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

    if effect_states['pixel_sort_enabled']:
        output_frame = effects.pixel_sort_horizontal(
            output_frame, motion_mask,
            brightness_threshold=50, minimum_segment_length=16
        )

    if effect_states['datamosh_enabled']:
        output_frame = effects.datamosh_vector(
            output_frame, prev_gray, motion_mask,
            displacement_strength=20
        )

    if effect_states['motion_mask_colormap_enabled']:
        output_frame = effects.show_motion_mask_effect(current_gray, prev_gray, output_frame)

    if effect_states['cycle_colormaps_enabled']:
        output_frame = effects.cycle_masks(current_gray, prev_gray, output_frame, 0, 0, 1.0)

    if effect_states['motion_smear_enabled']:
        output_frame = effects.motion_smear(output_frame, motion_mask, effect_params['motion_smear_amount'])

    if effect_states['glitch_slices_enabled']:
        output_frame = effects.glitch_slices(
            output_frame,
            slice_height=effect_params['glitch_slice_height'],
            max_horizontal_shift=effect_params['glitch_max_shift']
        )

    if effect_states['zoom_punch_enabled']:
        output_frame = effects.zoom_punch(output_frame, zoom_strength=effect_params['zoom_punch_strength'])

    if effect_states['posterize_enabled']:
        output_frame = effects.posterize(output_frame, color_levels=effect_params['posterize_levels'])

    if effect_states['feedback_loop_enabled']:
        render_buffers['feedback'] = effects.feedback_loop(
            output_frame, render_buffers.get('feedback'),
            effect_params['feedback_decay_rate']
        )
        output_frame = render_buffers['feedback']

    if effect_states['rgb_wave_enabled']:
        output_frame = effects.rgb_wave(output_frame, wave_amount=effect_params['rgb_wave_intensity'])

    if effect_states['edge_glow_enabled']:
        output_frame = effects.edge_glow(output_frame)

    if effect_states['chromatic_aberration_enabled']:
        output_frame = effects.chromatic_aberration(output_frame, channel_offset=effect_params['chromatic_offset'])

    if effect_states['vhs_enabled']:
        output_frame = effects.vhs_effect(output_frame, noise_intensity=effect_params['vhs_noise'])

    if effect_states['mirror_enabled']:
        output_frame = effects.mirror_effect(output_frame, current_mirror_mode)

    if effect_states['thermal_enabled']:
        output_frame = effects.thermal_vision(output_frame)

    if effect_states['negative_enabled']:
        output_frame = effects.negative(output_frame)

    if effect_states['pixelate_enabled']:
        output_frame = effects.pixelate(output_frame, pixel_block_size=effect_params['pixelate_size'])

    if effect_states['kaleidoscope_enabled']:
        output_frame = effects.kaleidoscope(output_frame, num_segments=effect_params['kaleidoscope_segments'])

    if effect_states['color_swap_enabled']:
        output_frame = effects.color_channel_swap(output_frame, current_color_swap_mode)

    if effect_states['emboss_enabled']:
        output_frame = effects.emboss(output_frame, emboss_strength=effect_params['emboss_strength'])

    if effect_states['radial_blur_enabled']:
        output_frame = effects.radial_blur(output_frame, blur_strength=effect_params['radial_blur_strength'])

    if effect_states['glitch_blocks_enabled']:
        output_frame = effects.glitch_blocks(output_frame, num_glitch_blocks=effect_params['glitch_blocks_count'])

    if effect_states['color_drift_enabled']:
        output_frame = effects.color_drift(output_frame, frame_number, drift_speed=effect_params['color_drift_speed'])

    if effect_states['slit_scan_enabled']:
        output_frame, render_buffers['slit_scan'] = effects.slit_scan(output_frame, render_buffers.get('slit_scan'))

    if effect_states['drunk_enabled']:
        output_frame = effects.drunk_effect(output_frame, frame_number, wobble_intensity=effect_params['drunk_intensity'])

    if effect_states['ascii_art_enabled']:
        output_frame = effects.ascii_art(output_frame)

    if effect_states['film_grain_enabled']:
        output_frame = effects.film_grain(output_frame, grain_intensity=effect_params['film_grain_intensity'])

    if effect_states['tv_static_enabled']:
        output_frame = effects.tv_static(output_frame, static_blend=effect_params['tv_static_blend'])

    if effect_states['wave_distort_enabled']:
        output_frame = effects.wave_distort(output_frame, frame_number, wave_amplitude=effect_params['wave_amplitude'])

    if effect_states['oil_paint_enabled']:
        output_frame = effects.oil_paint(output_frame)

    if effect_states['ghost_trail_enabled']:
        output_frame, render_buffers['ghost_trail'] = effects.ghost_trail(
            output_frame, render_buffers.get('ghost_trail'),
            fade_decay=effect_params['ghost_trail_decay']
        )

    if effect_states['tunnel_vision_enabled']:
        output_frame = effects.tunnel_vision(output_frame, vignette_intensity=effect_params['tunnel_vignette'])

    if effect_states['double_vision_enabled']:
        output_frame = effects.double_vision(output_frame, shift_offset=effect_params['double_vision_offset'])

    if effect_states['scanlines_enabled']:
        output_frame = effects.scanline_intensity(output_frame, scanline_darkness=effect_params['scanline_darkness'])

    if effect_states['rgb_split_radial_enabled']:
        output_frame = effects.rgb_split_radial(output_frame, split_strength=effect_params['rgb_split_strength'])

    if effect_states['sketch_enabled']:
        output_frame = effects.sketch_effect(output_frame)

    if effect_states['halftone_enabled']:
        output_frame = effects.halftone(output_frame, halftone_dot_size=effect_params['halftone_dot_size'])

    if effect_states['neon_edges_enabled']:
        output_frame = effects.neon_edges(output_frame, glow_blur_size=effect_params['neon_glow_size'])

    if effect_states['glitch_shift_enabled']:
        output_frame = effects.glitch_shift(output_frame, shift_intensity=effect_params['glitch_shift_intensity'])

    if effect_states['heat_distort_enabled']:
        output_frame = effects.heat_distort(output_frame, frame_number, distortion_intensity=effect_params['heat_distort_intensity'])

    if effect_states['cross_process_enabled']:
        output_frame = effects.cross_process(output_frame)

    if effect_states['duotone_enabled']:
        output_frame = effects.duotone(output_frame)

    if effect_states['pulse_zoom_enabled']:
        output_frame = effects.pulse_zoom(output_frame, frame_number, pulse_amount=effect_params['pulse_zoom_amount'])

    if effect_states['blocky_noise_enabled']:
        output_frame = effects.blocky_noise(output_frame, corruption_chance=effect_params['blocky_noise_chance'])

    if effect_states['retro_crt_enabled']:
        output_frame = effects.retro_crt(output_frame)

    if effect_states['time_echo_enabled']:
        output_frame, render_buffers['time_echo'] = effects.time_echo(
            output_frame, render_buffers.get('time_echo'),
            num_echo_frames=effect_params['time_echo_frames']
        )

    if effect_states['prism_enabled']:
        output_frame = effects.prism(output_frame, prism_offset=effect_params['prism_offset'])

    if effect_states['spiral_warp_enabled']:
        output_frame = effects.spiral_warp(output_frame, frame_number, warp_strength=effect_params['spiral_warp_strength'])

    if effect_states['digital_rain_enabled']:
        output_frame, render_buffers['digital_rain'] = effects.digital_rain(
            output_frame, render_buffers.get('digital_rain'),
            total_drops=effect_params['digital_rain_drops']
        )

    if effect_states['rotation_enabled']:
        output_frame = effects.rotate_frame(
            output_frame, frame_number,
            rotation_speed=effect_params['rotation_speed']
        )

    return output_frame, current_gray, render_buffers


# ----------------------------
# Media Source Management
# ----------------------------
# Global state for source switching
request_source_change = False
app_running = True


def initialize_media_source(selected_source_type, selected_source_path):
    """Initialize media source and return configuration dict."""
    config = {
        'source_type': selected_source_type,
        'source_path': selected_source_path,
        'is_video_file': False,
        'is_image_file': False,
        'image_frame': None,
        'camera': None,
        'frame_delay': 1,
        'total_video_frames': 0,
        'video_fps': 30,
        'window_title': "Glitch Mirror"
    }

    if selected_source_type == 'webcam':
        config['camera'] = cv2.VideoCapture(selected_source_path, cv2.CAP_DSHOW)
        config['camera'].set(cv2.CAP_PROP_FRAME_WIDTH, 500)
        config['camera'].set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
        config['frame_delay'] = 1
        config['window_title'] = "Glitch Mirror - Webcam"
        print("Using webcam")
    elif selected_source_type == 'image':
        config['image_frame'] = cv2.imread(selected_source_path)
        if config['image_frame'] is None:
            print(f"Error: Could not load image: {selected_source_path}")
            return None
        config['is_image_file'] = True
        config['frame_delay'] = 33
        config['total_video_frames'] = 1
        config['window_title'] = f"Glitch Mirror - Image: {os.path.basename(selected_source_path)}"
        print(f"Loaded image: {selected_source_path}")
        print(f"Dimensions: {config['image_frame'].shape[1]}x{config['image_frame'].shape[0]}")
    else:
        config['camera'] = cv2.VideoCapture(selected_source_path)
        config['is_video_file'] = True
        config['video_fps'] = config['camera'].get(cv2.CAP_PROP_FPS) or 30
        config['frame_delay'] = int(1000 / config['video_fps'])
        config['total_video_frames'] = int(config['camera'].get(cv2.CAP_PROP_FRAME_COUNT))
        config['window_title'] = f"Glitch Mirror - Video: {os.path.basename(selected_source_path)}"
        print(f"Loaded video: {selected_source_path}")
        print(f"FPS: {config['video_fps']}, Total frames: {config['total_video_frames']}")

    return config


def get_source_display_name(config):
    """Get a display name for the current source."""
    if config['is_image_file']:
        return f"Image: {os.path.basename(config['source_path'])}"
    elif config['is_video_file']:
        return f"Video: {os.path.basename(config['source_path'])}"
    else:
        return "Webcam"


def cleanup_media_source(config):
    """Release media source resources."""
    if config and config.get('camera'):
        try:
            config['camera'].release()
        except:
            pass
    cv2.destroyAllWindows()


def trigger_source_change():
    """Callback to trigger source change from control panel."""
    global request_source_change
    request_source_change = True


# Initial source selection
source_type, source_path = select_video_source()
media_config = initialize_media_source(source_type, source_path)

if media_config is None:
    print("Failed to initialize media source")
    sys.exit(1)

# Extract config to global variables for compatibility
is_video_file = media_config['is_video_file']
is_image_file = media_config['is_image_file']
image_frame = media_config['image_frame']
camera = media_config['camera']
frame_delay = media_config['frame_delay']
total_video_frames = media_config['total_video_frames']
video_fps = media_config['video_fps']
window_title = media_config['window_title']

# Video playback control
video_paused = False
video_loop = True  # Loop video by default

# ----------------------------
# Effect Toggle States (Dictionary for panel sync)
# ----------------------------
effect_states = {
    'motion_visualization_enabled': False,
    'glitch_slices_enabled': False,
    'zoom_punch_enabled': False,
    'feedback_loop_enabled': False,
    'rgb_wave_enabled': False,
    'motion_smear_enabled': False,
    'motion_mask_colormap_enabled': False,
    'cycle_colormaps_enabled': False,
    'datamosh_enabled': False,
    'pixel_sort_enabled': False,
    'posterize_enabled': False,
    'background_effect_enabled': False,
    'edge_glow_enabled': False,
    'chromatic_aberration_enabled': False,
    'vhs_enabled': False,
    'mirror_enabled': False,
    'thermal_enabled': False,
    'negative_enabled': False,
    'pixelate_enabled': False,
    'kaleidoscope_enabled': False,
    'color_swap_enabled': False,
    'emboss_enabled': False,
    'radial_blur_enabled': False,
    'glitch_blocks_enabled': False,
    'color_drift_enabled': False,
    'slit_scan_enabled': False,
    'drunk_enabled': False,
    'ascii_art_enabled': False,
    'film_grain_enabled': False,
    'tv_static_enabled': False,
    'wave_distort_enabled': False,
    'oil_paint_enabled': False,
    'ghost_trail_enabled': False,
    'tunnel_vision_enabled': False,
    'double_vision_enabled': False,
    'scanlines_enabled': False,
    'rgb_split_radial_enabled': False,
    'sketch_enabled': False,
    'halftone_enabled': False,
    'neon_edges_enabled': False,
    'glitch_shift_enabled': False,
    'heat_distort_enabled': False,
    'cross_process_enabled': False,
    'duotone_enabled': False,
    'pulse_zoom_enabled': False,
    'blocky_noise_enabled': False,
    'retro_crt_enabled': False,
    'time_echo_enabled': False,
    'prism_enabled': False,
    'spiral_warp_enabled': False,
    'digital_rain_enabled': False,
    'rotation_enabled': False,
}

# Additional state variables
previous_frame_grayscale = None
alternate_sort_direction = False
current_mirror_mode = 'horizontal'
current_color_swap_mode = 'rgb_to_bgr'
feedback_buffer = None
slit_scan_buffer = None
ghost_trail_buffer = None
time_echo_buffer = None
digital_rain_drops_state = None
live_preview_effects = True  # Toggle for showing effects in live preview

# ----------------------------
# Initialize Control Panel
# ----------------------------
control_panel = EffectControlPanel(effect_states, effect_params)
control_panel.create_panel()
control_panel.change_source_callback = trigger_source_change
control_panel.update_source_label(get_source_display_name(media_config))

# ----------------------------
# Initialize Audio Effects Panel (only for video files)
# ----------------------------
audio_panel = None
if is_video_file:
    audio_panel = AudioEffectsPanel()
    audio_panel.create_panel()
    audio_panel.set_video_source(source_path)

# ----------------------------
# Initialize Timeline Panel (only for video files)
# ----------------------------
timeline_panel = None
if is_video_file:
    timeline_panel = TimelinePanel(
        video_fps=video_fps,
        total_frames=total_video_frames,
        effect_states=effect_states,
        effect_params=effect_params
    )
    timeline_panel.create_panel()

# ----------------------------
# Image Mode Setup
# ----------------------------
if is_image_file:
    print("Image mode: Effects will be applied in real-time")
    print("Press ';' (semicolon) to save the current output with effects")
    print("Or click 'Render Video' button to save (it will prompt for image save)")
    print("Press 'Q' to quit")

# ----------------------------
# Rendering State
# ----------------------------
is_rendering = False
use_timeline_mode = False  # Toggle between manual effects and timeline-based effects


def save_image_with_effects():
    """Save the current image with effects applied."""
    if not is_image_file:
        return

    # Apply all effects to the image
    render_buffers = {}
    prev_gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    processed_frame, _, _ = apply_effects_to_frame(image_frame.copy(), prev_gray, total_frames_processed, render_buffers)

    # Ask for output file location
    output_path = filedialog.asksaveasfilename(
        title="Save Image with Effects",
        defaultextension=".png",
        filetypes=[
            ("PNG Image", "*.png"),
            ("JPEG Image", "*.jpg *.jpeg"),
            ("BMP Image", "*.bmp"),
            ("All files", "*.*")
        ],
        initialfile="output_with_effects.png"
    )

    if output_path:
        cv2.imwrite(output_path, processed_frame)
        print(f"Image saved to: {output_path}")
        control_panel.update_render_progress(100, f"Image saved: {output_path}")


def render_video():
    """Render the video with current effects at original framerate, preserving audio."""
    global is_rendering

    # Handle image export
    if is_image_file:
        save_image_with_effects()
        return

    if not is_video_file:
        control_panel.update_render_progress(0, "Error: Can only render video files, not webcam")
        print("Cannot render webcam feed - please load a video file")
        return

    if is_rendering:
        control_panel.update_render_progress(0, "Render already in progress...")
        return

    is_rendering = True

    # Ask for output file location
    output_path = filedialog.asksaveasfilename(
        title="Save Rendered Video",
        defaultextension=".mp4",
        filetypes=[
            ("MP4 Video", "*.mp4"),
            ("AVI Video", "*.avi"),
            ("All files", "*.*")
        ],
        initialfile="rendered_output.mp4"
    )

    if not output_path:
        is_rendering = False
        control_panel.update_render_progress(0, "Render cancelled")
        return

    print(f"Starting render to: {output_path}")
    control_panel.update_render_progress(0, "Starting render...")

    # Create a separate video capture for rendering
    render_cap = cv2.VideoCapture(source_path)

    if not render_cap.isOpened():
        control_panel.update_render_progress(0, "Error: Could not open video for rendering")
        is_rendering = False
        return

    # Get video properties
    original_width = int(render_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(render_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(render_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = render_cap.get(cv2.CAP_PROP_FPS) or 30

    # Output at original fps (no interpolation)
    output_fps = original_fps

    # Create temp file for video without audio
    temp_video_path = output_path + ".temp_video.mp4"

    # Setup video writer with H.264 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(temp_video_path, fourcc, output_fps, (original_width, original_height))

    if not out_writer.isOpened():
        control_panel.update_render_progress(0, "Error: Could not create output video")
        render_cap.release()
        is_rendering = False
        return

    # Initialize render buffers
    render_buffers = {}
    prev_gray = None
    frame_count = 0

    print(f"Rendering {total_frames} frames at {output_fps}fps (original framerate)")
    print(f"Resolution: {original_width}x{original_height}")
    if use_timeline_mode and timeline_panel:
        print("Using TIMELINE MODE for effects")
    else:
        print("Using CURRENT SETTINGS for effects")

    # Store original effect states/params to restore later if using timeline
    original_effect_states = copy.deepcopy(effect_states)
    original_effect_params = copy.deepcopy(effect_params)

    while True:
        ret, frame = render_cap.read()
        if not ret:
            break

        frame_count += 1

        # If using timeline mode, apply effects from keyframes at this time
        if use_timeline_mode and timeline_panel:
            current_time = frame_count / original_fps
            timeline_states, timeline_params = timeline_panel.get_active_effects_at_time(current_time)

            # Reset to original states first, then apply timeline overrides
            for key in effect_states:
                effect_states[key] = original_effect_states.get(key, False)
            for key in effect_params:
                effect_params[key] = original_effect_params.get(key, effect_params[key])

            # Apply timeline keyframe data
            if timeline_states:
                for key, value in timeline_states.items():
                    if key in effect_states:
                        effect_states[key] = value
            if timeline_params:
                for key, value in timeline_params.items():
                    if key in effect_params:
                        effect_params[key] = value

        # Apply effects to the current frame
        processed_frame, prev_gray, render_buffers = apply_effects_to_frame(
            frame, prev_gray, frame_count, render_buffers
        )

        # Write the processed frame
        out_writer.write(processed_frame)

        # Update progress
        progress = (frame_count / total_frames) * 100
        status = f"Rendering: {frame_count}/{total_frames} frames ({progress:.1f}%)"
        control_panel.update_render_progress(progress, status)

        # Allow GUI to update
        if frame_count % 10 == 0:
            control_panel.root.update()

    # Restore original effect states/params
    for key in effect_states:
        effect_states[key] = original_effect_states.get(key, False)
    for key in effect_params:
        effect_params[key] = original_effect_params.get(key, effect_params[key])

    # Cleanup video writing
    render_cap.release()
    out_writer.release()

    # Now merge audio using ffmpeg
    control_panel.update_render_progress(95, "Adding audio...")

    # Check if we have timeline audio tracks to mix
    timeline_audio_tracks = []
    if timeline_panel and use_timeline_mode:
        timeline_audio_tracks = timeline_panel.get_audio_events_for_render()

    # Check if we have processed audio from the audio panel
    processed_audio = None
    if audio_panel:
        processed_audio = audio_panel.get_audio_for_render()

    # Check if ffmpeg is available
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            video_duration = total_frames / original_fps

            if timeline_audio_tracks and len(timeline_audio_tracks) > 0:
                # Multi-track audio mixing from timeline
                print(f"Mixing {len(timeline_audio_tracks)} audio tracks from timeline...")
                control_panel.update_render_progress(96, f"Mixing {len(timeline_audio_tracks)} audio tracks...")

                # Build complex filter for mixing multiple audio tracks
                # First, create delayed/padded versions of each audio track
                inputs = ["-i", temp_video_path]
                filter_parts = []
                mix_inputs = []

                for i, track in enumerate(timeline_audio_tracks):
                    if os.path.exists(track['file_path']):
                        inputs.extend(["-i", track['file_path']])
                        input_idx = i + 1  # +1 because video is input 0

                        # Delay audio to start_time and adjust volume
                        delay_ms = int(track['start_time'] * 1000)
                        volume = track.get('volume', 1.0)

                        # Create filter for this track: delay and volume adjust
                        filter_parts.append(
                            f"[{input_idx}:a]"
                            f"atrim=start=0,"
                            f"asetpts=PTS-STARTPTS,"
                            f"adelay={delay_ms}|{delay_ms},"
                            f"volume={volume}"
                            f"[a{i}]"
                        )

                        mix_inputs.append(f"[a{i}]")

                if mix_inputs:
                    # Mix all audio tracks together
                    mix_filter = ";".join(filter_parts)
                    mix_filter += f";{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=longest:dropout_transition=0[aout]"

                    print("FFmpeg audio filter:", mix_filter)


                    ffmpeg_cmd = [
                        "ffmpeg", "-y",
                        *inputs,
                        "-filter_complex", mix_filter,
                        "-map", "0:v:0",
                        "-map", "[aout]",
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-shortest",
                        output_path
                    ]
                else:
                    # No valid audio tracks, just copy video
                    ffmpeg_cmd = [
                        "ffmpeg", "-y",
                        "-i", temp_video_path,
                        "-c:v", "copy",
                        output_path
                    ]

            elif processed_audio and os.path.exists(processed_audio):
                # Use processed audio from audio panel
                print(f"Using processed audio: {processed_audio}")
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", temp_video_path,
                    "-i", processed_audio,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-shortest",
                    output_path
                ]
            else:
                # Use original audio from video
                print("Using original audio from video...")
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", temp_video_path,
                    "-i", source_path,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-map", "0:v:0",
                    "-map", "1:a:0?",
                    "-shortest",
                    output_path
                ]

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                os.remove(temp_video_path)
                print("Audio merged successfully!")
            else:
                print(f"FFmpeg warning: {result.stderr}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_video_path, output_path)
                print("Saved without audio (ffmpeg merge failed)")

        except Exception as e:
            print(f"Error running ffmpeg: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_video_path, output_path)
            print("Saved without audio")
    else:
        print("FFmpeg not found - saving without audio")
        print("Install ffmpeg to preserve audio: https://ffmpeg.org/download.html")
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_video_path, output_path)

    is_rendering = False
    control_panel.update_render_progress(100, f"Done! Saved to {output_path}")
    print(f"Render complete! Output: {output_path}")
    print(f"Total frames: {frame_count}")


# Connect render callback to control panel
control_panel.render_callback = render_video

# ----------------------------
# Auto-Cycling State
# ----------------------------
auto_cycle_enabled = False
cycle_interval_seconds = 2.0
last_cycle_timestamp = time.time()

is_transitioning = False
transition_start_time = 0.0
transition_duration_seconds = 1.0
transition_from_preset = 0
transition_to_preset = 0



# ----------------------------
# Main Loop
# ----------------------------
current_colormap_preset = 0
total_frames_processed = 0

while app_running:
    # Check for source change request
    if request_source_change:
        request_source_change = False

        # Close existing panels (except control panel)
        if audio_panel:
            try:
                audio_panel.close()
            except:
                pass
            audio_panel = None

        if timeline_panel:
            try:
                timeline_panel.close()
            except:
                pass
            timeline_panel = None

        # Release current media source
        cleanup_media_source(media_config)

        # Show source selection dialog
        new_source_type, new_source_path = select_video_source()
        new_config = initialize_media_source(new_source_type, new_source_path)

        if new_config is None:
            print("Failed to initialize new media source, continuing with current")
            # Reinitialize old source
            new_config = initialize_media_source(source_type, source_path)
            if new_config is None:
                print("Critical error: cannot reinitialize source")
                break

        # Update global state
        media_config = new_config
        source_type = new_config['source_type']
        source_path = new_config['source_path']
        is_video_file = new_config['is_video_file']
        is_image_file = new_config['is_image_file']
        image_frame = new_config['image_frame']
        camera = new_config['camera']
        frame_delay = new_config['frame_delay']
        total_video_frames = new_config['total_video_frames']
        video_fps = new_config['video_fps']
        window_title = new_config['window_title']

        # Reset effect buffers
        previous_frame_grayscale = None
        feedback_buffer = None
        slit_scan_buffer = None
        ghost_trail_buffer = None
        time_echo_buffer = None
        digital_rain_drops_state = None

        # Update control panel source label
        control_panel.update_source_label(get_source_display_name(media_config))
        control_panel.update_render_progress(0, "Ready")

        # Initialize audio panel for video files
        if is_video_file:
            audio_panel = AudioEffectsPanel()
            audio_panel.create_panel()
            audio_panel.set_video_source(source_path)

            timeline_panel = TimelinePanel(
                video_fps=video_fps,
                total_frames=total_video_frames,
                effect_states=effect_states,
                effect_params=effect_params
            )
            timeline_panel.create_panel()

        # Print mode info
        if is_image_file:
            print("Image mode: Effects will be applied in real-time")
            print("Press ';' (semicolon) to save the current output with effects")

        # Reset frame counter
        total_frames_processed = 0
        continue

    total_frames_processed += 1

    # Handle image file mode
    if is_image_file:
        # For images, we always use the same source frame
        current_frame = image_frame.copy()
        frame_captured_successfully = True
    # Handle video file playback
    elif is_video_file and not video_paused:
        frame_captured_successfully, current_frame = camera.read()
        if not frame_captured_successfully:
            if video_loop:
                # Loop back to beginning
                camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_captured_successfully, current_frame = camera.read()
                if not frame_captured_successfully:
                    break
            else:
                break
    elif is_video_file and video_paused:
        # When paused, keep showing the last frame
        if 'current_frame' not in dir() or current_frame is None:
            frame_captured_successfully, current_frame = camera.read()
            if not frame_captured_successfully:
                break
        frame_captured_successfully = True
    else:
        # Webcam mode
        frame_captured_successfully, current_frame = camera.read()
        if not frame_captured_successfully:
            break

    current_frame_grayscale = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    output_frame = current_frame.copy()

    if previous_frame_grayscale is None:
        previous_frame_grayscale = current_frame_grayscale.copy()

    # ----------------------------
    # Motion Detection
    # ----------------------------
    motion_mask = effects.compute_motion_mask(
        current_frame_grayscale,
        previous_frame_grayscale,
        threshold=25,
        blur_kernel_size=5
    )

    # ----------------------------
    # Auto Preset Cycling Logic
    # ----------------------------
    current_time = time.time()
    transition_blend_factor = 1.0

    if is_transitioning:
        elapsed_transition_time = current_time - transition_start_time
        transition_blend_factor = elapsed_transition_time / transition_duration_seconds
        transition_blend_factor = np.clip(transition_blend_factor, 0.0, 1.0)
        # smooth step interpolation
        transition_blend_factor = transition_blend_factor * transition_blend_factor * (3 - 2 * transition_blend_factor)

        if transition_blend_factor >= 1.0:
            is_transitioning = False
            current_colormap_preset = transition_to_preset

    if auto_cycle_enabled and not is_transitioning:
        time_since_last_cycle = current_time - last_cycle_timestamp
        if time_since_last_cycle >= cycle_interval_seconds:
            transition_from_preset = current_colormap_preset
            transition_to_preset = (current_colormap_preset + 1) % 22
            is_transitioning = True
            transition_start_time = current_time
            last_cycle_timestamp = current_time

    # ----------------------------
    # Apply Effects Stack (using effect_states dict and effect_params)
    # Only apply if live preview is enabled (for performance)
    # ----------------------------

    # Toggle live preview with backtick key
    if was_key_just_pressed("`"):
        live_preview_effects = not live_preview_effects
        if control_panel.preview_effects_var:
            control_panel.preview_effects_var.set(live_preview_effects)
        print(f"Live preview effects: {'ON' if live_preview_effects else 'OFF'}")

    # Toggle timeline mode with Tab key
    if was_key_just_pressed("tab"):
        use_timeline_mode = not use_timeline_mode
        print(f"Timeline Mode: {'ON' if use_timeline_mode else 'OFF'}")
        # Update the control panel indicator
        if control_panel:
            control_panel.update_timeline_mode(use_timeline_mode)

    # Sync live_preview_effects with control panel checkbox
    if control_panel.preview_effects_var:
        live_preview_effects = control_panel.preview_effects_var.get()

    # If in timeline mode, apply effects from timeline keyframes
    if use_timeline_mode and timeline_panel and is_video_file:
        # Use actual video position (handles looping correctly)
        current_frame_pos = camera.get(cv2.CAP_PROP_POS_FRAMES)
        current_video_time = current_frame_pos / video_fps if video_fps > 0 else 0

        # Clamp to video duration to prevent going past end
        current_video_time = min(current_video_time, timeline_panel.total_duration)

        timeline_states, timeline_params = timeline_panel.get_active_effects_at_time(current_video_time)

        # Temporarily override effect_states and effect_params with timeline data
        if timeline_states:
            for key, value in timeline_states.items():
                effect_states[key] = value
        if timeline_params:
            for key, value in timeline_params.items():
                effect_params[key] = value

        # Sync timeline playhead with video position
        timeline_panel.current_time = current_video_time
        if int(current_frame_pos) % 5 == 0:  # Update UI every 5 frames for performance
            try:
                timeline_panel.update_time_display()
                timeline_panel.draw_timeline()
            except Exception:
                pass

    # Background scanline effect (applied to static areas)
    blurred_motion_mask = cv2.GaussianBlur(motion_mask, (9, 9), 0)
    static_background_mask = blurred_motion_mask < 5

    # Skip all effects processing if live preview is disabled (for performance)
    if live_preview_effects:
        if effect_states['background_effect_enabled']:
            background_copy = output_frame.copy()
            scanlined_background = background_copy.copy()
            frame_height = scanlined_background.shape[0]
            for row_index in range(0, frame_height, 3):
                scanlined_background[row_index:row_index + 1] = (
                    scanlined_background[row_index:row_index + 1] * 0.6
                ).astype(np.uint8)
            output_frame[static_background_mask] = scanlined_background[static_background_mask]

        if effect_states['motion_visualization_enabled']:
            output_frame = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

    if live_preview_effects and effect_states['pixel_sort_enabled']:
        if alternate_sort_direction:
            if total_frames_processed % 2 == 0:
                output_frame = effects.pixel_sort_horizontal(
                    output_frame, motion_mask,
                    brightness_threshold=50, minimum_segment_length=16
                )
            else:
                output_frame = effects.pixel_sort_vertical(
                    output_frame, motion_mask,
                    brightness_threshold=50, minimum_segment_length=16
                )
        else:
            output_frame = effects.pixel_sort_horizontal(
                output_frame, motion_mask,
                brightness_threshold=50, minimum_segment_length=16
            )

    if live_preview_effects and effect_states['datamosh_enabled']:
        output_frame = effects.datamosh_vector(
            output_frame, previous_frame_grayscale, motion_mask,
            displacement_strength=20
        )

    if live_preview_effects and effect_states['motion_mask_colormap_enabled']:
        output_frame = effects.show_motion_mask_effect(
            current_frame_grayscale, previous_frame_grayscale, output_frame
        )

    if live_preview_effects and effect_states['cycle_colormaps_enabled']:
        if is_transitioning:
            output_frame = effects.cycle_masks(
                current_frame_grayscale, previous_frame_grayscale, output_frame,
                transition_from_preset, transition_to_preset, transition_blend_factor
            )
        else:
            output_frame = effects.cycle_masks(
                current_frame_grayscale, previous_frame_grayscale, output_frame,
                current_colormap_preset, current_colormap_preset, 1.0
            )

    if live_preview_effects and effect_states['motion_smear_enabled']:
        output_frame = effects.motion_smear(
            output_frame, motion_mask, effect_params['motion_smear_amount']
        )

    if live_preview_effects and effect_states['glitch_slices_enabled']:
        output_frame = effects.glitch_slices(
            output_frame,
            slice_height=effect_params['glitch_slice_height'],
            max_horizontal_shift=effect_params['glitch_max_shift']
        )

    if live_preview_effects and effect_states['zoom_punch_enabled']:
        output_frame = effects.zoom_punch(
            output_frame,
            zoom_strength=effect_params['zoom_punch_strength']
        )

    if live_preview_effects and effect_states['posterize_enabled']:
        output_frame = effects.posterize(
            output_frame,
            color_levels=effect_params['posterize_levels']
        )

    if live_preview_effects and effect_states['feedback_loop_enabled']:
        feedback_buffer = effects.feedback_loop(
            output_frame, feedback_buffer,
            effect_params['feedback_decay_rate']
        )
        output_frame = feedback_buffer

    if live_preview_effects and effect_states['rgb_wave_enabled']:
        output_frame = effects.rgb_wave(
            output_frame,
            wave_amount=effect_params['rgb_wave_intensity']
        )

    if live_preview_effects and effect_states['edge_glow_enabled']:
        output_frame = effects.edge_glow(output_frame)

    if live_preview_effects and effect_states['chromatic_aberration_enabled']:
        output_frame = effects.chromatic_aberration(
            output_frame,
            channel_offset=effect_params['chromatic_offset']
        )

    if live_preview_effects and effect_states['vhs_enabled']:
        output_frame = effects.vhs_effect(
            output_frame,
            noise_intensity=effect_params['vhs_noise']
        )

    if live_preview_effects and effect_states['mirror_enabled']:
        output_frame = effects.mirror_effect(output_frame, current_mirror_mode)

    if live_preview_effects and effect_states['thermal_enabled']:
        output_frame = effects.thermal_vision(output_frame)

    if live_preview_effects and effect_states['negative_enabled']:
        output_frame = effects.negative(output_frame)

    if live_preview_effects and effect_states['pixelate_enabled']:
        output_frame = effects.pixelate(
            output_frame,
            pixel_block_size=effect_params['pixelate_size']
        )

    if live_preview_effects and effect_states['kaleidoscope_enabled']:
        output_frame = effects.kaleidoscope(
            output_frame,
            num_segments=effect_params['kaleidoscope_segments']
        )

    if live_preview_effects and effect_states['color_swap_enabled']:
        output_frame = effects.color_channel_swap(output_frame, current_color_swap_mode)

    if live_preview_effects and effect_states['emboss_enabled']:
        output_frame = effects.emboss(
            output_frame,
            emboss_strength=effect_params['emboss_strength']
        )

    if live_preview_effects and effect_states['radial_blur_enabled']:
        output_frame = effects.radial_blur(
            output_frame,
            blur_strength=effect_params['radial_blur_strength']
        )

    if live_preview_effects and effect_states['glitch_blocks_enabled']:
        output_frame = effects.glitch_blocks(
            output_frame,
            num_glitch_blocks=effect_params['glitch_blocks_count']
        )

    if live_preview_effects and effect_states['color_drift_enabled']:
        output_frame = effects.color_drift(
            output_frame, total_frames_processed,
            drift_speed=effect_params['color_drift_speed']
        )

    if live_preview_effects and effect_states['slit_scan_enabled']:
        output_frame, slit_scan_buffer = effects.slit_scan(output_frame, slit_scan_buffer)

    if live_preview_effects and effect_states['drunk_enabled']:
        output_frame = effects.drunk_effect(
            output_frame, total_frames_processed,
            wobble_intensity=effect_params['drunk_intensity']
        )

    if live_preview_effects and effect_states['ascii_art_enabled']:
        output_frame = effects.ascii_art(output_frame)

    if live_preview_effects and effect_states['film_grain_enabled']:
        output_frame = effects.film_grain(
            output_frame,
            grain_intensity=effect_params['film_grain_intensity']
        )

    if live_preview_effects and effect_states['tv_static_enabled']:
        output_frame = effects.tv_static(
            output_frame,
            static_blend=effect_params['tv_static_blend']
        )

    if live_preview_effects and effect_states['wave_distort_enabled']:
        output_frame = effects.wave_distort(
            output_frame, total_frames_processed,
            wave_amplitude=effect_params['wave_amplitude']
        )

    if live_preview_effects and effect_states['oil_paint_enabled']:
        output_frame = effects.oil_paint(output_frame)

    if live_preview_effects and effect_states['ghost_trail_enabled']:
        output_frame, ghost_trail_buffer = effects.ghost_trail(
            output_frame, ghost_trail_buffer,
            fade_decay=effect_params['ghost_trail_decay']
        )

    if live_preview_effects and effect_states['tunnel_vision_enabled']:
        output_frame = effects.tunnel_vision(
            output_frame,
            vignette_intensity=effect_params['tunnel_vignette']
        )

    if live_preview_effects and effect_states['double_vision_enabled']:
        output_frame = effects.double_vision(
            output_frame,
            shift_offset=effect_params['double_vision_offset']
        )

    if live_preview_effects and effect_states['scanlines_enabled']:
        output_frame = effects.scanline_intensity(
            output_frame,
            scanline_darkness=effect_params['scanline_darkness']
        )

    if live_preview_effects and effect_states['rgb_split_radial_enabled']:
        output_frame = effects.rgb_split_radial(
            output_frame,
            split_strength=effect_params['rgb_split_strength']
        )

    if live_preview_effects and effect_states['sketch_enabled']:
        output_frame = effects.sketch_effect(output_frame)

    if live_preview_effects and effect_states['halftone_enabled']:
        output_frame = effects.halftone(
            output_frame,
            halftone_dot_size=effect_params['halftone_dot_size']
        )

    if live_preview_effects and effect_states['neon_edges_enabled']:
        output_frame = effects.neon_edges(
            output_frame,
            glow_blur_size=effect_params['neon_glow_size']
        )

    if live_preview_effects and effect_states['glitch_shift_enabled']:
        output_frame = effects.glitch_shift(
            output_frame,
            shift_intensity=effect_params['glitch_shift_intensity']
        )

    if live_preview_effects and effect_states['heat_distort_enabled']:
        output_frame = effects.heat_distort(
            output_frame, total_frames_processed,
            distortion_intensity=effect_params['heat_distort_intensity']
        )

    if live_preview_effects and effect_states['cross_process_enabled']:
        output_frame = effects.cross_process(output_frame)

    if live_preview_effects and effect_states['duotone_enabled']:
        output_frame = effects.duotone(output_frame)

    if live_preview_effects and effect_states['pulse_zoom_enabled']:
        output_frame = effects.pulse_zoom(
            output_frame, total_frames_processed,
            pulse_amount=effect_params['pulse_zoom_amount']
        )

    if live_preview_effects and effect_states['blocky_noise_enabled']:
        output_frame = effects.blocky_noise(
            output_frame,
            corruption_chance=effect_params['blocky_noise_chance']
        )

    if live_preview_effects and effect_states['retro_crt_enabled']:
        output_frame = effects.retro_crt(output_frame)

    if live_preview_effects and effect_states['time_echo_enabled']:
        output_frame, time_echo_buffer = effects.time_echo(
            output_frame, time_echo_buffer,
            num_echo_frames=effect_params['time_echo_frames']
        )

    if live_preview_effects and effect_states['prism_enabled']:
        output_frame = effects.prism(
            output_frame,
            prism_offset=effect_params['prism_offset']
        )

    if live_preview_effects and effect_states['spiral_warp_enabled']:
        output_frame = effects.spiral_warp(
            output_frame, total_frames_processed,
            warp_strength=effect_params['spiral_warp_strength']
        )

    if live_preview_effects and effect_states['digital_rain_enabled']:
        output_frame, digital_rain_drops_state = effects.digital_rain(
            output_frame, digital_rain_drops_state,
            total_drops=effect_params['digital_rain_drops']
        )

    if live_preview_effects and effect_states['rotation_enabled']:
        output_frame = effects.rotate_frame(
            output_frame, total_frames_processed,
            rotation_speed=effect_params['rotation_speed']
        )

    # ----------------------------
    # Store Frame for Next Iteration
    # ----------------------------
    previous_frame_grayscale = current_frame_grayscale.copy()

    # ----------------------------
    # Keyboard Input Handling (syncs with control panel)
    # ----------------------------

    # Number keys for original effects
    if was_key_just_pressed("1"):
        effect_states['motion_mask_colormap_enabled'] = not effect_states['motion_mask_colormap_enabled']
    if was_key_just_pressed("2"):
        effect_states['glitch_slices_enabled'] = not effect_states['glitch_slices_enabled']
    if was_key_just_pressed("3"):
        effect_states['zoom_punch_enabled'] = not effect_states['zoom_punch_enabled']
    if was_key_just_pressed("4"):
        effect_states['feedback_loop_enabled'] = not effect_states['feedback_loop_enabled']
    if was_key_just_pressed("5"):
        effect_states['rgb_wave_enabled'] = not effect_states['rgb_wave_enabled']
    if was_key_just_pressed("6"):
        effect_states['posterize_enabled'] = not effect_states['posterize_enabled']
    if was_key_just_pressed("7"):
        effect_states['motion_smear_enabled'] = not effect_states['motion_smear_enabled']
    if was_key_just_pressed("8"):
        effect_states['cycle_colormaps_enabled'] = not effect_states['cycle_colormaps_enabled']
    if was_key_just_pressed("9"):
        effect_states['datamosh_enabled'] = not effect_states['datamosh_enabled']
    if was_key_just_pressed("0"):
        effect_states['ghost_trail_enabled'] = not effect_states['ghost_trail_enabled']
        if not effect_states['ghost_trail_enabled']:
            ghost_trail_buffer = None

    # Letter keys for effects
    if was_key_just_pressed("p"):
        effect_states['pixel_sort_enabled'] = not effect_states['pixel_sort_enabled']
    if was_key_just_pressed("/"):
        alternate_sort_direction = not alternate_sort_direction
        print(f"alternate_sort_direction = {alternate_sort_direction}")
    if was_key_just_pressed("c"):
        auto_cycle_enabled = not auto_cycle_enabled
        last_cycle_timestamp = time.time()
    if was_key_just_pressed("*"):
        effect_states['background_effect_enabled'] = not effect_states['background_effect_enabled']
        print(f"background_effect_enabled = {effect_states['background_effect_enabled']}")

    # Debug: visualize background mask when holding M
    if effect_states['background_effect_enabled'] and keyboard.is_pressed("m"):
        output_frame[static_background_mask] = (0, 255, 0)

    # Preset navigation
    if was_key_just_pressed("]"):
        current_colormap_preset = (current_colormap_preset + 1) % 22
    if was_key_just_pressed("["):
        current_colormap_preset = (current_colormap_preset - 1) % 22

    # Batch 1 effects
    if was_key_just_pressed("e"):
        effect_states['edge_glow_enabled'] = not effect_states['edge_glow_enabled']
    if was_key_just_pressed("a"):
        effect_states['chromatic_aberration_enabled'] = not effect_states['chromatic_aberration_enabled']
    if was_key_just_pressed("v"):
        effect_states['vhs_enabled'] = not effect_states['vhs_enabled']
    if was_key_just_pressed("r"):
        effect_states['mirror_enabled'] = not effect_states['mirror_enabled']
    if was_key_just_pressed("t"):
        # cycle through mirror modes
        available_mirror_modes = ['horizontal', 'vertical', 'quad']
        current_mode_index = available_mirror_modes.index(current_mirror_mode)
        current_mirror_mode = available_mirror_modes[(current_mode_index + 1) % len(available_mirror_modes)]
        print(f"current_mirror_mode = {current_mirror_mode}")
    if was_key_just_pressed("h"):
        effect_states['thermal_enabled'] = not effect_states['thermal_enabled']
    if was_key_just_pressed("n"):
        effect_states['negative_enabled'] = not effect_states['negative_enabled']
    if was_key_just_pressed("x"):
        effect_states['pixelate_enabled'] = not effect_states['pixelate_enabled']
    if was_key_just_pressed("k"):
        effect_states['kaleidoscope_enabled'] = not effect_states['kaleidoscope_enabled']
    if was_key_just_pressed("w"):
        effect_states['color_swap_enabled'] = not effect_states['color_swap_enabled']
    if was_key_just_pressed("s"):
        # cycle through color swap modes
        available_swap_modes = ['rgb_to_bgr', 'gbr', 'brg']
        current_swap_index = available_swap_modes.index(current_color_swap_mode)
        current_color_swap_mode = available_swap_modes[(current_swap_index + 1) % len(available_swap_modes)]
        print(f"current_color_swap_mode = {current_color_swap_mode}")
    if was_key_just_pressed("b"):
        effect_states['emboss_enabled'] = not effect_states['emboss_enabled']
    if was_key_just_pressed("l"):
        effect_states['radial_blur_enabled'] = not effect_states['radial_blur_enabled']

    # Batch 2 effects
    if was_key_just_pressed("g"):
        effect_states['glitch_blocks_enabled'] = not effect_states['glitch_blocks_enabled']
    if was_key_just_pressed("d"):
        effect_states['color_drift_enabled'] = not effect_states['color_drift_enabled']
    if was_key_just_pressed("z"):
        effect_states['slit_scan_enabled'] = not effect_states['slit_scan_enabled']
        if not effect_states['slit_scan_enabled']:
            slit_scan_buffer = None
    if was_key_just_pressed("u"):
        effect_states['drunk_enabled'] = not effect_states['drunk_enabled']
    if was_key_just_pressed("i"):
        effect_states['ascii_art_enabled'] = not effect_states['ascii_art_enabled']
    if was_key_just_pressed("f"):
        effect_states['film_grain_enabled'] = not effect_states['film_grain_enabled']
    if was_key_just_pressed("o"):
        effect_states['tv_static_enabled'] = not effect_states['tv_static_enabled']
    if was_key_just_pressed("y"):
        effect_states['wave_distort_enabled'] = not effect_states['wave_distort_enabled']
    if was_key_just_pressed("j"):
        effect_states['oil_paint_enabled'] = not effect_states['oil_paint_enabled']

    # Reset all effects
    if was_key_just_pressed("backspace"):
        reset_all_effects()

    # Function key effects (batch 3)
    if was_key_just_pressed("F1"):
        effect_states['tunnel_vision_enabled'] = not effect_states['tunnel_vision_enabled']
    if was_key_just_pressed("F2"):
        effect_states['double_vision_enabled'] = not effect_states['double_vision_enabled']
    if was_key_just_pressed("F3"):
        effect_states['scanlines_enabled'] = not effect_states['scanlines_enabled']
    if was_key_just_pressed("F4"):
        effect_states['rgb_split_radial_enabled'] = not effect_states['rgb_split_radial_enabled']
    if was_key_just_pressed("F5"):
        effect_states['sketch_enabled'] = not effect_states['sketch_enabled']
    if was_key_just_pressed("F6"):
        effect_states['halftone_enabled'] = not effect_states['halftone_enabled']
    if was_key_just_pressed("F7"):
        effect_states['neon_edges_enabled'] = not effect_states['neon_edges_enabled']
    if was_key_just_pressed("F8"):
        effect_states['glitch_shift_enabled'] = not effect_states['glitch_shift_enabled']
    if was_key_just_pressed("F9"):
        effect_states['heat_distort_enabled'] = not effect_states['heat_distort_enabled']
    if was_key_just_pressed("F10"):
        effect_states['cross_process_enabled'] = not effect_states['cross_process_enabled']
    if was_key_just_pressed("F11"):
        effect_states['duotone_enabled'] = not effect_states['duotone_enabled']
    if was_key_just_pressed("F12"):
        effect_states['pulse_zoom_enabled'] = not effect_states['pulse_zoom_enabled']

    # Special key effects
    if was_key_just_pressed("home"):
        effect_states['blocky_noise_enabled'] = not effect_states['blocky_noise_enabled']
    if was_key_just_pressed("end"):
        effect_states['retro_crt_enabled'] = not effect_states['retro_crt_enabled']
    if was_key_just_pressed("insert"):
        effect_states['time_echo_enabled'] = not effect_states['time_echo_enabled']
        if not effect_states['time_echo_enabled']:
            time_echo_buffer = None
    if was_key_just_pressed("delete"):
        effect_states['prism_enabled'] = not effect_states['prism_enabled']
    if was_key_just_pressed("page up"):
        effect_states['spiral_warp_enabled'] = not effect_states['spiral_warp_enabled']
    if was_key_just_pressed("page down"):
        effect_states['digital_rain_enabled'] = not effect_states['digital_rain_enabled']
        if not effect_states['digital_rain_enabled']:
            digital_rain_drops_state = None

    # Rotation effect
    if was_key_just_pressed("q"):
        effect_states['rotation_enabled'] = not effect_states['rotation_enabled']
        print(f"Rotation: {'ON' if effect_states['rotation_enabled'] else 'OFF'}")

    # Audio Panel Hotkeys
    if audio_panel:
        if was_key_just_pressed("+"):
            audio_panel.preview_effects()
        if was_key_just_pressed("-"):
            audio_panel.stop_audio()

    # Image Mode: Save with ';' key (since Ctrl+S may conflict)
    if is_image_file and was_key_just_pressed(";"):
        save_image_with_effects()


    # Video file playback controls
    if is_video_file:
        # Space to pause/unpause
        if was_key_just_pressed("space"):
            video_paused = not video_paused
            print(f"Video {'paused' if video_paused else 'playing'}")

        # Left/Right arrows to seek
        if was_key_just_pressed("left"):
            current_pos = camera.get(cv2.CAP_PROP_POS_FRAMES)
            new_pos = max(0, current_pos - 30)  # Go back ~1 second
            camera.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            print(f"Seeking to frame {int(new_pos)}")

        if was_key_just_pressed("right"):
            current_pos = camera.get(cv2.CAP_PROP_POS_FRAMES)
            new_pos = min(total_video_frames - 1, current_pos + 30)  # Skip forward ~1 second
            camera.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            print(f"Seeking to frame {int(new_pos)}")

        # L to toggle loop
        if was_key_just_pressed("\\"):
            video_loop = not video_loop
            print(f"Video loop: {'ON' if video_loop else 'OFF'}")

        # R to restart video
        if was_key_just_pressed("'"):
            camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            print("Video restarted")

    # Update control panel GUI (with exception protection)
    try:
        control_panel.update()
    except Exception:
        pass

    # Update audio panel GUI (if exists)
    try:
        if audio_panel:
            audio_panel.update()
    except Exception:
        pass

    # Update timeline panel GUI (if exists)
    try:
        if timeline_panel:
            timeline_panel.update()
    except Exception:
        pass

    # Quit application
    try:
        # Use frame_delay for video/image, 1 for webcam (for responsiveness)
        wait_time = frame_delay if (is_video_file or is_image_file) else 1
        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord("q"):
            app_running = False
            break
    except Exception:
        break

    # Apply preview scaling for performance
    preview_scale = control_panel.preview_scale_var.get() if hasattr(control_panel, 'preview_scale_var') else 1.0
    if preview_scale < 1.0:
        display_height = int(output_frame.shape[0] * preview_scale)
        display_width = int(output_frame.shape[1] * preview_scale)
        display_frame = cv2.resize(output_frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
    else:
        display_frame = output_frame

    cv2.imshow(window_title, display_frame)

# ----------------------------
# Cleanup
# ----------------------------
try:
    control_panel.close()
except Exception:
    pass
try:
    if audio_panel:
        audio_panel.close()
except Exception:
    pass
try:
    if timeline_panel:
        timeline_panel.close()
except Exception:
    pass
try:
    cleanup_media_source(media_config)
except Exception:
    pass
