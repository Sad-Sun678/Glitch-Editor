import cv2
import numpy as np
import keyboard
import effects
import time
import tkinter as tk
from tkinter import filedialog, ttk
import threading
import subprocess
import os
import shutil
from effects import change_knob

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
}


def select_video_source():
    """
    Display a dialog to choose between webcam and video file.
    Returns a tuple: (source_type, source_path_or_index)
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Create a simple dialog
    dialog = tk.Toplevel(root)
    dialog.title("Select Video Source")
    dialog.geometry("300x150")
    dialog.resizable(False, False)

    # Center the dialog
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() - 300) // 2
    y = (dialog.winfo_screenheight() - 150) // 2
    dialog.geometry(f"300x150+{x}+{y}")

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

    label = tk.Label(dialog, text="Choose video source:", font=("Arial", 12))
    label.pack(pady=20)

    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=10)

    webcam_btn = tk.Button(btn_frame, text="Webcam", command=use_webcam, width=12, height=2)
    webcam_btn.pack(side=tk.LEFT, padx=10)

    file_btn = tk.Button(btn_frame, text="Video File", command=use_video_file, width=12, height=2)
    file_btn.pack(side=tk.LEFT, padx=10)

    dialog.protocol("WM_DELETE_WINDOW", use_webcam)  # Default to webcam if closed
    dialog.grab_set()
    root.wait_window(dialog)

    return result['source'], result['path']


class EffectControlPanel:
    """A GUI panel with checkboxes and sliders to control effects."""

    def __init__(self, effect_states, effect_params):
        self.effect_states = effect_states
        self.effect_params = effect_params
        self.root = None
        self.running = False
        self.checkbox_vars = {}
        self.slider_vars = {}
        self.render_callback = None  # Will be set after initialization
        self.render_progress_var = None
        self.render_status_label = None
        self.preview_effects_var = None  # For toggling live effect preview

        # Define effects with their parameters
        # Format: (effect_key, display_name, param_key, param_min, param_max, param_default)
        self.effects_config = [
            # Original effects
            ('motion_mask_colormap', 'Motion Colormap', '1', None, None, None),
            ('glitch_slices', 'Glitch Slices', '2', 'glitch_slice_height', 2, 30, 12),
            ('zoom_punch', 'Zoom Punch', '3', 'zoom_punch_strength', 0.01, 0.2, 0.05),
            ('feedback_loop', 'Feedback Loop', '4', 'feedback_decay_rate', 0.5, 0.99, 0.9),
            ('rgb_wave', 'RGB Wave', '5', 'rgb_wave_intensity', 1, 50, 12),
            ('posterize', 'Posterize', '6', 'posterize_levels', 2, 16, 6),
            ('motion_smear', 'Motion Smear', '7', 'motion_smear_amount', 5, 150, 60),
            ('cycle_colormaps', 'Cycle Colormaps', '8', None, None, None),
            ('datamosh', 'Datamosh', '9', None, None, None),
            ('ghost_trail', 'Ghost Trail', '0', 'ghost_trail_decay', 0.5, 0.99, 0.85),

            # Batch 1
            ('edge_glow', 'Edge Glow', 'E', None, None, None),
            ('chromatic_aberration', 'Chromatic Aberration', 'A', 'chromatic_offset', 1, 20, 5),
            ('vhs', 'VHS Effect', 'V', 'vhs_noise', 5, 80, 25),
            ('mirror', 'Mirror', 'R', None, None, None),
            ('thermal', 'Thermal Vision', 'H', None, None, None),
            ('negative', 'Negative', 'N', None, None, None),
            ('pixelate', 'Pixelate', 'X', 'pixelate_size', 2, 32, 8),
            ('kaleidoscope', 'Kaleidoscope', 'K', 'kaleidoscope_segments', 3, 12, 6),
            ('color_swap', 'Color Swap', 'W', None, None, None),
            ('emboss', 'Emboss', 'B', 'emboss_strength', 0.5, 3.0, 1.0),
            ('radial_blur', 'Radial Blur', 'L', 'radial_blur_strength', 0.01, 0.1, 0.02),

            # Batch 2
            ('glitch_blocks', 'Glitch Blocks', 'G', 'glitch_blocks_count', 2, 20, 8),
            ('color_drift', 'Color Drift', 'D', 'color_drift_speed', 0.005, 0.1, 0.02),
            ('slit_scan', 'Slit Scan', 'Z', None, None, None),
            ('drunk', 'Drunk Effect', 'U', 'drunk_intensity', 5, 40, 15),
            ('ascii_art', 'ASCII Art', 'I', None, None, None),
            ('film_grain', 'Film Grain', 'F', 'film_grain_intensity', 5, 80, 30),
            ('tv_static', 'TV Static', 'O', 'tv_static_blend', 0.1, 0.8, 0.3),
            ('wave_distort', 'Wave Distort', 'Y', 'wave_amplitude', 5, 50, 20),
            ('oil_paint', 'Oil Paint', 'J', None, None, None),

            # Batch 3
            ('tunnel_vision', 'Tunnel Vision', 'F1', 'tunnel_vignette', 0.3, 1.0, 0.7),
            ('double_vision', 'Double Vision', 'F2', 'double_vision_offset', 5, 40, 15),
            ('scanlines', 'Scanlines', 'F3', 'scanline_darkness', 0.1, 0.8, 0.4),
            ('rgb_split_radial', 'RGB Split Radial', 'F4', 'rgb_split_strength', 2, 30, 10),
            ('sketch', 'Sketch', 'F5', None, None, None),
            ('halftone', 'Halftone', 'F6', 'halftone_dot_size', 2, 12, 4),
            ('neon_edges', 'Neon Edges', 'F7', 'neon_glow_size', 2, 15, 5),
            ('glitch_shift', 'Glitch Shift', 'F8', 'glitch_shift_intensity', 5, 50, 20),
            ('heat_distort', 'Heat Distort', 'F9', 'heat_distort_intensity', 2, 20, 8),
            ('cross_process', 'Cross Process', 'F10', None, None, None),
            ('duotone', 'Duotone', 'F11', None, None, None),
            ('pulse_zoom', 'Pulse Zoom', 'F12', 'pulse_zoom_amount', 0.01, 0.1, 0.03),
            ('blocky_noise', 'Blocky Noise', 'Home', 'blocky_noise_chance', 0.02, 0.3, 0.1),
            ('retro_crt', 'Retro CRT', 'End', None, None, None),
            ('time_echo', 'Time Echo', 'Ins', 'time_echo_frames', 2, 15, 5),
            ('prism', 'Prism', 'Del', 'prism_offset', 2, 20, 8),
            ('spiral_warp', 'Spiral Warp', 'PgUp', 'spiral_warp_strength', 0.1, 1.5, 0.5),
            ('digital_rain', 'Digital Rain', 'PgDn', 'digital_rain_drops', 50, 500, 200),
            ('pixel_sort', 'Pixel Sort', 'P', None, None, None),
            ('background_effect', 'Background Scanlines', '*', None, None, None),
        ]

    def create_panel(self):
        """Create the control panel window."""
        self.root = tk.Tk()
        self.root.title("Effect Control Panel")
        self.root.geometry("400x800")
        self.root.resizable(True, True)

        # Create main container with scrollbar
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create canvas and scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Title
        title_label = tk.Label(scrollable_frame, text="Effect Controls", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)

        # Button frame for Reset and Render
        button_frame = tk.Frame(scrollable_frame)
        button_frame.pack(pady=5, padx=10, fill=tk.X)

        # Reset button
        reset_btn = tk.Button(button_frame, text="Reset All Effects", command=self.reset_all,
                              bg="#ff6b6b", fg="white", font=("Arial", 10, "bold"))
        reset_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))

        # Render button
        render_btn = tk.Button(button_frame, text="Render Video", command=self.start_render,
                               bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        render_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))

        # Render progress section
        render_frame = tk.Frame(scrollable_frame)
        render_frame.pack(pady=5, padx=10, fill=tk.X)

        self.render_progress_var = tk.DoubleVar(value=0)
        self.render_progress_bar = ttk.Progressbar(render_frame, variable=self.render_progress_var,
                                                    maximum=100, mode='determinate')
        self.render_progress_bar.pack(fill=tk.X)

        self.render_status_label = tk.Label(render_frame, text="Ready to render", font=("Arial", 8))
        self.render_status_label.pack()

        # Preview toggle checkbox
        preview_frame = tk.Frame(scrollable_frame)
        preview_frame.pack(pady=5, padx=10, fill=tk.X)

        self.preview_effects_var = tk.BooleanVar(value=True)
        preview_checkbox = tk.Checkbutton(
            preview_frame,
            text="Show Effects in Live Preview (uncheck for better performance)",
            variable=self.preview_effects_var,
            font=("Arial", 9)
        )
        preview_checkbox.pack(anchor='w')

        # Hint label
        hint_label = tk.Label(preview_frame,
                              text="Tip: Press 'Space' to pause video, '`' to toggle live effects",
                              font=("Arial", 8), fg="gray")
        hint_label.pack(anchor='w')

        # Separator
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=10, padx=10)

        # Create effect controls
        for config in self.effects_config:
            self.create_effect_control(scrollable_frame, config)

        self.running = True

    def create_effect_control(self, parent, config):
        """Create a checkbox and optional slider for an effect."""
        # Handle different config lengths
        if len(config) == 6:
            # No parameter: (effect_key, display_name, hotkey, None, None, None)
            effect_key, display_name, hotkey, param_key, param_min, param_max = config
            param_default = None
        elif len(config) == 7:
            # Has parameter: (effect_key, display_name, hotkey, param_key, param_min, param_max, param_default)
            effect_key, display_name, hotkey, param_key, param_min, param_max, param_default = config
        else:
            # Fallback for unexpected formats
            effect_key = config[0]
            display_name = config[1]
            hotkey = config[2]
            param_key, param_min, param_max, param_default = None, None, None, None

        # Create frame for this effect
        effect_frame = tk.Frame(parent, relief=tk.GROOVE, borderwidth=1)
        effect_frame.pack(fill=tk.X, padx=5, pady=2)

        # Checkbox
        var = tk.BooleanVar(value=self.effect_states.get(f'{effect_key}_enabled', False))
        self.checkbox_vars[effect_key] = var

        checkbox = tk.Checkbutton(
            effect_frame,
            text=f"{display_name} [{hotkey}]",
            variable=var,
            command=lambda k=effect_key: self.toggle_effect(k),
            font=("Arial", 9)
        )
        checkbox.pack(anchor='w', padx=5)

        # Slider if effect has parameters
        if param_key and param_min is not None and param_max is not None:
            slider_frame = tk.Frame(effect_frame)
            slider_frame.pack(fill=tk.X, padx=20, pady=2)

            # Determine if we need float or int resolution
            if isinstance(param_min, float) or isinstance(param_max, float):
                resolution = 0.01
                slider_var = tk.DoubleVar(value=self.effect_params.get(param_key, param_default))
            else:
                resolution = 1
                slider_var = tk.IntVar(value=self.effect_params.get(param_key, param_default))

            self.slider_vars[param_key] = slider_var

            slider = tk.Scale(
                slider_frame,
                from_=param_min,
                to=param_max,
                orient=tk.HORIZONTAL,
                variable=slider_var,
                resolution=resolution,
                length=300,
                command=lambda val, pk=param_key: self.update_param(pk, val)
            )
            slider.pack(fill=tk.X)

    def toggle_effect(self, effect_key):
        """Toggle an effect on/off."""
        state = self.checkbox_vars[effect_key].get()
        self.effect_states[f'{effect_key}_enabled'] = state

    def update_param(self, param_key, value):
        """Update an effect parameter."""
        try:
            # Try to convert to appropriate type
            if '.' in str(value):
                self.effect_params[param_key] = float(value)
            else:
                self.effect_params[param_key] = int(float(value))
        except ValueError:
            self.effect_params[param_key] = float(value)

    def reset_all(self):
        """Reset all effects to off."""
        for key, var in self.checkbox_vars.items():
            var.set(False)
            self.effect_states[f'{key}_enabled'] = False

    def start_render(self):
        """Trigger the render callback."""
        if self.render_callback:
            self.render_callback()

    def update_render_progress(self, progress, status_text):
        """Update the render progress bar and status."""
        if self.render_progress_var:
            self.render_progress_var.set(progress)
        if self.render_status_label:
            self.render_status_label.config(text=status_text)
        if self.root:
            self.root.update_idletasks()

    def sync_from_keyboard(self):
        """Sync checkbox states from keyboard toggles."""
        for config in self.effects_config:
            effect_key = config[0]
            state_key = f'{effect_key}_enabled'
            if state_key in self.effect_states and effect_key in self.checkbox_vars:
                self.checkbox_vars[effect_key].set(self.effect_states[state_key])

    def update(self):
        """Update the panel (call from main loop)."""
        if self.root and self.running:
            try:
                self.sync_from_keyboard()
                self.root.update_idletasks()
                self.root.update()
            except tk.TclError:
                self.running = False

    def close(self):
        """Close the panel."""
        if self.root:
            self.running = False
            try:
                self.root.destroy()
            except:
                pass


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

    return output_frame, current_gray, render_buffers


# ----------------------------
# Video Source Setup
# ----------------------------
source_type, source_path = select_video_source()

if source_type == 'webcam':
    camera = cv2.VideoCapture(source_path, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
    is_video_file = False
    frame_delay = 1
    total_video_frames = 0
    print("Using webcam")
else:
    camera = cv2.VideoCapture(source_path)
    is_video_file = True
    video_fps = camera.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = int(1000 / video_fps)
    total_video_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Loaded video: {source_path}")
    print(f"FPS: {video_fps}, Total frames: {total_video_frames}")

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

# ----------------------------
# Rendering State
# ----------------------------
is_rendering = False


def render_video():
    """Render the video with current effects at original framerate, preserving audio."""
    global is_rendering

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

    while True:
        ret, frame = render_cap.read()
        if not ret:
            break

        frame_count += 1

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

    # Cleanup video writing
    render_cap.release()
    out_writer.release()

    # Now merge audio from original video using ffmpeg
    control_panel.update_render_progress(95, "Adding audio...")
    print("Adding audio from original video...")

    # Check if ffmpeg is available
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            # Use ffmpeg to combine video with original audio
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", temp_video_path,      # Input: rendered video (no audio)
                "-i", source_path,           # Input: original video (for audio)
                "-c:v", "copy",              # Copy video stream as-is
                "-c:a", "aac",               # Encode audio as AAC
                "-map", "0:v:0",             # Use video from first input
                "-map", "1:a:0?",            # Use audio from second input (if exists)
                "-shortest",                 # Match shortest stream
                output_path
            ]

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Remove temp file
                os.remove(temp_video_path)
                print("Audio merged successfully!")
            else:
                print(f"FFmpeg warning: {result.stderr}")
                # If ffmpeg fails, just rename temp to output
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_video_path, output_path)
                print("Saved without audio (ffmpeg merge failed)")

        except Exception as e:
            print(f"Error running ffmpeg: {e}")
            # Fallback: rename temp to output
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_video_path, output_path)
            print("Saved without audio")
    else:
        # FFmpeg not found, just rename temp to output
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
# Effect Strength Knobs (legacy - now uses effect_params)
# ----------------------------
effect_strength_values = [
    effect_params['rgb_wave_intensity'],
    effect_params['posterize_levels'],
    effect_params['motion_smear_amount'],
    effect_params['feedback_decay_rate']
]

# ----------------------------
# Main Loop
# ----------------------------
current_colormap_preset = 0
total_frames_processed = 0

while True:
    total_frames_processed += 1

    # Handle video file playback
    if is_video_file and not video_paused:
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

    # Sync live_preview_effects with control panel checkbox
    if control_panel.preview_effects_var:
        live_preview_effects = control_panel.preview_effects_var.get()

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

    # Effect strength adjustment
    if keyboard.is_pressed("+"):
        effect_strength_values = change_knob("up", effect_strength_values)
    if keyboard.is_pressed("-"):
        effect_strength_values = change_knob("down", effect_strength_values)

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

    # Update control panel GUI
    control_panel.update()

    # Quit application
    wait_time = frame_delay if is_video_file else 1
    if cv2.waitKey(wait_time) & 0xFF == ord("q"):
        break

    cv2.imshow("glitch mirror", output_frame)

# ----------------------------
# Cleanup
# ----------------------------
control_panel.close()
camera.release()
cv2.destroyAllWindows()
