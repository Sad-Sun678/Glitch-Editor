import cv2
import numpy as np
import keyboard
import effects
import time
import tkinter as tk
from tkinter import filedialog, ttk, simpledialog
import threading
import subprocess
import os
import shutil
import tempfile
import json

# Preset file paths
PRESETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
VIDEO_PRESETS_FILE = os.path.join(PRESETS_DIR, "video_presets.json")
AUDIO_PRESETS_FILE = os.path.join(PRESETS_DIR, "audio_presets.json")

# Create presets directory if it doesn't exist
os.makedirs(PRESETS_DIR, exist_ok=True)

# Try to import pygame for audio playback
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("pygame not installed - audio preview will be limited")

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
            ('rotation', 'Rotation', 'Q', 'rotation_speed', 0.1, 5.0, 0.5),
        ]

        # Load saved custom presets
        self.custom_presets = self.load_custom_presets()

    def load_custom_presets(self):
        """Load custom presets from JSON file."""
        try:
            if os.path.exists(VIDEO_PRESETS_FILE):
                with open(VIDEO_PRESETS_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading video presets: {e}")
        return {}

    def save_custom_presets(self):
        """Save custom presets to JSON file."""
        try:
            with open(VIDEO_PRESETS_FILE, 'w') as f:
                json.dump(self.custom_presets, f, indent=2)
            print(f"Video presets saved to {VIDEO_PRESETS_FILE}")
        except Exception as e:
            print(f"Error saving video presets: {e}")

    def save_current_as_preset(self):
        """Save current effect settings as a named preset."""
        name = simpledialog.askstring("Save Preset", "Enter preset name:",
                                       parent=self.root)
        if not name:
            return

        # Collect current state
        preset_data = {
            'effect_states': dict(self.effect_states),
            'effect_params': dict(self.effect_params)
        }
        self.custom_presets[name] = preset_data
        self.save_custom_presets()
        self.update_preset_dropdown()
        print(f"Saved video preset: {name}")

    def load_preset(self, preset_name):
        """Load a preset by name."""
        if preset_name not in self.custom_presets:
            return

        preset_data = self.custom_presets[preset_name]

        # Apply effect states
        if 'effect_states' in preset_data:
            for key, value in preset_data['effect_states'].items():
                if key in self.effect_states:
                    self.effect_states[key] = value

        # Apply effect params
        if 'effect_params' in preset_data:
            for key, value in preset_data['effect_params'].items():
                if key in self.effect_params:
                    self.effect_params[key] = value
                    if key in self.slider_vars:
                        self.slider_vars[key].set(value)

        # Sync checkboxes
        self.sync_from_keyboard()
        print(f"Loaded video preset: {preset_name}")

    def delete_preset(self):
        """Delete the currently selected preset."""
        if hasattr(self, 'preset_var') and self.preset_var.get():
            name = self.preset_var.get()
            if name in self.custom_presets:
                del self.custom_presets[name]
                self.save_custom_presets()
                self.update_preset_dropdown()
                print(f"Deleted video preset: {name}")

    def update_preset_dropdown(self):
        """Update the preset dropdown menu."""
        if hasattr(self, 'preset_dropdown'):
            menu = self.preset_dropdown['menu']
            menu.delete(0, 'end')
            for name in self.custom_presets.keys():
                menu.add_command(label=name, command=lambda n=name: self.on_preset_selected(n))
            if self.custom_presets:
                self.preset_var.set(list(self.custom_presets.keys())[0])
            else:
                self.preset_var.set('')

    def on_preset_selected(self, name):
        """Handle preset selection from dropdown."""
        self.preset_var.set(name)
        self.load_preset(name)

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

        # Preview scale slider for performance
        scale_frame = tk.Frame(preview_frame)
        scale_frame.pack(fill=tk.X, pady=5)

        scale_label = tk.Label(scale_frame, text="Preview Size:", font=("Arial", 9))
        scale_label.pack(side=tk.LEFT)

        self.preview_scale_var = tk.DoubleVar(value=1.0)
        self.preview_scale_slider = tk.Scale(
            scale_frame,
            from_=0.25,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.preview_scale_var,
            resolution=0.05,
            length=200,
            font=("Arial", 8)
        )
        self.preview_scale_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        scale_hint = tk.Label(scale_frame, text="(smaller = faster)", font=("Arial", 8), fg="gray")
        scale_hint.pack(side=tk.LEFT, padx=5)

        # Hint label
        hint_label = tk.Label(preview_frame,
                              text="Tip: Press 'Space' to pause video, '`' to toggle live effects",
                              font=("Arial", 8), fg="gray")
        hint_label.pack(anchor='w')

        # Separator
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=10, padx=10)

        # Custom Presets Section
        preset_frame = tk.LabelFrame(scrollable_frame, text="Custom Presets", font=("Arial", 10, "bold"))
        preset_frame.pack(pady=5, padx=10, fill=tk.X)

        # Save preset button
        save_preset_btn = tk.Button(preset_frame, text="💾 Save Current as Preset",
                                     command=self.save_current_as_preset,
                                     bg="#2196F3", fg="white")
        save_preset_btn.pack(fill=tk.X, padx=5, pady=2)

        # Preset dropdown and load
        preset_select_frame = tk.Frame(preset_frame)
        preset_select_frame.pack(fill=tk.X, padx=5, pady=2)

        self.preset_var = tk.StringVar()
        preset_names = list(self.custom_presets.keys()) if self.custom_presets else ['']
        self.preset_dropdown = tk.OptionMenu(preset_select_frame, self.preset_var,
                                              *preset_names if preset_names else [''])
        self.preset_dropdown.config(width=20)
        self.preset_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)

        load_preset_btn = tk.Button(preset_select_frame, text="Load",
                                     command=lambda: self.load_preset(self.preset_var.get()),
                                     bg="#4CAF50", fg="white", width=6)
        load_preset_btn.pack(side=tk.LEFT, padx=2)

        delete_preset_btn = tk.Button(preset_select_frame, text="Delete",
                                       command=self.delete_preset,
                                       bg="#f44336", fg="white", width=6)
        delete_preset_btn.pack(side=tk.LEFT, padx=2)

        if self.custom_presets:
            self.preset_var.set(list(self.custom_presets.keys())[0])

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


class AudioEffectsPanel:
    """A GUI panel for audio preview and effects."""

    def __init__(self, video_source_path=None):
        self.video_source_path = video_source_path
        self.root = None
        self.running = False
        self.temp_dir = tempfile.mkdtemp()
        self.original_audio_path = None
        self.video_original_audio_path = None  # Backup of video's original audio
        self.preview_audio_path = None
        self.processed_audio_path = None  # The "kept" audio for final render
        self.is_playing = False
        self.is_extracting = False
        self.is_processing = False
        self.preview_counter = 0  # For unique preview filenames
        self.pending_status = None  # For thread-safe status updates
        self.custom_audio_loaded = False  # Track if custom audio is loaded
        self.custom_audio_source = None  # Original path of custom audio file

        # Audio effect parameters
        self.audio_params = {
            'speed': 1.0,           # 0.5 to 2.0
            'pitch': 1.0,           # 0.5 to 2.0
            'bass': 0,              # -20 to 20 dB
            'treble': 0,            # -20 to 20 dB
            'echo_delay': 0,        # 0 to 1000 ms
            'echo_decay': 0.0,      # 0 to 0.9
            'reverb': 0,            # 0 to 100
            'distortion': 0,        # 0 to 100
            'lowpass': 20000,       # 200 to 20000 Hz
            'highpass': 20,         # 20 to 5000 Hz
            'volume': 1.0,          # 0 to 2.0
            'reverse': False,
            'bitcrush': 16,         # 4 to 16 bits
            'flanger_delay': 0,     # 0 to 20 ms
            'flanger_depth': 0,     # 0 to 10
            'tremolo_freq': 0,      # 0 to 20 Hz
            'tremolo_depth': 0.0,   # 0 to 1.0
            'vibrato_freq': 0,      # 0 to 20 Hz
            'vibrato_depth': 0.0,   # 0 to 1.0
            'phaser_speed': 0.0,    # 0 to 2 Hz
            'chorus_depth': 0,      # 0 to 10 ms
            'noise_amount': 0,      # 0 to 100
            'telephone': False,
            'mono': False,
        }

        self.slider_vars = {}
        self.presets = {
            'Normal': {'speed': 1.0, 'pitch': 1.0, 'bass': 0, 'treble': 0, 'echo_delay': 0,
                      'echo_decay': 0.0, 'reverb': 0, 'distortion': 0, 'lowpass': 20000,
                      'highpass': 20, 'volume': 1.0, 'reverse': False, 'bitcrush': 16,
                      'flanger_delay': 0, 'flanger_depth': 0, 'tremolo_freq': 0, 'tremolo_depth': 0.0,
                      'vibrato_freq': 0, 'vibrato_depth': 0.0, 'phaser_speed': 0.0, 'chorus_depth': 0,
                      'noise_amount': 0, 'telephone': False, 'mono': False},
            'Underwater': {'speed': 0.9, 'pitch': 0.85, 'bass': 10, 'treble': -15, 'echo_delay': 100,
                          'echo_decay': 0.4, 'reverb': 60, 'distortion': 0, 'lowpass': 800,
                          'highpass': 20, 'volume': 1.0, 'reverse': False, 'bitcrush': 16,
                          'flanger_delay': 5, 'flanger_depth': 3, 'tremolo_freq': 0, 'tremolo_depth': 0.0,
                          'vibrato_freq': 2, 'vibrato_depth': 0.3, 'phaser_speed': 0.3, 'chorus_depth': 4,
                          'noise_amount': 0, 'telephone': False, 'mono': False},
            'Radio': {'speed': 1.0, 'pitch': 1.0, 'bass': -10, 'treble': 5, 'echo_delay': 0,
                     'echo_decay': 0.0, 'reverb': 0, 'distortion': 10, 'lowpass': 4000,
                     'highpass': 300, 'volume': 1.0, 'reverse': False, 'bitcrush': 16,
                     'flanger_delay': 0, 'flanger_depth': 0, 'tremolo_freq': 0, 'tremolo_depth': 0.0,
                     'vibrato_freq': 0, 'vibrato_depth': 0.0, 'phaser_speed': 0.0, 'chorus_depth': 0,
                     'noise_amount': 15, 'telephone': False, 'mono': True},
            'Cave Echo': {'speed': 1.0, 'pitch': 0.95, 'bass': 5, 'treble': -5, 'echo_delay': 300,
                         'echo_decay': 0.6, 'reverb': 80, 'distortion': 0, 'lowpass': 20000,
                         'highpass': 20, 'volume': 0.9, 'reverse': False, 'bitcrush': 16,
                         'flanger_delay': 0, 'flanger_depth': 0, 'tremolo_freq': 0, 'tremolo_depth': 0.0,
                         'vibrato_freq': 0, 'vibrato_depth': 0.0, 'phaser_speed': 0.0, 'chorus_depth': 0,
                         'noise_amount': 0, 'telephone': False, 'mono': False},
            'Chipmunk': {'speed': 1.3, 'pitch': 1.5, 'bass': -5, 'treble': 10, 'echo_delay': 0,
                        'echo_decay': 0.0, 'reverb': 0, 'distortion': 0, 'lowpass': 20000,
                        'highpass': 20, 'volume': 1.0, 'reverse': False, 'bitcrush': 16,
                        'flanger_delay': 0, 'flanger_depth': 0, 'tremolo_freq': 0, 'tremolo_depth': 0.0,
                        'vibrato_freq': 0, 'vibrato_depth': 0.0, 'phaser_speed': 0.0, 'chorus_depth': 0,
                        'noise_amount': 0, 'telephone': False, 'mono': False},
            'Deep Voice': {'speed': 0.85, 'pitch': 0.7, 'bass': 15, 'treble': -10, 'echo_delay': 0,
                          'echo_decay': 0.0, 'reverb': 20, 'distortion': 0, 'lowpass': 20000,
                          'highpass': 20, 'volume': 1.0, 'reverse': False, 'bitcrush': 16,
                          'flanger_delay': 0, 'flanger_depth': 0, 'tremolo_freq': 0, 'tremolo_depth': 0.0,
                          'vibrato_freq': 0, 'vibrato_depth': 0.0, 'phaser_speed': 0.0, 'chorus_depth': 0,
                          'noise_amount': 0, 'telephone': False, 'mono': False},
            'Robot': {'speed': 1.0, 'pitch': 1.0, 'bass': 0, 'treble': 0, 'echo_delay': 50,
                     'echo_decay': 0.5, 'reverb': 30, 'distortion': 30, 'lowpass': 5000,
                     'highpass': 200, 'volume': 1.0, 'reverse': False, 'bitcrush': 8,
                     'flanger_delay': 2, 'flanger_depth': 2, 'tremolo_freq': 8, 'tremolo_depth': 0.3,
                     'vibrato_freq': 0, 'vibrato_depth': 0.0, 'phaser_speed': 0.0, 'chorus_depth': 0,
                     'noise_amount': 5, 'telephone': False, 'mono': False},
            'VHS Tape': {'speed': 0.98, 'pitch': 0.98, 'bass': 5, 'treble': -8, 'echo_delay': 20,
                        'echo_decay': 0.2, 'reverb': 10, 'distortion': 15, 'lowpass': 8000,
                        'highpass': 80, 'volume': 0.9, 'reverse': False, 'bitcrush': 12,
                        'flanger_delay': 3, 'flanger_depth': 2, 'tremolo_freq': 0, 'tremolo_depth': 0.0,
                        'vibrato_freq': 1, 'vibrato_depth': 0.1, 'phaser_speed': 0.0, 'chorus_depth': 2,
                        'noise_amount': 20, 'telephone': False, 'mono': False},
            'Nightmare': {'speed': 0.7, 'pitch': 0.6, 'bass': 15, 'treble': -10, 'echo_delay': 400,
                         'echo_decay': 0.7, 'reverb': 90, 'distortion': 20, 'lowpass': 3000,
                         'highpass': 20, 'volume': 0.8, 'reverse': False, 'bitcrush': 16,
                         'flanger_delay': 8, 'flanger_depth': 5, 'tremolo_freq': 3, 'tremolo_depth': 0.4,
                         'vibrato_freq': 2, 'vibrato_depth': 0.5, 'phaser_speed': 0.5, 'chorus_depth': 5,
                         'noise_amount': 10, 'telephone': False, 'mono': False},
            'Lo-Fi': {'speed': 1.0, 'pitch': 1.0, 'bass': 5, 'treble': -12, 'echo_delay': 0,
                     'echo_decay': 0.0, 'reverb': 15, 'distortion': 5, 'lowpass': 3500,
                     'highpass': 100, 'volume': 0.85, 'reverse': False, 'bitcrush': 10,
                     'flanger_delay': 0, 'flanger_depth': 0, 'tremolo_freq': 0, 'tremolo_depth': 0.0,
                     'vibrato_freq': 0, 'vibrato_depth': 0.0, 'phaser_speed': 0.0, 'chorus_depth': 0,
                     'noise_amount': 25, 'telephone': False, 'mono': False},
            'Telephone': {'speed': 1.0, 'pitch': 1.0, 'bass': -15, 'treble': -5, 'echo_delay': 0,
                         'echo_decay': 0.0, 'reverb': 0, 'distortion': 5, 'lowpass': 3400,
                         'highpass': 400, 'volume': 1.0, 'reverse': False, 'bitcrush': 14,
                         'flanger_delay': 0, 'flanger_depth': 0, 'tremolo_freq': 0, 'tremolo_depth': 0.0,
                         'vibrato_freq': 0, 'vibrato_depth': 0.0, 'phaser_speed': 0.0, 'chorus_depth': 0,
                         'noise_amount': 10, 'telephone': True, 'mono': True},
            'Haunted': {'speed': 0.9, 'pitch': 0.85, 'bass': 8, 'treble': -5, 'echo_delay': 250,
                       'echo_decay': 0.5, 'reverb': 70, 'distortion': 10, 'lowpass': 6000,
                       'highpass': 50, 'volume': 0.85, 'reverse': False, 'bitcrush': 16,
                       'flanger_delay': 10, 'flanger_depth': 4, 'tremolo_freq': 4, 'tremolo_depth': 0.3,
                       'vibrato_freq': 3, 'vibrato_depth': 0.4, 'phaser_speed': 0.8, 'chorus_depth': 6,
                       'noise_amount': 5, 'telephone': False, 'mono': False},
            'Alien': {'speed': 1.1, 'pitch': 1.3, 'bass': -5, 'treble': 8, 'echo_delay': 80,
                     'echo_decay': 0.3, 'reverb': 40, 'distortion': 15, 'lowpass': 12000,
                     'highpass': 200, 'volume': 1.0, 'reverse': False, 'bitcrush': 12,
                     'flanger_delay': 15, 'flanger_depth': 8, 'tremolo_freq': 6, 'tremolo_depth': 0.2,
                     'vibrato_freq': 8, 'vibrato_depth': 0.6, 'phaser_speed': 1.5, 'chorus_depth': 8,
                     'noise_amount': 0, 'telephone': False, 'mono': False},
            'Broken Speaker': {'speed': 1.0, 'pitch': 1.0, 'bass': 10, 'treble': -15, 'echo_delay': 0,
                              'echo_decay': 0.0, 'reverb': 0, 'distortion': 60, 'lowpass': 2500,
                              'highpass': 100, 'volume': 0.9, 'reverse': False, 'bitcrush': 6,
                              'flanger_delay': 0, 'flanger_depth': 0, 'tremolo_freq': 12, 'tremolo_depth': 0.5,
                              'vibrato_freq': 0, 'vibrato_depth': 0.0, 'phaser_speed': 0.0, 'chorus_depth': 0,
                              'noise_amount': 40, 'telephone': False, 'mono': True},
            'Dreamy': {'speed': 0.95, 'pitch': 1.05, 'bass': 3, 'treble': 5, 'echo_delay': 150,
                      'echo_decay': 0.4, 'reverb': 60, 'distortion': 0, 'lowpass': 15000,
                      'highpass': 20, 'volume': 0.9, 'reverse': False, 'bitcrush': 16,
                      'flanger_delay': 6, 'flanger_depth': 3, 'tremolo_freq': 0, 'tremolo_depth': 0.0,
                      'vibrato_freq': 1, 'vibrato_depth': 0.15, 'phaser_speed': 0.3, 'chorus_depth': 5,
                      'noise_amount': 0, 'telephone': False, 'mono': False},
        }

        # Load custom audio presets
        self.custom_presets = self.load_custom_presets()

    def load_custom_presets(self):
        """Load custom audio presets from JSON file."""
        try:
            if os.path.exists(AUDIO_PRESETS_FILE):
                with open(AUDIO_PRESETS_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading audio presets: {e}")
        return {}

    def save_custom_presets(self):
        """Save custom audio presets to JSON file."""
        try:
            with open(AUDIO_PRESETS_FILE, 'w') as f:
                json.dump(self.custom_presets, f, indent=2)
            print(f"Audio presets saved to {AUDIO_PRESETS_FILE}")
        except Exception as e:
            print(f"Error saving audio presets: {e}")

    def save_current_as_preset(self):
        """Save current audio settings as a named preset."""
        name = simpledialog.askstring("Save Audio Preset", "Enter preset name:",
                                       parent=self.root)
        if not name:
            return

        # Collect current state
        self.custom_presets[name] = dict(self.audio_params)
        self.save_custom_presets()
        self.update_custom_preset_dropdown()
        self.status_label.config(text=f"Saved preset: {name}")
        print(f"Saved audio preset: {name}")

    def load_custom_preset(self, preset_name):
        """Load a custom preset by name."""
        if preset_name not in self.custom_presets:
            return

        preset_data = self.custom_presets[preset_name]
        for key, value in preset_data.items():
            self.audio_params[key] = value
            if key in self.slider_vars:
                self.slider_vars[key].set(value)
            elif key == 'reverse' and hasattr(self, 'reverse_var'):
                self.reverse_var.set(value)
            elif key == 'mono' and hasattr(self, 'mono_var'):
                self.mono_var.set(value)
            elif key == 'telephone' and hasattr(self, 'telephone_var'):
                self.telephone_var.set(value)

        self.status_label.config(text=f"Loaded preset: {preset_name}")
        print(f"Loaded audio preset: {preset_name}")

    def delete_custom_preset(self):
        """Delete the currently selected custom preset."""
        if hasattr(self, 'custom_preset_var') and self.custom_preset_var.get():
            name = self.custom_preset_var.get()
            if name in self.custom_presets:
                del self.custom_presets[name]
                self.save_custom_presets()
                self.update_custom_preset_dropdown()
                self.status_label.config(text=f"Deleted preset: {name}")
                print(f"Deleted audio preset: {name}")

    def update_custom_preset_dropdown(self):
        """Update the custom preset dropdown menu."""
        if hasattr(self, 'custom_preset_dropdown'):
            menu = self.custom_preset_dropdown['menu']
            menu.delete(0, 'end')
            for name in self.custom_presets.keys():
                menu.add_command(label=name, command=lambda n=name: self.on_custom_preset_selected(n))
            if self.custom_presets:
                self.custom_preset_var.set(list(self.custom_presets.keys())[0])
            else:
                self.custom_preset_var.set('')

    def on_custom_preset_selected(self, name):
        """Handle custom preset selection from dropdown."""
        self.custom_preset_var.set(name)
        self.load_custom_preset(name)

    def set_video_source(self, path):
        """Set the video source and extract audio."""
        self.video_source_path = path
        if path and os.path.exists(path):
            self.extract_audio()

    def extract_audio(self):
        """Extract audio from video file using ffmpeg."""
        if not self.video_source_path or self.is_extracting:
            return

        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            print("FFmpeg not found - cannot extract audio")
            return

        self.is_extracting = True
        self.original_audio_path = os.path.join(self.temp_dir, "original_audio.wav")
        self.video_original_audio_path = os.path.join(self.temp_dir, "video_original_audio.wav")

        def extract_thread():
            try:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", self.video_source_path,
                    "-vn",  # No video
                    "-acodec", "pcm_s16le",
                    "-ar", "44100",
                    "-ac", "2",
                    self.original_audio_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"Audio extracted to: {self.original_audio_path}")
                    self.preview_audio_path = self.original_audio_path
                    # Make a backup copy of the video's original audio
                    shutil.copy(self.original_audio_path, self.video_original_audio_path)
                else:
                    print(f"Failed to extract audio: {result.stderr}")
                    self.original_audio_path = None
            except Exception as e:
                print(f"Error extracting audio: {e}")
                self.original_audio_path = None
            finally:
                self.is_extracting = False

        thread = threading.Thread(target=extract_thread, daemon=True)
        thread.start()

    def load_custom_audio(self):
        """Load a custom audio file (MP3 or WAV) to replace the video's audio."""
        if self.is_extracting or self.is_processing:
            self.status_label.config(text="Please wait for current operation...")
            return

        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.ogg *.flac *.aac *.m4a"),
                ("MP3 Files", "*.mp3"),
                ("WAV Files", "*.wav"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        self.status_label.config(text="Loading custom audio...")
        self.root.update()

        # Stop any playing audio
        self.stop_audio()

        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            self.status_label.config(text="FFmpeg not found!")
            return

        # Convert to WAV format for consistent processing
        new_audio_path = os.path.join(self.temp_dir, "custom_audio.wav")

        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", file_path,
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                "-ac", "2",
                new_audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                self.original_audio_path = new_audio_path
                self.preview_audio_path = new_audio_path
                self.custom_audio_loaded = True
                self.custom_audio_source = file_path
                self.processed_audio_path = None  # Reset kept audio

                # Update UI
                filename = os.path.basename(file_path)
                self.audio_source_label.config(text=f"Source: {filename}")
                self.keep_status.config(text="Using: Custom Audio (not yet kept)")
                self.status_label.config(text=f"Loaded: {filename}")
                print(f"Custom audio loaded: {file_path}")
            else:
                self.status_label.config(text="Failed to load audio file")
                print(f"FFmpeg error: {result.stderr}")

        except Exception as e:
            self.status_label.config(text=f"Error: {e}")
            print(f"Error loading custom audio: {e}")

    def restore_original_audio(self):
        """Restore the video's original audio."""
        if not self.video_original_audio_path or not os.path.exists(self.video_original_audio_path):
            self.status_label.config(text="Original audio not available")
            return

        self.stop_audio()

        # Restore from backup
        self.original_audio_path = os.path.join(self.temp_dir, "original_audio.wav")
        shutil.copy(self.video_original_audio_path, self.original_audio_path)
        self.preview_audio_path = self.original_audio_path
        self.custom_audio_loaded = False
        self.custom_audio_source = None
        self.processed_audio_path = None  # Reset kept audio

        # Update UI
        self.audio_source_label.config(text="Source: Video's Original Audio")
        self.keep_status.config(text="Using: Original Audio")
        self.status_label.config(text="Restored original audio")
        print("Restored video's original audio")

    def create_panel(self):
        """Create the audio effects panel window."""
        self.root = tk.Toplevel()
        self.root.title("Audio Effects Panel")
        self.root.geometry("450x700")
        self.root.resizable(True, True)

        # Create main container with scrollbar
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Title
        title_label = tk.Label(scrollable_frame, text="Audio Effects", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)

        # Status label
        self.status_label = tk.Label(scrollable_frame, text="Loading audio...", font=("Arial", 9), fg="gray")
        self.status_label.pack(pady=5)

        # Playback controls
        playback_frame = tk.Frame(scrollable_frame)
        playback_frame.pack(pady=10, padx=10, fill=tk.X)

        self.play_btn = tk.Button(playback_frame, text="▶ Play", command=self.play_audio,
                                   width=10, bg="#4CAF50", fg="white")
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(playback_frame, text="⬛ Stop", command=self.stop_audio,
                                   width=10, bg="#f44336", fg="white")
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.preview_btn = tk.Button(playback_frame, text="🔄 Preview Effects", command=self.preview_effects,
                                      width=15, bg="#2196F3", fg="white")
        self.preview_btn.pack(side=tk.LEFT, padx=5)

        # Keep button
        keep_frame = tk.Frame(scrollable_frame)
        keep_frame.pack(pady=5, padx=10, fill=tk.X)

        self.keep_btn = tk.Button(keep_frame, text="✓ Keep This Audio for Render", command=self.keep_audio,
                                   bg="#9C27B0", fg="white", font=("Arial", 10, "bold"))
        self.keep_btn.pack(fill=tk.X)

        self.keep_status = tk.Label(keep_frame, text="Using: Original Audio", font=("Arial", 8), fg="gray")
        self.keep_status.pack()

        # Separator
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=10, padx=10)

        # Audio Source Section
        source_frame = tk.LabelFrame(scrollable_frame, text="Audio Source", font=("Arial", 10, "bold"))
        source_frame.pack(pady=5, padx=10, fill=tk.X)

        self.audio_source_label = tk.Label(source_frame, text="Source: Video's Original Audio",
                                            font=("Arial", 8), fg="gray", wraplength=380)
        self.audio_source_label.pack(pady=2)

        source_btn_frame = tk.Frame(source_frame)
        source_btn_frame.pack(fill=tk.X, pady=5)

        self.load_audio_btn = tk.Button(source_btn_frame, text="📁 Load Custom Audio",
                                         command=self.load_custom_audio,
                                         bg="#FF9800", fg="white", width=18)
        self.load_audio_btn.pack(side=tk.LEFT, padx=5)

        self.restore_original_btn = tk.Button(source_btn_frame, text="↩ Restore Original",
                                               command=self.restore_original_audio,
                                               bg="#607D8B", fg="white", width=15)
        self.restore_original_btn.pack(side=tk.LEFT, padx=5)

        # Separator
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=10, padx=10)

        # Presets section
        preset_frame = tk.LabelFrame(scrollable_frame, text="Presets", font=("Arial", 10, "bold"))
        preset_frame.pack(pady=5, padx=10, fill=tk.X)

        preset_btn_frame = tk.Frame(preset_frame)
        preset_btn_frame.pack(pady=5, padx=5, fill=tk.X)

        # Create preset buttons in a grid
        row = 0
        col = 0
        for preset_name in self.presets.keys():
            btn = tk.Button(preset_btn_frame, text=preset_name, width=12,
                           command=lambda p=preset_name: self.apply_preset(p))
            btn.grid(row=row, column=col, padx=2, pady=2)
            col += 1
            if col >= 3:
                col = 0
                row += 1

        # Separator
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=10, padx=10)

        # Custom Presets Section
        custom_preset_frame = tk.LabelFrame(scrollable_frame, text="Custom Presets", font=("Arial", 10, "bold"))
        custom_preset_frame.pack(pady=5, padx=10, fill=tk.X)

        # Save preset button
        save_preset_btn = tk.Button(custom_preset_frame, text="💾 Save Current as Preset",
                                     command=self.save_current_as_preset,
                                     bg="#2196F3", fg="white")
        save_preset_btn.pack(fill=tk.X, padx=5, pady=2)

        # Custom preset dropdown and load
        custom_preset_select_frame = tk.Frame(custom_preset_frame)
        custom_preset_select_frame.pack(fill=tk.X, padx=5, pady=2)

        self.custom_preset_var = tk.StringVar()
        custom_preset_names = list(self.custom_presets.keys()) if self.custom_presets else ['']
        self.custom_preset_dropdown = tk.OptionMenu(custom_preset_select_frame, self.custom_preset_var,
                                                     *custom_preset_names if custom_preset_names else [''])
        self.custom_preset_dropdown.config(width=20)
        self.custom_preset_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)

        load_custom_btn = tk.Button(custom_preset_select_frame, text="Load",
                                     command=lambda: self.load_custom_preset(self.custom_preset_var.get()),
                                     bg="#4CAF50", fg="white", width=6)
        load_custom_btn.pack(side=tk.LEFT, padx=2)

        delete_custom_btn = tk.Button(custom_preset_select_frame, text="Delete",
                                       command=self.delete_custom_preset,
                                       bg="#f44336", fg="white", width=6)
        delete_custom_btn.pack(side=tk.LEFT, padx=2)

        if self.custom_presets:
            self.custom_preset_var.set(list(self.custom_presets.keys())[0])

        # Separator
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=10, padx=10)

        # Effect sliders - Basic
        basic_frame = tk.LabelFrame(scrollable_frame, text="Basic Effects", font=("Arial", 10, "bold"))
        basic_frame.pack(pady=5, padx=10, fill=tk.X)

        self.create_slider(basic_frame, 'speed', 'Speed', 0.5, 2.0, 1.0, 0.05)
        self.create_slider(basic_frame, 'pitch', 'Pitch', 0.5, 2.0, 1.0, 0.05)
        self.create_slider(basic_frame, 'volume', 'Volume', 0.0, 2.0, 1.0, 0.05)
        self.create_slider(basic_frame, 'bass', 'Bass (dB)', -20, 20, 0, 1)
        self.create_slider(basic_frame, 'treble', 'Treble (dB)', -20, 20, 0, 1)

        # Echo & Reverb
        echo_frame = tk.LabelFrame(scrollable_frame, text="Echo & Reverb", font=("Arial", 10, "bold"))
        echo_frame.pack(pady=5, padx=10, fill=tk.X)

        self.create_slider(echo_frame, 'echo_delay', 'Echo Delay (ms)', 0, 1000, 0, 10)
        self.create_slider(echo_frame, 'echo_decay', 'Echo Decay', 0.0, 0.9, 0.0, 0.05)
        self.create_slider(echo_frame, 'reverb', 'Reverb', 0, 100, 0, 5)

        # Distortion & Degradation
        distort_frame = tk.LabelFrame(scrollable_frame, text="Distortion & Degradation", font=("Arial", 10, "bold"))
        distort_frame.pack(pady=5, padx=10, fill=tk.X)

        self.create_slider(distort_frame, 'distortion', 'Distortion', 0, 100, 0, 5)
        self.create_slider(distort_frame, 'bitcrush', 'Bit Depth', 4, 16, 16, 1)
        self.create_slider(distort_frame, 'noise_amount', 'Noise/Static', 0, 100, 0, 5)

        # Filters
        filter_frame = tk.LabelFrame(scrollable_frame, text="Filters", font=("Arial", 10, "bold"))
        filter_frame.pack(pady=5, padx=10, fill=tk.X)

        self.create_slider(filter_frame, 'lowpass', 'Lowpass (Hz)', 200, 20000, 20000, 100)
        self.create_slider(filter_frame, 'highpass', 'Highpass (Hz)', 20, 5000, 20, 20)

        # Modulation Effects
        mod_frame = tk.LabelFrame(scrollable_frame, text="Modulation Effects", font=("Arial", 10, "bold"))
        mod_frame.pack(pady=5, padx=10, fill=tk.X)

        self.create_slider(mod_frame, 'flanger_delay', 'Flanger Delay', 0, 20, 0, 1)
        self.create_slider(mod_frame, 'flanger_depth', 'Flanger Depth', 0, 10, 0, 1)
        self.create_slider(mod_frame, 'tremolo_freq', 'Tremolo Speed', 0, 20, 0, 1)
        self.create_slider(mod_frame, 'tremolo_depth', 'Tremolo Depth', 0.0, 1.0, 0.0, 0.05)
        self.create_slider(mod_frame, 'vibrato_freq', 'Vibrato Speed', 0, 20, 0, 1)
        self.create_slider(mod_frame, 'vibrato_depth', 'Vibrato Depth', 0.0, 1.0, 0.0, 0.05)
        self.create_slider(mod_frame, 'phaser_speed', 'Phaser Speed', 0.0, 2.0, 0.0, 0.1)
        self.create_slider(mod_frame, 'chorus_depth', 'Chorus Depth', 0, 10, 0, 1)

        # Checkboxes for toggle effects
        toggle_frame = tk.LabelFrame(scrollable_frame, text="Toggle Effects", font=("Arial", 10, "bold"))
        toggle_frame.pack(pady=5, padx=10, fill=tk.X)

        self.reverse_var = tk.BooleanVar(value=False)
        reverse_cb = tk.Checkbutton(toggle_frame, text="Reverse Audio", variable=self.reverse_var,
                                     command=lambda: self.update_param('reverse', self.reverse_var.get()))
        reverse_cb.pack(anchor='w', padx=5)

        self.mono_var = tk.BooleanVar(value=False)
        mono_cb = tk.Checkbutton(toggle_frame, text="Mono (Single Channel)", variable=self.mono_var,
                                  command=lambda: self.update_param('mono', self.mono_var.get()))
        mono_cb.pack(anchor='w', padx=5)

        self.telephone_var = tk.BooleanVar(value=False)
        telephone_cb = tk.Checkbutton(toggle_frame, text="Telephone Effect", variable=self.telephone_var,
                                       command=lambda: self.update_param('telephone', self.telephone_var.get()))
        telephone_cb.pack(anchor='w', padx=5)

        # Reset button
        reset_btn = tk.Button(scrollable_frame, text="Reset All to Default", command=self.reset_effects,
                              bg="#ff9800", fg="white", font=("Arial", 10, "bold"))
        reset_btn.pack(pady=10)

        self.running = True
        self.update_status()

    def create_slider(self, parent, param_key, label, min_val, max_val, default, resolution):
        """Create a labeled slider for an audio parameter."""
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=2)

        lbl = tk.Label(frame, text=label, width=15, anchor='w')
        lbl.pack(side=tk.LEFT)

        if isinstance(default, float):
            var = tk.DoubleVar(value=self.audio_params.get(param_key, default))
        else:
            var = tk.IntVar(value=self.audio_params.get(param_key, default))

        self.slider_vars[param_key] = var

        slider = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                         variable=var, resolution=resolution, length=250,
                         command=lambda v, k=param_key: self.update_param(k, v))
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def update_param(self, key, value):
        """Update an audio parameter."""
        if key in ('reverse', 'mono', 'telephone'):
            self.audio_params[key] = bool(value)
        elif isinstance(value, str):
            try:
                self.audio_params[key] = float(value) if '.' in value else int(value)
            except:
                self.audio_params[key] = float(value)
        else:
            self.audio_params[key] = value

    def apply_preset(self, preset_name):
        """Apply a preset to all audio parameters."""
        if preset_name in self.presets:
            preset = self.presets[preset_name]
            for key, value in preset.items():
                self.audio_params[key] = value
                if key in self.slider_vars:
                    self.slider_vars[key].set(value)
                elif key == 'reverse' and hasattr(self, 'reverse_var'):
                    self.reverse_var.set(value)
                elif key == 'mono' and hasattr(self, 'mono_var'):
                    self.mono_var.set(value)
                elif key == 'telephone' and hasattr(self, 'telephone_var'):
                    self.telephone_var.set(value)
            print(f"Applied preset: {preset_name}")
            self.status_label.config(text=f"Preset applied: {preset_name}")

    def reset_effects(self):
        """Reset all effects to default."""
        self.apply_preset('Normal')

    def build_ffmpeg_filter(self):
        """Build the ffmpeg audio filter chain."""
        filters = []

        # Speed/tempo (atempo only works 0.5-2.0, chain for more extreme values)
        speed = self.audio_params['speed']
        if speed != 1.0:
            # atempo works between 0.5 and 2.0
            temp_speed = speed
            while temp_speed < 0.5:
                filters.append("atempo=0.5")
                temp_speed *= 2
            while temp_speed > 2.0:
                filters.append("atempo=2.0")
                temp_speed /= 2
            if temp_speed != 1.0:
                filters.append(f"atempo={temp_speed}")

        # Pitch (using asetrate+aresample)
        pitch = self.audio_params['pitch']
        if pitch != 1.0:
            new_rate = int(44100 * pitch)
            filters.append(f"asetrate={new_rate},aresample=44100")

        # Bass boost/cut
        bass = self.audio_params['bass']
        if bass != 0:
            filters.append(f"bass=g={bass}")

        # Treble boost/cut
        treble = self.audio_params['treble']
        if treble != 0:
            filters.append(f"treble=g={treble}")

        # Echo
        echo_delay = self.audio_params['echo_delay']
        echo_decay = self.audio_params['echo_decay']
        if echo_delay > 0 and echo_decay > 0:
            filters.append(f"aecho=0.8:0.88:{echo_delay}:{echo_decay}")

        # Reverb (using aecho to simulate)
        reverb = self.audio_params['reverb']
        if reverb > 0:
            reverb_delay = 40 + reverb
            reverb_decay = 0.3 + (reverb / 200)
            filters.append(f"aecho=0.8:0.9:{reverb_delay}|{reverb_delay*1.5}:{reverb_decay}|{reverb_decay*0.7}")

        # Flanger
        flanger_delay = self.audio_params.get('flanger_delay', 0)
        flanger_depth = self.audio_params.get('flanger_depth', 0)
        if flanger_delay > 0 and flanger_depth > 0:
            filters.append(f"flanger=delay={flanger_delay}:depth={flanger_depth}:speed=0.5")

        # Tremolo
        tremolo_freq = self.audio_params.get('tremolo_freq', 0)
        tremolo_depth = self.audio_params.get('tremolo_depth', 0)
        if tremolo_freq > 0 and tremolo_depth > 0:
            filters.append(f"tremolo=f={tremolo_freq}:d={tremolo_depth}")

        # Vibrato
        vibrato_freq = self.audio_params.get('vibrato_freq', 0)
        vibrato_depth = self.audio_params.get('vibrato_depth', 0)
        if vibrato_freq > 0 and vibrato_depth > 0:
            filters.append(f"vibrato=f={vibrato_freq}:d={vibrato_depth}")

        # Phaser
        phaser_speed = self.audio_params.get('phaser_speed', 0)
        if phaser_speed > 0:
            filters.append(f"aphaser=speed={phaser_speed}")

        # Chorus
        chorus_depth = self.audio_params.get('chorus_depth', 0)
        if chorus_depth > 0:
            filters.append(f"chorus=0.5:0.9:{chorus_depth}:0.4:0.25:2")

        # Distortion (using acrusher)
        distortion = self.audio_params['distortion']
        if distortion > 0:
            crush_bits = max(2, 16 - int(distortion / 10))
            filters.append(f"acrusher=bits={crush_bits}:mix={distortion/100}")

        # Telephone effect (bandpass simulation)
        if self.audio_params.get('telephone', False):
            filters.append("highpass=f=400,lowpass=f=3400")

        # Lowpass filter
        lowpass = self.audio_params['lowpass']
        if lowpass < 20000 and not self.audio_params.get('telephone', False):
            filters.append(f"lowpass=f={lowpass}")

        # Highpass filter
        highpass = self.audio_params['highpass']
        if highpass > 20 and not self.audio_params.get('telephone', False):
            filters.append(f"highpass=f={highpass}")

        # Bitcrush
        bitcrush = self.audio_params['bitcrush']
        if bitcrush < 16:
            filters.append(f"acrusher=bits={bitcrush}:mode=log")

        # Add noise/static - we'll handle this separately in apply_audio_effects
        # since it requires mixing two audio sources

        # Mono
        if self.audio_params.get('mono', False):
            filters.append("pan=mono|c0=0.5*c0+0.5*c1")

        # Volume
        volume = self.audio_params['volume']
        if volume != 1.0:
            filters.append(f"volume={volume}")

        # Reverse (should be last for proper effect)
        if self.audio_params['reverse']:
            filters.append("areverse")

        return ",".join(filters) if filters else "anull"

    def apply_audio_effects(self, output_path):
        """Apply current audio effects and save to output path."""
        if not self.original_audio_path or not os.path.exists(self.original_audio_path):
            print("No audio to process")
            return False

        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            print("FFmpeg not found")
            return False

        filter_chain = self.build_ffmpeg_filter()
        noise_amount = self.audio_params.get('noise_amount', 0)

        print(f"Applying audio filter: {filter_chain}")

        try:
            if noise_amount > 0:
                # Use complex filter to mix noise with audio
                noise_vol = noise_amount / 100  # Scale 0-100 to 0-1
                # Get audio duration first
                probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=noprint_wrappers=1:nokey=1", self.original_audio_path]
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                duration = float(probe_result.stdout.strip()) if probe_result.returncode == 0 else 10

                # Build complex filter: apply effects to input, generate noise, mix them
                if filter_chain != "anull":
                    complex_filter = f"[0:a]{filter_chain}[processed];anoisesrc=d={duration}:c=pink:a={noise_vol}[noise];[processed][noise]amix=inputs=2:duration=first:weights=1 {noise_vol}[out]"
                else:
                    complex_filter = f"[0:a]acopy[processed];anoisesrc=d={duration}:c=pink:a={noise_vol}[noise];[processed][noise]amix=inputs=2:duration=first:weights=1 {noise_vol}[out]"

                cmd = [
                    "ffmpeg", "-y",
                    "-i", self.original_audio_path,
                    "-filter_complex", complex_filter,
                    "-map", "[out]",
                    "-ar", "44100",
                    "-ac", "2",
                    output_path
                ]
            else:
                # Simple filter without noise
                cmd = [
                    "ffmpeg", "-y",
                    "-i", self.original_audio_path,
                    "-af", filter_chain,
                    "-ar", "44100",
                    "-ac", "2",
                    output_path
                ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Audio effects applied: {output_path}")
                return True
            else:
                print(f"FFmpeg error: {result.stderr}")
                return False
        except Exception as e:
            print(f"Error applying effects: {e}")
            return False

    def preview_effects(self):
        """Apply effects and preview the result."""
        if self.is_extracting:
            self.status_label.config(text="Still extracting audio...")
            return

        if not self.original_audio_path:
            self.status_label.config(text="No audio available")
            return

        if self.is_processing:
            self.status_label.config(text="Already processing...")
            return

        self.is_processing = True
        self.status_label.config(text="Applying effects...")
        self.root.update()

        # CRITICAL: Stop any playing audio and unload it first to release the file lock
        self.stop_audio()
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.music.unload()
            except:
                pass
            # Small delay to ensure file is released
            time.sleep(0.1)

        # Use unique filename for each preview to avoid file locking issues
        self.preview_counter += 1
        new_preview_path = os.path.join(self.temp_dir, f"preview_audio_{self.preview_counter}.wav")

        # Process synchronously to avoid threading issues with tkinter
        success = self.apply_audio_effects(new_preview_path)

        if success:
            # Clean up old preview file if different
            old_preview = self.preview_audio_path
            self.preview_audio_path = new_preview_path
            if old_preview and old_preview != self.original_audio_path and old_preview != new_preview_path:
                try:
                    os.remove(old_preview)
                except:
                    pass

            self.status_label.config(text="Effects applied!")
            self.is_processing = False
            # Auto-play
            self.play_audio()
        else:
            self.status_label.config(text="Failed to apply effects")
            self.preview_audio_path = self.original_audio_path
            self.is_processing = False

    def play_audio(self):
        """Play the current preview audio."""
        if self.is_extracting:
            self.status_label.config(text="Still extracting audio...")
            return

        if not PYGAME_AVAILABLE:
            self.status_label.config(text="pygame not installed - cannot play")
            return

        audio_path = self.preview_audio_path or self.original_audio_path

        if not audio_path or not os.path.exists(audio_path):
            self.status_label.config(text="No audio to play")
            return

        try:
            self.stop_audio()
            if PYGAME_AVAILABLE:
                try:
                    pygame.mixer.music.unload()
                except:
                    pass
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            self.is_playing = True
            self.status_label.config(text="Playing...")
        except Exception as e:
            print(f"Playback error: {e}")
            self.status_label.config(text=f"Playback error: {e}")

    def stop_audio(self):
        """Stop audio playback and release file lock."""
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
                self.is_playing = False
                self.status_label.config(text="Stopped")
            except:
                pass

    def keep_audio(self):
        """Keep the current effects for final render."""
        # Stop audio first to release any file locks
        self.stop_audio()

        # Determine the source type for status message
        if self.custom_audio_loaded:
            source_type = "Custom Audio"
        else:
            source_type = "Video Audio"

        if not self.preview_audio_path or not os.path.exists(self.preview_audio_path) or self.preview_audio_path == self.original_audio_path:
            # If no preview with effects, apply effects first
            self.processed_audio_path = os.path.join(self.temp_dir, "kept_audio.wav")
            if self.apply_audio_effects(self.processed_audio_path):
                self.keep_status.config(text=f"Using: {source_type} (Effects Applied)")
                print("Audio effects kept for final render")
            else:
                self.keep_status.config(text=f"Using: {source_type} (No effects)")
                # Still keep the source audio even if effects failed
                try:
                    shutil.copy(self.original_audio_path, self.processed_audio_path)
                except:
                    self.processed_audio_path = None
        else:
            # Copy preview to kept
            self.processed_audio_path = os.path.join(self.temp_dir, "kept_audio.wav")
            try:
                shutil.copy(self.preview_audio_path, self.processed_audio_path)
                self.keep_status.config(text=f"Using: {source_type} (Effects Applied)")
                print("Audio effects kept for final render")
            except Exception as e:
                print(f"Error keeping audio: {e}")
                self.keep_status.config(text=f"Using: {source_type} (Copy failed)")
                self.processed_audio_path = None

    def get_audio_for_render(self):
        """Get the audio path to use for final render."""
        if self.processed_audio_path and os.path.exists(self.processed_audio_path):
            return self.processed_audio_path
        return None  # Use original from video

    def update_status(self):
        """Update the status label based on current state."""
        if self.is_extracting:
            self.status_label.config(text="Extracting audio from video...")
        elif self.original_audio_path and os.path.exists(self.original_audio_path):
            self.status_label.config(text="Audio ready - click Play or Preview Effects")
        else:
            self.status_label.config(text="No audio loaded")

    def update(self):
        """Update the panel (call from main loop)."""
        if self.root and self.running:
            try:
                if self.is_extracting:
                    self.status_label.config(text="Extracting audio...")
                elif self.original_audio_path and os.path.exists(self.original_audio_path):
                    if not self.is_playing and "Extracting" in self.status_label.cget("text"):
                        self.status_label.config(text="Audio ready")
                self.root.update_idletasks()
                self.root.update()
            except tk.TclError:
                self.running = False

    def close(self):
        """Close the panel and cleanup."""
        self.stop_audio()
        if self.root:
            self.running = False
            try:
                self.root.destroy()
            except:
                pass
        # Cleanup temp files
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
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

    if effect_states['rotation_enabled']:
        output_frame = effects.rotate_frame(
            output_frame, frame_number,
            rotation_speed=effect_params['rotation_speed']
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

# ----------------------------
# Initialize Audio Effects Panel (only for video files)
# ----------------------------
audio_panel = None
if is_video_file:
    audio_panel = AudioEffectsPanel(source_path)
    audio_panel.set_video_source(source_path)  # ← ADD THIS LINE
    audio_panel.create_panel()

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

    # Now merge audio using ffmpeg
    control_panel.update_render_progress(95, "Adding audio...")

    # Check if we have processed audio from the audio panel
    processed_audio = None
    if audio_panel:
        processed_audio = audio_panel.get_audio_for_render()

    if processed_audio:
        print(f"Using processed audio: {processed_audio}")
    else:
        print("Using original audio from video...")

    # Check if ffmpeg is available
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            if processed_audio and os.path.exists(processed_audio):
                # Use processed audio from audio panel
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", temp_video_path,      # Input: rendered video (no audio)
                    "-i", processed_audio,       # Input: processed audio file
                    "-c:v", "copy",              # Copy video stream as-is
                    "-c:a", "aac",               # Encode audio as AAC
                    "-map", "0:v:0",             # Use video from first input
                    "-map", "1:a:0",             # Use audio from second input
                    "-shortest",                 # Match shortest stream
                    output_path
                ]
            else:
                # Use original audio from video
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
    if was_key_just_pressed("+"):
        audio_panel.preview_effects()
    if was_key_just_pressed("-"):
        audio_panel.stop_audio()


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

    # Update audio panel GUI (if exists)
    if audio_panel:
        audio_panel.update()

    # Quit application
    wait_time = frame_delay if is_video_file else 1
    if cv2.waitKey(wait_time) & 0xFF == ord("q"):
        break

    # Apply preview scaling for performance
    preview_scale = control_panel.preview_scale_var.get() if hasattr(control_panel, 'preview_scale_var') else 1.0
    if preview_scale < 1.0:
        display_height = int(output_frame.shape[0] * preview_scale)
        display_width = int(output_frame.shape[1] * preview_scale)
        display_frame = cv2.resize(output_frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
    else:
        display_frame = output_frame

    cv2.imshow("glitch mirror", display_frame)

# ----------------------------
# Cleanup
# ----------------------------
control_panel.close()
if audio_panel:
    audio_panel.close()
camera.release()
cv2.destroyAllWindows()
