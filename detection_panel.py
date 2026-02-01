"""
Detection Control Panel for Glitch Mirror
Provides UI controls for face/body detection and selective effects.
"""

import tkinter as tk
from tkinter import ttk
import os


class DetectionControlPanel:
    """UI Panel for controlling object detection and selective effects."""

    def __init__(self, detection_params):
        """
        Initialize the detection control panel.

        Args:
            detection_params: dict reference that will be updated with detection settings
        """
        self.root = None
        self.detection_params = detection_params
        self.created = False

        # UI variable references
        self.vars = {}

    def create_panel(self):
        """Create the detection control panel window."""
        if self.created:
            return

        self.root = tk.Toplevel()
        self.root.title("Detection & Selective Effects")
        self.root.geometry("380x700")
        self.root.minsize(350, 500)

        # Make it stay on top optionally
        # self.root.attributes('-topmost', True)

        # Create main scrollable frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Build UI sections
        self._create_detection_settings()
        self._create_face_effects()
        self._create_mask_settings()
        self._create_presets()

        self.created = True

    def _create_section_header(self, parent, text):
        """Create a styled section header."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(10, 5), padx=5)

        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=2)
        label = ttk.Label(frame, text=text, font=('Arial', 10, 'bold'))
        label.pack(anchor='w')

        return frame

    def _create_detection_settings(self):
        """Create detection enable/disable controls."""
        self._create_section_header(self.scrollable_frame, "Detection Settings")

        frame = ttk.Frame(self.scrollable_frame)
        frame.pack(fill=tk.X, padx=10, pady=5)

        # Enable detection master toggle
        self.vars['detection_enabled'] = tk.BooleanVar(value=self.detection_params.get('detection_enabled', False))
        ttk.Checkbutton(
            frame, text="Enable Detection",
            variable=self.vars['detection_enabled'],
            command=lambda: self._update_param('detection_enabled', self.vars['detection_enabled'].get())
        ).pack(anchor='w')

        # Face detection
        self.vars['detect_faces'] = tk.BooleanVar(value=self.detection_params.get('detect_faces', True))
        ttk.Checkbutton(
            frame, text="Detect Faces",
            variable=self.vars['detect_faces'],
            command=lambda: self._update_param('detect_faces', self.vars['detect_faces'].get())
        ).pack(anchor='w', padx=20)

        # Eye detection
        self.vars['detect_eyes'] = tk.BooleanVar(value=self.detection_params.get('detect_eyes', False))
        ttk.Checkbutton(
            frame, text="Detect Eyes",
            variable=self.vars['detect_eyes'],
            command=lambda: self._update_param('detect_eyes', self.vars['detect_eyes'].get())
        ).pack(anchor='w', padx=20)

        # Body detection
        self.vars['detect_bodies'] = tk.BooleanVar(value=self.detection_params.get('detect_bodies', False))
        ttk.Checkbutton(
            frame, text="Detect Bodies (Full Body)",
            variable=self.vars['detect_bodies'],
            command=lambda: self._update_param('detect_bodies', self.vars['detect_bodies'].get())
        ).pack(anchor='w', padx=20)

        # Show detection boxes (debug)
        self.vars['show_detection_boxes'] = tk.BooleanVar(value=self.detection_params.get('show_detection_boxes', False))
        ttk.Checkbutton(
            frame, text="Show Detection Boxes (Debug)",
            variable=self.vars['show_detection_boxes'],
            command=lambda: self._update_param('show_detection_boxes', self.vars['show_detection_boxes'].get())
        ).pack(anchor='w')

        # Detection sensitivity
        sens_frame = ttk.Frame(frame)
        sens_frame.pack(fill=tk.X, pady=5)
        ttk.Label(sens_frame, text="Detection Sensitivity:").pack(anchor='w')

        self.vars['detection_sensitivity'] = tk.DoubleVar(value=self.detection_params.get('detection_sensitivity', 1.1))
        sensitivity_scale = ttk.Scale(
            sens_frame, from_=1.05, to=1.5,
            variable=self.vars['detection_sensitivity'],
            orient='horizontal',
            command=lambda v: self._update_param('detection_sensitivity', float(v))
        )
        sensitivity_scale.pack(fill=tk.X)
        ttk.Label(sens_frame, text="(Lower = more sensitive, may have false positives)").pack(anchor='w')

        # Min neighbors
        neighbors_frame = ttk.Frame(frame)
        neighbors_frame.pack(fill=tk.X, pady=5)
        ttk.Label(neighbors_frame, text="Min Neighbors (accuracy):").pack(anchor='w')

        self.vars['min_neighbors'] = tk.IntVar(value=self.detection_params.get('min_neighbors', 5))
        neighbors_scale = ttk.Scale(
            neighbors_frame, from_=1, to=10,
            variable=self.vars['min_neighbors'],
            orient='horizontal',
            command=lambda v: self._update_param('min_neighbors', int(float(v)))
        )
        neighbors_scale.pack(fill=tk.X)

        # Use tracking (smoothing)
        self.vars['use_tracking'] = tk.BooleanVar(value=self.detection_params.get('use_tracking', True))
        ttk.Checkbutton(
            frame, text="Smooth Tracking (reduce jitter)",
            variable=self.vars['use_tracking'],
            command=lambda: self._update_param('use_tracking', self.vars['use_tracking'].get())
        ).pack(anchor='w')

    def _create_face_effects(self):
        """Create face-specific effect controls."""
        self._create_section_header(self.scrollable_frame, "Detection-Based Effects")

        frame = ttk.Frame(self.scrollable_frame)
        frame.pack(fill=tk.X, padx=10, pady=5)

        # Effect mode selection
        ttk.Label(frame, text="Face Effect Mode:").pack(anchor='w')

        self.vars['face_effect_mode'] = tk.StringVar(value=self.detection_params.get('face_effect_mode', 'none'))

        effect_modes = [
            ('none', 'None (Detection Only)'),
            ('pixelate', 'Pixelate Faces'),
            ('blur', 'Blur Faces'),
            ('glitch', 'Glitch Faces'),
            ('thermal', 'Thermal Vision'),
            ('neon_outline', 'Neon Outline'),
            ('cartoon', 'Cartoon Effect'),
            ('color_shift', 'Color Shift'),
            ('edge_highlight', 'Edge Highlight'),
            ('vignette', 'Face Vignette'),
            ('background_replace', 'Background Replace'),
            ('apply_all_effects', 'Apply All Effects to Faces'),
            ('apply_all_effects_inverted', 'Apply All Effects Outside Faces'),
        ]

        for value, text in effect_modes:
            ttk.Radiobutton(
                frame, text=text, value=value,
                variable=self.vars['face_effect_mode'],
                command=lambda: self._update_param('face_effect_mode', self.vars['face_effect_mode'].get())
            ).pack(anchor='w', padx=10)

        # Effect-specific parameters
        param_frame = ttk.LabelFrame(frame, text="Effect Parameters")
        param_frame.pack(fill=tk.X, pady=10)

        # Pixelate size
        px_frame = ttk.Frame(param_frame)
        px_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(px_frame, text="Pixelate Size:").pack(side='left')
        self.vars['face_pixelate_size'] = tk.IntVar(value=self.detection_params.get('face_pixelate_size', 10))
        ttk.Scale(
            px_frame, from_=4, to=30,
            variable=self.vars['face_pixelate_size'],
            orient='horizontal',
            command=lambda v: self._update_param('face_pixelate_size', int(float(v)))
        ).pack(side='right', fill=tk.X, expand=True)

        # Blur amount
        blur_frame = ttk.Frame(param_frame)
        blur_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(blur_frame, text="Blur Amount:").pack(side='left')
        self.vars['face_blur_amount'] = tk.IntVar(value=self.detection_params.get('face_blur_amount', 51))
        ttk.Scale(
            blur_frame, from_=11, to=101,
            variable=self.vars['face_blur_amount'],
            orient='horizontal',
            command=lambda v: self._update_param('face_blur_amount', int(float(v)) | 1)  # Ensure odd
        ).pack(side='right', fill=tk.X, expand=True)

        # Glitch intensity
        glitch_frame = ttk.Frame(param_frame)
        glitch_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(glitch_frame, text="Glitch Intensity:").pack(side='left')
        self.vars['face_glitch_intensity'] = tk.IntVar(value=self.detection_params.get('face_glitch_intensity', 20))
        ttk.Scale(
            glitch_frame, from_=5, to=50,
            variable=self.vars['face_glitch_intensity'],
            orient='horizontal',
            command=lambda v: self._update_param('face_glitch_intensity', int(float(v)))
        ).pack(side='right', fill=tk.X, expand=True)

        # Color shift hue
        hue_frame = ttk.Frame(param_frame)
        hue_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(hue_frame, text="Hue Shift:").pack(side='left')
        self.vars['face_hue_shift'] = tk.IntVar(value=self.detection_params.get('face_hue_shift', 90))
        ttk.Scale(
            hue_frame, from_=0, to=180,
            variable=self.vars['face_hue_shift'],
            orient='horizontal',
            command=lambda v: self._update_param('face_hue_shift', int(float(v)))
        ).pack(side='right', fill=tk.X, expand=True)

        # Neon color picker
        neon_frame = ttk.Frame(param_frame)
        neon_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(neon_frame, text="Neon Color:").pack(side='left')

        self.vars['neon_color'] = tk.StringVar(value=self.detection_params.get('neon_color', 'cyan'))
        neon_colors = [('cyan', 'Cyan'), ('magenta', 'Magenta'), ('yellow', 'Yellow'),
                       ('green', 'Green'), ('red', 'Red'), ('white', 'White')]

        neon_combo = ttk.Combobox(neon_frame, textvariable=self.vars['neon_color'],
                                   values=[c[0] for c in neon_colors], state='readonly', width=10)
        neon_combo.pack(side='right')
        neon_combo.bind('<<ComboboxSelected>>',
                        lambda e: self._update_param('neon_color', self.vars['neon_color'].get()))

    def _create_mask_settings(self):
        """Create mask/blending settings."""
        self._create_section_header(self.scrollable_frame, "Mask Settings")

        frame = ttk.Frame(self.scrollable_frame)
        frame.pack(fill=tk.X, padx=10, pady=5)

        # Mask type
        ttk.Label(frame, text="Mask Shape:").pack(anchor='w')
        self.vars['mask_type'] = tk.StringVar(value=self.detection_params.get('mask_type', 'ellipse'))

        ttk.Radiobutton(
            frame, text="Ellipse (smooth)", value='ellipse',
            variable=self.vars['mask_type'],
            command=lambda: self._update_param('mask_type', self.vars['mask_type'].get())
        ).pack(anchor='w', padx=10)

        ttk.Radiobutton(
            frame, text="Rectangle (sharp)", value='rectangle',
            variable=self.vars['mask_type'],
            command=lambda: self._update_param('mask_type', self.vars['mask_type'].get())
        ).pack(anchor='w', padx=10)

        # Padding
        pad_frame = ttk.Frame(frame)
        pad_frame.pack(fill=tk.X, pady=5)
        ttk.Label(pad_frame, text="Mask Padding:").pack(side='left')
        self.vars['mask_padding'] = tk.IntVar(value=self.detection_params.get('mask_padding', 20))
        ttk.Scale(
            pad_frame, from_=0, to=100,
            variable=self.vars['mask_padding'],
            orient='horizontal',
            command=lambda v: self._update_param('mask_padding', int(float(v)))
        ).pack(side='right', fill=tk.X, expand=True)

        # Feather
        feather_frame = ttk.Frame(frame)
        feather_frame.pack(fill=tk.X, pady=5)
        ttk.Label(feather_frame, text="Edge Feather:").pack(side='left')
        self.vars['mask_feather'] = tk.IntVar(value=self.detection_params.get('mask_feather', 20))
        ttk.Scale(
            feather_frame, from_=0, to=50,
            variable=self.vars['mask_feather'],
            orient='horizontal',
            command=lambda v: self._update_param('mask_feather', int(float(v)))
        ).pack(side='right', fill=tk.X, expand=True)

        # Background color (for background replace)
        bg_frame = ttk.LabelFrame(frame, text="Background Replace Color")
        bg_frame.pack(fill=tk.X, pady=5)

        self.vars['bg_color'] = tk.StringVar(value=self.detection_params.get('bg_color', 'black'))
        bg_colors = ['black', 'white', 'green', 'blue', 'red', 'gray']

        for color in bg_colors:
            ttk.Radiobutton(
                bg_frame, text=color.title(), value=color,
                variable=self.vars['bg_color'],
                command=lambda: self._update_param('bg_color', self.vars['bg_color'].get())
            ).pack(side='left', padx=5)

    def _create_presets(self):
        """Create preset buttons."""
        self._create_section_header(self.scrollable_frame, "Quick Presets")

        frame = ttk.Frame(self.scrollable_frame)
        frame.pack(fill=tk.X, padx=10, pady=5)

        presets = [
            ("Privacy Blur", self._apply_privacy_preset),
            ("Glitch Portrait", self._apply_glitch_preset),
            ("Neon Face", self._apply_neon_preset),
            ("Thermal Scan", self._apply_thermal_preset),
            ("Green Screen", self._apply_greenscreen_preset),
            ("Cartoon Face", self._apply_cartoon_preset),
            ("Reset All", self._reset_preset),
        ]

        # Create 2-column grid
        for i, (name, callback) in enumerate(presets):
            row = i // 2
            col = i % 2
            btn = ttk.Button(frame, text=name, command=callback, width=15)
            btn.grid(row=row, column=col, padx=5, pady=2)

    def _update_param(self, key, value):
        """Update a detection parameter."""
        self.detection_params[key] = value

    def _apply_privacy_preset(self):
        """Apply privacy/anonymization preset."""
        self.vars['detection_enabled'].set(True)
        self.vars['detect_faces'].set(True)
        self.vars['face_effect_mode'].set('blur')
        self.vars['face_blur_amount'].set(71)
        self._sync_all_params()

    def _apply_glitch_preset(self):
        """Apply glitch face preset."""
        self.vars['detection_enabled'].set(True)
        self.vars['detect_faces'].set(True)
        self.vars['face_effect_mode'].set('glitch')
        self.vars['face_glitch_intensity'].set(30)
        self._sync_all_params()

    def _apply_neon_preset(self):
        """Apply neon outline preset."""
        self.vars['detection_enabled'].set(True)
        self.vars['detect_faces'].set(True)
        self.vars['face_effect_mode'].set('neon_outline')
        self.vars['neon_color'].set('cyan')
        self._sync_all_params()

    def _apply_thermal_preset(self):
        """Apply thermal vision preset."""
        self.vars['detection_enabled'].set(True)
        self.vars['detect_faces'].set(True)
        self.vars['face_effect_mode'].set('thermal')
        self._sync_all_params()

    def _apply_greenscreen_preset(self):
        """Apply green screen preset."""
        self.vars['detection_enabled'].set(True)
        self.vars['detect_faces'].set(True)
        self.vars['face_effect_mode'].set('background_replace')
        self.vars['bg_color'].set('green')
        self.vars['mask_padding'].set(30)
        self.vars['mask_feather'].set(30)
        self._sync_all_params()

    def _apply_cartoon_preset(self):
        """Apply cartoon face preset."""
        self.vars['detection_enabled'].set(True)
        self.vars['detect_faces'].set(True)
        self.vars['face_effect_mode'].set('cartoon')
        self._sync_all_params()

    def _reset_preset(self):
        """Reset all detection settings."""
        self.vars['detection_enabled'].set(False)
        self.vars['detect_faces'].set(True)
        self.vars['detect_eyes'].set(False)
        self.vars['detect_bodies'].set(False)
        self.vars['show_detection_boxes'].set(False)
        self.vars['face_effect_mode'].set('none')
        self.vars['detection_sensitivity'].set(1.1)
        self.vars['min_neighbors'].set(5)
        self.vars['use_tracking'].set(True)
        self.vars['mask_type'].set('ellipse')
        self.vars['mask_padding'].set(20)
        self.vars['mask_feather'].set(20)
        self._sync_all_params()

    def _sync_all_params(self):
        """Sync all UI variables to the detection_params dict."""
        for key, var in self.vars.items():
            self.detection_params[key] = var.get()

    def update(self):
        """Update the panel (call from main loop)."""
        if self.root:
            try:
                self.root.update()
            except tk.TclError:
                pass

    def close(self):
        """Close the panel."""
        if self.root:
            try:
                self.root.destroy()
            except:
                pass
            self.root = None
            self.created = False
