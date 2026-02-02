"""
Detection Control Panel for Glitch Mirror
Provides UI controls for face/body detection and selective effects.
Includes per-region independent effect stacks and video texture support.
"""

import tkinter as tk
from tkinter import ttk, filedialog
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

        # Region effect manager reference (will be set from main.py)
        self.region_effect_manager = None

        # GIF exporter reference (will be set from main.py)
        self.gif_exporter = None

        # Detection tracker reference (will be set from main.py)
        self.detection_tracker = None

        # Data visualization reference (will be set from main.py)
        self.data_viz = None

        # Per-region UI controls
        self.region_controls = {}
        self.current_region_id = None

    def create_panel(self):
        """Create the detection control panel window."""
        if self.created:
            return

        self.root = tk.Toplevel()
        self.root.title("Detection & Effects")
        self.root.geometry("400x700")
        self.root.minsize(300, 400)

        # Create main scrollable frame with proper resize handling
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Use grid for better resize behavior
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        # Create window in canvas
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Configure scroll region when frame size changes
        def _configure_scroll(event):
            try:
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            except tk.TclError:
                pass

        # Configure canvas width when window resizes
        def _configure_canvas(event):
            try:
                # Make scrollable frame fill canvas width
                self.canvas.itemconfig(self.canvas_window, width=event.width)
            except tk.TclError:
                pass

        self.scrollable_frame.bind("<Configure>", _configure_scroll)
        self.canvas.bind("<Configure>", _configure_canvas)

        self.canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Enable mouse wheel scrolling (only when over this window)
        def _on_mousewheel(event):
            try:
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except tk.TclError:
                pass

        def _bind_mousewheel(event):
            self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event):
            self.canvas.unbind_all("<MouseWheel>")

        self.canvas.bind("<Enter>", _bind_mousewheel)
        self.canvas.bind("<Leave>", _unbind_mousewheel)

        # Create notebook (tabs) for better organization
        self.notebook = ttk.Notebook(self.scrollable_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Detection Settings
        self.tab_detection = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_detection, text="🎯 Detection")

        # Tab 2: Effects
        self.tab_effects = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_effects, text="✨ Effects")

        # Tab 3: Visualization
        self.tab_viz = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_viz, text="📊 Visualization")

        # Tab 4: Tools (Custom Masks, GIF, Presets)
        self.tab_tools = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_tools, text="🔧 Tools")

        # Build each tab
        self._create_detection_tab()
        self._create_effects_tab()
        self._create_viz_tab()
        self._create_tools_tab()

        self.created = True

    def _create_section_header(self, parent, text, collapsed=False):
        """Create a styled section header with optional collapse."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(8, 3), padx=3)

        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=2)
        label = ttk.Label(frame, text=text, font=('Arial', 9, 'bold'))
        label.pack(anchor='w')

        return frame

    def _create_detection_tab(self):
        """Create the Detection Settings tab."""
        parent = self.tab_detection

        # === IMAGE MODE: Single-shot detection ===
        image_frame = ttk.LabelFrame(parent, text="📷 Image Mode (Single-Shot Detection)", padding=5)
        image_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(image_frame, text="For static images - detect once instead of continuous",
                  font=('Arial', 8, 'italic')).pack(anchor='w')

        # Single-shot mode toggle
        self.vars['single_shot_mode'] = tk.BooleanVar(value=self.detection_params.get('single_shot_mode', False))
        ttk.Checkbutton(
            image_frame, text="Enable Single-Shot Mode (for images)",
            variable=self.vars['single_shot_mode'],
            command=lambda: self._update_param('single_shot_mode', self.vars['single_shot_mode'].get())
        ).pack(anchor='w')

        # Detect button row
        detect_btn_row = ttk.Frame(image_frame)
        detect_btn_row.pack(fill=tk.X, pady=5)

        self.detect_now_btn = ttk.Button(detect_btn_row, text="🔍 Detect Now",
                                          command=self._trigger_detection, width=12)
        self.detect_now_btn.pack(side='left', padx=2)

        ttk.Button(detect_btn_row, text="🔒 Keep Good",
                   command=self._keep_selected_detections, width=10).pack(side='left', padx=2)

        ttk.Button(detect_btn_row, text="🗑 Clear All",
                   command=self._clear_all_detections, width=10).pack(side='left', padx=2)

        # Detection count status
        self.single_shot_status = ttk.Label(image_frame, text="No detections yet", font=('Arial', 8))
        self.single_shot_status.pack(anchor='w')

        # === Detection types ===
        types_frame = ttk.LabelFrame(parent, text="What to Detect", padding=5)
        types_frame.pack(fill=tk.X, padx=5, pady=5)

        detection_types = [
            ('detect_faces', '😀 Faces', True),
            ('detect_eyes', '👁 Eyes', False),
            ('detect_bodies', '🧍 Full Body', False),
            ('detect_upper_body', '👤 Upper Body', False),
            ('detect_smiles', '😊 Smiles', False),
            ('detect_cats', '🐱 Cat Faces', False),
            ('detect_plates', '🚗 License Plates', False),
        ]

        for i, (key, label, default) in enumerate(detection_types):
            self.vars[key] = tk.BooleanVar(value=self.detection_params.get(key, default))
            cb = ttk.Checkbutton(
                types_frame, text=label,
                variable=self.vars[key],
                command=lambda k=key: self._update_param(k, self.vars[k].get())
            )
            cb.grid(row=i // 2, column=i % 2, sticky='w', padx=5, pady=1)

        # === Detection Sensitivity ===
        sens_frame = ttk.LabelFrame(parent, text="Detection Sensitivity", padding=5)
        sens_frame.pack(fill=tk.X, padx=5, pady=5)

        # Sensitivity slider
        sens_row = ttk.Frame(sens_frame)
        sens_row.pack(fill=tk.X)
        ttk.Label(sens_row, text="Sensitivity:").pack(side='left')
        self.vars['detection_sensitivity'] = tk.DoubleVar(value=self.detection_params.get('detection_sensitivity', 1.1))
        ttk.Scale(sens_row, from_=1.05, to=1.5, variable=self.vars['detection_sensitivity'],
                  orient='horizontal',
                  command=self._on_sensitivity_changed).pack(side='left', fill=tk.X, expand=True)
        self.sens_value_label = ttk.Label(sens_row, text=f"{self.detection_params.get('detection_sensitivity', 1.1):.2f}", width=5)
        self.sens_value_label.pack(side='right')

        ttk.Label(sens_frame, text="Lower = more sensitive (more detections, more false positives)",
                  font=('Arial', 8, 'italic')).pack(anchor='w')

        # Min neighbors slider
        neighbors_row = ttk.Frame(sens_frame)
        neighbors_row.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(neighbors_row, text="Min Neighbors:").pack(side='left')
        self.vars['min_neighbors'] = tk.IntVar(value=self.detection_params.get('min_neighbors', 5))
        ttk.Scale(neighbors_row, from_=1, to=10, variable=self.vars['min_neighbors'],
                  orient='horizontal',
                  command=self._on_min_neighbors_changed).pack(side='left', fill=tk.X, expand=True)
        self.neighbors_value_label = ttk.Label(neighbors_row, text=str(self.detection_params.get('min_neighbors', 5)), width=3)
        self.neighbors_value_label.pack(side='right')

        ttk.Label(sens_frame, text="Higher = more strict (fewer false positives)",
                  font=('Arial', 8, 'italic')).pack(anchor='w')

        # === Debug & Performance (for video mode) ===
        debug_frame = ttk.LabelFrame(parent, text="Video Mode Options", padding=5)
        debug_frame.pack(fill=tk.X, padx=5, pady=5)

        self.vars['detection_enabled'] = tk.BooleanVar(value=self.detection_params.get('detection_enabled', False))
        ttk.Checkbutton(
            debug_frame, text="🔍 Enable Continuous Detection (for video)",
            variable=self.vars['detection_enabled'],
            command=lambda: self._update_param('detection_enabled', self.vars['detection_enabled'].get())
        ).pack(anchor='w')

        self.vars['show_detection_boxes'] = tk.BooleanVar(value=self.detection_params.get('show_detection_boxes', False))
        ttk.Checkbutton(
            debug_frame, text="Show Detection Boxes",
            variable=self.vars['show_detection_boxes'],
            command=lambda: self._update_param('show_detection_boxes', self.vars['show_detection_boxes'].get())
        ).pack(anchor='w')

        self.vars['use_tracking'] = tk.BooleanVar(value=self.detection_params.get('use_tracking', True))
        ttk.Checkbutton(
            debug_frame, text="Smooth Tracking (reduce jitter)",
            variable=self.vars['use_tracking'],
            command=lambda: self._update_param('use_tracking', self.vars['use_tracking'].get())
        ).pack(anchor='w')

        # Frame skip slider
        ttk.Label(debug_frame, text="Frame Skip (higher = faster):", font=('Arial', 8)).pack(anchor='w', pady=(5, 0))
        self.vars['detection_frame_skip'] = tk.IntVar(value=self.detection_params.get('detection_frame_skip', 3))
        skip_frame = ttk.Frame(debug_frame)
        skip_frame.pack(fill=tk.X)
        ttk.Scale(skip_frame, from_=1, to=10, variable=self.vars['detection_frame_skip'],
                  orient='horizontal', command=self._update_frame_skip).pack(side='left', fill=tk.X, expand=True)
        self.frame_skip_label = ttk.Label(skip_frame, text="3", width=3)
        self.frame_skip_label.pack(side='right')

        # === Detection Management ===
        self._create_detection_management_section(parent)

    def _create_detection_management_section(self, parent):
        """Create the detection management section (ignore/select/keep)."""
        mgmt_frame = ttk.LabelFrame(parent, text="Manage Detections", padding=5)
        mgmt_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(mgmt_frame, text="Select detections to edit effects, ignore false positives, or keep good ones",
                  font=('Arial', 8, 'italic'), wraplength=350).pack(anchor='w')

        # Listbox
        list_frame = ttk.Frame(mgmt_frame)
        list_frame.pack(fill=tk.X, pady=5)

        self.detection_listbox = tk.Listbox(list_frame, height=6, selectmode=tk.EXTENDED, font=('Consolas', 9))
        self.detection_listbox.pack(side='left', fill=tk.X, expand=True)

        list_scroll = ttk.Scrollbar(list_frame, orient='vertical', command=self.detection_listbox.yview)
        list_scroll.pack(side='right', fill='y')
        self.detection_listbox.config(yscrollcommand=list_scroll.set)

        # Buttons row 1: Basic actions
        btn_row1 = ttk.Frame(mgmt_frame)
        btn_row1.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row1, text="🔄 Refresh", command=self._refresh_detection_list, width=9).pack(side='left', padx=1)
        ttk.Button(btn_row1, text="📌 Select", command=self._select_detections, width=9).pack(side='left', padx=1)
        ttk.Button(btn_row1, text="📍 Deselect", command=self._deselect_detections, width=9).pack(side='left', padx=1)

        # Buttons row 2: Keep/Ignore
        btn_row2 = ttk.Frame(mgmt_frame)
        btn_row2.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row2, text="🔒 Keep", command=self._keep_selected_detections, width=9).pack(side='left', padx=1)
        ttk.Button(btn_row2, text="🚫 Ignore", command=self._ignore_selected_detections, width=9).pack(side='left', padx=1)
        ttk.Button(btn_row2, text="🔗 Link", command=self._connect_selected, width=9).pack(side='left', padx=1)

        # Buttons row 3: Clear actions
        btn_row3 = ttk.Frame(mgmt_frame)
        btn_row3.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row3, text="Unkeep", command=self._unkeep_selected_detections, width=9).pack(side='left', padx=1)
        ttk.Button(btn_row3, text="Unignore", command=self._unignore_selected_detections, width=9).pack(side='left', padx=1)
        ttk.Button(btn_row3, text="Clear Sel", command=self._clear_selection, width=9).pack(side='left', padx=1)

        # Status
        self.detection_status_label = ttk.Label(mgmt_frame, text="0 active, 0 kept, 0 ignored, 0 selected", font=('Arial', 8))
        self.detection_status_label.pack(anchor='w')

        # Tip for workflow
        ttk.Label(mgmt_frame, text="💡 Tip: Select faces → go to Effects tab to apply effects",
                  font=('Arial', 8, 'italic'), foreground='gray').pack(anchor='w', pady=(5, 0))

    def _create_effects_tab(self):
        """Create the Effects tab - shows only selected detections for editing."""
        parent = self.tab_effects

        # === Selected Detections for Editing ===
        selected_frame = ttk.LabelFrame(parent, text="📌 Selected Detections", padding=5)
        selected_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(selected_frame, text="Select faces in Detection tab, they appear here for editing",
                  font=('Arial', 8, 'italic')).pack(anchor='w')

        # Selected detections listbox (shows faces and their parts)
        sel_list_frame = ttk.Frame(selected_frame)
        sel_list_frame.pack(fill=tk.X, pady=5)

        self.selected_detections_listbox = tk.Listbox(sel_list_frame, height=5, selectmode=tk.SINGLE, font=('Consolas', 9))
        self.selected_detections_listbox.pack(side='left', fill=tk.X, expand=True)
        self.selected_detections_listbox.bind('<<ListboxSelect>>', self._on_effect_detection_selected)

        sel_list_scroll = ttk.Scrollbar(sel_list_frame, orient='vertical', command=self.selected_detections_listbox.yview)
        sel_list_scroll.pack(side='right', fill='y')
        self.selected_detections_listbox.config(yscrollcommand=sel_list_scroll.set)

        # Refresh button
        ttk.Button(selected_frame, text="🔄 Refresh Selected List", command=self._refresh_selected_for_effects).pack(anchor='w', pady=2)

        self.selected_effect_status = ttk.Label(selected_frame, text="0 selected for editing", font=('Arial', 8))
        self.selected_effect_status.pack(anchor='w')

        # =============================================
        # FACE EFFECTS SECTION - Applies to selected faces only
        # =============================================
        face_section = ttk.LabelFrame(parent, text="🎭 FACE EFFECTS (Selected Faces Only)", padding=5)
        face_section.pack(fill=tk.X, padx=5, pady=5)

        # Enable/Show face effects toggle
        face_toggle_row = ttk.Frame(face_section)
        face_toggle_row.pack(fill=tk.X, pady=2)

        self.vars['show_face_effects'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            face_toggle_row, text="👁 Show Face Effects (disable for performance)",
            variable=self.vars['show_face_effects'],
            command=self._toggle_face_effects
        ).pack(side='left')

        # Status label
        self.face_effect_status = ttk.Label(face_section, text="No face selected",
                                            font=('Arial', 9, 'bold'), foreground='gray')
        self.face_effect_status.pack(anchor='w', pady=(5, 2))

        # Live Face Effect Controls
        live_effect_frame = ttk.Frame(face_section)
        live_effect_frame.pack(fill=tk.X, pady=5)

        # Effect type dropdown - changes apply LIVE
        effect_row = ttk.Frame(live_effect_frame)
        effect_row.pack(fill=tk.X, pady=3)
        ttk.Label(effect_row, text="Effect:").pack(side='left')
        self.vars['face_effect_type'] = tk.StringVar(value='none')

        # Complete list of all available effects
        all_effects = [
            'none',
            'restore_original',  # Special: removes all effects, shows base image
            # Basic effects
            'pixelate', 'blur', 'glitch', 'thermal', 'negative',
            # Artistic
            'cartoon', 'sketch', 'oil_paint', 'emboss', 'posterize',
            # Color effects
            'color_shift', 'rgb_shift', 'chromatic', 'duotone', 'cross_process',
            # Distortion
            'wave_distort', 'heat_distort', 'drunk', 'spiral_warp', 'kaleidoscope', 'mirror',
            # Glow/Edge
            'neon_glow', 'edge_glow', 'edge_highlight', 'halftone',
            # Noise/Grain
            'vhs', 'film_grain', 'tv_static', 'blocky_noise',
            # Vintage/Retro
            'scanlines', 'retro_crt',
            # Motion/Time
            'ghost_trail', 'motion_blur', 'radial_blur', 'zoom_blur',
            # Special
            'tunnel_vision', 'double_vision', 'prism', 'rgb_split_radial',
            'glitch_blocks', 'glitch_shift',
        ]
        self.face_effect_combo = ttk.Combobox(
            effect_row, textvariable=self.vars['face_effect_type'],
            values=all_effects,
            state='readonly', width=18)
        self.face_effect_combo.pack(side='left', padx=5)
        self.face_effect_combo.bind('<<ComboboxSelected>>', self._on_face_effect_changed)

        # === Dynamic Parameter Sliders Frame ===
        # This frame holds sliders that change based on selected effect
        self.face_params_frame = ttk.Frame(live_effect_frame)
        self.face_params_frame.pack(fill=tk.X, pady=5)

        # Define all possible parameters with their ranges and defaults
        # Format: param_name: (label, min, max, default, resolution)
        self.face_effect_params = {
            # Universal
            'intensity': ('Intensity', 0, 100, 50, 1),
            'color_shift': ('Color/Hue Shift', 0, 180, 90, 1),
            # Pixelate
            'pixelate_size': ('Block Size', 2, 50, 10, 1),
            # Blur
            'blur_amount': ('Blur Amount', 1, 101, 31, 2),
            # Posterize
            'posterize_levels': ('Color Levels', 2, 16, 6, 1),
            # Wave/Distortion
            'wave_amplitude': ('Wave Amount', 1, 50, 20, 1),
            'wave_frequency': ('Wave Speed', 1, 20, 5, 1),
            # Chromatic
            'chromatic_offset': ('Offset', 1, 30, 5, 1),
            # VHS/Noise
            'noise_intensity': ('Noise', 1, 100, 25, 1),
            # Film Grain
            'grain_intensity': ('Grain', 1, 100, 30, 1),
            # TV Static
            'static_blend': ('Blend', 0, 100, 30, 1),
            # Scanlines
            'scanline_darkness': ('Darkness', 0, 100, 40, 1),
            # Ghost Trail
            'trail_decay': ('Decay', 50, 99, 85, 1),
            # Kaleidoscope
            'segments': ('Segments', 2, 16, 6, 1),
            # Emboss
            'emboss_strength': ('Strength', 0, 100, 50, 1),
            # Radial Blur
            'radial_strength': ('Strength', 1, 50, 10, 1),
            # Tunnel Vision
            'vignette_intensity': ('Intensity', 0, 100, 70, 1),
            # Double Vision
            'offset': ('Offset', 1, 50, 15, 1),
            # Halftone
            'dot_size': ('Dot Size', 1, 20, 4, 1),
            # Neon Glow
            'glow_size': ('Glow Size', 1, 20, 5, 1),
            # Glitch
            'glitch_intensity': ('Glitch Amount', 1, 50, 20, 1),
            'glitch_blocks': ('Block Count', 1, 20, 8, 1),
            # Heat Distort
            'heat_intensity': ('Heat', 1, 30, 8, 1),
            # Drunk
            'wobble_intensity': ('Wobble', 1, 50, 15, 1),
            # Prism
            'prism_offset': ('Prism Offset', 1, 30, 8, 1),
            # Spiral
            'spiral_strength': ('Spiral', 0, 100, 50, 1),
            # Blocky Noise
            'block_chance': ('Chance', 1, 50, 10, 1),
            # RGB Split
            'split_strength': ('Split', 1, 30, 10, 1),
        }

        # Map effects to their relevant parameters
        self.effect_param_map = {
            'none': [],
            'pixelate': ['pixelate_size'],
            'blur': ['blur_amount'],
            'glitch': ['glitch_intensity'],
            'thermal': [],
            'negative': [],
            'cartoon': ['intensity'],
            'sketch': ['intensity'],
            'oil_paint': [],
            'emboss': ['emboss_strength'],
            'posterize': ['posterize_levels'],
            'color_shift': ['color_shift'],
            'rgb_shift': ['chromatic_offset'],
            'chromatic': ['chromatic_offset'],
            'duotone': [],
            'cross_process': [],
            'wave_distort': ['wave_amplitude', 'wave_frequency'],
            'heat_distort': ['heat_intensity'],
            'drunk': ['wobble_intensity'],
            'spiral_warp': ['spiral_strength'],
            'kaleidoscope': ['segments'],
            'mirror': [],
            'neon_glow': ['glow_size'],
            'edge_glow': [],
            'edge_highlight': ['intensity'],
            'halftone': ['dot_size'],
            'vhs': ['noise_intensity'],
            'film_grain': ['grain_intensity'],
            'tv_static': ['static_blend'],
            'blocky_noise': ['block_chance'],
            'scanlines': ['scanline_darkness'],
            'retro_crt': [],
            'ghost_trail': ['trail_decay'],
            'motion_blur': ['blur_amount'],
            'radial_blur': ['radial_strength'],
            'zoom_blur': ['intensity'],
            'tunnel_vision': ['vignette_intensity'],
            'double_vision': ['offset'],
            'prism': ['prism_offset'],
            'rgb_split_radial': ['split_strength'],
            'glitch_blocks': ['glitch_blocks', 'glitch_intensity'],
            'glitch_shift': ['glitch_intensity'],
        }

        # Storage for dynamically created slider widgets
        self.face_param_widgets = {}
        self.vars['face_params'] = {}

        # Initialize with no sliders (will be populated when effect selected)
        self._update_face_param_sliders()

        # === Effect Stack Section ===
        stack_frame = ttk.LabelFrame(live_effect_frame, text="Effect Stack", padding=3)
        stack_frame.pack(fill=tk.X, pady=5)

        # Apply to Stack button - adds current effect to stack
        stack_btn_row = ttk.Frame(stack_frame)
        stack_btn_row.pack(fill=tk.X, pady=2)
        ttk.Button(stack_btn_row, text="➕ Add to Stack",
                   command=self._add_effect_to_stack, width=14).pack(side='left', padx=2)
        ttk.Button(stack_btn_row, text="🗑 Clear Stack",
                   command=self._clear_effect_stack, width=12).pack(side='left', padx=2)

        # Stack display label - shows current stack
        self.effect_stack_label = ttk.Label(stack_frame, text="Stack: (empty)",
                                            font=('Arial', 8), wraplength=350)
        self.effect_stack_label.pack(anchor='w', pady=2)

        # Bake/Lock buttons
        bake_row = ttk.Frame(face_section)
        bake_row.pack(fill=tk.X, pady=5)
        ttk.Button(bake_row, text="🔒 Bake Stack",
                   command=self._bake_face_effect, width=12).pack(side='left', padx=2)
        ttk.Button(bake_row, text="🔓 Unbake",
                   command=self._unbake_face_effect, width=10).pack(side='left', padx=2)
        ttk.Button(bake_row, text="🔄 Reset Face",
                   command=self._reset_face_effect, width=12).pack(side='left', padx=2)

        # Baked faces info
        self.baked_info_label = ttk.Label(face_section, text="Baked: 0 faces",
                                          font=('Arial', 8, 'italic'))
        self.baked_info_label.pack(anchor='w')

        ttk.Button(face_section, text="🗑 Clear All Baked Effects",
                   command=self._clear_all_baked).pack(anchor='w', pady=2)

        # Preview toggle - shows baked effects in live preview
        preview_frame = ttk.Frame(face_section)
        preview_frame.pack(fill=tk.X, pady=5)

        self.vars['show_face_effects'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            preview_frame, text="👁 Show face effects in preview",
            variable=self.vars['show_face_effects'],
            command=self._toggle_face_effects
        ).pack(side='left')

        ttk.Label(preview_frame, text="(includes baked & live effects)",
                  font=('Arial', 7, 'italic'), foreground='gray').pack(side='left', padx=5)

        # =============================================
        # BACKGROUND INFO SECTION
        # =============================================
        bg_section = ttk.LabelFrame(parent, text="🌐 BACKGROUND/GLOBAL EFFECTS", padding=5)
        bg_section.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(bg_section, text="Global effects are controlled in the main Effects panel.",
                  font=('Arial', 8)).pack(anchor='w')
        ttk.Label(bg_section, text="They apply to the entire frame EXCEPT selected faces.",
                  font=('Arial', 8, 'italic')).pack(anchor='w')

        # Option to exclude faces from background effects
        self.vars['exclude_faces_from_bg'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            bg_section, text="🚫 Exclude selected faces from background effects",
            variable=self.vars['exclude_faces_from_bg'],
            command=self._toggle_exclude_faces
        ).pack(anchor='w', pady=5)

        # Legacy compatibility vars (hidden)
        self.vars['per_region_mode'] = tk.BooleanVar(value=True)
        self.vars['region_effect_type'] = tk.StringVar(value='none')
        self.vars['region_intensity'] = tk.IntVar(value=50)
        self.vars['region_color_shift'] = tk.IntVar(value=90)
        self.vars['face_effect_mode'] = tk.StringVar(value='none')

        # === Texture Overlay Section ===
        tex_frame = ttk.LabelFrame(parent, text="🖼 Texture Overlay (Image/Video)", padding=5)
        tex_frame.pack(fill=tk.X, padx=5, pady=5)

        # Load buttons row
        tex_row = ttk.Frame(tex_frame)
        tex_row.pack(fill=tk.X, pady=2)
        ttk.Button(tex_row, text="📹 Load Video", command=self._load_video_texture, width=12).pack(side='left', padx=2)
        ttk.Button(tex_row, text="🖼 Load Image", command=self._load_image_texture, width=12).pack(side='left', padx=2)

        # Status labels
        self.video_path_label = ttk.Label(tex_frame, text="No texture loaded", font=('Arial', 8))
        self.video_path_label.pack(anchor='w')
        self.image_path_label = ttk.Label(tex_frame, text="", font=('Arial', 8))
        self.image_path_label.pack(anchor='w')

        # Blend mode selector
        blend_row = ttk.Frame(tex_frame)
        blend_row.pack(fill=tk.X, pady=2)
        ttk.Label(blend_row, text="Blend:").pack(side='left')
        self.vars['region_blend_mode'] = tk.StringVar(value='replace')
        blend_combo = ttk.Combobox(blend_row, textvariable=self.vars['region_blend_mode'],
                                    values=['replace', 'multiply', 'screen', 'overlay'],
                                    state='readonly', width=10)
        blend_combo.pack(side='left', padx=5)

        # Opacity slider
        opacity_row = ttk.Frame(tex_frame)
        opacity_row.pack(fill=tk.X, pady=2)
        ttk.Label(opacity_row, text="Opacity:").pack(side='left')
        self.vars['region_opacity'] = tk.DoubleVar(value=1.0)
        opacity_scale = ttk.Scale(opacity_row, from_=0.0, to=1.0, variable=self.vars['region_opacity'],
                                   orient='horizontal', length=120)
        opacity_scale.pack(side='left', padx=5)
        self.texture_opacity_label = ttk.Label(opacity_row, text="100%", font=('Arial', 8))
        self.texture_opacity_label.pack(side='left')
        self.vars['region_opacity'].trace_add('write', self._update_texture_opacity_label)

        # Apply texture button - applies to selected faces
        apply_tex_row = ttk.Frame(tex_frame)
        apply_tex_row.pack(fill=tk.X, pady=5)
        ttk.Button(apply_tex_row, text="✅ Apply Texture to Selected",
                   command=self._apply_texture_to_selected, width=22).pack(side='left', padx=2)
        ttk.Button(apply_tex_row, text="🗑 Clear Texture",
                   command=self._clear_texture_from_selected, width=12).pack(side='left', padx=2)

        # Note about workflow
        ttk.Label(tex_frame, text="💡 Load texture, then click 'Apply' to add to selected faces",
                  font=('Arial', 7, 'italic'), foreground='gray').pack(anchor='w')

        # Hidden region selector (for compatibility)
        self.vars['selected_region'] = tk.StringVar(value='face_0')
        self.vars['region_color_shift'] = tk.IntVar(value=90)

        # Track loaded texture type
        self._loaded_texture_type = None  # 'video' or 'image'
        self._loaded_texture_path = None

    def _create_viz_tab(self):
        """Create the Visualization tab."""
        parent = self.tab_viz

        # Data viz enable
        viz_frame = ttk.LabelFrame(parent, text="Data Visualization", padding=5)
        viz_frame.pack(fill=tk.X, padx=5, pady=5)

        self.vars['dataviz_enabled'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            viz_frame, text="📊 Enable Data Visualization",
            variable=self.vars['dataviz_enabled'],
            command=self._toggle_data_viz
        ).pack(anchor='w')

        # Selective mode - only apply to selected faces
        self.vars['dataviz_selective'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            viz_frame, text="🎯 Selected Faces Only",
            variable=self.vars['dataviz_selective'],
            command=self._toggle_dataviz_selective
        ).pack(anchor='w')

        ttk.Label(viz_frame, text="(When checked, visualization only shows on selected faces)",
                  font=('Arial', 7, 'italic'), foreground='gray').pack(anchor='w')

        # Style selector
        style_row = ttk.Frame(viz_frame)
        style_row.pack(fill=tk.X, pady=3)
        ttk.Label(style_row, text="Style:").pack(side='left')
        self.dataviz_style = ttk.Combobox(style_row, state='readonly', width=12,
                                           values=['default', 'cyberpunk', 'matrix', 'hud', 'thermal', 'glitch', 'minimal', 'neon'])
        self.dataviz_style.set('default')
        self.dataviz_style.pack(side='left', padx=5)
        self.dataviz_style.bind('<<ComboboxSelected>>', self._update_dataviz_style)

        # Options checkboxes
        opts_frame = ttk.Frame(viz_frame)
        opts_frame.pack(fill=tk.X, pady=3)

        self.vars['show_labels'] = tk.BooleanVar(value=True)
        self.vars['show_id'] = tk.BooleanVar(value=True)
        self.vars['show_coordinates'] = tk.BooleanVar(value=False)
        self.vars['show_center'] = tk.BooleanVar(value=False)

        ttk.Checkbutton(opts_frame, text="Labels", variable=self.vars['show_labels'],
                        command=self._update_dataviz_settings).pack(side='left')
        ttk.Checkbutton(opts_frame, text="IDs", variable=self.vars['show_id'],
                        command=self._update_dataviz_settings).pack(side='left')
        ttk.Checkbutton(opts_frame, text="Coords", variable=self.vars['show_coordinates'],
                        command=self._update_dataviz_settings).pack(side='left')

        # === Box Colors Section ===
        box_color_frame = ttk.LabelFrame(parent, text="Box Colors", padding=5)
        box_color_frame.pack(fill=tk.X, padx=5, pady=5)

        # Color presets for boxes
        box_color_row = ttk.Frame(box_color_frame)
        box_color_row.pack(fill=tk.X, pady=2)
        ttk.Label(box_color_row, text="Box Color:").pack(side='left')
        self.vars['box_color'] = tk.StringVar(value='cyan')
        self.box_color_combo = ttk.Combobox(box_color_row, textvariable=self.vars['box_color'],
                                             values=['cyan', 'magenta', 'yellow', 'green', 'red', 'blue', 'white', 'orange', 'purple', 'pink'],
                                             state='readonly', width=10)
        self.box_color_combo.pack(side='left', padx=5)

        ttk.Button(box_color_row, text="Apply to Selected", command=self._apply_box_color_to_selected,
                   width=15).pack(side='left', padx=3)

        # Connection lines
        conn_frame = ttk.LabelFrame(parent, text="Connection Lines", padding=5)
        conn_frame.pack(fill=tk.X, padx=5, pady=5)

        self.vars['show_connections'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(conn_frame, text="Show Auto Connections", variable=self.vars['show_connections'],
                        command=self._update_dataviz_settings).pack(anchor='w')

        conn_style_row = ttk.Frame(conn_frame)
        conn_style_row.pack(fill=tk.X, pady=2)
        ttk.Label(conn_style_row, text="Style:").pack(side='left')
        self.connection_style = ttk.Combobox(conn_style_row, state='readonly', width=12,
                                              values=['line', 'dashed', 'dotted', 'curved', 'lightning'])
        self.connection_style.set('line')
        self.connection_style.pack(side='left', padx=5)
        self.connection_style.bind('<<ComboboxSelected>>', self._update_dataviz_settings)

        # Connection color and thickness row
        conn_color_row = ttk.Frame(conn_frame)
        conn_color_row.pack(fill=tk.X, pady=2)
        ttk.Label(conn_color_row, text="Line Color:").pack(side='left')
        self.vars['connection_color'] = tk.StringVar(value='cyan')
        self.conn_color_combo = ttk.Combobox(conn_color_row, textvariable=self.vars['connection_color'],
                                              values=['cyan', 'magenta', 'yellow', 'green', 'red', 'blue', 'white', 'orange', 'purple', 'pink'],
                                              state='readonly', width=10)
        self.conn_color_combo.pack(side='left', padx=5)

        # Connection thickness
        ttk.Label(conn_color_row, text="Thick:").pack(side='left', padx=(10, 0))
        self.vars['connection_thickness'] = tk.IntVar(value=2)
        ttk.Scale(conn_color_row, from_=1, to=10, variable=self.vars['connection_thickness'],
                  orient='horizontal', length=60).pack(side='left', padx=2)
        self.conn_thick_label = ttk.Label(conn_color_row, text="2", width=2)
        self.conn_thick_label.pack(side='left')
        self.vars['connection_thickness'].trace_add('write', self._on_thickness_change)

        # Apply button for connection settings
        conn_apply_row = ttk.Frame(conn_frame)
        conn_apply_row.pack(fill=tk.X, pady=3)
        ttk.Button(conn_apply_row, text="✅ Apply Line Settings", command=self._apply_connection_settings,
                   width=18).pack(side='left', padx=2)
        self.conn_status_label = ttk.Label(conn_apply_row, text="", font=('Arial', 8, 'italic'))
        self.conn_status_label.pack(side='left', padx=5)

        self.vars['connect_same'] = tk.BooleanVar(value=True)
        self.vars['connect_different'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(conn_frame, text="Connect same type", variable=self.vars['connect_same'],
                        command=self._update_dataviz_settings).pack(anchor='w')
        ttk.Checkbutton(conn_frame, text="Connect different types", variable=self.vars['connect_different'],
                        command=self._update_dataviz_settings).pack(anchor='w')

        # Connection groups section
        conn_groups_frame = ttk.LabelFrame(parent, text="Connection Groups (Different Colors)", padding=5)
        conn_groups_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(conn_groups_frame, text="Create colored connection groups:", font=('Arial', 8)).pack(anchor='w')

        # Group color selector
        group_row = ttk.Frame(conn_groups_frame)
        group_row.pack(fill=tk.X, pady=2)
        ttk.Label(group_row, text="Group Color:").pack(side='left')
        self.vars['group_color'] = tk.StringVar(value='yellow')
        self.group_color_combo = ttk.Combobox(group_row, textvariable=self.vars['group_color'],
                                               values=['yellow', 'cyan', 'magenta', 'green', 'red', 'blue', 'white', 'orange', 'purple', 'pink'],
                                               state='readonly', width=10)
        self.group_color_combo.pack(side='left', padx=5)

        # Group buttons
        group_btn_row = ttk.Frame(conn_groups_frame)
        group_btn_row.pack(fill=tk.X, pady=3)
        ttk.Button(group_btn_row, text="🔗 Connect Selected", command=self._connect_selected_with_color,
                   width=16).pack(side='left', padx=2)
        ttk.Button(group_btn_row, text="🗑 Clear Connections", command=self._clear_all_connections,
                   width=14).pack(side='left', padx=2)

        # Connection groups list
        self.connection_groups_list = tk.Listbox(conn_groups_frame, height=3, font=('Consolas', 8))
        self.connection_groups_list.pack(fill=tk.X, pady=3)
        ttk.Button(conn_groups_frame, text="Delete Selected Group", command=self._delete_connection_group).pack(anchor='w')

        # Effects
        fx_frame = ttk.LabelFrame(parent, text="Visual Effects", padding=5)
        fx_frame.pack(fill=tk.X, padx=5, pady=5)

        self.vars['animate_boxes'] = tk.BooleanVar(value=False)
        self.vars['scan_line'] = tk.BooleanVar(value=False)
        self.vars['info_panel'] = tk.BooleanVar(value=False)

        ttk.Checkbutton(fx_frame, text="✨ Animate Boxes (Pulse)", variable=self.vars['animate_boxes'],
                        command=self._update_dataviz_settings).pack(anchor='w')
        ttk.Checkbutton(fx_frame, text="📺 Scan Line Effect", variable=self.vars['scan_line'],
                        command=self._update_dataviz_settings).pack(anchor='w')
        ttk.Checkbutton(fx_frame, text="📋 Info Panel", variable=self.vars['info_panel'],
                        command=self._update_dataviz_settings).pack(anchor='w')

        # Info panel position
        self.info_panel_pos = ttk.Combobox(fx_frame, state='readonly', width=12,
                                            values=['top_left', 'top_right', 'bottom_left', 'bottom_right'])
        self.info_panel_pos.set('top_left')
        self.info_panel_pos.pack(anchor='w', pady=2)
        self.info_panel_pos.bind('<<ComboboxSelected>>', self._update_dataviz_settings)

        # Per-detection style
        det_style_frame = ttk.LabelFrame(parent, text="Style Selected Detections", padding=5)
        det_style_frame.pack(fill=tk.X, padx=5, pady=5)

        style_row = ttk.Frame(det_style_frame)
        style_row.pack(fill=tk.X)
        ttk.Label(style_row, text="Style:").pack(side='left')
        self.det_viz_style_combo = ttk.Combobox(style_row, state='readonly', width=12,
                                                 values=['(global)', 'cyberpunk', 'matrix', 'hud', 'thermal', 'glitch', 'neon'])
        self.det_viz_style_combo.set('(global)')
        self.det_viz_style_combo.pack(side='left', padx=3)

        label_row = ttk.Frame(det_style_frame)
        label_row.pack(fill=tk.X, pady=2)
        ttk.Label(label_row, text="Label:").pack(side='left')
        self.det_custom_label_entry = ttk.Entry(label_row, width=15)
        self.det_custom_label_entry.pack(side='left', padx=3)

        ttk.Button(det_style_frame, text="Apply to Selected", command=self._apply_viz_style_to_selected).pack(anchor='w', pady=3)

    def _create_tools_tab(self):
        """Create the Tools tab (custom masks, GIF, presets)."""
        parent = self.tab_tools

        # Custom masks
        mask_frame = ttk.LabelFrame(parent, text="Custom Mask Regions", padding=5)
        mask_frame.pack(fill=tk.X, padx=5, pady=5)

        import object_detection
        self.custom_mask_manager = object_detection.get_custom_mask_manager()

        self.vars['custom_masks_enabled'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(mask_frame, text="Enable Custom Masks", variable=self.vars['custom_masks_enabled'],
                        command=self._toggle_custom_masks).pack(anchor='w')

        mask_btn_row = ttk.Frame(mask_frame)
        mask_btn_row.pack(fill=tk.X, pady=3)
        ttk.Button(mask_btn_row, text="+ Rectangle", command=lambda: self._add_custom_region('rectangle'), width=10).pack(side='left', padx=1)
        ttk.Button(mask_btn_row, text="+ Ellipse", command=lambda: self._add_custom_region('ellipse'), width=10).pack(side='left', padx=1)
        ttk.Button(mask_btn_row, text="Delete", command=self._delete_selected_region, width=8).pack(side='left', padx=1)

        # Region selector - moved up so user selects region first
        ttk.Label(mask_frame, text="Select Region:", font=('Arial', 8)).pack(anchor='w')
        self.custom_region_selector = ttk.Combobox(mask_frame, state='readonly', width=18)
        self.custom_region_selector.pack(anchor='w', pady=2)
        self.custom_region_selector.bind('<<ComboboxSelected>>', self._on_custom_region_selected)

        # Custom mask effect - use full list from CustomMaskManager
        mask_effect_row = ttk.Frame(mask_frame)
        mask_effect_row.pack(fill=tk.X, pady=2)
        ttk.Label(mask_effect_row, text="Effect:").pack(side='left')
        # Get effects from CustomMaskManager, includes 'restore_original' to cut through all effects
        custom_mask_effects = object_detection.CustomMaskManager.AVAILABLE_EFFECTS
        self.custom_effect_type = ttk.Combobox(mask_effect_row, state='readonly', width=15,
                                                values=custom_mask_effects)
        self.custom_effect_type.set('none')
        self.custom_effect_type.pack(side='left', padx=3)

        # Parameter controls
        self.vars['custom_intensity'] = tk.IntVar(value=50)
        self.vars['custom_feather'] = tk.IntVar(value=10)
        self.vars['custom_invert'] = tk.BooleanVar(value=False)
        self.vars['custom_show_border'] = tk.BooleanVar(value=True)

        # Intensity slider
        intensity_row = ttk.Frame(mask_frame)
        intensity_row.pack(fill=tk.X, pady=2)
        ttk.Label(intensity_row, text="Intensity:").pack(side='left')
        ttk.Scale(intensity_row, from_=0, to=100, variable=self.vars['custom_intensity'],
                  orient='horizontal', length=100).pack(side='left', padx=3)
        self.custom_intensity_label = ttk.Label(intensity_row, text="50", width=3)
        self.custom_intensity_label.pack(side='left')
        self.vars['custom_intensity'].trace_add('write', self._update_custom_intensity_label)

        # Feather slider
        feather_row = ttk.Frame(mask_frame)
        feather_row.pack(fill=tk.X, pady=2)
        ttk.Label(feather_row, text="Feather:").pack(side='left')
        ttk.Scale(feather_row, from_=0, to=50, variable=self.vars['custom_feather'],
                  orient='horizontal', length=100).pack(side='left', padx=3)
        self.custom_feather_label = ttk.Label(feather_row, text="10", width=3)
        self.custom_feather_label.pack(side='left')
        self.vars['custom_feather'].trace_add('write', self._update_custom_feather_label)

        # Checkboxes row
        checkbox_row = ttk.Frame(mask_frame)
        checkbox_row.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(checkbox_row, text="Invert", variable=self.vars['custom_invert']).pack(side='left', padx=3)
        ttk.Checkbutton(checkbox_row, text="Show Border", variable=self.vars['custom_show_border']).pack(side='left', padx=3)

        # Apply button row
        apply_row = ttk.Frame(mask_frame)
        apply_row.pack(fill=tk.X, pady=5)
        ttk.Button(apply_row, text="✅ Apply Effect", command=self._apply_custom_mask_effect,
                   width=14).pack(side='left', padx=2)
        ttk.Button(apply_row, text="🔄 Reset", command=self._reset_custom_mask_effect,
                   width=8).pack(side='left', padx=2)

        # Status label
        self.custom_mask_status = ttk.Label(mask_frame, text="Create a mask and select an effect", font=('Arial', 8, 'italic'))
        self.custom_mask_status.pack(anchor='w')

        # GIF Export
        gif_frame = ttk.LabelFrame(parent, text="GIF Export", padding=5)
        gif_frame.pack(fill=tk.X, padx=5, pady=5)

        # Frames slider
        frames_row = ttk.Frame(gif_frame)
        frames_row.pack(fill=tk.X)
        ttk.Label(frames_row, text="Frames:").pack(side='left')
        self.vars['gif_num_frames'] = tk.IntVar(value=60)
        ttk.Scale(frames_row, from_=10, to=300, variable=self.vars['gif_num_frames'],
                  orient='horizontal', command=self._update_render_duration_label).pack(side='left', fill=tk.X, expand=True)
        self.gif_frame_count_label = ttk.Label(frames_row, text="60", width=4)
        self.gif_frame_count_label.pack(side='right')

        # FPS slider
        fps_row = ttk.Frame(gif_frame)
        fps_row.pack(fill=tk.X)
        ttk.Label(fps_row, text="FPS:").pack(side='left')
        self.vars['gif_fps'] = tk.IntVar(value=15)
        ttk.Scale(fps_row, from_=5, to=30, variable=self.vars['gif_fps'],
                  orient='horizontal', command=self._update_render_duration_label).pack(side='left', fill=tk.X, expand=True)
        self.gif_fps_label = ttk.Label(fps_row, text="15", width=4)
        self.gif_fps_label.pack(side='right')

        self.gif_duration_label = ttk.Label(gif_frame, text="Duration: ~4.0 sec", font=('Arial', 8, 'bold'))
        self.gif_duration_label.pack(anchor='w')

        # Buttons
        gif_btn_row = ttk.Frame(gif_frame)
        gif_btn_row.pack(fill=tk.X, pady=3)
        self.gif_render_btn = ttk.Button(gif_btn_row, text="🎬 Render GIF", command=self._start_gif_render, width=12)
        self.gif_render_btn.pack(side='left', padx=2)
        self.gif_cancel_btn = ttk.Button(gif_btn_row, text="Cancel", command=self._cancel_gif_render, width=8, state='disabled')
        self.gif_cancel_btn.pack(side='left', padx=2)
        ttk.Button(gif_btn_row, text="💾 Save", command=self._export_gif, width=8).pack(side='left', padx=2)

        # Progress
        self.gif_progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(gif_frame, variable=self.gif_progress_var, maximum=100).pack(fill=tk.X, pady=2)
        self.gif_status_label = ttk.Label(gif_frame, text="Ready", font=('Arial', 8))
        self.gif_status_label.pack(anchor='w')
        self.gif_rendered_label = ttk.Label(gif_frame, text="Rendered: 0 frames", font=('Arial', 8))
        self.gif_rendered_label.pack(anchor='w')

        # Presets
        preset_frame = ttk.LabelFrame(parent, text="Quick Presets", padding=5)
        preset_frame.pack(fill=tk.X, padx=5, pady=5)

        presets = [
            ("Privacy Blur", self._apply_privacy_preset),
            ("Glitch Portrait", self._apply_glitch_preset),
            ("Neon Face", self._apply_neon_preset),
            ("Reset All", self._reset_preset),
        ]

        preset_row = ttk.Frame(preset_frame)
        preset_row.pack(fill=tk.X)
        for name, callback in presets:
            ttk.Button(preset_row, text=name, command=callback, width=12).pack(side='left', padx=1, pady=1)

        # Store references for GIF rendering
        self.apply_effects_func = None
        self.current_source_frame = None
        self.video_path = None

    def _update_frame_skip(self, value=None):
        """Update frame skip settings."""
        skip = self.vars['detection_frame_skip'].get()
        self.detection_params['detection_frame_skip'] = skip
        self.frame_skip_label.config(text=f"Skip: {skip} (detect every {skip} frame{'s' if skip > 1 else ''})")

        # Update the frame skipper if available
        try:
            from performance import frame_skipper
            frame_skipper.set_skip_interval('face_detection', skip)
            frame_skipper.set_skip_interval('eye_detection', skip)
            frame_skipper.set_skip_interval('body_detection', max(skip, 5))
            frame_skipper.set_skip_interval('upper_body_detection', max(skip, 5))
        except ImportError:
            pass

    def _on_region_selected(self, event=None):
        """Handle region selection change."""
        region_id = self.vars['selected_region'].get()
        self.current_region_id = region_id

        if self.region_effect_manager:
            effect = self.region_effect_manager.get_region_effect(region_id)
            self.vars['region_effect_type'].set(effect.get('effect_type', 'none'))
            self.vars['region_intensity'].set(effect.get('intensity', 50))
            self.vars['region_color_shift'].set(effect.get('color_shift', 90))
            self.vars['region_blend_mode'].set(effect.get('blend_mode', 'replace'))
            self.vars['region_opacity'].set(effect.get('blend_opacity', 1.0))

            # Update path labels
            video_path = effect.get('video_path')
            image_path = effect.get('image_path')
            if video_path:
                self.video_path_label.config(text=os.path.basename(video_path))
            else:
                self.video_path_label.config(text="No video loaded")
            if image_path:
                self.image_path_label.config(text=os.path.basename(image_path))
            else:
                self.image_path_label.config(text="No image loaded")

    def _on_region_effect_changed(self, event=None):
        """Handle region effect type change."""
        self._apply_region_settings()

    def _on_region_intensity_changed(self, value):
        """Handle region intensity change."""
        pass  # Apply on button click

    def _on_region_color_changed(self, value):
        """Handle region color shift change."""
        pass  # Apply on button click

    def _on_blend_mode_changed(self, event=None):
        """Handle blend mode change."""
        self._apply_region_settings()

    def _on_opacity_changed(self, value):
        """Handle opacity change."""
        pass  # Apply on button click

    def _update_texture_opacity_label(self, *args):
        """Update the opacity percentage label."""
        if hasattr(self, 'texture_opacity_label'):
            opacity = self.vars['region_opacity'].get()
            self.texture_opacity_label.config(text=f"{int(opacity * 100)}%")

    def _load_video_texture(self):
        """Load a video file for texture (stores it for later application)."""
        file_path = filedialog.askopenfilename(
            title="Select Video for Texture Overlay",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self._loaded_texture_type = 'video'
            self._loaded_texture_path = file_path
            self.video_path_label.config(text=f"Video: {os.path.basename(file_path)}")
            self.image_path_label.config(text="")
            print(f"Loaded video texture: {file_path}")
            print("Click 'Apply Texture to Selected' to apply to selected faces")

    def _load_image_texture(self):
        """Load an image file for texture (stores it for later application)."""
        file_path = filedialog.askopenfilename(
            title="Select Image for Texture Overlay",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self._loaded_texture_type = 'image'
            self._loaded_texture_path = file_path
            self.image_path_label.config(text=f"Image: {os.path.basename(file_path)}")
            self.video_path_label.config(text="")
            print(f"Loaded image texture: {file_path}")
            print("Click 'Apply Texture to Selected' to apply to selected faces")

    def _apply_texture_to_selected(self):
        """Apply the loaded texture to all selected faces."""
        if not self.region_effect_manager:
            print("No region effect manager available")
            return

        if not hasattr(self, '_loaded_texture_type') or self._loaded_texture_type is None:
            print("No texture loaded. Load a video or image first.")
            return

        if not hasattr(self, '_loaded_texture_path') or self._loaded_texture_path is None:
            print("No texture path set. Load a video or image first.")
            return

        # Get selected detection IDs from data_viz
        data_viz = self.data_viz if hasattr(self, 'data_viz') and self.data_viz else None
        if not data_viz:
            print("No data visualization manager available")
            return

        selected_ids = data_viz.get_selected_list()
        if not selected_ids:
            print("No faces selected. Select faces in the Detection tab first.")
            return

        # Get blend mode and opacity
        blend_mode = self.vars['region_blend_mode'].get()
        opacity = self.vars['region_opacity'].get()

        # Apply texture to each selected face
        applied_count = 0
        for det_id in selected_ids:
            region_id = f"face_{det_id}"

            # Load the texture for this region
            if self._loaded_texture_type == 'video':
                if self.region_effect_manager.load_video_texture(region_id, self._loaded_texture_path):
                    # Add to effect stack (texture_region_id used by effect to find the texture)
                    self.region_effect_manager.add_effect_to_stack(
                        region_id, 'video_texture',
                        texture_region_id=region_id,
                        blend_mode=blend_mode,
                        opacity=opacity
                    )
                    applied_count += 1
            elif self._loaded_texture_type == 'image':
                if self.region_effect_manager.load_image_texture(region_id, self._loaded_texture_path):
                    # Add to effect stack (texture_region_id used by effect to find the texture)
                    self.region_effect_manager.add_effect_to_stack(
                        region_id, 'image_texture',
                        texture_region_id=region_id,
                        blend_mode=blend_mode,
                        opacity=opacity
                    )
                    applied_count += 1

        print(f"Applied {self._loaded_texture_type} texture to {applied_count} face(s)")
        self._update_face_effect_status()

    def _clear_texture_from_selected(self):
        """Clear texture effects from selected faces."""
        if not self.region_effect_manager:
            return

        # Get selected detection IDs
        data_viz = self.data_viz if hasattr(self, 'data_viz') and self.data_viz else None
        if not data_viz:
            return

        selected_ids = data_viz.get_selected_list()
        if not selected_ids:
            print("No faces selected")
            return

        cleared_count = 0
        for det_id in selected_ids:
            region_id = f"face_{det_id}"
            # Clear the effect stack for this region
            if region_id in self.region_effect_manager.effect_stacks:
                # Remove only texture effects
                stack = self.region_effect_manager.effect_stacks[region_id]
                new_stack = [e for e in stack if e.get('effect_type') not in ('video_texture', 'image_texture')]
                self.region_effect_manager.effect_stacks[region_id] = new_stack
                cleared_count += 1
            # Also reset the region effect
            self.region_effect_manager.reset_region(region_id)

        print(f"Cleared texture from {cleared_count} face(s)")
        self._update_face_effect_status()

    def _refresh_detection_list(self):
        """Refresh the list of active detections."""
        self.detection_listbox.delete(0, tk.END)

        # Get tracker reference
        tracker = self.detection_tracker

        if tracker is None:
            self.detection_listbox.insert(tk.END, "(No tracker available)")
            return

        # Get all tracked detections
        tracked = tracker.get_all_tracked()
        ignored = tracker.get_ignored_list()
        kept = tracker.get_kept_list() if hasattr(tracker, 'get_kept_list') else []

        # Get data viz for selection info
        data_viz = self.data_viz if hasattr(self, 'data_viz') and self.data_viz else None
        selected = data_viz.get_selected_list() if data_viz else []

        for det_id, info in tracked.items():
            det_type = info.get('type', 'unknown')
            rect = info.get('rect', (0, 0, 0, 0))

            status_flags = []
            if det_id in kept:
                status_flags.append("🔒KEPT")
            if det_id in ignored:
                status_flags.append("🚫IGN")
            if det_id in selected:
                status_flags.append("📌SEL")

            status_str = f" [{', '.join(status_flags)}]" if status_flags else ""
            display = f"ID:{det_id} - {det_type} @ ({rect[0]},{rect[1]}){status_str}"
            self.detection_listbox.insert(tk.END, display)

        # Update status label
        num_active = len(tracked)
        num_kept = len(kept) if kept else 0
        num_ignored = len(ignored)
        num_selected = len(selected) if selected else 0
        self.detection_status_label.config(
            text=f"{num_active} total, {num_kept} kept, {num_ignored} ignored, {num_selected} selected"
        )

        # Also update single-shot status if available
        if hasattr(self, 'single_shot_status'):
            self.single_shot_status.config(text=f"{num_active} detections found, {num_kept} kept")

    def _get_selected_detection_ids(self):
        """Get detection IDs from listbox selection."""
        selection = self.detection_listbox.curselection()
        det_ids = []
        for idx in selection:
            item = self.detection_listbox.get(idx)
            # Parse ID from "ID:X - ..." format
            if item.startswith("ID:"):
                try:
                    det_id = int(item.split(" - ")[0].replace("ID:", ""))
                    det_ids.append(det_id)
                except:
                    pass
        return det_ids

    def _ignore_selected_detections(self):
        """Ignore the selected detections in the listbox."""
        det_ids = self._get_selected_detection_ids()
        tracker = self.detection_tracker
        if tracker is None:
            return

        for det_id in det_ids:
            tracker.ignore_detection(det_id)

        self._refresh_detection_list()
        print(f"Ignored {len(det_ids)} detection(s)")

    def _unignore_selected_detections(self):
        """Unignore the selected detections."""
        det_ids = self._get_selected_detection_ids()
        tracker = self.detection_tracker
        if tracker is None:
            return

        for det_id in det_ids:
            tracker.unignore_detection(det_id)

        self._refresh_detection_list()
        print(f"Unignored {len(det_ids)} detection(s)")

    def _clear_ignored(self):
        """Clear all ignored detections."""
        tracker = self.detection_tracker
        if tracker is None:
            return

        tracker.clear_ignored()
        self._refresh_detection_list()
        print("Cleared all ignored detections")

    # === Single-Shot Detection Methods ===

    def _trigger_detection(self):
        """Trigger a single detection pass (for image mode)."""
        # Enable single-shot mode when using the detect button
        # This ensures detection works without continuous mode enabled
        self.detection_params['single_shot_mode'] = True
        self.vars['single_shot_mode'].set(True)

        # Set flag to trigger detection on next frame
        self.detection_params['trigger_single_detection'] = True
        print("Detection triggered - processing...")

        # Update status
        if hasattr(self, 'single_shot_status'):
            self.single_shot_status.config(text="Detecting...")

        # Refresh list after a short delay to allow detection to complete
        if self.root:
            self.root.after(500, self._refresh_detection_list)

    def _keep_selected_detections(self):
        """Keep/lock the selected detections (won't be cleared on re-detect)."""
        det_ids = self._get_selected_detection_ids()
        tracker = self.detection_tracker
        if tracker is None:
            return

        for det_id in det_ids:
            tracker.keep_detection(det_id)

        self._refresh_detection_list()
        print(f"Kept {len(det_ids)} detection(s) - they will persist through re-detection")

    def _unkeep_selected_detections(self):
        """Remove keep/lock from selected detections."""
        det_ids = self._get_selected_detection_ids()
        tracker = self.detection_tracker
        if tracker is None:
            return

        for det_id in det_ids:
            tracker.unkeep_detection(det_id)

        self._refresh_detection_list()
        print(f"Unkept {len(det_ids)} detection(s)")

    def _clear_all_detections(self):
        """Clear all detections (except kept ones)."""
        tracker = self.detection_tracker
        if tracker is None:
            return

        tracker.clear_non_kept()
        self._refresh_detection_list()
        print("Cleared all non-kept detections")

    # === Effects Tab Methods ===

    def _refresh_selected_for_effects(self):
        """Refresh the selected detections list in the effects tab."""
        if not hasattr(self, 'selected_detections_listbox'):
            return

        self.selected_detections_listbox.delete(0, tk.END)

        # Get tracker and data viz
        tracker = self.detection_tracker

        data_viz = self.data_viz if hasattr(self, 'data_viz') and self.data_viz else None
        if not data_viz or not tracker:
            self.selected_detections_listbox.insert(tk.END, "(Select faces in Detection tab)")
            return

        selected = data_viz.get_selected_list()
        tracked = tracker.get_all_tracked()

        if not selected:
            self.selected_detections_listbox.insert(tk.END, "(No detections selected)")
            self.selected_effect_status.config(text="0 selected for editing")
            return

        # Add selected detections and their child parts (eyes within faces, etc.)
        count = 0
        for det_id in selected:
            if det_id in tracked:
                info = tracked[det_id]
                det_type = info.get('type', 'unknown')
                rect = info.get('rect', (0, 0, 0, 0))

                # Get current effect if any
                effect_str = ""
                if self.region_effect_manager:
                    region_key = f"{det_type}_{det_id}"
                    effect = self.region_effect_manager.get_region_effect(region_key)
                    if effect.get('effect_type', 'none') != 'none':
                        effect_str = f" → {effect['effect_type']}"

                display = f"ID:{det_id} - {det_type}{effect_str}"
                self.selected_detections_listbox.insert(tk.END, display)
                count += 1

        self.selected_effect_status.config(text=f"{count} selected for editing")

        # Update face effect status and load current effect for first selected
        self._update_face_effect_status()
        self._load_face_effect_from_selected()

    def _on_effect_detection_selected(self, event=None):
        """Handle selection of a detection in the effects list."""
        if not hasattr(self, 'selected_detections_listbox'):
            return

        selection = self.selected_detections_listbox.curselection()
        if not selection:
            return

        item = self.selected_detections_listbox.get(selection[0])
        if item.startswith("("):
            return

        # Parse ID
        try:
            det_id = int(item.split(" - ")[0].replace("ID:", ""))
        except:
            return

        # Get detection info
        tracker = self.detection_tracker
        if tracker is None:
            return

        tracked = tracker.get_all_tracked()
        if det_id in tracked:
            info = tracked[det_id]
            det_type = info.get('type', 'unknown')

            # Store current editing target
            self._current_editing_id = det_id
            self._current_editing_type = det_type

            # Load current effect settings for legacy UI
            if self.region_effect_manager:
                region_key = f"{det_type}_{det_id}"
                effect = self.region_effect_manager.get_region_effect(region_key)
                self.vars['region_effect_type'].set(effect.get('effect_type', 'none'))
                self.vars['region_intensity'].set(effect.get('intensity', 50))
                self.vars['region_color_shift'].set(effect.get('color_shift', 90))

    def _apply_effect_to_selected(self):
        """Apply the current effect settings to the selected detection."""
        if not hasattr(self, '_current_editing_id') or self._current_editing_id is None:
            print("No detection selected for editing")
            return

        if not self.region_effect_manager:
            return

        det_id = self._current_editing_id
        det_type = self._current_editing_type
        region_key = f"{det_type}_{det_id}"

        effect_type = self.vars['region_effect_type'].get()
        intensity = self.vars['region_intensity'].get()
        color_shift = self.vars['region_color_shift'].get()

        self.region_effect_manager.set_region_effect(
            region_key,
            effect_type,
            intensity=intensity,
            color_shift=color_shift
        )

        # Auto-enable per_region_mode so effects are visible
        self.detection_params['per_region_mode'] = True
        if 'per_region_mode' in self.vars:
            self.vars['per_region_mode'].set(True)

        print(f"Applied {effect_type} to {det_type} ID:{det_id}")
        self._refresh_selected_for_effects()

    def _reset_selected_effect(self):
        """Reset the effect on the currently selected detection."""
        if not hasattr(self, '_current_editing_id') or self._current_editing_id is None:
            return

        if not self.region_effect_manager:
            return

        det_id = self._current_editing_id
        det_type = self._current_editing_type
        region_key = f"{det_type}_{det_id}"

        self.region_effect_manager.reset_region(region_key)
        self.vars['region_effect_type'].set('none')
        self.vars['region_intensity'].set(50)

        print(f"Reset effect on {det_type} ID:{det_id}")
        self._refresh_selected_for_effects()

    def _copy_effect_to_all_selected(self):
        """Copy the current effect to all selected detections."""
        if not self.region_effect_manager:
            return

        data_viz = self.data_viz if hasattr(self, 'data_viz') and self.data_viz else None
        tracker = self.detection_tracker
        if not data_viz or not tracker:
            return

        selected = data_viz.get_selected_list()
        tracked = tracker.get_all_tracked()

        effect_type = self.vars['region_effect_type'].get()
        intensity = self.vars['region_intensity'].get()
        color_shift = self.vars['region_color_shift'].get()

        count = 0
        for det_id in selected:
            if det_id in tracked:
                info = tracked[det_id]
                det_type = info.get('type', 'unknown')
                region_key = f"{det_type}_{det_id}"

                self.region_effect_manager.set_region_effect(
                    region_key,
                    effect_type,
                    intensity=intensity,
                    color_shift=color_shift
                )
                count += 1

        # Auto-enable per_region_mode so effects are visible
        if count > 0:
            self.detection_params['per_region_mode'] = True
            if 'per_region_mode' in self.vars:
                self.vars['per_region_mode'].set(True)

        print(f"Applied {effect_type} to {count} selected detection(s)")
        self._refresh_selected_for_effects()

    def _apply_quick_effect_to_selected(self, effect_type):
        """Apply a quick effect to all selected detections."""
        if not self.region_effect_manager:
            return

        data_viz = self.data_viz if hasattr(self, 'data_viz') and self.data_viz else None
        tracker = self.detection_tracker
        if not data_viz or not tracker:
            return

        selected = data_viz.get_selected_list()
        tracked = tracker.get_all_tracked()

        count = 0
        for det_id in selected:
            if det_id in tracked:
                info = tracked[det_id]
                det_type = info.get('type', 'unknown')
                region_key = f"{det_type}_{det_id}"

                self.region_effect_manager.set_region_effect(
                    region_key,
                    effect_type,
                    intensity=50
                )
                count += 1

        # Auto-enable per_region_mode so effects are visible
        if count > 0:
            self.detection_params['per_region_mode'] = True
            if 'per_region_mode' in self.vars:
                self.vars['per_region_mode'].set(True)

        print(f"Applied {effect_type} to {count} selected detection(s)")
        self._refresh_selected_for_effects()

    def _apply_effect_stack_to_selected(self):
        """Bake/capture the current global effects onto selected faces permanently."""
        data_viz = self.data_viz if hasattr(self, 'data_viz') and self.data_viz else None
        if not data_viz:
            print("Data visualization not available")
            return

        if not self.region_effect_manager:
            print("Region effect manager not available")
            return

        selected = data_viz.get_selected_list()
        if not selected:
            print("No faces selected - select some faces first")
            return

        # Use stored tracker reference instead of importing main
        if not self.detection_tracker:
            print("Detection tracker not available")
            return

        # Signal main loop to bake effects on next frame
        # We store the IDs and main.py will capture the effected regions
        self.detection_params['bake_effect_stack'] = True
        self.detection_params['bake_selected_ids'] = list(selected)

        # Update UI label
        if hasattr(self, 'stack_mode_label'):
            self.stack_mode_label.config(text=f"Baking effects for {len(selected)} face(s)...")

        print(f"Baking current effect stack onto {len(selected)} selected face(s)")
        print("Effects will be permanently captured - you can now change global effects for other faces")

        # Auto-enable per_region_mode so baked effects are visible
        self.detection_params['per_region_mode'] = True
        if 'per_region_mode' in self.vars:
            self.vars['per_region_mode'].set(True)

        # Schedule UI update after baking completes
        if self.root:
            self.root.after(200, self._update_bake_status)

    def _update_bake_status(self):
        """Update UI after baking completes."""
        if hasattr(self, 'stack_mode_label'):
            # Count how many faces have baked textures
            count = 0
            if self.region_effect_manager:
                count = len(self.region_effect_manager.baked_textures)
            self.stack_mode_label.config(text=f"Baked: {count} face(s) with effects")
        self._refresh_selected_for_effects()

    def _clear_effect_stack_mode(self):
        """Clear all baked effect textures."""
        if not self.region_effect_manager:
            print("Region effect manager not available")
            return

        # Clear all baked textures
        self.region_effect_manager.baked_textures.clear()

        # Reset any regions that were using baked_texture effect type
        for region_id, effect in self.region_effect_manager.region_effects.items():
            if effect.get('effect_type') == 'baked_texture':
                effect['effect_type'] = 'none'

        # Update UI label
        if hasattr(self, 'stack_mode_label'):
            self.stack_mode_label.config(text="Baked effects cleared")

        print("All baked effect textures cleared")

    def _toggle_per_region_mode(self):
        """Toggle the per-region effects visibility."""
        enabled = self.vars['per_region_mode'].get()
        self.detection_params['per_region_mode'] = enabled
        print(f"Per-region effects: {'ON' if enabled else 'OFF'}")

    # =============================================
    # LIVE FACE EFFECT METHODS
    # =============================================

    def _toggle_face_effects(self):
        """Toggle visibility of face effects (for performance)."""
        enabled = self.vars['show_face_effects'].get()
        self.detection_params['show_face_effects'] = enabled
        self.detection_params['per_region_mode'] = enabled
        print(f"Face effects visibility: {'ON' if enabled else 'OFF'}")

    def _toggle_exclude_faces(self):
        """Toggle whether selected faces are excluded from background effects."""
        enabled = self.vars['exclude_faces_from_bg'].get()
        self.detection_params['exclude_faces_from_bg'] = enabled
        print(f"Exclude faces from background: {'ON' if enabled else 'OFF'}")

    def _get_current_selected_face(self):
        """Get the currently selected face for live editing."""
        if not self.data_viz:
            return None, None

        selected = self.data_viz.get_selected_list()
        if not selected:
            return None, None

        # Use the first selected face for live editing
        det_id = selected[0]

        tracker = self.detection_tracker
        if not tracker:
            return None, None

        tracked = tracker.get_all_tracked()
        if det_id not in tracked:
            return None, None

        info = tracked[det_id]
        det_type = info.get('type', 'face')
        return det_id, det_type

    def _update_face_effect_status(self):
        """Update the status label for face effects."""
        det_id, det_type = self._get_current_selected_face()

        if det_id is None:
            if hasattr(self, 'face_effect_status'):
                self.face_effect_status.config(text="No face selected", foreground='gray')
        else:
            # Check if this face has a baked effect or a stack
            status_suffix = ""
            if self.region_effect_manager:
                region_key = f"{det_type}_{det_id}"
                if self.region_effect_manager.has_baked_texture(region_key):
                    status_suffix = " [BAKED]"
                else:
                    stack = self.region_effect_manager.get_effect_stack(region_key)
                    if stack:
                        status_suffix = f" [STACK: {len(stack)}]"

            if hasattr(self, 'face_effect_status'):
                color = 'green'
                if '[BAKED]' in status_suffix:
                    color = 'blue'
                elif '[STACK' in status_suffix:
                    color = 'orange'
                self.face_effect_status.config(
                    text=f"Editing: {det_type.title()} #{det_id}{status_suffix}",
                    foreground=color
                )

        # Update baked count
        if hasattr(self, 'baked_info_label') and self.region_effect_manager:
            count = len(self.region_effect_manager.baked_textures)
            self.baked_info_label.config(text=f"Baked: {count} face(s)")

        # Update stack display
        self._update_stack_display()

    def _load_face_effect_from_selected(self):
        """Load the current face effect settings from the first selected face."""
        det_id, det_type = self._get_current_selected_face()

        if det_id is None or not self.region_effect_manager:
            return

        region_key = f"{det_type}_{det_id}"

        # Don't load if baked (can't edit baked faces without clearing)
        if self.region_effect_manager.has_baked_texture(region_key):
            return

        effect = self.region_effect_manager.get_region_effect(region_key)

        # Update UI without triggering live apply
        self._loading_effect = True
        try:
            # Set effect type first
            effect_type = effect.get('effect_type', 'none')
            self.vars['face_effect_type'].set(effect_type)

            # Update sliders for this effect type
            self._update_face_param_sliders()

            # Load all dynamic parameters from the effect
            if 'face_params' in self.vars:
                for param_name, var in self.vars['face_params'].items():
                    if param_name in effect:
                        var.set(effect[param_name])
                        # Update the label if widget exists
                        if param_name in self.face_param_widgets:
                            label = self.face_param_widgets[param_name].get('label')
                            if label:
                                label.config(text=str(int(effect[param_name])))
        finally:
            self._loading_effect = False

    def _on_face_effect_changed(self, event=None):
        """Handle live face effect type change - applies immediately."""
        # Update the dynamic parameter sliders for this effect
        self._update_face_param_sliders()
        self._apply_live_face_effect()

    def _update_face_param_sliders(self):
        """Update the parameter sliders based on selected effect."""
        # Clear existing parameter widgets
        for widget in self.face_params_frame.winfo_children():
            widget.destroy()
        self.face_param_widgets.clear()

        effect_type = self.vars.get('face_effect_type', tk.StringVar(value='none')).get()

        # Get parameters for this effect
        param_names = self.effect_param_map.get(effect_type, [])

        if not param_names:
            # No parameters - show info label
            ttk.Label(self.face_params_frame, text="(No adjustable parameters)",
                      font=('Arial', 8, 'italic')).pack(anchor='w')
            return

        # Create sliders for each parameter
        for param_name in param_names:
            if param_name not in self.face_effect_params:
                continue

            label, min_val, max_val, default_val, resolution = self.face_effect_params[param_name]

            # Create row frame
            row = ttk.Frame(self.face_params_frame)
            row.pack(fill=tk.X, pady=2)

            # Label
            ttk.Label(row, text=f"{label}:", width=12).pack(side='left')

            # Variable
            if param_name not in self.vars.get('face_params', {}):
                if 'face_params' not in self.vars:
                    self.vars['face_params'] = {}
                self.vars['face_params'][param_name] = tk.DoubleVar(value=default_val)

            var = self.vars['face_params'][param_name]

            # Value label
            value_label = ttk.Label(row, text=str(int(var.get())), width=4)
            value_label.pack(side='right')

            # Slider
            def make_callback(pname, vlabel):
                def callback(val):
                    vlabel.config(text=str(int(float(val))))
                    self._apply_live_face_effect()
                return callback

            slider = ttk.Scale(
                row, from_=min_val, to=max_val, variable=var,
                orient='horizontal', command=make_callback(param_name, value_label)
            )
            slider.pack(side='left', fill=tk.X, expand=True, padx=5)

            self.face_param_widgets[param_name] = {
                'slider': slider,
                'label': value_label,
                'var': var
            }

    def _get_face_effect_params(self):
        """Get all current face effect parameters as a dict."""
        params = {}
        effect_type = self.vars.get('face_effect_type', tk.StringVar(value='none')).get()

        # Get dynamic parameters
        face_params = self.vars.get('face_params', {})
        param_names = self.effect_param_map.get(effect_type, [])

        for param_name in param_names:
            if param_name in face_params:
                params[param_name] = int(face_params[param_name].get())

        return params

    def _apply_live_face_effect(self):
        """Apply current face effect settings to all selected faces LIVE."""
        # Don't apply while loading effect from selection
        if getattr(self, '_loading_effect', False):
            return

        if not self.region_effect_manager:
            return

        if not self.vars.get('show_face_effects', tk.BooleanVar(value=True)).get():
            return  # Face effects disabled for performance

        data_viz = self.data_viz
        tracker = self.detection_tracker
        if not data_viz or not tracker:
            return

        selected = data_viz.get_selected_list()
        if not selected:
            return

        tracked = tracker.get_all_tracked()

        effect_type = self.vars['face_effect_type'].get()

        # Gather all parameters for this effect
        effect_params = self._get_face_effect_params()

        # Apply to all selected faces
        for det_id in selected:
            if det_id in tracked:
                info = tracked[det_id]
                det_type = info.get('type', 'face')
                region_key = f"{det_type}_{det_id}"

                # Don't overwrite baked textures
                if self.region_effect_manager.has_baked_texture(region_key):
                    continue

                self.region_effect_manager.set_region_effect(
                    region_key,
                    effect_type,
                    **effect_params
                )

        # Ensure effects are visible
        self.detection_params['per_region_mode'] = True

    def _bake_face_effect(self):
        """Bake the current effect stack onto selected faces (makes it permanent)."""
        data_viz = self.data_viz
        tracker = self.detection_tracker
        if not data_viz or not tracker:
            print("Data viz or tracker not available")
            return

        selected = data_viz.get_selected_list()
        if not selected:
            print("No faces selected to bake")
            return

        # Check if there's an effect to bake (either stacked or live)
        effect_type = self.vars.get('face_effect_type', tk.StringVar(value='none')).get()
        has_live_effect = effect_type != 'none'

        # Check if any selected face has a non-empty stack
        has_stack = False
        if self.region_effect_manager:
            tracked = tracker.get_all_tracked()
            for det_id in selected:
                if det_id in tracked:
                    info = tracked[det_id]
                    det_type = info.get('type', 'face')
                    region_key = f"{det_type}_{det_id}"
                    stack = self.region_effect_manager.get_effect_stack(region_key)
                    if stack:
                        has_stack = True
                        break

        if not has_live_effect and not has_stack:
            print("No effects to bake - add effects to the stack first or select a live effect")
            return

        # Signal main loop to bake on next frame
        self.detection_params['bake_effect_stack'] = True
        self.detection_params['bake_selected_ids'] = list(selected)

        stack_info = "stack" if has_stack else f"'{effect_type}'"
        print(f"Baking {stack_info} effect to {len(selected)} face(s)")

        # Reset UI to 'none' after baking - the effect is now baked, not live
        def after_bake():
            self._loading_effect = True
            try:
                self.vars['face_effect_type'].set('none')
                self._update_face_param_sliders()
                self._update_face_effect_status()
            finally:
                self._loading_effect = False

        # Update UI after bake (give time for main loop to process)
        if self.root:
            self.root.after(300, after_bake)

    def _add_effect_to_stack(self):
        """Add the current effect to the effect stack for selected faces."""
        if not self.region_effect_manager:
            print("Region effect manager not available")
            return

        data_viz = self.data_viz
        tracker = self.detection_tracker
        if not data_viz or not tracker:
            print("Data viz or tracker not available")
            return

        selected = data_viz.get_selected_list()
        if not selected:
            print("No faces selected")
            return

        effect_type = self.vars.get('face_effect_type', tk.StringVar(value='none')).get()
        if effect_type == 'none':
            print("No effect to add - select an effect first")
            return

        tracked = tracker.get_all_tracked()

        # Build effect parameters (without effect_type - that's passed separately)
        effect_params = {}
        if 'face_params' in self.vars:
            for param_name, var in self.vars['face_params'].items():
                try:
                    effect_params[param_name] = var.get()
                except:
                    pass

        # Add effect to stack for each selected face
        for det_id in selected:
            if det_id in tracked:
                info = tracked[det_id]
                det_type = info.get('type', 'face')
                region_key = f"{det_type}_{det_id}"
                stack_size = self.region_effect_manager.add_effect_to_stack(
                    region_key, effect_type, **effect_params
                )
                print(f"Added '{effect_type}' to stack for {region_key} (stack size: {stack_size})")

        # Update the stack display
        self._update_stack_display()

        # Optionally reset the dropdown to 'none' after adding
        # self.vars['face_effect_type'].set('none')
        # self._update_face_param_sliders()

    def _clear_effect_stack(self):
        """Clear the effect stack for selected faces."""
        if not self.region_effect_manager:
            return

        data_viz = self.data_viz
        tracker = self.detection_tracker
        if not data_viz or not tracker:
            return

        selected = data_viz.get_selected_list()
        tracked = tracker.get_all_tracked()

        for det_id in selected:
            if det_id in tracked:
                info = tracked[det_id]
                det_type = info.get('type', 'face')
                region_key = f"{det_type}_{det_id}"
                self.region_effect_manager.clear_effect_stack(region_key)
                print(f"Cleared effect stack for {region_key}")

        # Update display
        self._update_stack_display()

    def _unbake_face_effect(self):
        """Unbake/clear baked effects from selected faces."""
        if not self.region_effect_manager:
            return

        data_viz = self.data_viz
        tracker = self.detection_tracker
        if not data_viz or not tracker:
            return

        selected = data_viz.get_selected_list()
        tracked = tracker.get_all_tracked()

        unbaked_count = 0
        for det_id in selected:
            if det_id in tracked:
                info = tracked[det_id]
                det_type = info.get('type', 'face')
                region_key = f"{det_type}_{det_id}"

                # Clear baked texture
                self.region_effect_manager.clear_baked_texture(region_key)

                # Reset effect type from baked_texture to none
                effect = self.region_effect_manager.get_region_effect(region_key)
                if effect.get('effect_type') == 'baked_texture':
                    self.region_effect_manager.set_region_effect(region_key, 'none')

                unbaked_count += 1

        print(f"Unbaked {unbaked_count} face(s)")
        self._update_face_effect_status()
        self._update_stack_display()

    def _update_stack_display(self):
        """Update the effect stack display label."""
        if not hasattr(self, 'effect_stack_label'):
            return

        if not self.region_effect_manager:
            self.effect_stack_label.config(text="Stack: (no manager)")
            return

        data_viz = self.data_viz
        tracker = self.detection_tracker
        if not data_viz or not tracker:
            self.effect_stack_label.config(text="Stack: (empty)")
            return

        selected = data_viz.get_selected_list()
        if not selected:
            self.effect_stack_label.config(text="Stack: (no face selected)")
            return

        tracked = tracker.get_all_tracked()

        # Get stack from first selected face
        for det_id in selected:
            if det_id in tracked:
                info = tracked[det_id]
                det_type = info.get('type', 'face')
                region_key = f"{det_type}_{det_id}"
                stack = self.region_effect_manager.get_effect_stack(region_key)

                if not stack:
                    self.effect_stack_label.config(text="Stack: (empty)")
                else:
                    effect_names = [e.get('effect_type', '?') for e in stack]
                    stack_text = " → ".join(effect_names)
                    self.effect_stack_label.config(text=f"Stack: {stack_text}")
                break
        else:
            self.effect_stack_label.config(text="Stack: (empty)")

    def _reset_face_effect(self):
        """Reset effect on currently selected faces (clears stack, baked, and live)."""
        if not self.region_effect_manager:
            return

        data_viz = self.data_viz
        tracker = self.detection_tracker
        if not data_viz or not tracker:
            return

        selected = data_viz.get_selected_list()
        tracked = tracker.get_all_tracked()

        for det_id in selected:
            if det_id in tracked:
                info = tracked[det_id]
                det_type = info.get('type', 'face')
                region_key = f"{det_type}_{det_id}"

                # Clear effect stack
                self.region_effect_manager.clear_effect_stack(region_key)
                # Clear baked texture if exists
                self.region_effect_manager.clear_baked_texture(region_key)
                # Reset to none
                self.region_effect_manager.set_region_effect(region_key, 'none')

        # Reset UI
        self.vars['face_effect_type'].set('none')

        # Reset all dynamic parameters to defaults
        if 'face_params' in self.vars:
            for param_name, var in self.vars['face_params'].items():
                if param_name in self.face_effect_params:
                    default_val = self.face_effect_params[param_name][3]  # Index 3 is default
                    var.set(default_val)

        # Update sliders to show "no parameters"
        self._update_face_param_sliders()

        print(f"Reset effects on {len(selected)} face(s)")
        self._update_face_effect_status()
        self._update_stack_display()

    def _apply_to_all_selected_faces(self):
        """Apply current face effect to ALL selected faces."""
        self._apply_live_face_effect()
        print("Applied current effect to all selected faces")

    def _clear_all_baked(self):
        """Clear all baked face effects and stacks."""
        if not self.region_effect_manager:
            return

        self.region_effect_manager.baked_textures.clear()
        self.region_effect_manager.effect_stacks.clear()

        # Reset any regions using baked_texture
        for region_id, effect in self.region_effect_manager.region_effects.items():
            if effect.get('effect_type') == 'baked_texture':
                effect['effect_type'] = 'none'

        print("Cleared all baked face effects")
        self._update_face_effect_status()

    def _select_detections(self):
        """Select detections for visualization operations."""
        det_ids = self._get_selected_detection_ids()
        if not self.data_viz:
            return

        tracker = self.detection_tracker
        tracked = tracker.get_all_tracked() if tracker else {}

        for det_id in det_ids:
            # Get rect and type from tracker if available
            rect = None
            det_type = 'faces'
            if det_id in tracked:
                info = tracked[det_id]
                rect = info.get('rect')
                det_type = info.get('type', 'faces')
            self.data_viz.select_detection(det_id, rect=rect, det_type=det_type)

        self._refresh_detection_list()
        print(f"Selected {len(det_ids)} detection(s)")

    def _deselect_detections(self):
        """Deselect detections."""
        det_ids = self._get_selected_detection_ids()
        if not self.data_viz:
            return

        for det_id in det_ids:
            self.data_viz.deselect_detection(det_id)

        self._refresh_detection_list()
        print(f"Deselected {len(det_ids)} detection(s)")

    def _clear_selection(self):
        """Clear all selections."""
        if not self.data_viz:
            return

        self.data_viz.clear_selection()
        self._refresh_detection_list()
        print("Cleared all selections")

    def _connect_selected(self):
        """Create connections between all selected detections."""
        if not self.data_viz:
            return

        self.data_viz.connect_selected()
        print("Connected all selected detections")

    def _clear_connections(self):
        """Clear all manual connections."""
        if not self.data_viz:
            return

        self.data_viz.clear_connections()
        print("Cleared all manual connections")

    def _apply_viz_style_to_selected(self):
        """Apply visualization style to selected detections."""
        if not self.data_viz:
            return

        det_ids = self._get_selected_detection_ids()
        style = self.det_viz_style_combo.get()
        custom_label = self.det_custom_label_entry.get().strip()

        for det_id in det_ids:
            settings = {}
            if style != '(use global)':
                settings['style'] = style
            if custom_label:
                settings['custom_label'] = custom_label

            self.data_viz.set_detection_settings(det_id, **settings)

        print(f"Applied style to {len(det_ids)} detection(s)")

    def _apply_region_settings(self):
        """Apply current settings to selected region."""
        if not self.region_effect_manager:
            return

        region_id = self.vars['selected_region'].get()
        effect_type = self.vars['region_effect_type'].get()

        self.region_effect_manager.set_region_effect(
            region_id,
            effect_type,
            intensity=self.vars['region_intensity'].get(),
            color_shift=self.vars['region_color_shift'].get(),
            blend_mode=self.vars['region_blend_mode'].get(),
            blend_opacity=self.vars['region_opacity'].get()
        )
        print(f"Applied {effect_type} to {region_id}")

    def _reset_region(self):
        """Reset selected region to default."""
        if not self.region_effect_manager:
            return

        region_id = self.vars['selected_region'].get()
        self.region_effect_manager.reset_region(region_id)
        self._on_region_selected()  # Refresh UI
        print(f"Reset {region_id} to defaults")

    def _copy_to_all_regions(self):
        """Copy current region settings to all regions of same type."""
        if not self.region_effect_manager:
            return

        region_id = self.vars['selected_region'].get()
        region_type = region_id.rsplit('_', 1)[0]  # 'face', 'eye', or 'body'

        effect_type = self.vars['region_effect_type'].get()
        intensity = self.vars['region_intensity'].get()
        color_shift = self.vars['region_color_shift'].get()
        blend_mode = self.vars['region_blend_mode'].get()
        opacity = self.vars['region_opacity'].get()

        # Apply to all regions of same type (0-4)
        for i in range(5):
            target_id = f"{region_type}_{i}"
            self.region_effect_manager.set_region_effect(
                target_id,
                effect_type,
                intensity=intensity,
                color_shift=color_shift,
                blend_mode=blend_mode,
                blend_opacity=opacity
            )

        print(f"Copied settings to all {region_type} regions")

    def _refresh_regions(self):
        """Refresh region list based on detected regions."""
        # This could be enhanced to show only actually detected regions
        pass

    def _toggle_custom_masks(self):
        """Toggle custom mask system."""
        enabled = self.vars['custom_masks_enabled'].get()
        self.custom_mask_manager.enabled = enabled

    def _add_custom_region(self, shape):
        """Add a new custom mask region."""
        # Add region in center of typical frame
        region_id = self.custom_mask_manager.add_region(100, 100, 150, 150, shape)
        self._update_region_selector()
        self.custom_mask_manager.select_region(region_id)
        self.custom_region_selector.set(region_id)

    def _delete_selected_region(self):
        """Delete the selected custom region."""
        region_id = self.custom_mask_manager.selected_region_id
        if region_id:
            self.custom_mask_manager.remove_region(region_id)
            self._update_region_selector()

    def _clear_custom_regions(self):
        """Clear all custom regions."""
        self.custom_mask_manager.clear_all()
        self._update_region_selector()

    def _update_region_selector(self):
        """Update the region selector combobox."""
        regions = self.custom_mask_manager.get_all_regions()
        region_ids = [r.region_id for r in regions]
        self.custom_region_selector['values'] = region_ids
        if not region_ids:
            self.custom_region_selector.set('')

    def _on_custom_region_selected(self, event):
        """Handle region selection from combobox."""
        region_id = self.custom_region_selector.get()
        if region_id:
            self.custom_mask_manager.select_region(region_id)
            region = self.custom_mask_manager.get_region(region_id)
            if region:
                self.custom_effect_type.set(region.effect_type)
                self.vars['custom_intensity'].set(region.effect_params.get('intensity', 50))
                self.vars['custom_feather'].set(region.feather)
                self.vars['custom_invert'].set(region.invert_mask)
                self.vars['custom_show_border'].set(region.show_border)

    def _update_custom_region_effect(self, event=None):
        """Update the effect type for selected region."""
        region_id = self.custom_mask_manager.selected_region_id
        if region_id:
            region = self.custom_mask_manager.get_region(region_id)
            if region:
                region.effect_type = self.custom_effect_type.get()

    def _update_custom_region_params(self, event=None):
        """Update parameters for selected region."""
        region_id = self.custom_mask_manager.selected_region_id
        if region_id:
            region = self.custom_mask_manager.get_region(region_id)
            if region:
                region.effect_params['intensity'] = self.vars['custom_intensity'].get()
                region.feather = self.vars['custom_feather'].get()
                region.invert_mask = self.vars['custom_invert'].get()
                region.show_border = self.vars['custom_show_border'].get()

    def _update_custom_intensity_label(self, *args):
        """Update the intensity label when slider changes."""
        if hasattr(self, 'custom_intensity_label'):
            val = self.vars['custom_intensity'].get()
            self.custom_intensity_label.config(text=str(int(val)))

    def _update_custom_feather_label(self, *args):
        """Update the feather label when slider changes."""
        if hasattr(self, 'custom_feather_label'):
            val = self.vars['custom_feather'].get()
            self.custom_feather_label.config(text=str(int(val)))

    def _apply_custom_mask_effect(self):
        """Apply the current effect settings to the selected custom mask region."""
        region_id = self.custom_mask_manager.selected_region_id
        if not region_id:
            # Try to get from combobox
            region_id = self.custom_region_selector.get()
            if region_id:
                self.custom_mask_manager.select_region(region_id)

        if not region_id:
            if hasattr(self, 'custom_mask_status'):
                self.custom_mask_status.config(text="No region selected")
            print("No custom mask region selected")
            return

        region = self.custom_mask_manager.get_region(region_id)
        if not region:
            if hasattr(self, 'custom_mask_status'):
                self.custom_mask_status.config(text="Region not found")
            return

        # Apply all settings to the region
        effect_type = self.custom_effect_type.get()
        region.effect_type = effect_type
        region.effect_params['intensity'] = self.vars['custom_intensity'].get()
        region.effect_params['blur_amount'] = max(3, self.vars['custom_intensity'].get() // 2 * 2 + 1)  # Ensure odd
        region.effect_params['pixelate_size'] = max(2, self.vars['custom_intensity'].get() // 5)
        region.effect_params['color_shift'] = self.vars['custom_intensity'].get()
        region.feather = self.vars['custom_feather'].get()
        region.invert_mask = self.vars['custom_invert'].get()
        region.show_border = self.vars['custom_show_border'].get()

        # Update status
        if hasattr(self, 'custom_mask_status'):
            if effect_type == 'restore_original':
                self.custom_mask_status.config(text=f"Applied: Show original (cuts through effects)")
            elif effect_type == 'none':
                self.custom_mask_status.config(text=f"Effect cleared from {region_id}")
            else:
                self.custom_mask_status.config(text=f"Applied: {effect_type} to {region_id}")

        print(f"Applied effect '{effect_type}' to custom mask '{region_id}'")

    def _reset_custom_mask_effect(self):
        """Reset the selected custom mask region to no effect."""
        region_id = self.custom_mask_manager.selected_region_id
        if not region_id:
            region_id = self.custom_region_selector.get()

        if not region_id:
            return

        region = self.custom_mask_manager.get_region(region_id)
        if region:
            region.effect_type = 'none'
            region.effect_params = {
                'intensity': 50,
                'blur_amount': 31,
                'pixelate_size': 10,
                'color_shift': 90,
                'blend_opacity': 1.0,
            }
            region.feather = 10
            region.invert_mask = False

            # Update UI
            self.custom_effect_type.set('none')
            self.vars['custom_intensity'].set(50)
            self.vars['custom_feather'].set(10)
            self.vars['custom_invert'].set(False)

            if hasattr(self, 'custom_mask_status'):
                self.custom_mask_status.config(text=f"Reset {region_id}")

            print(f"Reset custom mask '{region_id}'")

    def _toggle_data_viz(self):
        """Toggle data visualization."""
        enabled = self.vars['dataviz_enabled'].get()
        self.data_viz.enabled = enabled
        print(f"Data visualization enabled: {enabled}")

    def _toggle_dataviz_selective(self):
        """Toggle selective mode for data visualization (only selected faces)."""
        self.data_viz.selective_mode = self.vars['dataviz_selective'].get()
        mode_str = "SELECTED FACES ONLY" if self.data_viz.selective_mode else "ALL FACES"
        print(f"Data visualization mode: {mode_str}")

    def _update_dataviz_style(self, event=None):
        """Update the box style."""
        self.data_viz.box_style = self.dataviz_style.get()

    def _update_dataviz_settings(self, event=None):
        """Update all data visualization settings."""
        self.data_viz.show_labels = self.vars['show_labels'].get()
        self.data_viz.show_id = self.vars['show_id'].get()
        self.data_viz.show_coordinates = self.vars['show_coordinates'].get()
        self.data_viz.show_center_point = self.vars['show_center'].get()
        self.data_viz.show_connections = self.vars['show_connections'].get()
        self.data_viz.connection_style = self.connection_style.get()
        self.data_viz.connect_same_type = self.vars['connect_same'].get()
        self.data_viz.connect_different_types = self.vars['connect_different'].get()
        self.data_viz.animate_boxes = self.vars['animate_boxes'].get()
        self.data_viz.scan_line_enabled = self.vars['scan_line'].get()
        self.data_viz.info_panel_enabled = self.vars['info_panel'].get()
        self.data_viz.info_panel_position = self.info_panel_pos.get()

    def _get_color_bgr(self, color_name):
        """Convert color name to BGR tuple."""
        color_map = {
            'cyan': (255, 255, 0),
            'magenta': (255, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'white': (255, 255, 255),
            'orange': (0, 165, 255),
            'purple': (128, 0, 128),
            'pink': (203, 192, 255),
        }
        return color_map.get(color_name, (0, 255, 255))

    def _apply_box_color_to_selected(self):
        """Apply the selected box color to all selected detections."""
        if not self.data_viz:
            print("No data_viz available")
            return

        selected_ids = self.data_viz.get_selected_list()
        if not selected_ids:
            print("No detections selected - select faces in Detection tab first")
            return

        color_name = self.vars['box_color'].get()
        color_bgr = self._get_color_bgr(color_name)

        for det_id in selected_ids:
            # Get or create settings for this detection
            settings = self.data_viz.get_detection_settings(det_id)
            settings['custom_color'] = color_bgr
            print(f"  Set det_id {det_id} color to {color_name} {color_bgr}")

        print(f"Applied {color_name} box color to {len(selected_ids)} detection(s)")

    def _on_thickness_change(self, *args):
        """Update thickness label when slider changes."""
        if hasattr(self, 'conn_thick_label'):
            thickness = int(self.vars['connection_thickness'].get())
            self.conn_thick_label.config(text=str(thickness))

    def _apply_connection_settings(self):
        """Apply connection line color and thickness settings."""
        if not self.data_viz:
            print("No data_viz available")
            return

        color_name = self.vars['connection_color'].get()
        color_bgr = self._get_color_bgr(color_name)
        thickness = int(self.vars['connection_thickness'].get())

        # Apply to data_viz
        self.data_viz.connection_color = color_bgr
        self.data_viz.connection_thickness = thickness

        # Update status
        if hasattr(self, 'conn_status_label'):
            self.conn_status_label.config(text=f"Applied: {color_name}, {thickness}px")

        print(f"Applied connection settings: color={color_name} {color_bgr}, thickness={thickness}")

    def _update_connection_color(self, event=None):
        """Update the global connection line color (called on combobox change)."""
        # Just update the preview - actual apply happens with button
        pass

    def _update_connection_thickness(self, event=None):
        """Update connection line thickness label."""
        if hasattr(self, 'conn_thick_label'):
            thickness = int(self.vars['connection_thickness'].get())
            self.conn_thick_label.config(text=str(thickness))

    def _connect_selected_with_color(self):
        """Connect all selected detections with the specified group color."""
        if not self.data_viz:
            print("No data_viz available")
            return

        selected_ids = self.data_viz.get_selected_list()
        if len(selected_ids) < 2:
            print("Need at least 2 selected detections to create connections")
            return

        color_name = self.vars['group_color'].get()
        color_bgr = self._get_color_bgr(color_name)

        # Create connections between all selected with this color
        # Store as a "connection group" with color info
        if not hasattr(self.data_viz, 'connection_groups'):
            self.data_viz.connection_groups = []

        group = {
            'ids': list(selected_ids),
            'color': color_bgr,
            'color_name': color_name
        }
        self.data_viz.connection_groups.append(group)

        print(f"Created connection group: IDs={selected_ids}, color={color_name} {color_bgr}")

        # Update the groups list UI
        self._refresh_connection_groups_list()
        print(f"Created {color_name} connection group with {len(selected_ids)} detections")

    def _refresh_connection_groups_list(self):
        """Refresh the connection groups listbox."""
        if not hasattr(self, 'connection_groups_list'):
            return

        self.connection_groups_list.delete(0, tk.END)

        if not self.data_viz or not hasattr(self.data_viz, 'connection_groups'):
            return

        for i, group in enumerate(self.data_viz.connection_groups):
            color_name = group.get('color_name', 'unknown')
            num_ids = len(group.get('ids', []))
            self.connection_groups_list.insert(tk.END, f"Group {i+1}: {color_name} ({num_ids} faces)")

    def _delete_connection_group(self):
        """Delete the selected connection group."""
        if not hasattr(self, 'connection_groups_list'):
            return

        selection = self.connection_groups_list.curselection()
        if not selection:
            return

        idx = selection[0]

        if self.data_viz and hasattr(self.data_viz, 'connection_groups'):
            if 0 <= idx < len(self.data_viz.connection_groups):
                del self.data_viz.connection_groups[idx]
                self._refresh_connection_groups_list()
                print(f"Deleted connection group {idx + 1}")

    def _clear_all_connections(self):
        """Clear all manual connections and connection groups."""
        if not self.data_viz:
            return

        self.data_viz.clear_connections()
        if hasattr(self.data_viz, 'connection_groups'):
            self.data_viz.connection_groups = []
        self._refresh_connection_groups_list()
        print("Cleared all connections")

    def _update_custom_label(self, det_type):
        """Update custom label for a detection type."""
        entry = self.custom_label_entries.get(det_type)
        if entry:
            text = entry.get().strip()
            if text:
                self.data_viz.custom_labels[det_type] = text
            elif det_type in self.data_viz.custom_labels:
                del self.data_viz.custom_labels[det_type]

    def _update_render_duration_label(self, event=None):
        """Update the estimated duration label based on frames and FPS."""
        num_frames = self.vars['gif_num_frames'].get()
        fps = self.vars['gif_fps'].get()
        duration = num_frames / fps if fps > 0 else 0

        self.gif_frame_count_label.config(text=f"{num_frames} frames")
        self.gif_fps_label.config(text=f"{fps} fps")
        self.gif_duration_label.config(text=f"Duration: ~{duration:.1f} sec")

    def _start_gif_render(self):
        """Start rendering GIF frames."""
        if self.gif_exporter is None:
            print("GIF exporter not initialized!")
            return

        if self.apply_effects_func is None:
            print("Effects function not set! Cannot render.")
            self.gif_status_label.config(text="Error: Effects function not set")
            return

        if self.current_source_frame is None:
            print("No source frame available!")
            self.gif_status_label.config(text="Error: No source frame")
            return

        # Disable render button, enable cancel
        self.gif_render_btn.config(state='disabled')
        self.gif_cancel_btn.config(state='normal')
        self.gif_progress_var.set(0)

        num_frames = self.vars['gif_num_frames'].get()
        fps = self.vars['gif_fps'].get()

        # Start rendering in a separate function that can be called iteratively
        # For now, do it synchronously but with progress updates
        self.gif_status_label.config(text="Rendering...")

        def progress_callback(progress, status):
            self.gif_progress_var.set(progress)
            self.gif_status_label.config(text=status)
            self.root.update_idletasks()

        # Check if we have a video source or static image
        if self.video_path and os.path.exists(self.video_path):
            # Render from video
            success = self.gif_exporter.render_gif_from_video(
                video_path=self.video_path,
                apply_effects_func=self.apply_effects_func,
                start_frame=0,
                num_frames=num_frames,
                fps=fps,
                detection_params=self.detection_params,
                region_effect_manager=self.region_effect_manager,
                progress_callback=progress_callback
            )
        else:
            # Render from static image
            success = self.gif_exporter.render_gif_frames(
                source_frame=self.current_source_frame,
                apply_effects_func=self.apply_effects_func,
                num_frames=num_frames,
                fps=fps,
                detection_params=self.detection_params,
                region_effect_manager=self.region_effect_manager,
                progress_callback=progress_callback
            )

        # Re-enable buttons
        self.gif_render_btn.config(state='normal')
        self.gif_cancel_btn.config(state='disabled')

        if success:
            frame_count = self.gif_exporter.get_frame_count()
            self.gif_rendered_label.config(text=f"Rendered: {frame_count} frames")
            self.gif_status_label.config(text=f"Done! {frame_count} frames ready to export")
        else:
            self.gif_status_label.config(text="Render failed or cancelled")

    def _cancel_gif_render(self):
        """Cancel ongoing GIF render."""
        if self.gif_exporter:
            self.gif_exporter.cancel_render()
            self.gif_status_label.config(text="Cancelling...")

    def _export_gif(self):
        """Export rendered frames as GIF."""
        if self.gif_exporter is None or self.gif_exporter.get_frame_count() == 0:
            print("No frames to export! Render frames first.")
            self.gif_status_label.config(text="No frames to export - render first")
            return

        # Get save path
        file_path = filedialog.asksaveasfilename(
            title="Save GIF",
            defaultextension=".gif",
            filetypes=[("GIF files", "*.gif"), ("All files", "*.*")]
        )

        if file_path:
            fps = self.vars['gif_fps'].get()
            self.gif_status_label.config(text="Exporting...")
            self.root.update_idletasks()

            success = self.gif_exporter.export_gif(file_path, fps=fps)
            if success:
                self.gif_status_label.config(text=f"Saved: {os.path.basename(file_path)}")
            else:
                self.gif_status_label.config(text="Export failed!")

    def _clear_gif_frames(self):
        """Clear all rendered GIF frames."""
        if self.gif_exporter:
            self.gif_exporter.clear()
            self.gif_rendered_label.config(text="Rendered: 0 frames")
            self.gif_status_label.config(text="Cleared")
            self.gif_progress_var.set(0)

    def _update_gif_status(self):
        """Update GIF status display."""
        if self.gif_exporter:
            frame_count = self.gif_exporter.get_frame_count()
            if self.gif_exporter.is_rendering:
                self.gif_status_label.config(text=self.gif_exporter.render_status)
                self.gif_progress_var.set(self.gif_exporter.render_progress)
            self.gif_rendered_label.config(text=f"Rendered: {frame_count} frames")

    def set_render_source(self, frame, video_path=None):
        """Set the source frame/video for rendering."""
        self.current_source_frame = frame
        self.video_path = video_path

    def set_apply_effects_func(self, func):
        """Set the function used to apply effects during rendering."""
        self.apply_effects_func = func

    def _update_param(self, key, value):
        """Update a detection parameter."""
        self.detection_params[key] = value

    def _on_sensitivity_changed(self, value):
        """Handle sensitivity slider change."""
        val = float(value)
        self.detection_params['detection_sensitivity'] = val
        self.sens_value_label.config(text=f"{val:.2f}")

    def _on_min_neighbors_changed(self, value):
        """Handle min neighbors slider change."""
        val = int(float(value))
        self.detection_params['min_neighbors'] = val
        self.neighbors_value_label.config(text=str(val))

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
