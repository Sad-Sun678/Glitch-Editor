"""Effect Control Panel - GUI for controlling video effects."""
import tkinter as tk
from tkinter import ttk, simpledialog
import os
import json

from utils.panel_utils import VIDEO_PRESETS_FILE


class EffectControlPanel:
    """A GUI panel with checkboxes and sliders to control effects."""

    def __init__(self, effect_states, effect_params):
        self.effect_states = effect_states
        self.effect_params = effect_params
        self.root = None
        self.running = False
        self.checkbox_vars = {}
        self.slider_vars = {}
        self.render_callback = None
        self.change_source_callback = None  # Callback for changing media source
        self.render_progress_var = None
        self.render_status_label = None
        self.preview_effects_var = None
        self.timeline_mode_label = None
        self.timeline_mode_var = None
        self.preview_scale_var = None
        self.show_fps_var = None  # FPS counter toggle
        self.source_label = None  # Label showing current source

        # Define effects with their parameters
        self.effects_config = [
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
            ('glitch_blocks', 'Glitch Blocks', 'G', 'glitch_blocks_count', 2, 20, 8),
            ('color_drift', 'Color Drift', 'D', 'color_drift_speed', 0.005, 0.1, 0.02),
            ('slit_scan', 'Slit Scan', 'Z', None, None, None),
            ('drunk', 'Drunk Effect', 'U', 'drunk_intensity', 5, 40, 15),
            ('ascii_art', 'ASCII Art', 'I', None, None, None),
            ('film_grain', 'Film Grain', 'F', 'film_grain_intensity', 5, 80, 30),
            ('tv_static', 'TV Static', 'O', 'tv_static_blend', 0.1, 0.8, 0.3),
            ('wave_distort', 'Wave Distort', 'Y', 'wave_amplitude', 5, 50, 20),
            ('oil_paint', 'Oil Paint', 'J', None, None, None),
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
        name = simpledialog.askstring("Save Preset", "Enter preset name:", parent=self.root)
        if not name:
            return
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
        if 'effect_states' in preset_data:
            for key, value in preset_data['effect_states'].items():
                if key in self.effect_states:
                    self.effect_states[key] = value
        if 'effect_params' in preset_data:
            for key, value in preset_data['effect_params'].items():
                if key in self.effect_params:
                    self.effect_params[key] = value
                    if key in self.slider_vars:
                        self.slider_vars[key].set(value)
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
        self.root.minsize(350, 400)  # Set minimum size to prevent resize crashes

        # Resize handling with debounce
        self._resize_after_id = None
        self._canvas = None
        self._scrollable_frame = None

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_frame)
        self._canvas = canvas
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        self._scrollable_frame = scrollable_frame

        def _on_frame_configure(event):
            """Handle scrollable frame resize with debounce."""
            try:
                if self._resize_after_id:
                    self.root.after_cancel(self._resize_after_id)
            except:
                pass
            self._resize_after_id = self.root.after(50, self._update_scroll_region)

        scrollable_frame.bind("<Configure>", _on_frame_configure)
        self._canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Handle canvas resize to update window width
        def _on_canvas_configure(event):
            try:
                canvas.itemconfig(self._canvas_window, width=event.width)
            except tk.TclError:
                pass
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            try:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except tk.TclError:
                pass

        # Bind mousewheel only when over this canvas
        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event):
            try:
                canvas.unbind_all("<MouseWheel>")
            except tk.TclError:
                pass

        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        title_label = tk.Label(scrollable_frame, text="Effect Controls", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)

        # Source info and change button
        source_frame = tk.Frame(scrollable_frame, bg="#1a1a2e")
        source_frame.pack(pady=5, padx=10, fill=tk.X)

        self.source_label = tk.Label(source_frame, text="Source: Not set", font=("Arial", 9),
                                      fg="#888", bg="#1a1a2e")
        self.source_label.pack(side=tk.LEFT, padx=5, pady=5)

        change_source_btn = tk.Button(source_frame, text="Change Source", command=self.request_source_change,
                                       bg="#9C27B0", fg="white", font=("Arial", 9))
        change_source_btn.pack(side=tk.RIGHT, padx=5, pady=5)

        button_frame = tk.Frame(scrollable_frame)
        button_frame.pack(pady=5, padx=10, fill=tk.X)

        reset_btn = tk.Button(button_frame, text="Reset All Effects", command=self.reset_all,
                              bg="#ff6b6b", fg="white", font=("Arial", 10, "bold"))
        reset_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))

        render_btn = tk.Button(button_frame, text="Render Video", command=self.start_render,
                               bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        render_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))

        render_frame = tk.Frame(scrollable_frame)
        render_frame.pack(pady=5, padx=10, fill=tk.X)

        self.render_progress_var = tk.DoubleVar(value=0)
        self.render_progress_bar = ttk.Progressbar(render_frame, variable=self.render_progress_var,
                                                    maximum=100, mode='determinate')
        self.render_progress_bar.pack(fill=tk.X)

        self.render_status_label = tk.Label(render_frame, text="Ready to render", font=("Arial", 8))
        self.render_status_label.pack()

        preview_frame = tk.Frame(scrollable_frame)
        preview_frame.pack(pady=5, padx=10, fill=tk.X)

        self.preview_effects_var = tk.BooleanVar(value=True)
        preview_checkbox = tk.Checkbutton(preview_frame,
            text="Show Effects in Live Preview (uncheck for better performance)",
            variable=self.preview_effects_var, font=("Arial", 9))
        preview_checkbox.pack(anchor='w')

        scale_frame = tk.Frame(preview_frame)
        scale_frame.pack(fill=tk.X, pady=5)

        scale_label = tk.Label(scale_frame, text="Preview Size:", font=("Arial", 9))
        scale_label.pack(side=tk.LEFT)

        self.preview_scale_var = tk.DoubleVar(value=1.0)
        self.preview_scale_slider = tk.Scale(scale_frame, from_=0.25, to=1.0, orient=tk.HORIZONTAL,
            variable=self.preview_scale_var, resolution=0.05, length=200, font=("Arial", 8))
        self.preview_scale_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        hint_label = tk.Label(preview_frame,
            text="Tip: Press 'Space' to pause, '`' for effects, 'Tab' for timeline mode",
            font=("Arial", 8), fg="gray")
        hint_label.pack(anchor='w')

        # FPS display toggle
        fps_frame = tk.Frame(preview_frame)
        fps_frame.pack(fill=tk.X, pady=2)
        self.show_fps_var = tk.BooleanVar(value=False)
        fps_check = tk.Checkbutton(fps_frame, text="Show FPS Counter",
            variable=self.show_fps_var, font=("Arial", 9))
        fps_check.pack(side=tk.LEFT)

        self.timeline_mode_var = tk.BooleanVar(value=False)
        timeline_frame = tk.Frame(scrollable_frame, bg="#1a1a2e")
        timeline_frame.pack(fill=tk.X, padx=10, pady=5)

        self.timeline_mode_label = tk.Label(timeline_frame,
            text="Timeline Mode: OFF (using manual effects)",
            font=("Arial", 9, "bold"), fg="#888", bg="#1a1a2e")
        self.timeline_mode_label.pack(side=tk.LEFT, padx=5, pady=5)

        tk.Label(timeline_frame, text="(Press Tab to toggle)", font=("Arial", 8),
                 fg="#666", bg="#1a1a2e").pack(side=tk.LEFT)

        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=10, padx=10)

        preset_frame = tk.LabelFrame(scrollable_frame, text="Custom Presets", font=("Arial", 10, "bold"))
        preset_frame.pack(pady=5, padx=10, fill=tk.X)

        save_preset_btn = tk.Button(preset_frame, text="Save Current as Preset",
            command=self.save_current_as_preset, bg="#2196F3", fg="white")
        save_preset_btn.pack(fill=tk.X, padx=5, pady=2)

        preset_select_frame = tk.Frame(preset_frame)
        preset_select_frame.pack(fill=tk.X, padx=5, pady=2)

        self.preset_var = tk.StringVar()
        preset_names = list(self.custom_presets.keys()) if self.custom_presets else ['']
        self.preset_dropdown = tk.OptionMenu(preset_select_frame, self.preset_var,
                                              *preset_names if preset_names else [''])
        self.preset_dropdown.config(width=20)
        self.preset_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)

        load_preset_btn = tk.Button(preset_select_frame, text="Load",
            command=lambda: self.load_preset(self.preset_var.get()), bg="#4CAF50", fg="white", width=6)
        load_preset_btn.pack(side=tk.LEFT, padx=2)

        delete_preset_btn = tk.Button(preset_select_frame, text="Delete",
            command=self.delete_preset, bg="#f44336", fg="white", width=6)
        delete_preset_btn.pack(side=tk.LEFT, padx=2)

        if self.custom_presets:
            self.preset_var.set(list(self.custom_presets.keys())[0])

        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, pady=10, padx=10)

        for config in self.effects_config:
            self.create_effect_control(scrollable_frame, config)

        self.running = True

    def create_effect_control(self, parent, config):
        """Create a checkbox and optional slider for an effect."""
        if len(config) == 6:
            effect_key, display_name, hotkey, param_key, param_min, param_max = config
            param_default = None
        elif len(config) == 7:
            effect_key, display_name, hotkey, param_key, param_min, param_max, param_default = config
        else:
            effect_key, display_name, hotkey = config[0], config[1], config[2]
            param_key, param_min, param_max, param_default = None, None, None, None

        effect_frame = tk.Frame(parent, relief=tk.GROOVE, borderwidth=1)
        effect_frame.pack(fill=tk.X, padx=5, pady=2)

        var = tk.BooleanVar(value=self.effect_states.get(f'{effect_key}_enabled', False))
        self.checkbox_vars[effect_key] = var

        checkbox = tk.Checkbutton(effect_frame, text=f"{display_name} [{hotkey}]",
            variable=var, command=lambda k=effect_key: self.toggle_effect(k), font=("Arial", 9))
        checkbox.pack(anchor='w', padx=5)

        if param_key and param_min is not None and param_max is not None:
            slider_frame = tk.Frame(effect_frame)
            slider_frame.pack(fill=tk.X, padx=20, pady=2)

            if isinstance(param_min, float) or isinstance(param_max, float):
                resolution = 0.01
                slider_var = tk.DoubleVar(value=self.effect_params.get(param_key, param_default))
            else:
                resolution = 1
                slider_var = tk.IntVar(value=self.effect_params.get(param_key, param_default))

            self.slider_vars[param_key] = slider_var

            slider = tk.Scale(slider_frame, from_=param_min, to=param_max, orient=tk.HORIZONTAL,
                variable=slider_var, resolution=resolution, length=300,
                command=lambda val, pk=param_key: self.update_param(pk, val))
            slider.pack(fill=tk.X)

    def toggle_effect(self, effect_key):
        """Toggle an effect on/off."""
        state = self.checkbox_vars[effect_key].get()
        self.effect_states[f'{effect_key}_enabled'] = state

    def update_param(self, param_key, value):
        """Update an effect parameter."""
        try:
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

    def request_source_change(self):
        """Request a source change via callback."""
        if self.change_source_callback:
            self.change_source_callback()

    def update_source_label(self, source_text):
        """Update the source label with current source info."""
        try:
            if self.source_label:
                self.source_label.config(text=f"Source: {source_text}")
        except tk.TclError:
            pass

    def update_render_progress(self, progress, status_text):
        """Update the render progress bar and status."""
        try:
            if self.render_progress_var:
                self.render_progress_var.set(progress)
            if self.render_status_label:
                self.render_status_label.config(text=status_text)
            if self.root:
                self.root.update_idletasks()
        except tk.TclError:
            pass

    def sync_from_keyboard(self):
        """Sync checkbox states from keyboard toggles."""
        for config in self.effects_config:
            effect_key = config[0]
            state_key = f'{effect_key}_enabled'
            if state_key in self.effect_states and effect_key in self.checkbox_vars:
                self.checkbox_vars[effect_key].set(self.effect_states[state_key])

    def _update_scroll_region(self):
        """Update scroll region safely."""
        try:
            if self._canvas and self._scrollable_frame:
                self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        except tk.TclError:
            pass
        finally:
            self._resize_after_id = None

    def update(self):
        """Update the panel (call from main loop). Main.py handles actual Tkinter updates."""
        if self.root and self.running:
            try:
                self.sync_from_keyboard()
            except tk.TclError:
                self.running = False
            except RuntimeError:
                pass  # Ignore RuntimeError during resize

    def update_timeline_mode(self, is_enabled):
        """Update the timeline mode indicator label."""
        try:
            if self.timeline_mode_label:
                if is_enabled:
                    self.timeline_mode_label.config(
                        text="Timeline Mode: ON (using timeline effects)",
                        fg="#4CAF50")
                else:
                    self.timeline_mode_label.config(
                        text="Timeline Mode: OFF (using manual effects)",
                        fg="#888")
                if self.timeline_mode_var:
                    self.timeline_mode_var.set(is_enabled)
        except tk.TclError:
            pass

    def close(self):
        """Close the panel."""
        self.running = False
        if self.root:
            try:
                self.root.destroy()
            except:
                pass
