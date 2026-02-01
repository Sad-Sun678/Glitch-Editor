"""Timeline Panel - GUI for scheduling effects and audio on a timeline."""
import tkinter as tk
from tkinter import ttk, simpledialog, filedialog, messagebox
import os
import json
import shutil
import subprocess
import tempfile
import threading
import time

from panel_utils import VIDEO_PRESETS_FILE, PYGAME_AVAILABLE

if PYGAME_AVAILABLE:
    import pygame

class TimelinePanel:
    """A timeline editor for scheduling effects and audio tracks."""

    def __init__(self, video_fps=30, total_frames=0, effect_states=None, effect_params=None):
        self.video_fps = video_fps
        self.total_frames = total_frames
        self.total_duration = total_frames / video_fps if video_fps > 0 else 0
        self.effect_states = effect_states or {}
        self.effect_params = effect_params or {}
        self.root = None
        self.running = False
        self.temp_dir = tempfile.mkdtemp()

        # Timeline data
        self.effect_keyframes = []  # List of {start_time, end_time, preset_name, preset_data}
        self.audio_tracks = []  # List of {start_time, file_path, volume, name, processed_path}

        # UI state
        self.canvas = None
        self.timeline_canvas = None
        self.current_time = 0
        self.zoom_level = 1.0  # Pixels per second
        self.scroll_offset = 0
        self.selected_keyframe = None
        self.selected_audio = None
        self.is_playing = False

        # Track heights
        self.effect_track_height = 40
        self.audio_track_height = 35
        self.timeline_height = 30
        self.track_padding = 5

        # Load video presets for reference
        self.video_presets = self.load_video_presets()

        # Pygame channels for multi-track audio
        self.audio_channels = {}
        if PYGAME_AVAILABLE:
            pygame.mixer.set_num_channels(16)  # Allow up to 16 simultaneous audio tracks

    def load_video_presets(self):
        """Load video presets from file."""
        try:
            if os.path.exists(VIDEO_PRESETS_FILE):
                with open(VIDEO_PRESETS_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading video presets: {e}")
        return {}

    def create_panel(self):
        """Create the timeline panel window."""
        self.root = tk.Toplevel()
        self.root.title("Timeline Editor")
        self.root.geometry("1000x500")
        self.root.resizable(True, True)
        self.root.minsize(600, 300)  # Set minimum size to prevent resize crashes

        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top toolbar
        toolbar = tk.Frame(main_frame, height=40, bg="#333")
        toolbar.pack(fill=tk.X, side=tk.TOP)
        toolbar.pack_propagate(False)

        # Toolbar buttons with tooltips
        tk.Button(toolbar, text="➕ Add Effect", command=self.add_effect_keyframe,
                  bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(toolbar, text="🎵 Add Audio", command=self.add_audio_track,
                  bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(toolbar, text="🗑️ Delete", command=self.delete_selected,
                  bg="#f44336", fg="white").pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Separator(toolbar, orient='vertical').pack(side=tk.LEFT, fill='y', padx=5, pady=5)

        tk.Button(toolbar, text="💾 Save", command=self.save_timeline,
                  bg="#FF9800", fg="white").pack(side=tk.LEFT, padx=3, pady=5)
        tk.Button(toolbar, text="📂 Load", command=self.load_timeline,
                  bg="#9C27B0", fg="white").pack(side=tk.LEFT, padx=3, pady=5)

        ttk.Separator(toolbar, orient='vertical').pack(side=tk.LEFT, fill='y', padx=5, pady=5)

        tk.Button(toolbar, text="▶ Preview", command=self.preview_timeline,
                  bg="#00BCD4", fg="white").pack(side=tk.LEFT, padx=3, pady=5)
        tk.Button(toolbar, text="⬛ Stop", command=self.stop_preview,
                  bg="#607D8B", fg="white").pack(side=tk.LEFT, padx=3, pady=5)

        ttk.Separator(toolbar, orient='vertical').pack(side=tk.LEFT, fill='y', padx=5, pady=5)

        # Help button
        tk.Button(toolbar, text="❓ Help", command=self.show_help,
                  bg="#555", fg="white").pack(side=tk.LEFT, padx=3, pady=5)

        # Zoom controls
        tk.Label(toolbar, text="Zoom:", bg="#333", fg="white").pack(side=tk.LEFT, padx=(10, 3), pady=5)
        self.zoom_scale = tk.Scale(toolbar, from_=10, to=200, orient=tk.HORIZONTAL,
                                    length=80, command=self.on_zoom_change, bg="#333", fg="white",
                                    highlightthickness=0)
        self.zoom_scale.set(50)
        self.zoom_scale.pack(side=tk.LEFT, padx=3, pady=5)

        # Current time display
        self.time_label = tk.Label(toolbar, text="Time: 0.00s / 0.00s", bg="#333", fg="#00FF00",
                                    font=("Consolas", 10, "bold"))
        self.time_label.pack(side=tk.RIGHT, padx=10, pady=5)

        # Timeline area with scrollbar
        timeline_container = tk.Frame(main_frame)
        timeline_container.pack(fill=tk.BOTH, expand=True)

        # Horizontal scrollbar
        self.h_scrollbar = tk.Scrollbar(timeline_container, orient=tk.HORIZONTAL)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Track labels frame (left side)
        labels_frame = tk.Frame(timeline_container, width=130, bg="#2a2a2a")
        labels_frame.pack(side=tk.LEFT, fill=tk.Y)
        labels_frame.pack_propagate(False)

        # Timeline labels with clearer hierarchy
        tk.Label(labels_frame, text="TIME RULER", bg="#2a2a2a", fg="#888",
                 font=("Arial", 8)).pack(pady=(8, 2))
        tk.Label(labels_frame, text="(click to seek)", bg="#2a2a2a", fg="#555",
                 font=("Arial", 7)).pack(pady=(0, 5))

        tk.Label(labels_frame, text="EFFECTS", bg="#2a2a2a", fg="#4CAF50",
                 font=("Arial", 9, "bold")).pack(pady=(10, 2))
        tk.Label(labels_frame, text="(video presets)", bg="#2a2a2a", fg="#3d8b40",
                 font=("Arial", 7)).pack(pady=(0, 8))

        tk.Label(labels_frame, text="AUDIO", bg="#2a2a2a", fg="#2196F3",
                 font=("Arial", 9, "bold")).pack(pady=(5, 2))
        tk.Label(labels_frame, text="(drag to move)", bg="#2a2a2a", fg="#1565C0",
                 font=("Arial", 7)).pack(pady=(0, 5))

        # Canvas for timeline
        self.timeline_canvas = tk.Canvas(timeline_container, bg="#1a1a1a",
                                          xscrollcommand=self.h_scrollbar.set)
        self.timeline_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.h_scrollbar.config(command=self.timeline_canvas.xview)

        # Bind events
        self.timeline_canvas.bind("<Button-1>", self.on_canvas_click)
        self.timeline_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.timeline_canvas.bind("<Double-Button-1>", self.on_canvas_double_click)
        self.timeline_canvas.bind("<Configure>", self.on_canvas_resize)

        # Bottom panel for editing selected item
        edit_frame = tk.LabelFrame(main_frame, text="Edit Selected", height=120)
        edit_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
        edit_frame.pack_propagate(False)

        # Edit controls container
        self.edit_container = tk.Frame(edit_frame)
        self.edit_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.edit_info_label = tk.Label(self.edit_container,
                                         text="Click on a green effect block or blue audio block to select and edit it",
                                         font=("Arial", 10), fg="#888")
        self.edit_info_label.pack(pady=20)

        # Text input area for manual keyframe entry
        manual_frame = tk.LabelFrame(main_frame, text="Quick Add Effect (type: START-END:PRESET_NAME)")
        manual_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)

        manual_inner = tk.Frame(manual_frame)
        manual_inner.pack(fill=tk.X, padx=5, pady=3)

        self.manual_entry = tk.Entry(manual_inner, width=40, font=("Consolas", 10))
        self.manual_entry.pack(side=tk.LEFT, padx=5, pady=3)
        self.manual_entry.bind("<FocusIn>", self._clear_placeholder)
        self.manual_entry.bind("<FocusOut>", self._restore_placeholder)
        self._placeholder_text = "e.g. 5-10:Glitch or 0-end:MyPreset"
        self.manual_entry.insert(0, self._placeholder_text)
        self.manual_entry.config(fg="gray")

        tk.Button(manual_inner, text="➕ Add Effect", command=self.add_from_manual_entry,
                  bg="#4CAF50", fg="white", font=("Arial", 9)).pack(side=tk.LEFT, padx=5, pady=3)

        # Show available presets
        tk.Label(manual_inner, text="Presets:", font=("Arial", 8), fg="#888").pack(side=tk.LEFT, padx=(15, 3))
        preset_names = list(self.video_presets.keys())[:5]  # Show first 5
        preset_text = ", ".join(preset_names) if preset_names else "(none saved)"
        if len(self.video_presets) > 5:
            preset_text += f" +{len(self.video_presets)-5} more"
        tk.Label(manual_inner, text=preset_text, font=("Arial", 8), fg="#4CAF50").pack(side=tk.LEFT)

        # Status bar at bottom
        status_frame = tk.Frame(main_frame, bg="#222", height=25)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(status_frame,
            text="Press TAB in the video window to toggle Timeline Mode | Double-click ruler to add effects | Drag blocks to reposition",
            bg="#222", fg="#888", font=("Arial", 8))
        self.status_label.pack(side=tk.LEFT, padx=10, pady=3)

        # Video duration info
        duration_text = f"Video: {self.total_duration:.1f}s ({self.total_frames} frames @ {self.video_fps:.1f}fps)"
        tk.Label(status_frame, text=duration_text, bg="#222", fg="#4CAF50",
                 font=("Arial", 8)).pack(side=tk.RIGHT, padx=10, pady=3)

        self.running = True
        self.draw_timeline()

    def on_zoom_change(self, value):
        """Handle zoom level change."""
        self.zoom_level = float(value) / 10.0  # Convert to pixels per second
        self.draw_timeline()

    def on_canvas_resize(self, event):
        """Handle canvas resize with crash protection."""
        if not self.running:
            return
        try:
            # Check for valid dimensions before proceeding
            if event.width < 10 or event.height < 10:
                return

            # Debounce rapid resize events
            if hasattr(self, '_resize_after_id') and self._resize_after_id:
                try:
                    self.root.after_cancel(self._resize_after_id)
                except:
                    pass
            self._resize_after_id = self.root.after(100, self._do_resize)
        except (tk.TclError, RuntimeError):
            pass
        except Exception:
            pass

    def _do_resize(self):
        """Actually perform the resize draw."""
        if not self.running:
            return
        try:
            if self.running and self.timeline_canvas:
                # Verify canvas still exists and has valid dimensions
                canvas_width = self.timeline_canvas.winfo_width()
                canvas_height = self.timeline_canvas.winfo_height()
                if canvas_width > 10 and canvas_height > 10:
                    self.draw_timeline()
        except (tk.TclError, RuntimeError):
            pass
        except Exception:
            pass
        finally:
            self._resize_after_id = None

    def time_to_x(self, time_sec):
        """Convert time in seconds to x pixel position."""
        return int(time_sec * self.zoom_level * 10) + 10

    def x_to_time(self, x):
        """Convert x pixel position to time in seconds."""
        return max(0, (x - 10) / (self.zoom_level * 10))

    def draw_timeline(self):
        """Draw the complete timeline with crash protection."""
        if not self.timeline_canvas or not self.running:
            return

        try:
            # Verify canvas dimensions are valid before drawing
            canvas_width = self.timeline_canvas.winfo_width()
            canvas_height = self.timeline_canvas.winfo_height()
            if canvas_height <= 10 or canvas_width <= 10:  # Canvas not ready yet
                return

            self.timeline_canvas.delete("all")

            total_width = max(self.time_to_x(self.total_duration) + 100, canvas_width)

            # Set scroll region
            self.timeline_canvas.config(scrollregion=(0, 0, total_width, canvas_height))

            # Draw time ruler
            self.draw_time_ruler()

            # Draw effect track background
            effect_y = self.timeline_height + self.track_padding
            self.timeline_canvas.create_rectangle(0, effect_y, total_width,
                                                   effect_y + self.effect_track_height,
                                                   fill="#2d3d2d", outline="")

            # Draw audio track background
            audio_y = effect_y + self.effect_track_height + self.track_padding
            self.timeline_canvas.create_rectangle(0, audio_y, total_width,
                                                   audio_y + self.audio_track_height * 4,
                                                   fill="#2d2d3d", outline="")
            # Draw effect keyframes
            for i, kf in enumerate(self.effect_keyframes):
                self.draw_effect_keyframe(kf, i, effect_y)

            # Draw audio tracks
            for i, track in enumerate(self.audio_tracks):
                self.draw_audio_track(track, i, audio_y)

            # Draw playhead
            self.draw_playhead()

        except tk.TclError:
            # Canvas was destroyed during drawing
            pass

    def draw_time_ruler(self):
        """Draw the time ruler at the top."""
        canvas_width = max(self.timeline_canvas.winfo_width(),
                           self.time_to_x(self.total_duration) + 100)

        # Background
        self.timeline_canvas.create_rectangle(0, 0, canvas_width, self.timeline_height,
                                               fill="#333", outline="")

        # Time markers
        interval = 1.0  # 1 second intervals
        if self.zoom_level < 2:
            interval = 5.0
        elif self.zoom_level < 5:
            interval = 2.0
        elif self.zoom_level > 15:
            interval = 0.5

        t = 0
        while t <= self.total_duration:
            x = self.time_to_x(t)
            # Major tick
            self.timeline_canvas.create_line(x, self.timeline_height - 10, x, self.timeline_height,
                                              fill="#888")
            # Time label
            if t == int(t):
                label = f"{int(t)}s"
            else:
                label = f"{t:.1f}s"
            self.timeline_canvas.create_text(x, self.timeline_height - 15, text=label,
                                              fill="white", font=("Arial", 8))
            t += interval

    def draw_effect_keyframe(self, keyframe, index, y_offset):
        """Draw a single effect keyframe block."""
        x1 = self.time_to_x(keyframe['start_time'])
        x2 = self.time_to_x(keyframe['end_time'])
        y1 = y_offset + 2
        y2 = y_offset + self.effect_track_height - 2

        # Determine color based on selection
        if self.selected_keyframe == index:
            fill_color = "#7CB342"
            outline_color = "#FFFFFF"
        else:
            fill_color = "#4CAF50"
            outline_color = "#2E7D32"

        # Draw block
        self.timeline_canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color,
                                               outline=outline_color, width=2,
                                               tags=f"effect_{index}")

        # Draw label
        label = keyframe.get('preset_name', 'Effect')[:15]
        self.timeline_canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2,
                                          text=label, fill="white",
                                          font=("Arial", 8, "bold"),
                                          tags=f"effect_{index}")

    def draw_audio_track(self, track, index, y_offset):
        """Draw a single audio track block."""
        x1 = self.time_to_x(track['start_time'])

        # Estimate duration from file or use default
        duration = track.get('duration', 5.0)
        x2 = self.time_to_x(track['start_time'] + duration)

        y1 = y_offset + (index % 4) * self.audio_track_height + 2
        y2 = y1 + self.audio_track_height - 4

        # Determine color based on selection
        if self.selected_audio == index:
            fill_color = "#42A5F5"
            outline_color = "#FFFFFF"
        else:
            fill_color = "#2196F3"
            outline_color = "#1565C0"

        # Draw block
        self.timeline_canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color,
                                               outline=outline_color, width=2,
                                               tags=f"audio_{index}")

        # Draw label
        label = track.get('name', 'Audio')[:12]
        self.timeline_canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2,
                                          text=f"🎵 {label}", fill="white",
                                          font=("Arial", 8),
                                          tags=f"audio_{index}")

    def draw_playhead(self):
        """Draw the current time playhead."""
        x = self.time_to_x(self.current_time)
        canvas_height = self.timeline_canvas.winfo_height()

        self.timeline_canvas.create_line(x, 0, x, canvas_height,
                                          fill="#FF5722", width=2, tags="playhead")
        self.timeline_canvas.create_polygon(x - 6, 0, x + 6, 0, x, 10,
                                             fill="#FF5722", tags="playhead")

    def on_canvas_click(self, event):
        """Handle click on canvas."""
        x = self.timeline_canvas.canvasx(event.x)
        y = event.y

        # Check if clicking on time ruler - set playhead
        if y < self.timeline_height:
            self.current_time = self.x_to_time(x)
            self.update_time_display()
            self.draw_timeline()
            return

        # Check if clicking on effect keyframes
        effect_y = self.timeline_height + self.track_padding
        if effect_y <= y <= effect_y + self.effect_track_height:
            click_time = self.x_to_time(x)
            for i, kf in enumerate(self.effect_keyframes):
                if kf['start_time'] <= click_time <= kf['end_time']:
                    self.selected_keyframe = i
                    self.selected_audio = None
                    self.show_effect_edit_panel(i)
                    self.draw_timeline()
                    return
            # Clicked empty space in effect track
            self.selected_keyframe = None
            self.selected_audio = None
            self.clear_edit_panel()
            self.draw_timeline()
            return

        # Check if clicking on audio tracks
        audio_y = effect_y + self.effect_track_height + self.track_padding
        if y >= audio_y:
            click_time = self.x_to_time(x)
            for i, track in enumerate(self.audio_tracks):
                duration = track.get('duration', 5.0)
                track_y1 = audio_y + (i % 4) * self.audio_track_height
                track_y2 = track_y1 + self.audio_track_height
                if (track['start_time'] <= click_time <= track['start_time'] + duration and
                    track_y1 <= y <= track_y2):
                    self.selected_audio = i
                    self.selected_keyframe = None
                    self.show_audio_edit_panel(i)
                    self.draw_timeline()
                    return

        # Clicked empty space
        self.selected_keyframe = None
        self.selected_audio = None
        self.clear_edit_panel()
        self.draw_timeline()

    def on_canvas_drag(self, event):
        """Handle drag on canvas for moving items."""
        x = self.timeline_canvas.canvasx(event.x)
        new_time = self.x_to_time(x)

        if self.selected_keyframe is not None:
            kf = self.effect_keyframes[self.selected_keyframe]
            duration = kf['end_time'] - kf['start_time']
            kf['start_time'] = max(0, min(new_time, self.total_duration - duration))
            kf['end_time'] = kf['start_time'] + duration
            self.draw_timeline()

        elif self.selected_audio is not None:
            track = self.audio_tracks[self.selected_audio]
            track['start_time'] = max(0, new_time)
            self.draw_timeline()

    def on_canvas_double_click(self, event):
        """Handle double-click to edit item."""
        # Double-click on time ruler to add keyframe at that time
        y = event.y
        if y < self.timeline_height:
            x = self.timeline_canvas.canvasx(event.x)
            self.current_time = self.x_to_time(x)
            self.add_effect_keyframe_at_time(self.current_time)

    def update_time_display(self):
        """Update the time label."""
        try:
            if self.time_label:
                self.time_label.config(text=f"Time: {self.current_time:.2f}s / {self.total_duration:.2f}s")
        except tk.TclError:
            pass

    def _clear_placeholder(self, event):
        """Clear placeholder text when entry is focused."""
        if self.manual_entry.get() == self._placeholder_text:
            self.manual_entry.delete(0, tk.END)
            self.manual_entry.config(fg="white")

    def _restore_placeholder(self, event):
        """Restore placeholder text if entry is empty."""
        if not self.manual_entry.get().strip():
            self.manual_entry.insert(0, self._placeholder_text)
            self.manual_entry.config(fg="gray")

    def show_help(self):
        """Show help dialog with timeline usage instructions."""
        help_dialog = tk.Toplevel(self.root)
        help_dialog.title("Timeline Editor Help")
        help_dialog.geometry("500x450")
        help_dialog.transient(self.root)

        # Make it scrollable
        canvas = tk.Canvas(help_dialog, bg="#1a1a2e")
        scrollbar = tk.Scrollbar(help_dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#1a1a2e")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Help content
        help_text = """
TIMELINE EDITOR GUIDE

HOW TO USE:
1. Press TAB to enable Timeline Mode (shows in Control Panel)
2. Add effects and audio to the timeline
3. Play your video - effects apply at scheduled times
4. Render to export with timeline effects baked in

ADDING EFFECTS:
• Click "Add Effect" button and select a preset
• Or double-click on the time ruler to add at that time
• Or type in Quick Add: 5-10:PresetName (seconds)
• Use "end" for video end: 0-end:MyPreset

ADDING AUDIO:
• Click "Add Audio" and select a file (MP3, WAV, etc.)
• Audio will be placed at the current playhead position
• Drag audio blocks left/right to reposition

EDITING:
• Click on any block to select it
• Drag blocks to move them on the timeline
• Use the Edit panel below to adjust times/settings
• Click "Delete" to remove selected block

KEYBOARD SHORTCUTS:
• TAB - Toggle Timeline Mode on/off
• Space - Pause/Resume video
• Left/Right Arrow - Seek video

RENDERING:
When you render with Timeline Mode ON:
• Effects are applied based on your timeline
• Multiple audio tracks are mixed together
• Original video effects are ignored

TIPS:
• Save your timeline frequently!
• Create video presets FIRST in the Control Panel
• Zoom slider helps with precision placement
• Click the time ruler to jump to that time
"""
        tk.Label(scrollable_frame, text=help_text, bg="#1a1a2e", fg="white",
                 font=("Consolas", 9), justify=tk.LEFT, anchor="nw",
                 padx=15, pady=10).pack(fill=tk.BOTH, expand=True)

        tk.Button(help_dialog, text="Got it!", command=help_dialog.destroy,
                  bg="#4CAF50", fg="white", font=("Arial", 10)).pack(pady=10)

    def add_effect_keyframe(self):
        """Add a new effect keyframe via dialog."""
        self.add_effect_keyframe_at_time(self.current_time)

    def add_effect_keyframe_at_time(self, start_time):
        """Add effect keyframe at specific time."""
        # Create dialog for preset selection
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Effect Keyframe")
        dialog.geometry("400x350")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Select Effect Preset:", font=("Arial", 10, "bold")).pack(pady=10)

        # Preset listbox
        preset_frame = tk.Frame(dialog)
        preset_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        scrollbar = tk.Scrollbar(preset_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        preset_listbox = tk.Listbox(preset_frame, yscrollcommand=scrollbar.set, height=10)
        preset_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=preset_listbox.yview)

        # Add "Current Settings" option
        preset_listbox.insert(tk.END, "[Current Effect Settings]")

        # Add saved presets
        for name in self.video_presets.keys():
            preset_listbox.insert(tk.END, name)

        preset_listbox.selection_set(0)

        # Time inputs
        time_frame = tk.Frame(dialog)
        time_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(time_frame, text="Start (sec):").pack(side=tk.LEFT)
        start_entry = tk.Entry(time_frame, width=8)
        start_entry.insert(0, f"{start_time:.2f}")
        start_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(time_frame, text="End (sec):").pack(side=tk.LEFT, padx=(10, 0))
        end_entry = tk.Entry(time_frame, width=8)
        end_entry.insert(0, f"{min(start_time + 5, self.total_duration):.2f}")
        end_entry.pack(side=tk.LEFT, padx=5)

        result = {'confirmed': False}

        def confirm():
            selection = preset_listbox.curselection()
            if not selection:
                return

            selected_name = preset_listbox.get(selection[0])
            try:
                start = float(start_entry.get())
                end = float(end_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid time values")
                return

            if end <= start:
                messagebox.showerror("Error", "End time must be after start time")
                return

            # Get preset data
            if selected_name == "[Current Effect Settings]":
                preset_data = {
                    'effect_states': dict(self.effect_states),
                    'effect_params': dict(self.effect_params)
                }
                preset_name = "Current Settings"
            else:
                preset_data = self.video_presets[selected_name]
                preset_name = selected_name

            self.effect_keyframes.append({
                'start_time': start,
                'end_time': end,
                'preset_name': preset_name,
                'preset_data': preset_data
            })

            # Sort keyframes by start time
            self.effect_keyframes.sort(key=lambda x: x['start_time'])

            result['confirmed'] = True
            dialog.destroy()

        tk.Button(dialog, text="Add Keyframe", command=confirm,
                  bg="#4CAF50", fg="white").pack(pady=10)

        dialog.wait_window()
        if result['confirmed']:
            self.draw_timeline()

    def add_audio_track(self):
        """Add a new audio track."""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.ogg *.flac *.aac *.m4a"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        # Get audio duration using ffprobe
        duration = 5.0  # Default
        try:
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                   "-of", "default=noprint_wrappers=1:nokey=1", file_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                duration = float(result.stdout.strip())
        except:
            pass

        # Convert to WAV for playback
        processed_path = os.path.join(self.temp_dir, f"audio_track_{len(self.audio_tracks)}.wav")
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", file_path,
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                "-ac", "2",
                processed_path
            ]
            subprocess.run(cmd, capture_output=True, text=True)
        except:
            processed_path = file_path

        self.audio_tracks.append({
            'start_time': self.current_time,
            'file_path': file_path,
            'processed_path': processed_path,
            'duration': duration,
            'volume': 1.0,
            'name': os.path.basename(file_path)[:20]
        })

        self.draw_timeline()

    def add_from_manual_entry(self):
        """Parse and add from manual text entry."""
        text = self.manual_entry.get().strip()
        if not text or text.startswith("Example:"):
            return

        try:
            # Parse format: start-end:preset_name
            time_part, preset_name = text.split(":")
            start_str, end_str = time_part.split("-")

            start_time = float(start_str)
            if end_str.lower() == "end":
                end_time = self.total_duration
            else:
                end_time = float(end_str)

            # Find preset
            if preset_name in self.video_presets:
                preset_data = self.video_presets[preset_name]
            else:
                preset_data = {
                    'effect_states': dict(self.effect_states),
                    'effect_params': dict(self.effect_params)
                }

            self.effect_keyframes.append({
                'start_time': start_time,
                'end_time': end_time,
                'preset_name': preset_name,
                'preset_data': preset_data
            })

            self.effect_keyframes.sort(key=lambda x: x['start_time'])
            self.manual_entry.delete(0, tk.END)
            self.draw_timeline()

        except Exception as e:
            messagebox.showerror("Parse Error", f"Could not parse: {text}\nFormat: start-end:preset_name\nError: {e}")

    def delete_selected(self):
        """Delete the selected keyframe or audio track."""
        if self.selected_keyframe is not None:
            del self.effect_keyframes[self.selected_keyframe]
            self.selected_keyframe = None
            self.clear_edit_panel()
            self.draw_timeline()
        elif self.selected_audio is not None:
            del self.audio_tracks[self.selected_audio]
            self.selected_audio = None
            self.clear_edit_panel()
            self.draw_timeline()

    def show_effect_edit_panel(self, index):
        """Show edit controls for effect keyframe."""
        try:
            self.clear_edit_panel()
            kf = self.effect_keyframes[index]

            # Info row
            info_frame = tk.Frame(self.edit_container)
            info_frame.pack(fill=tk.X, pady=2)

            tk.Label(info_frame, text=f"Effect: {kf['preset_name']}", font=("Arial", 10, "bold"),
                     fg="#4CAF50").pack(side=tk.LEFT)
            tk.Label(info_frame, text=f"  ({kf['end_time'] - kf['start_time']:.1f}s duration)",
                     font=("Arial", 9), fg="#888").pack(side=tk.LEFT)

            # Time edit row
            time_frame = tk.Frame(self.edit_container)
            time_frame.pack(fill=tk.X, pady=2)

            tk.Label(time_frame, text="Start (sec):").pack(side=tk.LEFT)
            start_var = tk.DoubleVar(value=kf['start_time'])
            start_entry = tk.Entry(time_frame, textvariable=start_var, width=8)
            start_entry.pack(side=tk.LEFT, padx=5)

            tk.Label(time_frame, text="End (sec):").pack(side=tk.LEFT, padx=(10, 0))
            end_var = tk.DoubleVar(value=kf['end_time'])
            end_entry = tk.Entry(time_frame, textvariable=end_var, width=8)
            end_entry.pack(side=tk.LEFT, padx=5)

            def apply_changes():
                try:
                    kf['start_time'] = max(0, start_var.get())
                    kf['end_time'] = min(end_var.get(), self.total_duration)
                    if kf['end_time'] <= kf['start_time']:
                        kf['end_time'] = kf['start_time'] + 0.5
                    self.draw_timeline()
                except Exception:
                    pass

            tk.Button(time_frame, text="Apply", command=apply_changes,
                      bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=10)
        except tk.TclError:
            pass

    def show_audio_edit_panel(self, index):
        """Show edit controls for audio track."""
        try:
            self.clear_edit_panel()
            track = self.audio_tracks[index]

            # Info row
            info_frame = tk.Frame(self.edit_container)
            info_frame.pack(fill=tk.X, pady=2)

            tk.Label(info_frame, text=f"Audio: {track['name']}", font=("Arial", 10, "bold"),
                     fg="#2196F3").pack(side=tk.LEFT)
            tk.Label(info_frame, text=f"  ({track.get('duration', 0):.1f}s)",
                     font=("Arial", 9), fg="#888").pack(side=tk.LEFT)

            # Controls row
            controls_frame = tk.Frame(self.edit_container)
            controls_frame.pack(fill=tk.X, pady=2)

            tk.Label(controls_frame, text="Start Time (sec):").pack(side=tk.LEFT)
            start_var = tk.DoubleVar(value=track['start_time'])
            start_entry = tk.Entry(controls_frame, textvariable=start_var, width=8)
            start_entry.pack(side=tk.LEFT, padx=5)

            tk.Label(controls_frame, text="Volume:").pack(side=tk.LEFT, padx=(10, 0))
            vol_var = tk.DoubleVar(value=track.get('volume', 1.0))
            vol_scale = tk.Scale(controls_frame, from_=0, to=2, resolution=0.1,
                                 orient=tk.HORIZONTAL, variable=vol_var, length=100)
            vol_scale.pack(side=tk.LEFT, padx=5)

            def apply_changes():
                try:
                    track['start_time'] = max(0, start_var.get())
                    track['volume'] = max(0, min(2.0, vol_var.get()))
                    self.draw_timeline()
                except Exception:
                    pass

            tk.Button(controls_frame, text="Apply", command=apply_changes,
                      bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=10)

            # Preview button
            tk.Button(controls_frame, text="▶ Play", command=lambda: self.play_audio_track(index),
                      bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=5)
        except tk.TclError:
            pass

    def clear_edit_panel(self):
        """Clear the edit panel and show helpful instructions."""
        try:
            for widget in self.edit_container.winfo_children():
                widget.destroy()

            # Show helpful info when nothing is selected
            info_frame = tk.Frame(self.edit_container)
            info_frame.pack(pady=10)

            self.edit_info_label = tk.Label(info_frame,
                text="Click on a green effect block or blue audio block to select and edit it",
                font=("Arial", 10), fg="#888")
            self.edit_info_label.pack(pady=5)

            # Quick stats
            stats_text = f"Timeline: {len(self.effect_keyframes)} effects, {len(self.audio_tracks)} audio tracks"
            tk.Label(info_frame, text=stats_text, font=("Arial", 9), fg="#555").pack(pady=2)
        except tk.TclError:
            pass

    def play_audio_track(self, index):
        """Play a single audio track for preview."""
        if not PYGAME_AVAILABLE:
            return

        track = self.audio_tracks[index]
        audio_path = track.get('processed_path') or track.get('file_path')

        if audio_path and os.path.exists(audio_path):
            try:
                sound = pygame.mixer.Sound(audio_path)
                sound.set_volume(track.get('volume', 1.0))
                sound.play()
            except Exception as e:
                print(f"Error playing audio: {e}")

    def preview_timeline(self):
        """Start timeline preview playback."""
        self.is_playing = True
        self.current_time = 0
        self._preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self._preview_thread.start()

    def stop_preview(self):
        """Stop timeline preview."""
        self.is_playing = False
        if PYGAME_AVAILABLE:
            pygame.mixer.stop()

    def _preview_loop(self):
        """Preview playback loop."""
        start_real_time = time.time()
        triggered_audio = set()

        while self.is_playing and self.current_time < self.total_duration:
            self.current_time = time.time() - start_real_time

            # Trigger audio tracks
            for i, track in enumerate(self.audio_tracks):
                if i not in triggered_audio and self.current_time >= track['start_time']:
                    self.play_audio_track(i)
                    triggered_audio.add(i)

            # Update UI
            try:
                self.root.after(0, self.update_time_display)
                self.root.after(0, self.draw_timeline)
            except:
                break

            time.sleep(0.05)

        self.is_playing = False

    def save_timeline(self):
        """Save timeline to JSON file."""
        file_path = filedialog.asksaveasfilename(
            title="Save Timeline",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            initialfile="timeline_project"
        )

        if not file_path:
            return

        data = {
            'video_fps': self.video_fps,
            'total_frames': self.total_frames,
            'effect_keyframes': self.effect_keyframes,
            'audio_tracks': [
                {
                    'start_time': t['start_time'],
                    'file_path': t['file_path'],
                    'duration': t['duration'],
                    'volume': t['volume'],
                    'name': t['name']
                }
                for t in self.audio_tracks
            ]
        }

        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Timeline saved to: {file_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save timeline: {e}")

    def load_timeline(self):
        """Load timeline from JSON file."""
        file_path = filedialog.askopenfilename(
            title="Load Timeline",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )

        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            self.effect_keyframes = data.get('effect_keyframes', [])

            # Reload audio tracks
            self.audio_tracks = []
            for track_data in data.get('audio_tracks', []):
                # Re-process audio file if it exists
                file_path = track_data['file_path']
                if os.path.exists(file_path):
                    processed_path = os.path.join(self.temp_dir, f"audio_track_{len(self.audio_tracks)}.wav")
                    try:
                        cmd = [
                            "ffmpeg", "-y", "-i", file_path,
                            "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                            processed_path
                        ]
                        subprocess.run(cmd, capture_output=True, text=True)
                    except:
                        processed_path = file_path

                    track_data['processed_path'] = processed_path
                    self.audio_tracks.append(track_data)

            self.draw_timeline()
            print(f"Timeline loaded from: {file_path}")

        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load timeline: {e}")

    def get_active_effects_at_time(self, time_sec):
        """Get the combined effect states/params for a given time."""
        active_states = {}
        active_params = {}

        for kf in self.effect_keyframes:
            if kf['start_time'] <= time_sec < kf['end_time']:
                preset_data = kf.get('preset_data', {})
                if 'effect_states' in preset_data:
                    active_states.update(preset_data['effect_states'])
                if 'effect_params' in preset_data:
                    active_params.update(preset_data['effect_params'])

        return active_states, active_params

    def get_audio_events_for_render(self):
        """Get audio events formatted for render mixing."""
        return [
            {
                'start_time': t['start_time'],
                'file_path': t.get('processed_path') or t['file_path'],
                'duration': t['duration'],
                'volume': t.get('volume', 1.0)
            }
            for t in self.audio_tracks
        ]

    def update(self):
        """Update the panel (call from main loop)."""
        if self.root and self.running:
            try:
                
                    self.root.update_idletasks()
                    self.root.update()
            except (tk.TclError, RuntimeError):
                self.running = False
            except Exception:
                pass  # Ignore other errors during update

    def close(self):
        """Close the panel."""
        self.running = False
        self.stop_preview()
        if self.root:
            try:
                
                    self.root.destroy()
            except:
                pass
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass

