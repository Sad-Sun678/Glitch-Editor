"""Audio Effects Panel - GUI for audio effects and preview."""
import tkinter as tk
from tkinter import ttk, simpledialog, filedialog
import os
import json
import shutil
import subprocess
import tempfile
import threading
import time

from panel_utils import AUDIO_PRESETS_FILE, PYGAME_AVAILABLE

if PYGAME_AVAILABLE:
    import pygame

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
        print(f"Path exists calling audio extractor {path}")
        if path and os.path.exists(path):
            print("extracting audio")
            self.extract_audio()
            print("extracted")
            print(f"{self.extract_audio() }")

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
        self.root.minsize(350, 400)  # Set minimum size to prevent resize crashes

        # Resize handling with debounce
        self._resize_after_id = None
        self._canvas = None
        self._canvas_window = None

        # Create main container with scrollbar
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_frame)
        self._canvas = canvas
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

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

        self.preview_btn = tk.Button(playback_frame, text="Preview Effects", command=self.preview_effects,
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

        # Export section
        export_frame = tk.Frame(source_frame)
        export_frame.pack(fill=tk.X, pady=5)

        self.export_audio_btn = tk.Button(export_frame, text="💾 Export Audio",
                                           command=self.export_audio,
                                           bg="#00BCD4", fg="white", width=15)
        self.export_audio_btn.pack(side=tk.LEFT, padx=5)

        tk.Label(export_frame, text="(Export with effects applied)", font=("Arial", 8), fg="gray").pack(side=tk.LEFT)

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

    def _update_scroll_region(self):
        """Update scroll region safely."""
        try:
            if self._canvas:
                self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        except tk.TclError:
            pass
        finally:
            self._resize_after_id = None

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
            except (tk.TclError, RuntimeError):
                self.running = False
            except Exception:
                pass  # Ignore other errors during update

    def close(self):
        """Close the panel and cleanup."""
        self.running = False
        self.stop_audio()
        if self.root:
            try:
                
                    self.root.destroy()
            except:
                pass
        # Cleanup temp files
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass

    def export_audio(self):
        """Export the current processed audio to a file."""
        # Determine which audio to export
        if self.processed_audio_path and os.path.exists(self.processed_audio_path):
            source_audio = self.processed_audio_path
            default_name = "exported_audio_processed"
        elif self.preview_audio_path and os.path.exists(self.preview_audio_path):
            source_audio = self.preview_audio_path
            default_name = "exported_audio_preview"
        elif self.original_audio_path and os.path.exists(self.original_audio_path):
            source_audio = self.original_audio_path
            default_name = "exported_audio"
        else:
            self.status_label.config(text="No audio to export")
            return

        # Ask for save location
        output_path = filedialog.asksaveasfilename(
            title="Export Audio",
            defaultextension=".wav",
            filetypes=[
                ("WAV Audio", "*.wav"),
                ("MP3 Audio", "*.mp3"),
                ("All Files", "*.*")
            ],
            initialfile=default_name
        )

        if not output_path:
            return

        self.status_label.config(text="Exporting audio...")
        self.root.update()

        try:
            if output_path.lower().endswith('.mp3'):
                # Convert to MP3 using ffmpeg
                cmd = [
                    "ffmpeg", "-y",
                    "-i", source_audio,
                    "-codec:a", "libmp3lame",
                    "-qscale:a", "2",
                    output_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    self.status_label.config(text=f"Exported to {os.path.basename(output_path)}")
                    print(f"Audio exported to: {output_path}")
                else:
                    self.status_label.config(text="Export failed")
                    print(f"FFmpeg error: {result.stderr}")
            else:
                # Just copy WAV file
                shutil.copy(source_audio, output_path)
                self.status_label.config(text=f"Exported to {os.path.basename(output_path)}")
                print(f"Audio exported to: {output_path}")

        except Exception as e:
            self.status_label.config(text=f"Export error: {e}")
            print(f"Export error: {e}")

