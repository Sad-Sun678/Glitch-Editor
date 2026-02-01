"""Shared utilities for panel modules."""
import os
import threading

# Preset file paths
PRESETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
VIDEO_PRESETS_FILE = os.path.join(PRESETS_DIR, "video_presets.json")
AUDIO_PRESETS_FILE = os.path.join(PRESETS_DIR, "audio_presets.json")
TIMELINE_FILE = os.path.join(PRESETS_DIR, "timeline_project.json")

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
