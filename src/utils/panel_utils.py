"""Shared utilities for panel modules."""
import os
import sys
import threading


def get_base_path():
    """Get the base path for resources - works for both dev and PyInstaller exe."""
    if getattr(sys, 'frozen', False):
        # Running as compiled exe
        return sys._MEIPASS
    else:
        # Running as script
        return os.path.dirname(os.path.abspath(__file__))


def get_user_data_path():
    """Get path for user data (presets, etc.) that persists after packaging."""
    if getattr(sys, 'frozen', False):
        # When packaged, use a folder next to the exe for user data
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))


# Base path for bundled resources
BASE_PATH = get_base_path()

# User data path for presets (writable location)
USER_DATA_PATH = get_user_data_path()

# Preset file paths - use user data path so presets can be saved
PRESETS_DIR = os.path.join(USER_DATA_PATH, "presets")
VIDEO_PRESETS_FILE = os.path.join(PRESETS_DIR, "video_presets.json")
AUDIO_PRESETS_FILE = os.path.join(PRESETS_DIR, "audio_presets.json")
TIMELINE_FILE = os.path.join(PRESETS_DIR, "timeline_project.json")

# Create presets directory if it doesn't exist
os.makedirs(PRESETS_DIR, exist_ok=True)

# Pygame availability flag - defer initialization until needed
_pygame_checked = False
_pygame_initialized = False
PYGAME_AVAILABLE = False


def _check_pygame_available():
    """Check if pygame can be imported without initializing it."""
    global _pygame_checked, PYGAME_AVAILABLE
    if _pygame_checked:
        return PYGAME_AVAILABLE
    _pygame_checked = True
    try:
        import importlib.util
        spec = importlib.util.find_spec("pygame")
        PYGAME_AVAILABLE = spec is not None
    except Exception:
        PYGAME_AVAILABLE = False
    if not PYGAME_AVAILABLE:
        print("pygame not installed - audio preview will be limited")
    return PYGAME_AVAILABLE


def init_pygame_mixer():
    """Initialize pygame mixer on demand. Returns True if successful."""
    global PYGAME_AVAILABLE, _pygame_initialized
    if _pygame_initialized:
        return PYGAME_AVAILABLE
    _pygame_initialized = True
    try:
        import pygame
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        PYGAME_AVAILABLE = True
        return True
    except ImportError:
        PYGAME_AVAILABLE = False
        print("pygame not installed - audio preview will be limited")
        return False
    except Exception as e:
        PYGAME_AVAILABLE = False
        print(f"pygame mixer initialization failed: {e}")
        return False


# Check availability at import time (but don't import pygame itself)
_check_pygame_available()
