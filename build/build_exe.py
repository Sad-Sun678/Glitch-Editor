"""
Build script for creating Glitch Mirror executable.
Run this script to package the application as a standalone .exe

Requirements:
    pip install pyinstaller

Usage:
    From project root:
        python build/build_exe.py
"""

import os
import sys
import subprocess
import shutil

# Get project root (parent of build folder)
BUILD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BUILD_DIR)


def get_cv2_data_path():
    """Get the path to OpenCV's data files."""
    try:
        import cv2
        return cv2.data.haarcascades
    except:
        return None


def build():
    print(f"Project directory: {PROJECT_DIR}")
    print(f"Build directory: {BUILD_DIR}")

    # Check if PyInstaller is installed
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Get OpenCV data path for cascade files
    cv2_data_path = get_cv2_data_path()
    if cv2_data_path:
        print(f"OpenCV data path: {cv2_data_path}")
    else:
        print("WARNING: Could not find OpenCV data path")

    # Verify main.py exists
    main_script = os.path.join(PROJECT_DIR, 'main.py')
    if not os.path.exists(main_script):
        print(f"ERROR: main.py not found at {main_script}")
        return False

    # Hidden imports that PyInstaller might miss
    hidden_imports = [
        'cv2',
        'numpy',
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.simpledialog',
        'tkinter.messagebox',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'pygame',
        'pygame.mixer',
    ]

    # Build the PyInstaller command
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--name=GlitchMirror',
        '--onedir',
        '--windowed',
        '--noconfirm',
        f'--distpath={os.path.join(PROJECT_DIR, "dist")}',
        f'--workpath={os.path.join(BUILD_DIR, "temp")}',
        f'--specpath={BUILD_DIR}',
    ]

    # Add hidden imports
    for imp in hidden_imports:
        cmd.extend(['--hidden-import', imp])

    # Add source directories
    src_dir = os.path.join(PROJECT_DIR, 'src')
    cmd.extend(['--add-data', f'{src_dir};src'])

    # Add OpenCV cascade data files
    if cv2_data_path and os.path.exists(cv2_data_path):
        cmd.extend(['--add-data', f'{cv2_data_path};cv2/data'])

    # Add presets directory if it exists
    presets_dir = os.path.join(PROJECT_DIR, 'presets')
    if os.path.exists(presets_dir):
        cmd.extend(['--add-data', f'{presets_dir};presets'])

    # Add the main script
    cmd.append(main_script)

    print("\nBuilding executable...")
    print(f"Command: {' '.join(cmd)}\n")

    # Run PyInstaller
    try:
        subprocess.check_call(cmd, cwd=PROJECT_DIR)
        print("\n" + "="*50)
        print("BUILD SUCCESSFUL!")
        print("="*50)
        print(f"\nExecutable created in: {os.path.join(PROJECT_DIR, 'dist', 'GlitchMirror')}")
        print("\nTo run on another machine:")
        print("1. Copy the entire 'GlitchMirror' folder from 'dist/'")
        print("2. Make sure ffmpeg is installed on the target machine")
        print("3. Run GlitchMirror.exe")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nBUILD FAILED: {e}")
        return False


if __name__ == '__main__':
    success = build()
    sys.exit(0 if success else 1)
