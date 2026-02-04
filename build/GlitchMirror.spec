# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Glitch Mirror
To build: pyinstaller GlitchMirror.spec
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Get cv2 data path for cascade files
try:
    import cv2
    cv2_data_path = cv2.data.haarcascades
except:
    cv2_data_path = None

block_cipher = None

# Collect all necessary data files
datas = []

# Add OpenCV cascade files
if cv2_data_path and os.path.exists(cv2_data_path):
    datas.append((cv2_data_path, 'cv2/data'))

# Add presets folder if it exists
if os.path.exists('presets'):
    datas.append(('presets', 'presets'))

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'cv2',
    'numpy',
    'numpy.core._methods',
    'numpy.lib.format',
    'tkinter',
    'tkinter.ttk',
    'tkinter.filedialog',
    'tkinter.simpledialog',
    'tkinter.messagebox',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'PIL._tkinter_finder',
    'pygame',
    'pygame.mixer',
    'pygame.base',
    'pygame.constants',
    'pygame.mixer_music',
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GlitchMirror',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Set to True for debugging (shows console window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GlitchMirror',
)
