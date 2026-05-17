# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


PROJECT_ROOT = Path(r"G:\731")
ENTRY_SCRIPT = PROJECT_ROOT / "drivesense" / "frontend" / "gui.py"


def add_data_path(source: Path, target: str) -> list[tuple[str, str]]:
    return [(str(source), target)] if source.exists() else []


datas: list[tuple[str, str]] = []
datas += add_data_path(PROJECT_ROOT / "weights", "weights")
datas += add_data_path(PROJECT_ROOT / "runs_timm", "runs_timm")
datas += add_data_path(PROJECT_ROOT / "benchmark_results", "benchmark_results")
datas += add_data_path(PROJECT_ROOT / ".env", ".")

# Runtime packages that actually need bundled non-Python assets.
datas += collect_data_files("ultralytics")
datas += collect_data_files("timm", includes=["**/*.json", "**/*.txt"])
datas += collect_data_files("faster_whisper")

# CTranslate2 ships native libraries that are required at runtime.
binaries: list[tuple[str, str]] = []
binaries += collect_dynamic_libs("ctranslate2")

# Keep hidden imports narrow. Torch / torchvision / PyQt5 already have
# first-party PyInstaller hooks; forcing collect_all() on them makes DLL
# discovery explode and is the main reason packaging stalls.
hiddenimports: list[str] = []
hiddenimports += collect_submodules("timm")
hiddenimports += collect_submodules("faster_whisper")
hiddenimports += collect_submodules("pyttsx3.drivers")
hiddenimports += [
    "sounddevice",
    "dotenv",
    "openai",
]

excludes = [
    "torchaudio",
    "tkinter",
    "matplotlib.tests",
    "numpy.tests",
]


a = Analysis(
    [str(ENTRY_SCRIPT)],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DriveSense",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="DriveSense",
)
