# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


PROJECT_ROOT = Path(r"G:\731")
ENTRY_SCRIPT = PROJECT_ROOT / "drivesense" / "frontend" / "gui.py"


def add_data_dir(source: Path, target: str) -> list[tuple[str, str]]:
    return [(str(source), target)] if source.exists() else []


datas: list[tuple[str, str]] = []
datas += add_data_dir(PROJECT_ROOT / "weights", "weights")
datas += add_data_dir(PROJECT_ROOT / "runs_timm", "runs_timm")
datas += add_data_dir(PROJECT_ROOT / "benchmark_results", "benchmark_results")
datas += add_data_dir(PROJECT_ROOT / ".env", ".")

binaries: list[tuple[str, str]] = []
hiddenimports: list[str] = []

for package_name in [
    "ultralytics",
    "torch",
    "torchvision",
    "torchaudio",
    "faster_whisper",
    "ctranslate2",
    "pyttsx3",
    "openai",
    "dotenv",
    "sounddevice",
]:
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package_name)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports


a = Analysis(
    [str(ENTRY_SCRIPT)],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DriveSense",
)
