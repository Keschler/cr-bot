# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

import torch
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    is_module_or_submodule,
)


if torch.version.cuda is not None:
    raise SystemExit(
        "capture.spec must be built from the CPU virtualenv. "
        "Run: venv-cpu/bin/pyinstaller --clean capture.spec"
    )


EXCLUDED_MODULES = (
    "cupy",
    "jax",
    "jaxlib",
    "tensorflow",
    "tensorboard",
    "torch.utils.tensorboard",
    "triton",
)

EXCLUDED_PATH_PARTS = (
    "/__pycache__/",
    "/cuda/",
    "/cudnn/",
    "/jax/",
    "/jaxlib/",
    "/nvidia/",
    "/tests/",
    "/torch/include/",
    "/torch/test/",
    "/triton/",
)

EXCLUDED_NAME_PARTS = (
    "cublas",
    "cudart",
    "cudnn",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "libcuda",
    "libnccl",
    "libnv",
    "nvjitlink",
    "nvrtc",
)


def keep_module(name):
    return not any(is_module_or_submodule(name, excluded) for excluded in EXCLUDED_MODULES)


def keep_toc_entry(entry):
    dest, src, _typecode = entry
    text = f"/{dest.replace(chr(92), '/')}/{str(src).replace(chr(92), '/')}".lower()
    return (
        not any(part in text for part in EXCLUDED_PATH_PARTS)
        and not any(part in text for part in EXCLUDED_NAME_PARTS)
    )


datas = [
    ("assets/models", "assets/models"),
    ("assets/templates", "assets/templates"),
    (
        "vendor/external/KataCR/katacr/yolov8",
        "vendor/external/KataCR/katacr/yolov8",
    ),
    (
        "vendor/external/KataCR/katacr/utils/fonts",
        "vendor/external/KataCR/katacr/utils/fonts",
    ),
]
datas += collect_data_files(
    "ultralytics",
    includes=["cfg/**/*.yaml", "assets/**/*"],
    excludes=["**/__pycache__/**", "**/tests/**"],
)

hiddenimports = [
    "scripts.run_seed_inference",
]
for package in (
    "katacr.constants",
    "katacr.utils",
    "katacr.yolov8",
    "ultralytics",
):
    hiddenimports += collect_submodules(package, filter=keep_module)


a = Analysis(
    ["capture.py"],
    pathex=[".", "vendor/external/KataCR"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["packaging/pyi_runtime_cpu.py"],
    excludes=list(EXCLUDED_MODULES),
    noarchive=False,
    optimize=0,
)

a.binaries = [entry for entry in a.binaries if keep_toc_entry(entry)]
a.datas = [entry for entry in a.datas if keep_toc_entry(entry)]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="capture",
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
