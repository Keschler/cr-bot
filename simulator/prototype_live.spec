"""PyInstaller one-file build for the Linux prototype live controller.

Build with the repository's inference environment, for example:

    ../outputs/venv/bin/python -m PyInstaller --clean --noconfirm \
        prototype_live.spec

The resulting native executable is written to ``simulator/dist/``.  This
spec intentionally targets Linux x86-64; PyInstaller builds are native and
must be produced separately for other operating systems/architectures.
"""

from pathlib import Path
import os
import shutil
import sys

from PyInstaller.building.build_main import Analysis, EXE, PYZ
from PyInstaller.utils.hooks import collect_submodules


try:
    _spec_path = Path(SPEC).resolve()  # type: ignore[name-defined]
except NameError:  # pragma: no cover - useful when linting this file directly
    _spec_path = Path(__file__).resolve()

SIMULATOR_ROOT = _spec_path.parent
CHECKOUT_ROOT = SIMULATOR_ROOT.parent
SOURCE_ROOT = CHECKOUT_ROOT / "src"
ASSETS_ROOT = CHECKOUT_ROOT / "assets"
VENDOR_ROOT = CHECKOUT_ROOT / "vendor"
KATACR_SOURCE_ROOT = VENDOR_ROOT / "external" / "KataCR"
DEFAULT_CHECKPOINT = (
    SIMULATOR_ROOT
    / "outputs"
    / "simulator"
    / "training"
    / "prototype-fast-current"
    / "prototype.pt"
)
# Training outputs are intentionally ignored by Git.  A build can select a
# compatible checkpoint from another location without staging it in the repo.
CHECKPOINT = Path(
    os.environ.get("PROTOTYPE_LIVE_CHECKPOINT", str(DEFAULT_CHECKPOINT))
).expanduser()

for _import_root in (CHECKOUT_ROOT, SOURCE_ROOT, KATACR_SOURCE_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))


def _require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"prototype-live build input is missing: {path}")
    return path


def _helper_binary(name: str) -> tuple[str, str]:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(
            f"cannot build prototype-live: required helper {name!r} is not on PATH"
        )
    return str(Path(path).resolve()), "."


datas = [
    (str(_require(CHECKOUT_ROOT / "assets" / "models")), "assets/models"),
    (str(_require(CHECKOUT_ROOT / "assets" / "templates")), "assets/templates"),
    (
        str(_require(ASSETS_ROOT / "templates" / "cr-api-assets" / "cards-150")),
        "assets/templates/cr-api-assets/cards-150",
    ),
    (
        str(_require(ASSETS_ROOT / "templates" / "cr-api-assets" / "cards-gold")),
        "assets/templates/cr-api-assets/cards-gold",
    ),
    (
        str(_require(KATACR_SOURCE_ROOT / "katacr" / "utils" / "fonts")),
        "katacr/utils/fonts",
    ),
    (
        str(
            _require(
                VENDOR_ROOT
                / "external"
                / "Clash-Royale-Detection-Dataset"
                / "version_info"
                / "segment_v0.1.csv"
            )
        ),
        "vendor/external/Clash-Royale-Detection-Dataset/version_info",
    ),
    (str(_require(SIMULATOR_ROOT / "rulesets")), "simulator/rulesets"),
    (str(_require(CHECKPOINT)), "simulator/outputs/simulator/training/prototype-fast-current"),
]

# AppDetector resolves this tracker file through KATACR_ROOT.  Keep the
# config in that path, and also beside the packaged katacr modules for code
# that resolves resources relative to __file__.
for _yaml in sorted((KATACR_SOURCE_ROOT / "katacr" / "yolov8").glob("*.yaml")):
    datas.append((str(_yaml), "vendor/external/KataCR/katacr/yolov8"))
    datas.append((str(_yaml), "katacr/yolov8"))


binaries = [_helper_binary("adb"), _helper_binary("ffmpeg")]

# These modules are loaded by name by the visual extractor or by the
# checkpoint/runtime.  Keep the collection focused on inference-related
# KataCR packages; collecting the entire repository would pull in training,
# policy, and dataset tooling that the live loop never imports.
hiddenimports = [
    "scripts.run_seed_inference",
    "simulator.physical_lab.prototype_controller",
    "simulator.rl.prototype",
    "simulator.rl._compat",
    "simulator.rl.provenance",
    "katacr.constants.label_list",
    "katacr.build_dataset.utils.split_part",
    "katacr.build_dataset.utils.datapath_manager",
    "katacr.build_dataset.constant",
    "katacr.utils.related_pkgs.utility",
    "katacr.yolov8.train",
    "katacr.yolov8.custom_model",
    "katacr.yolov8.custom_predict",
    "katacr.yolov8.custom_utils",
    "katacr.yolov8.custom_result",
    "katacr.yolov8.custom_trackers",
]
hiddenimports += collect_submodules("torch.distributed")
hiddenimports = sorted(set(hiddenimports))

a = Analysis(
    [str(SIMULATOR_ROOT / "run_prototype_live.py")],
    pathex=[
        str(CHECKOUT_ROOT),
        str(SIMULATOR_ROOT),
        str(SOURCE_ROOT),
        str(KATACR_SOURCE_ROOT),
    ],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(SIMULATOR_ROOT / "packaging" / "prototype_live_runtime.py")],
    excludes=[
        "IPython",
        "jupyter",
        "notebook",
        "pandas",
        "pytest",
        "sklearn",
        "tensorboard",
        "tensorboardX",
        "jax",
        "jaxlib",
        "triton",
        "yt_dlp",
        "onnxruntime",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="prototype-live-linux-x86_64",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)
