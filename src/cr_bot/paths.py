from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

APP_ROOT = REPO_ROOT
ASSETS_DIR = APP_ROOT / "assets"
MODELS_DIR = ASSETS_DIR / "models"
TEMPLATES_DIR = ASSETS_DIR / "templates"
PICTURES_DIR = ASSETS_DIR / "pictures"
CACHE_DIR = APP_ROOT / "outputs" / "cache"

KATACR_ROOT = APP_ROOT / "vendor/external/KataCR"
KATACR_DATASET_ROOT = APP_ROOT / "vendor/external/Clash-Royale-Detection-Dataset"
