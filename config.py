"""
Thin loader over config.yaml.

config.yaml is the single source of truth for all tunable values. This module
parses it once at import time and exposes attribute-style access plus a few
derived constants used everywhere (Flask, paths, thresholds).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def _load_yaml() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


_CFG: dict[str, Any] = _load_yaml()


def get(*keys: str, default: Any = None) -> Any:
    """Dotted-key access: get('detection', 'confidence_threshold')."""
    node: Any = _CFG
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def _abs(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else BASE_DIR / p


# Paths
MODELS_DIR = _abs(get("paths", "models_dir"))
UPLOADS_DIR = _abs(get("paths", "uploads_dir"))
RESULTS_DIR = _abs(get("paths", "results_dir"))
ANNOTATED_DIR = _abs(get("paths", "annotated_dir"))
CROPS_DIR = _abs(get("paths", "crops_dir"))
LOGS_DIR = _abs(get("paths", "logs_dir"))
CSV_LOG_PATH = _abs(get("paths", "verification_log_csv"))

for _d in (MODELS_DIR, UPLOADS_DIR, RESULTS_DIR, ANNOTATED_DIR, CROPS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Database
DB_PATH = _abs(get("database", "path"))

# Flask
SECRET_KEY = os.getenv(get("flask", "secret_key_env"), "dev-secret-key")
FLASK_DEBUG = os.getenv(get("flask", "debug_env"), "True").lower() == "true"
FLASK_PORT = int(os.getenv(get("flask", "port_env"), str(get("flask", "default_port"))))
MAX_CONTENT_LENGTH = int(get("flask", "max_content_length_mb")) * 1024 * 1024
ADMIN_PASSWORD = os.getenv(get("flask", "admin_password_env"), get("flask", "default_admin_password"))

# Input
ALLOWED_IMAGE_EXTS = set(get("input", "allowed_image_extensions"))
ALLOWED_VIDEO_EXTS = set(get("input", "allowed_video_extensions"))
MIN_IMG_W = int(get("input", "min_image_width"))
MIN_IMG_H = int(get("input", "min_image_height"))
MAX_VIDEO_SEC = int(get("input", "max_video_duration_sec"))
KEYFRAME_FPS = float(get("input", "keyframe_fps"))

# Preprocessing
INPUT_SIZE = tuple(get("preprocessing", "input_size"))
CLAHE_CLIP = float(get("preprocessing", "clahe_clip_limit"))
CLAHE_TILE = tuple(get("preprocessing", "clahe_tile_grid"))
BLUR_LAP_THRESH = float(get("preprocessing", "blur_laplacian_threshold"))
WIENER_NOISE_VAR = float(get("preprocessing", "wiener_noise_var"))
ENABLE_CLAHE = bool(get("preprocessing", "enable_clahe"))
ENABLE_WIENER = bool(get("preprocessing", "enable_wiener"))

# Detection
DEFAULT_MODEL_VARIANT = get("detection", "default_model_variant")
MODEL_VARIANTS = get("detection", "model_variants")
DET_CONF = float(get("detection", "confidence_threshold"))
NMS_IOU = float(get("detection", "nms_iou_threshold"))
ASPECT_MIN = float(get("detection", "plate_aspect_ratio_min"))
ASPECT_MAX = float(get("detection", "plate_aspect_ratio_max"))
MIN_BBOX_AREA = float(get("detection", "min_bbox_area_px"))

# Cropping / OCR prep
CROP_EXPAND = float(get("cropping", "crop_expand_ratio"))
DESKEW_MAX_ANGLE = float(get("cropping", "deskew_max_angle_deg"))
MEDIAN_K = int(get("cropping", "median_kernel_size"))
UNSHARP_RADIUS = int(get("cropping", "unsharp_radius"))
UNSHARP_PCT = int(get("cropping", "unsharp_percent"))
UNSHARP_THRESH = int(get("cropping", "unsharp_threshold"))

# OCR
EASYOCR_LANGS = list(get("ocr", "easyocr_languages"))
EASYOCR_GPU = bool(get("ocr", "easyocr_gpu"))
TESSERACT_CONFIG = get("ocr", "tesseract_config")
PLATE_REGEX_CAR = get("ocr", "plate_regex_car")
PLATE_REGEX_MOTO = get("ocr", "plate_regex_motorcycle")
MIN_TEXT_HEIGHT_RATIO = float(get("ocr", "min_text_height_ratio", default=0.6))


def _normalize_subs(raw: dict) -> dict[str, list[str]]:
    """Accept either str values (legacy) or list[str] values from YAML."""
    out: dict[str, list[str]] = {}
    for k, v in (raw or {}).items():
        out[str(k)] = [str(x) for x in v] if isinstance(v, list) else [str(v)]
    return out


CHAR_SUBS: dict[str, list[str]] = _normalize_subs(get("ocr", "char_substitutions"))

# Verification
OCR_CONF_THRESH = float(get("verification", "ocr_confidence_threshold"))
OCR_CONF_THRESH_EXACT = float(
    get("verification", "ocr_confidence_threshold_exact", default=OCR_CONF_THRESH)
)
FUZZY_MAX_DIST = int(get("verification", "fuzzy_match_max_distance"))
FUZZY_FLAG = get("verification", "fuzzy_match_flag")
LOW_CONF_EXACT_FLAG = get(
    "verification",
    "low_conf_exact_flag",
    default="LOW_OCR_CONFIDENCE_EXACT - relaxed gate (matched in DB)",
)

# Region codes
REGION_CODES: set[str] = set(get("region_codes", default=[]))

# UI
BBOX_COLORS: dict[str, tuple[int, int, int]] = {
    k: tuple(v) for k, v in get("ui", "bbox_colors").items()
}
BBOX_THICKNESS = int(get("ui", "bbox_thickness"))
FONT_SCALE = float(get("ui", "font_scale"))
FONT_THICKNESS = int(get("ui", "font_thickness"))

# Timing
TARGET_MS = int(get("timing", "target_processing_time_ms"))
