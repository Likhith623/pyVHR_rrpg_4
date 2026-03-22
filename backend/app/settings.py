from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR.parent

MODEL_DIRS = [
    BASE_DIR / "models",
    PROJECT_ROOT / "models",
    PROJECT_ROOT,
]

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_MB = 500
