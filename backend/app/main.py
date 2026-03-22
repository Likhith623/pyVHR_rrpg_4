from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .model_service import InferenceService, ModelRepository
from .schemas import HealthResponse, ModelsResponse, PredictResponse
from .settings import ALLOWED_VIDEO_EXTENSIONS, MAX_UPLOAD_MB

app = FastAPI(
    title="NeuroPulse API",
    description="Upload a video and run deepfake inference using a named ensemble model artifact.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

repo = ModelRepository()
inference = InferenceService(repo)
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/api/v1/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    return ModelsResponse(available_models=repo.list_available_models())


@app.post("/api/v1/predict", response_model=PredictResponse)
async def predict(
    model_name: str | None = Form(default=None, description="Optional model filename stem"),
    video: UploadFile = File(...),
) -> PredictResponse:
    if not video.filename:
        raise HTTPException(status_code=400, detail="Video filename is required")

    ext = Path(video.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}",
        )

    data = await video.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Video exceeds max upload size of {MAX_UPLOAD_MB}MB",
        )

    try:
        selected_model = repo.resolve_model_name(model_name)
        with tempfile.TemporaryDirectory(prefix="neuropulse_") as tmpdir:
            tmp_path = Path(tmpdir) / video.filename
            with tmp_path.open("wb") as f:
                f.write(data)

            result = inference.predict(tmp_path, selected_model)

        return PredictResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


@app.get("/")
def root() -> dict:
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "service": "NeuroPulse API",
        "docs": "/docs",
        "predict": "POST /api/v1/predict",
        "models": "GET /api/v1/models",
    }


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/styles.css")
def frontend_styles() -> FileResponse:
    css_path = FRONTEND_DIR / "styles.css"
    if css_path.exists():
        return FileResponse(css_path)
    raise HTTPException(status_code=404, detail="styles.css not found")


@app.get("/app.js")
def frontend_script() -> FileResponse:
    js_path = FRONTEND_DIR / "app.js"
    if js_path.exists():
        return FileResponse(js_path)
    raise HTTPException(status_code=404, detail="app.js not found")
