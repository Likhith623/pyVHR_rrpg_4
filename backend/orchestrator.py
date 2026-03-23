# ═══════════════════════════════════════════════════════════════════════════
# orchestrator.py  —  NeuroPulse | Main Orchestrator API
#
# Run:  uvicorn orchestrator:app --host 0.0.0.0 --port 8000
# Env:  requirements_orchestrator.txt  (httpx, fastapi, uvicorn, joblib, numpy, scipy)
#
# Architecture:
#   1.  Client uploads video → POST /api/v1/predict
#   2.  Orchestrator saves video to a temp file once.
#   3.  Concurrently fans out to 3 sub-services via HTTP multipart:
#         • http://localhost:8001/predict  (rPPG)
#         • http://localhost:8002/predict  (EfficientNet-B4)
#         • http://localhost:8003/predict  (Swin-Tiny)
#   4.  Combines the 3 probabilities using Optuna-tuned ensemble weights +
#       temperature scaling + Youden threshold from ensemble_model.pkl.
#   5.  Returns rich JSON response to the client.
#
#   If any single sub-service fails / times out, that stream is set to 0.5
#   (neutral) and the ensemble continues with the remaining streams.
#   If ALL sub-services fail, a 503 error is returned.
# ═══════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import warnings
from typing import Any

import httpx
import joblib
import numpy as np
from scipy.special import expit, logit

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")

app = FastAPI(title="NeuroPulse Deepfake Detection API", version="4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

EPS      = 1e-7
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Sub-service URLs (override with environment variables if needed) ────────
RPPG_URL   = os.environ.get("RPPG_URL",   "http://localhost:8001/predict")
EFF_URL    = os.environ.get("EFF_URL",    "http://localhost:8002/predict")
SWIN_URL   = os.environ.get("SWIN_URL",   "http://localhost:8003/predict")

# Timeout (seconds) for each sub-service call — rPPG can be slow (MediaPipe)
RPPG_TIMEOUT = float(os.environ.get("RPPG_TIMEOUT", "180"))
CNN_TIMEOUT  = float(os.environ.get("CNN_TIMEOUT",  "120"))

ALLOWED_EXTS = {"mp4", "avi", "mov", "mkv", "webm"}


# ═════════════════════════════════════════════════════════════════════════════
# ENSEMBLE CONFIGURATION  — loaded from ensemble_model.pkl
# ═════════════════════════════════════════════════════════════════════════════

def _load_ensemble() -> dict:
    """
    Load ensemble_model.pkl and extract the 3-stream weights for
    [rPPG, EfficientNet, Swin].

    The pkl may have been saved with 4 models (rPPG, EfficientNet, Xception,
    Swin) from the fusion notebook.  We detect this via input_model_names and
    extract / renormalise the 3 relevant weights.
    """
    pkl_path = os.path.join(BASE_DIR, "ensemble_model.pkl")
    json_path = os.path.join(BASE_DIR, "ensemble_model.json")

    artifact: dict = {}

    if os.path.exists(pkl_path):
        artifact = joblib.load(pkl_path)
        print(f"  Loaded ensemble_model.pkl (version={artifact.get('version','?')})", flush=True)
    elif os.path.exists(json_path):
        import json
        with open(json_path) as fh:
            artifact = json.load(fh)
        print("  Loaded ensemble_model.json (pkl not found)", flush=True)
    else:
        print(
            "  WARNING: Neither ensemble_model.pkl nor ensemble_model.json found. "
            "Using default weights [0.1063, 0.3435, 0.5501].",
            flush=True,
        )
        artifact = {
            "optuna_weights":   [0.1063, 0.3435, 0.5501],
            "temperature_T":    1.0,
            "threshold_youden": 0.5,
            "threshold_cv_f1":  0.5,
            "ensemble_method":  "NeuroPulse_3Stream_Optuna_Default",
            "input_model_names": ["rppg", "efficientnet", "swin"],
        }

    # Determine which weights to use for [rppg, efficientnet, swin]
    all_weights    = np.array(artifact.get("optuna_weights", [0.25, 0.25, 0.25, 0.25]), dtype=float)
    input_names    = artifact.get("input_model_names", [])

    # Streams we actually use
    WANTED_STREAMS = ["rppg", "efficientnet", "swin"]

    if input_names and set(WANTED_STREAMS).issubset(set(input_names)):
        # Extract the indices for the 3 wanted streams
        wanted_idx = [input_names.index(m) for m in WANTED_STREAMS]
        w3         = all_weights[wanted_idx]
    elif len(all_weights) == 3:
        # Already 3 weights in the right order
        w3 = all_weights.copy()
    elif len(all_weights) == 4:
        # 4-weight pkl: order is [rPPG, EfficientNet, Xception, Swin]
        # Drop index 2 (Xception) — matches predict.py legacy behaviour
        w3 = np.delete(all_weights, 2)
    else:
        w3 = np.ones(3) / 3.0

    # Renormalise
    if w3.sum() < EPS:
        w3 = np.ones(3) / 3.0
    else:
        w3 = w3 / w3.sum()

    return {
        "W3":        w3,
        "T_SCALE":   float(artifact.get("temperature_T",    1.0)),
        "THRESHOLD": float(artifact.get("threshold_youden", 0.5)),
        "THRESH_F1": float(artifact.get("threshold_cv_f1",  0.5)),
        "MODEL_NAME": str(artifact.get("ensemble_method",   "NeuroPulse_3Stream_Optuna")),
    }


print("[Orchestrator] Loading ensemble configuration...", flush=True)
_cfg       = _load_ensemble()
W3         = _cfg["W3"]
T_SCALE    = _cfg["T_SCALE"]
THRESHOLD  = _cfg["THRESHOLD"]
THRESH_F1  = _cfg["THRESH_F1"]
MODEL_NAME = _cfg["MODEL_NAME"]

print(
    f"  Ensemble: W={[round(w, 4) for w in W3.tolist()]}  "
    f"T={T_SCALE:.4f}  THR={THRESHOLD:.4f}  "
    f"streams=[rPPG, EfficientNet, Swin]",
    flush=True,
)
print("[Orchestrator] Ready on port 8000.", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# ENSEMBLE LOGIC
# ═════════════════════════════════════════════════════════════════════════════

def run_ensemble(
    p_rppg: float,
    p_eff:  float,
    p_swin: float,
) -> tuple[float, str, float]:
    """
    Combine 3 model probabilities with Optuna weights + temperature scaling.

    Returns:
        (p_final, label, confidence_pct)
        label      — "FAKE" or "REAL"
        confidence — percentage in (0, 100]
    """
    probs = np.array([p_rppg, p_eff, p_swin], dtype=float).clip(EPS, 1.0 - EPS)
    p_raw = float(W3 @ probs)

    # Temperature calibration (skip if T ≈ 1)
    if abs(T_SCALE - 1.0) > 1e-4:
        p_final = float(expit(logit(np.clip(p_raw, EPS, 1.0 - EPS)) / T_SCALE))
    else:
        p_final = p_raw

    is_fake    = p_final >= THRESHOLD
    label      = "FAKE" if is_fake else "REAL"
    confidence = p_final if is_fake else (1.0 - p_final)
    return p_final, label, round(confidence * 100, 2)


# ═════════════════════════════════════════════════════════════════════════════
# ASYNC SUB-SERVICE CALLER
# ═════════════════════════════════════════════════════════════════════════════

async def _call_service(
    client:   httpx.AsyncClient,
    url:      str,
    video_bytes: bytes,
    filename: str,
    timeout:  float,
    service:  str,
) -> tuple[float, str | None]:
    """
    POST *video_bytes* to *url* as multipart/form-data.

    Returns:
        (probability, error_message_or_None)
        On error: probability = 0.5 (neutral), error_message = description.
    """
    try:
        files    = {"video": (filename, video_bytes, "application/octet-stream")}
        response = await client.post(url, files=files, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        prob = float(data.get("probability", 0.5))
        prob = float(np.clip(prob, 0.0, 1.0))
        return prob, None
    except httpx.TimeoutException:
        return 0.5, f"{service} service timed out after {timeout}s"
    except httpx.ConnectError:
        return 0.5, f"{service} service is not reachable at {url}"
    except httpx.HTTPStatusError as exc:
        return 0.5, f"{service} service returned HTTP {exc.response.status_code}: {exc.response.text[:200]}"
    except Exception as exc:
        return 0.5, f"{service} service unexpected error: {exc}"


# ═════════════════════════════════════════════════════════════════════════════
# FASTAPI ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check() -> dict:
    """Check orchestrator + all sub-services."""
    service_health: dict[str, Any] = {}

    async with httpx.AsyncClient() as client:
        for name, base_url, port in [
            ("rppg",        RPPG_URL.replace("/predict", ""),  8001),
            ("efficientnet", EFF_URL.replace("/predict", ""), 8002),
            ("swin",        SWIN_URL.replace("/predict", ""),  8003),
        ]:
            try:
                r = await client.get(f"{base_url}/health", timeout=5.0)
                service_health[name] = r.json() if r.status_code == 200 else {"status": "error", "code": r.status_code}
            except Exception as exc:
                service_health[name] = {"status": "unreachable", "error": str(exc)}

    return {
        "status":            "ok",
        "version":           "4.0",
        "model":             MODEL_NAME,
        "ensemble_streams":  ["rPPG", "EfficientNet", "Swin"],
        "weights":           [round(w, 4) for w in W3.tolist()],
        "temperature_T":     round(T_SCALE, 4),
        "threshold":         round(THRESHOLD, 4),
        "sub_services":      service_health,
    }


@app.post("/api/v1/predict")
async def predict_endpoint(video: UploadFile = File(...)) -> dict:
    """
    Accept a video file, fan out to all 3 sub-services concurrently,
    ensemble their probabilities, and return the final verdict.
    """
    # ── 1. Validate file extension ─────────────────────────────────────────
    raw_name = video.filename or "upload.mp4"
    ext      = raw_name.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: .{ext}")

    # ── 2. Read bytes ONCE ─────────────────────────────────────────────────
    video_bytes = await video.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Empty file received.")

    filename = f"video.{ext}"

    # ── 3. Call all 3 sub-services concurrently ────────────────────────────
    async with httpx.AsyncClient() as client:
        rppg_task = _call_service(client, RPPG_URL, video_bytes, filename, RPPG_TIMEOUT, "rPPG")
        eff_task  = _call_service(client, EFF_URL,  video_bytes, filename, CNN_TIMEOUT,  "EfficientNet")
        swin_task = _call_service(client, SWIN_URL, video_bytes, filename, CNN_TIMEOUT,  "Swin")

        (p_rppg, err_rppg), (p_eff, err_eff), (p_swin, err_swin) = await asyncio.gather(
            rppg_task, eff_task, swin_task
        )

    # ── 4. Check for total failure ─────────────────────────────────────────
    errors = [e for e in (err_rppg, err_eff, err_swin) if e]
    if len(errors) == 3:
        raise HTTPException(
            status_code=503,
            detail=(
                "All sub-services failed. "
                f"rPPG: {err_rppg} | EfficientNet: {err_eff} | Swin: {err_swin}"
            ),
        )

    # ── 5. Run ensemble ────────────────────────────────────────────────────
    p_final, label, confidence = run_ensemble(p_rppg, p_eff, p_swin)

    # ── 6. Build response ──────────────────────────────────────────────────
    response: dict[str, Any] = {
        # Primary result
        "label":            label,
        "prediction":       1 if label == "FAKE" else 0,
        "probability":      round(p_final, 4),
        "p_fake":           round(p_final, 4),
        "confidence":       confidence,

        # Per-stream probabilities
        "P_rPPG":           round(p_rppg, 4),
        "P_efficientnet":   round(p_eff,  4),
        "P_swin":           round(p_swin, 4),

        # Ensemble metadata
        "model_name":       MODEL_NAME,
        "ensemble_streams": ["rPPG", "EfficientNet", "Swin"],
        "weights":          [round(w, 4) for w in W3.tolist()],
        "temperature_T":    round(T_SCALE, 4),
        "threshold":        round(THRESHOLD, 4),
        "threshold_f1":     round(THRESH_F1, 4),
    }

    # Include any non-fatal service warnings
    service_warnings: dict[str, str] = {}
    if err_rppg:
        service_warnings["rppg"]        = err_rppg
    if err_eff:
        service_warnings["efficientnet"] = err_eff
    if err_swin:
        service_warnings["swin"]        = err_swin
    if service_warnings:
        response["service_warnings"] = service_warnings

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("orchestrator:app", host="0.0.0.0", port=8000, reload=False)
