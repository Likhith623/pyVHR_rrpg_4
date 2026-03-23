# ═════════════════════════════════════════════════════════════════════════════
# service_rppg.py  —  NeuroPulse | rPPG Microservice
# ═════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import os

# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL MACOS / APPLE SILICON FIX
# Prevents Segmentation Faults during joblib.load() by disabling conflicting 
# OpenMP and BLAS multithreading pools before C-extensions initialize.
# ─────────────────────────────────────────────────────────────────────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# ─────────────────────────────────────────────────────────────────────────────

import tempfile
import shutil
import warnings

import numpy as np
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: Joblib / NumPy unpickler patch
# Needed when the .joblib files were saved with an older NumPy that serialises
# BitGenerator objects differently from the current NumPy version.
# ─────────────────────────────────────────────────────────────────────────────
_originalFindClass = joblib.numpy_pickle.NumpyUnpickler.find_class


def _patchedFindClass(self, moduleName, className):
    if moduleName == "numpy.random._pickle" and className == "__bit_generator_ctor":
        return _safeBitGeneratorConstructor
    return _originalFindClass(self, moduleName, className)


def _safeBitGeneratorConstructor(bitGeneratorInput="PCG64", *args, **kwargs):
    generatorName = "PCG64"
    if isinstance(bitGeneratorInput, str):
        generatorName = bitGeneratorInput.rsplit(".", 1)[-1]
    elif hasattr(bitGeneratorInput, "__name__"):
        generatorName = bitGeneratorInput.__name__
    elif hasattr(bitGeneratorInput, "__class__"):
        generatorName = bitGeneratorInput.__class__.__name__

    if not hasattr(np.random, generatorName):
        generatorName = "PCG64"

    baseClass = getattr(np.random, generatorName)

    class PatchedGenerator(baseClass):
        def __setstate__(self, stateData):
            # 1. Legacy tuple format
            if isinstance(stateData, tuple):
                stateData = {
                    "bit_generator": self.__class__.__name__,
                    "state":         stateData[1] if len(stateData) > 1 else {},
                    "has_uint32":    stateData[2] if len(stateData) > 2 else 0,
                    "uinteger":      stateData[3] if len(stateData) > 3 else 0,
                }

            # 2. Non-dict state
            if not isinstance(stateData, dict):
                stateData = baseClass().state
                stateData["bit_generator"] = self.__class__.__name__
                super().__setstate__(stateData)
                return

            # 3. Inner 'state' key is a SeedSequence / non-dict
            stateData = dict(stateData)
            inner = stateData.get("state")
            if not isinstance(inner, dict):
                try:
                    raw = inner.generate_state(4, dtype=np.uint64)
                    stateData["state"] = {
                        "state": int(raw[0]) | (int(raw[1]) << 64),
                        "inc":   int(raw[2]) | (int(raw[3]) << 64),
                    }
                except Exception:
                    stateData["state"] = baseClass().state["state"]

            # 4. Guarantee Cython class-name match
            stateData["bit_generator"] = self.__class__.__name__
            stateData.setdefault("has_uint32", 0)
            stateData.setdefault("uinteger",   0)
            super().__setstate__(stateData)

    return PatchedGenerator.__new__(PatchedGenerator)


joblib.numpy_pickle.NumpyUnpickler.find_class = _patchedFindClass
# ─────────────────────────────────────────────────────────────────────────────


import cv2
import mediapipe as mp
from scipy.signal import butter, filtfilt, welch

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI(title="NeuroPulse rPPG Service", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

EPS      = 1e-7
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load rPPG artifacts ────────────────────────────────────────────────────
print("[rPPG Service] Loading artifacts...", flush=True)

_rppg_dir = os.path.join(BASE_DIR, "rppg")
for _fname in ("best_rppg_ml_model.joblib", "rppg_scaler.joblib", "rppg_selector.joblib"):
    _fp = os.path.join(_rppg_dir, _fname)
    if not os.path.exists(_fp):
        raise FileNotFoundError(f"rPPG artefact not found: {_fp}")

_RPPG_SCALER   = joblib.load(os.path.join(_rppg_dir, "rppg_scaler.joblib"))
_RPPG_SELECTOR = joblib.load(os.path.join(_rppg_dir, "rppg_selector.joblib"))
_RPPG_MODEL    = joblib.load(os.path.join(_rppg_dir, "best_rppg_ml_model.joblib"))
print("  rPPG ML artifacts loaded", flush=True)

# ── MediaPipe FaceMesh ─────────────────────────────────────────────────────
_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
print("  MediaPipe FaceMesh loaded", flush=True)
print("[rPPG Service] Ready on port 8001.", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# ROI LANDMARK DEFINITIONS  (9 facial regions, 468 FaceMesh landmarks)
# ═════════════════════════════════════════════════════════════════════════════
ROI_LANDMARKS: dict[str, list[int]] = {
    "forehead": [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ],
    "left_cheek":     [50, 187, 123, 116, 143, 156, 70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "right_cheek":    [280, 411, 352, 345, 372, 383, 301, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    "chin":           [175, 171, 208, 199, 428, 395, 200, 175, 152, 377, 400, 378],
    "nose":           [4, 5, 6, 168, 195, 197, 1, 2, 98, 327, 326, 0],
    "left_jaw":       [172, 136, 150, 149, 176, 148, 152, 377, 378, 400],
    "right_jaw":      [397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150],
    "left_forehead":  [105, 66, 107, 55, 65, 52, 53, 46, 70, 63],
    "right_forehead": [334, 296, 336, 285, 295, 282, 283, 276, 301, 293],
}


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL EXTRACTION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _extract_roi_rgb(
    frame_bgr: np.ndarray,
    landmarks_px: list,
    indices: list[int],
) -> np.ndarray | None:
    """Extract mean RGB from a convex-hull facial ROI."""
    pts = np.array(
        [
            [landmarks_px[i].x * frame_bgr.shape[1],
             landmarks_px[i].y * frame_bgr.shape[0]]
            for i in indices
            if i < len(landmarks_px)
        ],
        dtype=np.float32,
    )
    if len(pts) < 3:
        return None
    hull = cv2.convexHull(pts.astype(np.int32))
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if mask.sum() < 1000:
        return None
    roi = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(float)
    return rgb[mask > 0].mean(axis=0)


def _chrom_rppg(rgb_signals: list, fps: float = 30.0) -> np.ndarray:
    """CHROM rPPG algorithm (de Haan & Jeanne, IEEE TBME 2013)."""
    sig = np.array(rgb_signals, dtype=float)
    if len(sig) < 10:
        return np.zeros(len(sig))

    mn    = sig.mean(axis=0) + EPS
    sig_n = sig / mn
    Xs    = 3 * sig_n[:, 0] - 2 * sig_n[:, 1]
    Ys    = 1.5 * sig_n[:, 0] + sig_n[:, 1] - 1.5 * sig_n[:, 2]
    alpha = (Xs.std() + EPS) / (Ys.std() + EPS)
    rppg  = Xs - alpha * Ys

    nyq    = fps / 2.0
    lo, hi = 0.7 / nyq, min(4.0 / nyq, 0.99)
    try:
        b, a = butter(3, [lo, hi], btype="band")
        rppg = filtfilt(b, a, rppg)
    except Exception:
        pass
    return rppg


def _extract_117_features(roi_signals_dict: dict, fps: float = 30.0) -> np.ndarray:
    """Extract 117-dimensional rPPG feature vector from 9 facial ROIs."""
    features:     list[float] = []
    roi_names                 = list(roi_signals_dict.keys())
    rppg_signals: dict        = {}

    # ── Per-ROI spectral + HRV features (9 ROIs × 10 features = 90) ─────────
    for roi_name, rgb_seq in roi_signals_dict.items():
        rppg = _chrom_rppg(rgb_seq, fps)
        rppg_signals[roi_name] = rppg

        if len(rppg) < 5:
            features.extend([0.0] * 10)
            continue

        freqs, psd = welch(rppg, fs=fps, nperseg=min(len(rppg), 64), nfft=1024)
        hr_mask    = (freqs >= 0.7) & (freqs <= 4.0)
        psd_hr     = psd[hr_mask]
        psd_hr_sum = psd_hr.sum() + EPS

        snr      = float(psd_hr.sum() / (psd.sum() + EPS))
        purity   = float(psd_hr.max() / psd_hr_sum) if psd_hr.size > 0 else 0.0
        entropy  = float(-np.sum((psd_hr / psd_hr_sum) * np.log(psd_hr / psd_hr_sum + EPS)))
        dom_f    = float(freqs[hr_mask][psd_hr.argmax()]) if psd_hr.size > 0 else 0.0
        centroid = float(np.sum(freqs[hr_mask] * psd_hr) / psd_hr_sum)
        rmssd    = float(np.sqrt(np.mean(np.diff(rppg) ** 2)))
        sdnn     = float(rppg.std())
        energy   = float(np.sum(rppg ** 2))
        rms      = float(np.sqrt(np.mean(rppg ** 2)))
        crest    = float(np.abs(rppg).max() / (rms + EPS))
        bpm      = dom_f * 60.0
        features.extend([snr, purity, entropy, dom_f, centroid,
                         rmssd, sdnn, energy, crest, bpm])

    # ── Cross-ROI Pearson correlations (9 choose 2 = 36 pairs) ───────────────
    rppg_list = [rppg_signals[r] for r in roi_names]
    min_len   = min(len(s) for s in rppg_list)
    if min_len >= 5:
        for ii in range(len(rppg_list)):
            for jj in range(ii + 1, len(rppg_list)):
                a = rppg_list[ii][:min_len]
                b = rppg_list[jj][:min_len]
                if a.std() > EPS and b.std() > EPS:
                    corr = float(np.corrcoef(a, b)[0, 1])
                else:
                    corr = 0.0
                features.append(corr)
    else:
        features.extend([0.0] * 36)

    # ── Clamp / pad to exactly 117 ────────────────────────────────────────────
    features = features[:117]
    while len(features) < 117:
        features.append(0.0)
    return np.array(features, dtype=np.float32)


def predict_rppg(video_path: str, max_frames: int = 60) -> float:
    """
    Run full rPPG pipeline on *video_path*.
    Returns P(fake) in [0, 1]; returns 0.5 on failure (neutral / uncertain).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.5

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps     = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    n_samp  = min(max_frames, max(total, 1))
    indices = np.linspace(0, max(total - 1, 0), n_samp, dtype=int)

    roi_rgb_seqs: dict[str, list] = {k: [] for k in ROI_LANDMARKS}

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        # Quality gate: reject blurry frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 10.0:
            continue

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            continue

        lms = result.multi_face_landmarks[0].landmark
        for roi_name, roi_indices in ROI_LANDMARKS.items():
            mean_rgb = _extract_roi_rgb(frame, lms, roi_indices)
            if mean_rgb is not None:
                roi_rgb_seqs[roi_name].append(mean_rgb)

    cap.release()

    # Need at least 3 ROIs with ≥10 usable frames each
    valid_rois = {k: v for k, v in roi_rgb_seqs.items() if len(v) >= 10}
    if len(valid_rois) < 3:
        return 0.5

    features          = _extract_117_features(valid_rois, fps=fps)
    # Sanitise: replace NaN / Inf before scaling
    features          = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    features_scaled   = _RPPG_SCALER.transform(features.reshape(1, -1))
    features_selected = _RPPG_SELECTOR.transform(features_scaled)
    prob              = float(_RPPG_MODEL.predict_proba(features_selected)[0, 1])
    return float(np.clip(prob, 0.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# FASTAPI ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

ALLOWED_EXTS = {"mp4", "avi", "mov", "mkv", "webm"}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "rppg", "port": 8001}


@app.post("/predict")
async def predict_endpoint(video: UploadFile = File(...)):
    ext = (video.filename or "upload.mp4").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: .{ext}")

    tmp_dir  = tempfile.mkdtemp(prefix="neuropulse_rppg_")
    tmp_path = os.path.join(tmp_dir, f"video.{ext}")
    try:
        content = await video.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file received.")
        with open(tmp_path, "wb") as fh:
            fh.write(content)

        prob = predict_rppg(tmp_path)
        return {"probability": round(prob, 4), "model": "rppg", "status": "ok"}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"rPPG inference error: {exc}") from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service_rppg:app", host="0.0.0.0", port=8001, reload=False)
