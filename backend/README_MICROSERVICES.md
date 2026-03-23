# NeuroPulse — Microservices Architecture

## Why this architecture?

The original `predict.py` loaded ALL models in a **single process**, which means:
- `mediapipe` (rPPG) must be compatible with `torch` (CNNs)
- `mediapipe 0.10.14` requires `protobuf ~= 3.20` which conflicts with newer `torch`
- Result: the server crashes on import or produces silent wrong results

**The fix**: each model runs in its own process with its own virtual environment.
Zero dependency conflicts — ever.

---

## Directory Structure

```
project/
├── orchestrator.py            # Main API (port 8000)  — no torch
├── service_rppg.py            # rPPG service  (port 8001) — no torch
├── service_efficientnet.py    # EfficientNet-B4 (port 8002) — torch only
├── service_swin.py            # Swin-Tiny (port 8003) — torch only
│
├── requirements_rppg.txt      # venv_rppg deps
├── requirements_cnn.txt       # venv_cnn deps (shared by 8002 + 8003)
├── requirements_orchestrator.txt  # venv_orch deps
│
├── start_services.sh          # Launch script (Linux/Mac)
│
├── ensemble_model.pkl         # ← from fusion_ensemble notebook
├── ensemble_model.json        # ← optional JSON copy
│
├── models/
│   ├── efficientnet_model.pth # ← from model_efficientnet notebook
│   └── swin_model.pth         # ← from model_swin notebook
│
└── rppg/
    ├── best_rppg_ml_model.joblib  # ← from model_rppg notebook
    ├── rppg_scaler.joblib
    └── rppg_selector.joblib
```

---

## Setup (one-time)

```bash
mkdir -p logs

# ── venv for rPPG (mediapipe, sklearn, xgboost — NO torch) ────────────────
python -m venv venv_rppg
source venv_rppg/bin/activate
pip install -r requirements_rppg.txt
deactivate

# ── venv for CNN services (torch, timm, facenet-pytorch — NO mediapipe) ───
python -m venv venv_cnn
source venv_cnn/bin/activate
pip install -r requirements_cnn.txt
deactivate

# ── venv for Orchestrator (httpx, fastapi, numpy — nothing else) ──────────
python -m venv venv_orch
source venv_orch/bin/activate
pip install -r requirements_orchestrator.txt
deactivate
```

---

## Running

```bash
chmod +x start_services.sh
./start_services.sh
```

Or manually (in separate terminals):

```bash
# Terminal 1
source venv_rppg/bin/activate
uvicorn service_rppg:app --host 0.0.0.0 --port 8001

# Terminal 2
source venv_cnn/bin/activate
uvicorn service_efficientnet:app --host 0.0.0.0 --port 8002

# Terminal 3
source venv_cnn/bin/activate
uvicorn service_swin:app --host 0.0.0.0 --port 8003

# Terminal 4 (after the others are healthy)
source venv_orch/bin/activate
uvicorn orchestrator:app --host 0.0.0.0 --port 8000
```

---

## API Usage

### Health check
```bash
curl http://localhost:8000/health
```

### Predict
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "video=@/path/to/video.mp4"
```

### Response
```json
{
  "label": "FAKE",
  "prediction": 1,
  "probability": 0.8312,
  "p_fake": 0.8312,
  "confidence": 83.12,

  "P_rPPG": 0.7421,
  "P_efficientnet": 0.8901,
  "P_swin": 0.8104,

  "model_name": "NeuroPulse_3Stream_Optuna",
  "ensemble_streams": ["rPPG", "EfficientNet", "Swin"],
  "weights": [0.1063, 0.3435, 0.5501],
  "temperature_T": 1.1234,
  "threshold": 0.4823,
  "threshold_f1": 0.4701
}
```

---

## Architecture

```
Client (video upload)
        │
        ▼
┌───────────────────────────┐
│   orchestrator.py :8000   │  ← minimal deps (httpx, fastapi, numpy)
│   POST /api/v1/predict    │
└──────────┬────────────────┘
           │  asyncio.gather (concurrent, all 3 at once)
     ┌─────┴──────┬──────────────┐
     ▼            ▼              ▼
┌─────────┐  ┌──────────┐  ┌──────────┐
│ :8001   │  │  :8002   │  │  :8003   │
│ rPPG    │  │ EffNet   │  │ Swin     │
│ venv    │  │ venv_cnn │  │ venv_cnn │
│ _rppg   │  │          │  │          │
└─────────┘  └──────────┘  └──────────┘
     │            │              │
   P_rPPG    P_effnet        P_swin
     └────────────┴──────────────┘
                  │
         Optuna weights + Temperature
         scaling + Youden threshold
                  │
             FAKE / REAL
```

---

## Fault tolerance

- If **one** service is down → that stream returns 0.5 (neutral), ensemble continues
- If **two** services are down → same behaviour, `service_warnings` key in response
- If **all three** fail → HTTP 503

---

## Ensemble weights (from ensemble_model.pkl)

The orchestrator reads `ensemble_model.pkl` produced by `fusion_ensemble.ipynb`.
It correctly handles both:
- **3-model PKL** (`input_model_names = [rppg, efficientnet, swin]`)
- **4-model PKL** (`input_model_names = [rppg, efficientnet, xception, swin]`)
  → Xception weight is dropped and the remaining 3 are renormalised.

The `ensemble_model.json` is also accepted if the `.pkl` is missing.

---

## Environment variable overrides

```bash
export RPPG_URL="http://rppg-host:8001/predict"
export EFF_URL="http://cnn-host:8002/predict"
export SWIN_URL="http://cnn-host:8003/predict"
export RPPG_TIMEOUT=180   # seconds
export CNN_TIMEOUT=120    # seconds
```

This lets you run each service on a different machine if needed.
