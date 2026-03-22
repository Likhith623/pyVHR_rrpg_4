# NeuroPulse Backend (FastAPI)

Production-style backend for video deepfake inference.

## What It Does

- Accepts a video upload.
- Auto-selects model by priority (or `DEFAULT_MODEL_NAME` if set).
- Loads the corresponding artifact from `backend/models/`.
- Returns Deepfake/Real prediction + confidence.

## Supported Model Artifacts

- `.joblib`
- `.pkl` / `.pickle`
- `.pt` / `.pth`
- `.csv` (for lookup-style ensemble files with `P_final`/`P_CNN`/`P_rPPG`)

## Start Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open app at `http://127.0.0.1:8000`.
Open docs at `http://127.0.0.1:8000/docs`.

## Model Placement

Put your final ensemble artifact into:

- `backend/models/`

Example:

- `backend/models/final_ensemble.joblib`

The backend automatically selects model in this order:

1. `DEFAULT_MODEL_NAME` env var (if present)
2. First model matching keywords like `final_ensemble` or `ensemble`
3. First available model artifact

If you still want explicit selection, send optional `model_name` in multipart form.

## Notes

If your model needs custom preprocessing, expose a `predict_video(path)` method in the loaded object, or adapt logic in `app/model_service.py`.
