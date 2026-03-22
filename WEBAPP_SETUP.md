# NeuroPulse Web App Setup

This repository now includes:

- `backend/` FastAPI API for model-name based video inference.
- `frontend/` polished browser UI for upload-only prediction flow.

## 1) Start Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Place your final model artifact in:

- `backend/models/`

Examples:

- `backend/models/final_ensemble.joblib`
- `backend/models/ensemble_final_predictions.csv`

Then use model name only:

- `final_ensemble`
- `ensemble_final_predictions`

Note: model name input is optional now. Backend auto-selects model.

## 2) Open Frontend

Frontend is auto-served by backend at:

- `http://127.0.0.1:8000`

## API Endpoints

- `GET /api/v1/health`
- `GET /api/v1/models`
- `POST /api/v1/predict` with multipart form fields:
  - `model_name` (optional)
  - `video`
