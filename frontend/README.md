# NeuroPulse Frontend

Research-style upload interface for one-step deepfake prediction.

## Run Locally

Recommended: serve frontend via backend (automatic at `/`).

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open:

- `http://127.0.0.1:8000`

User flow:

1. Upload video
2. Click Analyze
3. Receive prediction
