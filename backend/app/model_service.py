from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .settings import MODEL_DIRS


@dataclass
class LoadedModel:
    name: str
    kind: str
    artifact_path: Path
    model: Any


class ModelRepository:
    """Loads and caches models by name, using filename discovery."""

    SUPPORTED_EXTENSIONS = (".joblib", ".pkl", ".pickle", ".pt", ".pth", ".csv")

    def __init__(self) -> None:
        self._cache: dict[str, LoadedModel] = {}

    def list_available_models(self) -> list[str]:
        names: set[str] = set()
        for model_dir in MODEL_DIRS:
            if not model_dir.exists():
                continue
            for path in model_dir.iterdir():
                if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    names.add(path.stem)
        return sorted(names)

    def resolve_model_name(self, requested_name: str | None = None) -> str:
        available = self.list_available_models()
        if not available:
            raise FileNotFoundError(
                f"No model artifacts found. Put model files in one of: {MODEL_DIRS}"
            )

        if requested_name:
            return requested_name

        env_default = os.getenv("DEFAULT_MODEL_NAME", "").strip()
        if env_default:
            if env_default in available:
                return env_default
            starts = [name for name in available if name.startswith(env_default)]
            if starts:
                return starts[0]

        preferred_keywords = [
            "final_ensemble",
            "ensemble_final",
            "best_ensemble",
            "ensemble",
        ]
        for keyword in preferred_keywords:
            for name in available:
                if keyword in name.lower():
                    return name

        return available[0]

    def _find_artifact(self, model_name: str) -> Path:
        # Prefer exact stem match, then prefix match.
        for model_dir in MODEL_DIRS:
            if not model_dir.exists():
                continue
            exact = [
                p for p in model_dir.iterdir()
                if p.is_file() and p.stem == model_name and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]
            if exact:
                return exact[0]

        for model_dir in MODEL_DIRS:
            if not model_dir.exists():
                continue
            prefixed = sorted(
                [
                    p for p in model_dir.iterdir()
                    if p.is_file()
                    and p.stem.startswith(model_name)
                    and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
                ]
            )
            if prefixed:
                return prefixed[0]

        raise FileNotFoundError(
            f"Model '{model_name}' not found. Place artifact in one of: {MODEL_DIRS}"
        )

    def load(self, model_name: str) -> LoadedModel:
        if model_name in self._cache:
            return self._cache[model_name]

        artifact = self._find_artifact(model_name)
        suffix = artifact.suffix.lower()

        if suffix == ".joblib":
            import joblib

            loaded = LoadedModel(model_name, "joblib", artifact, joblib.load(artifact))
        elif suffix in {".pkl", ".pickle"}:
            with artifact.open("rb") as f:
                loaded = LoadedModel(model_name, "pickle", artifact, pickle.load(f))
        elif suffix in {".pt", ".pth"}:
            try:
                torch = __import__("torch")
            except ImportError as exc:
                raise RuntimeError("torch is required to load .pt/.pth models") from exc
            loaded = LoadedModel(model_name, "torch", artifact, torch.load(artifact, map_location="cpu"))
        elif suffix == ".csv":
            loaded = LoadedModel(model_name, "csv", artifact, pd.read_csv(artifact))
        else:
            raise ValueError(f"Unsupported model artifact extension: {suffix}")

        self._cache[model_name] = loaded
        return loaded


class InferenceService:
    """Adapter that predicts deepfake probability from a video for various model types."""

    def __init__(self, repo: ModelRepository) -> None:
        self.repo = repo

    def predict(self, video_path: Path, model_name: str) -> dict:
        loaded = self.repo.load(model_name)

        if loaded.kind == "csv":
            probability = self._predict_from_csv_lookup(loaded.model, video_path)
            details = {
                "artifact": str(loaded.artifact_path),
                "mode": "csv_lookup",
            }
            return self._build_result(model_name, probability, details)

        features = self._extract_video_features(video_path)
        probability = self._predict_with_model(loaded.model, features, video_path)

        details = {
            "artifact": str(loaded.artifact_path),
            "mode": loaded.kind,
            "feature_dim": len(features),
        }
        return self._build_result(model_name, probability, details)

    @staticmethod
    def _build_result(model_name: str, probability: float, details: dict) -> dict:
        prob = float(np.clip(probability, 0.0, 1.0))
        prediction = int(prob >= 0.5)
        label = "Deepfake" if prediction == 1 else "Real"
        confidence = prob * 100 if prediction == 1 else (1 - prob) * 100
        return {
            "model_name": model_name,
            "prediction": prediction,
            "label": label,
            "probability": prob,
            "confidence": float(confidence),
            "details": details,
        }

    @staticmethod
    def _predict_from_csv_lookup(df: pd.DataFrame, video_path: Path) -> float:
        # Supports ensemble CSV artifacts with columns like: video_id, P_final or P_CNN.
        score_col = None
        for candidate in ("P_final", "probability", "P_CNN", "P_rPPG"):
            if candidate in df.columns:
                score_col = candidate
                break

        if score_col is None:
            raise ValueError(
                "CSV model must contain one of these columns: P_final, probability, P_CNN, P_rPPG"
            )

        if "video_id" not in df.columns:
            # If no identifier exists, return dataset mean as prior.
            return float(df[score_col].mean())

        key = video_path.name
        matches = df[df["video_id"].astype(str).str.contains(key, case=False, regex=False)]
        if len(matches) > 0:
            return float(matches.iloc[0][score_col])

        return float(df[score_col].mean())

    @staticmethod
    def _extract_video_features(video_path: Path) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("Could not open uploaded video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        sample_limit = min(64, frame_count) if frame_count > 0 else 0
        sample_idxs = set(np.linspace(0, max(frame_count - 1, 0), num=sample_limit, dtype=int))

        means, stds, sharpness = [], [], []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in sample_idxs:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                means.append(float(gray.mean()))
                stds.append(float(gray.std()))
                sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            idx += 1
        cap.release()

        duration = frame_count / fps if fps > 0 else 0.0
        features = np.array(
            [
                fps,
                frame_count,
                width,
                height,
                duration,
                float(np.mean(means) if means else 0.0),
                float(np.std(means) if means else 0.0),
                float(np.mean(stds) if stds else 0.0),
                float(np.std(stds) if stds else 0.0),
                float(np.mean(sharpness) if sharpness else 0.0),
                float(np.std(sharpness) if sharpness else 0.0),
            ],
            dtype=np.float32,
        )
        return features

    @staticmethod
    def _predict_with_model(model: Any, features: np.ndarray, video_path: Path) -> float:
        x = features.reshape(1, -1)

        if hasattr(model, "predict_video"):
            prob = model.predict_video(str(video_path))
            return float(np.squeeze(prob))

        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(x)
            if np.ndim(probas) == 2 and probas.shape[1] >= 2:
                return float(probas[0, 1])
            return float(np.squeeze(probas))

        if hasattr(model, "predict"):
            pred = model.predict(x)
            val = float(np.squeeze(pred))
            return float(np.clip(val, 0.0, 1.0))

        if callable(model):
            val = model(x)
            return float(np.clip(float(np.squeeze(val)), 0.0, 1.0))

        raise RuntimeError(
            "Loaded model object is not directly inferable. "
            "Expose predict_video(path) or sklearn-like predict_proba(X)."
        )
