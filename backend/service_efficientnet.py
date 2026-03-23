# ═══════════════════════════════════════════════════════════════════════════
# service_efficientnet.py  —  NeuroPulse | EfficientNet-B4 Microservice
#
# Run:  uvicorn service_efficientnet:app --host 0.0.0.0 --port 8002
# Env:  requirements_cnn.txt  (torch, timm, facenet-pytorch, albumentations)
# ═══════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import os
import math
import tempfile
import shutil
import warnings

import numpy as np
import cv2
import torch
import torch.nn as nn
import timm
import albumentations as A

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI(title="NeuroPulse EfficientNet Service", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

ALLOWED_EXTS = {"mp4", "avi", "mov", "mkv", "webm"}


# ═════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION  — must match the training notebook exactly
# ═════════════════════════════════════════════════════════════════════════════

class TemporalAttention(nn.Module):
    """Multi-head self-attention with masked average pooling."""

    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention  = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout    = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # mask: BoolTensor [B, T], True = valid frame
        kpm      = ~mask if mask is not None else None
        attn_out, attn_weights = self.attention(x, x, x, key_padding_mask=kpm)
        x        = self.layer_norm(x + self.dropout(attn_out))

        if mask is not None:
            me     = mask.unsqueeze(-1).float()
            pooled = (x * me).sum(1) / me.sum(1).clamp(min=1)
        else:
            pooled = x.mean(1)
        return pooled, attn_weights


class SpatioTemporalDeepfakeCNN(nn.Module):
    """
    EfficientNet-B4 backbone → BiLSTM → Multi-head attention → classifier.
    Matches model_efficientnet.ipynb training configuration exactly.
    """

    def __init__(
        self,
        model_name:      str   = "efficientnet_b4",
        hidden_dim:      int   = 256,
        dropout:         float = 0.3,
        pretrained:      bool  = True,
        temporal_type:   str   = "bilstm_attention",
        lstm_hidden:     int   = 256,
        lstm_layers:     int   = 2,
        attention_heads: int   = 4,
        freeze_backbone: bool  = False,
    ):
        super().__init__()
        self.temporal_type = temporal_type

        # Backbone — EfficientNet-B4 (1792-dim output)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool="avg", drop_path_rate=0.2,
        )
        self.backbone_dim = self.backbone.num_features

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if temporal_type in ("bilstm", "bilstm_attention"):
            self.temporal = nn.LSTM(
                input_size=self.backbone_dim, hidden_size=lstm_hidden,
                num_layers=lstm_layers, batch_first=True, bidirectional=True,
                dropout=dropout if lstm_layers > 1 else 0,
            )
            temporal_out_dim      = lstm_hidden * 2
            self.temporal_dropout = nn.Dropout(p=dropout)
            if temporal_type == "bilstm_attention":
                self.temporal_attention = TemporalAttention(
                    feature_dim=lstm_hidden * 2,
                    num_heads=attention_heads,
                    dropout=dropout,
                )

        elif temporal_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.backbone_dim, nhead=attention_heads,
                dim_feedforward=self.backbone_dim * 2,
                dropout=dropout, activation="gelu", batch_first=True,
            )
            self.temporal           = nn.TransformerEncoder(encoder_layer, num_layers=lstm_layers)
            self.temporal_attention = TemporalAttention(
                feature_dim=self.backbone_dim,
                num_heads=attention_heads,
                dropout=dropout,
            )
            temporal_out_dim = self.backbone_dim

        else:
            raise ValueError(f"Unknown temporal_type: {temporal_type!r}")

        self.temporal_out_dim = temporal_out_dim

        # Classifier head — LayerNorm instead of BatchNorm for small batch stability
        self.classifier = nn.Sequential(
            nn.Linear(temporal_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),   nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        frames: torch.Tensor,
        mask:   torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        features       = self.backbone(frames.view(B * T, C, H, W)).view(B, T, -1)

        if self.temporal_type == "bilstm":
            with torch.backends.cudnn.flags(enabled=False):
                lstm_out, _ = self.temporal(features)
            if mask is not None:
                me     = mask.unsqueeze(-1).float()
                pooled = (lstm_out * me).sum(1) / me.sum(1).clamp(min=1)
            else:
                pooled = lstm_out.mean(1)

        elif self.temporal_type == "bilstm_attention":
            with torch.backends.cudnn.flags(enabled=False):
                lstm_out, _ = self.temporal(features)
            lstm_out = self.temporal_dropout(lstm_out)
            pooled, _ = self.temporal_attention(lstm_out, mask)

        elif self.temporal_type == "transformer":
            attn_mask = ~mask if mask is not None else None
            trans_out  = self.temporal(features, src_key_padding_mask=attn_mask)
            pooled, _  = self.temporal_attention(trans_out, mask)

        return self.classifier(pooled).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════

print("[EfficientNet Service] Loading model...", flush=True)

_eff_path = os.path.join(BASE_DIR, "models", "efficientnet_model.pth")
if not os.path.exists(_eff_path):
    raise FileNotFoundError(f"EfficientNet weights not found: {_eff_path}")

_eff_model = SpatioTemporalDeepfakeCNN(
    model_name="efficientnet_b4",
    pretrained=False,
    hidden_dim=256,
    lstm_hidden=256,
    lstm_layers=2,
    attention_heads=4,
    dropout=0.5,
    freeze_backbone=False,
).to(DEVICE)

# weights_only=True: safe loading, prevents arbitrary code execution (torch >= 2.0)
_eff_model.load_state_dict(
    torch.load(_eff_path, map_location=DEVICE, weights_only=True)
)
_eff_model.eval()
print(f"  EfficientNet-B4 loaded on {DEVICE}", flush=True)

# ── MTCNN face detector ────────────────────────────────────────────────────
from facenet_pytorch import MTCNN as _MTCNN  # noqa: E402 (after torch is imported)

_mtcnn_224 = _MTCNN(
    image_size=224, margin=20, min_face_size=60,
    thresholds=[0.6, 0.7, 0.7], factor=0.709,
    post_process=False, device=DEVICE, keep_all=False,
)
print("  MTCNN (224) loaded", flush=True)
print("[EfficientNet Service] Ready on port 8002.", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# FACE EXTRACTION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _center_crop(rgb_frame: np.ndarray, target: int) -> np.ndarray:
    h, w   = rgb_frame.shape[:2]
    s      = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return cv2.resize(rgb_frame[y0:y0 + s, x0:x0 + s], (target, target))


@torch.no_grad()
def _extract_face(
    mtcnn_detector,
    rgb_frame: np.ndarray,
    img_size:  int,
) -> np.ndarray:
    """Extract a face crop; fall back to centre-crop on failure."""
    from PIL import Image as _PIL_Image
    pil_img = _PIL_Image.fromarray(rgb_frame)
    try:
        face_tensor = mtcnn_detector(pil_img)
        if face_tensor is not None:
            face_np = face_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            # Quality gate: reject blurry face crops
            gray_var = cv2.Laplacian(
                cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY), cv2.CV_64F
            ).var()
            if gray_var >= 20.0:
                return face_np
        return _center_crop(rgb_frame, img_size)
    except Exception:
        return _center_crop(rgb_frame, img_size)


def _extract_faces_from_video(
    video_path: str,
    n_frames:   int,
    img_size:   int,
) -> np.ndarray:
    """Uniformly sample *n_frames* face crops from *video_path*."""
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return np.zeros((n_frames, img_size, img_size, 3), dtype=np.uint8)

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    faces: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            faces.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
            continue
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = _extract_face(_mtcnn_224, rgb, img_size)
        faces.append(face)

    cap.release()
    return np.array(faces, dtype=np.uint8)


# ═════════════════════════════════════════════════════════════════════════════
# TTA INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def _build_transform(img_size: int, mean: list, std: list, n_frames: int) -> A.Compose:
    """Build albumentations pipeline that transforms all frames identically."""
    extra = {f"image{i}": "image" for i in range(1, n_frames)}
    return A.Compose(
        [A.Resize(img_size, img_size), A.Normalize(mean=mean, std=std)],
        additional_targets=extra,
    )


def _run_cnn_tta(
    model:        nn.Module,
    faces_np:     np.ndarray,
    img_size:     int,
    mean:         list,
    std:          list,
    n_tta_passes: int,
) -> float:
    """
    Run Test-Time Augmentation over *n_tta_passes* passes.
    Returns the mean sigmoid probability across all passes.
    """
    tf        = _build_transform(img_size, mean, std, len(faces_np))
    all_probs: list[float] = []

    for tta_idx in range(n_tta_passes):
        frames: list[np.ndarray] = []
        for f in faces_np:
            aug = f.copy()
            if tta_idx == 1:
                aug = np.fliplr(aug).copy()
            elif tta_idx == 2:
                aug = np.clip(aug.astype(np.int32) + 15, 0, 255).astype(np.uint8)
            elif tta_idx == 3:
                aug = np.clip(aug.astype(np.int32) - 15, 0, 255).astype(np.uint8)
            elif tta_idx == 4:
                aug = cv2.GaussianBlur(aug, (3, 3), 0)
            frames.append(aug)

        # Build albumentations call with additional_targets
        kwargs = {"image": frames[0]}
        for k in range(1, len(frames)):
            kwargs[f"image{k}"] = frames[k]
        result = tf(**kwargs)

        tensor_list: list[torch.Tensor] = []
        for k in range(len(frames)):
            t = result["image"] if k == 0 else result[f"image{k}"]
            if isinstance(t, np.ndarray):
                t = torch.from_numpy(t.transpose(2, 0, 1)).float()
            tensor_list.append(t)

        # Stack → [1, T, C, H, W]
        fstack = torch.stack(tensor_list).unsqueeze(0).to(DEVICE)
        mask   = torch.ones(1, len(frames), dtype=torch.bool, device=DEVICE)

        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
            logit_out = model(fstack, mask)

        prob = float(torch.sigmoid(logit_out.squeeze()).item())
        all_probs.append(prob)

    return float(np.mean(all_probs))


def predict_efficientnet(video_path: str) -> float:
    """Full EfficientNet-B4 inference: face extraction → 5-pass TTA."""
    faces = _extract_faces_from_video(video_path, n_frames=16, img_size=224)
    return _run_cnn_tta(
        _eff_model, faces,
        img_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD,
        n_tta_passes=5,
    )


# ═════════════════════════════════════════════════════════════════════════════
# FASTAPI ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok", "service": "efficientnet", "device": str(DEVICE), "port": 8002}


@app.post("/predict")
async def predict_endpoint(video: UploadFile = File(...)):
    ext = (video.filename or "upload.mp4").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: .{ext}")

    tmp_dir  = tempfile.mkdtemp(prefix="neuropulse_eff_")
    tmp_path = os.path.join(tmp_dir, f"video.{ext}")
    try:
        content = await video.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file received.")
        with open(tmp_path, "wb") as fh:
            fh.write(content)

        prob = predict_efficientnet(tmp_path)
        return {"probability": round(prob, 4), "model": "efficientnet", "status": "ok"}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"EfficientNet inference error: {exc}") from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service_efficientnet:app", host="0.0.0.0", port=8002, reload=False)
