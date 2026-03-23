# ═══════════════════════════════════════════════════════════════════════════
# service_swin.py  —  NeuroPulse | Swin Transformer Microservice
#
# Run:  uvicorn service_swin:app --host 0.0.0.0 --port 8003
# Env:  requirements_cnn.txt  (torch, timm, facenet-pytorch, albumentations)
# ═══════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import math
import os
import tempfile
import shutil
import warnings

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI(title="NeuroPulse Swin Transformer Service", version="1.0")
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
# MODEL DEFINITIONS  — must match training notebook exactly
# ═════════════════════════════════════════════════════════════════════════════

class EfficientChannelAttention(nn.Module):
    """ECA-Net: Efficient Channel Attention (Wang et al., CVPR 2020)."""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        k = int(abs((math.log2(channels) + b) / gamma))
        k = k if k % 2 else k + 1          # ensure odd kernel
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() == 3:                    # [B, T, C]
            B, T, C = x.shape
            x = x.reshape(B * T, C)
        y = torch.sigmoid(self.conv(x.unsqueeze(1)).squeeze(1))
        x = x * y
        if len(orig_shape) == 3:
            x = x.reshape(orig_shape)
        return x


class SpatioTemporalSwinCNN(nn.Module):
    """
    Swin-Tiny backbone + ECA + on-the-fly DCT branch → BiLSTM → MHA → classifier.
    Matches model_swin.ipynb training configuration exactly.
    """

    def __init__(
        self,
        model_name:      str   = "swin_tiny_patch4_window7_224",
        pretrained:      bool  = True,
        hidden_dim:      int   = 192,
        lstm_hidden:     int   = 256,
        num_layers:      int   = 2,
        dropout:         float = 0.3,
        attention_heads: int   = 4,
        drop_path_rate:  float = 0.2,
        freeze_backbone: bool  = False,
    ):
        super().__init__()

        # Pre-compute 64×64 DCT matrix (registered as non-trainable buffer)
        dct_m = np.empty((64, 64))
        for k in range(64):
            for n in range(64):
                dct_m[k, n] = math.cos(math.pi * k * (2.0 * n + 1) / 128.0)
        dct_m[0, :] /= math.sqrt(2.0)
        dct_m       *= math.sqrt(2.0 / 64)
        self.register_buffer("dct_m", torch.from_numpy(dct_m).float())

        # ImageNet normalisation constants (for DCT de-normalisation)
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

        # Backbone — Swin-Tiny (768-dim output)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool="avg",
            drop_path_rate=drop_path_rate,
        )
        cnn_out_dim = self.backbone.num_features   # 768

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ECA on 768-dim spatial features
        self.channel_attention = EfficientChannelAttention(cnn_out_dim)

        # Spatial projection: 768 → hidden_dim*2 (384)
        self.input_proj = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # DCT frequency branch: 128 → hidden_dim (192)
        self.freq_encoder = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # pack_padded_sequence BiLSTM: 384 → 512
        self.temporal = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        temporal_out_dim      = lstm_hidden * 2   # 512
        self.temporal_dropout = nn.Dropout(p=dropout)

        # MHA on 512-dim LSTM output
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=temporal_out_dim,
            num_heads=attention_heads,
            dropout=dropout * 0.5,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(temporal_out_dim)

        # Fused: 512 + 192 = 704 → classifier
        fused_dim = temporal_out_dim + hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and "backbone" not in name:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for pname, param in module.named_parameters():
                    if "weight_ih" in pname:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in pname:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in pname:
                        nn.init.zeros_(param.data)
                        # Set forget-gate bias = 1 (LSTM best practice)
                        n = param.data.size(0)
                        param.data[n // 4: n // 2].fill_(1.0)

    def _rgb_to_dct_features(self, x: torch.Tensor) -> torch.Tensor:
        """On-the-fly 2-D DCT frequency features from ImageNet-normalised frames."""
        x_denorm = x * self.imagenet_std + self.imagenet_mean
        gray = (
            0.299 * x_denorm[:, 0]
            + 0.587 * x_denorm[:, 1]
            + 0.114 * x_denorm[:, 2]
        )
        # Downsample to 64×64
        down = F.interpolate(
            gray.unsqueeze(1), size=(64, 64),
            mode="bilinear", align_corners=False,
        ).squeeze(1)
        # 2-D DCT via pre-computed matrix
        dct_feat = torch.matmul(torch.matmul(self.dct_m, down), self.dct_m.t())
        dct_feat = torch.log(torch.abs(dct_feat) + 1e-6)
        # Extract 8×8 block statistics → 128-dim vector
        blocks = dct_feat.unfold(1, 8, 8).unfold(2, 8, 8)   # [B, 8, 8, 8, 8]
        means  = blocks.mean(dim=(3, 4)).reshape(x.size(0), 64)
        stds   = blocks.std(dim=(3, 4)).reshape(x.size(0), 64)
        return torch.cat([means, stds], dim=1)               # [B, 128]

    def forward(
        self,
        x:    torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, C, H, W = x.size()
        x_flat         = x.view(B * T, C, H, W)

        # ── Spatial branch: skip padded frames in backbone for memory efficiency ──
        if mask is not None:
            rmf     = mask.view(-1)                          # [B*T] bool
            rs      = self.backbone(x_flat[rmf])             # only valid frames
            sf      = torch.zeros(B * T, rs.shape[-1], device=x.device, dtype=rs.dtype)
            sf[rmf] = rs
            spatial = sf.view(B, T, -1)
        else:
            spatial = self.backbone(x_flat).view(B, T, -1)

        spatial = self.channel_attention(spatial)

        # ── Frequency branch (computed on ALL frames, including padded) ──
        freq = self.freq_encoder(
            self._rgb_to_dct_features(x_flat)
        ).view(B, T, -1)

        if mask is not None:
            fm          = mask.unsqueeze(-1).float()
            freq_pooled = (freq * fm).sum(1) / fm.sum(1).clamp(min=1)
        else:
            freq_pooled = freq.mean(1)

        # ── Temporal branch: pack_padded_sequence BiLSTM ──────────────────────
        projected = self.input_proj(spatial)

        if mask is not None:
            lengths = mask.sum(1).clamp(min=1).cpu().long()
            packed  = torch.nn.utils.rnn.pack_padded_sequence(
                projected, lengths, batch_first=True, enforce_sorted=False,
            )
            with torch.backends.cudnn.flags(enabled=False):
                packed_out, _ = self.temporal(packed)
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=T,
            )
        else:
            with torch.backends.cudnn.flags(enabled=False):
                lstm_out, _ = self.temporal(projected)

        lstm_out = self.temporal_dropout(lstm_out)

        # Multi-head attention with residual + LayerNorm
        kpm     = ~mask if mask is not None else None
        attn_out, _ = self.temporal_attention(
            lstm_out, lstm_out, lstm_out, key_padding_mask=kpm,
        )
        temporal_features = self.attn_norm(lstm_out + attn_out)

        # Masked average pooling
        if mask is not None:
            tm              = mask.unsqueeze(-1).float()
            temporal_pooled = (temporal_features * tm).sum(1) / tm.sum(1).clamp(min=1)
        else:
            temporal_pooled = temporal_features.mean(1)

        # ── Classifier on fused [temporal_pooled || freq_pooled] ─────────────
        fused = torch.cat([temporal_pooled, freq_pooled], dim=-1)
        return self.classifier(fused).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════

print("[Swin Service] Loading model...", flush=True)

_swin_path = os.path.join(BASE_DIR, "models", "swin_model.pth")
if not os.path.exists(_swin_path):
    raise FileNotFoundError(f"Swin weights not found: {_swin_path}")

_swin_model = SpatioTemporalSwinCNN(
    model_name="swin_tiny_patch4_window7_224",
    pretrained=False,
    hidden_dim=192,
    lstm_hidden=256,
    num_layers=2,
    attention_heads=4,
    dropout=0.3,
    drop_path_rate=0.2,
    freeze_backbone=False,
).to(DEVICE)

_swin_model.load_state_dict(
    torch.load(_swin_path, map_location=DEVICE, weights_only=True)
)
_swin_model.eval()
print(f"  Swin-Tiny loaded on {DEVICE}", flush=True)

# ── MTCNN with eye-alignment support ──────────────────────────────────────
from facenet_pytorch import MTCNN as _MTCNN  # noqa: E402

_mtcnn_224_align = _MTCNN(
    image_size=224, margin=20, min_face_size=60,
    thresholds=[0.6, 0.7, 0.7], factor=0.709,
    post_process=False, device=DEVICE, keep_all=False,
)
print("  MTCNN (224 + align) loaded", flush=True)
print("[Swin Service] Ready on port 8003.", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# FACE EXTRACTION HELPERS  (with eye-alignment — matches Swin training)
# ═════════════════════════════════════════════════════════════════════════════

def _center_crop(rgb_frame: np.ndarray, target: int) -> np.ndarray:
    h, w   = rgb_frame.shape[:2]
    s      = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return cv2.resize(rgb_frame[y0:y0 + s, x0:x0 + s], (target, target))


def _eye_align_rgb(rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Rotate face so that the eye line is horizontal (< 2° → skip)."""
    lm    = landmarks[0]
    dy    = lm[1][1] - lm[0][1]
    dx    = lm[1][0] - lm[0][0]
    angle = np.degrees(np.arctan2(dy, dx))
    if abs(angle) <= 2.0:
        return rgb
    eye_center = (
        float((lm[0][0] + lm[1][0]) / 2),
        float((lm[0][1] + lm[1][1]) / 2),
    )
    h, w = rgb.shape[:2]
    M    = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    return cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR)


@torch.no_grad()
def _extract_face_aligned(
    mtcnn_detector,
    rgb_frame: np.ndarray,
    img_size:  int,
) -> np.ndarray:
    """Extract face with optional eye alignment; fall back to centre-crop."""
    from PIL import Image as _PIL_Image
    pil_img = _PIL_Image.fromarray(rgb_frame)
    try:
        boxes, probs, lms = mtcnn_detector.detect(pil_img, landmarks=True)
        if (
            lms is not None
            and len(lms) > 0
            and probs[0] is not None
            and probs[0] >= 0.9
        ):
            rgb_frame = _eye_align_rgb(rgb_frame, lms)
            pil_img   = _PIL_Image.fromarray(rgb_frame)

        face_tensor = mtcnn_detector(pil_img)
        if face_tensor is not None:
            face_np  = face_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
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
    """Uniformly sample *n_frames* aligned face crops from *video_path*."""
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
        face = _extract_face_aligned(_mtcnn_224_align, rgb, img_size)
        faces.append(face)

    cap.release()
    return np.array(faces, dtype=np.uint8)


# ═════════════════════════════════════════════════════════════════════════════
# TTA INFERENCE  (6-pass, matching Swin training)
# ═════════════════════════════════════════════════════════════════════════════

def _build_transform(img_size: int, mean: list, std: list, n_frames: int) -> A.Compose:
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
            elif tta_idx == 5:
                h, w   = aug.shape[:2]
                ch, cw = int(h * 0.93), int(w * 0.93)
                y0, x0 = (h - ch) // 2, (w - cw) // 2
                aug    = cv2.resize(aug[y0:y0 + ch, x0:x0 + cw], (w, h))
            frames.append(aug)

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

        fstack = torch.stack(tensor_list).unsqueeze(0).to(DEVICE)
        mask   = torch.ones(1, len(frames), dtype=torch.bool, device=DEVICE)

        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
            logit_out = _swin_model(fstack, mask)

        prob = float(torch.sigmoid(logit_out.squeeze()).item())
        all_probs.append(prob)

    return float(np.mean(all_probs))


def predict_swin(video_path: str) -> float:
    """Full Swin-Tiny inference: face extraction + alignment → 6-pass TTA."""
    faces = _extract_faces_from_video(video_path, n_frames=16, img_size=224)
    return _run_cnn_tta(
        _swin_model, faces,
        img_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD,
        n_tta_passes=6,
    )


# ═════════════════════════════════════════════════════════════════════════════
# FASTAPI ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok", "service": "swin", "device": str(DEVICE), "port": 8003}


@app.post("/predict")
async def predict_endpoint(video: UploadFile = File(...)):
    ext = (video.filename or "upload.mp4").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: .{ext}")

    tmp_dir  = tempfile.mkdtemp(prefix="neuropulse_swin_")
    tmp_path = os.path.join(tmp_dir, f"video.{ext}")
    try:
        content = await video.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file received.")
        with open(tmp_path, "wb") as fh:
            fh.write(content)

        prob = predict_swin(tmp_path)
        return {"probability": round(prob, 4), "model": "swin", "status": "ok"}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Swin inference error: {exc}") from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service_swin:app", host="0.0.0.0", port=8003, reload=False)
