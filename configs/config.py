"""
Configuration for Neuro-Pulse rPPG system.
"""

# ─── Video / Camera Settings ───────────────────────────────────────
CAMERA_INDEX = 0
FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ─── Face Detection ────────────────────────────────────────────────
FACE_DETECTION_CONFIDENCE = 0.5
FACE_MESH_CONFIDENCE = 0.5

# ─── ROI Regions (MediaPipe Face Mesh landmark indices) ────────────
ROI_FOREHEAD = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
                150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,
                67, 109]
ROI_LEFT_CHEEK = [187, 123, 116, 117, 118, 119, 120, 121, 128, 245,
                  193, 55, 65, 52, 53]
ROI_RIGHT_CHEEK = [411, 352, 345, 346, 347, 348, 349, 350, 357, 465,
                   417, 285, 295, 282, 283]

# ─── rPPG Signal Processing ───────────────────────────────────────
SIGNAL_BUFFER_SECONDS = 10
RPPG_METHOD = "GREEN"  # GREEN, CHROM, POS
BANDPASS_LOW = 0.7      # Hz (~42 bpm)
BANDPASS_HIGH = 3.5     # Hz (~210 bpm)
BANDPASS_ORDER = 4
DETREND_LAMBDA = 300

# ─── Welch PSD ─────────────────────────────────────────────────────
WELCH_NPERSEG = 256
WELCH_NOVERLAP = 128
WELCH_NFFT = 1024

# ─── Liveness Thresholds ──────────────────────────────────────────
LIVENESS_BPM_MIN = 45.0
LIVENESS_BPM_MAX = 150.0
LIVENESS_SNR_THRESHOLD = 3.0        # dB
LIVENESS_SPECTRAL_PURITY_THRESHOLD = 0.35
LIVENESS_PEAK_PROMINENCE_THRESHOLD = 0.15
LIVENESS_ROI_CORRELATION_THRESHOLD = 0.6
LIVENESS_SIGNAL_STATIONARITY_THRESHOLD = 0.7
LIVENESS_MIN_CHECKS_PASSED = 5      # out of 7

# ─── Feature Extraction (Deepfake Detection) ──────────────────────
FEATURE_WINDOW_SECONDS = 10
FEATURE_STRIDE_SECONDS = 5

# ─── Kaggle Dataset Paths ─────────────────────────────────────────
KAGGLE_DATASET_ROOT = "/kaggle/input/datasets/likhithvasireddy/deepfake-video-dataset-dip/content/drive/MyDrive/face_dataset_dip"
KAGGLE_REAL_DIR = f"{KAGGLE_DATASET_ROOT}/real_videos"
KAGGLE_FAKE_DIR = f"{KAGGLE_DATASET_ROOT}/deepfake_videos"
