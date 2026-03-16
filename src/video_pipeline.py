"""
Use Case 2: Video Deepfake Detection — Feature Extraction Pipeline

Processes video files (real/fake), extracts 35 rPPG features per video,
and saves the feature matrix + labels as .npy / .csv for ML training.

Designed for Kaggle dataset at: /kaggle/input/face-dataset-dip/{real, deepfake}
"""
import os
import sys
import glob
import time
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

sys.path.insert(0, ".")
from configs.config import (
    ROI_FOREHEAD, ROI_LEFT_CHEEK, ROI_RIGHT_CHEEK,
    BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER,
    FEATURE_WINDOW_SECONDS, FEATURE_STRIDE_SECONDS,
    KAGGLE_REAL_DIR, KAGGLE_FAKE_DIR,
)
from src.rppg_extractor import (
    SignalBuffer, get_roi_mean_rgb, RPPG_METHODS,
    bandpass_filter, detrend_signal,
)
from src.feature_extractor import extract_features, FEATURE_NAMES


# Module-level shared FaceMesh — reused across all videos
_FACE_MESH = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def extract_features_from_video(video_path, method="GREEN", max_frames=300):
    """
    Process a single video file and extract rPPG features.

    Pipeline:
      1. Read video frames
      2. Detect face with MediaPipe Face Mesh
      3. Extract mean RGB from 3 ROIs per frame
      4. Apply rPPG method (GREEN/CHROM/POS)
      5. Bandpass filter + detrend
      6. Extract 35-dim feature vector

    Args:
        video_path: path to video file
        method: rPPG method name
        max_frames: cap on frames to process (300 = ~10s at 30fps)

    Returns:
        features: (35,) numpy array or None on failure
        metadata: dict with video info
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, {"error": f"Cannot open {video_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rgb_forehead = []
    rgb_left_cheek = []
    rgb_right_cheek = []
    frames_with_face = 0
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        frame_h, frame_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _FACE_MESH.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            rgb_fh = get_roi_mean_rgb(frame, landmarks, ROI_FOREHEAD, frame_h, frame_w)
            rgb_lc = get_roi_mean_rgb(frame, landmarks, ROI_LEFT_CHEEK, frame_h, frame_w)
            rgb_rc = get_roi_mean_rgb(frame, landmarks, ROI_RIGHT_CHEEK, frame_h, frame_w)

            if rgb_fh is not None and rgb_lc is not None and rgb_rc is not None:
                rgb_forehead.append(rgb_fh)
                rgb_left_cheek.append(rgb_lc)
                rgb_right_cheek.append(rgb_rc)
                frames_with_face += 1

    cap.release()

    metadata = {
        "video": os.path.basename(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "processed_frames": frame_count,
        "face_frames": frames_with_face,
    }

    # Need at least 2 seconds of face data
    min_frames = int(2.0 * fps)
    if frames_with_face < min_frames:
        metadata["error"] = f"Insufficient face frames: {frames_with_face}/{min_frames}"
        return None, metadata

    # Convert to arrays
    rgb_fh_arr = np.array(rgb_forehead)
    rgb_lc_arr = np.array(rgb_left_cheek)
    rgb_rc_arr = np.array(rgb_right_cheek)

    # Extract rPPG using chosen method
    extract_fn = RPPG_METHODS.get(method, RPPG_METHODS["GREEN"])
    bvp_fh = bandpass_filter(detrend_signal(extract_fn(rgb_fh_arr)), fps, BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER)
    bvp_lc = bandpass_filter(detrend_signal(extract_fn(rgb_lc_arr)), fps, BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER)
    bvp_rc = bandpass_filter(detrend_signal(extract_fn(rgb_rc_arr)), fps, BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER)

    # Extract 35 features
    features, _ = extract_features(bvp_fh, bvp_lc, bvp_rc, fps, BANDPASS_LOW, BANDPASS_HIGH)

    return features, metadata


def process_dataset(real_dir=None, fake_dir=None, method="GREEN",
                    max_frames=300, output_dir="./output"):
    """
    Process entire dataset: extract features from all real and fake videos.

    Args:
        real_dir: directory of real videos
        fake_dir: directory of deepfake videos
        method: rPPG method
        max_frames: max frames per video
        output_dir: where to save feature files

    Saves:
        features.npy  — (N, 35) feature matrix
        labels.npy    — (N,) binary labels (0=real, 1=fake)
        features.csv  — same data with column names for inspection
        metadata.csv  — processing metadata per video
        raw_signals/  — raw BVP signals per video (optional)
    """
    real_dir = real_dir or KAGGLE_REAL_DIR
    fake_dir = fake_dir or KAGGLE_FAKE_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Gather all video files
    video_exts = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]

    real_videos = []
    for ext in video_exts:
        real_videos.extend(glob.glob(os.path.join(real_dir, ext)))
        real_videos.extend(glob.glob(os.path.join(real_dir, ext.upper())))

    fake_videos = []
    for ext in video_exts:
        fake_videos.extend(glob.glob(os.path.join(fake_dir, ext)))
        fake_videos.extend(glob.glob(os.path.join(fake_dir, ext.upper())))

    print(f"[INFO] Found {len(real_videos)} real videos, {len(fake_videos)} fake videos")

    if len(real_videos) == 0 and len(fake_videos) == 0:
        print("[ERROR] No videos found. Check dataset paths:")
        print(f"  Real: {real_dir}")
        print(f"  Fake: {fake_dir}")
        return

    all_features = []
    all_labels = []
    all_metadata = []

    # Process real videos (label=0)
    print("\n--- Processing REAL videos ---")
    for vpath in tqdm(real_videos, desc="Real"):
        features, meta = extract_features_from_video(vpath, method, max_frames)
        meta["label"] = 0
        meta["class"] = "real"
        all_metadata.append(meta)
        if features is not None:
            all_features.append(features)
            all_labels.append(0)

    # Process fake videos (label=1)
    print("\n--- Processing FAKE videos ---")
    for vpath in tqdm(fake_videos, desc="Fake"):
        features, meta = extract_features_from_video(vpath, method, max_frames)
        meta["label"] = 1
        meta["class"] = "deepfake"
        all_metadata.append(meta)
        if features is not None:
            all_features.append(features)
            all_labels.append(1)

    if len(all_features) == 0:
        print("[ERROR] No features extracted. Check video files and face detection.")
        return

    # Convert to arrays
    X = np.array(all_features, dtype=np.float64)
    y = np.array(all_labels, dtype=np.int64)

    # Save as .npy
    np.save(os.path.join(output_dir, "features.npy"), X)
    np.save(os.path.join(output_dir, "labels.npy"), y)

    # Save as .csv with feature names
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = y
    df.to_csv(os.path.join(output_dir, "features.csv"), index=False)

    # Save metadata
    meta_df = pd.DataFrame(all_metadata)
    meta_df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

    print(f"\n[DONE] Extracted features from {len(all_features)} videos")
    print(f"  Real: {np.sum(y == 0)} | Fake: {np.sum(y == 1)}")
    print(f"  Feature shape: {X.shape}")
    print(f"  Saved to: {output_dir}/")


if __name__ == "__main__":
    process_dataset()
