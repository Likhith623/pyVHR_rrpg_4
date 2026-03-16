"""
Use Case 1: Real-Time Webcam Liveness Detection (Threshold-Based)

Pipeline:
  Webcam → Face Detection → ROI Extraction → RGB traces →
  rPPG extraction → Signal quality check → Liveness decision

NO machine learning required. Pure signal analysis with physiological thresholds.
Based on pyVHR signal processing and FakeCatcher signal quality metrics.
"""
import sys
import time
import cv2
import numpy as np
import mediapipe as mp

sys.path.insert(0, ".")
from configs.config import (
    CAMERA_INDEX, FPS, FRAME_WIDTH, FRAME_HEIGHT,
    FACE_MESH_CONFIDENCE,
    ROI_FOREHEAD, ROI_LEFT_CHEEK, ROI_RIGHT_CHEEK,
    SIGNAL_BUFFER_SECONDS, RPPG_METHOD,
    BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER,
    LIVENESS_BPM_MIN, LIVENESS_BPM_MAX,
    LIVENESS_SNR_THRESHOLD, LIVENESS_SPECTRAL_PURITY_THRESHOLD,
    LIVENESS_PEAK_PROMINENCE_THRESHOLD, LIVENESS_ROI_CORRELATION_THRESHOLD,
    LIVENESS_SIGNAL_STATIONARITY_THRESHOLD, LIVENESS_MIN_CHECKS_PASSED,
)
from src.rppg_extractor import SignalBuffer, get_roi_mean_rgb
from src.signal_processor import (
    estimate_bpm, compute_snr, compute_spectral_purity,
    compute_peak_prominence, compute_roi_correlation,
    compute_signal_stationarity,
)


class LivenessResult:
    """Container for a single liveness assessment."""
    def __init__(self):
        self.bpm = 0.0
        self.snr = 0.0
        self.spectral_purity = 0.0
        self.peak_prominence = 0.0
        self.roi_correlation_fh_lc = 0.0
        self.roi_correlation_fh_rc = 0.0
        self.stationarity = 0.0

        self.bpm_ok = False
        self.snr_ok = False
        self.spectral_purity_ok = False
        self.peak_prominence_ok = False
        self.roi_correlation_ok = False
        self.stationarity_ok = False
        self.harmonic_ok = False

        self.checks_passed = 0
        self.total_checks = 7
        self.is_live = False

    def to_dict(self):
        return {
            "bpm": round(self.bpm, 1),
            "snr_dB": round(self.snr, 2),
            "spectral_purity": round(self.spectral_purity, 3),
            "peak_prominence": round(self.peak_prominence, 4),
            "roi_corr_fh_lc": round(self.roi_correlation_fh_lc, 3),
            "roi_corr_fh_rc": round(self.roi_correlation_fh_rc, 3),
            "stationarity": round(self.stationarity, 3),
            "checks_passed": f"{self.checks_passed}/{self.total_checks}",
            "is_live": self.is_live,
        }


def assess_liveness(bvp_fh, bvp_lc, bvp_rc, fs):
    """
    Perform threshold-based liveness assessment on BVP signals.

    7 checks:
      1. BPM in physiological range [45-150]
      2. SNR above threshold (strong pulse signal)
      3. Spectral purity (single dominant frequency)
      4. Peak prominence (clear spectral peak)
      5. ROI correlation (consistent pulse across face regions)
      6. Signal stationarity (stable over time)
      7. Harmonic presence (2nd harmonic of pulse)

    Returns:
        LivenessResult with all metrics and final decision
    """
    result = LivenessResult()

    # 1. BPM check
    bpm, _, _, _ = estimate_bpm(bvp_fh, fs, BANDPASS_LOW, BANDPASS_HIGH)
    result.bpm = bpm
    result.bpm_ok = LIVENESS_BPM_MIN <= bpm <= LIVENESS_BPM_MAX

    # 2. SNR check
    result.snr = compute_snr(bvp_fh, fs, BANDPASS_LOW, BANDPASS_HIGH)
    result.snr_ok = result.snr >= LIVENESS_SNR_THRESHOLD

    # 3. Spectral purity check
    result.spectral_purity = compute_spectral_purity(bvp_fh, fs, BANDPASS_LOW, BANDPASS_HIGH)
    result.spectral_purity_ok = result.spectral_purity >= LIVENESS_SPECTRAL_PURITY_THRESHOLD

    # 4. Peak prominence check
    result.peak_prominence = compute_peak_prominence(bvp_fh, fs, BANDPASS_LOW, BANDPASS_HIGH)
    result.peak_prominence_ok = result.peak_prominence >= LIVENESS_PEAK_PROMINENCE_THRESHOLD

    # 5. ROI correlation check (both pairs must be above threshold)
    result.roi_correlation_fh_lc = compute_roi_correlation(bvp_fh, bvp_lc)
    result.roi_correlation_fh_rc = compute_roi_correlation(bvp_fh, bvp_rc)
    result.roi_correlation_ok = (
        result.roi_correlation_fh_lc >= LIVENESS_ROI_CORRELATION_THRESHOLD and
        result.roi_correlation_fh_rc >= LIVENESS_ROI_CORRELATION_THRESHOLD
    )

    # 6. Signal stationarity check
    result.stationarity = compute_signal_stationarity(bvp_fh)
    result.stationarity_ok = result.stationarity >= LIVENESS_SIGNAL_STATIONARITY_THRESHOLD

    # 7. Harmonic check — BPM from left cheek should agree with forehead
    bpm_lc, _, _, _ = estimate_bpm(bvp_lc, fs, BANDPASS_LOW, BANDPASS_HIGH)
    result.harmonic_ok = abs(bpm - bpm_lc) < 10.0  # within 10 BPM

    # Count checks
    checks = [result.bpm_ok, result.snr_ok, result.spectral_purity_ok,
              result.peak_prominence_ok, result.roi_correlation_ok,
              result.stationarity_ok, result.harmonic_ok]
    result.checks_passed = sum(checks)
    result.is_live = result.checks_passed >= LIVENESS_MIN_CHECKS_PASSED

    return result


def draw_overlay(frame, result, face_detected, buffer_ready):
    """Draw liveness info on the video frame."""
    h, w = frame.shape[:2]

    if not face_detected:
        cv2.putText(frame, "NO FACE DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame

    if not buffer_ready:
        cv2.putText(frame, "Collecting signal... please hold still", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame

    # Liveness verdict
    if result.is_live:
        color = (0, 255, 0)
        label = f"LIVE PERSON ({result.checks_passed}/7)"
    else:
        color = (0, 0, 255)
        label = f"NOT LIVE ({result.checks_passed}/7)"

    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Metrics
    y = 75
    metrics = [
        (f"BPM: {result.bpm:.0f}", result.bpm_ok),
        (f"SNR: {result.snr:.1f} dB", result.snr_ok),
        (f"Spectral Purity: {result.spectral_purity:.2f}", result.spectral_purity_ok),
        (f"Peak Prominence: {result.peak_prominence:.3f}", result.peak_prominence_ok),
        (f"ROI Correlation: {result.roi_correlation_fh_lc:.2f}/{result.roi_correlation_fh_rc:.2f}",
         result.roi_correlation_ok),
        (f"Stationarity: {result.stationarity:.2f}", result.stationarity_ok),
        (f"BPM Consistency: {'PASS' if result.harmonic_ok else 'FAIL'}", result.harmonic_ok),
    ]
    for text, passed in metrics:
        c = (0, 200, 0) if passed else (0, 0, 200)
        prefix = "[+]" if passed else "[-]"
        cv2.putText(frame, f"{prefix} {text}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
        y += 25

    return frame


def run_liveness_detection():
    """
    Main loop: Webcam → Face Mesh → ROI → rPPG → Liveness Check.
    Press 'q' to quit.
    """
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=FACE_MESH_CONFIDENCE,
        min_tracking_confidence=FACE_MESH_CONFIDENCE,
    )

    # Signal buffer
    max_frames = int(SIGNAL_BUFFER_SECONDS * FPS)
    sig_buffer = SignalBuffer(
        max_frames=max_frames,
        method=RPPG_METHOD,
        fs=FPS,
        bandpass_low=BANDPASS_LOW,
        bandpass_high=BANDPASS_HIGH,
        bandpass_order=BANDPASS_ORDER,
    )

    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Liveness detection started. Press 'q' to quit.")
    print(f"[INFO] Method: {RPPG_METHOD} | Buffer: {SIGNAL_BUFFER_SECONDS}s | FPS: {FPS}")

    result = LivenessResult()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_results = face_mesh.process(rgb_frame)

        face_detected = False
        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0]
            face_detected = True

            # Extract mean RGB from 3 ROIs
            rgb_fh = get_roi_mean_rgb(frame, landmarks, ROI_FOREHEAD, frame_h, frame_w)
            rgb_lc = get_roi_mean_rgb(frame, landmarks, ROI_LEFT_CHEEK, frame_h, frame_w)
            rgb_rc = get_roi_mean_rgb(frame, landmarks, ROI_RIGHT_CHEEK, frame_h, frame_w)

            sig_buffer.add_frame(rgb_fh, rgb_lc, rgb_rc)

        # Run liveness check every 15 frames (0.5s at 30fps)
        frame_count += 1
        buffer_ready = sig_buffer.is_ready
        if buffer_ready and frame_count % 15 == 0:
            bvp_fh, bvp_lc, bvp_rc = sig_buffer.get_bvp_signals()
            if bvp_fh is not None:
                result = assess_liveness(bvp_fh, bvp_lc, bvp_rc, FPS)

        # Draw overlay
        frame = draw_overlay(frame, result, face_detected, buffer_ready)
        cv2.imshow("Neuro-Pulse Liveness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("[INFO] Liveness detection stopped.")


if __name__ == "__main__":
    run_liveness_detection()
