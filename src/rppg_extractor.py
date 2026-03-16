"""
rPPG signal extraction from video frames using GREEN, CHROM, and POS methods.
Uses MediaPipe Face Mesh for ROI detection.
Implements bandpass filtering and signal detrending (pyVHR-aligned).
"""
import numpy as np
from scipy.signal import butter, filtfilt
from collections import deque


# ─── rPPG Extraction Methods ──────────────────────────────────────

def extract_green(rgb_mean):
    """
    GREEN channel method (Verkruysse et al., 2008).
    Simply uses the green channel as BVP proxy.
    Args:
        rgb_mean: (N, 3) array of mean R, G, B per frame
    Returns:
        bvp: (N,) green channel trace
    """
    return rgb_mean[:, 1].copy()


def extract_chrom(rgb_mean, fs=30):
    """
    CHROM method (De Haan & Jeanne, 2013).
    Chrominance-based rPPG using linear combination of color channels.
    Args:
        rgb_mean: (N, 3) array of mean R, G, B per frame
        fs: sampling rate
    Returns:
        bvp: (N,) CHROM signal
    """
    r = rgb_mean[:, 0]
    g = rgb_mean[:, 1]
    b = rgb_mean[:, 2]

    # Normalize by mean
    r_norm = r / (np.mean(r) + 1e-8)
    g_norm = g / (np.mean(g) + 1e-8)
    b_norm = b / (np.mean(b) + 1e-8)

    # CHROM signal
    xs = 3 * r_norm - 2 * g_norm
    ys = 1.5 * r_norm + g_norm - 1.5 * b_norm

    # Windowed standard deviation ratio
    win_size = int(1.6 * fs)
    if win_size < 2:
        win_size = 2

    bvp = np.zeros(len(r))
    for i in range(len(r)):
        start = max(0, i - win_size // 2)
        end = min(len(r), i + win_size // 2)
        std_xs = np.std(xs[start:end])
        std_ys = np.std(ys[start:end])
        alpha = std_xs / (std_ys + 1e-8)
        bvp[i] = xs[i] - alpha * ys[i]

    return bvp


def extract_pos(rgb_mean, fs=30):
    """
    POS method (Wang et al., 2017).
    Plane-Orthogonal-to-Skin.
    Args:
        rgb_mean: (N, 3) array of mean R, G, B per frame
        fs: sampling rate
    Returns:
        bvp: (N,) POS signal
    """
    n = rgb_mean.shape[0]
    win_size = int(1.6 * fs)
    if win_size < 2:
        win_size = 2

    bvp = np.zeros(n)

    for i in range(n):
        start = max(0, i - win_size + 1)
        end = i + 1
        if end - start < 3:
            continue

        window = rgb_mean[start:end, :]
        # Temporal normalization
        mean_rgb = np.mean(window, axis=0)
        mean_rgb[mean_rgb < 1e-8] = 1e-8
        cn = window / mean_rgb

        # Projection
        s1 = cn[:, 1] - cn[:, 2]
        s2 = cn[:, 1] + cn[:, 2] - 2.0 * cn[:, 0]

        std_s1 = np.std(s1)
        std_s2 = np.std(s2)
        alpha = std_s1 / (std_s2 + 1e-8)

        pulse = s1 + alpha * s2
        bvp[i] = pulse[-1]

    return bvp


RPPG_METHODS = {
    "GREEN": extract_green,
    "CHROM": extract_chrom,
    "POS": extract_pos,
}


# ─── Signal Conditioning ─────────────────────────────────────────

def bandpass_filter(signal, fs, lowcut=0.7, highcut=3.5, order=4):
    """
    Butterworth bandpass filter for isolating pulse frequency band.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    low = max(low, 0.001)
    high = min(high, 0.999)
    if low >= high:
        return signal
    b, a = butter(order, [low, high], btype='band')
    if len(signal) < 3 * max(len(a), len(b)):
        return signal
    return filtfilt(b, a, signal)


def detrend_signal(signal, lam=300):
    """
    Smoothness-priors detrending (Tarvainen et al., 2002).
    Removes slow drift while preserving pulse signal.
    """
    n = len(signal)
    if n < 5:
        return signal
    identity = np.eye(n)
    d2 = np.zeros((n - 2, n))
    for i in range(n - 2):
        d2[i, i] = 1
        d2[i, i + 1] = -2
        d2[i, i + 2] = 1
    inv = np.linalg.solve(identity + lam ** 2 * d2.T @ d2, signal)
    return signal - inv


# ─── ROI Extraction via MediaPipe ────────────────────────────────

def get_roi_mean_rgb(frame, landmarks, roi_indices, frame_h, frame_w):
    """
    Extract mean RGB from a facial ROI defined by landmark indices.
    Args:
        frame: BGR image (H, W, 3)
        landmarks: MediaPipe face landmarks
        roi_indices: list of landmark indices defining the ROI
        frame_h, frame_w: frame dimensions
    Returns:
        mean_rgb: (3,) mean R, G, B values, or None if ROI is invalid
    """
    points = []
    for idx in roi_indices:
        lm = landmarks.landmark[idx]
        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        points.append((x, y))

    if len(points) < 3:
        return None

    points = np.array(points, dtype=np.int32)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame_w, x_max)
    y_max = min(frame_h, y_max)

    if x_max <= x_min or y_max <= y_min:
        return None

    roi = frame[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None

    # BGR to RGB, then mean
    mean_bgr = np.mean(roi.reshape(-1, 3), axis=0)
    mean_rgb = mean_bgr[::-1]  # BGR -> RGB
    return mean_rgb.astype(np.float64)


class SignalBuffer:
    """
    Rolling buffer for accumulating RGB traces and extracting BVP signals.
    """

    def __init__(self, max_frames, method="GREEN", fs=30,
                 bandpass_low=0.7, bandpass_high=3.5, bandpass_order=4):
        self.max_frames = max_frames
        self.method = method
        self.fs = fs
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.bandpass_order = bandpass_order

        self.rgb_forehead = deque(maxlen=max_frames)
        self.rgb_left_cheek = deque(maxlen=max_frames)
        self.rgb_right_cheek = deque(maxlen=max_frames)

    def add_frame(self, rgb_fh, rgb_lc, rgb_rc):
        """Add one frame's mean RGB values for each ROI."""
        if rgb_fh is not None:
            self.rgb_forehead.append(rgb_fh)
        if rgb_lc is not None:
            self.rgb_left_cheek.append(rgb_lc)
        if rgb_rc is not None:
            self.rgb_right_cheek.append(rgb_rc)

    @property
    def is_ready(self):
        """Need at least 2 seconds of data."""
        min_frames = int(2.0 * self.fs)
        return (len(self.rgb_forehead) >= min_frames and
                len(self.rgb_left_cheek) >= min_frames and
                len(self.rgb_right_cheek) >= min_frames)

    def get_bvp_signals(self):
        """
        Extract BVP signals from all three ROIs.
        Returns:
            bvp_fh, bvp_lc, bvp_rc: filtered BVP signals
        """
        if not self.is_ready:
            return None, None, None

        extract_fn = RPPG_METHODS.get(self.method, extract_green)

        signals = []
        for buf in [self.rgb_forehead, self.rgb_left_cheek, self.rgb_right_cheek]:
            rgb_array = np.array(list(buf))
            raw_bvp = extract_fn(rgb_array) if self.method != "GREEN" else extract_fn(rgb_array)
            # Detrend
            bvp = detrend_signal(raw_bvp)
            # Bandpass filter
            bvp = bandpass_filter(bvp, self.fs, self.bandpass_low,
                                  self.bandpass_high, self.bandpass_order)
            signals.append(bvp)

        return signals[0], signals[1], signals[2]

    def reset(self):
        self.rgb_forehead.clear()
        self.rgb_left_cheek.clear()
        self.rgb_right_cheek.clear()
