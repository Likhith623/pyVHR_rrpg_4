"""
rPPG signal extraction methods: GREEN, CHROM, POS.
Implements the core algorithms for extracting blood volume pulse (BVP)
signals from RGB traces captured from facial skin ROIs.
"""
import numpy as np
from scipy.signal import butter, filtfilt, detrend


def _bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """Apply zero-phase Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def _detrend_signal(signal, lam=300):
    """Smoothness priors detrending (Tarvainen et al. 2002)."""
    n = len(signal)
    if n < 4:
        return signal - np.mean(signal)
    I = np.eye(n)
    D2 = np.zeros((n - 2, n))
    for i in range(n - 2):
        D2[i, i] = 1
        D2[i, i + 1] = -2
        D2[i, i + 2] = 1
    inv_mat = np.linalg.solve(I + lam ** 2 * D2.T @ D2, np.eye(n))
    trend = inv_mat @ signal
    return signal - trend


def extract_green(rgb_traces, fs, bandpass_low=0.7, bandpass_high=3.5,
                  detrend_lambda=300):
    """
    GREEN channel method (Verkruysse et al., 2008).
    Simplest rPPG — the green channel contains the strongest PPG signal.

    Args:
        rgb_traces: (N, 3) array of [R, G, B] mean values per frame
        fs: sampling frequency (fps)
    Returns:
        bvp: band-pass filtered BVP signal
    """
    green = rgb_traces[:, 1].copy()
    green = _detrend_signal(green, detrend_lambda)
    green = (green - np.mean(green)) / (np.std(green) + 1e-8)
    bvp = _bandpass_filter(green, bandpass_low, bandpass_high, fs)
    return bvp


def extract_chrom(rgb_traces, fs, bandpass_low=0.7, bandpass_high=3.5,
                  detrend_lambda=300):
    """
    CHROM method (De Haan & Jeanne, 2013).
    Chrominance-based method — combines R, G, B channels to reduce
    motion artifacts and illumination noise.

    Args:
        rgb_traces: (N, 3) array
        fs: sampling frequency
    Returns:
        bvp: BVP signal
    """
    r = rgb_traces[:, 0].astype(np.float64)
    g = rgb_traces[:, 1].astype(np.float64)
    b = rgb_traces[:, 2].astype(np.float64)

    # Normalize by temporal mean
    r = r / (np.mean(r) + 1e-8)
    g = g / (np.mean(g) + 1e-8)
    b = b / (np.mean(b) + 1e-8)

    # CHROM projection
    x_s = 3 * r - 2 * g
    y_s = 1.5 * r + g - 1.5 * b

    # Bandpass first
    x_f = _bandpass_filter(x_s, bandpass_low, bandpass_high, fs)
    y_f = _bandpass_filter(y_s, bandpass_low, bandpass_high, fs)

    alpha = np.std(x_f) / (np.std(y_f) + 1e-8)
    bvp = x_f - alpha * y_f

    bvp = _detrend_signal(bvp, detrend_lambda)
    bvp = (bvp - np.mean(bvp)) / (np.std(bvp) + 1e-8)
    return bvp


def extract_pos(rgb_traces, fs, bandpass_low=0.7, bandpass_high=3.5,
                detrend_lambda=300, window_size=None):
    """
    POS method (Wang et al., 2017).
    Plane-Orthogonal-to-Skin — projects RGB signals onto a plane
    orthogonal to the skin tone direction.

    Args:
        rgb_traces: (N, 3) array
        fs: sampling frequency
        window_size: temporal window (frames). Default = 1.6 * fs
    Returns:
        bvp: BVP signal
    """
    if window_size is None:
        window_size = int(1.6 * fs)

    n = rgb_traces.shape[0]
    bvp = np.zeros(n)

    for i in range(n):
        start = max(0, i - window_size + 1)
        end = i + 1
        if end - start < 3:
            continue

        c = rgb_traces[start:end].astype(np.float64)
        mean_c = np.mean(c, axis=0) + 1e-8
        cn = c / mean_c  # normalize

        # Projection
        s1 = cn[:, 1] - cn[:, 2]  # G - B
        s2 = cn[:, 1] + cn[:, 2] - 2 * cn[:, 0]  # G + B - 2R

        alpha = np.std(s1) / (np.std(s2) + 1e-8)
        h = s1 + alpha * s2

        bvp[start:end] += (h - np.mean(h))

    bvp = _detrend_signal(bvp, detrend_lambda)
    bvp = (bvp - np.mean(bvp)) / (np.std(bvp) + 1e-8)
    bvp = _bandpass_filter(bvp, bandpass_low, bandpass_high, fs)
    return bvp


RPPG_METHODS = {
    "GREEN": extract_green,
    "CHROM": extract_chrom,
    "POS": extract_pos,
}


def extract_rppg(rgb_traces, fs, method="GREEN", **kwargs):
    """
    Extract rPPG BVP signal using the specified method.

    Args:
        rgb_traces: (N, 3) array of RGB mean values
        fs: sampling rate
        method: one of 'GREEN', 'CHROM', 'POS'
    Returns:
        bvp: 1D BVP signal array
    """
    if method not in RPPG_METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(RPPG_METHODS.keys())}")
    return RPPG_METHODS[method](rgb_traces, fs, **kwargs)
