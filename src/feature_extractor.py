"""
Comprehensive rPPG feature extraction for deepfake detection.
Extracts 35 features from BVP signals across multiple ROIs,
aligned with FakeCatcher (SNR, PSD, MAD, SD, PCC) feature set.

Feature categories:
  1. Spectral features (per ROI): SNR, spectral purity, peak prominence,
     dominant freq, harmonic ratio, spectral entropy, spectral centroid
  2. Temporal features (per ROI): MAD, SD, zero-crossing rate, kurtosis, skewness
  3. Cross-ROI features: Pearson correlation, coherence, phase difference
  4. Signal quality: stationarity, BPM consistency
"""
import numpy as np
from scipy.signal import coherence, welch
from scipy.stats import kurtosis, skew, pearsonr

from src.signal_processor import (
    compute_welch_psd,
    estimate_bpm,
    compute_snr,
    compute_spectral_purity,
    compute_peak_prominence,
    compute_roi_correlation,
    compute_signal_stationarity,
    compute_harmonic_ratio,
)


FEATURE_NAMES = [
    # ─── Forehead spectral (7) ───
    "fh_snr", "fh_spectral_purity", "fh_peak_prominence",
    "fh_dominant_freq", "fh_harmonic_ratio", "fh_spectral_entropy",
    "fh_spectral_centroid",
    # ─── Forehead temporal (5) ───
    "fh_mad", "fh_std", "fh_zcr", "fh_kurtosis", "fh_skewness",
    # ─── Left cheek spectral (7) ───
    "lc_snr", "lc_spectral_purity", "lc_peak_prominence",
    "lc_dominant_freq", "lc_harmonic_ratio", "lc_spectral_entropy",
    "lc_spectral_centroid",
    # ─── Left cheek temporal (5) ───
    "lc_mad", "lc_std", "lc_zcr", "lc_kurtosis", "lc_skewness",
    # ─── Cross-ROI features (8) ───
    "corr_fh_lc", "corr_fh_rc", "corr_lc_rc",
    "coherence_fh_lc", "coherence_fh_rc", "coherence_lc_rc",
    "phase_diff_fh_lc", "phase_diff_fh_rc",
    # ─── Global signal quality (3) ───
    "bpm_estimate", "signal_stationarity", "bpm_consistency",
]


def _spectral_entropy(psd):
    """Shannon entropy of normalized PSD."""
    psd_norm = psd / (np.sum(psd) + 1e-10)
    psd_norm = psd_norm[psd_norm > 0]
    return -np.sum(psd_norm * np.log2(psd_norm + 1e-10))


def _spectral_centroid(freqs, psd):
    """Weighted mean frequency."""
    total = np.sum(psd) + 1e-10
    return np.sum(freqs * psd) / total


def _zero_crossing_rate(signal):
    """Fraction of successive samples that cross zero."""
    if len(signal) < 2:
        return 0.0
    crossings = np.sum(np.abs(np.diff(np.sign(signal))) > 0)
    return crossings / (len(signal) - 1)


def _mean_coherence(bvp1, bvp2, fs, freq_low=0.7, freq_high=3.5):
    """Mean magnitude-squared coherence in physiological band."""
    nperseg = min(256, min(len(bvp1), len(bvp2)))
    if nperseg < 4:
        return 0.0
    freqs, coh = coherence(bvp1, bvp2, fs=fs, nperseg=nperseg)
    mask = (freqs >= freq_low) & (freqs <= freq_high)
    if not np.any(mask):
        return 0.0
    return np.mean(coh[mask])


def _phase_difference(bvp1, bvp2, fs, freq_low=0.7, freq_high=3.5):
    """Phase difference at dominant frequency using cross-spectral density."""
    nperseg = min(256, min(len(bvp1), len(bvp2)))
    if nperseg < 4:
        return 0.0
    from scipy.signal import csd
    freqs, pxy = csd(bvp1, bvp2, fs=fs, nperseg=nperseg)
    mask = (freqs >= freq_low) & (freqs <= freq_high)
    if not np.any(mask):
        return 0.0
    peak_idx = np.argmax(np.abs(pxy[mask]))
    phase = np.angle(pxy[mask][peak_idx])
    return np.abs(phase)


def _extract_single_roi_features(bvp, fs, freq_low=0.7, freq_high=3.5):
    """Extract 12 features from a single ROI BVP signal."""
    if len(bvp) < 10:
        return [0.0] * 12

    # Spectral
    snr = compute_snr(bvp, fs, freq_low, freq_high)
    sp = compute_spectral_purity(bvp, fs, freq_low, freq_high)
    pp = compute_peak_prominence(bvp, fs, freq_low, freq_high)
    _, dom_freq, freqs, psd = estimate_bpm(bvp, fs, freq_low, freq_high)
    hr = compute_harmonic_ratio(bvp, fs, freq_low, freq_high)

    mask = (freqs >= freq_low) & (freqs <= freq_high)
    masked_psd = psd[mask] if np.any(mask) else psd
    masked_freqs = freqs[mask] if np.any(mask) else freqs
    se = _spectral_entropy(masked_psd)
    sc = _spectral_centroid(masked_freqs, masked_psd)

    # Temporal
    mad = np.mean(np.abs(bvp - np.mean(bvp)))
    std = np.std(bvp)
    zcr = _zero_crossing_rate(bvp)
    kurt = kurtosis(bvp) if len(bvp) > 3 else 0.0
    skewness = skew(bvp) if len(bvp) > 3 else 0.0

    return [snr, sp, pp, dom_freq, hr, se, sc,
            mad, std, zcr, kurt, skewness]


def extract_features(bvp_forehead, bvp_left_cheek, bvp_right_cheek, fs,
                     freq_low=0.7, freq_high=3.5):
    """
    Extract complete 35-dimensional feature vector from three ROI BVP signals.

    Args:
        bvp_forehead: BVP from forehead ROI
        bvp_left_cheek: BVP from left cheek ROI
        bvp_right_cheek: BVP from right cheek ROI
        fs: sampling rate
    Returns:
        features: numpy array of shape (35,)
        feature_names: list of 35 feature names
    """
    # Per-ROI features (12 each for forehead and left cheek = 24)
    fh_feat = _extract_single_roi_features(bvp_forehead, fs, freq_low, freq_high)
    lc_feat = _extract_single_roi_features(bvp_left_cheek, fs, freq_low, freq_high)

    # Cross-ROI correlations (3)
    corr_fh_lc = compute_roi_correlation(bvp_forehead, bvp_left_cheek)
    corr_fh_rc = compute_roi_correlation(bvp_forehead, bvp_right_cheek)
    corr_lc_rc = compute_roi_correlation(bvp_left_cheek, bvp_right_cheek)

    # Cross-ROI coherence (3)
    coh_fh_lc = _mean_coherence(bvp_forehead, bvp_left_cheek, fs, freq_low, freq_high)
    coh_fh_rc = _mean_coherence(bvp_forehead, bvp_right_cheek, fs, freq_low, freq_high)
    coh_lc_rc = _mean_coherence(bvp_left_cheek, bvp_right_cheek, fs, freq_low, freq_high)

    # Phase differences (2)
    phase_fh_lc = _phase_difference(bvp_forehead, bvp_left_cheek, fs, freq_low, freq_high)
    phase_fh_rc = _phase_difference(bvp_forehead, bvp_right_cheek, fs, freq_low, freq_high)

    # Global features (3)
    bpm, _, _, _ = estimate_bpm(bvp_forehead, fs, freq_low, freq_high)
    stationarity = compute_signal_stationarity(bvp_forehead)

    # BPM consistency across ROIs
    bpm_lc, _, _, _ = estimate_bpm(bvp_left_cheek, fs, freq_low, freq_high)
    bpm_rc, _, _, _ = estimate_bpm(bvp_right_cheek, fs, freq_low, freq_high)
    bpm_values = [bpm, bpm_lc, bpm_rc]
    bpm_mean = np.mean(bpm_values)
    bpm_consistency = 1.0 - (np.std(bpm_values) / (bpm_mean + 1e-8))
    bpm_consistency = max(0.0, min(1.0, bpm_consistency))

    features = np.array(
        fh_feat + lc_feat +
        [corr_fh_lc, corr_fh_rc, corr_lc_rc,
         coh_fh_lc, coh_fh_rc, coh_lc_rc,
         phase_fh_lc, phase_fh_rc,
         bpm, stationarity, bpm_consistency],
        dtype=np.float64
    )

    # Replace NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=40.0, neginf=-40.0)

    return features, FEATURE_NAMES
