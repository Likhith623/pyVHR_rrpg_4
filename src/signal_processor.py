"""
Signal processing utilities for rPPG analysis.
Welch PSD estimation, SNR computation, BPM extraction,
spectral purity, peak prominence, and signal quality metrics.
"""
import numpy as np
from scipy.signal import welch, find_peaks
from scipy.stats import pearsonr


def compute_welch_psd(bvp, fs, nperseg=256, noverlap=128, nfft=1024):
    """
    Compute Power Spectral Density using Welch's method.

    Returns:
        freqs: frequency array (Hz)
        psd: power spectral density array
    """
    nperseg = min(nperseg, len(bvp))
    noverlap = min(noverlap, nperseg - 1)
    freqs, psd = welch(bvp, fs=fs, nperseg=nperseg,
                       noverlap=noverlap, nfft=nfft)
    return freqs, psd


def estimate_bpm(bvp, fs, freq_low=0.7, freq_high=3.5,
                 nperseg=256, noverlap=128, nfft=1024):
    """
    Estimate heart rate (BPM) from BVP signal using Welch PSD peak.

    Returns:
        bpm: estimated beats per minute
        peak_freq: dominant frequency in Hz
        freqs: frequency array
        psd: PSD array
    """
    freqs, psd = compute_welch_psd(bvp, fs, nperseg, noverlap, nfft)

    # Mask to physiological range
    mask = (freqs >= freq_low) & (freqs <= freq_high)
    if not np.any(mask):
        return 0.0, 0.0, freqs, psd

    masked_psd = psd[mask]
    masked_freqs = freqs[mask]

    peak_idx = np.argmax(masked_psd)
    peak_freq = masked_freqs[peak_idx]
    bpm = peak_freq * 60.0

    return bpm, peak_freq, freqs, psd


def compute_snr(bvp, fs, freq_low=0.7, freq_high=3.5, bandwidth=0.2,
                nperseg=256, noverlap=128, nfft=1024):
    """
    Signal-to-Noise Ratio in dB.
    Signal power = power within ±bandwidth of peak frequency.
    Noise power = remaining power in physiological band.
    """
    freqs, psd = compute_welch_psd(bvp, fs, nperseg, noverlap, nfft)
    mask = (freqs >= freq_low) & (freqs <= freq_high)
    if not np.any(mask):
        return -np.inf

    masked_psd = psd[mask]
    masked_freqs = freqs[mask]

    peak_idx = np.argmax(masked_psd)
    peak_freq = masked_freqs[peak_idx]

    signal_mask = np.abs(masked_freqs - peak_freq) <= bandwidth
    noise_mask = ~signal_mask

    signal_power = np.sum(masked_psd[signal_mask])
    noise_power = np.sum(masked_psd[noise_mask])

    if noise_power < 1e-10:
        return 40.0  # very clean signal
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def compute_spectral_purity(bvp, fs, freq_low=0.7, freq_high=3.5,
                            bandwidth=0.2, nperseg=256, noverlap=128,
                            nfft=1024):
    """
    Spectral purity: ratio of peak power to total power in band.
    High value = signal dominated by single frequency (real pulse).
    """
    freqs, psd = compute_welch_psd(bvp, fs, nperseg, noverlap, nfft)
    mask = (freqs >= freq_low) & (freqs <= freq_high)
    if not np.any(mask):
        return 0.0

    masked_psd = psd[mask]
    masked_freqs = freqs[mask]

    peak_idx = np.argmax(masked_psd)
    peak_freq = masked_freqs[peak_idx]

    signal_mask = np.abs(masked_freqs - peak_freq) <= bandwidth
    peak_power = np.sum(masked_psd[signal_mask])
    total_power = np.sum(masked_psd)

    if total_power < 1e-10:
        return 0.0
    return peak_power / total_power


def compute_peak_prominence(bvp, fs, freq_low=0.7, freq_high=3.5,
                            nperseg=256, noverlap=128, nfft=1024):
    """
    Prominence of the dominant spectral peak.
    Higher prominence = clearer physiological signal.
    """
    freqs, psd = compute_welch_psd(bvp, fs, nperseg, noverlap, nfft)
    mask = (freqs >= freq_low) & (freqs <= freq_high)
    if not np.any(mask):
        return 0.0

    masked_psd = psd[mask]
    peaks, properties = find_peaks(masked_psd, prominence=0)
    if len(peaks) == 0:
        return 0.0

    return np.max(properties["prominences"])


def compute_roi_correlation(bvp_roi1, bvp_roi2):
    """
    Pearson correlation between BVP signals from two ROIs.
    Real faces: high correlation (same heart drives both).
    Fake: low/random correlation.
    """
    if len(bvp_roi1) < 3 or len(bvp_roi2) < 3:
        return 0.0
    min_len = min(len(bvp_roi1), len(bvp_roi2))
    corr, _ = pearsonr(bvp_roi1[:min_len], bvp_roi2[:min_len])
    if np.isnan(corr):
        return 0.0
    return corr


def compute_signal_stationarity(bvp, n_segments=4):
    """
    Measure signal stationarity by comparing variance across segments.
    Real pulse: relatively stationary.
    Returns: 1 - coefficient_of_variation of segment variances.
    """
    if len(bvp) < n_segments * 2:
        return 0.0
    segments = np.array_split(bvp, n_segments)
    variances = [np.var(seg) for seg in segments]
    mean_var = np.mean(variances)
    if mean_var < 1e-10:
        return 1.0
    cv = np.std(variances) / mean_var
    return max(0.0, 1.0 - cv)


def compute_harmonic_ratio(bvp, fs, freq_low=0.7, freq_high=3.5,
                           nperseg=256, noverlap=128, nfft=1024):
    """
    Ratio of power at 2nd harmonic to fundamental.
    Real pulse has characteristic harmonics.
    """
    freqs, psd = compute_welch_psd(bvp, fs, nperseg, noverlap, nfft)
    mask = (freqs >= freq_low) & (freqs <= freq_high)
    if not np.any(mask):
        return 0.0

    masked_psd = psd[mask]
    masked_freqs = freqs[mask]

    peak_idx = np.argmax(masked_psd)
    fund_freq = masked_freqs[peak_idx]
    harm_freq = 2 * fund_freq

    # Find power near harmonic
    harm_mask = (freqs >= harm_freq - 0.15) & (freqs <= harm_freq + 0.15)
    if not np.any(harm_mask):
        return 0.0

    fund_power = masked_psd[peak_idx]
    harm_power = np.max(psd[harm_mask])

    if fund_power < 1e-10:
        return 0.0
    return harm_power / fund_power
