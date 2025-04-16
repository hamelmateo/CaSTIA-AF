import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, firwin, savgol_filter, sosfilt, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit, OptimizeWarning
from typing import List
import warnings
from sklearn.metrics import r2_score




def process_trace(raw_trace: list[float], sigma: float = 1.0) -> tuple[np.ndarray, float]:
    raw = np.array(raw_trace)
    #detrended, r2 = detrend_exponential(raw)
    #filtered = highpass_filter(trace, cutoff, fs, order, btype, mode)
    detrended = fir_filter(raw, cutoff=0.001, fs=1.0, numtaps=201)
    smoothed = gaussian_smooth(detrended, sigma)
    normalized = normalize_trace(smoothed, 'deltaf')
    #normalized = normalize_trace(smoothed)

    processed_trace = normalized

    return processed_trace


def highpass_filter(trace, cutoff, fs=1.0, order=2, btype='highpass', mode='sos'):
    if mode == 'ba':
        b, a = butter(order, cutoff, btype=btype, analog=False, fs=fs)
        return filtfilt(b, a, trace)
    elif mode == 'sos':
        sos = butter(order, cutoff, btype=btype, analog=False, output='sos', fs=fs)
        return sosfilt(sos, trace)
    else:
        raise ValueError(f"Unsupported filter mode: {mode}. Use 'sos' or 'ba'.")


def fir_filter(trace, cutoff=0.001, fs=1.0, numtaps=201):
    if numtaps % 2 == 0:
        numtaps += 1  # Ensure it's odd
    fir_coeff = firwin(numtaps, cutoff=cutoff, fs=fs, pass_zero=False)
    return filtfilt(fir_coeff, [1.0], trace)


def diff_filter(trace, **kwargs):
    return np.diff(trace, prepend=trace[0])


def savgol_baseline_subtract(trace, window_length=21, polyorder=3):
    if window_length % 2 == 0:
        window_length += 1  # must be odd
    baseline = savgol_filter(trace, window_length, polyorder)
    return trace - baseline


def detrend_exponential(trace: np.ndarray) -> tuple[np.ndarray, float]:
    def exp_model(t, A, k, C):
        return A * np.exp(-k * t) + C

    t = np.arange(len(trace))
    try:
        popt, _ = curve_fit(
            exp_model,
            t,
            trace,
            p0=[trace[0] - trace[-1], 0.01, trace[-1]],
            bounds=([0, 0, 0], [np.inf, 1.0, np.inf]),
            maxfev=10000
        )
        fitted = exp_model(t, *popt)
        detrended = trace - fitted
        r_squared = r2_score(trace, fitted)
        return detrended, r_squared
    except Exception as e:
        print(f"Exponential fitting failed: {e}")
        return trace, 0.0


def gaussian_smooth(trace, sigma):
    return gaussian_filter1d(trace, sigma=sigma)


def normalize_trace(trace, method='deltaf', min_range=1e-2):
    """
    Normalize an intensity trace using different methods.

    Args:
        trace (np.ndarray): Input intensity trace (float).
        method (str): Normalization method:
            - 'minmax' : scale to [0, 1] using min/max
            - 'percentile' : scale to [0, 1] using 10th percentile as baseline
            - 'deltaf' : ΔF/F₀ using 10th percentile as F₀
            - 'zscore' : standard score normalization
        min_range (float): Minimum allowed dynamic range to avoid division by near-zero

    Returns:
        np.ndarray: Normalized trace
    """
    trace = np.array(trace, dtype=float)

    if method == 'minmax':
        min_val = np.min(trace)
        max_val = np.max(trace)
        denom = max_val - min_val
        if denom < min_range:
            return trace - min_val
        return (trace - min_val) / (denom + 1e-8)

    elif method == 'percentile':
        baseline = np.percentile(trace, 10)
        peak = np.max(trace)
        denom = peak - baseline
        if denom < min_range:
            return trace - baseline
        return (trace - baseline) / (denom + 1e-8)

    elif method == 'deltaf':
        F0 = max(np.percentile(trace, 10), min_range)
        deltaf = (trace - F0) / (F0 + 1e-8)
        max_abs = np.max(np.abs(deltaf))
        return deltaf / (max_abs + 1e-8) if max_abs >= min_range else deltaf

    elif method == 'zscore':
        mean = np.mean(trace)
        std = np.std(trace)
        if std < min_range:
            return trace - mean
        return (trace - mean) / (std + 1e-8)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

