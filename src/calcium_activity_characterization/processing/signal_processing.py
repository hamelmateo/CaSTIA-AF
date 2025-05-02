"""Module containing the SignalProcessor class for processing calcium imaging traces.

Example:
    >>> from src.analysis.signal_processing import SignalProcessor
    >>> processor = SignalProcessor(mode="butterworth", params=SIGNAL_PROCESSING_PARAMETERS["butterworth"])
    >>> processed = processor.run(raw_trace)
"""

import numpy as np
from scipy.signal import butter, firwin, savgol_filter, sosfilt, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import pywt


class SignalProcessor:
    """
    Class for pptional detrending, smoothing, and normalizing calcium intensity traces.

    Attributes:
        mode (str): Detrending method name (e.g., 'butterworth', 'wavelet', etc.)
        params (dict): Dictionary of parameters specific to the selected method
    """

    def __init__(self, params: dict, pipeline: dict):
        """
        Initialize the SignalProcessor.

        Args:
            mode (str): The detrending method to use.
            params (dict): Parameters specific to the selected method.
        """
        self.use_detrending = pipeline.get("detrending", True)
        self.use_smoothing = pipeline.get("smoothing", True)
        self.use_normalization = pipeline.get("normalization", True)

        self.detrending_mode = pipeline.get("detrending_mode", "butterworth")
        self.detrending_params = params.get("methods", {}).get(self.detrending_mode, {})

        self.normalize_method = pipeline.get("normalizing_method", "deltaf")

        self.sigma = params.get("sigma", 1.0)


    def run(self, raw_trace: list[float]) -> np.ndarray:
        """
        Execute full processing: detrending → smoothing → normalization.

        Args:
            raw_trace (list[float]): Raw calcium trace.

        Returns:
            np.ndarray: Processed trace.
        """
        raw_trace = np.array(raw_trace, dtype=float)

        if self.use_detrending:
            processed_trace = self._detrend(raw_trace)
            processed_trace = self._cut_trace_start(processed_trace, 125)

        if self.use_smoothing:
            processed_trace = gaussian_filter1d(processed_trace, sigma=self.sigma)

        if self.use_normalization:
            processed_trace = self._normalize(processed_trace)

        return processed_trace

    def _cut_trace_start(self, trace: np.ndarray, num_points: int) -> np.ndarray:
        """
        Remove initial timepoints from the trace.

        Args:
            trace (np.ndarray): Input trace.
            num_points (int): Number of initial timepoints to discard.

        Returns:
            np.ndarray: Truncated trace.
        """
        return trace[num_points:] if len(trace) > num_points else np.array([], dtype=trace.dtype)

    def _detrend(self, trace: np.ndarray) -> np.ndarray:
        """
        Apply the selected detrending method.

        Args:
            trace (np.ndarray): Raw trace.

        Returns:
            np.ndarray: Detrended trace.
        """
        if self.detrending_mode == "wavelet":
            wavelet = self.detrending_params.get("wavelet", "db4")
            level = self.detrending_params.get("level")
            coeffs = pywt.wavedec(trace, wavelet, mode="periodization", level=level)
            coeffs[0] = np.zeros_like(coeffs[0])
            return pywt.waverec(coeffs, wavelet, mode="periodization")[:len(trace)]

        elif self.detrending_mode == "fir":
            cutoff = self.detrending_params.get("cutoff", 0.001)
            numtaps = self.detrending_params.get("numtaps", 201)
            fs = self.detrending_params.get("sampling_freq", 1.0)
            if numtaps % 2 == 0:
                numtaps += 1
            fir_coeff = firwin(numtaps, cutoff=cutoff, fs=fs, pass_zero=False)
            return filtfilt(fir_coeff, [1.0], trace)

        elif self.detrending_mode == "butterworth":
            cutoff = self.detrending_params.get("cutoff", 0.003)
            order = self.detrending_params.get("order", 6)
            mode = self.detrending_params.get("mode", "ba")
            btype = self.detrending_params.get("btype", "highpass")
            fs = self.detrending_params.get("sampling_freq", 1.0)
            if mode == "ba":
                b, a = butter(order, cutoff, btype=btype, fs=fs)
                return filtfilt(b, a, trace)
            else:
                sos = butter(order, cutoff, btype=btype, output=mode, fs=fs)
                return sosfilt(sos, trace)

        elif self.detrending_mode == "exponentialfit":
            return self._detrend_exponential(trace)[0]

        elif self.detrending_mode == "diff":
            return np.diff(trace, prepend=trace[0])

        elif self.detrending_mode == "savgol":
            window_length = self.params.get("window_length", 101)
            polyorder = self.params.get("polyorder", 2)
            if window_length % 2 == 0:
                window_length += 1
            baseline = savgol_filter(trace, window_length, polyorder)
            return trace - baseline

        elif self.detrending_mode == "movingaverage":
            window = self.params.get("window_size", 101)
            baseline = np.convolve(trace, np.ones(window)/window, mode='same')
            return trace - baseline

        else:
            raise ValueError(f"Unsupported detrending mode: {self.mode}")

    def _detrend_exponential(self, trace: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Fit and subtract exponential decay curve from the trace.

        Args:
            trace (np.ndarray): Input trace.

        Returns:
            tuple[np.ndarray, float]: Detrended trace and R² score.
        """
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
                maxfev=10000,
            )
            fitted = exp_model(t, *popt)
            return trace - fitted, r2_score(trace, fitted)
        except Exception:
            return trace, 0.0

    def _normalize(self, trace: np.ndarray, min_range: float = 1e-2) -> np.ndarray:
        """
        Normalize a trace using the specified method.

        Args:
            trace (np.ndarray): Input trace.
            min_range (float): Minimum dynamic range to avoid divide-by-zero.

        Returns:
            np.ndarray: Normalized trace.
        """
        method = self.normalize_method

        if method == "minmax":
            min_val, max_val = np.min(trace), np.max(trace)
            denom = max_val - min_val
            return (trace - min_val) / (denom + 1e-8) if denom >= min_range else trace - min_val

        elif method == "percentile":
            baseline = np.percentile(trace, 10)
            peak = np.max(trace)
            denom = peak - baseline
            return (trace - baseline) / (denom + 1e-8) if denom >= min_range else trace - baseline

        elif method == "deltaf":
            F0 = max(np.percentile(trace, 10), min_range)
            deltaf = (trace - F0) / (F0 + 1e-8)
            max_abs = np.max(np.abs(deltaf))
            return deltaf / (max_abs + 1e-8) if max_abs >= min_range else deltaf

        elif method == "zscore":
            mean, std = np.mean(trace), np.std(trace)
            return (trace - mean) / (std + 1e-8) if std >= min_range else trace - mean

        else:
            raise ValueError(f"Unknown normalization method: {method}")