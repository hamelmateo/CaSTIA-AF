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


class SignalProcessor:
    """
    Class for detrending, smoothing, and normalizing calcium intensity traces.

    Attributes:
        mode (str): Detrending method name (e.g., 'butterworth', 'wavelet', etc.)
        params (dict): Dictionary of parameters specific to the selected method
    """

    def __init__(self, mode: str, params: dict):
        """
        Initialize the SignalProcessor.

        Args:
            mode (str): The detrending method to use.
            params (dict): Parameters specific to the selected method.
        """
        self.mode = mode
        self.params = params
        self.sigma = params.get("sigma", 1.0)
        self.normalize_method = params.get("normalize_method", "deltaf")

    def run(self, raw_trace: list[float]) -> np.ndarray:
        """
        Execute full processing: detrending → smoothing → normalization.

        Args:
            raw_trace (list[float]): Raw calcium trace.

        Returns:
            np.ndarray: Processed trace.
        """
        trace = np.array(raw_trace, dtype=float)
        detrended = self._detrend(trace)
        detrended = self._cut_trace_start(detrended, 125)
        smoothed = gaussian_filter1d(detrended, sigma=self.sigma)
        return self._normalize(smoothed)

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
        if self.mode == "wavelet":
            import pywt
            wavelet = self.params.get("wavelet", "db4")
            level = self.params.get("level")
            coeffs = pywt.wavedec(trace, wavelet, mode="periodization", level=level)
            coeffs[0] = np.zeros_like(coeffs[0])
            return pywt.waverec(coeffs, wavelet, mode="periodization")[:len(trace)]

        elif self.mode == "fir":
            cutoff = self.params.get("cutoff", 0.001)
            numtaps = self.params.get("numtaps", 201)
            if numtaps % 2 == 0:
                numtaps += 1
            fir_coeff = firwin(numtaps, cutoff=cutoff, fs=1.0, pass_zero=False)
            return filtfilt(fir_coeff, [1.0], trace)

        elif self.mode == "butterworth":
            cutoff = self.params.get("cutoff", 0.003)
            order = self.params.get("order", 6)
            mode = self.params.get("mode", "ba")
            if mode == "ba":
                b, a = butter(order, cutoff, btype='highpass', fs=1.0)
                return filtfilt(b, a, trace)
            else:
                sos = butter(order, cutoff, btype='highpass', output='sos', fs=1.0)
                return sosfilt(sos, trace)

        elif self.mode == "exponentialfit":
            return self._detrend_exponential(trace)[0]

        elif self.mode == "diff":
            return np.diff(trace, prepend=trace[0])

        elif self.mode == "savgol":
            presigma = self.params.get("presmoothing_sigma", 0.0)
            if presigma > 0:
                trace = gaussian_filter1d(trace, presigma)
            window_length = self.params.get("window_length", 101)
            polyorder = self.params.get("polyorder", 2)
            if window_length % 2 == 0:
                window_length += 1
            baseline = savgol_filter(trace, window_length, polyorder)
            return trace - baseline

        elif self.mode == "movingaverage":
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

        if method == "none":
            return trace

        elif method == "minmax":
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