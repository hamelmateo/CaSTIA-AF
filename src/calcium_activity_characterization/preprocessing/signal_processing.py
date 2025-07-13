#!/usr/bin/env python3
"""
Module: signal_processing.py

Refactored SignalProcessor class for calcium imaging signal pipelines.

Usage Example:
    >>> processor = SignalProcessor(config)
    >>> processor.run(trace, input_version="raw", output_version="processed")
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, firwin, filtfilt, savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RANSACRegressor, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
import logging

from calcium_activity_characterization.data.peaks import PeakDetector
from calcium_activity_characterization.preprocessing.local_minima_detrender import LocalMinimaDetrender
from calcium_activity_characterization.preprocessing.double_curve_detrender import DoubleCurveDetrender

logger = logging.getLogger(__name__)


class SignalProcessor:
    """
    Pipeline to process calcium imaging signals with support for both
    filtering-based and baseline subtraction detrending methods.

    Args:
        config (dict): Configuration dictionary for the processing pipeline.
    """

    def __init__(self, config: dict):
        self.config = config
        self.trace_versions: dict[str, np.ndarray] = {}

        self.apply_flags = config.get("apply", {})
        self.cut_length = config.get("cut_trace_num_points", 0)
        self.pre_sigma = config.get("presmoothing_sigma", 1.0)
        self.sigma = config.get("smoothing_sigma", 1.0)
        self.normalize_method = config.get("normalizing_method", "zscore")
        self.normalize_params = config.get("normalization_parameters", {})

        self.detrending_mode = config.get("detrending_mode", "movingaverage")
        self.detrending_params = config.get("methods", {}).get(self.detrending_mode, {})

    def run(self, trace: np.ndarray) -> np.ndarray:
        """
        Run the signal processing pipeline on the provided Trace object.

        Args:
            trace (Trace): Trace object to process.
            input_version (str): Key of the initial trace version to process.
            output_version (str): Key to assign the final processed trace.

        Returns:
            np.ndarray: Processed trace data.
        """
        processed_trace = trace

        if self.apply_flags.get("detrending", False):
            processed_trace = self._detrend(processed_trace)
            self.trace_versions["detrended"] = processed_trace.copy()
            
        if self.apply_flags.get("normalization", False):
            processed_trace = self._normalize(processed_trace)
            self.trace_versions["normalized"] = processed_trace.copy()
            
        if self.apply_flags.get("smoothing", False):
            processed_trace = gaussian_filter1d(processed_trace, sigma=self.sigma)
            self.trace_versions["smoothed"] = processed_trace.copy()
            
        return processed_trace

    def get_intermediate_versions(self) -> dict[str, np.ndarray]:
        """
        Get all intermediate trace versions generated during processing.

        Returns:
            dict[str, np.ndarray]: Dictionary of trace versions.
        """
        return self.trace_versions

    def _detrend(self, trace: np.ndarray) -> np.ndarray:
        """
        Apply the selected detrending method to the trace.
        This method supports both filtering-based detrending (e.g., Butterworth, FIR, wavelet)
        and baseline subtraction methods (e.g., moving average, polynomial fitting).

        Args:
            trace (np.ndarray): Raw trace data.

        Returns:
            np.ndarray: Detrended trace.
        """
        if self.detrending_mode in {"butterworth", "fir", "wavelet"}:
            return self._detrend_filter_based(trace), {}
        elif self.detrending_mode in {"movingaverage", "polynomial", "robustpoly", "exponentialfit", "savgol"}:
            return self._detrend_baseline_subtraction(trace), {}
        elif self.detrending_mode == "doublecurvefitting":
            return self._detrend_double_curve_fitting(trace), {}
        elif self.detrending_mode == "localminima":
            return self._detrend_local_minima(trace)
        else:
            raise ValueError(f"Unsupported detrending mode: {self.detrending_mode}")

    def _detrend_filter_based(self, trace: np.ndarray) -> np.ndarray:
        """
        Apply filtering-based detrending methods such as Butterworth, FIR, or wavelet.

        Args:
            trace (np.ndarray): Raw trace data.

        Returns:
            np.ndarray: Detrended trace.
        """
        trace = self._cut_trace(trace, self.cut_length)
        mode = self.detrending_mode

        if mode == "butterworth":
            cutoff = self.detrending_params.get("cutoff", 0.003)
            order = self.detrending_params.get("order", 6)
            fs = self.detrending_params.get("sampling_freq", 1.0)
            b, a = butter(order, cutoff, btype='highpass', fs=fs)
            return filtfilt(b, a, trace)

        elif mode == "fir":
            cutoff = self.detrending_params.get("cutoff", 0.001)
            numtaps = self.detrending_params.get("numtaps", 201)
            fs = self.detrending_params.get("sampling_freq", 1.0)
            fir_coeff = firwin(numtaps, cutoff=cutoff, fs=fs, pass_zero=False)
            return filtfilt(fir_coeff, [1.0], trace)

        elif mode == "wavelet":
            import pywt
            level = self.detrending_params.get("level")
            wavelet = self.detrending_params.get("wavelet", "db4")
            coeffs = pywt.wavedec(trace, wavelet, level=level, mode="periodization")
            coeffs[0] = np.zeros_like(coeffs[0])
            return pywt.waverec(coeffs, wavelet, mode="periodization")[:len(trace)]

    def _detrend_baseline_subtraction(self, trace: np.ndarray) -> np.ndarray:
        """
        Apply baseline subtraction methods such as moving average, polynomial fitting, or robust regression.
        This method also detects peaks, masks them, interpolates the trace, fits a baseline,
        and computes the residual.

        Args:
            trace (np.ndarray): Raw trace data.

        Returns:
            np.ndarray: Detrended trace.
        """
        # Detect peaks and create a mask for long peaks
        detector = PeakDetector(self.detrending_params.get("peak_detector_params"))
        peak_list = detector.run(trace)

        # Create a mask for the detected peaks
        mask = np.zeros_like(trace, dtype=bool)
        for peak in peak_list:
            mask[peak.rel_start_time:peak.rel_end_time + 1] = True
        self.trace_versions["mask"] = mask.astype(np.float32)
        
        # Interpolate the trace where the mask is True
        x = np.arange(len(trace))
        interpolated = trace.copy()
        interpolated[mask] = np.interp(x[mask], x[~mask], trace[~mask])
        self.trace_versions["interpolated"] = interpolated.copy()
        
        # Cut the trace to remove initial frames if specified
        cut_interpolated = self._cut_trace(interpolated, self.cut_length)
        baseline = self._fit_baseline(cut_interpolated)
        self.trace_versions["baseline"] = baseline.copy()
        
        # Compute the residual and store it
        residual = cut_interpolated - baseline
        self.trace_versions["residual"] = residual.copy()
        
        # Compute the final detrended trace
        cut_original = self._cut_trace(trace, self.cut_length)
        return cut_original - baseline


    def _detrend_double_curve_fitting(self, trace: np.ndarray) -> np.ndarray:
        """
        Detrend trace using the DoubleCurveDetrender.

        Args:
            trace (np.ndarray): Raw input trace.

        Returns:
            np.ndarray: Final detrended trace
        """
        try:
            detrender = DoubleCurveDetrender(self.detrending_params, trace, self.cut_length)
            detrended = detrender.run()
            self.trace_versions.update(detrender.get_intermediate_versions())
            return detrended
        except Exception as e:
            logger.error(f"Double curve fitting failed: {e}")
            return trace.copy()


    def _detrend_local_minima(self, trace: np.ndarray) -> np.ndarray:
        """
        Detrend trace using the LocalMinimaDetrender.

        Args:
            trace (np.ndarray): Input trace (usually smoothed).

        Returns:
            np.ndarray: Final detrended trace.
        """
        try:
            detrender = LocalMinimaDetrender(self.detrending_params, trace)
            detrended = detrender.run()
            self.trace_versions.update(detrender.get_intermediate_versions())

            detrended = self._cut_trace(detrended, self.cut_length)
            return detrended
        except Exception as e:
            logger.error(f"Local minima detrending failed: {e}")
            return trace.copy()

    def _fit_baseline(self, trace: np.ndarray) -> np.ndarray:
        """
        Fit a baseline to the trace using the specified detrending method.
        This method supports various baseline fitting techniques such as moving average,
        polynomial fitting, robust regression, and exponential fitting.

        Args:
            trace (np.ndarray): Input trace data.

        Returns:
            np.ndarray: Fitted baseline.
        """
        mode = self.detrending_mode
        if mode == "movingaverage":
            window = self.detrending_params.get("window_size", 101)
            return np.convolve(np.pad(trace, (window//2, window//2), mode='reflect'),
                               np.ones(window)/window, mode='valid')
        elif mode == "polynomial":
            degree = self.detrending_params.get("degree", 2)
            x = np.arange(len(trace))
            coeffs = np.polyfit(x, trace, deg=degree)
            return np.polyval(coeffs, x)
        elif mode == "robustpoly":
            degree = self.detrending_params.get("degree", 2)
            method = self.detrending_params.get("method", "huber")
            x = np.arange(len(trace)).reshape(-1, 1)
            if method == "ransac":
                model = make_pipeline(PolynomialFeatures(degree), RANSACRegressor(residual_threshold=5.0))
            else:
                model = make_pipeline(PolynomialFeatures(degree), HuberRegressor())
            model.fit(x, trace)
            return model.predict(x)
        elif mode == "exponentialfit":
            def exp_model(t, A, k, C):
                return A * np.exp(-k * t) + C
            t = np.arange(len(trace))
            popt, _ = curve_fit(exp_model, t, trace,
                                p0=[trace[0] - trace[-1], 0.01, trace[-1]],
                                bounds=([0, 0, 0], [np.inf, 1.0, np.inf]),
                                maxfev=10000)
            return exp_model(t, *popt)
        elif mode == "savgol":
            win = self.detrending_params.get("window_length", 101)
            poly = self.detrending_params.get("polyorder", 2)
            return savgol_filter(trace, win, poly)

    def _normalize(self, trace: np.ndarray) -> np.ndarray:
        """
        Normalize the trace using the specified normalization method.
        This method supports various normalization techniques such as min-max scaling,
        percentile-based normalization, ΔF/F, and z-score normalization.

        Args:
            trace (np.ndarray): Input trace data.

        Returns:
            np.ndarray: Normalized trace.
        """
        eps = self.normalize_params.get("epsilon", 1e-8)
        min_range = self.normalize_params.get("min_range", 1e-2)
        method = self.normalize_method

        if method == "minmax":
            min_val, max_val = np.min(trace), np.max(trace)
            denom = max_val - min_val
            return (trace - min_val) / (denom + eps) if denom >= min_range else trace - min_val

        elif method == "percentile":
            q = self.normalize_params.get("percentile_baseline", 10)
            baseline = np.percentile(trace, q)
            peak = np.max(trace)
            return (trace - baseline) / (peak - baseline + eps)

        elif method == "deltaf":
            q = self.normalize_params.get("percentile_baseline", 10)
            F0 = max(np.percentile(trace, q), min_range)
            deltaf = (trace - F0) / (F0 + eps)
            return deltaf / (np.max(np.abs(deltaf)) + eps)

        elif method == "zscore":
            # TODO: implement a z score that takes only the std of the noise into account instead of the whole signal.
            mean, std = np.mean(trace), np.std(trace)
            return (trace - mean) / (std + eps) if std >= min_range else trace - mean
        
        elif method == "zscore_residual":
            return self._zscore_residual(
                trace, 
                sigma=self.normalize_params.get("sigma", 2.0),
                clip_percentile=self.normalize_params.get("clip_percentile", 80.0),
                eps=eps
            )

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def _cut_trace(self, trace: np.ndarray, nb_frames: int) -> np.ndarray:
        """
        Remove the initial frames from the trace to avoid early photobleaching effects.

        Args:
            trace (np.ndarray): Input trace data.

        Returns:
            np.ndarray: Truncated trace.
        """
        return trace[nb_frames:] if len(trace) > nb_frames else np.array([], dtype=trace.dtype)


    def _zscore_residual(
        self,
        trace: np.ndarray,
        sigma: float = 5.0,
        clip_percentile: float = 95.0,
        eps: float = 1e-8
    ) -> np.ndarray:
        """
        Normalize the trace using noise-based z-score computed from residuals.

        Args:
            trace (np.ndarray): Detrended trace.
            sigma (float): Gaussian smoothing sigma.
            clip_percentile (float): Threshold to exclude outlier residuals.
            eps (float): Small value to avoid division by zero.
            plot (bool): Whether to display diagnostic plots.
            label (str): Trace label.

        Returns:
            np.ndarray: z-scored trace.
        """
        try:
            smoothed = gaussian_filter1d(trace, sigma=sigma)
            residual = trace - smoothed

            abs_residual = np.abs(residual)
            clip_threshold = np.percentile(abs_residual, clip_percentile)

            # Keep only residuals within threshold to compute std
            valid_residuals = residual[abs_residual < clip_threshold]

            if len(valid_residuals) < 20:
                raise ValueError("Too few valid residuals to compute robust std.")

            sigma_noise = np.std(valid_residuals)
            logger.info(f"Computed noise std: {sigma_noise:.4f} from {len(valid_residuals)} valid residuals.")

            self.trace_versions["zscore_residual_smoothed"] = smoothed.copy()
            self.trace_versions["zscore_residual"] = residual.copy()
            self.trace_versions["zscore_residual_clip_threshold"] = np.full_like(trace, clip_threshold)
            self.trace_versions["zscore_valid_residual"] = valid_residuals.copy()

            return trace / (sigma_noise + eps)

        except Exception as e:
            logger.error(f"Failed to compute residual z-score: {e}")
            return trace.copy()
