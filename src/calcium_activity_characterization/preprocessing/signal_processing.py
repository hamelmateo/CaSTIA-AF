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
from calcium_activity_characterization.logger import logger

from calcium_activity_characterization.config.presets import SignalProcessingConfig
from calcium_activity_characterization.data.peaks import PeakDetector
from calcium_activity_characterization.preprocessing.local_minima_detrender import LocalMinimaDetrender
from calcium_activity_characterization.preprocessing.double_curve_detrender import DoubleCurveDetrender




class SignalProcessor:
    """
    Pipeline to process calcium imaging signals with support for both
    filtering-based and baseline subtraction detrending methods.

    Args:
        config (dict): Configuration dictionary for the processing pipeline.
    """

    def __init__(self, config: SignalProcessingConfig):
        self.trace_versions: dict[str, np.ndarray] = {}

        self.pipeline = config.pipeline
        self.cut_length = config.detrending.params.cut_trace_num_points
        self.sigma = config.smoothing_sigma

        self.normalize_method = config.normalization.method
        self.normalize_params = config.normalization.params

        self.detrending_mode = config.detrending.method
        self.detrending_params = config.detrending.params

    def run(self, trace: np.ndarray) -> np.ndarray:
        """
        Run the signal processing pipeline on the provided Trace object.

        Args:
            trace (Trace): Trace object to process.

        Returns:
            np.ndarray: Processed trace data.
        """
        processed_trace = trace

        if self.pipeline.detrending:
            processed_trace = self._detrend(processed_trace)
            self.trace_versions["detrended"] = processed_trace.copy()
            
        if self.pipeline.normalization:
            processed_trace = self._normalize(processed_trace)
            self.trace_versions["normalized"] = processed_trace.copy()

        if self.pipeline.smoothing:
            processed_trace = gaussian_filter1d(processed_trace, sigma=self.sigma)
            self.trace_versions["smoothed"] = processed_trace.copy()
            
        return processed_trace

    def get_intermediate_versions(self) -> dict[str, np.ndarray]:
        """
        Get all intermediate trace versions generated during processing.

        Returns:
            dict[str, np.ndarray]: dictionary of trace versions.
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
            return self._detrend_filter_based(trace)
        elif self.detrending_mode in {"movingaverage", "polynomial", "robustpoly", "exponentialfit", "savgol"}:
            return self._detrend_baseline_subtraction(trace)
        elif self.detrending_mode == "doublecurvefitting":
            return self._detrend_double_curve_fitting(trace)
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
            cutoff = self.detrending_params.cutoff
            order = self.detrending_params.order
            fs = self.detrending_params.sampling_freq
            b, a = butter(order, cutoff, btype='highpass', fs=fs)
            return filtfilt(b, a, trace)

        elif mode == "fir":
            cutoff = self.detrending_params.cutoff
            numtaps = self.detrending_params.numtaps
            fs = self.detrending_params.sampling_freq
            fir_coeff = firwin(numtaps, cutoff=cutoff, fs=fs, pass_zero=False)
            return filtfilt(fir_coeff, [1.0], trace)

        elif mode == "wavelet":
            import pywt
            level = self.detrending_params.level
            wavelet = self.detrending_params.wavelet
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
        detector = PeakDetector(self.detrending_params.baseline_detection_params)
        peak_list = detector.run(trace)

        # Create a mask for the detected peaks
        mask = np.zeros_like(trace, dtype=bool)
        for peak in peak_list:
            mask[peak.activation_start_time:peak.activation_end_time + 1] = True
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
            window = self.detrending_params.window_size
            return np.convolve(np.pad(trace, (window//2, window//2), mode='reflect'),
                               np.ones(window)/window, mode='valid')
        elif mode == "polynomial":
            degree = self.detrending_params.degree
            x = np.arange(len(trace))
            coeffs = np.polyfit(x, trace, deg=degree)
            return np.polyval(coeffs, x)
        elif mode == "robustpoly":
            degree = self.detrending_params.degree
            method = self.detrending_params.method
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
            win = self.detrending_params.window_length
            poly = self.detrending_params.polyorder
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
        eps = self.normalize_params.epsilon
        method = self.normalize_method
        
        if method == "minmax":
            min_range = self.normalize_params.min_range
            min_val, max_val = np.min(trace), np.max(trace)
            denom = max_val - min_val
            return (trace - min_val) / (denom + eps) if denom >= min_range else trace - min_val

        elif method == "percentile":
            q = self.normalize_params.percentile_baseline
            baseline = np.percentile(trace, q)
            peak = np.max(trace)
            return (trace - baseline) / (peak - baseline + eps)

        elif method == "deltaf":
            min_range = self.normalize_params.min_range
            q = self.normalize_params.percentile_baseline
            F0 = max(np.percentile(trace, q), min_range)
            deltaf = (trace - F0) / (F0 + eps)
            return deltaf / (np.max(np.abs(deltaf)) + eps)
        
        elif method == "zscore":
            return self._zscore(
                trace, 
                sigma=self.normalize_params.smoothing_sigma,
                residuals_clip_percentile=self.normalize_params.residuals_clip_percentile,
                eps=eps,
                residuals_min_number=self.normalize_params.residuals_min_number
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


    def _zscore(
        self,
        trace: np.ndarray,
        sigma: float = 5.0,
        residuals_clip_percentile: float = 95.0,
        eps: float = 1e-8,
        residuals_min_number: int = 20
    ) -> np.ndarray:
        """
        Normalize the trace using noise-based z-score computed from residuals.

        Args:
            trace (np.ndarray): Detrended trace.
            sigma (float): Gaussian smoothing sigma.
            clip_percentile (float): Threshold to exclude outlier residuals.
            eps (float): Small value to avoid division by zero.

        Returns:
            np.ndarray: z-scored trace.
        """
        try:
            smoothed = gaussian_filter1d(trace, sigma=sigma)
            residual = trace - smoothed

            abs_residual = np.abs(residual)
            clip_threshold = np.percentile(abs_residual, residuals_clip_percentile)

            # Keep only residuals within threshold to compute std
            valid_residuals = residual[abs_residual < clip_threshold]

            if len(valid_residuals) < residuals_min_number:
                raise ValueError("Too few valid residuals to compute robust std.")

            sigma_noise = np.std(valid_residuals)

            self.trace_versions["zscore_residual_smoothed"] = smoothed.copy()
            self.trace_versions["zscore_residual"] = residual.copy()
            self.trace_versions["zscore_residual_clip_threshold"] = np.full_like(trace, clip_threshold)
            self.trace_versions["zscore_valid_residual"] = valid_residuals.copy()

            return trace / (sigma_noise + eps)

        except Exception as e:
            logger.error(f"Failed to compute residual z-score: {e}")
            return trace.copy()
