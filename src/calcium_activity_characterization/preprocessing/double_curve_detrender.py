# double_curve_detrender.py
# Usage Example:
#   >>> detrender = DoubleCurveDetrender(config, trace, cut_length)
#   >>> detrended = detrender.run()
#   >>> versions = detrender.get_intermediate_versions()

from typing import Dict, Tuple
import numpy as np
import logging
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class DoubleCurveDetrender:
    """
    Applies double curve fitting based detrending to a calcium trace
    using iterative masking and interpolation over a moving average baseline.

    Args:
        config (dict): Configuration parameters for curve fitting and masking.
        trace (np.ndarray): Raw intensity trace.
        cut_length (int): Final trace length after cutting.
    """

    def __init__(self, config: Dict, trace: np.ndarray, cut_length: int) -> None:
        self.config = config
        self.trace = trace
        self.cut_length = cut_length
        self.trace_versions: Dict[str, np.ndarray] = {}

    def run(self) -> np.ndarray:
        """
        Run the full double curve fitting detrending pipeline.

        Returns:
            np.ndarray: Detrended trace.
        """
        try:
            trace = self._cut_trace(self.trace, self.cut_length)
            method = self.config.get("fit_method", "movingaverage")
            window = self.config.get("window_size", 801)
            mask_method = self.config.get("mask_method", "percentile")
            percentile_bounds = self.config.get("percentile_bounds", [10, 90])
            max_iter = self.config.get("max_iterations", 5)

            baseline = np.zeros_like(trace)
            interpolated = trace.copy()
            for i in range(max_iter):
                if method == "movingaverage":
                    baseline = np.convolve(
                        np.pad(interpolated, (window // 2, window // 2), mode='reflect'),
                        np.ones(window) / window,
                        mode='valid')
                else:
                    raise ValueError(f"Unsupported fit_method: {method}")

                self.trace_versions[f"baseline_{i+1}"] = baseline.copy()
                residual = trace - baseline
                self.trace_versions[f"residual_{i+1}"] = residual.copy()

                if mask_method == "histogram":
                    mask = self._mask_residual_histogram(residual)
                elif mask_method == "percentile":
                    mask = self._mask_residual_percentile(residual, *percentile_bounds)
                else:
                    raise ValueError(f"Invalid mask_method: {mask_method}")

                if not np.any(mask):
                    logger.warning(f"Iteration {i+1}: All points masked. Stopping early.")
                    break

                interpolated = trace.copy()
                interpolated[~mask] = np.nan
                self.trace_versions[f"masked_trace_{i+1}"] = interpolated.copy()
                interpolated = self._linear_interpolate_nans(interpolated)
                self.trace_versions[f"interpolated_trace_{i+1}"] = interpolated.copy()

            final_baseline = baseline
            trace = self._cut_trace(trace, self.cut_length)
            final_baseline = self._cut_trace(final_baseline, self.cut_length)
            return trace - final_baseline
        except Exception as e:
            logger.error(f"Double curve fitting failed: {e}")
            return self.trace.copy()

    def get_intermediate_versions(self) -> Dict[str, np.ndarray]:
        """
        Return intermediate versions from each iteration.

        Returns:
            Dict[str, np.ndarray]: Dictionary of trace versions.
        """
        return self.trace_versions

    def _cut_trace(self, trace: np.ndarray, length: int) -> np.ndarray:
        return trace[:length]

    def _linear_interpolate_nans(self, arr: np.ndarray) -> np.ndarray:
        x = np.arange(len(arr))
        valid = ~np.isnan(arr)
        return np.interp(x, x[valid], arr[valid])

    def _mask_residual_percentile(self, residual: np.ndarray, lower_q: float = 10, upper_q: float = 90) -> np.ndarray:
        lower, upper = np.percentile(residual, [lower_q, upper_q])
        return (residual >= lower) & (residual <= upper)

    def _mask_residual_histogram(self, residual: np.ndarray, bins: int = 100) -> np.ndarray:
        hist, bin_edges = np.histogram(residual, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        A, mu, sigma = self._fit_gaussian_to_histogram(hist, bin_centers)
        if np.isnan(mu) or np.isnan(sigma):
            logger.warning("Skipping residual filtering: invalid Gaussian fit.")
            return np.ones_like(residual, dtype=bool)
        lower, upper = self._get_fwhm_bounds(mu, sigma)
        return (residual >= lower) & (residual <= upper)

    def _fit_gaussian_to_histogram(self, hist: np.ndarray, bin_centers: np.ndarray) -> Tuple[float, float, float]:
        try:
            def gaussian(x, A, mu, sigma):
                return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
            A_init = np.max(hist)
            mu_init = bin_centers[np.argmax(hist)]
            sigma_init = np.std(bin_centers)
            popt, _ = curve_fit(gaussian, bin_centers, hist, p0=[A_init, mu_init, sigma_init], maxfev=100000)
            return tuple(popt)
        except Exception as e:
            logger.error(f"Gaussian fitting failed: {e}")
            return (np.nan, np.nan, np.nan)

    def _get_fwhm_bounds(self, mu: float, sigma: float) -> Tuple[float, float]:
        fwhm = 2.355 * sigma
        return mu - fwhm / 2, mu + fwhm / 2
