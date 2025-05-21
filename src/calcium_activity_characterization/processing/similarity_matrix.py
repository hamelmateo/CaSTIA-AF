"""
Module for computing similarity matrices between calcium signal traces.

Example usage:
    >>> from calcium_activity_characterization.processing.similarity_matrix import SimilarityMatrixComputer
    >>> from config import CORRELATION_PARAMETERS
    >>> 
    >>> traces = np.array([[0, 1, 1, 0, 1], [1, 1, 0, 0, 1]])
    >>> computer = SimilarityMatrixComputer(CORRELATION_PARAMETERS)
    >>> sim_matrix = computer.compute(traces, lag_range=10)
"""

import numpy as np
from scipy.signal import correlate
from scipy.stats import pearsonr, spearmanr
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)


class SimilarityMatrixComputer:
    """
    Class for computing pairwise similarity matrices between calcium activity traces.

    Supports multiple similarity metrics:
        - cross_correlation: Normalized maximum cross-correlation across lag range.
        - jaccard: Jaccard index on binarized signals, evaluated across lag range.
        - pearson: Pearson correlation, evaluated across lag range.
        - spearman: Spearman rank correlation, evaluated across lag range.

    Parameters:
        config (dict): Configuration dictionary with 'method' and per-method parameters.
    """

    def __init__(self, config: dict):
        """
        Initialize the SimilarityMatrixComputer with configuration parameters.

        Args:
            config (dict): Configuration dictionary containing method and parameter definitions.
        """
        self.method: str = config["method"]
        self.params: dict = config["params"].get(self.method, {})

    def compute(self, traces: np.ndarray, lag_range: Optional[int] = None) -> np.ndarray:
        """
        Compute a similarity matrix from an array of traces.

        Args:
            traces (np.ndarray): Array of shape (num_cells, num_timepoints).
            lag_range (int, optional): Maximum time lag to consider for similarity.

        Returns:
            np.ndarray: Symmetric similarity matrix of shape (num_cells, num_cells).
        """
        if self.method == "cross_correlation":
            return self._compute_cross_correlation(traces, lag_range)
        elif self.method == "jaccard":
            return self._compute_lagged_similarity(traces, self._jaccard_similarity, lag_range)
        elif self.method == "pearson":
            return self._compute_lagged_similarity(traces, self._pearson_similarity, lag_range)
        elif self.method == "spearman":
            return self._compute_lagged_similarity(traces, self._spearman_similarity, lag_range)
        else:
            raise ValueError(f"Unsupported similarity method: {self.method}")

    def _compute_cross_correlation(self, traces: np.ndarray, lag_range: Optional[int]) -> np.ndarray:
        """
        Compute cross-correlation-based similarity matrix.

        For each pair of traces, shifts one relative to the other across lag_range,
        computes normalized cross-correlation, and returns the maximum value.

        Args:
            traces (np.ndarray): 2D array of shape (n_cells, timepoints).
            lag_range (int): Max number of timepoints to lag for comparison.

        Returns:
            np.ndarray: Similarity matrix of shape (n_cells, n_cells).
        """
        mode = self.params.get("mode", "full")
        method = self.params.get("method", "direct")

        n_cells = traces.shape[0]
        sim_matrix = np.zeros((n_cells, n_cells), dtype=float)

        for i in range(n_cells):
            for j in range(i, n_cells):
                trace_i = traces[i]
                trace_j = traces[j]
                corr = correlate(trace_i, trace_j, mode=mode, method=method)
                lags = np.arange(-len(trace_j) + 1, len(trace_i))
                if lag_range is not None:
                    valid = (lags >= -lag_range) & (lags <= lag_range)
                    corr = corr[valid]
                max_corr = np.max(corr) / ((np.linalg.norm(trace_i) * np.linalg.norm(trace_j)) + 1e-8)
                sim_matrix[i, j] = sim_matrix[j, i] = max_corr

        return sim_matrix

    def _compute_lagged_similarity(self, traces: np.ndarray, metric_fn: Callable[[np.ndarray, np.ndarray], float], lag_range: int) -> np.ndarray:
        """
        Compute pairwise similarity matrix using custom metric function across time lags.

        Args:
            traces (np.ndarray): 2D array of shape (n_cells, timepoints).
            metric_fn (Callable): Function to compute similarity between two aligned traces.
            lag_range (int): Maximum lag to consider in both directions.

        Returns:
            np.ndarray: Similarity matrix of shape (n_cells, n_cells).
        """
        n_cells = traces.shape[0]
        sim_matrix = np.zeros((n_cells, n_cells), dtype=float)

        for i in range(n_cells):
            for j in range(i, n_cells):
                a, b = traces[i], traces[j]
                max_sim = self._max_lagged_similarity(a, b, metric_fn, lag_range)
                sim_matrix[i, j] = sim_matrix[j, i] = max_sim

        return sim_matrix

    def _max_lagged_similarity(self, a: np.ndarray, b: np.ndarray, metric_fn: Callable[[np.ndarray, np.ndarray], float], lag_range: int) -> float:
        """
        Compute the maximum similarity between two signals across lags.

        Args:
            a, b (np.ndarray): Signals to compare.
            metric_fn (Callable): Similarity function.
            lag_range (int): Maximum shift to apply.

        Returns:
            float: Maximum similarity score across all valid lags.
        """
        T = len(a)
        max_sim = -np.inf

        for lag in range(-lag_range, lag_range + 1):
            if lag < 0:
                a_lag, b_lag = a[:lag], b[-lag:]
            elif lag > 0:
                a_lag, b_lag = a[lag:], b[:-lag]
            else:
                a_lag, b_lag = a, b

            if len(a_lag) < 2:
                continue

            try:
                sim = metric_fn(a_lag, b_lag)
                max_sim = max(max_sim, sim)
            except Exception:
                continue

        return max_sim if max_sim != -np.inf else 0.0

    def _pearson_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute Pearson correlation coefficient between two signals.

        Args:
            a, b (np.ndarray): Aligned 1D arrays.

        Returns:
            float: Pearson correlation.
        """
        r, _ = pearsonr(a, b)
        return r

    def _spearman_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute Spearman rank correlation between two signals.

        Args:
            a, b (np.ndarray): Aligned 1D arrays.

        Returns:
            float: Spearman rank correlation.
        """
        r, _ = spearmanr(a, b)
        return r

    def _jaccard_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute Jaccard similarity between binarized versions of two signals.

        Args:
            a, b (np.ndarray): Aligned 1D arrays.

        Returns:
            float: Jaccard index.
        """
        a_bin = np.array(a > 0, dtype=bool)
        b_bin = np.array(b > 0, dtype=bool)
        intersection = np.logical_and(a_bin, b_bin).sum()
        union = np.logical_or(a_bin, b_bin).sum()
        return intersection / union if union > 0 else 0.0