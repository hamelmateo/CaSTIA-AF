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
from typing import Callable, Optional, Tuple
import logging
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

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
        Initialize the SimilarityMatrixComputer.

        Args:
            config (dict): Configuration dictionary.
        """
        self.method: str = config["method"]
        self.params: dict = config["params"].get(self.method, {})
        self.parallelize: bool = config.get("parallelize", True)

    def compute(self, traces: np.ndarray, lag_range: Optional[int] = None) -> np.ndarray:
        """
        Compute similarity matrix using selected method and config parallelization flag.

        Args:
            traces (np.ndarray): Input signals (num_cells x num_timepoints).
            lag_range (int, optional): Lag window for similarity.
        
        Returns:
            np.ndarray: Similarity matrix.
        """
        if self.parallelize:
            return self._compute_parallel(traces, lag_range)
        else:
            return self._compute_serial(traces, lag_range)

    def _compute_parallel(self, traces: np.ndarray, lag_range: Optional[int] = None, n_jobs: int = None) -> np.ndarray:
            """
            Compute similarity matrix in parallel.

            Args:
                traces (np.ndarray): Input signals (num_cells x num_timepoints).
                lag_range (int, optional): Lag window for similarity.
                n_jobs (int, optional): Number of parallel workers.

            Returns:
                np.ndarray: Similarity matrix.
            """
            n = traces.shape[0]
            pairs = [(i, j) for i in range(n) for j in range(i, n)]
            tasks = [(i, j, traces[i], traces[j], self.method, self.params, lag_range) for (i, j) in pairs]

            results = [None] * len(tasks)

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(SimilarityMatrixComputer.compute_similarity_pair, task): idx
                    for idx, task in enumerate(tasks)
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="Computing similarity matrix"):
                    idx = futures[future]
                    results[idx] = future.result()

            sim_matrix = np.zeros((n, n))
            for (i, j), val in zip(pairs, results):
                sim_matrix[i, j] = sim_matrix[j, i] = val
            return sim_matrix

    def _compute_serial(self, traces: np.ndarray, lag_range: Optional[int] = None) -> np.ndarray:
        """
        Compute similarity matrix in serial using tqdm.

        Args:
            traces (np.ndarray): Input signals (num_cells x num_timepoints).
            lag_range (int, optional): Lag window for similarity.

        Returns:
            np.ndarray: Similarity matrix.
        """
        n = traces.shape[0]
        sim_matrix = np.zeros((n, n))

        print(f"Serial computing similarity matrix for {n*(n+1)//2} pairs of traces using {self.method} method...")

        for i in tqdm(range(n), desc=f"Computing (serial, {self.method})"):
            for j in range(i, n):
                args = (i, j, traces[i], traces[j], self.method, self.params, lag_range)
                sim = self.compute_similarity_pair(args)
                sim_matrix[i, j] = sim_matrix[j, i] = sim

        return sim_matrix

    @staticmethod
    def compute_similarity_pair(args: Tuple[int, int, np.ndarray, np.ndarray, str, dict, Optional[int]]) -> float:
        """
        Compute similarity for a pair of traces.

        Args:
            args (tuple): (i, j, trace_i, trace_j, method, params, lag_range)

        Returns:
            float: Similarity score.
        """
        i, j, a, b, method, params, lag_range = args
        def norm_cross_corr(a, b):
            mode = params.get("mode", "full")
            method_type = params.get("method", "direct")
            corr = correlate(a, b, mode=mode, method=method_type)
            lags = np.arange(-len(b) + 1, len(a))
            if lag_range is not None:
                corr = corr[(lags >= -lag_range) & (lags <= lag_range)]
            return np.max(corr) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8)

        def jaccard(a, b):
            a_bin, b_bin = a > 0, b > 0
            inter = np.logical_and(a_bin, b_bin).sum()
            union = np.logical_or(a_bin, b_bin).sum()
            return inter / union if union > 0 else 0.0

        def max_lagged(a, b, fn):
            T = len(a)
            max_sim = -np.inf
            for lag in range(-lag_range, lag_range + 1):
                if lag < 0:
                    a_lag, b_lag = a[:lag], b[-lag:]
                elif lag > 0:
                    a_lag, b_lag = a[lag:], b[:-lag]
                else:
                    a_lag, b_lag = a, b
                if len(a_lag) < 2: continue
                try:
                    max_sim = max(max_sim, fn(a_lag, b_lag))
                except Exception: continue
            return max_sim if max_sim != -np.inf else 0.0

        if method == "cross_correlation":
            return norm_cross_corr(a, b)
        elif method == "jaccard":
            return max_lagged(a, b, jaccard)
        elif method == "pearson":
            return max_lagged(a, b, lambda x, y: pearsonr(x, y)[0])
        elif method == "spearman":
            return max_lagged(a, b, lambda x, y: spearmanr(x, y)[0])
        else:
            raise ValueError(f"Unsupported method: {method}")

    def _compute_cross_correlation(self, traces: np.ndarray, lag_range: Optional[int] = None) -> np.ndarray:
        """
        Compute normalized cross-correlation similarity matrix.

        Args:
            traces (np.ndarray): Calcium traces.
            lag_range (int, optional): Time lag window.

        Returns:
            np.ndarray: Similarity matrix.
        """
        return self.compute_parallel(traces, lag_range)

    def _compute_lagged_similarity(self, traces: np.ndarray, metric_fn: Callable[[np.ndarray, np.ndarray], float], lag_range: int) -> np.ndarray:
        """
        Compute lagged metric-based similarity matrix.

        Args:
            traces (np.ndarray): Input traces.
            metric_fn (Callable): Metric function.
            lag_range (int): Max lag.

        Returns:
            np.ndarray: Similarity matrix.
        """
        return self.compute_parallel(traces, lag_range)

    def _pearson_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute Pearson correlation.

        Args:
            a (np.ndarray): Signal 1.
            b (np.ndarray): Signal 2.

        Returns:
            float: Correlation score.
        """
        r, _ = pearsonr(a, b)
        return r

    def _spearman_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute Spearman rank correlation.

        Args:
            a (np.ndarray): Signal 1.
            b (np.ndarray): Signal 2.

        Returns:
            float: Rank correlation.
        """
        r, _ = spearmanr(a, b)
        return r

    def _jaccard_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute Jaccard index.

        Args:
            a (np.ndarray): Signal 1.
            b (np.ndarray): Signal 2.

        Returns:
            float: Jaccard similarity.
        """
        a_bin, b_bin = a > 0, b > 0
        intersection = np.logical_and(a_bin, b_bin).sum()
        union = np.logical_or(a_bin, b_bin).sum()
        return intersection / union if union > 0 else 0.0
