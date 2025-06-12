"""
Module for running Granger causality (pairwise or multivariate) on calcium activity clusters.

Example usage:
    >>> from calcium_activity_characterization.analysis.gc_analyzer import GCAnalyzer
    >>> config = {
    ...     "mode": "pairwise",
    ...     "trace": "gc_trace",
    ...     "parameters": {
    ...         "pairwise": {
    ...             "lag_order": 3,
    ...             "pvalue_threshold": 0.001,
    ...             "threshold_links": True,
    ...             "window_size": 150,
    ...             "min_cells": 3
    ...         }
    ...     }
    }
    >>> analyzer = GCAnalyzer(config)
    >>> cells = [...]  # List of Cell objects
    >>> center_time = 1000  # Example center time
    >>> gc_matrix = analyzer.run(cells, center_time)
"""

import logging
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Optional

from calcium_activity_characterization.data.cells import Cell


logger = logging.getLogger(__name__)


class GCAnalyzer:
    """
    General-purpose analyzer for computing Granger causality (GC), supporting both
    pairwise and multivariate modes.
    """

    def __init__(self, config: dict, DEVICES_CORES: int = 4):
        """
        Initialize the GC analyzer.

        Args:
            config (dict): Configuration dictionary with keys:
                - 'mode': 'pairwise' or 'multivariate'
                - 'parameters': dict with mode-specific parameters
        """
        self.DEVICES_CORES = DEVICES_CORES

        self.mode = config.get("mode", "pairwise")
        self.trace = config.get("trace", "gc_trace")
        self.params = config.get("parameters", {}).get(self.mode, {})


    def run(self, cells: List[Cell], center_time: int) -> Optional[np.ndarray]:
        """
        Run the GC analysis on a cluster of cells.

        Args:
            cells (List[Cell]): Cluster member cells.
            center_time (int): Center time of the analysis window.

        Returns:
            Optional[np.ndarray]: GC matrix (NxN) or None.
        """
        trace_matrix = self._extract_trace_matrix(cells, center_time)
        if trace_matrix is None:
            print("Cluster skipped: insufficient valid traces")
            return None

        if self.mode == "pairwise":
            return self._run_pairwise_gc(trace_matrix)
        elif self.mode == "multivariate":
            return self._run_multivariate_gc(trace_matrix)
        else:
            raise ValueError(f"Unsupported GC mode: {self.mode}")

    def _extract_trace_matrix(self, cells: List[Cell], center_time: int) -> Optional[np.ndarray]:
        """
        Extract a trace matrix from the cells' GC traces, sorted by label.

        Args:
            cells (List[Cell]): Cluster member cells.
            center_time (int): Center time of the analysis window.
        
        Returns:
            Optional[np.ndarray]: Trace matrix (T, N) or None if insufficient data.
        """
        self.window_size = self.params.get("window_size", 150)
        self.min_cells = self.params.get("min_cells", 3)

        # Sort cells by label
        sorted_cells = sorted(cells, key=lambda cell: cell.label)
        trace_length = max(len(cell.trace.versions["gc_trace"]) for cell in sorted_cells if hasattr(cell, self.trace))
        half = self.window_size // 2
        start = max(0, center_time - half)
        end = start + self.window_size
        if end > trace_length:
            end = trace_length
            start = max(0, end - self.window_size)

        traces = []
        for cell in sorted_cells:
                trace = getattr(cell, self.trace, [])
                if len(trace) < end:
                    return None
                window = trace[start:end]
                if np.std(window) > 0:
                    traces.append(window)

        if len(traces) < self.min_cells:
            return None

        return np.array(traces).T  # shape (T, N)

    def _run_multivariate_gc(self, trace_matrix: np.ndarray) -> np.ndarray:
        """
        Run multivariate Granger causality on the trace matrix.

        Args:
            trace_matrix (np.ndarray): Trace matrix (T, N).

        Returns:
            np.ndarray: GC matrix (N, N) with causality test statistics.
        """

        try:
            self.lag_order = self.params.get("lag_order", 5)
            model = VAR(trace_matrix)
            results = model.fit(maxlags=self.lag_order)
            N = trace_matrix.shape[1]
            gc_matrix = np.zeros((N, N))

            for caused in range(N):
                for causing in range(N):
                    if caused == causing:
                        continue
                    test = results.test_causality(caused, causing, kind='f')
                    gc_matrix[causing, caused] = test.test_statistic

            return gc_matrix
        except Exception as e:
            print(f"Multivariate GC failed: {e}")
            return np.zeros((trace_matrix.shape[1], trace_matrix.shape[1]))

    def _run_pairwise_gc(self, trace_matrix: np.ndarray) -> np.ndarray:
        """
        Run pairwise Granger causality on the trace matrix.

        Args:
            trace_matrix (np.ndarray): Trace matrix (T, N).

        Returns:
            np.ndarray: GC matrix (N, N) with causality test statistics.
        """
        self.lag_order = self.params.get("lag_order", 3)
        self.pvalue_threshold = self.params.get("pvalue_threshold", 0.001)
        self.threshold_links = self.params.get("threshold_links", True)

        N = trace_matrix.shape[1]
        gc_matrix = np.zeros((N, N))

        tasks = [
            (i, j, trace_matrix[:, j], trace_matrix[:, i], self.lag_order, self.pvalue_threshold, self.threshold_links)
            for i in range(N) for j in range(N) if i != j
        ]

        with ProcessPoolExecutor(max_workers=self.DEVICES_CORES) as executor:
            futures = {
                executor.submit(GCAnalyzer._compute_pairwise_gc, i, j, x, y, lag, pval, thresh): (i, j)
                for (i, j, x, y, lag, pval, thresh) in tasks
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Running pairwise GC"):
                i, j = futures[future]
                try:
                    gc_matrix[i, j] = future.result()
                except Exception as e:
                    print(f"GC failed for {i}->{j}: {e}")
                    gc_matrix[i, j] = 0

        return gc_matrix
    

    @staticmethod

    def _compute_pairwise_gc(i, j, x, y, lag_order, pvalue_threshold, threshold_links):
        data = np.column_stack([y, x])  # y, x
        result = grangercausalitytests(data, maxlag=lag_order, verbose=False)
        pvalue = result[lag_order][0]['ssr_ftest'][1]
        return pvalue if not threshold_links or pvalue < pvalue_threshold else 0.0