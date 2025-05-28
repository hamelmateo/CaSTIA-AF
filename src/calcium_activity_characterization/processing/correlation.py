"""
Module for slicing calcium traces and computing time-lagged similarity matrices
across multiple time windows using the SimilarityMatrixComputer.

Example usage:
    >>> from calcium_activity_characterization.processing.correlation import CorrelationAnalyzer
    >>> from config import CORRELATION_PARAMETERS
    >>> analyzer = CorrelationAnalyzer(CORRELATION_PARAMETERS)
    >>> similarity_matrices = analyzer.run(cells=active_cells)
"""

import numpy as np
from typing import List
import logging
from calcium_activity_characterization.processing.similarity_matrix import SimilarityMatrixComputer
from calcium_activity_characterization.data.cells import Cell

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Analyze cell-cell similarity over time by computing similarity matrices
    in a sliding-window, time-lagged manner.

    All traces used are binarized (cell.trace.binary).
    """

    def __init__(self, config: dict, DEVICES_CORES: int = 4):
        """
        Initialize the CorrelationAnalyzer with time windowing and similarity settings.

        Args:
            config (dict): CORRELATION_PARAMETERS config block
        """
        self.DEVICES_CORES = DEVICES_CORES

        self.window_size = config.get("window_size", 1800)
        self.step_percent = config.get("step_percent", 0.1)
        self.lag_percent = config.get("lag_percent", 0.1)
        self.parallelize = config.get("parallelize", True)

        self.step_size = int(self.step_percent * self.window_size)
        self.lag_range = int(self.lag_percent * self.window_size)

        self.similarity_computer = SimilarityMatrixComputer(config)

    def run(self, cells: List[Cell], single_window: bool = False) -> List[np.ndarray]:
        """
        Compute similarity matrices across time windows.

        Args:
            cells (List[Cell]): List of active Cell objects with binarized traces.
            single_window (bool): If True, compute one similarity matrix over full trace.

        Returns:
            List[np.ndarray]: List of similarity matrices, one per window.
        """
        if not cells:
            logger.warning("No cells provided for correlation analysis.")
            return []

        binary_traces = [np.array(cell.trace.binary, dtype=int) for cell in cells]
        trace_length = len(binary_traces[0])
        trace_matrix = np.stack(binary_traces, axis=0)

        if single_window:
            logger.info("Computing correlation over full trace.")
            sim_matrix = self.similarity_computer.compute(trace_matrix, lag_range=self.lag_range)
            return [sim_matrix]

        if trace_length < self.window_size:
            logger.warning("Trace shorter than window size; skipping analysis.")
            return []

        similarity_matrices = []
        for start in range(0, trace_length - self.window_size + 1, self.step_size):
            window = trace_matrix[:, start : start + self.window_size]
            sim = self.similarity_computer.compute(traces=window, lag_range=self.lag_range)
            similarity_matrices.append(sim)

        return similarity_matrices