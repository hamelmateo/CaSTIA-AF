import numpy as np
import logging
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)



class Distribution:
    """
    Represents basic statistical properties of a numerical distribution.

    Attributes:
        values (List[float]): Original distribution values.
        mean (float): Mean of the values.
        std (float): Standard deviation of the values.
        min (float): Minimum value.
        max (float): Maximum value.
        count (int): Number of elements.
    """

    def __init__(self, values: List[float]) -> None:
        """
        Initialize with a list of numerical values.

        Args:
            values (List[float]): Numerical values to analyze.
        """
        self.values: List[float] = values
        self.mean: float = float(np.mean(values)) if values else 0.0
        self.std: float = float(np.std(values)) if values else 0.0
        self.min: float = float(np.min(values)) if values else 0.0
        self.max: float = float(np.max(values)) if values else 0.0
        self.count: int = len(values)

    def __repr__(self) -> str:
        return f"<DistributionStats mean={self.mean:.2f}, std={self.std:.2f}, n={self.count}>"

    def as_dict(self) -> dict:
        """
        Export statistics as a dictionary.

        Returns:
            dict: Dictionary with keys: mean, std, min, max, count.
        """
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "count": self.count
        }

    
    def compute_histogram(self, bin_width: Optional[float] = None, bin_count: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Compute a histogram from a list of values, with either fixed bin width or bin count.

        Args:
            bin_width (Optional[float]): Fixed width for bins. Overrides bin_count if provided.
            bin_count (Optional[int]): Number of bins to use (ignored if bin_width is provided).

        Returns:
            Dict[str, List[float]]: A dictionary with "bins" and "counts" keys.
                                    "bins" contains bin edges, "counts" the frequency in each bin.

        Raises:
            ValueError: If input data is empty or invalid.
        """
        n_total = len(self.values)
        clean_data = [x for x in self.values if x is not None and not np.isnan(x)]
        n_valid = len(clean_data)
        n_filtered = n_total - n_valid

        if n_valid == 0:
            logger.warning(f"Filtered data is empty or non-finite in compute_histogram_func. Filtered out {n_filtered} / {n_total} entries.")
            return {"bins": [], "counts": []}

        if n_filtered > 0:
            logger.warning(f"Filtered out {n_filtered} non-finite values in compute_histogram_func (kept {n_valid} / {n_total}).")

        data_array = np.asarray(clean_data, dtype=float)

        try:
            if bin_width is not None:
                min_val = np.min(data_array)
                max_val = np.max(data_array)
                bins = np.arange(min_val, max_val + bin_width, bin_width)
            else:
                bins = bin_count  # Will be interpreted by np.histogram

            counts, edges = np.histogram(data_array, bins=bins)

            return {
                "bins": edges.tolist(),
                "counts": counts.tolist()
            }

        except Exception as e:
            logger.error(f"Failed to compute histogram: {e}")
            raise


    @classmethod
    def from_values(cls, values: List[float]) -> "Distribution":
        """
        Construct a DistributionStats instance from values.

        Args:
            values (List[float]): List of numerical values.

        Returns:
            DistributionStats: Computed statistics.
        """
        return cls(values)


def compute_histogram_func(data: List[float], bin_width: Optional[float] = None, bin_count: Optional[int] = 10) -> Dict[str, List[float]]:
    """
    Compute a histogram from a list of values, with either fixed bin width or bin count.

    Args:
        data (List[float]): Input numeric data (e.g. peak durations, amplitudes).
        bin_width (Optional[float]): Fixed width for bins. Overrides bin_count if provided.
        bin_count (Optional[int]): Number of bins to use (ignored if bin_width is provided).

    Returns:
        Dict[str, List[float]]: A dictionary with "bins" and "counts" keys.
                                "bins" contains bin edges, "counts" the frequency in each bin.

    Raises:
        ValueError: If input data is empty or invalid.
    """
    n_total = len(data)
    clean_data = [x for x in data if x is not None and not np.isnan(x)]
    n_valid = len(clean_data)
    n_filtered = n_total - n_valid

    if n_valid == 0:
        logger.warning(f"Filtered data is empty or non-finite in compute_histogram_func. Filtered out {n_filtered} / {n_total} entries.")
        return {"bins": [], "counts": []}

    if n_filtered > 0:
        logger.warning(f"Filtered out {n_filtered} non-finite values in compute_histogram_func (kept {n_valid} / {n_total}).")

    data_array = np.asarray(clean_data, dtype=float)

    try:
        if bin_width is not None:
            min_val = np.min(data_array)
            max_val = np.max(data_array)
            bins = np.arange(min_val, max_val + bin_width, bin_width)
        else:
            bins = bin_count  # Will be interpreted by np.histogram

        counts, edges = np.histogram(data_array, bins=bins)

        return {
            "bins": edges.tolist(),
            "counts": counts.tolist()
        }

    except Exception as e:
        logger.error(f"Failed to compute histogram: {e}")
        raise




def compute_peak_frequency_over_time(
    start_times_per_cell: List[List[int]],
    trace_length: int,
    window_size: int = 200,
    step_size: int = 50
) -> List[float]:
    """
    Compute population-level peak frequency evolution over time.

    Args:
        start_times_per_cell (List[List[int]]): Each sublist contains the start times of peaks for a cell.
        trace_length (int): Length of the time trace.
        window_size (int): Size of the sliding window (in frames).
        step_size (int): Step size for moving the window.

    Returns:
        List[float]: Normalized peak frequency (peaks / (cells * window)) per window.
    """
    try:
        if not isinstance(start_times_per_cell, list) or not all(isinstance(lst, list) for lst in start_times_per_cell):
            raise TypeError("start_times_per_cell must be a list of lists of integers.")

        if not isinstance(trace_length, int) or trace_length <= 0:
            raise ValueError("trace_length must be a positive integer.")

        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer.")

        if not isinstance(step_size, int) or step_size <= 0:
            raise ValueError("step_size must be a positive integer.")

        if trace_length < window_size:
            logger.warning("Trace length is shorter than the window size; returning empty frequency evolution.")
            return []

        freq_evo = []
        n_cells = len(start_times_per_cell)

        for start in range(0, trace_length - window_size + 1, step_size):
            end = start + window_size
            peak_count = sum(
                sum(start <= t < end for t in start_times if isinstance(t, int))
                for start_times in start_times_per_cell
            )
            norm_freq = peak_count / (window_size * n_cells) if n_cells > 0 else 0.0
            freq_evo.append(norm_freq)

        return freq_evo

    except Exception as e:
        logger.error(f"Failed to compute peak frequency evolution: {e}")
        return []