import numpy as np
from calcium_activity_characterization.logger import logger
from typing import Literal
import matplotlib.pyplot as plt
import pandas as pd





class Distribution:
    """
    Represents basic statistical properties of a numerical distribution.

    Attributes:
        values (list[float]): Original distribution values.
        mean (float): Mean of the values.
        std (float): Standard deviation of the values.
        min (float): Minimum value.
        max (float): Maximum value.
        count (int): Number of elements.
        binning_mode (str): Either 'count' or 'width'.
        bin_param (float): Value used for bins (count or width).
    """

    def __init__(self, values: list[float], binning_mode: Literal["count", "width"], bin_param: float) -> None:
        """
        Initialize with a list of numerical values.

        Args:
            values (list[float]): Numerical values to analyze.
        """
        self.values: list[float] = values
        self.mean: float = float(np.mean(values)) if values else 0.0
        self.std: float = float(np.std(values)) if values else 0.0
        self.min: float = float(np.min(values)) if values else 0.0
        self.max: float = float(np.max(values)) if values else 0.0
        self.count: int = len(values)
        self.binning_mode: Literal["count", "width"] = binning_mode
        self.bin_param: float = bin_param

    def __repr__(self) -> str:
        return f"<DistributionStats mean={self.mean:.2f}, std={self.std:.2f}, n={self.count}>"

    def as_dict(self) -> dict:
        """
        Export statistics as a dictionary.

        Returns:
            dict: dictionary with keys: mean, std, min, max, count.
        """
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "count": self.count
        }

    
    def compute_histogram(self) -> dict[str, list[float]]:
        """
        Compute a histogram from a list of values, with either fixed bin width or bin count.

        Args:
            bin_width (Optional[float]): Fixed width for bins. Overrides bin_count if provided.
            bin_count (Optional[int]): Number of bins to use (ignored if bin_width is provided).

        Returns:
            dict[str, list[float]]: A dictionary with "bins" and "counts" keys.
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
            if self.binning_mode == "count":
                bins = self.bin_param  # Will be interpreted by np.histogram
            elif self.binning_mode == "width":
                start = np.floor(min(self.values))  # ensure bin starts at an integer below min
                stop = np.ceil(max(self.values))    # ensure bin ends at an integer above max
                bins = np.arange(start, stop, self.bin_param)
            else:
                raise logger.error(f"Invalid binning mode: {self.binning_mode}. Use 'count' or 'width'.")
            
            counts, edges = np.histogram(data_array, bins=bins)

            return {
                "bins": edges.tolist(),
                "counts": counts.tolist()
            }

        except Exception as e:
            logger.error(f"Failed to compute histogram: {e}")
            raise

    def plot_histogram(self, title: str, xlabel: str) -> plt.Figure:
        """
        Generate a matplotlib histogram with overlaid stats text.

        Args:
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.

        Returns:
            plt.Figure: The matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        hist = self.compute_histogram()

        if hist["bins"] and hist["counts"]:
            ax.bar(hist["bins"][:-1], hist["counts"],width=hist["bins"][1] - hist["bins"][0],
                   align='edge', color="steelblue")

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.grid(True)

        ax.text(0.98, 0.98, self.get_stats_text(), transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

        return fig
    
    def get_stats_text(self) -> str:
        """Return stats as multiline string for display."""
        return (
            f"Mean: {self.mean:.3f}\n"
            f"Std: {self.std:.3f}\n"
            f"Min: {self.min:.3f}\n"
            f"Max: {self.max:.3f}\n"
            f"Count: {self.count}"
        )

    @classmethod
    def from_values(cls, values: list[float], binning_mode: Literal["count", "width"] | None = "count", bin_param: float | None = 10) -> "Distribution":
        """
        Construct a DistributionStats instance from values.

        Args:
            values (list[float]): list of numerical values.

        Returns:
            DistributionStats: Computed statistics.
        """
        filtered = [v for v in values if v is not None]
        n_filtered = len([v for v in values if v is None])
        if n_filtered > 0:
            logger.info(f"Filtered out {n_filtered} None values in from_values (kept {len(filtered)} / {len(values)}).")
        return cls(filtered,binning_mode=binning_mode, bin_param=bin_param)

def compute_histogram_func(data: list[float], bin_width: float | None = None, bin_count: int | None = 10) -> dict[str, list[float]]:
    """
    Compute a histogram from a list of values, with either fixed bin width or bin count.

    Args:
        data (list[float]): Input numeric data (e.g. peak durations, amplitudes).
        bin_width (Optional[float]): Fixed width for bins. Overrides bin_count if provided.
        bin_count (Optional[int]): Number of bins to use (ignored if bin_width is provided).

    Returns:
        dict[str, list[float]]: A dictionary with "bins" and "counts" keys.
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
    start_times_per_cell: list[list[int]],
    trace_length: int,
    window_size: int = 200,
    step_size: int = 50
) -> list[float]:
    """
    Compute population-level peak frequency evolution over time.

    Args:
        start_times_per_cell (list[list[int]]): Each sublist contains the start times of peaks for a cell.
        trace_length (int): Length of the time trace.
        window_size (int): Size of the sliding window (in frames).
        step_size (int): Step size for moving the window.

    Returns:
        list[float]: Normalized peak frequency (peaks / (cells * window)) per window.
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
    

def detect_asymmetric_iqr_outliers(
    df: pd.DataFrame = None,
    column: str = None,
    k_lower: float = 1.5,
    k_upper: float = 3.0
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """
    Detect outliers using asymmetric IQR fences.

    Args:
        df (pd.DataFrame): DataFrame to analyze.
        column (str): Column name to check for outliers.
        k_lower (float): Multiplier for lower bound (default 1.5).
        k_upper (float): Multiplier for upper bound (default 3.0).
        return_rows (bool): If True, return full rows for outliers (requires df).

    Returns:
        tuple: (inliers, outliers, lower_bound, upper_bound)
    """
    data = df[column]
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - k_lower * iqr
    upper_bound = q3 + k_upper * iqr

    df_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    df_inliers = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df_inliers, df_outliers, lower_bound, upper_bound