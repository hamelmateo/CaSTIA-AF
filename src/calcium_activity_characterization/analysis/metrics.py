import numpy as np
from calcium_activity_characterization.logger import logger
from typing import Literal, Optional
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
    df: pd.DataFrame | None = None,
    column: str | None = None,
    k_lower: float = 1.5,
    k_upper: float = 3.0,
    groupby_col: Optional[str] = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """
    Detect outliers using asymmetric IQR fences, optionally **per group**.

    If `groupby_col` is provided, the IQR and bounds are computed independently for
    each group, and inliers/outliers are concatenated. The function also returns a
    *global* lower/upper bound computed as the min/max across groups for convenience.

    Args:
        df: DataFrame to analyze (must contain `column`; and `groupby_col` if used).
        column: Numeric column name to check for outliers.
        k_lower: Multiplier for the lower IQR fence (default 1.5).
        k_upper: Multiplier for the upper IQR fence (default 3.0).
        groupby_col: Column to group by when computing fences. If None, computes on
            the full population.
        verbose: If True, logs detailed per-group/global info.

    Returns:
        Tuple:
            - inliers (pd.DataFrame): rows within fences (group-wise if applicable)
            - outliers (pd.DataFrame): rows outside fences (group-wise if applicable)
            - lower_bound (float): global lower fence = min of per-group lowers
            - upper_bound (float): global upper fence = max of per-group uppers
    """
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input `df` must be a non-empty pandas DataFrame.")
        if column is None:
            raise ValueError("Parameter `column` must be provided.")
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")
        if groupby_col is not None and groupby_col not in df.columns:
            raise KeyError(f"groupby_col '{groupby_col}' not found in DataFrame.")

        work = df.copy()
        work[column] = pd.to_numeric(work[column], errors="coerce")
        work = work.dropna(subset=[column])
        if work.empty:
            logger.warning("detect_asymmetric_iqr_outliers: no valid numeric rows in '%s'.", column)
            return work, work, float("nan"), float("nan")

        def _bounds(series: pd.Series) -> tuple[float, float]:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - k_lower * iqr
            upper = q3 + k_upper * iqr
            return float(lower), float(upper)

        if groupby_col is None:
            lower, upper = _bounds(work[column])
            mask_out = (work[column] < lower) | (work[column] > upper)
            df_outliers = work.loc[mask_out]
            df_inliers = work.loc[~mask_out]
            if verbose:
                logger.info(
                    "Global fences on '%s' -> lower=%.5g, upper=%.5g; "
                    "outliers=%d / %d rows",
                    column, lower, upper, len(df_outliers), len(work)
                )
            return df_inliers, df_outliers, lower, upper

        inliers_list: list[pd.DataFrame] = []
        outliers_list: list[pd.DataFrame] = []
        lowers: list[float] = []
        uppers: list[float] = []

        for key, g in work.groupby(groupby_col, dropna=False):
            if g.empty:
                continue
            lower_g, upper_g = _bounds(g[column])
            lowers.append(lower_g)
            uppers.append(upper_g)

            mask_out_g = (g[column] < lower_g) | (g[column] > upper_g)
            outliers_g = g.loc[mask_out_g]
            inliers_g = g.loc[~mask_out_g]
            inliers_list.append(inliers_g)
            outliers_list.append(outliers_g)

            if verbose:
                logger.info(
                    "Group '%s' -> lower=%.5g, upper=%.5g; "
                    "outliers=%d / %d rows",
                    key, lower_g, upper_g, len(outliers_g), len(g)
                )

        df_inliers = pd.concat(inliers_list, axis=0, ignore_index=False) if inliers_list else work.iloc[0:0]
        df_outliers = pd.concat(outliers_list, axis=0, ignore_index=False) if outliers_list else work.iloc[0:0]

        global_lower = float(np.nanmin(lowers)) if lowers else float("nan")
        global_upper = float(np.nanmax(uppers)) if uppers else float("nan")

        if verbose:
            logger.info(
                "Grouped by '%s' -> global lower=%.5g, upper=%.5g; "
                "total outliers=%d / %d",
                groupby_col, global_lower, global_upper, len(df_outliers), len(work)
            )

        return df_inliers, df_outliers, global_lower, global_upper

    except Exception as exc:
        logger.exception("detect_asymmetric_iqr_outliers failed: %s", exc)
        empty = df.iloc[0:0] if isinstance(df, pd.DataFrame) else pd.DataFrame()
        return empty, empty, float("nan"), float("nan")
