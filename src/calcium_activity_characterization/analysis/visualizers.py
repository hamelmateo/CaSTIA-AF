import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import imread
from pathlib import Path
from matplotlib import cm
from typing import Optional, Callable, Union
from calcium_activity_characterization.logger import logger
from calcium_activity_characterization.analysis.metrics import detect_asymmetric_iqr_outliers


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str = "Count",
    bin_width: Optional[float] = None,
    bin_count: Optional[int] = None,
    log_scale_datasets: list[str] = [],  # If non-empty, y-axis will be log-scaled
    x_axis_boundaries: Optional[tuple[float, float]] = None,
    y_axis_boundaries: Optional[tuple[float, float]] = None,
    filter_outliers: bool = False,
    outliers_bounds: Optional[tuple[float, float]] = None,
    return_outliers: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Plot a **single** histogram for `column` using the provided DataFrame `df`.

    Args:
        df (pd.DataFrame): Input data. Must contain the `column`.
        column (str): Column name to plot.
        title (str): Global title of the plot.
        ylabel (str, optional): Y-axis label. Defaults to "Count".
        bin_width (float, optional): Width of histogram bins. If provided, overrides `bin_count`.
        bin_count (int, optional): Number of bins (used only when `bin_width` is None).
        log_scale_datasets (list[str], optional): If non-empty, y-axis will be log-scaled.
        horizontal_layout (bool, optional): Ignored (kept for backward compatibility).
        x_axis_boundaries (tuple[float, float], optional): X-axis limits as (min, max).
        y_axis_boundaries (tuple[float, float], optional): Y-axis limits as (min, max).
        filter_outliers (bool, optional): If True, remove outliers using asymmetric-IQR before plotting.
        outliers_bounds (tuple[float, float], optional): Quantile bounds for outlier detection (lower, upper).
            If None, defaults to (0.25, 0.75).
        return_outliers (bool, optional): If True, return DataFrame of removed outliers.

    Returns:
        Optional[pd.DataFrame]: DataFrame of removed outliers if `return_outliers` is True; otherwise None.

    Raises:
        KeyError: If `column` is not in `df`.
        ValueError: If `df` is empty or `column` has no valid (non-NaN) values.
    """
    try:
        if df is None or df.empty:
            logger.warning(f"No data to plot for column '{column}' — DataFrame is empty.")
            raise ValueError("Input DataFrame is empty.")

        if column not in df.columns:
            logger.error(f"Column '{column}' not found in DataFrame columns: {list(df.columns)}")
            raise KeyError(f"Column '{column}' not found in DataFrame.")

        data = df[column].dropna()
        if data.empty:
            logger.warning(f"Column '{column}' has no valid values (all NaN after dropna()).")
            raise ValueError(f"Column '{column}' has no valid values.")

        removed_outliers_frames: pd.DataFrame = pd.DataFrame()

        if filter_outliers:
            q_lower, q_upper = (outliers_bounds if outliers_bounds is not None else (0.25, 0.75))
            try:
                filtered_df, outliers, lower_bound, upper_bound = detect_asymmetric_iqr_outliers(
                    df.copy(), column, q_lower, q_upper
                )
                data = filtered_df[column].dropna()
                removed_outliers_frames = outliers.copy()
                logger.info(
                    "plot_histogram: removed %d outliers out of %d on '%s' (lower=%.5g, upper=%.5g)",
                    len(outliers), df.shape[0], column, lower_bound, upper_bound
                )
            except Exception as e:
                logger.exception("Outlier detection failed for column '%s': %s", column, e)
                # Fail safe: continue without filtering

        # Determine bins
        bins = None
        if bin_width is not None:
            try:
                min_val, max_val = float(np.min(data)), float(np.max(data))
                width = float(bin_width)
                if width <= 0:
                    raise ValueError("bin_width must be > 0")
                bins = max(1, int(math.ceil((max_val - min_val) / width)))
            except Exception as e:
                logger.exception("Invalid bin_width for column '%s': %s", column, e)
                bins = bin_count if bin_count is not None else 30
        else:
            bins = bin_count if bin_count is not None else 30

        # Plot
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        try:
            sns.histplot(data, bins=bins, kde=False, ax=ax)
        except Exception as e:
            logger.exception("Seaborn histplot failed — falling back to matplotlib: %s", e)
            ax.hist(data.values, bins=bins)

        ax.set_title(title)
        ax.set_xlabel(column)
        ax.set_ylabel(ylabel)

        # Optional axis limits
        if x_axis_boundaries is not None:
            try:
                ax.set_xlim(x_axis_boundaries[0], x_axis_boundaries[1])
            except Exception as e:
                logger.exception("Failed to apply x-axis boundaries %s: %s", x_axis_boundaries, e)
        if y_axis_boundaries is not None:
            try:
                ax.set_ylim(y_axis_boundaries[0], y_axis_boundaries[1])
            except Exception as e:
                logger.exception("Failed to apply y-axis boundaries %s: %s", y_axis_boundaries, e)

        # Log scale (triggered if list is non-empty to preserve signature compatibility)
        if isinstance(log_scale_datasets, list) and len(log_scale_datasets) > 0:
            try:
                ax.set_yscale("log")
            except Exception as e:
                logger.exception("Failed to set log scale for y-axis: %s", e)

        # Stats box
        try:
            stats_text = (
                f"Mean: {data.mean():.3g}\n"
                f"Std: {data.std(ddof=1):.3g}\n"
                f"Min: {data.min():.3g}\n"
                f"Max: {data.max():.3g}\n"
                f"Count: {data.size}"
            )
            ax.text(
                0.98,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                ha="right",
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"),
            )
        except Exception as e:
            logger.exception("Failed to add stats textbox: %s", e)

        plt.tight_layout()
        plt.show()

        if return_outliers:
            if not removed_outliers_frames.empty:
                logger.info(
                    "Returning removed outliers DataFrame with %d entries.",
                    len(removed_outliers_frames),
                )
                return removed_outliers_frames
            else:
                logger.info("Outlier filtering requested, but no outliers were removed.")
                return pd.DataFrame(columns=df.columns)

        return None

    except Exception as e:
        # Top-level safety net — we log and re-raise to aid debugging upstream
        logger.exception("plot_histogram failed: %s", e)
        raise

def visualize_image(
    image_source: dict[str, str],
    image_name: str,
    title: str = "Image Title",
    figsize: tuple = (6, 5),
    show_axes: bool = False
) -> None:
    """
    Display a **single** image.

    Args:
        image_source: Directory path, direct file path, or dict[label -> folder].
        image_name: Raster image name when `image_source` is a directory.
        title: Figure title.
        figsize_per_plot: Ignored (kept to avoid breaking existing calls).
        show_axes: If True, keep axes; otherwise hide.

    Returns:
        None

    Raises:
        FileNotFoundError: If the resolved image file does not exist.
        ValueError: If image cannot be loaded or is empty.
    """
    try:
        dataset_name, source_path = next(iter(image_source.items()))
        img_path = Path(source_path) / image_name

        try:
            img = imread(img_path)
        except Exception as e:
            logger.exception("Failed to read image '%s': %s", img_path, e)
            raise ValueError(f"Could not read image at '{img_path}'") from e

        if img is None or (hasattr(img, "size") and getattr(img, "size", 0) == 0):
            logger.error("Loaded image is empty: '%s'", img_path)
            raise ValueError(f"Loaded image is empty: '{img_path}'")

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img, aspect="auto")
        ax.set_title(f"{title} - {dataset_name}")
        if not show_axes:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.exception("visualize_image failed: %s", e)
        raise


def plot_pie_chart(
    df: pd.DataFrame,
    column: str,
    title: str = "Category Distribution",
    value_label_formatter: Optional[Callable[[float], str]] = None,
    label_prefix: Optional[str] = None,
    palette: str = "tab10",
    figsize: tuple = (6, 5)
) -> None:
    """
    Plot a **single** pie chart showing the distribution of a categorical column for the entire DataFrame.

    This refactors the prior per-dataset faceted version into a one-figure renderer. All rows in `df` are
    considered together—no grouping by `dataset` is performed.

    Args:
        df (pd.DataFrame): DataFrame containing the categorical column.
        column (str): Categorical column to summarize (e.g., 'in_event', 'is_active').
        title (str): Figure title.
        value_label_formatter (callable, optional): Formatter for the percentage text drawn on slices.
            Signature must be ``f(pct: float) -> str`` where ``pct`` is the percentage (0-100).
        label_prefix (str, optional): Optional prefix prepended to category labels.
        palette (str): Name of a matplotlib colormap for slice colors (default: 'tab10').

    Returns:
        None

    Raises:
        KeyError: If `column` is not present in `df`.
        ValueError: If `df` is empty or the column has no valid values.
    """
    try:
        if df is None or df.empty:
            logger.warning("plot_pie_chart: empty DataFrame")
            raise ValueError("Input DataFrame is empty.")
        if column not in df.columns:
            logger.error("plot_pie_chart: column '%s' not found in DataFrame columns: %s", column, list(df.columns))
            raise KeyError(f"Column '{column}' not found in DataFrame.")

        series = df[column].dropna()
        if series.empty:
            logger.warning("plot_pie_chart: column '%s' has no non-NaN values", column)
            raise ValueError(f"Column '{column}' has no valid values.")

        # Determine unique categories and colors (stable ordering)
        try:
            all_categories = sorted(series.unique().tolist())
        except Exception:
            # Fallback in case of un-orderable mixed types
            all_categories = list(pd.unique(series))

        counts = series.value_counts().reindex(all_categories).fillna(0).astype(int)
        total = int(counts.sum())
        if total == 0:
            logger.warning("plot_pie_chart: total count is zero for column '%s'", column)
            raise ValueError("No data to plot (all categories have zero count).")

        # Colors by category using the selected colormap
        try:
            cmap = cm.get_cmap(palette)
            denom = max(len(all_categories) - 1, 1)
            color_map = {cat: cmap(i / denom) for i, cat in enumerate(all_categories)}
            colors = [color_map[c] for c in counts.index]
        except Exception as e:
            logger.exception("plot_pie_chart: failed to build colors from palette '%s': %s", palette, e)
            colors = None  # Let matplotlib choose defaults

        # Labels shown around the pie (category + counts). If a custom formatter is supplied,
        # we follow the previous behavior and skip outer labels to reduce clutter.
        labels = [
            f"{label_prefix + ' ' if label_prefix else ''}{k} ({v})" for k, v in counts.items()
        ]
        pie_labels = labels if value_label_formatter is None else None

        fig, ax = plt.subplots(figsize=figsize)
        try:
            wedges, texts, autotexts = ax.pie(
                counts,
                labels=pie_labels,
                autopct=value_label_formatter or (lambda p: f"{p:.1f}%" if p > 0 else ""),
                startangle=90,
                colors=colors,
            )
        except Exception as e:
            logger.exception("plot_pie_chart: pie rendering failed: %s", e)
            raise

        # Styling
        try:
            if autotexts is not None:
                for autotext in autotexts:
                    autotext.set_color("black")
                    autotext.set_fontsize(9)
            if texts is not None:
                for text in texts:
                    text.set_color("black")
                    text.set_fontsize(9)
        except Exception as e:
            logger.exception("plot_pie_chart: label styling failed: %s", e)

        ax.set_title(title)
        ax.axis('equal')  # ensure pie is a circle
        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.exception("plot_pie_chart failed: %s", e)
        raise


def plot_bar(
    df: pd.DataFrame,
    axis_column: str,
    value_column: str,
    title: str,
    hue_column: Optional[str] = None,
    ylabel: str = "Count",
    xlabel: str = "Dataset",
    rotation: int = 45,
    palette: str = "muted",
) -> None:
    """
    Plot a single bar chart from a tidy DataFrame.

    The function renders a seaborn barplot (with dodge disabled), optionally split by a
    hue column. Input is assumed to be pre-aggregated or already at the desired granularity.

    Args:
        df (pd.DataFrame): DataFrame containing the plotting columns.
        axis_column (str): Column mapped to the x-axis (categorical recommended).
        value_column (str): Column with numeric values to plot on the y-axis.
        title (str): Plot title.
        hue_column (Optional[str]): Optional categorical column to split bars.
        ylabel (str): Y-axis label. Defaults to "Count".
        xlabel (str): X-axis label. Defaults to "Dataset".
        rotation (int): Rotation angle for x-axis tick labels. Defaults to 45 degrees.
        palette (str): Seaborn/Matplotlib palette name. Defaults to "muted".

    Returns:
        None

    Raises:
        KeyError: If required columns are missing.
        ValueError: If the DataFrame is empty or `value_column` has no numeric data after cleaning.
    """
    try:
        if df is None or df.empty:
            logger.warning("plot_bar: received empty DataFrame")
            raise ValueError("Input DataFrame is empty.")

        missing = [c for c in [axis_column, value_column] if c not in df.columns]
        if missing:
            logger.error("plot_bar: missing required columns: %s (available=%s)", missing, list(df.columns))
            raise KeyError(f"Missing required column(s): {missing}")
        if hue_column is not None and hue_column not in df.columns:
            logger.warning("plot_bar: hue_column '%s' not found; proceeding without hue.", hue_column)
            hue_column = None

        # Ensure value column is numeric
        clean_df = df.copy()
        clean_df[value_column] = pd.to_numeric(clean_df[value_column], errors="coerce")
        before = len(clean_df)
        clean_df = clean_df.dropna(subset=[value_column])
        dropped = before - len(clean_df)
        if dropped > 0:
            logger.info("plot_bar: dropped %d rows with non-numeric '%s'", dropped, value_column)
        if clean_df.empty:
            logger.error("plot_bar: no valid numeric data to plot after cleaning '%s'", value_column)
            raise ValueError("No valid numeric data to plot.")

        plt.figure(figsize=(6.5, 4.0))
        try:
            sns.barplot(
                data=clean_df,
                x=axis_column,
                y=value_column,
                hue=hue_column,
                dodge=False,
                palette=palette,
                legend=False,
                errorbar=None,
            )
        except Exception as e:
            logger.exception("plot_bar: seaborn.barplot failed, falling back to matplotlib: %s", e)
            # Fallback: aggregate by (axis, hue) mean
            if hue_column is None:
                grouped = clean_df.groupby(axis_column, as_index=False)[value_column].mean()
                plt.bar(grouped[axis_column], grouped[value_column])
            else:
                grouped = clean_df.groupby([axis_column, hue_column], as_index=False)[value_column].mean()
                # Simple grouped bars fallback
                xcats = grouped[axis_column].unique().tolist()
                hues = grouped[hue_column].unique().tolist()
                x = np.arange(len(xcats))
                width = 0.8 / max(len(hues), 1)
                for i, h in enumerate(hues):
                    sub = grouped[grouped[hue_column] == h]
                    heights = [sub.loc[sub[axis_column] == xc, value_column].mean() for xc in xcats]
                    plt.bar(x + i * width, heights, width=width, label=str(h))
                plt.xticks(x + (len(hues) - 1) * width / 2, xcats)

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        try:
            plt.xticks(rotation=rotation)
        except Exception as e:
            logger.exception("plot_bar: failed to rotate xticks: %s", e)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.exception("plot_bar failed: %s", e)
        raise


def plot_histogram_by_group(
    df: pd.DataFrame,
    value_column: str,
    group_column: str,
    title: str,
    ylabel: str = "Count",
    bin_width: Optional[float] = None,
    bin_count: Optional[int] = None,
    log_scale_datasets: list[str] = [],  # if non-empty, use log y-scale to keep API compatible
    palette: str = "muted",
    multiple: str = "dodge",
    x_axis_boundaries: Optional[tuple[float, float]] = None,
    y_axis_boundaries: Optional[tuple[float, float]] = None,
    filter_outliers: bool = False,
    outliers_bounds: Optional[tuple[float, float]] = None,
    return_outliers: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Plot a **single** histogram of `value_column` for the entire DataFrame, colored by `group_column`.

    Args:
        df (pd.DataFrame): Input data containing `value_column` and `group_column`.
        value_column (str): Numeric column to plot on the x-axis.
        group_column (str): Categorical column used for hue coloring.
        title (str): Figure title.
        ylabel (str): Y-axis label. Defaults to "Count".
        bin_width (float, optional): Bin width; if provided, overrides `bin_count`.
        bin_count (int, optional): Number of bins (used when `bin_width` is None). Defaults to None -> 30.
        log_scale_datasets (list[str]): If non-empty, sets y-axis to log scale.
        palette (str): Seaborn palette name.
        multiple (str): How to render overlaps: 'layer', 'dodge', 'fill', or 'stack'.
        x_axis_boundaries (tuple[float, float], optional): X-axis limits (min, max).
        y_axis_boundaries (tuple[float, float], optional): Y-axis limits (min, max).
        filter_outliers (bool, optional): If True, remove outliers using asymmetric-IQR before plotting.
        outliers_bounds (tuple[float, float], optional): Quantile bounds (lower, upper) for outlier detection.
            If None, defaults to (0.25, 0.75).
        return_outliers (bool, optional): If True, return a DataFrame of removed outliers.

    Returns:
        Optional[pd.DataFrame]: Removed outliers if requested; else None.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If `df` is empty or the filtered plotting data is empty.
    """
    try:
        if df is None or df.empty:
            logger.warning("plot_histogram_by_group: empty DataFrame")
            raise ValueError("Input DataFrame is empty.")

        for col in [value_column, group_column]:
            if col not in df.columns:
                logger.error(
                    "plot_histogram_by_group: column '%s' missing. Available columns: %s",
                    col,
                    list(df.columns),
                )
                raise KeyError(f"Missing required column: {col}")

        # Ensure numeric value column
        work = df[[value_column, group_column]].copy()
        work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
        work = work.dropna(subset=[value_column, group_column])
        if work.empty:
            logger.warning("plot_histogram_by_group: no valid rows after cleaning")
            raise ValueError("No valid rows to plot after cleaning.")

        removed_outliers_frames: pd.DataFrame = pd.DataFrame()

        # Optional outlier filtering
        if filter_outliers:
            q_lower, q_upper = (outliers_bounds if outliers_bounds is not None else (0.25, 0.75))
            try:
                filtered_df, outliers, lower_bound, upper_bound = detect_asymmetric_iqr_outliers(
                    work, value_column, q_lower, q_upper
                )
                logger.info(
                    "plot_histogram_by_group: removed %d outliers out of %d on '%s' (lower=%.5g, upper=%.5g)",
                    len(outliers), work.shape[0], value_column, lower_bound, upper_bound
                )
                work = filtered_df
                removed_outliers_frames = outliers.copy()
            except Exception as e:
                logger.exception("plot_histogram_by_group: outlier detection failed: %s", e)
                # Continue without filtering on error

        # Determine bins
        bins = None
        if bin_width is not None:
            try:
                mn = float(work[value_column].min())
                mx = float(work[value_column].max())
                width = float(bin_width)
                if width <= 0:
                    raise ValueError("bin_width must be > 0")
                bins = max(1, int(math.ceil((mx - mn) / width)))
            except Exception as e:
                logger.exception("plot_histogram_by_group: invalid bin_width: %s", e)
                bins = bin_count if bin_count is not None else 30
        else:
            bins = bin_count if bin_count is not None else 30

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        try:
            sns.histplot(
                data=work,
                x=value_column,
                hue=group_column,
                bins=bins,
                multiple=multiple,
                kde=False,
                ax=ax,
                palette=palette,
                edgecolor=None,
            )
        except Exception as e:
            logger.exception("plot_histogram_by_group: seaborn.histplot failed: %s", e)
            # Minimal fallback: draw a plain histogram without hue
            ax.hist(work[value_column].values, bins=bins)

        ax.set_title(title)
        ax.set_xlabel(value_column)
        ax.set_ylabel(ylabel)

        # Optional axis limits
        if x_axis_boundaries is not None:
            try:
                ax.set_xlim(x_axis_boundaries[0], x_axis_boundaries[1])
            except Exception as e:
                logger.exception("plot_histogram_by_group: failed to set x limits %s: %s", x_axis_boundaries, e)
        if y_axis_boundaries is not None:
            try:
                ax.set_ylim(y_axis_boundaries[0], y_axis_boundaries[1])
            except Exception as e:
                logger.exception("plot_histogram_by_group: failed to set y limits %s: %s", y_axis_boundaries, e)

        # Log y-scale toggle for compatibility with previous API
        if isinstance(log_scale_datasets, list) and len(log_scale_datasets) > 0:
            try:
                ax.set_yscale("log")
            except Exception as e:
                logger.exception("plot_histogram_by_group: failed to set log y-scale: %s", e)

        plt.tight_layout()
        plt.show()

        if return_outliers:
            if not removed_outliers_frames.empty:
                logger.info(
                    "plot_histogram_by_group: returning %d removed outliers",
                    len(removed_outliers_frames),
                )
                return removed_outliers_frames
            else:
                logger.info("plot_histogram_by_group: outlier return requested, none removed")
                return pd.DataFrame(columns=df.columns)

        return None

    except Exception as e:
        logger.exception("plot_histogram_by_group failed: %s", e)
        raise



def plot_scatter_size_coded(
    df: pd.DataFrame,
    x_col: str = "occurrences in individual events",
    y_col: str = "occurrences in sequential events",
    size_scale: float = 20,
    figsize: tuple = (6, 4),
) -> None:
    """
    Scatter plot where the marker size and color represent the count of overlapping points.

    Args:
        df: DataFrame with occurrence counts.
        x_col: Column for individual event counts.
        y_col: Column for sequential event counts.
        size_scale: Multiplier to scale marker sizes.
        figsize: Figure size.

    Returns:
        None
    """
    if df.empty:
        logger.warning(f"No data to plot for column '{x_col}'")
        return
    
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"DataFrame must contain '{x_col}' and '{y_col}'")
    counts = df.groupby([x_col, y_col]).size().reset_index(name="count")
    plt.figure(figsize=figsize)
    sc = plt.scatter(
        counts[x_col],
        counts[y_col],
        s=counts["count"] * size_scale,
        c=counts["count"],
        cmap="viridis",
        alpha=0.7,
        edgecolor="k",
    )
    plt.colorbar(sc, label="Number of Cells")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("Size coded Scatter: Overlap Density")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_scatter_hexbin(
    df: pd.DataFrame,
    x_col: str = "occurrences in individual events",
    y_col: str = "occurrences in sequential events",
    gridsize: int = 15,
    figsize: tuple = (8, 6),
) -> None:
    """
    Hexbin density plot to visualize point overlap in bins.

    Args:
        df: DataFrame with occurrence counts.
        x_col: Column for individual event counts.
        y_col: Column for sequential event counts.
        gridsize: Number of hexagons in the x-direction.
        figsize: Figure size.

    Returns:
        None
    """
    if df.empty:
        logger.warning(f"No data to plot for column '{x_col}'")
        return
    
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"DataFrame must contain '{x_col}' and '{y_col}'")
    x = df[x_col]
    y = df[y_col]
    plt.figure(figsize=figsize)
    hb = plt.hexbin(x, y, gridsize=gridsize, cmap="Blues", mincnt=1)
    plt.colorbar(hb, label="Count in Bin")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("Hexbin Density: Individual vs Sequential Occurrences")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_violin(
    df: pd.DataFrame,
    x: Optional[str],
    y: str,
    title: str,
    hue: Optional[str] = None,
    order: Optional[list] = None,
    hue_order: Optional[list] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    palette: Union[str, list, dict] = "muted",
    bw: Optional[Union[float, str]] = None,
    inner: Optional[str] = "quartile",
    cut: float = 0.0,
    linewidth: float = 0.8,
    width: float = 0.8,
    alpha: float = 0.9,
    dodge: bool = False,
    show_points: bool = False,
    point_style: str = "strip",  # 'strip' or 'swarm'
    point_size: float = 3.5,
    jitter: float = 0.2,
    figsize: tuple[float, float] = (7.0, 4.5),
    x_tick_rotation: int = 0,
    y_axis_boundaries: Optional[tuple[float, float]] = None,
    x_axis_boundaries: Optional[tuple[float, float]] = None,
    log_y: bool = False,
    filter_outliers: bool = False,
    outliers_bounds: Optional[tuple[float, float]] = None,
) -> None:
    """
    Render a flexible, single-figure violin plot with optional hue split, bandwidth control,
    and overlayed points.

    Args:
        df: Tidy DataFrame containing at least the `y` column, and `x`/`hue` if used.
        x: Categorical column for the x-axis (set to None for a single violin of all data).
        y: Numeric column to plot on the y-axis.
        title: Figure title.
        hue: Optional categorical column to split violins.
        order: Explicit order for `x` categories.
        hue_order: Explicit order for `hue` categories.
        xlabel: X-axis label (defaults to `x` if None).
        ylabel: Y-axis label (defaults to `y` if None).
        palette: Seaborn/Matplotlib palette name, list of colors, or dict mapping.
        bw: KDE bandwidth (float) or a method string like 'scott'/'silverman'. If None, seaborn default is used.
        inner: Inner representation: None|'box'|'quartile'|'point'|'stick'.
        cut: How far the density extends beyond extreme datapoints.
        linewidth: Line width of violin edges.
        width: Width of each violin.
        alpha: Face opacity of the violins (0–1).
        dodge: If True, draw separate violins for hue levels at each category.
        show_points: If True, overlay points (strip/swarm) to show raw samples.
        point_style: 'strip' (faster) or 'swarm' (collision-avoiding).
        point_size: Marker size for point overlay.
        jitter: Horizontal jitter for strip points.
        figsize: Figure size in inches.
        x_tick_rotation: Degrees to rotate x tick labels.
        y_axis_boundaries: Optional y-limits (ymin, ymax).
        x_axis_boundaries: Optional x-limits (xmin, xmax) — useful when x is numeric.
        log_y: If True, set y-axis to logarithmic scale.
        filter_outliers: If True, remove outliers using asymmetric-IQR before plotting.
        outliers_bounds: Quantile bounds (lower, upper) for outlier detection; default (0.25, 0.75).

    Returns:
        None

    Raises:
        KeyError: If required columns are missing.
        ValueError: If the DataFrame is empty or the plotting subset is empty.
    """
    try:
        if df is None or df.empty:
            logger.warning("plot_violin: received empty DataFrame")
            raise ValueError("Input DataFrame is empty.")

        required = [y] + ([x] if x is not None else []) + ([hue] if hue is not None else [])
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.error("plot_violin: missing required columns: %s (available=%s)", missing, list(df.columns))
            raise KeyError(f"Missing required column(s): {missing}")

        work = df.copy()
        work[y] = pd.to_numeric(work[y], errors="coerce")
        work = work.dropna(subset=[y] + ([x] if x is not None else []) + ([hue] if hue is not None else []))
        if work.empty:
            logger.warning("plot_violin: no valid rows to plot after cleaning")
            raise ValueError("No valid rows to plot after cleaning.")

        if filter_outliers:
            try:
                ql, qu = outliers_bounds if outliers_bounds is not None else (0.25, 0.75)
                filtered_df, outliers, lb, ub = detect_asymmetric_iqr_outliers(work, y, ql, qu)
                logger.info(
                    "plot_violin: removed %d outliers out of %d on '%s' (lower=%.5g, upper=%.5g)",
                    len(outliers), work.shape[0], y, lb, ub
                )
                work = filtered_df
            except Exception as e:
                logger.exception("plot_violin: outlier filtering failed — continuing without: %s", e)

        fig, ax = plt.subplots(figsize=figsize)

        # Build common kwargs; seaborn changed some params across versions, so we guard optional ones
        violin_kwargs = dict(
            data=work,
            x=x,
            y=y,
            hue=hue,
            order=order,
            hue_order=hue_order,
            palette=palette,
            inner=inner,
            cut=cut,
            linewidth=linewidth,
            width=width,
            dodge=dodge,
            ax=ax,
        )
        # Bandwidth: only pass if user specified, to avoid version warnings
        if bw is not None:
            violin_kwargs["bw"] = bw

        try:
            v = sns.violinplot(**violin_kwargs)
            # Apply alpha to collections
            for coll in ax.collections:
                try:
                    coll.set_alpha(alpha)
                except Exception:
                    pass
        except Exception as e:
            logger.exception("plot_violin: seaborn.violinplot failed, attempting matplotlib fallback: %s", e)
            # Very simple fallback — single violin using matplotlib; no hue support
            try:
                parts = ax.violinplot(dataset=[work[y].values], showmeans=False, showmedians=True)
                for pc in parts['bodies']:
                    pc.set_alpha(alpha)
                ax.set_xticks([1])
                ax.set_xticklabels([x if isinstance(x, str) else "Values"])
            except Exception as e2:
                logger.exception("plot_violin: matplotlib fallback failed: %s", e2)
                raise

        # Optional points overlay
        if show_points:
            try:
                if point_style == "swarm":
                    sns.swarmplot(
                        data=work, x=x, y=y, hue=hue if dodge else None, dodge=dodge,
                        size=point_size, ax=ax, color="k", alpha=min(1.0, alpha + 0.05)
                    )
                else:
                    sns.stripplot(
                        data=work, x=x, y=y, hue=hue if dodge else None, dodge=dodge,
                        size=point_size, jitter=jitter, ax=ax, color="k", alpha=min(1.0, alpha + 0.05)
                    )
            except Exception as e:
                logger.exception("plot_violin: point overlay failed: %s", e)

        ax.set_title(title)
        ax.set_xlabel(xlabel if xlabel is not None else (x if x is not None else ""))
        ax.set_ylabel(ylabel if ylabel is not None else y)

        if x_tick_rotation:
            try:
                plt.setp(ax.get_xticklabels(), rotation=x_tick_rotation)
            except Exception as e:
                logger.exception("plot_violin: rotating xticklabels failed: %s", e)

        if y_axis_boundaries is not None:
            try:
                ax.set_ylim(y_axis_boundaries)
            except Exception as e:
                logger.exception("plot_violin: setting y-axis boundaries failed: %s", e)
        if x_axis_boundaries is not None:
            try:
                ax.set_xlim(x_axis_boundaries)
            except Exception as e:
                logger.exception("plot_violin: setting x-axis boundaries failed: %s", e)

        if log_y:
            try:
                ax.set_yscale("log")
            except Exception as e:
                logger.exception("plot_violin: setting log y failed: %s", e)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.exception("plot_violin failed: %s", e)
        raise


def plot_points_mean_std(
    df: pd.DataFrame,
    x: Optional[str],
    y: str,
    title: str,
    hue: Optional[str] = None,
    order: Optional[list] = None,
    hue_order: Optional[list] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    palette: Union[str, list, dict, None] = "muted",
    dodge: bool = False,
    point_style: str = "strip",  # 'strip' or 'swarm'
    point_size: float = 3.5,
    point_alpha: float = 0.85,
    jitter: float = 0.2,
    capsize: float = 0.1,
    figsize: tuple[float, float] = (7.0, 4.5),
    x_tick_rotation: int = 0,
    y_axis_boundaries: Optional[tuple[float, float]] = None,
    x_axis_boundaries: Optional[tuple[float, float]] = None,
    log_y: bool = False,
    filter_outliers: bool = False,
    outliers_bounds: Optional[tuple[float, float]] = None,
    legend: bool = True,
    show_mean_labels: bool = False,
) -> None:
    """
    Plot raw points (strip or swarm) with a mean point and standard deviation whiskers per group.

    This helper mirrors the API of your violin plot: it supports optional hue splitting with
    dodging, custom palettes, axis labels, and robust error handling. The mean±std overlay is
    rendered using ``seaborn.pointplot`` (with fallbacks for different seaborn versions).

    Args:
        df: Tidy DataFrame containing at least the `y` column, and `x`/`hue` if used.
        x: Categorical column for the x-axis (set to None to aggregate across all rows).
        y: Numeric column to plot on the y-axis.
        title: Figure title.
        hue: Optional categorical column to split groups.
        order: Explicit category order for `x`.
        hue_order: Explicit category order for `hue`.
        xlabel: X-axis label (defaults to `x` if None).
        ylabel: Y-axis label (defaults to `y` if None).
        palette: Seaborn/Matplotlib palette name, list, or dict (mapping hue levels to colors).
        dodge: If True, draw separate summaries for hue levels at each category.
        point_style: 'strip' or 'swarm' for the raw points layer.
        point_size: Marker size of the raw points layer.
        point_alpha: Opacity of the raw points layer.
        jitter: Horizontal jitter for strip points (ignored for swarm).
        capsize: Size of the error bar caps (in axis fraction units).
        figsize: Figure size in inches.
        x_tick_rotation: Degrees to rotate x tick labels.
        y_axis_boundaries: Optional y-limits (ymin, ymax).
        x_axis_boundaries: Optional x-limits (xmin, xmax).
        log_y: If True, set y-axis to logarithmic scale.
        filter_outliers: If True, remove outliers using asymmetric-IQR before plotting.
        outliers_bounds: Quantile bounds (lower, upper) for outlier detection; default (0.25, 0.75).
        legend: Whether to show legend for the hue summary layer.
        show_mean_labels: If True, annotate each summary point with the mean value.

    Returns:
        None

    Raises:
        KeyError: If required columns are missing.
        ValueError: If the DataFrame is empty or no valid rows remain after cleaning.
    """
    try:
        if df is None or df.empty:
            logger.warning("plot_points_mean_std: received empty DataFrame")
            raise ValueError("Input DataFrame is empty.")

        required = [y] + ([x] if x is not None else []) + ([hue] if hue is not None else [])
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.error("plot_points_mean_std: missing required columns: %s (available=%s)", missing, list(df.columns))
            raise KeyError(f"Missing required column(s): {missing}")

        work = df.copy()
        work[y] = pd.to_numeric(work[y], errors="coerce")
        work = work.dropna(subset=[y] + ([x] if x is not None else []) + ([hue] if hue is not None else []))
        if work.empty:
            logger.warning("plot_points_mean_std: no valid rows to plot after cleaning")
            raise ValueError("No valid rows to plot after cleaning.")

        if filter_outliers:
            try:
                ql, qu = outliers_bounds if outliers_bounds is not None else (0.25, 0.75)
                filtered_df, outliers, lb, ub = detect_asymmetric_iqr_outliers(work, y, ql, qu)
                logger.info(
                    "plot_points_mean_std: removed %d outliers out of %d on '%s' (lower=%.5g, upper=%.5g)",
                    len(outliers), work.shape[0], y, lb, ub
                )
                work = filtered_df
            except Exception as e:
                logger.exception("plot_points_mean_std: outlier filtering failed — continuing without: %s", e)

        fig, ax = plt.subplots(figsize=figsize)

        # --- Raw points layer ---
        try:
            if point_style == "swarm":
                sns.swarmplot(
                    data=work,
                    x=x,
                    y=y,
                    hue=hue if dodge else None,
                    dodge=dodge,
                    palette=palette if (hue is not None) else None,
                    size=point_size,
                    alpha=point_alpha,
                    ax=ax,
                )
            else:
                sns.stripplot(
                    data=work,
                    x=x,
                    y=y,
                    hue=hue if dodge else None,
                    dodge=dodge,
                    palette=palette if (hue is not None) else None,
                    size=point_size,
                    jitter=jitter,
                    alpha=point_alpha,
                    ax=ax,
                )
        except Exception as e:
            logger.exception("plot_points_mean_std: points layer failed: %s", e)

        # --- Mean ± std overlay ---
        try:
            # New seaborn API (>=0.12): errorbar='sd'
            sns.pointplot(
                data=work,
                x=x,
                y=y,
                hue=hue,
                order=order,
                hue_order=hue_order,
                dodge=dodge,
                palette=palette,
                errorbar='sd',
                markers='o',
                capsize=capsize,
                ax=ax,
            )
        except TypeError:
            # Backward compatibility: older seaborn versions use estimator + ci='sd'
            import numpy as _np
            sns.pointplot(
                data=work,
                x=x,
                y=y,
                hue=hue,
                order=order,
                hue_order=hue_order,
                dodge=dodge,
                palette=palette,
                estimator=_np.mean,
                ci='sd',
                markers='o',
                capsize=capsize,
                ax=ax,
            )
        except Exception as e:
            logger.exception("plot_points_mean_std: summary overlay failed: %s", e)

        if not legend:
            try:
                ax.get_legend().remove()
            except Exception:
                pass

        if show_mean_labels:
            try:
                # Compute per-group means to annotate
                group_cols = ([] if x is None else [x]) + ([] if hue is None else [hue])
                means = work.groupby(group_cols, dropna=False)[y].mean().reset_index()
                # Resolve category order for positioning
                x_levels = order if (order is not None) else (sorted(work[x].unique()) if x is not None else [None])
                h_levels = hue_order if (hue_order is not None) else (sorted(work[hue].unique()) if hue is not None else [None])
                for _, row in means.iterrows():
                    xpos = x_levels.index(row[x]) if x is not None else 0
                    if hue is not None and dodge:
                        i = h_levels.index(row[hue])
                        n = max(len(h_levels), 1)
                        xpos = xpos - 0.4 + (i + 0.5) * (0.8 / n)
                    ax.text(xpos, row[y], f"{row[y]:.2f}", ha='center', va='bottom', fontsize=8, color='black')
            except Exception as e:
                logger.exception("plot_points_mean_std: annotating means failed: %s", e)

        ax.set_title(title)
        ax.set_xlabel(xlabel if xlabel is not None else (x if x is not None else ""))
        ax.set_ylabel(ylabel if ylabel is not None else y)

        if x_tick_rotation:
            try:
                plt.setp(ax.get_xticklabels(), rotation=x_tick_rotation)
            except Exception as e:
                logger.exception("plot_points_mean_std: rotating xticklabels failed: %s", e)

        if y_axis_boundaries is not None:
            try:
                ax.set_ylim(y_axis_boundaries)
            except Exception as e:
                logger.exception("plot_points_mean_std: setting y-axis boundaries failed: %s", e)
        if x_axis_boundaries is not None:
            try:
                ax.set_xlim(x_axis_boundaries)
            except Exception as e:
                logger.exception("plot_points_mean_std: setting x-axis boundaries failed: %s", e)

        if log_y:
            try:
                ax.set_yscale("log")
            except Exception as e:
                logger.exception("plot_points_mean_std: setting log y failed: %s", e)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.exception("plot_points_mean_std failed: %s", e)
        raise
