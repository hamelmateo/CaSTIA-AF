import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import imread
from pathlib import Path
from matplotlib import cm
from typing import Optional, Callable, Union, Literal, Iterable
from calcium_activity_characterization.logger import logger
from calcium_activity_characterization.analysis.metrics import detect_asymmetric_iqr_outliers
from contextlib import contextmanager
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import PercentFormatter
import json


@contextmanager
def illustrator_text_context(prefer_font: str = "Arial", keep_text: bool = True):
    """
    RC context to keep text editable in Adobe Illustrator for SVG/PDF exports.

    Args:
        prefer_font (str): A system font Illustrator will find (e.g., 'Arial' or 'Helvetica').
        keep_text (bool): If True, keep fonts as text (not outlines).

    Notes:
        - SVG: svg.fonttype='none' preserves text as text.
        - PDF: pdf.fonttype=42 embeds TrueType; Illustrator reads it as text.
        - TeX rendering is disabled to avoid outlines.
        - Only rcParams supported by the current Matplotlib are applied.
    """
    desired_rc = {
        "svg.fonttype": "none" if keep_text else "path",
        # "svg.embed_char_paths": False,  # DO NOT set; not supported on your version
        "pdf.fonttype": 42,              # TrueType
        "ps.fonttype": 42,               # TrueType (if ever exporting PS/EPS)
        "text.usetex": False,            # avoid TeX (often outlines)
        "mathtext.default": "regular",
        "font.family": "sans-serif",
        "font.sans-serif": [prefer_font, "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "axes.unicode_minus": False,     # safer minus glyph in AI
    }

    # Filter to only keys supported by this Matplotlib build
    safe_rc = {k: v for k, v in desired_rc.items() if k in mpl.rcParams}
    missing = set(desired_rc) - set(safe_rc)
    if missing:
        logger.info("Skipping unsupported rcParams for this Matplotlib: %s", sorted(missing))

    with mpl.rc_context(safe_rc):
        yield


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
    outliers_bygroup: Optional[str] = None,
    return_outliers: bool = False,
    save_svg_path: str | Path | None = None,
    export_format: str = "svg",
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
        outliers_bygroup (Optional[str], optional): Column name to group by for outlier detection.
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
            return None

        if column not in df.columns:
            logger.error(f"Column '{column}' not found in DataFrame columns: {list(df.columns)}")
            return None

        data = df[column].dropna()
        if data.empty:
            logger.warning(f"Column '{column}' has no valid values (all NaN after dropna()).")
            return None

        removed_outliers_frames: pd.DataFrame = pd.DataFrame()

        if filter_outliers:
            q_lower, q_upper = (outliers_bounds if outliers_bounds is not None else (0.25, 0.75))
            try:
                filtered_df, outliers, lower_bound, upper_bound = detect_asymmetric_iqr_outliers(
                    df.copy(), column, q_lower, q_upper, outliers_bygroup
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

        plt.tight_layout()

        # --- Save (flatten on white to avoid PDF darkening) ---
        if save_svg_path is not None:
            try:
                out_path = Path(save_svg_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fmt = export_format.lower()
                if fmt not in {"svg", "pdf"}:
                    fmt = "svg"
                plt.savefig(
                    out_path,
                    format=fmt,
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor="white",
                    transparent=True,
                    metadata={"Creator": "plot_violin"},
                )
                logger.info("plot_violin: figure saved to %s (%s)", out_path, fmt)
            except Exception as e:
                logger.exception("plot_violin: failed to save %s to '%s': %s", export_format, save_svg_path, e)

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
        return None

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
        _, source_path = next(iter(image_source.items()))
        img_path = Path(source_path) / image_name

        try:
            img = imread(img_path)
        except Exception as e:
            logger.exception("Failed to read image '%s': %s", img_path, e)
            return None

        if img is None or (hasattr(img, "size") and getattr(img, "size", 0) == 0):
            logger.error("Loaded image is empty: '%s'", img_path)
            return None

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img, aspect="auto")
        ax.set_title(title)
        if not show_axes:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.exception("visualize_image failed: %s", e)
        return None


def plot_pie_chart(
    df: pd.DataFrame,
    column: str,
    title: str = "Category Distribution",
    value_label_formatter: Callable[[float], str] | None = None,
    label_prefix: str | None = None,
    palette: str = "tab10",
    figsize: tuple[float, float] = (6, 5),
    save_svg_path: Path | str | None = None,
    show: bool = True,
    *,
    editable_text: bool = True,
    prefer_font: str = "Arial",
    export_format: str = "svg",  # "svg" or "pdf"
) -> Path | None:
    """
    Plot a single pie chart for the distribution of a categorical column and
    optionally save it as an SVG.

    Args:
        df (pd.DataFrame): DataFrame containing the categorical column.
        column (str): Categorical column to summarize (e.g., 'in_event', 'is_active').
        title (str): Figure title.
        value_label_formatter (Callable[[float], str] | None): Formatter for the percentage text
            drawn on slices. Signature must be f(pct: float) -> str where pct is in [0, 100].
        label_prefix (str | None): Optional prefix prepended to category labels.
        palette (str): Matplotlib colormap name for slice colors (default: 'tab10').
        figsize (tuple[float, float]): Figure size in inches.
        save_svg_path (Path | str | None): If provided, saves the figure as an SVG to this path.
        show (bool): Whether to display the figure window (plt.show()).

    Returns:
        Path | None: Path to the saved SVG if saved successfully; otherwise None.

    Raises:
        None: Errors are logged and the function returns early for robustness.
    """
    try:
        if df is None or df.empty:
            logger.warning("plot_pie_chart: empty DataFrame")
            return None
        if column not in df.columns:
            logger.error("plot_pie_chart: column '%s' not found in DataFrame columns: %s", column, list(df.columns))
            return None

        series = df[column].dropna()
        if series.empty:
            logger.warning("plot_pie_chart: column '%s' has no non-NaN values", column)
            return None

        # Stable category order
        try:
            all_categories = sorted(series.unique().tolist())
        except Exception:
            all_categories = list(pd.unique(series))

        counts = series.value_counts().reindex(all_categories).fillna(0).astype(int)
        total = int(counts.sum())
        if total == 0:
            logger.warning("plot_pie_chart: total count is zero for column '%s'", column)
            return None

        # Colors by category using the selected colormap
        colors = None
        try:
            cmap = cm.get_cmap(palette)
            denom = max(len(all_categories) - 1, 1)
            color_map = {cat: cmap(i / denom) for i, cat in enumerate(all_categories)}
            colors = [color_map[c] for c in counts.index]
        except Exception as e:
            logger.exception("plot_pie_chart: failed to build colors from palette '%s': %s", palette, e)

        # Labels around the pie (category + counts). If a custom formatter is supplied,
        # we skip outer labels to reduce clutter.
        labels = [
            f"{label_prefix + ' ' if label_prefix else ''}{k} ({v})" for k, v in counts.items()
        ]
        pie_labels = labels if value_label_formatter is None else None

        # === wrap the plotting & saving inside the rc context ===
        with illustrator_text_context(prefer_font=prefer_font, keep_text=editable_text):
            fig, ax = plt.subplots(figsize=figsize)
            try:
                wedges, texts, autotexts = ax.pie(
                    counts,
                    labels=pie_labels,
                    autopct=value_label_formatter or (lambda p: f"{p:.1f}%" if p > 0 else ""),
                    startangle=90,
                    colors=colors,
                    wedgeprops=dict(linewidth=0.0, edgecolor="black"),  # print-friendly
                )
            except Exception as e:
                logger.exception("plot_pie_chart: pie rendering failed: %s", e)
                plt.close(fig)
                return None

        # Styling
        try:
            if autotexts is not None:
                for autotext in autotexts:
                    autotext.set_color("black")
                    autotext.set_fontsize(9)
            if pie_labels is not None and texts is not None:
                for text in texts:
                    text.set_color("black")
                    text.set_fontsize(9)
        except Exception as e:
            logger.exception("plot_pie_chart: label styling failed: %s", e)

        ax.set_title(title)
        ax.axis("equal")
        plt.tight_layout()

        saved_path: Path | None = None
        if save_svg_path is not None:
            try:
                saved_path = Path(save_svg_path)
                saved_path.parent.mkdir(parents=True, exist_ok=True)
                fmt = export_format.lower()
                if fmt not in {"svg", "pdf"}:
                    fmt = "svg"
                plt.savefig(
                    saved_path,
                    format=fmt,
                    transparent=True,
                    bbox_inches="tight",
                    metadata={"Creator": "plot_pie_chart"},
                )
            except Exception as e:
                logger.exception("plot_pie_chart: failed to save %s to '%s': %s", export_format, save_svg_path, e)
                saved_path = None

        if show:
            plt.show()
        else:
            plt.close(fig)

        return saved_path
    
    except Exception as e:
        logger.exception("plot_pie_chart: unexpected error: %s", e)
        return None


def plot_category_distribution_by_dataset(
    df: pd.DataFrame,
    category_col: str,
    dataset_col: str,
    title: str = "Category Distribution by Dataset",
    normalize: bool = True,
    palette: str = "tab10",
    figsize: tuple[float, float] = (8.0, 5.0),
    save_path: str | Path | None = None,
    export_format: str = "svg",
    show: bool = True,
) -> Path | None:
    """Plot a multi-dataset "bar pie" as 100% stacked columns.

    Each dataset is a vertical bar split into colored segments whose heights
    correspond to the category distribution within that dataset. This is the
    bar-chart analog of multiple pie charts, allowing side-by-side comparison.

    Args:
        df: Input DataFrame containing at least *category_col* and *dataset_col*.
        category_col: Column name with category labels (e.g., "phenotype").
        dataset_col: Column name indicating dataset/group (e.g., "Dataset").
        title: Figure title.
        normalize: If True, bars are scaled to 100% within each dataset (recommended).
        palette: Matplotlib colormap name used to color categories consistently.
        figsize: Figure size in inches.
        save_path: If provided, write the figure to this path.
        export_format: Either "svg" or "pdf".
        show: If True, display the figure; otherwise close it after saving.

    Returns:
        Path | None: The saved figure path if written; otherwise None.
    """
    try:
        if df is None or df.empty:
            logger.warning("plot_category_distribution_by_dataset: empty DataFrame")
            return None
        for col in (category_col, dataset_col):
            if col not in df.columns:
                logger.error("plot_category_distribution_by_dataset: missing column '%s' in DataFrame", col)
                return None

        data = df[[dataset_col, category_col]].dropna()
        if data.empty:
            logger.warning("plot_category_distribution_by_dataset: no non-NaN rows after dropna")
            return None

        # Build counts matrix: rows=categories, cols=datasets
        counts = (
            data.groupby([dataset_col, category_col])
                .size()
                .unstack(fill_value=0)
                .T  # categories as rows for consistent stacking order
        )
        # Determine plotting order
        datasets = counts.columns.tolist()
        categories = counts.index.tolist()

        # Normalize to proportions per dataset
        if normalize:
            col_sums = counts.sum(axis=0).replace(0, np.nan)
            props = counts.div(col_sums, axis=1).fillna(0.0)
        else:
            props = counts.astype(float)

        # Colors per category
        try:
            from matplotlib import cm as _cm
            cmap = _cm.get_cmap(palette)
            denom = max(len(categories) - 1, 1)
            color_map = {cat: cmap(i / denom) for i, cat in enumerate(categories)}
        except Exception as e:
            logger.exception("plot_category_distribution_by_dataset: failed to build colors from palette '%s': %s", palette, e)
            # fallback: grayscale ramp
            color_map = {cat: (i / max(len(categories)-1, 1),) * 3 for i, cat in enumerate(categories)}

        # Plot stacked vertical bars
        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(datasets))
        bottoms = np.zeros(len(datasets))
        for cat in categories:
            heights = props.loc[cat, datasets].to_numpy()
            bars = ax.bar(x, heights, bottom=bottoms, color=color_map[cat], edgecolor="none")
            bottoms += heights

        # Axes & labels
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=0)
        if normalize:
            ax.set_ylabel("Percentage")
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        else:
            ax.set_ylabel("Count")
        ax.set_title(title)
        ax.set_xlim(-0.5, len(datasets) - 0.5)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

        # Legend outside on the right
        handles = [plt.Line2D([0], [0], color=color_map[cat], lw=6) for cat in categories]
        ax.legend(handles, categories, title="Category", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

        plt.tight_layout()

        out_path: Path | None = None
        if save_path is not None:
            try:
                out_path = Path(save_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fmt = export_format.lower()
                if fmt not in {"svg", "pdf"}:
                    fmt = "svg"
                plt.savefig(out_path, format=fmt, bbox_inches="tight", metadata={"Creator": "plot_category_distribution_by_dataset"})
            except Exception as e:
                logger.exception("plot_category_distribution_by_dataset: failed to save %s to '%s': %s", export_format, save_path, e)
                out_path = None

        if show:
            plt.show()
        else:
            plt.close(fig)
        return out_path

    except Exception as e:
        logger.exception("plot_category_distribution_by_dataset: unexpected error: %s", e)
        return None



def plot_bar(
    df: pd.DataFrame,
    axis_column: str,
    value_column: str,
    title: str,
    hue_column: Optional[str] = None,
    ylabel: str = "Count",
    xlabel: str = "Dataset",
    rotation: int = 45,
    palette: Optional[str] = "muted",
    save_svg_path: str | Path | None = None,  # you already added this
    export_format: str = "svg",  # "svg" or "pdf"
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
            return None

        missing = [c for c in [axis_column, value_column] if c not in df.columns]
        if missing:
            logger.error("plot_bar: missing required columns: %s (available=%s)", missing, list(df.columns))
            return None
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
            return None

        plt.figure(figsize=(6.5, 4.0))
        if hue_column is None:
            hue_column = axis_column

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

        # Save (SVG/PDF) with editable text
        if save_svg_path is not None:
            try:
                svg_path = Path(save_svg_path)
                svg_path.parent.mkdir(parents=True, exist_ok=True)
                fmt = export_format.lower()
                if fmt not in {"svg", "pdf"}:
                    fmt = "svg"
                plt.savefig(
                    svg_path,
                    format=fmt,
                    transparent=True,
                    bbox_inches="tight",
                    metadata={"Creator": "plot_histogram_by_group"},
                )
            except Exception as e:
                logger.exception("plot_histogram_by_group: failed to save %s to '%s': %s", export_format, save_svg_path, e)

        plt.show()

    except Exception as e:
        logger.exception("plot_bar failed: %s", e)
        return None


def plot_histogram_by_group(
    df: pd.DataFrame,
    value_column: str,
    group_column: str,
    title: str,
    ylabel: str = "Count",
    bin_width: float | None = None,
    bin_count: int | None = None,
    log_scale_datasets: list[str] = [],
    palette: str = "muted",
    multiple: str = "dodge",
    x_axis_boundaries: tuple[float, float] | None = None,
    y_axis_boundaries: tuple[float, float] | None = None,
    filter_outliers: bool = False,
    outliers_bounds: tuple[float, float] | None = None,
    outliers_bygroup: str | None = None,
    return_outliers: bool = False,
    save_svg_path: str | Path | None = None,  # you already added this
    *,
    editable_text: bool = True,
    prefer_font: str = "Arial",
    export_format: str = "svg",  # "svg" or "pdf"
) -> pd.DataFrame | None:
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
        outliers_bygroup (Optional[str], optional): Column name to group by for outlier detection.
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
            return None

        for col in [value_column, group_column]:
            if col not in df.columns:
                logger.error(
                    "plot_histogram_by_group: column '%s' missing. Available columns: %s",
                    col,
                    list(df.columns),
                )
                return None

        # Ensure numeric value column
        work = df[[value_column, group_column]].copy()
        work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
        work = work.dropna(subset=[value_column, group_column])
        if work.empty:
            logger.warning("plot_histogram_by_group: no valid rows after cleaning")
            return None

        removed_outliers_frames: pd.DataFrame = pd.DataFrame()

        # Optional outlier filtering
        if filter_outliers:
            q_lower, q_upper = (outliers_bounds if outliers_bounds is not None else (0.25, 0.75))
            try:
                filtered_df, outliers, lower_bound, upper_bound = detect_asymmetric_iqr_outliers(
                    work, value_column, q_lower, q_upper, outliers_bygroup
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

        with illustrator_text_context(prefer_font=prefer_font, keep_text=editable_text):
            fig, ax = plt.subplots(figsize=(5, 4))
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

        # Save (SVG/PDF) with editable text
        if save_svg_path is not None:
            try:
                svg_path = Path(save_svg_path)
                svg_path.parent.mkdir(parents=True, exist_ok=True)
                fmt = export_format.lower()
                if fmt not in {"svg", "pdf"}:
                    fmt = "svg"
                plt.savefig(
                    svg_path,
                    format=fmt,
                    transparent=True,
                    bbox_inches="tight",
                    metadata={"Creator": "plot_histogram_by_group"},
                )
            except Exception as e:
                logger.exception("plot_histogram_by_group: failed to save %s to '%s': %s", export_format, save_svg_path, e)

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
        return None



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
        return None

    if x_col not in df.columns or y_col not in df.columns:
        logger.error(f"plot_scatter_size_coded: DataFrame must contain '{x_col}' and '{y_col}'")
        return None
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
        logger.error(f"plot_scatter_hexbin: DataFrame must contain '{x_col}' and '{y_col}'")
        return None
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
    x: str | None,
    y: str,
    title: str,
    hue: str | None = None,
    order: list[str] | None = None,
    hue_order: list[str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    palette: str | list | dict | None = "muted",
    bw: float | str | None = None,
    inner: str | None = "quartile",
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
    y_axis_boundaries: tuple[float, float] | None = None,
    x_axis_boundaries: tuple[float, float] | None = None,
    log_y: bool = False,
    filter_outliers: bool = False,
    outliers_bygroup: str | None = None,
    outliers_bounds: tuple[float, float] | None = None,
    save_svg_path: str | Path | None = None,
    export_format: str = "svg",

    # --- Box overlay (classic "violin + box" style) ---
    overlay_box: bool = True,
    box_whis: float | tuple[float, float] = 1.5,   # Tukey whiskers by default
    box_linewidth: float = 1.2,
    showfliers: bool = False,

    # --- Matching colors / lines ---
    match_box_to_violin: bool = True,              # box fill matches violin fill color(s)
    shared_line_color: str | tuple[float, ...] = "0.35",  # one grey for all lines
    box_alpha: float = 1.0,                         # opacity for the IQR box
) -> None:
    """
    Render a violin plot with optional classic box overlay (median/IQR + whiskers), with the
    option to match the box fill to the violin color and use a unified grey for all linework.

    Args:
        df: Tidy DataFrame; must contain `y` (and `x`/`hue` if used).
        x: Categorical column for x-axis; None -> single violin of all data.
        y: Numeric column for y-axis.
        title: Plot title.
        hue: Optional grouping column within each x level.
        order, hue_order: Explicit orders.
        xlabel, ylabel: Axis labels; default to column names if None.
        palette: Seaborn palette / list / dict.
        bw, inner, cut, linewidth, width, alpha, dodge: Violin styling.
        show_points, point_style, point_size, jitter: Raw points overlay.
        figsize, x_tick_rotation, y_axis_boundaries, x_axis_boundaries, log_y: Layout/axes.
        filter_outliers, outliers_bygroup, outliers_bounds: Optional pre-filtering.
        save_svg_path, export_format: Saving (svg/pdf, flattened on white).

        overlay_box: Draw boxplot inside violins.
        box_whis: Whisker rule; 1.5 = Tukey.
        box_linewidth: Width of box/whisker/median lines.
        showfliers: Draw outlier fliers (dots) from boxplot layer.

        match_box_to_violin: If True, the IQR box fill color(s) match the violin color(s).
        shared_line_color: Grey used for *all* linework (violin edges + box lines).
        box_alpha: Opacity of the IQR box fill.

    Returns:
        None
    """
    try:
        if df is None or df.empty:
            logger.warning("plot_violin: received empty DataFrame")
            return None

        required = [y] + ([x] if x is not None else []) + ([hue] if hue is not None else [])
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.error("plot_violin: missing required columns: %s (available=%s)", missing, list(df.columns))
            return None

        work = df.copy()
        work[y] = pd.to_numeric(work[y], errors="coerce")
        work = work.dropna(subset=[y] + ([x] if x is not None else []) + ([hue] if hue is not None else []))
        if work.empty:
            logger.warning("plot_violin: no valid rows to plot after cleaning")
            return None

        if filter_outliers:
            try:
                ql, qu = outliers_bounds if outliers_bounds is not None else (0.25, 0.75)
                # Expect a helper in your codebase:
                filtered_df, outliers, lb, ub = detect_asymmetric_iqr_outliers(  # type: ignore[name-defined]
                    work, y, ql, qu, outliers_bygroup
                )
                logger.info(
                    "plot_violin: removed %d outliers out of %d on '%s' (lower=%.5g, upper=%.5g)",
                    len(outliers), work.shape[0], y, lb, ub
                )
                work = filtered_df
            except Exception as e:
                logger.exception("plot_violin: outlier filtering failed — continuing without: %s", e)

        fig, ax = plt.subplots(figsize=figsize)

        # --- Draw violins ---
        violin_kwargs: dict[str, object] = dict(
            data=work,
            x=x,
            y=y,
            hue=hue,
            order=order,
            hue_order=hue_order,
            inner=None if overlay_box else inner,  # avoid doubled quartile lines when we draw our box
            cut=cut,
            palette=palette,
            linewidth=linewidth,
            width=width,
            dodge=dodge,
            ax=ax,
        )
        if bw is not None:
            violin_kwargs["bw"] = bw

        try:
            sns.violinplot(**violin_kwargs)
            # apply opacity to the violin bodies
            for coll in ax.collections:
                try:
                    coll.set_alpha(alpha)
                    coll.set_edgecolor(shared_line_color)  # unified grey edge
                except Exception:
                    pass
        except Exception as e:
            logger.exception("plot_violin: seaborn.violinplot failed, attempting matplotlib fallback: %s", e)
            try:
                parts = ax.violinplot(dataset=[work[y].values], showmeans=False, showmedians=True)
                for pc in parts["bodies"]:
                    pc.set_alpha(alpha)
                    pc.set_edgecolor(shared_line_color)
                ax.set_xticks([1])
                ax.set_xticklabels([x if isinstance(x, str) else "Values"])
            except Exception as e2:
                logger.exception("plot_violin: matplotlib fallback failed: %s", e2)
                return None

        # --- Box overlay (median/IQR + Tukey whiskers), matching colors/lines ---
        if overlay_box:
            try:
                box_width = max(0.05, min(0.95, width * 0.15))

                # If we want the box face(s) to match the violin(s), let the palette set the colors.
                # (Don't pass a facecolor in boxprops; we set alpha separately.)
                box_palette = palette if match_box_to_violin else None

                bp_ax = sns.boxplot(
                    data=work,
                    x=x,
                    y=y,
                    hue=hue if dodge else None,
                    order=order,
                    hue_order=hue_order,
                    whis=box_whis,
                    showcaps=False,
                    showfliers=showfliers,
                    dodge=dodge,
                    width=box_width,
                    palette=box_palette,  # <- matches violin fill colors
                    ax=ax,
                    boxprops=dict(
                        edgecolor=shared_line_color,
                        linewidth=box_linewidth,
                        alpha=box_alpha,      # keep palette color but apply our alpha
                    ),
                    whiskerprops=dict(color=shared_line_color, linewidth=box_linewidth),
                    capprops=dict(color=shared_line_color, linewidth=box_linewidth),
                    medianprops=dict(color=shared_line_color, linewidth=box_linewidth),
                    zorder=5,
                )

                # If match_box_to_violin is False, make the boxes white fill but grey lines:
                if not match_box_to_violin:
                    for patch in ax.artists:  # each box is a PathPatch
                        try:
                            patch.set_facecolor("white")
                            patch.set_edgecolor(shared_line_color)
                            patch.set_alpha(box_alpha)
                        except Exception:
                            pass

            except Exception as e:
                logger.exception("plot_violin: box overlay failed: %s", e)

        # --- Optional raw points on top ---
        if show_points:
            try:
                if point_style == "swarm":
                    sns.swarmplot(
                        data=work, x=x, y=y, hue=hue if dodge else None, dodge=dodge,
                        size=point_size, ax=ax, color="k", alpha=min(1.0, alpha + 0.05), zorder=6
                    )
                else:
                    sns.stripplot(
                        data=work, x=x, y=y, hue=hue if dodge else None, dodge=dodge,
                        size=point_size, jitter=jitter, ax=ax, color="k", alpha=min(1.0, alpha + 0.05), zorder=6
                    )
            except Exception as e:
                logger.exception("plot_violin: point overlay failed: %s", e)

        # --- Labels / axes ---
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

        # --- Save (flatten on white to avoid PDF darkening) ---
        if save_svg_path is not None:
            try:
                out_path = Path(save_svg_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fmt = export_format.lower()
                if fmt not in {"svg", "pdf"}:
                    fmt = "svg"
                plt.savefig(
                    out_path,
                    format=fmt,
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor="white",
                    transparent=True,
                    metadata={"Creator": "plot_violin"},
                )
                logger.info("plot_violin: figure saved to %s (%s)", out_path, fmt)
            except Exception as e:
                logger.exception("plot_violin: failed to save %s to '%s': %s", export_format, save_svg_path, e)

        plt.show()

    except Exception as e:
        logger.exception("plot_violin failed: %s", e)
        return None


def plot_points_mean_std(
    df: pd.DataFrame,
    x: str | None,
    y: str,
    title: str,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_points: bool = True,
    point_style: str = "swarm",           # 'strip' or 'swarm'
    point_size: float = 3.5,
    point_alpha: float = 0.85,
    filter_outliers: bool = False,
    outliers_bounds: tuple[float, float] | None = None,
    outliers_bygroup: str | None = None,
    figsize: tuple[float, float] = (7.0, 4.5),
    x_tick_rotation: int = 0,
    y_axis_boundaries: tuple[float, float] | None = None,
    x_axis_boundaries: tuple[float, float] | None = None,
    log_y: bool = False,
    mean_color: str | None = None,               # Single color for all mean±std
    mean_palette: str | list[str] | None = None, # NEW: color each x-category (e.g., "tab10")
    save_svg_path: str | Path | None = None,
    export_format: str = "svg",
) -> None:
    """
    Plot optional raw points plus a mean±std overlay per x-category.
    - No connecting line between category means.
    - `mean_palette` colors each category differently; otherwise use `mean_color` (or default).

    Args:
        df: Tidy DataFrame with columns `y` and optionally `x`.
        x: Categorical column for x-axis (None => one aggregated group).
        y: Numeric column to plot.
        title: Figure title.
        xlabel: X label (defaults to `x` if None).
        ylabel: Y label (defaults to `y` if None).
        show_points: If True, show raw points beneath the summary.
        point_style: 'strip' or 'swarm' for the raw points.
        point_size: Marker size for raw points.
        point_alpha: Alpha for raw points.
        filter_outliers: Apply asymmetric-IQR filtering via `detect_asymmetric_iqr_outliers`.
        outliers_bounds: Bounds tuple forwarded to outlier detector.
        outliers_bygroup: Optional group column for outlier detector.
        figsize: Figure size.
        x_tick_rotation: Rotation of x tick labels.
        y_axis_boundaries: Optional (ymin, ymax).
        x_axis_boundaries: Optional (xmin, xmax).
        log_y: Log-scale y if True.
        mean_color: Single color for all means (ignored when `mean_palette` is set).
        mean_palette: Palette name or list to color each x-category differently.
        save_svg_path: If set, save the figure to this path.
        export_format: "svg" or "pdf".

    Returns:
        None
    """
    try:
        if df is None or df.empty:
            logger.warning("plot_points_mean_std: received empty DataFrame")
            return None

        req = [y] + ([x] if x is not None else [])
        missing = [c for c in req if c not in df.columns]
        if missing:
            logger.error("plot_points_mean_std: missing columns: %s (available=%s)", missing, list(df.columns))
            return None

        work = df.copy()
        work[y] = pd.to_numeric(work[y], errors="coerce")
        work = work.dropna(subset=req)
        if work.empty:
            logger.warning("plot_points_mean_std: no valid rows to plot after cleaning")
            return None

        if filter_outliers:
            try:
                ql, qu = outliers_bounds if outliers_bounds is not None else (0.25, 0.75)
                filtered_df, outliers, lb, ub = detect_asymmetric_iqr_outliers(work, y, ql, qu, outliers_bygroup)
                logger.info(
                    "plot_points_mean_std: removed %d/%d outliers on '%s' (lower=%.5g, upper=%.5g)",
                    len(outliers), work.shape[0], y, lb, ub,
                )
                work = filtered_df
            except Exception as e:
                logger.exception("plot_points_mean_std: outlier filtering failed — continuing without: %s", e)

        fig, ax = plt.subplots(figsize=figsize)

        # Optional raw points
        if show_points:
            try:
                if point_style == "swarm":
                    sns.swarmplot(data=work, x=x, y=y, size=point_size, alpha=point_alpha, ax=ax)
                else:
                    sns.stripplot(data=work, x=x, y=y, size=point_size, alpha=point_alpha, ax=ax, jitter=True)
            except Exception as e:
                logger.exception("plot_points_mean_std: points layer failed: %s", e)

        # === Mean ± std (no connecting line) ===
        try:
            # If a palette is provided, color each x-category separately by mapping hue=x
            if mean_palette is not None:
                # New seaborn (>=0.12): errorbar="sd"; else fallback to estimator/ci
                try:
                    sns.pointplot(
                        data=work,
                        x=x,
                        y=y,
                        hue=x,          # color per category
                        palette=mean_palette,
                        dodge=False,        # overplot at same x position
                        join=False,         # NO LINES
                        markers="o",
                        capsize=0.1,
                        errorbar="sd",
                        ax=ax,
                        legend=False,       # suppress duplicate legend
                    )
                except TypeError:
                    sns.pointplot(
                        data=work,
                        x=x,
                        y=y,
                        hue=x,
                        palette=mean_palette,
                        dodge=False,
                        join=False,
                        markers="o",
                        capsize=0.1,
                        estimator=np.mean,
                        ci="sd",
                        ax=ax,
                        legend=False,
                    )

            else:
                # Single-color (or default) seaborn pointplot, with join=False to avoid lines
                try:
                    sns.pointplot(
                        data=work,
                        x=x,
                        y=y,
                        errorbar="sd",
                        markers="o",
                        capsize=0.1,
                        linestyles='none',       
                        ax=ax,
                        color=mean_color,    # None -> default
                    )
                except TypeError:
                    # Older seaborn
                    sns.pointplot(
                        data=work,
                        x=x,
                        y=y,
                        estimator=np.mean,
                        ci="sd",
                        markers="o",
                        capsize=0.1,
                        join=False,          # <- no connecting line
                        ax=ax,
                        color=mean_color,
                    )
        except Exception as e:
            logger.exception("plot_points_mean_std: summary overlay failed: %s", e)

        # Labels / limits / scales
        try:
            ax.set_title(title)
            ax.set_xlabel(xlabel if xlabel is not None else (x if x is not None else ""))
            ax.set_ylabel(ylabel if ylabel is not None else y)
            if x_tick_rotation:
                plt.setp(ax.get_xticklabels(), rotation=x_tick_rotation)
            if y_axis_boundaries is not None:
                ax.set_ylim(y_axis_boundaries)
            if x_axis_boundaries is not None:
                ax.set_xlim(x_axis_boundaries)
            if log_y:
                ax.set_yscale("log")
        except Exception as e:
            logger.exception("plot_points_mean_std: axis setup failed: %s", e)

        plt.tight_layout()

        if save_svg_path is not None:
            try:
                out_path = Path(save_svg_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fmt = export_format.lower()
                if fmt not in {"svg", "pdf"}:
                    fmt = "svg"
                plt.savefig(
                    out_path,
                    format=fmt,
                    transparent=True,
                    bbox_inches="tight",
                    metadata={"Creator": "plot_points_mean_std"},
                )
            except Exception as e:
                logger.exception("plot_points_mean_std: failed to save %s to '%s': %s", export_format, save_svg_path, e)

        plt.show()

    except Exception as e:
        logger.exception("plot_points_mean_std failed: %s", e)
        return None

def plot_points_mean_std_continuous(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    hue: Optional[str] = None,           # optional — used only for raw scatter coloring
    palette: Union[str, list, dict, None] = "muted",
    point_size: float = 25.0,            # matplotlib scatter size (points^2)
    point_alpha: float = 0.85,
    jitter_x: float = 0.0,               # add small horizontal jitter to raw points
    figsize: tuple[float, float] = (7.0, 4.5),
    x_axis_boundaries: Optional[tuple[float, float]] = None,
    y_axis_boundaries: Optional[tuple[float, float]] = None,
    log_y: bool = False,

    # Mean ± std overlay controls
    strategy: str = "exact",             # "exact" (group by exact x), or "bins" (group into numeric bins)
    round_x_decimals: Optional[int] = None,
    bins: Optional[int] = None,
    bin_width: Optional[float] = None,
    error_cap_size: float = 3.0,
    error_line_width: float = 1.5,
    summary_marker_size: float = 45.0,
    style: str = "whitegrid",

    # NEW: control whether to draw raw points
    show_points: bool = True,

    # NEW: outlier filtering
    filter_outliers: bool = False,
    outliers_bounds: Optional[tuple[float, float]] = None,
    outliers_bygroup: Optional[str] = None,
) -> None:
    """
    Plot raw points (continuous x) with a mean point and standard deviation error bars per x grouping.

    Args:
        ...
        show_points: If False, hide the raw scatter and only show mean ± std.
        filter_outliers: If True, remove outliers before plotting.
        outliers_bounds: Quantile bounds for outlier detection; default (0.25, 0.75).
        outliers_bygroup: Optional grouping variable for outlier detection.
    """
    try:
        if df is None or df.empty:
            logger.warning("plot_points_mean_std_continuous: received empty DataFrame")
            return None

        required = [x, y] + ([hue] if (hue is not None and show_points) else [])
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.error("plot_points_mean_std_continuous: missing required columns: %s (available=%s)",
                         missing, list(df.columns))
            return None

        work = df.copy()
        work[x] = pd.to_numeric(work[x], errors="coerce")
        work[y] = pd.to_numeric(work[y], errors="coerce")
        if hue is not None and show_points:
            work[hue] = work[hue].astype("category")

        work = work.dropna(subset=[x, y])
        if work.empty:
            logger.warning("plot_points_mean_std_continuous: no valid rows to plot after cleaning")
            return None

        # --- Outlier filtering ---
        if filter_outliers:
            try:
                ql, qu = outliers_bounds if outliers_bounds is not None else (0.25, 0.75)
                filtered_df, outliers, lb, ub = detect_asymmetric_iqr_outliers(work, y, ql, qu, outliers_bygroup)
                logger.info(
                    "plot_points_mean_std_continuous: removed %d outliers out of %d on '%s' (lower=%.5g, upper=%.5g)",
                    len(outliers), work.shape[0], y, lb, ub
                )
                work = filtered_df
            except Exception as e:
                logger.exception("plot_points_mean_std_continuous: outlier filtering failed — continuing without: %s", e)

        sns.set(style=style)
        fig, ax = plt.subplots(figsize=figsize)

        # ----- Raw points -----
        if show_points:
            x_vals = work[x].to_numpy()
            y_vals = work[y].to_numpy()

            if jitter_x and jitter_x > 0:
                rng = np.random.default_rng(seed=42)
                x_vals = x_vals + rng.uniform(-jitter_x, jitter_x, size=x_vals.shape)

            if hue is not None:
                sns.scatterplot(
                    data=work,
                    x=x,
                    y=y,
                    hue=hue,
                    palette=palette,
                    alpha=point_alpha,
                    s=point_size,
                    ax=ax,
                    legend=True,
                )
            else:
                ax.scatter(x_vals, y_vals, alpha=point_alpha, s=point_size, color="black")

        # ----- Mean ± std overlay -----
        if strategy == "exact":
            gx = work.copy()
            if round_x_decimals is not None:
                gx[x] = gx[x].round(round_x_decimals)
            grouped = gx.groupby(x, dropna=False)[y]
            centers = grouped.mean().index.to_numpy(dtype=float)
            means = grouped.mean().to_numpy()
            stds = grouped.std(ddof=0).fillna(0.0).to_numpy()
        elif strategy == "bins":
            x_min, x_max = float(work[x].min()), float(work[x].max())
            if bins is None:
                if bin_width is not None and bin_width > 0:
                    bins = max(1, int(np.ceil((x_max - x_min) / bin_width)))
                else:
                    bins = 10
            if bin_width is not None and bin_width > 0 and (bins is None or isinstance(bins, int)):
                edges = np.arange(x_min, x_max + bin_width, bin_width)
            else:
                edges = np.linspace(x_min, x_max, int(bins) + 1)
            binned = work.copy()
            binned["_bin"] = pd.cut(binned[x], bins=edges, include_lowest=True)
            grouped = binned.groupby("_bin")[y]
            means = grouped.mean().to_numpy()
            stds = grouped.std(ddof=0).fillna(0.0).to_numpy()
            centers = np.array([(iv.left + iv.right) / 2.0 for iv in grouped.mean().index.to_numpy()])
        else:
            logger.error("plot_points_mean_std_continuous: invalid strategy '%s'", strategy)
            return None

        try:
            ax.errorbar(
                centers,
                means,
                yerr=stds,
                fmt="o",
                ms=np.sqrt(summary_marker_size),
                capsize=error_cap_size,
                linewidth=error_line_width,
                zorder=3,
                color="red"
            )
        except Exception as e:
            logger.exception("plot_points_mean_std_continuous: errorbar overlay failed: %s", e)

        if not show_points:
            leg = ax.get_legend()
            if leg is not None:
                try:
                    leg.remove()
                except Exception as e:
                    logger.exception("plot_points_mean_std_continuous: removing legend failed: %s", e)

        ax.set_title(title)
        ax.set_xlabel(xlabel if xlabel is not None else x)
        ax.set_ylabel(ylabel if ylabel is not None else y)

        if x_axis_boundaries is not None:
            try:
                ax.set_xlim(x_axis_boundaries)
            except Exception as e:
                logger.exception("plot_points_mean_std_continuous: setting x-limits failed: %s", e)
        if y_axis_boundaries is not None:
            try:
                ax.set_ylim(y_axis_boundaries)
            except Exception as e:
                logger.exception("plot_points_mean_std_continuous: setting y-limits failed: %s", e)

        if log_y:
            try:
                ax.set_yscale("log")
            except Exception as e:
                logger.exception("plot_points_mean_std_continuous: setting log y failed: %s", e)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.exception("plot_points_mean_std_continuous failed: %s", e)
        return None

def plot_xy_with_regression(
    data: pd.DataFrame,
    x_col: str = "Number of cells",
    y_col: str = "Number of global events",
    *,
    corr_method: Literal["pearson", "spearman", "kendall"] = "pearson",
    annotate_r: bool = True,
    r_text_format: str = "{method_cap} r = {corr:.2f}",
    r_position: tuple[float, float] = (0.05, 0.95),
    r_fontsize: int = 12,
    r_bbox: Optional[dict[str, any]] = None,
    order: int = 1,
    ci: int = 95,
    robust: bool = False,
    truncate: bool = True,
    hue: Optional[str] = None,
    height: float = 8.0,
    aspect: float = 1.0,
    markers: Union[str, tuple[str, ...]] = "o",
    palette: str = "muted",
    scatter_kws: Optional[dict[str, any]] = None,
    line_kws: Optional[dict[str, any]] = None,
    title: str = "Influence of the number of cells over cells global events activity",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    style: str = "whitegrid",
    dropna: bool = True
) -> None:
    """
    Creates a scatter + regression plot between two numeric columns and annotates with correlation.
    Displays the plot directly without returning anything.

    Args:
        data: Input DataFrame containing x and y columns.
        x_col: Column name for x-axis values.
        y_col: Column name for y-axis values.
        corr_method: Correlation method ('pearson', 'spearman', 'kendall').
        annotate_r: Whether to annotate the correlation.
        r_text_format: Text format for correlation.
        r_position: Position of correlation text in Axes coordinates.
        r_fontsize: Font size for annotation text.
        r_bbox: Bounding box style for annotation.
        order: Polynomial order for regression line.
        ci: Confidence interval for regression.
        robust: Whether to use robust regression.
        truncate: Truncate regression line to data range.
        hue: Optional column name for color grouping.
        height: Figure height.
        aspect: Aspect ratio (width = height * aspect).
        markers: Marker style(s).
        palette: Seaborn palette name.
        scatter_kws: Keyword arguments for scatter plot.
        line_kws: Keyword arguments for regression line.
        title: Plot title.
        xlabel: Optional override for x-axis label.
        ylabel: Optional override for y-axis label.
        xlim: Optional x-axis limits.
        ylim: Optional y-axis limits.
        style: Seaborn style.
        dropna: Drop NA values before plotting.
    """
    sns.set(style=style)

    if scatter_kws is None:
        scatter_kws = {"alpha": 0.6, "s": 80, "color": "blue", "edgecolor": "black"}
    if line_kws is None:
        line_kws = {"color": "red", "linewidth": 2, "linestyle": "--"}
    if r_bbox is None:
        r_bbox = dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white")

    if dropna:
        data = data.dropna(subset=[x_col, y_col])

    corr = data[x_col].corr(data[y_col], method=corr_method)

    g = sns.lmplot(
        data=data,
        x=x_col,
        y=y_col,
        hue=hue,
        height=height,
        aspect=aspect,
        ci=ci,
        robust=robust,
        order=order,
        truncate=truncate,
        scatter_kws=scatter_kws,
        line_kws=line_kws,
        markers=markers,
        palette=palette
    )

    if annotate_r:
        g.ax.text(
            r_position[0],
            r_position[1],
            r_text_format.format(method_cap=corr_method.capitalize(), corr=corr),
            transform=g.ax.transAxes,
            fontsize=r_fontsize,
            verticalalignment='top',
            bbox=r_bbox
        )

    g.set_axis_labels(xlabel if xlabel else x_col, ylabel if ylabel else y_col)
    g.fig.suptitle(title, fontsize=14, y=1.02)

    if xlim:
        g.set(xlim=xlim)
    if ylim:
        g.set(ylim=ylim)

    plt.tight_layout()
    plt.show()

def plot_heatmap(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    *,
    value: str | None = None,
    stat: Literal["count", "mean", "sum", "median"] = "count",
    bins_x: int | np.ndarray = 30,
    bins_y: int | np.ndarray = 30,
    range_x: tuple[float, float] | None = None,
    range_y: tuple[float, float] | None = None,
    dropna: bool = True,
    # transforms
    log_scale: bool = False,
    normalize: Literal["row", "col"] | None = None,
    clip_quantiles: tuple[float, float] | None = None,
    # plotting
    cmap: str | list = "viridis",
    robust: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    annot: bool = False,
    fmt: str = ".2f",
    cbar: bool = True,
    linewidths: float = 0.0,
    linecolor: str | None = None,
    square: bool = False,
    xtick_rotation: int = 0,
    ytick_rotation: int = 0,
    figsize: tuple[float, float] = (7.0, 5.0),
    save_svg_path: str | Path | None = None,
    export_format: str = "svg",
) -> plt.Axes | None:
    """
    Plot a 2D binned heatmap from numeric x/y columns using pandas cut + aggregation.

    Args:
        df: Input DataFrame with numeric columns `x` and `y`.
        x: Name of the numeric column for the x-axis.
        y: Name of the numeric column for the y-axis.
        title: Plot title.
        value: Optional numeric column used with stat {"mean","sum","median"}.
        stat: Aggregation within each bin. "count" ignores `value`.
        bins_x: Number of bins or explicit bin edges (array-like) for x.
        bins_y: Number of bins or explicit bin edges (array-like) for y.
        range_x: Optional (min,max) for x binning.
        range_y: Optional (min,max) for y binning.
        dropna: If True, drop rows where x/y (and value, if used) are NaN.
        log_scale: If True, applies log1p to the binned matrix before coloring.
        normalize: Optional min-max normalization per row or column on the binned matrix.
        clip_quantiles: Optional (low,high) quantiles to clip matrix values before plotting.
        cmap: Colormap.
        robust: Seaborn robust scaling for color limits.
        vmin, vmax, center: Color scaling parameters for seaborn.heatmap.
        annot: If True, annotate each cell with its numeric value.
        fmt: Format string for annotations.
        cbar: Show colorbar.
        linewidths: Grid line width between cells.
        linecolor: Grid line color between cells.
        square: Force square cells.
        xtick_rotation, ytick_rotation: Tick rotations in degrees.
        figsize: Figure size.

    Returns:
        matplotlib.axes.Axes on success, else None.
    """
    try:
        # Basic checks
        if df is None or df.empty:
            logger.warning("plot_heatmap: received empty DataFrame")
            return None

        missing = [col for col in [x, y] if col not in df.columns]
        if missing:
            logger.error("plot_heatmap: missing columns: %s (available=%s)", missing, list(df.columns))
            return None

        work_cols = [x, y] + ([value] if value and stat in {"mean", "sum", "median"} else [])
        work = df[work_cols].copy()
        for col in [x, y] + ([value] if value else []):
            work[col] = pd.to_numeric(work[col], errors="coerce")

        if dropna:
            work = work.dropna(subset=[x, y] + ([value] if value and stat != "count" else []))
        if work.empty:
            logger.warning("plot_heatmap: no valid data after cleaning")
            return None

        # Build bins
        x_min = work[x].min() if range_x is None else range_x[0]
        x_max = work[x].max() if range_x is None else range_x[1]
        y_min = work[y].min() if range_y is None else range_y[0]
        y_max = work[y].max() if range_y is None else range_y[1]

        x_bins = (np.linspace(x_min, x_max, int(bins_x) + 1)
                  if isinstance(bins_x, int) else np.asarray(bins_x, dtype=float))
        y_bins = (np.linspace(y_min, y_max, int(bins_y) + 1)
                  if isinstance(bins_y, int) else np.asarray(bins_y, dtype=float))

        # Bin using left-closed, right-open intervals so labels match left edges (0,1,2…)
        x_cat = pd.cut(work[x], bins=x_bins, include_lowest=True, right=False, ordered=True)
        y_cat = pd.cut(work[y], bins=y_bins, include_lowest=True, right=False, ordered=True)

        # Aggregation
        if stat == "count":
            mat = pd.crosstab(y_cat, x_cat, dropna=False)
        else:
            if not value:
                logger.error("plot_heatmap: stat %r requires `value` column.", stat)
                return None
            aggfunc = {"mean": "mean", "sum": "sum", "median": "median"}[stat]
            grouped = (work.assign(_xbin=x_cat, _ybin=y_cat)
                            .groupby(["_ybin", "_xbin"], observed=False)[value]
                            .agg(aggfunc))
            mat = grouped.unstack("_xbin")

        # Ensure 2D frame with all bins, even if empty
        # Convert IntervalIndex to numeric left edges for both axes (your “use the values themselves” request)
        def _ensure_all_bins(idx: pd.CategoricalIndex) -> list:
            # idx has .categories which are IntervalIndex
            return list(idx.categories)

        full_y_intervals = _ensure_all_bins(y_cat.cat)
        full_x_intervals = _ensure_all_bins(x_cat.cat)

        mat = mat.reindex(index=full_y_intervals, columns=full_x_intervals)

        # Extract left edges for clean integer-like tick labels
        x_labels = [int(iv.left) for iv in mat.columns]
        y_labels = [int(iv.left) for iv in mat.index]

        H = mat.to_numpy(dtype=float)

        # Optional transforms
        if clip_quantiles is not None:
            try:
                ql, qh = clip_quantiles
                finite_vals = H[np.isfinite(H)]
                if finite_vals.size:
                    lo, hi = np.quantile(finite_vals, [ql, qh])
                    H = np.clip(H, lo, hi)
            except Exception as e:
                logger.exception("plot_heatmap: clip_quantiles failed: %s", e)

        if log_scale:
            with np.errstate(invalid="ignore"):
                H = np.log1p(H)

        if normalize in {"row", "col"}:
            try:
                if normalize == "row":
                    mn = np.nanmin(H, axis=1, keepdims=True)
                    mx = np.nanmax(H, axis=1, keepdims=True)
                    denom = np.where(np.isfinite(mx - mn) & ((mx - mn) > 0), mx - mn, np.nan)
                    H = (H - mn) / denom
                else:  # "col"
                    mn = np.nanmin(H, axis=0, keepdims=True)
                    mx = np.nanmax(H, axis=0, keepdims=True)
                    denom = np.where(np.isfinite(mx - mn) & ((mx - mn) > 0), mx - mn, np.nan)
                    H = (H - mn) / denom
            except Exception as e:
                logger.exception("plot_heatmap: normalize failed: %s", e)

        # Build a frame back with left-edge labels so seaborn shows 0,1,2… not centers
        mat_plot = pd.DataFrame(H, index=y_labels, columns=x_labels)

        if stat == "count" and np.nansum(H) == 0:
            logger.warning("plot_heatmap: all bins empty (no points fell into ranges)")

        if not np.isfinite(H).any():
            logger.error("plot_heatmap: resulting matrix has no finite values")
            return None

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            data=mat_plot,
            cmap=cmap,
            robust=robust,
            vmin=vmin, vmax=vmax, center=center,
            annot=annot, fmt=fmt,
            cbar=cbar,
            linewidths=linewidths,
            linecolor=linecolor,
            square=square,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)

        # Keep low y at bottom
        ax.invert_yaxis()

        # Rotate ticks if requested
        plt.setp(ax.get_xticklabels(), rotation=xtick_rotation)
        plt.setp(ax.get_yticklabels(), rotation=ytick_rotation)

        plt.tight_layout()

        if save_svg_path is not None:
            try:
                out_path = Path(save_svg_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fmt = export_format.lower()
                if fmt not in {"svg", "pdf"}:
                    fmt = "svg"
                plt.savefig(
                    out_path,
                    format=fmt,
                    transparent=True,
                    bbox_inches="tight",
                    metadata={"Creator": "plot_points_mean_std"},
                )
            except Exception as e:
                logger.exception("plot_points_mean_std: failed to save %s to '%s': %s", export_format, save_svg_path, e)

        plt.show()
        return ax

    except Exception as e:
        logger.exception("plot_heatmap failed: %s", e)
        return None
    

def _parse_event_list(value: any) -> list[str]:
    """Best-effort parser for an event-ID list cell.

    Accepts Python lists/tuples/sets, JSON-encoded strings, comma-separated strings,
    or scalar (which will yield a single-item list). All items are converted to str.

    Args:
        value: Cell content to parse.

    Returns:
        list[str]: Parsed list of event IDs as strings. Empty on failure.
    """
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return []
        # Already a list-like
        if isinstance(value, (list, tuple, set)):
            return [str(x) for x in value]
        # JSON string like "[1, 2, 3]" or "[\"E1\", \"E2\"]"
        if isinstance(value, str):
            v = value.strip()
            if v.startswith("[") and v.endswith("]"):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return [str(x) for x in parsed]
                except Exception:
                    # fallthrough to comma-split
                    pass
            # Comma-separated fallback: "1,2,3" -> ["1","2","3"]
            if "," in v:
                return [s.strip() for s in v.split(",") if s.strip()]
            # Single scalar string -> [value]
            if v:
                return [v]
        # Single scalar -> [value]
        return [str(value)]
    except Exception:
        logger.exception("Failed to parse event list value: %r", value)
        return []

def plot_early_peakers_heatmap(
    cells: pd.DataFrame,
    *,
    cell_col: str = "Cell ID",
    occurrences_col: str = "Occurrences in global events as early peaker",
    events_col: str = "Early peaker event IDs",
    event_ids: list[str] | None = None,
    total_events: int | None = None,
    output_svg: str | Path = "early_peakers_heatmap.svg",
    title: str = "Early Peakers Participation",
    show_labels: bool = False,
    return_labels: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[str], list[str]] | None:
    """Plot and save a black/white heatmap of early-peaker participation by **event ID**.

    This version uses a per-cell column of event IDs (JSON/list) to place black squares
    exactly where a given cell was an early peaker for a specific event. Columns are the
    set of unique event IDs across all cells (or the provided *event_ids* order), and rows
    are the cells with at least one early-peaker occurrence.

    Args:
        cells: DataFrame with per-cell metadata.
        cell_col: Column holding cell identifiers.
        occurrences_col: Column holding the precomputed count of times a cell is an early peaker.
        events_col: Column holding the event-ID lists (can be JSON string or Python list).
        event_ids: Optional explicit ordering of event IDs (items will be converted to str).
            If None, will derive from *events_col* (sorted lexicographically as strings).
        total_events: **Deprecated**. Ignored when *event_ids* or *events_col* are present.
            Kept for backward compatibility; logged as a warning if provided.
        output_svg: Path to the SVG file to write.
        title: Title string for the figure.
        show_labels: If True, draw row/column tick labels. Off by default to keep figures compact.
        return_labels: If True, return a tuple of (matrix, row_labels, col_labels). Otherwise just the matrix.

    Returns:
        numpy.ndarray | tuple[numpy.ndarray, list[str], list[str]] | None:
            Binary matrix of shape [n_cells_with_occ>0, n_events]. If *return_labels* is True,
            returns (matrix, row_labels (cell IDs), col_labels (event IDs as str)). Returns None on error or when no data.
    """
    try:
        # Validate required columns
        for col in (cell_col, occurrences_col, events_col):
            if col not in cells.columns:
                raise KeyError(
                    f"Missing required column '{col}' in cells DataFrame. Available: {list(cells.columns)}"
                )

        # Filter to cells with >0 occurrences
        df = cells[[cell_col, occurrences_col, events_col]].copy()
        df = df[df[occurrences_col].astype(float) > 0]
        if df.empty:
            logger.warning("No cells with '%s' > 0. Nothing to plot.", occurrences_col)
            return None

        # Normalize types
        df[occurrences_col] = df[occurrences_col].round().astype(int).clip(lower=0)
        # Parse event lists
        df["__event_list__"] = df[events_col].apply(_parse_event_list)
        # Sort rows by occurrence count then cell ID for reproducibility
        df.sort_values(by=[occurrences_col, cell_col], ascending=[False, True], inplace=True)

        # Determine column order (event IDs)
        if event_ids is not None:
            col_labels = [str(e) for e in event_ids]
        else:
            # Union of all events across cells
            all_events: set[str] = set()
            for lst in df["__event_list__"]:
                all_events.update(lst)
            if not all_events:
                logger.warning("No event IDs found in '%s'. Nothing to plot.", events_col)
                return None
            col_labels = sorted(all_events)  # lexicographic; customize if needed

        if total_events is not None:
            logger.warning(
                "'total_events' is deprecated and ignored. Using %d unique event IDs.",
                len(col_labels),
            )

        # Build the binary matrix
        row_labels = df[cell_col].astype(str).tolist()
        n_rows = len(row_labels)
        n_cols = len(col_labels)
        index_by_event = {ev: j for j, ev in enumerate(col_labels)}

        matrix = np.zeros((n_rows, n_cols), dtype=int)
        for i, ev_list in enumerate(df["__event_list__"]):
            for ev in ev_list:
                j = index_by_event.get(ev)
                if j is not None:
                    matrix[i, j] = 1

        logger.info(
            "Early peakers event-matrix: %d cells x %d events; black squares: %d",
            n_rows,
            n_cols,
            int(matrix.sum()),
        )

        # --- Plot ---
        # Heuristic sizing: keep things readable without exploding figure size
        # Base cell size; scale modestly with counts
        cell_h = 0.05  # inches per cell row
        cell_w = 0.05  # inches per event column
        fig_w = max(3.0, n_cols * cell_w)
        fig_h = max(3.0, n_rows * cell_h)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        cmap = ListedColormap(["white", "black"])
        norm = BoundaryNorm([0, 0.5, 1], ncolors=2)
        ax.imshow(
            matrix,
            cmap=cmap,
            norm=norm,
            aspect="auto",
            interpolation="none",
            origin="upper",
        )

        # Grid lines at every cell
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.6)

        # Labels (optional; can get crowded)
        if show_labels:
            ax.set_xticks(np.arange(n_cols))
            ax.set_yticks(np.arange(n_rows))
            ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
            ax.set_yticklabels(row_labels, fontsize=8)
            ax.tick_params(axis="both", which="both", length=0)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title)

        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
        plt.margins(0)

        output_path = Path(output_svg)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, format="svg", bbox_inches="tight", pad_inches=0)
        plt.show()
        plt.close(fig)
        logger.info("Saved early peakers heatmap SVG to: %s", output_path)

        if return_labels:
            return matrix, row_labels, col_labels
        return matrix

    except Exception as e:
        logger.exception("Failed to create/save early peakers heatmap: %s", e)
        return None
