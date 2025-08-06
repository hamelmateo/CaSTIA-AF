import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import imread
from pathlib import Path
from matplotlib import cm
from calcium_activity_characterization.logger import logger

def plot_histogram_by_dataset(
    df: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str = "Count",
    bin_width: float  = None,
    bin_count: int  = None,
    n_cols: int  = 2,
    log_scale_datasets: list[str]  = [],
    horizontal_layout: bool  = False
) -> None:
    """
    Plot histograms of a column per dataset, with stats annotation.

    Args:
        df (pd.DataFrame): The dataset (must include 'dataset' column).
        column (str): Column name to plot.
        title (str): Global title of the plot.
        bin_width (float, optional): Width of histogram bins.
        bin_count (int, optional): Number of bins (ignored if bin_width is used).
        n_cols (int, optional): Number of columns in the subplot grid (default: 3).
        log_scale_datasets (list[str], optional): list of dataset names to plot with log scale on y-axis.
        horizontal_layout (bool, optional): If True, prioritizes horizontal layout (max 2 rows).
    """
    if df.empty:
        logger.warning(f"No data to plot for column '{column}'")
        return
    datasets = sorted(df["dataset"].unique())
    n = len(datasets)

    if horizontal_layout:
        rows = min(2, math.ceil(n / n_cols))
        n_cols = math.ceil(n / rows)
    else:
        rows = math.ceil(n / n_cols)

    fig, axs = plt.subplots(rows, n_cols, figsize=(5 * n_cols, 4 * rows))
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    for i, dataset in enumerate(datasets):
        subset = df[df["dataset"] == dataset]
        clean = subset[column].dropna()

        if clean.empty:
            axs[i].set_title(f"{dataset} (No Data)")
            axs[i].axis("off")
            continue

        if bin_width is not None:
            min_val, max_val = clean.min(), clean.max()
            bins = int((max_val - min_val) / bin_width) + 1
        else:
            bins = bin_count

        sns.histplot(clean, bins=bins, kde=False, ax=axs[i])
        axs[i].set_title(dataset)
        axs[i].set_xlabel(column)
        axs[i].set_ylabel(ylabel)

        if dataset in log_scale_datasets:
            axs[i].set_yscale("log")

        stats_text = (
            f"Mean: {clean.mean():.2f}\n"
            f"Std: {clean.std():.2f}\n"
            f"Min: {clean.min():.2f}\n"
            f"Max: {clean.max():.2f}\n"
            f"Count: {clean.count()}"
        )
        axs[i].text(
            0.98, 0.98, stats_text,
            transform=axs[i].transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round")
        )

    for j in range(len(datasets), len(axs)):
        axs[j].axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def visualize_image(
    dataset_paths: dict[str, str],
    image_name: str = "signal-processing/raster_plot.png",
    title: str = "Binary Activity Raster Plots by Dataset",
    n_cols: int = 2,
    figsize_per_plot: tuple = (6, 5)
) -> None:
    """
    Plot raster images side by side for each dataset.

    Args:
        dataset_paths (dict[str, str]): dict mapping dataset labels to folder paths.
        image_name (str): Raster image file name (default: 'raster_plot.png').
        title (str): Global title.
        n_cols (int): Number of subplot columns (default: 2).
        figsize_per_plot (tuple): Size of each subplot (width, height).
    """
    datasets = sorted(dataset_paths.items())
    n = len(datasets)
    n_rows = int(np.ceil(n / n_cols))

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
    )

    # Always convert axs into a 1D list for consistent indexing
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]

    for i, (label, path) in enumerate(datasets):
        raster_path = Path(path) / image_name
        if raster_path.exists():
            img = imread(raster_path)
            axs[i].imshow(img, aspect="auto")
            axs[i].set_title(label)
            axs[i].axis("off")
        else:
            axs[i].set_title(f"{label} (Not Found)")
            axs[i].axis("off")

    # Hide any unused axes
    for j in range(len(datasets), len(axs)):
        axs[j].axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_pie_chart_by_dataset(
    df: pd.DataFrame,
    column: str,
    title: str = "Category Distribution",
    n_cols: int = 2,
    value_label_formatter: callable  = None,
    label_prefix: str  = None,
    palette: str = "tab10"
) -> None:
    """
    Plot pie charts showing distribution of a categorical column per dataset.

    Args:
        df (pd.DataFrame): DataFrame with 'dataset' and a categorical column.
        column (str): Categorical column to summarize (e.g., 'in_event', 'is_active').
        title (str): Global plot title.
        n_cols (int): Number of subplot columns.
        value_label_formatter (callable, optional): Custom formatter for pie slice labels.
        label_prefix (str, optional): Optional prefix for category labels.
        palette (str): Name of matplotlib colormap (default: 'tab10').
    """
    if df.empty:
        logger.warning(f"No data to plot for column '{column}'")
        return
    
    datasets = sorted(df["dataset"].unique())
    n = len(datasets)
    n_rows = math.ceil(n / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    # Compute all unique categories once and assign consistent colors
    all_categories = sorted(df[column].dropna().unique())
    cmap = cm.get_cmap(palette)
    colors = {cat: cmap(i / max(len(all_categories)-1, 1)) for i, cat in enumerate(all_categories)}

    for i, dataset in enumerate(datasets):
        subset = df[df["dataset"] == dataset]
        counts = subset[column].value_counts().reindex(all_categories).fillna(0).astype(int)
        total = counts.sum()

        if total == 0:
            axs[i].set_title(f"{dataset} (No Data)")
            axs[i].axis("off")
            continue

        labels = [
            f"{label_prefix + ' ' if label_prefix else ''}{k} ({v})"
            for k, v in counts.items()
        ]
        pie_labels = labels if value_label_formatter is None else None

        wedges, texts, autotexts = axs[i].pie(
            counts,
            labels=pie_labels,
            autopct=value_label_formatter or (lambda p: f"{p:.1f}%" if p > 0 else ""),
            startangle=90,
            colors=[colors[k] for k in counts.index]
        )

        for autotext in autotexts:
            autotext.set_color("black")
            autotext.set_fontsize(8)

        for text in texts:
            text.set_fontsize(8)
            text.set_color("black")

        axs[i].set_title(dataset)

    for j in range(len(datasets), len(axs)):
        axs[j].axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_bar_by_dataset(
    df: pd.DataFrame,
    axis_column: str,
    value_column: str,
    title: str,
    hue_column: str  = None,
    ylabel: str  = "Count",
    xlabel: str  = "Dataset",
    rotation: int  = 45,
    palette: str  = "muted"
) -> None:
    """
    Plot a bar chart of a metric aggregated per dataset.

    Args:
        df (pd.DataFrame): DataFrame containing 'dataset' and value column.
        value_column (str): Column with numeric values to plot.
        title (str): Plot title.
        hue_column (Optional[str]): Optional hue to split bars (e.g., concentration).
        ylabel (str): Y-axis label.
        xlabel (str): X-axis label.
        rotation (int): Rotation of x-axis labels.
        palette (str): Color palette.
    """
    if df.empty:
        logger.warning(f"No data to plot for column '{value_column}'")
        return
    plt.figure(figsize=(5, 3))
    sns.barplot(data=df, x=axis_column, y=value_column, hue=hue_column, dodge=False, palette=palette, legend=False)
    plt.title(title)
    plt.xticks(rotation=rotation)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.show()


def plot_histogram_by_group(
    df: pd.DataFrame,
    value_column: str,
    group_column: str,
    title: str,
    ylabel: str = "Count",
    bin_width: float = None,
    bin_count: int = None,
    n_cols: int = 2,
    log_scale_datasets: list[str] = [],
    horizontal_layout: bool = False,
    palette: str = "muted", # tab20, Set2, Dark2, Paired, Accent, Pastel1, Pastel2, muted
    multiple: str = "dodge"
) -> None:
    """
    Plot histograms of a numeric column per dataset, colored by group.

    Args:
        df (pd.DataFrame): The dataset (must include 'dataset', value_column, and group_column).
        value_column (str): Column name with numeric values (e.g., 'fwy_duration').
        group_column (str): Column name for grouping/coloring (e.g., 'cluster_label').
        title (str): Global plot title.
        bin_width (float, optional): Bin width for histograms.
        bin_count (int, optional): Number of bins (ignored if bin_width is provided).
        n_cols (int, optional): Number of subplot columns.
        log_scale_datasets (list[str], optional): Datasets to use log scale for y-axis.
        horizontal_layout (bool, optional): Layout mode.
        palette (str): Seaborn color palette name.
    """
    if df.empty:
        logger.warning(f"No data to plot for column '{value_column}'")
        return
    datasets = sorted(df["dataset"].unique())
    n = len(datasets)

    if horizontal_layout:
        rows = min(2, math.ceil(n / n_cols))
        n_cols = math.ceil(n / rows)
    else:
        rows = math.ceil(n / n_cols)

    fig, axs = plt.subplots(rows, n_cols, figsize=(5 * n_cols, 4 * rows))
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    for i, dataset in enumerate(datasets):
        subset = df[df["dataset"] == dataset]
        clean = subset[[value_column, group_column]].dropna()

        if clean.empty:
            axs[i].set_title(f"{dataset} (No Data)")
            axs[i].axis("off")
            continue

        if bin_width is not None:
            min_val, max_val = clean[value_column].min(), clean[value_column].max()
            bins = int((max_val - min_val) / bin_width) + 1
        else:
            bins = bin_count

        sns.histplot(
            data=clean,
            x=value_column,
            hue=group_column,
            bins=bins,
            multiple=multiple,  # Alternatives: 'layer', 'dodge', 'fill', 'stack'
            kde=False,
            ax=axs[i],
            palette=palette,
            edgecolor=None
        )

        axs[i].set_title(dataset)
        axs[i].set_xlabel(value_column)
        axs[i].set_ylabel(ylabel)

        if dataset in log_scale_datasets:
            axs[i].set_yscale("log")

    for j in range(len(datasets), len(axs)):
        axs[j].axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


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