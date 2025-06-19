import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from matplotlib.image import imread
from pathlib import Path

def plot_metric_by_dataset(
    df: pd.DataFrame,
    column: str,
    title: str,
    bin_width: float = None,
    bin_count: int = None,
    n_cols: int = 2,
    log_scale_datasets: list[str] = [],
    horizontal_layout: bool = False
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
        log_scale_datasets (list[str], optional): List of dataset names to plot with log scale on y-axis.
        horizontal_layout (bool, optional): If True, prioritizes horizontal layout (max 2 rows).
    """
    datasets = sorted(df["dataset"].unique())
    n = len(datasets)

    if horizontal_layout:
        rows = min(2, math.ceil(n / n_cols))
        n_cols = math.ceil(n / rows)
    else:
        rows = math.ceil(n / n_cols)

    fig, axs = plt.subplots(rows, n_cols, figsize=(6 * n_cols, 5 * rows))
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
        axs[i].set_ylabel("Frequency")

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

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_raster_plots_by_dataset(
    dataset_paths: Dict[str, str],
    image_name: str = "raster_plot.png",
    title: str = "Binary Activity Raster Plots by Dataset",
    n_cols: int = 2,
    figsize_per_plot: tuple = (6, 5)
) -> None:
    """
    Plot raster images side by side for each dataset.

    Args:
        dataset_paths (Dict[str, str]): Dict mapping dataset labels to folder paths.
        image_name (str): Raster image file name (default: 'raster_plot.png').
        title (str): Global title.
        n_cols (int): Number of subplot columns (default: 3).
        figsize_per_plot (tuple): Size of each subplot (width, height).
    """
    datasets = sorted(dataset_paths.items())
    n = len(datasets)
    n_rows = int(np.ceil(n / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows))
    axs = axs.flatten()

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

    for j in range(len(datasets), len(axs)):
        axs[j].axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_category_distribution_by_dataset(
    df: pd.DataFrame,
    column: str,
    category_order: List[str],
    colors: Dict[str, str],
    title: str = "Category Distribution",
    n_cols: int = 2,
    value_label_formatter: Optional[callable] = None,
    label_prefix: Optional[str] = None
) -> None:
    """
    Plot pie charts showing distribution of a categorical column per dataset.

    Args:
        df (pd.DataFrame): DataFrame with a 'dataset' column and category column.
        column (str): Categorical column to summarize (e.g., 'in_event' or 'is_active').
        category_order (List[str]): Order of categories to display.
        colors (Dict[str, str]): Mapping of category names to colors.
        title (str): Global plot title.
        n_cols (int): Number of subplot columns.
        value_label_formatter (callable, optional): Function to customize pie label values.
        label_prefix (str, optional): Optional prefix for category labels (e.g., 'Active', 'Inactive').
    """
    datasets = sorted(df["dataset"].unique())
    n = len(datasets)
    n_rows = int(np.ceil(n / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axs = axs.flatten()

    for i, dataset in enumerate(datasets):
        subset = df[df["dataset"] == dataset]
        counts = subset[column].value_counts().reindex(category_order).fillna(0).astype(int)
        total = counts.sum()

        if total == 0:
            axs[i].set_title(f"{dataset} (No Data)")
            axs[i].axis("off")
            continue

        labels = [f"{label_prefix + ' ' if label_prefix else ''}{k.capitalize()} ({v})" for k, v in counts.items()]
        pie_labels = labels if value_label_formatter is None else None

        axs[i].pie(
            counts,
            labels=pie_labels,
            autopct=value_label_formatter or (lambda p: f"{p:.1f}%" if p > 0 else ""),
            startangle=90,
            colors=[colors.get(k, "#cccccc") for k in counts.index]
        )
        axs[i].set_title(dataset)

    for j in range(len(datasets), len(axs)):
        axs[j].axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
