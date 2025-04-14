import matplotlib.pyplot as plt
from itertools import product
from sklearn.preprocessing import StandardScaler
import umap
import numpy as np
from src.core.cell import Cell
from typing import List
from scipy.signal import butter, sosfilt, filtfilt



def explore_umap_parameters(
    active_cells: List[Cell],
    neighbors_list: List[int] = [5, 10, 20],
    min_dist_list: List[float] = [0.1, 0.5, 1.0],
    normalize_options: List[bool] = [True, False],
    n_components: int = 2,
) -> None:
    """
    Explore UMAP dimensionality reduction over a grid of parameters and plot results.

    Args:
        active_cells (List[Cell]): List of Cell objects with intensity traces.
        neighbors_list (List[int]): Values for n_neighbors.
        min_dist_list (List[float]): Values for min_dist.
        normalize_options (List[bool]): Whether to normalize input data.
        n_components (int): Number of UMAP output dimensions (default: 2).
    """
    traces = [cell.raw_intensity_trace for cell in active_cells if len(cell.raw_intensity_trace) > 0]
    if not traces:
        print("No valid traces found for UMAP.")
        return

    total_plots = len(neighbors_list) * len(min_dist_list) * len(normalize_options)
    n_cols = len(min_dist_list)
    n_rows = len(neighbors_list) * len(normalize_options)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), squeeze=False)

    for (i, n_neighbors), (j, min_dist), (k, normalize) in product(
        enumerate(neighbors_list), enumerate(min_dist_list), enumerate(normalize_options)
    ):
        row = i * len(normalize_options) + k
        col = j

        if normalize:
            data = StandardScaler().fit_transform(traces)
        else:
            data = np.array(traces)

        try:
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
            embedding = reducer.fit_transform(data)
        except Exception as e:
            print(f"UMAP failed for neighbors={n_neighbors}, min_dist={min_dist}, normalize={normalize}: {e}")
            continue

        ax = axes[row][col]
        ax.scatter(embedding[:, 0], embedding[:, 1], s=10, c='blue', alpha=0.6)
        ax.set_title(f"neighbors={n_neighbors}\nmin_dist={min_dist}\nnormalize={normalize}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("UMAP Parameter Sweep", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


def explore_processing_parameters(cell, sigmas, cutoffs, fs=1.0, order=2):
    """
    Explore combinations of Gaussian smoothing (sigma) and high-pass filtering (cutoff)
    and plot the resulting normalized traces for visual comparison. The raw trace is shown separately.

    Args:
        cell (Cell): Cell object with raw intensity_trace.
        sigmas (list[float]): List of sigma values for Gaussian smoothing.
        cutoffs (list[float]): List of cutoff frequencies for high-pass filter.
        fs (float): Sampling frequency in Hz (default 1.0).
        order (int): Filter order for high-pass (default 2).
    """
    raw = cell.raw_intensity_trace
    n_rows = len(cutoffs)
    n_cols = len(sigmas)

    # Plot the raw trace separately
    plt.figure(figsize=(8, 3))
    plt.plot(raw, color='black')
    plt.title(f"Raw Intensity Trace - Cell {cell.label}")
    plt.xlabel("Timepoint")
    plt.ylabel("Intensity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot the parameter sweep
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 2.5*n_rows), sharex=True, sharey=True)

    for i, cutoff in enumerate(cutoffs):
        for j, sigma in enumerate(sigmas):
            cell.get_processed_trace(sigma=sigma)
            trace = cell.processed_intensity_trace

            ax = axes[i][j] if n_rows > 1 else axes[j]

            if trace is None or len(trace) == 0:
                ax.set_title(f"σ={sigma}, cutoff={cutoff} (empty)")
                continue

            ax.plot(trace, color='blue')
            ax.set_title(f"σ={sigma}, cutoff={cutoff}")
            ax.grid(True)

    fig.suptitle(f"Processed Trace Grid (Cell {cell.label})", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def highpass_filter_param_tuning(cell, orders, cutoffs, fs=1.0, sigma = 1.0, btype = 'highpass'):
    """
    Visualize the effect of different high-pass filter parameters (cutoff, order)
    on the intensity trace of a single cell using direct filter application.

    Args:
        cell (Cell): Cell object with raw_intensity_trace.
        cutoffs (list[float]): List of high-pass filter cutoff frequencies.
        orders (list[int]): List of filter orders.
        fs (float): Sampling frequency in Hz (default: 1.0).
        btype (str): Filter type (default: 'highpass').
    """
    raw = cell.raw_intensity_trace
    n_rows = len(cutoffs)
    n_cols = len(orders)

    # Plot the raw trace separately
    plt.figure(figsize=(8, 3))
    plt.plot(raw, color='black')
    plt.title(f"Raw Intensity Trace - Cell {cell.label}")
    plt.xlabel("Timepoint")
    plt.ylabel("Intensity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot the parameter sweep
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 2.5*n_rows), sharex=True, sharey=True)

    for i, cutoff in enumerate(cutoffs):
        for j, order in enumerate(orders):
            cell.get_processed_trace(sigma=sigma)
            trace = cell.processed_intensity_trace

            ax = axes[i][j] if n_rows > 1 else axes[j]

            if trace is None or len(trace) == 0:
                ax.set_title(f"order={order}, cutoff={cutoff} (empty)")
                continue

            ax.plot(trace, color='blue')
            ax.set_title(f"order={order}, cutoff={cutoff}")
            ax.grid(True)

    fig.suptitle(f"Processed Trace Grid (Cell {cell.label})", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()