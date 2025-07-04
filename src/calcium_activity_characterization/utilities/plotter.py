import matplotlib.pyplot as plt
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def plot_minima_diagnostics(
    trace: np.ndarray,
    anchor_idx: List[int],
    inserted_idx: List[int],
    discarded1: List[int],
    discarded2: List[int],
    discarded3: List[int],
    output_dir: Path
) -> None:
    """
    Plot trace with all categories of local minima overlaid.

    Args:
        trace (np.ndarray): The smoothed intensity trace.
        anchor_idx (List[int]): Final retained anchor indices.
        inserted_idx (List[int]): Newly added anchors.
        discarded1 (List[int]): Discarded after shoulder rejection.
        discarded2 (List[int]): Discarded after angle filtering.
        discarded3 (List[int]): Reserved for future use.
        output_dir (Path): Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(trace, label="Trace", lw=1.5)

    if anchor_idx:
        ax.scatter(anchor_idx, trace[anchor_idx], c="red", label="Final Anchors", s=15)
    if inserted_idx:
        ax.scatter(inserted_idx, trace[inserted_idx], c="orange", label="Inserted Anchors", s=15)
    if discarded1:
        ax.scatter(discarded1, trace[discarded1], c="gray", label="Shoulder Rejected", s=10, alpha=0.7)
    if discarded2:
        ax.scatter(discarded2, trace[discarded2], c="deepskyblue", label="Angle Rejected", s=10, alpha=0.5)
    if discarded3:
        ax.scatter(discarded3, trace[discarded3], c="lightgreen", label="Extra Discarded", s=10, alpha=0.5)

    ax.set_title("Local Minima Filtering Diagnostics")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    counter = 0
    while (output_dir / f"local_minimas_filtering_{counter}.png").exists():
        counter += 1
    save_path = output_dir / f"local_minimas_filtering_{counter}.png"
    plt.savefig(save_path)
    plt.close()

def plot_final_baseline_fit(
    trace: np.ndarray,
    baseline: np.ndarray,
    anchor_idx: List[int],
    detrended: np.ndarray,
    label: str,
    output_dir: Path,
    model_name: str
) -> None:
    """
    Plot the raw trace, baseline, anchor points, and final detrended result.

    Args:
        trace (np.ndarray): Smoothed input trace.
        baseline (np.ndarray): Fitted baseline.
        anchor_idx (List[int]): Anchor indices used for baseline fit.
        detrended (np.ndarray): Detrended result.
        label (str): Label suffix for filename.
        output_dir (Path): Directory to save plot.
        model_name (str): Label for the model type.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(trace, label="Trace")
    axs[0].scatter(anchor_idx, trace[anchor_idx], c='red', label="Anchors", s=15)
    axs[0].legend()
    axs[0].set_title("Raw Trace with Anchor Points")

    axs[1].plot(trace, label="Trace")
    axs[1].plot(baseline, label=f"Baseline ({model_name})")
    axs[1].legend()
    axs[1].set_title("Fitted Baseline")

    axs[2].plot(detrended, label="Detrended Trace")
    axs[2].legend()
    axs[2].set_title("Detrended Output")

    for ax in axs:
        ax.grid(True)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    counter = 0
    while (output_dir / f"baseline_fit_{counter}.png").exists():
        counter += 1
    save_path = output_dir / f"baseline_fit_{counter}.png"
    plt.savefig(save_path)
    plt.close()


def show_cell_plot(cell) -> None:
    """
    Plot the intensity trace of a single cell.

    Args:
        cell (Cell): The cell object whose intensity trace will be plotted.
    """
    if not cell.raw_intensity_trace:
        logger.info(f"Cell {cell.label} has no intensity data to plot.")
        return

    try:
        fig, ax = plt.subplots()
        ax.plot(cell.raw_intensity_trace, label=f"Cell {cell.label} ({cell.centroid[1]}, {cell.centroid[0]})")
        ax.set_title(f"Intensity Profile for Cell {cell.label}")
        ax.set_xlabel("Timepoint")
        ax.set_ylabel("Mean Intensity")
        ax.legend()
        ax.grid(True)
        plt.show(block=False)
    except Exception as e:
        logger.error(f"Failed to plot intensity profile for cell {cell.label}: {e}")



def plot_arcos_binarized_data(df: pd.DataFrame, track_id: int):
    """
    Plot raw intensity, detrended/rescaled intensity, and binarized data
    for a specific trackID.

    Args:
        df (pd.DataFrame): DataFrame containing the binarized data.
        track_id (int): Track identifier (cell label).
    """
    cell_data = df[df['trackID'] == track_id].sort_values('frame')

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(cell_data['frame'], cell_data['intensity'], 
            label='Raw Intensity', color='gray', linestyle='--', alpha=0.6)

    ax.plot(cell_data['frame'], cell_data['intensity.resc'], 
            label='Detrended & Rescaled Intensity', color='blue')

    # Plot binarized data (scaled for visibility)
    ax.step(cell_data['frame'], cell_data['intensity.bin'] * cell_data['intensity.resc'].max(), 
            label='Binarized Activity', color='red', linewidth=2, where='post')

    ax.set_xlabel('Frame (Time)')
    ax.set_ylabel('Intensity')
    ax.set_title(f'ARCOS Binarization - Cell {track_id}')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()




def plot_umap(
    embedding: np.ndarray,
    output_path: Path = None,
    title: str = "UMAP Projection",
    n_neighbors: int = None,
    min_dist: float = None,
    n_components: int = None,
    labels: np.ndarray = None
) -> None:
    """
    Plot the UMAP embedding.

    Args:
        embedding (np.ndarray): UMAP embedding (2D array).
        output_path (Path): Path to save the plot (optional).
        title (str): Title of the plot.
        n_neighbors (int): The size of the local neighborhood used for manifold approximation.
        min_dist (float): The minimum distance between points in the low-dimensional space.
        n_components (int): The number of dimensions for the UMAP embedding.
        labels (np.ndarray): Optional cluster labels for coloring points.
    """
    plt.figure(figsize=(8, 6))
    
    if labels is not None:
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
        plt.colorbar(scatter, label="Cluster Labels")
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=10, c='blue', alpha=0.6)

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True)

    # Add UMAP parameters as text on the plot
    if n_neighbors is not None or min_dist is not None or n_components is not None:
        param_text = "\n".join([
            f"n_neighbors = {n_neighbors}" if n_neighbors is not None else "",
            f"min_dist = {min_dist}" if min_dist is not None else "",
            f"n_components = {n_components}" if n_components is not None else ""
        ]).strip()
        plt.gca().text(
            0.95, 0.95, param_text, transform=plt.gca().transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.8)
        )

    plt.tight_layout()

    if output_path:
        np.save(output_path, embedding)
        logger.info(f"UMAP saved to {output_path}")
    else:
        plt.show()

