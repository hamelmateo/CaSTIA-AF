import matplotlib.pyplot as plt
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from calcium_activity_characterization.data.cells import Cell

logger = logging.getLogger(__name__)


def show_cell_plot(cell: Cell) -> None:
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

