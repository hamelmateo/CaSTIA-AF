import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from calcium_activity_characterization.logger import logger
from pathlib import Path
from typing import list, tuple

from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.utilities.plotter import plot_umap




def run_umap_on_cells(
    active_cells: list[Cell],
    output_path: Path,
    n_neighbors: int,
    min_dist: float,
    n_components: int,
    normalize: bool
) -> np.ndarray:
    """
    Run UMAP dimensionality reduction on active cell intensity traces.

    Args:
        active_cells (list[Cell]): list of valid Cell objects with intensity traces.
        n_neighbors (int): Number of neighbors for UMAP.
        min_dist (float): Minimum distance parameter for UMAP.
        n_components (int): Target dimensionality.
        normalize (bool): Whether to standardize the data before UMAP.

    Returns:
        np.ndarray: 2D array representing UMAP embedding.
    """
    traces = [cell.trace.versions["processed"] for cell in active_cells
          if len(cell.trace.versions["processed"]) > 0 and not cell.exclude_from_umap]

    if normalize:
        traces = StandardScaler().fit_transform(traces)
    else:
        traces = np.array(traces)

    try:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric='correlation', random_state=42)
        embedding = reducer.fit_transform(traces)
    except Exception as e:
        logger.error(f"UMAP failed: {e}")
        raise

    plot_umap(embedding, output_path, "cells_UMAP", n_neighbors, min_dist, n_components)

    return embedding

def run_umap_with_clustering(
    active_cells: list[Cell],
    n_neighbors: int,
    min_dist: float,
    n_components: int,
    normalize: bool,
    eps: float,
    min_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform UMAP followed by DBSCAN clustering on active cells.

    Args:
        active_cells (list[Cell]): list of valid Cell objects with intensity traces.
        n_neighbors (int): Number of neighbors for UMAP.
        min_dist (float): Minimum distance for UMAP.
        n_components (int): Target dimensionality.
        normalize (bool): Whether to standardize the traces.
        eps (float): DBSCAN epsilon radius.
        min_samples (int): Minimum samples per cluster for DBSCAN.

    Returns:
        tuple[np.ndarray, np.ndarray]: UMAP embeddings and DBSCAN cluster labels.
    """
    traces = [cell.trace.versions["processed"] for cell in active_cells if len(cell.trace.versions["processed"]) > 0]

    if normalize:
        traces = StandardScaler().fit_transform(traces)
    else:
        traces = np.array(traces)

    try:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
        embedding = reducer.fit_transform(traces)
    except Exception as e:
        logger.error(f"UMAP failed: {e}")
        raise

    try:
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(embedding)
    except Exception as e:
        logger.error(f"DBSCAN failed: {e}")
        raise

    plt.figure(figsize=(8, 6))
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_points = list(labels).count(-1)

    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    plt.title(f"UMAP + DBSCAN Clustering\n{num_clusters} clusters, {noise_points} noise points")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    param_text = (
        f"UMAP:\n  neighbors={n_neighbors}\n  min_dist={min_dist}\n  normalized={normalize}\n"
        f"DBSCAN:\n  eps={eps}\n  min_samples={min_samples}"
    )
    plt.gca().text(0.97, 0.97, param_text, transform=plt.gca().transAxes,
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.8))

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return embedding, labels
