
from calcium_activity_characterization.logger import logger
import random
from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import colorsys
from calcium_activity_characterization.data.cells import Cell




def generate_distinct_colors(n_colors=60, min_s=0.6, max_v=0.9):
    """
    Generate distinct colors by sweeping hue space in HSV.

    Args:
        n_colors (int): Number of colors.
        min_s (float): Minimum saturation.
        max_v (float): Maximum brightness.

    Returns:
        list[tuple[float, float, float]]: RGB colors in 0–1 range.
    """
    colors = []
    for i in range(n_colors * 2):  # Oversample and filter
        h = i / (n_colors * 2)
        s = np.random.uniform(min_s, 1.0)
        v = np.random.uniform(0.6, max_v)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        if (r + g + b) / 3 < max_v:  # discard overly bright
            colors.append((r, g, b))
        if len(colors) >= n_colors:
            break
    random.shuffle(colors)
    return colors


def plot_similarity_matrices(output_path: Path, similarity_matrices: list[np.ndarray]) -> None:
    """
    Plot all similarity matrices as heatmaps.

    Args:
        output_path (Path): Directory to save the plots.
        similarity_matrices (list[np.ndarray]): list of similarity matrices (N x N).
    """
    if not similarity_matrices:
        logger.warning("No similarity matrices to plot.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    for i, sim in enumerate(similarity_matrices):
        plt.figure(figsize=(8, 6))
        plt.imshow(sim, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label="Similarity")
        plt.title(f"Cell-Cell Similarity Matrix - Window {i}")
        plt.xlabel("Cell Index")
        plt.ylabel("Cell Index")
        plt.tight_layout()
        save_path = output_path / f"similarity_matrix_window_{i:03d}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved similarity matrix: {save_path}")


        

def save_clusters_on_overlay(overlay_path: Path, output_path: Path, clustered_labels: list[np.ndarray], cells: list[Cell]) -> None:
    """
    Overlay and save clustering results on the grayscale background image for each time window.

    Args:
        overlay_path (Path): Path to grayscale overlay image (TIF).
        cells (list[Cell]): list of active Cell objects.
        clustered_labels (list[np.ndarray]): list of cluster label arrays (one per window).
        output_dir (Path): Directory to save colored overlays.
    """
    overlay_img = tifffile.imread(str(overlay_path))
    h, w = overlay_img.shape
    output_path.mkdir(parents=True, exist_ok=True)

    for window_idx, cluster_labels in enumerate(clustered_labels):
        color_overlay = np.stack([overlay_img]*3, axis=-1).astype(np.uint8)
        unique_labels = sorted(set(cluster_labels))
        base_cmap = plt.get_cmap('tab10')
        color_map = {
            label: np.array(base_cmap(i % 10)[:3]) * 255 if label != -1 else np.array([0, 0, 0])
            for i, label in enumerate(unique_labels)
        }

        for cell, label in zip(cells, cluster_labels):
            color = color_map[label].astype(np.uint8)
            for y, x in cell.pixel_coords:
                color_overlay[y, x] = color

        save_path = output_path / f"cluster_overlay_window_{window_idx:03d}.png"
        plt.imsave(save_path, color_overlay)
        print(f"✅ Saved: {save_path}")


def plot_dendrogram(output_path: Path, dendrogram_data: dict) -> None:
    """
    Plot and save a dendrogram.

    Args:
        output_path (Path): Directory to save the dendrogram.
        dendrogram_data (dict): Dendrogram data.
    """
    from scipy.cluster.hierarchy import dendrogram
    from scipy.spatial.distance import squareform

    plt.figure(figsize=(10, 6))
    dendrogram(dendrogram_data)
    plt.title("Dendrogram")
    plt.xlabel("Cell Index")
    plt.ylabel("Distance")
    plt.tight_layout()
    save_path = output_path / "dendrogram.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Dendrogram saved to {save_path}")
