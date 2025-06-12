from pathlib import Path
import tifffile
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import random
import logging
from typing import List, Optional
import colorsys
import networkx as nx

from calcium_activity_characterization.data.cells import Cell


logger = logging.getLogger(__name__)


# ==========================
# DIVERSE FUNCTIONS
# ==========================


def get_config_with_fallback(config: dict, key: str, default: dict = None) -> dict:
    """
    Try to retrieve a config block. Log a warning and return default if not found.

    Args:
        config (dict): Top-level config dictionary.
        key (str): Name of the config block to retrieve.
        default (dict): Fallback value if key is missing.

    Returns:
        dict: The found or default config.
    """
    if key not in config:
        logging.warning(f"⚠️ Config parameter '{key}' not found. Using default: {default}")
        return default if default is not None else {}
    return config[key]


def generate_random_cell_overlay(cells: List[Cell], output_path: Path) -> None:
    """
    Generate a grayscale image with random intensities assigned per cell.

    Args:
        cells (List[Cell]): List of Cell objects.
        output_path (Path): Save path.
    """
    overlay = np.zeros((1536, 1536), dtype=np.uint16)
    for cell in cells:
        if not cell.pixel_coords.size:
            continue
        val = random.randint(10_000, 60_000)
        for y, x in cell.pixel_coords:
            overlay[y, x] = val
    save_tif_image(overlay, output_path)
    logger.info(f"Random cell overlay saved at: {output_path}")



# ==========================
# LOADING FUNCTIONS
# ==========================


def load_images(dir: Path = None) -> List[np.ndarray]:
    """
    Load all .tif images from a specified directory.

    Args:
        dir (Path): Path to the directory containing .tif images.

    Returns:
        List[np.ndarray]: List of loaded images as numpy arrays.

    Raises:
        FileNotFoundError: If no .tif images are found in the directory.
        ValueError: If any image fails to load.
    """
    images = list(dir.glob("*.TIF"))
    if not images:
        logger.error(f"No .tif images found in {dir}")
        raise FileNotFoundError(f"No .tif images found in {dir}")

    logger.info(f"Found {len(images)} images in directory: {dir}")
    loaded_images = [tifffile.imread(str(image)) for image in images]

    if any(img is None for img in loaded_images):
        logger.error(f"Could not load one or more images from {dir}")
        raise ValueError(f"Failed to load one or more images from {dir}")

    return loaded_images


def load_pickle_file(file_path: Path):
    """
    Load any Python object (e.g. DataFrame, dict, list) from a pickle file.

    Args:
        file_path (Path): Path to the .pkl file.

    Returns:
        object: The loaded Python object.
    """
    if not file_path.exists():
        logger.warning(f"File {file_path} not loaded (missing or load=False). Returning empty list.")
        return []

    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Successfully loaded pickle from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load pickle file from {file_path}: {e}")
        raise


def load_existing_img(img_path: Path) -> np.ndarray:
    """
    Load a 16-bit grayscale image from a .tif file.

    Args:
        img_path (Path): Path to the .tif file.

    Returns:
        np.ndarray: Loaded image.
    """
    return tifffile.imread(str(img_path))



# ==========================
# SAVING FUNCTIONS
# ==========================


def save_tif_image(image: np.ndarray, file_path: Path, photometric: str = 'minisblack', imagej: bool = True) -> None:
    """
    Save an image as .TIF.

    Args:
        image (np.ndarray): Image data.
        file_path (Path): Save path.
        photometric (str): Interpretation mode.
        imagej (bool): Save in ImageJ-compatible format.
    """
    tifffile.imwrite(file_path, image.astype(np.uint16), photometric=photometric, imagej=imagej)
    logger.info(f"Saved .tif image to: {file_path}")


def save_pickle_file(data: object, file_path: Path) -> None:
    """
    Save a Python object using pickle.

    Args:
        data (object): Python object to save.
        file_path (Path): Save path.
    """
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"Pickle saved to: {file_path}")


def save_image_histogram(image_path: Path, output_path: Path, title="Pixel Intensity Histogram", bins=65536) -> None:
    """
    Save a histogram plot of 16-bit grayscale image intensities.

    Args:
        image_path (Path): Path to the input image.
        output_path (Path): Path to save PNG.
        title (str): Plot title.
        bins (int): Number of bins.
    """
    img = tifffile.imread(str(image_path))
    flattened = img.ravel()

    plt.figure(figsize=(10, 6))
    plt.hist(flattened, bins=bins, color='green')
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Histogram saved to {output_path}")



def generate_distinct_colors(n_colors=60, min_s=0.6, max_v=0.9):
    """
    Generate distinct colors by sweeping hue space in HSV.

    Args:
        n_colors (int): Number of colors.
        min_s (float): Minimum saturation.
        max_v (float): Maximum brightness.

    Returns:
        List[Tuple[float, float, float]]: RGB colors in 0–1 range.
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


def plot_raster(
    output_path: Path,
    cells: List[Cell],
    clustered: bool = False,
    cluster_labels: Optional[np.ndarray] = None
) -> None:
    """
    Plot a raster plot of binary traces for cells.

    Args:
        output_path (Path): Directory to save the plot.
        cells (List[Cell]): List of Cell objects.
        clustered (bool): Whether to plot clustered data.
        cluster_labels (Optional[np.ndarray]): Cluster labels for each cell.
    """
    logger = logging.getLogger(__name__)
    if not cells:
        logger.warning("No cells provided for raster plot.")
        return

    if clustered:
        if cluster_labels is None or len(cluster_labels) != len(cells):
            logger.error("Clustered plot requested but valid cluster_labels not provided.")
            return

        cluster_labels = np.array(cluster_labels)
        cells_sorted = [cell for _, cell in sorted(zip(cluster_labels, cells), key=lambda x: x[0])]
        label_order = [label for label, _ in sorted(zip(cluster_labels, cells), key=lambda x: x[0])]
        binarized_matrix = np.array([cell.trace.binary for cell in cells_sorted])

        n_clusters = len([l for l in set(label_order) if l != -1])
        filtered_colors = generate_distinct_colors(n_clusters)
        label_to_color = {}
        for i, label in enumerate([l for l in sorted(set(label_order)) if l != -1]):
            color = filtered_colors[i % len(filtered_colors)]
            label_to_color[label] = color

        h, w = binarized_matrix.shape
        rgb_image = np.ones((h, w, 3), dtype=np.float32)

        for row_idx, (row, cluster_id) in enumerate(zip(binarized_matrix, label_order)):
            if cluster_id == -1:
                continue
            color = label_to_color.get(cluster_id, np.array([0.8, 0.8, 0.8]))
            rgb_image[row_idx, row == 1] = color

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(rgb_image, aspect='auto')
        ax.set_title("Binarized Activity (sorted by cluster)")
        ax.set_xlabel("Time")
        ax.set_yticks([])
        ax.set_yticklabels([])

    else:
        binarized_matrix = np.array([cell.trace.binary for cell in cells])
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(binarized_matrix, aspect='auto', cmap='Greys', interpolation='nearest')
        ax.set_title("Binarized Activity Raster Plot")
        ax.set_ylabel("Cell Index")
        ax.set_xlabel("Time")

    plt.tight_layout()
    filename = "clustered_raster_plot.png" if clustered else "raster_plot.png"
    save_path = output_path / filename
    plt.savefig(save_path, dpi=600)
    plt.close()
    logger.info(f"Raster plot saved to {save_path}")


def plot_impulse_raster(save_path: Path, cells: List[Cell]) -> None:
    """
    Plot a raster-like image where each row is a cell, and impulses are drawn
    as single-frame activations at rel_start_time.

    Args:
        cells (List[Cell]): List of Cell objects.
        save_path (Path): Path to save the output image.
    """
    try:
        n_cells = len(cells)
        max_time = max((
            max((p.rel_start_time for p in cell.trace.peaks), default=0)
            for cell in cells
        ), default=0) + 1

        impulse_raster = np.zeros((n_cells, max_time), dtype=int)
        for i, cell in enumerate(cells):
            for peak in cell.trace.peaks:
                if 0 <= peak.rel_start_time < max_time:
                    impulse_raster[i, peak.rel_start_time] = 1

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(impulse_raster, aspect="auto", cmap="Greys", interpolation="nearest")
        ax.set_title("Impulse Raster Plot (rel_start_time only)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Cell Index")

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_file = save_path / "impulse_raster.png" if save_path.is_dir() else save_path
        plt.savefig(save_file, dpi=600)
        plt.close(fig)
        logger.info(f"Impulse raster plot saved to {save_file}")

    except Exception as e:
        logger.error(f"Failed to generate impulse raster plot: {e}")


def plot_similarity_matrices(output_path: Path, similarity_matrices: list[np.ndarray]) -> None:
    """
    Plot all similarity matrices as heatmaps.

    Args:
        output_path (Path): Directory to save the plots.
        similarity_matrices (list[np.ndarray]): List of similarity matrices (N x N).
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
        cells (list[Cell]): List of active Cell objects.
        clustered_labels (list[np.ndarray]): List of cluster label arrays (one per window).
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

