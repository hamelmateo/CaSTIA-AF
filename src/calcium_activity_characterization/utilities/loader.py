import os
from pathlib import Path
import tifffile
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import logging
from typing import List, Optional
import colorsys
from PIL import Image

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


def load_image_fast(img_path: Path) -> np.ndarray:
    try:
        return tifffile.memmap(str(img_path))
    except Exception as e:
        logger.error(f"Failed to memmap {img_path.name}: {e}")
        raise


# ==========================
# SAVING FUNCTIONS
# ==========================


def save_tif_image(image: np.ndarray, file_path: str, photometric: str = "minisblack", imagej: bool = True) -> None:
    """
    Save a 2D image as a .tif file. Automatically creates the output directory if it does not exist.

    Args:
        image (np.ndarray): Image array to save.
        file_path (str): Full path (including filename) to save the image.
        photometric (str): Photometric interpretation for TIFF. Default is "minisblack".
        imagej (bool): Whether to save in ImageJ-compatible format. Default is True.
    """
    try:
        output_dir = os.path.dirname(file_path)
        os.makedirs(output_dir, exist_ok=True)
        tifffile.imwrite(file_path, image.astype(np.uint16), photometric=photometric, imagej=imagej)
        logger.info(f"Saved image to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save TIFF image to {file_path}: {e}")
        raise

def save_rgb_image(
    image: np.ndarray,
    filepath: Path | str
) -> None:
    """
    Save an RGB uint8 image as a PNG file.

    Args:
        image (np.ndarray): H×W×3 uint8 RGB array.
        filepath (Path | str): Path to the output PNG file.

    Raises:
        ValueError: If the image is not H×W×3 uint8.
        IOError: If saving the file fails.
    """
    try:
        # Validate image
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 RGB image of shape H×W×3, got shape {image.shape} and dtype {image.dtype}"
            )

        # Ensure output directory exists
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Save using PIL
        img_pil = Image.fromarray(image, mode='RGB')
        img_pil.save(out_path, format='PNG')
        logger.info(f"Saved PNG image to {out_path}")

    except Exception as e:
        logger.error(f"Failed to save RGB PNG: {e}")
        raise


def save_pickle_file(obj: any, file_path: str) -> None:
    """
    Save an object to a pickle file. Creates directories if needed.

    Args:
        obj (Any): The Python object to serialize.
        file_path (str): Full path to the pickle file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Saved pickle to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save pickle to {file_path}: {e}")
        raise


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
    cut_trace: int = 0
) -> None:
    """
    Plot a raster plot of binary traces for cells, with optional time shift (cut_trace).

    Args:
        output_path (Path): Directory to save the plot.
        cells (List[Cell]): List of Cell objects.
        cut_trace (int): Offset to add to the time axis (x-axis) for synchronization.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not cells:
            logger.warning("No cells provided for raster plot.")
            return

        binarized_matrix = np.array([cell.trace.binary for cell in cells])
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(binarized_matrix, aspect='auto', cmap='Greys', interpolation='nearest')
        ax.set_title("Binarized Activity Raster Plot")
        ax.set_ylabel("Cell Index")
        ax.set_xlabel("Time")

        # Add offset to x-axis if cut_trace > 0
        n_time = binarized_matrix.shape[1]
        ax.set_xticks(np.arange(0, n_time, max(1, n_time // 17)))
        ax.set_xticklabels([str(cut_trace + x) for x in ax.get_xticks().astype(int)])

        plt.tight_layout()
        plt.savefig(output_path, dpi=600)
        plt.close()
        logger.info(f"Raster plot saved to {output_path}")

    except Exception as e:
        logger.error(f"Failed to generate raster plot: {e}")
        raise


def plot_raster_heatmap(
    output_path: Path,
    cells: List[Cell],
    clip_percentile: float = 98.0,
    cut_trace: int = 0
) -> None:
    """
    Plot a raster of processed traces for a list of cells with a continuous colormap.
    Optionally adds an offset (cut_trace) to the time axis for synchronization.

    Args:
        output_path (Path): Path to save the output figure.
        cells (List): List of Cell objects, each with 'processed_intensity_trace' (np.ndarray).
        clip_percentile (float): Percentile for upper clipping of intensities.
        cut_trace (int): Offset to add to the time axis (x-axis) for synchronization.
    
    Raises:
        ValueError: If no valid cells with processed traces are found.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not cells:
            logger.warning("No cells provided for raster plot.")
            return

        trace_matrix = []
        for idx, cell in enumerate(cells):
            trace = cell.trace.versions["processed"]
            if trace is None:
                logger.warning(f"Cell {idx} has no 'processed_intensity_trace'. Skipping.")
                continue
            trace_matrix.append(trace)

        if not trace_matrix:
            raise ValueError("No valid processed traces found in cells.")

        trace_array = np.array(trace_matrix.copy())

        # Clip upper percentile to enhance contrast
        upper_clip = np.percentile(trace_array, clip_percentile)
        trace_array_clipped = np.clip(trace_array, a_min=None, a_max=upper_clip)

        fig, ax = plt.subplots(figsize=(12, 6))
        cmap = plt.get_cmap("viridis") 
        im = ax.imshow(trace_array_clipped, aspect='auto', cmap=cmap, interpolation='nearest')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Processed Trace Intensity")

        ax.set_title("Processed Trace Raster Heatmap")
        ax.set_ylabel("Cell Index")
        ax.set_xlabel("Time")

        # Add offset to x-axis if cut_trace > 0
        if cut_trace > 0:
            n_time = trace_array_clipped.shape[1]
            ax.set_xticks(np.arange(0, n_time, max(1, n_time // 17)))
            ax.set_xticklabels([str(cut_trace + x) for x in ax.get_xticks().astype(int)])

        plt.tight_layout()
        plt.savefig(output_path, dpi=600)
        plt.close()
        logger.info(f"Processed trace raster plot saved to {output_path}")

    except Exception as e:
        logger.error(f"Failed to generate processed trace raster plot: {e}")
        raise


def plot_impulse_raster(save_path: Path, cells: List[Cell]) -> None:
    """
    Plot a raster-like image where each row is a cell, and impulses are drawn
    as single-frame activations at activation_start_time.

    Args:
        cells (List[Cell]): List of Cell objects.
        save_path (Path): Path to save the output image.
    """
    try:
        n_cells = len(cells)
        max_time = max((
            max((p.activation_start_time for p in cell.trace.peaks), default=0)
            for cell in cells
        ), default=0) + 1

        impulse_raster = np.zeros((n_cells, max_time), dtype=int)
        for i, cell in enumerate(cells):
            for peak in cell.trace.peaks:
                if 0 <= peak.activation_start_time < max_time:
                    impulse_raster[i, peak.activation_start_time] = 1

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(impulse_raster, aspect="auto", cmap="Greys", interpolation="nearest")
        ax.set_title("Impulse Raster Plot (activation_start_time only)")
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

