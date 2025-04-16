from pathlib import Path
import tifffile
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import random
import logging
from typing import List

from src.core.cell import Cell
from src.analysis.signal_processing import process_trace, detrend_exponential, fir_filter, gaussian_smooth, normalize_trace, highpass_filter

logger = logging.getLogger(__name__)


# ==========================
# DIVERSE FUNCTIONS
# ==========================

def preprocess_images(dir: Path, roi_scale: float, pattern: str, padding: int = 5) -> np.ndarray:
    """
    Preprocess .tif images by renaming, loading, and cropping.

    This function renames .tif images in a directory with padded numbers, 
    loads them into memory, and crops them to a specified region of interest (ROI).

    Args:
        dir (Path): Path to the folder containing .tif images.
        roi_scale (float): Scale factor for cropping (e.g., 0.75).
        pattern (str): Regex pattern to match filenames for renaming.
        padding (int): Number of digits for zero-padding in filenames.

    Returns:
        np.ndarray: Array of cropped images.

    Raises:
        FileNotFoundError: If no .tif images are found in the directory.
        ValueError: If any image fails to load.
    """
    rename_files_with_padding(dir, pattern, padding)
    
    loaded_images = load_images(dir)

    cropped_images = [crop_image(img, roi_scale) for img in loaded_images]
    
    return np.array(cropped_images)


def crop_image(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Crop the image based on the ROI scale.

    Args:
        image (np.ndarray): The input image to crop.
        scale (float): Scale factor for cropping.

    Returns:
        np.ndarray: Cropped image.
    """
    height, width = image.shape[:2]
    crop_h, crop_w = int(height * scale), int(width * scale)
    start_h, start_w = (height - crop_h) // 2, (width - crop_w) // 2
    return image[start_h:start_h + crop_h, start_w:start_w + crop_w]


def rename_files_with_padding(directory: Path, pattern: str, padding: int = 5) -> None:
    """
    Rename files in a directory to pad numeric portions with leading zeros.

    Args:
        directory (Path): Path to files.
        pattern (str): Regex pattern with numeric group.
        padding (int): Digits to pad to.
    """
    regex = re.compile(pattern)
    for file in directory.glob("*.TIF"):
        match = regex.search(file.name)
        if match:
            number = match.group(1)
            if len(number) >= padding:
                continue
            padded_number = number.zfill(padding)
            new_name = file.name.replace(f"t{number}", f"t{padded_number}")
            new_path = directory / new_name
            file.rename(new_path)
            logger.info(f"Renamed: {file.name} -> {new_name}")


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


def load_cells_from_pickle(file_path: Path, load: bool = False) -> List[Cell]:
    """
    Load a pickle file.

    Args:
        file_path (Path): Path to the pickle file.
        load (bool): Whether to actually load the file (useful for toggling).

    Returns:
        List[Cell]: List of Cell objects.
    """
    if load and file_path.exists():
        with open(file_path, "rb") as f:
            cells = pickle.load(f)
        logger.info(f"Loaded {len(cells)} cells from {file_path}")
        return cells
    else:
        logger.warning(f"File {file_path} not loaded (missing or load=False). Returning empty list.")
        return []


def load_existing_img(img_path: Path) -> np.ndarray:
    """
    Load a 16-bit grayscale image from a .tif file.

    Args:
        img_path (Path): Path to the .tif file.

    Returns:
        np.ndarray: Loaded image.
    """
    logger.debug(f"Loading image from: {img_path}")
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



# ==========================
# PLOTTING FUNCTIONS
# ==========================


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




# ==========================


def run_processing_pipeline(cells: List[Cell], processing_configs: List[dict]):
    plot_all_cells_processing_stages(cells, processing_configs)


def plot_all_cells_processing_stages(cells: List[Cell], processing_configs: List[dict]) -> None:
    """
    Plot the processing stages for all cells across multiple processing configurations.

    Args:
        cells (List[Cell]): List of Cell objects.
        processing_configs (List[dict]): List of parameter dictionaries, one per config.

    Returns:
        None
    """
    stages_labels = ["Raw", "Detrended", "Smoothed", "ΔF/F₀"]

    for config_idx, config in enumerate(processing_configs):
        
        
        sigma = config.get("sigma", 1.0)
        cutoff = config.get("cutoff", 0.001)
        numtaps = config.get("numtaps", 201)
        fs = config.get("fs", 1.0)
        """order = config.get("order", 2)"""

        method = config.get("method", "deltaf")

        n_rows = len(cells)
        n_cols = len(stages_labels)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2.5 * n_rows), sharey=False, sharex=True)
        if n_rows == 1:
            axes = [axes]  # Ensure iterable if single cell

        for i, cell in enumerate(cells):
            trace = np.array(cell.raw_intensity_trace, dtype=float)

            raw = trace.copy()
            #detrended = highpass_filter(raw, cutoff=cutoff, fs=fs, order=order)
            detrended = fir_filter(raw, cutoff=cutoff, fs=fs, numtaps=numtaps)
            smoothed = gaussian_smooth(detrended, sigma=sigma)
            normalized = normalize_trace(smoothed, method=method)
            stages = [raw, detrended, smoothed, normalized]

            for j, (data, label) in enumerate(zip(stages, stages_labels)):
                ax = axes[i][j] if n_rows > 1 else axes[j]
                ax.plot(data, color='blue')
                ax.set_title(f"{label}" if i == 0 else "")
                ax.set_xlabel("Time")
                if j == 0:
                    ax.set_ylabel(f"Cell {cell.label}")
                ax.grid(True)

        fig.suptitle(f"Processing Stages - Config {config_idx+1}: σ={sigma}, cutoff={cutoff}, numtaps={numtaps}", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


def plot_trace_grid(cell, sigmas, cutoffs, fs=1.0, order=2):
    raw = cell.raw_intensity_trace
    n_rows = len(cutoffs)
    n_cols = len(sigmas)

    plt.figure(figsize=(8, 3))
    plt.plot(raw, color='black')
    plt.title(f"Raw Intensity Trace - Cell {cell.label}")
    plt.xlabel("Timepoint")
    plt.ylabel("Intensity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 2.5*n_rows), sharex=True, sharey=True)

    for i, cutoff in enumerate(cutoffs):
        for j, sigma in enumerate(sigmas):
            trace = process_trace(raw, sigma=sigma, cutoff=cutoff, fs=fs, order=order)
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


def plot_highpass_grid(cell, cutoffs, orders, fs=1.0, sigma=1.0, btype='highpass'):
    raw = cell.raw_intensity_trace
    n_rows = len(cutoffs)
    n_cols = len(orders)

    plt.figure(figsize=(8, 3))
    plt.plot(raw, color='black')
    plt.title(f"Raw Intensity Trace - Cell {cell.label}")
    plt.xlabel("Timepoint")
    plt.ylabel("Intensity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 2.5*n_rows), sharex=True, sharey=True)

    for i, cutoff in enumerate(cutoffs):
        for j, order in enumerate(orders):
            trace = process_trace(raw, sigma=sigma, cutoff=cutoff, fs=fs, order=order, btype=btype)
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
