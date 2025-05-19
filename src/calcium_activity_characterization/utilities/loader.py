from pathlib import Path
import tifffile
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import random
import logging
from typing import List

from calcium_activity_characterization.data.cells import Cell


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


def plot_binarized_raster(output_path: Path, cells: List[Cell]) -> None:
    if not cells:
        logger.warning("No cells provided for raster plot.")
        return

    binarized_matrix = np.array([cell.binary_trace for cell in cells])
    plt.figure(figsize=(12, 6))
    plt.imshow(binarized_matrix, aspect='auto', cmap='Greys', interpolation='nearest')
    plt.xlabel("Time")
    plt.ylabel("Cell Index")
    plt.title("Binarized Activity Raster Plot")
    plt.colorbar(label="Activity (0/1)")
    plt.tight_layout()
    save_path = output_path / "raster_plot.png"
    plt.savefig(save_path, dpi=600)
    plt.close()
    logger.info(f"Raster plot saved to {output_path}")



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