from pathlib import Path
import tifffile
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import random
import logging
from typing import List
import pandas as pd

from data.cells import Cell
from processing.signal_processing import SignalProcessor
from config.config import DETRENDING_MODE, SIGNAL_PROCESSING_PARAMETERS


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

def plot_all_cells_processing_stages(cells: list, processing_configs: list, save_dir: Path = None) -> None:
    """
    Plot and save the processing stages (Raw -> Processed) for multiple cells
    under multiple processing configurations.

    Args:
        cells (List[Cell]): List of Cell objects.
        processing_configs (List[dict]): List of processing parameter dictionaries.
        save_dir (Path, optional): Directory to save the plots. Defaults to None (just show).
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for config_idx, params in enumerate(processing_configs):
        n_rows = len(cells)
        n_cols = 2  # Only raw and processed

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2.5 * n_rows), sharey=False, sharex=True)
        if n_rows == 1:
            axes = [axes]  # Ensure iterable if only one cell

        for i, cell in enumerate(cells):
            raw = np.array(cell.raw_intensity_trace, dtype=float)
            processor = SignalProcessor(DETRENDING_MODE, SIGNAL_PROCESSING_PARAMETERS[DETRENDING_MODE])
            processed = processor.run(raw)


            # Plot raw
            ax_raw = axes[i][0] if n_rows > 1 else axes[0]
            ax_raw.plot(raw, color='black')
            ax_raw.set_title("Raw")
            ax_raw.set_xlabel("Timepoint")
            if i == 0:
                ax_raw.set_ylabel(f"Cell {cell.label}")

            # Plot processed
            ax_proc = axes[i][1] if n_rows > 1 else axes[1]
            ax_proc.plot(processed, color='blue')
            ax_proc.set_title("Processed")
            ax_proc.set_xlabel("Timepoint")

        fig.suptitle(f"Processing Stages - Config {config_idx+1} [{DETRENDING_MODE.upper()}]", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_dir:
            plot_filename = save_dir / f"processing_config_{config_idx+1}.png"
            fig.savefig(plot_filename)
            plt.close(fig)
            print(f"Saved plot to {plot_filename}")
        else:
            plt.show()


def save_event_summary_to_excel(events_df: pd.DataFrame, output_path: Path) -> None:
    """
    Save a summary of tracked ARCOS events as an Excel file.

    Each row represents one event, with columns:
        - event_id
        - n_frames: number of unique frames
        - n_cells: number of unique cells (trackID)
        - start_frame: first frame index
        - end_frame: last frame index
        - frames: list of frame indices
        - cells: list of cell IDs

    Args:
        events_df (pd.DataFrame): Output from track_events_dataframe
        output_path (Path): Path to save the Excel file
    """
    if "event_id" not in events_df.columns:
        raise ValueError("Missing 'event_id' column in event dataframe.")

    summary = (
        events_df.groupby("event_id")
        .agg(
            frames=("frame", lambda x: sorted(x.unique())),
            cells=("trackID", lambda x: sorted(x.unique())),
        )
        .reset_index()
    )

    summary["n_frames"] = summary["frames"].apply(len)
    summary["n_cells"] = summary["cells"].apply(len)
    summary["start_frame"] = summary["frames"].apply(lambda x: x[0] if x else None)
    summary["end_frame"] = summary["frames"].apply(lambda x: x[-1] if x else None)

    # Optional: move columns into order
    summary = summary[
        ["event_id", "n_frames", "n_cells", "start_frame", "end_frame", "frames", "cells"]
    ]

    excel_path = output_path / "event_summary.xlsx"
    summary.to_excel(excel_path, index=False)
    logger.info(f"Saved event summary to {excel_path}")
