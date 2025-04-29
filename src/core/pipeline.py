import numpy as np
import tifffile
import os
import logging
from pathlib import Path
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

from src.core.cell import Cell
from src.io.loader import (
    preprocess_images,
    save_tif_image,
    rename_files_with_padding,
    load_images,
    crop_image,
)
from src.core.segmentation import segmented

logger = logging.getLogger(__name__)

def cells_segmentation(
    input_dir: Path,
    roi_scale: float,
    file_pattern: str,
    padding: int,
    overlay_path: Path,
    save_overlay: bool,
    nuclei_mask_path: Path
) -> np.ndarray:
    """
    Perform segmentation of nuclei and save the resulting mask.

    Args:
        input_dir (Path): Path to the directory containing input images.
        roi_scale (float): Scale factor for cropping the images.
        file_pattern (str): File pattern to match input images.
        padding (int): Padding for filenames.
        overlay_path (Path): Path to save the overlay image.
        save_overlay (bool): Whether to save the overlay image.
        nuclei_mask_path (Path): Path to save the nuclei mask.

    Returns:
        np.ndarray: The resulting nuclei mask.
    """
    nucleis_imgs = preprocess_images(input_dir, roi_scale, file_pattern, padding)
    logger.debug("Running segmentation...")
    nuclei_mask = segmented(nucleis_imgs, overlay_path, save_overlay)
    save_tif_image(nuclei_mask, nuclei_mask_path)
    return nuclei_mask

def convert_mask_to_cells(nuclei_mask: np.ndarray) -> List[Cell]:
    """
    Convert a labeled nuclei mask into a list of Cell objects.

    Args:
        nuclei_mask (np.ndarray): The labeled mask of nuclei.

    Returns:
        List[Cell]: A list of Cell objects representing the detected cells.
    """
    logger.debug("Converting labeled mask to Cell objects...")
    cells = []
    label = 1
    while np.any(nuclei_mask == label):
        pixel_coords = np.argwhere(nuclei_mask == label)
        if pixel_coords.size > 0:
            centroid = np.array(np.mean(pixel_coords, axis=0), dtype=int)
            cell = Cell(label=label, centroid=centroid, pixel_coords=pixel_coords)
            if (
                centroid[0] < 20 or centroid[1] < 20 or
                centroid[0] > nuclei_mask.shape[0] - 20 or
                centroid[1] > nuclei_mask.shape[1] - 20
            ):
                cell.is_valid = False
            cells.append(cell)
        label += 1
    return cells


def compute_intensity_for_image(image_path: Path, cell_coords: List[np.ndarray], roi_scale: float) -> List[float]:
    """
    Compute mean intensity for each cell in a single image.

    Args:
        image_path (Path): Path to a FITC image.
        cell_coords (List[np.ndarray]): List of pixel coordinates for each cell.
        roi_scale (float): Scale factor for cropping.

    Returns:
        List[float]: Mean intensity values for each cell.
    """
    img = tifffile.imread(str(image_path))
    img = crop_image(img, roi_scale)
    mean_intensity = []
    for coords in cell_coords:
        intensities = [img[y, x] for y, x in coords]
        mean_intensity.append(float(np.mean(intensities)))
    return mean_intensity


def get_cells_intensity_profiles_parallelized(
    cells: List[Cell],
    input_dir: Path,
    pattern: str,
    padding: int,
    roi_scale: float
) -> None:
    """
    Calculate the intensity profiles for each cell using parallel processing
    and update each cell with its processed trace.

    Args:
        cells (List[Cell]): List of Cell objects to process.
        input_dir (Path): Path to the directory containing FITC images.
        pattern (str): File pattern to match FITC images.
        padding (int): Padding for filenames.
        roi_scale (float): Scale factor for cropping the images.
    """
    rename_files_with_padding(input_dir, pattern, padding)
    image_paths = sorted(input_dir.glob("*.TIF"))
    logger.debug("Sorted image paths")
    cell_coords = [cell.pixel_coords for cell in cells]
    func = partial(compute_intensity_for_image, cell_coords=cell_coords, roi_scale=roi_scale)
    logger.debug("Partial function created for multiprocessing")

    try:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(tqdm(executor.map(func, image_paths), total=len(image_paths), desc="Processing Images"))
    except Exception as e:
        logger.error(f"Parallel intensity computation failed: {e}")
        raise

    logger.debug("Executor map completed. Results collected.")
    
    results_per_cell = list(zip(*results))
    for cell, trace in zip(cells, results_per_cell):
        cell.raw_intensity_trace = list(map(int, trace))      

    logger.info("Raw intensity traces set for all cells.")
    

def get_cells_intensity_profiles(
    cells: List[Cell],
    input_dir: Path,
    roi_scale: float,
    file_pattern: str,
    padding: int
) -> None:
    """
    Calculate the intensity profiles for each cell based on their timepoints.

    Args:
        cells (List[Cell]): List of Cell objects to process.
        input_dir (Path): Path to the directory containing FITC images.
        roi_scale (float): Scale factor for cropping the images.
        file_pattern (str): File pattern to match FITC images.
        padding (int): Padding for filenames.
    """
    calcium_imgs = preprocess_images(input_dir, roi_scale, file_pattern, padding)
    if calcium_imgs.size > 0:
        logger.info(f"{len(calcium_imgs)} FITC images loaded.")
        for idx, img in enumerate(calcium_imgs):
            logger.debug(f"Processing image {idx + 1}/{len(calcium_imgs)}...")
            for cell in cells:
                cell.add_mean_intensity(img)
    else:
        logger.warning("No FITC images found.")