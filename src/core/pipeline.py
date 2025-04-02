import numpy as np
from pathlib import Path
from src.core.cell import Cell
from src.io.loader import save_pickle_file, load_and_crop_images, save_tif_image
from src.core.segmentation import segmented
import tifffile
from typing import List
from src.io.loader import rename_files_with_padding, crop_image
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import os



def cells_segmentation(input_dir: Path, roi_scale: float, file_pattern: str, padding: int, overlay_path: Path, save_overlay: bool, nuclei_mask_path: Path) -> np.ndarray:
    """
    Perform segmentation of nuclei and save the resulting mask.

    Parameters:
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

    nucleis_imgs = load_and_crop_images(input_dir, roi_scale, file_pattern, padding)

    print("[DEBUG] Running segmentation...")
    nuclei_mask = segmented(nucleis_imgs, overlay_path, save_overlay)

    save_tif_image(nuclei_mask, nuclei_mask_path)

    return nuclei_mask





def convert_mask_to_cells(nuclei_mask: np.ndarray) -> list[Cell]:
    """
    Convert a labeled nuclei mask into a list of Cell objects and save them to a file.

    Parameters:
        nuclei_mask (np.ndarray): The labeled mask of nuclei.

    Returns:
        list[Cell]: A list of Cell objects representing the detected cells.
    """

    print("[DEBUG] Converting labeled mask to Cell objects...")
    cells = []
    label = 0
    while np.any(nuclei_mask == label):
        # Get the coordinates of all pixels with the current label
        pixel_coords = np.argwhere(nuclei_mask == label)
        
        if pixel_coords.size > 0:
            # Calculate centroid as the mean of the pixel coordinates
            centroid = np.array(np.mean(pixel_coords, axis=0), dtype=int)
            cell = Cell(label=label, centroid=centroid, pixel_coords=pixel_coords)
            
            # Check if the centroid is less than 20 pixels from the border
            if (centroid[0] < 20 or centroid[1] < 20 or 
                centroid[0] > nuclei_mask.shape[0] - 20 or 
                centroid[1] > nuclei_mask.shape[1] - 20):
                cell.is_valid = False
            
            cells.append(cell)
        
        label += 1

    return cells



def get_cells_intensity_profiles(cells: list[Cell], input_dir: Path, roi_scale: float, file_pattern: str, padding: int) -> None:
    """
    Calculate the intensity profiles for each cell based on their timepoints.

    Parameters:
        cells (list[Cell]): List of Cell objects to process.
        input_dir (Path): Path to the directory containing FITC images.
        roi_scale (float): Scale factor for cropping the images.
        file_pattern (str): File pattern to match FITC images.
        padding (int): Padding for filenames.
    """

    calcium_imgs = load_and_crop_images(input_dir, roi_scale, file_pattern, padding)

    if calcium_imgs.size > 0:
        print(f"[INFO] {len(calcium_imgs)} FITC images loaded.")
        for idx, img in enumerate(calcium_imgs):
            print(f"[DEBUG] Processing image {idx + 1}/{len(calcium_imgs)}...")
            for cell in cells:
                cell.add_mean_intensity(img)  # Add the current image as a timepoint for each cell
    else:
        print("[INFO] No FITC images found.")




def compute_intensity_for_image(image_path, cell_coords, roi_scale) -> List[float]:
    
    img = tifffile.imread(str(image_path)) 
    crop_image(img, roi_scale)  # stream from disk
    mean_intensity = []
    for coords in cell_coords:
        intensities = [img[y, x] for y, x in coords]
        mean_intensity.append(float(np.mean(intensities)))
    return mean_intensity



def get_cells_intensity_profiles_parallelized(cells, input_dir, pattern, padding, roi_scale):
    """
    Calculate the intensity profiles for each cell using parallel processing.

    Parameters:
        cells (list[Cell]): List of Cell objects to process.
        input_dir (Path): Path to the directory containing FITC images.
        pattern (str): File pattern to match FITC images.
        padding (int): Padding for filenames.
        roi_scale (float): Scale factor for cropping the images.
    """
    # Get list of image paths only (don’t load yet)
    rename_files_with_padding(input_dir, pattern, padding)
    image_paths = sorted(input_dir.glob("*.TIF"))
    print("[DEBUG] Sorted image paths")

    # Only send coordinates to workers
    cell_coords = [cell.pixel_coords for cell in cells]

    # Create a partial function for parallel execution
    func = partial(compute_intensity_for_image, cell_coords=cell_coords, roi_scale=roi_scale)
    print(f"[DEBUG] Partial function created with input directory")

    # Use tqdm to track progress

    with ProcessPoolExecutor(max_workers= os.cpu_count()) as executor:
        # Wrap the executor.map with tqdm for progress tracking
        results = list(tqdm(executor.map(func, image_paths), total=len(image_paths), desc="Processing Images"))

    print("[DEBUG] Executor map completed. Results collected.")

    # Reconstruct full trace for each cell
    results_per_cell = list(zip(*results))  # shape: [num_cells][num_timepoints]
    for cell, trace in zip(cells, results_per_cell):
        cell.intensity_trace = list(map(int, trace))
    print("[INFO] Intensity traces reconstructed for all cells.")
