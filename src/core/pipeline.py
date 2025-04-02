import numpy as np
from pathlib import Path
from src.core.cell import Cell
from src.io.loader import save_pickle_file, load_and_crop_images, save_tif_image
from src.core.segmentation import segmented




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





def convert_mask_to_cells(nuclei_mask: np.ndarray, output_file_path: Path) -> list[Cell]:
    """
    Convert a labeled nuclei mask into a list of Cell objects and save them to a file.

    Parameters:
        nuclei_mask (np.ndarray): The labeled mask of nuclei.
        output_file (Path): Path to save the list of Cell objects.

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

    save_pickle_file(cells, output_file_path)

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

    # Plot the intensity traces of the first 5 active cells
    active_cells = [cell for cell in cells if cell.is_valid]
    for i, cell in enumerate(active_cells[200:205]):
        print(f"[INFO] Plotting intensity profile for Cell {cell.label}...")
        cell.plot_intensity_profile()