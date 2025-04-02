from pathlib import Path
from src.core.cell import Cell
import tifffile
import numpy as np
import re
import pickle


def load_and_crop_images(dir: Path, roi_scale: float, pattern: str, padding: int = 5) -> np.ndarray:
    """
    Load .tif images in the folder (16-bit grayscale), rename them with padded numbers, 
    and crop them to a specific ROI.

    Parameters:
        dir (Path): Path to folder.
        roi_scale (float): Scale factor for cropping (e.g., 0.75 for 75%).
        pattern (str): Regex pattern to match filenames (e.g., "20250326_w3FITC_t(\d+).TIF").
        padding (int): Number of digits to pad the numbers to (default is 5).
    
    Returns:
        (np.ndarray): An array of cropped images from the folder.
    """
    # Rename files with padded numbers
    rename_files_with_padding(dir, pattern, padding)

    # Load images after renaming
    images = list(dir.glob("*.TIF"))
    if not images:
        raise FileNotFoundError(f"No .tif images found in {dir}")
    
    print(f"[INFO] Number of images found: {len(images)}")  # Print the number of images found

    loaded_images = [tifffile.imread(str(image)) for image in images]

    if any(img is None for img in loaded_images):
        raise ValueError(f"Could not load one or more images from {dir}")

    # Crop images to the specified ROI scale
    cropped_images = [crop_image(img, roi_scale) for img in loaded_images]

    return np.array(cropped_images)



def crop_image(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Crop the image based on the ROI scale.

    Parameters:
        image (np.ndarray): The input image to crop.
        scale (float): The scale factor for cropping (e.g., 0.75 for 75%).

    Returns:
        np.ndarray: The cropped image.
    """
    height, width = image.shape[:2]
    crop_h, crop_w = int(height * scale), int(width * scale)
    start_h, start_w = (height - crop_h) // 2, (width - crop_w) // 2
    return image[start_h:start_h + crop_h, start_w:start_w + crop_w]



def rename_files_with_padding(directory: Path, pattern: str, padding: int = 5) -> None:
    """
    Rename files in the directory by adding leading zeros to numbers in filenames.

    Parameters:
        directory (Path): Path to the directory containing the files.
        pattern (str): Regex pattern to match filenames (e.g., "20250326_w3FITC_t(\d+).TIF").
        padding (int): Number of digits to pad the numbers to (default is 5).
    """
    files = list(directory.glob("*.TIF"))
    regex = re.compile(pattern)

    for file in files:
        match = regex.search(file.name)
        if match:
            number = match.group(1)
            
            # Skip renaming if the number is already padded
            if len(number) >= padding:
                continue

            # Add leading zeros to the number
            padded_number = number.zfill(padding)
            new_name = file.name.replace(f"t{number}", f"t{padded_number}")
            new_path = directory / new_name

            # Rename the file
            file.rename(new_path)
            print(f"[INFO] Renamed: {file.name} -> {new_name}")



def load_existing_cells(file_path: str, load: bool = False) -> list[Cell]:
    """Load cells from a pickle file."""
    
    if load and file_path.exists():
        with open(file_path, "rb") as f:
            cells = pickle.load(f)
        print(f"[INFO] Loaded {len(cells)} cells from {file_path}")
        return cells



def load_existing_nuclei_mask(nuclei_mask_path: str) -> np.ndarray:
    """Load an existing nuclei mask from a file."""

    print("[DEBUG] Loading existing nuclei mask from file.")
    return tifffile.imread(nuclei_mask_path)



def save_tif_image(image: np.ndarray, file_path: Path, photometric: str = 'minisblack', imagej: bool = True) -> None:
    """
    Save a NumPy array as a .TIF image.

    Parameters:
        image (np.ndarray): The image to save.
        file_path (Path): Path to save the .TIF image.
        photometric (str): Photometric interpretation ('minisblack' for grayscale, etc.).
        imagej (bool): Whether to save the image in ImageJ-compatible format.
    """
    tifffile.imwrite(file_path, image.astype(np.uint16), photometric=photometric, imagej=imagej)
    print(f"[INFO] Image saved to {file_path}")



def save_pickle_file(data: object, file_path: Path) -> None:
    """
    Save a Python object to a pickle file.

    Parameters:
        data (object): The Python object to save (e.g., list of cells).
        file_path (Path): Path to save the pickle file.
    """
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[INFO] Data saved to {file_path}")
