from pathlib import Path
import cv2
import numpy as np

def load_images(dir: Path) -> np.ndarray:
    """
    Load .tif images in the folder (16-bit grayscale).

    Parameters:
        dir (Path): Path to folder.
    
    Returns:
        (np.ndarray): An array of images from the folder.
    """

    images = list(dir.glob("*.TIF"))
    if not images:
        raise FileNotFoundError(f"No .tif images found in {dir}")
    
    print(f"[INFO] Number of images found: {len(images)}")  # Print the number of images found

    loaded_images = [cv2.imread(str(image), cv2.IMREAD_UNCHANGED) for image in images]

    if any(img is None for img in loaded_images):
        raise ValueError(f"Could not load one or more images from {dir}")

    return np.array(loaded_images)
