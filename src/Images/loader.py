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
    
    loaded_images = cv2.imread(str(images))

    if loaded_images is None:
        raise ValueError(f"Could not load image: {images}")

    return loaded_images
