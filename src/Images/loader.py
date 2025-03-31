from pathlib import Path
import cv2
import numpy as np


def load_and_crop_images(dir: Path, roi_scale: float) -> np.ndarray:
    """
    Load .tif images in the folder (16-bit grayscale) and crop them to a specific ROI.

    Parameters:
        dir (Path): Path to folder.
        roi_scale (float): Scale factor for cropping (e.g., 0.75 for 75%).
    
    Returns:
        (np.ndarray): An array of cropped images from the folder.
    """

    images = list(dir.glob("*.TIF"))
    if not images:
        raise FileNotFoundError(f"No .tif images found in {dir}")
    
    print(f"[INFO] Number of images found: {len(images)}")  # Print the number of images found

    loaded_images = [cv2.imread(str(image), cv2.IMREAD_UNCHANGED) for image in images]

    if any(img is None for img in loaded_images):
        raise ValueError(f"Could not load one or more images from {dir}")

    cropped_images = [crop_image(img, roi_scale) for img in loaded_images]  # Crop images to 75% of original size

    return np.array(loaded_images)


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