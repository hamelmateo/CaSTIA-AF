import numpy as np
from deepcell.applications import Mesmer
from deepcell.utils.plot_utils import make_outline_overlay
from pathlib import Path
import logging

from utilities.loader import save_tif_image

logger = logging.getLogger(__name__)

def segmented(images: np.ndarray, output_path: Path, save_overlay: bool) -> np.ndarray:
    """
    Perform nuclear segmentation on 16-bit grayscale Hoechst images using the Mesmer model.

    Args:
        images (np.ndarray): 3D NumPy array of shape (T, H, W) representing grayscale images.
        output_path (Path): Path to save the overlay image.
        save_overlay (bool): Whether to generate and save a colored overlay of segmentation.

    Returns:
        np.ndarray: A labeled mask (2D) where each nucleus is assigned a unique integer label.

    Raises:
        ValueError: If `images` is not provided or is not a valid 3D uint16 array.
        TypeError: If `images` is not a NumPy array.
    """
    if images is None:
        raise ValueError("No image provided. You must pass a 3D grayscale image stack as `images`.")

    if not isinstance(images, np.ndarray):
        raise TypeError("Input `images` must be a NumPy array.")

    if images.ndim != 3:
        raise ValueError(f"Expected a 3D array (list of grayscale images), got shape {images.shape}.")

    if images.dtype != np.uint16:
        raise ValueError(f"Expected dtype=np.uint16 for 16-bit grayscale image, got {images.dtype}.")

    # Stack input as required by Mesmer (grayscale in channel 0, zeros in channel 1)
    images_hd = np.stack((images, np.zeros_like(images)), axis=-1)
    logger.debug(f"Image shape after stacking: {images_hd.shape}")

    try:
        app = Mesmer()
        nuclei_mask = app.predict(
            images_hd,
            image_mpp=0.5,
            compartment='nuclear',
            postprocess_kwargs_nuclear={
                'maxima_threshold': 0.2,
                'maxima_smooth': 2.5,
                'interior_threshold': 0.05,
                'interior_smooth': 1,
                'small_objects_threshold': 25,
                'fill_holes_threshold': 15,
                'radius': 2
            }
        )
        nuclei_mask = nuclei_mask[0, ..., 0]  # remove batch and channel dims
        logger.info(f"Segmentation mask generated with shape: {nuclei_mask.shape}")

        if save_overlay:
            overlay_image = make_outline_overlay(images_hd, nuclei_mask[np.newaxis, ..., np.newaxis])
            save_tif_image(overlay_image[0, ..., 0], output_path)

        return nuclei_mask

    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise
