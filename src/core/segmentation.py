from deepcell.applications import Mesmer
from deepcell.utils.plot_utils import make_outline_overlay
import numpy as np
from matplotlib import pyplot as plt



def segmented(images: np.ndarray = None, output_path: str = "overlay.png", save_overlay: bool = True) -> np.ndarray:
    """
    Perform nuclear segmentation on 16-bit grayscale Hoechst images using the Mesmer model.

    Args:
        images (np.ndarray): 2D NumPy array representing a single-channel (grayscale) image
                             with dtype=np.uint16. Required shape: (H, W).
        output_path (str): Path to save the overlay image. Default is "overlay.png".
        save_overlay (bool): Whether to generate and save a colored overlay of segmentation.

    Returns:
        np.ndarray: A labeled mask (2D) of the same height and width as the input image.
                    Each nucleus is assigned a unique integer label.
    
    Raises:
        ValueError: If `images` is not provided or is not a 2D uint16 array.
    """

    # --- Validation ---
    if images is None:
        raise ValueError("No image provided. You must pass a 2D grayscale image as `images`.")

    if not isinstance(images, np.ndarray):
        raise TypeError("Input `images` must be a NumPy array.")
    
    if images.ndim != 3:
        raise ValueError(f"Expected a 3D array (list of grayscale images), got shape {images.shape}.")
    
    if images.dtype != np.uint16:
        raise ValueError(f"Expected dtype=np.uint16 for 16-bit grayscale image, got {images.dtype}.")

    # --- Prepare image for Mesmer ---
    images_hd = np.stack((images, np.zeros_like(images)), axis=-1)  # shape: (H, W, 2)
    #images_4d = np.expand_dims(images_4d, axis=0)  # shape: (1, H, W, 2)

    print(f"[INFO] Image shape after stacking: {images_hd.shape}")

    # --- Load Mesmer model ---
    app = Mesmer()

    # --- Perform segmentation ---
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
    )  # shape: (1, H, W, 1)

    nuclei_mask = nuclei_mask[0, ..., 0]  # remove batch and channel dimensions (keep only the mask of the first image)
    print(f"[INFO] Segmentation mask shape: {nuclei_mask.shape}")

    # --- Save overlay if needed ---
    if save_overlay:
        overlay_image = make_outline_overlay(images_hd, nuclei_mask[np.newaxis, ..., np.newaxis])
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(overlay_image[0, ..., 0])
        ax.set_title("Nuclear Segmentation Overlay")
        ax.axis("off")
        fig.savefig(output_path)
        plt.close(fig)
        print(f"[INFO] Overlay image saved to {output_path}")

    return nuclei_mask




