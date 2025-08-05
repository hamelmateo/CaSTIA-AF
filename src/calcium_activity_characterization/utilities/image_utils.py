import numpy as np

from calcium_activity_characterization.logger import logger


def convert_grayscale16_to_rgb(
    img16: np.ndarray
) -> np.ndarray:
    """
    Convert a 2D 16-bit (or float) image into an 8-bit RGB array.

    Usage example:
        >>> rgb = convert_grayscale16_to_rgb(raw16)

    Args:
        img16 (np.ndarray): 2D array of dtype uint16 or float.

    Returns:
        np.ndarray: H×W×3 uint8 RGB version of the input, linearly scaled.
    """
    if img16.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {img16.shape}")

    # Force float for scaling
    data = img16.astype(np.float32)

    mn, mx = float(data.min()), float(data.max())
    logger.debug(f"Grayscale16 → float32: min={mn}, max={mx}")

    if mx <= mn:
        raise ValueError("Zero dynamic range: all pixels have the same value")

    # normalize to [0,1]
    norm = (data - mn) / (mx - mn)
    # scale to [0,255]
    gray8 = (norm * 255.0).round().astype(np.uint8)
    logger.debug(f"After scaling: dtype={gray8.dtype}, min={gray8.min()}, max={gray8.max()}")

    # Stack into RGB
    rgb = np.stack([gray8, gray8, gray8], axis=-1)
    return rgb

def render_cell_outline_overlay(
    background_image: np.ndarray,
    outline_mask: np.ndarray
) -> np.ndarray:
    """
    Create an RGB overlay image by converting a 16-bit grayscale background to RGB
    and overlaying a red outline where the mask is True.

    Args:
        background_image (np.ndarray):
            2D array (H×W) of dtype uint16 or float representing a grayscale image.
        outline_mask (np.ndarray):
            2D boolean array (H×W) where True indicates outline pixels.

    Returns:
        np.ndarray: H×W×3 uint8 RGB image with red outlines.

    Raises:
        ValueError: If shapes are incompatible or inputs invalid.
    """
    try:
        # Convert background to 8-bit RGB
        rgb = convert_grayscale16_to_rgb(background_image)

        # Ensure mask is boolean and shape matches
        mask = np.asarray(outline_mask, dtype=bool)
        if mask.shape != rgb.shape[:2]:
            raise ValueError(
                f"Outline mask shape {mask.shape} does not match "
                f"background image shape {rgb.shape[:2]}"
            )

        # Overlay red on masked pixels
        overlay = rgb.copy()
        overlay[mask, 0] = 255
        overlay[mask, 1] = 0
        overlay[mask, 2] = 0

        return overlay

    except Exception as e:
        logger.error(f"RenderCellOutlineOverlay failed: {e}")
        raise