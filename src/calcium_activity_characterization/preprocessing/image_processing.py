"""
Module defining the ImageProcessor class for renaming, cleaning, and cropping calcium imaging frames.

Example:
    >>> processor = ImageProcessor(config)
    >>> stack = processor.process_all(images_dir, file_pattern)
    >>> img = processor.process_single_image(Path("frame0005.TIF"))
"""

import re
import numpy as np
from pathlib import Path
from typing import List
from scipy.ndimage import uniform_filter

from calcium_activity_characterization.utilities.loader import (
    load_existing_img,
    get_config_with_fallback
)

import logging
logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Applies preprocessing steps (padding, cropping, hot pixel correction) to calcium imaging frames.

    Args:
        config (dict): Configuration dictionary with the following keys:
            - apply: dict with keys 'padding', 'cropping', 'hot_pixel_cleaning'
            - padding_digits: int
            - roi_scale: float
            - hot_pixel_cleaning: dict with 'method', 'threshold', etc.
    """

    def __init__(self, config: dict):
        self.config = config
        self.apply = get_config_with_fallback(config, "apply", {})
        self.padding_digits = get_config_with_fallback(config, "padding_digits", 5)
        self.roi_scale = get_config_with_fallback(config, "roi_scale", 1.0)
        self.hot_cfg = get_config_with_fallback(config, "hot_pixel_cleaning", {})

    def process_all(self, images_dir: Path, file_pattern: str) -> List[np.ndarray]:
        """
        Load and process all TIF images in a directory.

        Args:
            images_dir (Path): Directory containing TIF files.
            file_pattern (str): Regex to identify frame number for padding.

        Returns:
            List[np.ndarray]: List of cleaned, cropped images.
        """
        if self.apply.get("padding", True):
            self.rename_with_padding(images_dir, file_pattern)

        image_paths = sorted(images_dir.glob("*.TIF"))
        images = [self.process_single_image(p) for p in image_paths]
        return np.stack(images, axis=0)

    def process_single_image(self, path: Path) -> np.ndarray:
        """
        Load, clean, and crop a single image.

        Args:
            path (Path): Path to TIF file.

        Returns:
            np.ndarray: Preprocessed 2D image.
        """
        img = load_existing_img(path)

        if self.apply.get("cropping", True):
            img = self._crop_image(img)

        if self.apply.get("hot_pixel_cleaning", False):
            img = self._clean_single_image(img, image_name=path.name)

        return img

    def rename_with_padding(self, images_dir: Path, file_pattern: str) -> None:
        """
        Rename TIF files to ensure frame numbers are zero-padded.

        Args:
            images_dir (Path): Folder containing image files.
            file_pattern (str): Regex pattern to extract frame number.
        """
        regex = re.compile(file_pattern)
        for file in images_dir.glob("*.TIF"):
            match = regex.search(file.name)
            if match:
                number = match.group(1)
                if len(number) >= self.padding_digits:
                    continue
                padded_number = number.zfill(self.padding_digits)
                new_name = file.name.replace(f"t{number}", f"t{padded_number}")
                new_path = images_dir / new_name
                file.rename(new_path)
                logger.info(f"Renamed: {file.name} -> {new_name}")


    def _crop_image(self, img: np.ndarray) -> np.ndarray:
        """
        Crop the image around its center using the configured ROI scale.

        Args:
            img (np.ndarray): Input 2D image.

        Returns:
            np.ndarray: Cropped image.
        """
        if self.roi_scale >= 1.0:
            return img

        height, width = img.shape[:2]
        crop_h = int(height * self.roi_scale)
        crop_w = int(width * self.roi_scale)
        start_h = (height - crop_h) // 2
        start_w = (width - crop_w) // 2
        return img[start_h:start_h + crop_h, start_w:start_w + crop_w]



    def _clean_single_image(self, img: np.ndarray, image_name: str = "") -> np.ndarray:
        """
        Clean hot pixels in a single image using thresholding method.

        Args:
            img (np.ndarray): Image array.
            image_name (str): Name used in logging.

        Returns:
            np.ndarray: Cleaned image.
        """
        method = get_config_with_fallback(self.hot_cfg, "method", "replace")
        window = get_config_with_fallback(self.hot_cfg, "window_size", 3)

        if get_config_with_fallback(self.hot_cfg, "use_auto_threshold", True):
            threshold = self._compute_auto_threshold(
                img,
                percentile=get_config_with_fallback(self.hot_cfg, "percentile", 99.9),
                scale=get_config_with_fallback(self.hot_cfg, "mad_scale", 10.0)
            )
        else:
            threshold = get_config_with_fallback(self.hot_cfg, "static_threshold", 1000.0)

        if method == "replace":
            return self._replace_hot_pixels(img, threshold, window, image_name)
        elif method == "clip":
            cleaned = np.clip(img, 0, threshold)
            if np.max(cleaned) > threshold:
                logger.warning(f"⚠️ Clipping failed in {image_name}")
            return cleaned
        else:
            raise ValueError(f"Unknown hot pixel cleaning method: {method}")

    @staticmethod
    def _replace_hot_pixels(img: np.ndarray, threshold: float, window: int, image_name: str = "") -> np.ndarray:
        """
        Replace pixels above threshold with local mean.

        Args:
            img (np.ndarray): Image to correct.
            threshold (float): Intensity threshold.
            window (int): Local window size.
            image_name (str): Logging context.

        Returns:
            np.ndarray: Corrected image.
        """
        mask = img > threshold
        num_hot = np.sum(mask)
        if num_hot == 0:
            return img

        local_mean = uniform_filter(img.astype(float), size=window, mode='reflect')
        corrected = img.copy()
        corrected[mask] = local_mean[mask]

        logger.info(f"🧽 Replaced {num_hot} hot pixels > {threshold} in {image_name or 'image'}")
        return corrected

    @staticmethod
    def _compute_auto_threshold(img: np.ndarray, percentile: float = 99.9, scale: float = 10.0) -> float:
        """
        Compute a dynamic threshold from percentile + MAD.

        Args:
            img (np.ndarray): Image data.
            percentile (float): Brightness percentile.
            scale (float): MAD scaling factor.

        Returns:
            float: Computed threshold.
        """
        base = np.percentile(img, percentile)
        mad = np.median(np.abs(img - base))
        return base + scale * mad
