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
import cupy as cp
import cupyx.scipy.ndimage

from calcium_activity_characterization.config.presets import ImageProcessingConfig
from calcium_activity_characterization.io.images_loader import (
    load_existing_img,
    load_image_fast
)

from calcium_activity_characterization.logger import logger



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

    def __init__(self, config: ImageProcessingConfig):
        self.config = config
        self.pipeline = config.pipeline
        self.padding_digits = config.padding_digits
        self.roi_scale = config.roi_scale
        self.hot_cfg = config.hot_pixel_cleaning

    def process_all(self, images_dir: Path, file_pattern: str) -> list[np.ndarray]:
        """
        Load and process all TIF images in a directory.

        Args:
            images_dir (Path): Directory containing TIF files.
            file_pattern (str): Regex to identify frame number for padding.

        Returns:
            list[np.ndarray]: list of cleaned, cropped images.
        """
        if self.pipeline.padding:
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
        try:
            img = load_image_fast(path)
        except Exception as e:
            logger.warning(f"⚠️ Fast load failed for {path.name}: {e}. Falling back to standard loader.") 
            img = load_existing_img(path)
        if self.pipeline.cropping:
            img = self._crop_image(img)
        if self.pipeline.hot_pixel_cleaning:
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

    def _clean_single_image_gpu(self, img: np.ndarray, image_name: str = "") -> np.ndarray:
        """
        GPU-compatible hot pixel cleaning. Keeps final result on CPU.
        Uses MAD and cp.percentile, with CuPy acceleration.

        Args:
            img (np.ndarray): Image array.
            image_name (str): Name used in logging.

        Returns:
            np.ndarray: Cleaned image (still on CPU).
        """
        try:
            window = self.hot_cfg.window_size

            img_gpu = cp.asarray(img, dtype=cp.float32)

            if self.hot_cfg.use_auto_threshold:
                percentile = self.hot_cfg.percentile
                scale = self.hot_cfg.mad_scale

                stride = 4
                sampled = img_gpu[::stride, ::stride]  # Downsample by 4 in both axes → 1/16th the pixels
                base = cp.percentile(sampled, percentile)

                residual = cp.abs(img_gpu - base)
                mad = cp.median(residual)
                threshold = base + scale * mad

            else:
                threshold = self.hot_cfg.static_threshold

            # Try uniform filter instead of median if you want speed over accuracy
            local_median = cupyx.scipy.ndimage.median_filter(img_gpu, size=window)
            # local_median = cupyx.scipy.ndimage.uniform_filter(img_gpu, size=window)

            diff = cp.abs(img_gpu - local_median)
            hot_mask = diff > threshold

            corrected = img_gpu.copy()
            corrected[hot_mask] = local_median[hot_mask]

            num_hot = int(cp.sum(hot_mask).get())
            if num_hot > 0:
                logger.info(f"Replaced {num_hot} hot pixels > {threshold:.2f} in {image_name or 'image'}")

            return corrected.get()

        except Exception as e:
            logger.error(f"GPU hot pixel correction failed for {image_name}: {e}")
            raise

    def _clean_single_image(self, img: np.ndarray, image_name: str = "") -> np.ndarray:
        """
        Clean hot pixels in a single image using thresholding method.

        Args:
            img (np.ndarray): Image array.
            image_name (str): Name used in logging.

        Returns:
            np.ndarray: Cleaned image.
        """
        method = self.hot_cfg.method
        window = self.hot_cfg.window_size

        if self.hot_cfg.use_auto_threshold:
            threshold = self._compute_auto_threshold(
                img,
                percentile=self.hot_cfg.percentile,
                scale=self.hot_cfg.mad_scale
            )

        else:
            threshold = self.hot_cfg.static_threshold
        if method == "replace":
            images = self._replace_hot_pixels(img, threshold, window, image_name)
            return images
        elif method == "clip":
            cleaned = np.clip(img, 0, threshold)

            if np.max(cleaned) > threshold:
                logger.warning(f"⚠️ Clipping failed in {image_name}")
            return cleaned
        else:
            raise ValueError(f"Unknown hot pixel cleaning method: {method}")

    @staticmethod
    def _replace_hot_pixels(img: np.ndarray, threshold: float, window: int, image_name: str | None = "") -> np.ndarray:
        """
        Replace hot pixels using local mean around each pixel individually.
        Much faster when very few hot pixels are present.

        Args:
            img (np.ndarray): Input image (uint16 or float32).
            threshold (float): Intensity threshold to detect hot pixels.
            window (int): Window size for local averaging.
            image_name (str): Optional name for logging context.

        Returns:
            np.ndarray: Cleaned image.
        """
        mask = img > threshold
        hot_indices = np.argwhere(mask)

        num_hot = len(hot_indices)
        if num_hot == 0:
            return img

        corrected = img.copy()
        r = window // 2
        H, W = img.shape

        for y, x in hot_indices:
            y_min = max(0, y - r)
            y_max = min(H, y + r + 1)
            x_min = max(0, x - r)
            x_max = min(W, x + r + 1)

            patch = img[y_min:y_max, x_min:x_max]
            corrected[y, x] = np.mean(patch)

        #logger.info(f"🧽 Locally replaced {num_hot} hot pixels > {threshold} in {image_name or 'image'}")
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
        stride = 4
        sampled = img[::stride, ::stride]
        base = np.percentile(sampled, 99.9)
        mad = np.median(np.abs(img - base))
        return base + scale * mad