import numpy as np
import matplotlib.pyplot as plt
from src.config.config import SMALL_OBJECT_THRESHOLD
import logging
from typing import Optional
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfilt
from src.analysis.signal_processing import process_trace

logger = logging.getLogger(__name__)

class Cell:
    """
    Represents a nucleus (cell) identified in a segmented image.

    Attributes:
        label (int): Unique label ID assigned to the cell.
        centroid (np.ndarray): Y, X coordinates of the center of mass.
        pixel_coords (np.ndarray): Array of (y, x) pixel coordinates belonging to the cell.
        raw_intensity_trace (list[int]): Mean intensity values over time.
        is_valid (bool): Whether the cell passes quality control (e.g., pixel count >= threshold).
    """

    def __init__(
        self,
        label: int,
        centroid: Optional[np.ndarray] = None,
        pixel_coords: Optional[np.ndarray] = None
    ) -> None:
        self.label = label
        self.centroid = centroid if centroid is not None else np.array([0, 0], dtype=int)
        self.pixel_coords = pixel_coords if pixel_coords is not None else np.empty((0, 2), dtype=int)
        self.raw_intensity_trace: list[float] = []
        self.processed_intensity_trace: list[float] = []
        self.is_valid: bool = len(self.pixel_coords) >= SMALL_OBJECT_THRESHOLD
        self.exclude_from_umap = False
        self.has_good_exponential_fit: bool = True


    def add_mean_intensity(self, image: np.ndarray) -> None:
        """
        Add the mean intensity value from a new image (timepoint) to the cell.

        Args:
            image (np.ndarray): A 2D grayscale image.
        """
        try:
            intensities = [image[y, x] for y, x in self.pixel_coords]
            mean_intensity = np.mean(intensities)
            self.raw_intensity_trace.append(mean_intensity)
        except Exception as e:
            logger.error(f"Failed to compute mean intensity for cell {self.label}: {e}")

    def plot_raw_intensity_profile(self) -> None:
        """
        Plot the mean intensity profile of the cell over time.
        """
        if len(self.raw_intensity_trace) == 0:
            logger.info(f"Cell {self.label} has no intensity data to plot.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.raw_intensity_trace, label=f"Cell {self.label}")
        plt.title(f"Raw Intensity Profile for Cell {self.label}")
        plt.xlabel("Timepoint")
        plt.ylabel("Mean Intensity")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_processed_intensity_profile(self) -> None:
        """
        Plot the mean intensity profile of the cell over time.
        """
        if len(self.processed_intensity_trace) == 0:
            logger.info(f"Cell {self.label} has no intensity data to plot.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.processed_intensity_trace, label=f"Cell {self.label}")
        plt.title(f"Processed Intensity Profile for Cell {self.label}")
        plt.xlabel("Timepoint")
        plt.ylabel("Mean Intensity")
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_processed_trace(self, sigma: float = 1.0) -> None:
        """
        Apply the full processing pipeline to the raw trace:
        detrending, smoothing, and ΔF/F₀ normalization.

        Args:
            sigma (float): Gaussian smoothing sigma.
        """
        if not self.raw_intensity_trace:
            logger.info(f"Cell {self.label} has no intensity data to process.")
            self.processed_intensity_trace = []
            return

        try:
            self.processed_intensity_trace = process_trace(
                self.raw_intensity_trace, sigma=sigma
            )
            """self.processed_intensity_trace, r_squared = process_trace(
                self.raw_intensity_trace, sigma=sigma
            )
            self.has_good_exponential_fit = r_squared >= 0.8"""
        except Exception as e:
            logger.error(f"Failed to process trace for cell {self.label}: {e}")
            self.processed_intensity_trace = []
            self.has_good_exponential_fit = False
