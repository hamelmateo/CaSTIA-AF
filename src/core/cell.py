import numpy as np
import matplotlib.pyplot as plt
from src.config.config import SMALL_OBJECT_THRESHOLD
import logging
from typing import Optional
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)

class Cell:
    """
    Represents a nucleus (cell) identified in a segmented image.

    Attributes:
        label (int): Unique label ID assigned to the cell.
        centroid (np.ndarray): Y, X coordinates of the center of mass.
        pixel_coords (np.ndarray): Array of (y, x) pixel coordinates belonging to the cell.
        intensity_trace (list[int]): Mean intensity values over time.
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
        self.intensity_trace: list[int] = []
        self.is_valid: bool = len(self.pixel_coords) >= SMALL_OBJECT_THRESHOLD

    def add_mean_intensity(self, image: np.ndarray) -> None:
        """
        Add the mean intensity value from a new image (timepoint) to the cell.

        Args:
            image (np.ndarray): A 2D grayscale image.
        """
        try:
            intensities = [image[y, x] for y, x in self.pixel_coords]
            mean_intensity = int(np.mean(intensities))
            self.intensity_trace.append(mean_intensity)
        except Exception as e:
            logger.error(f"Failed to compute mean intensity for cell {self.label}: {e}")

    def plot_intensity_profile(self) -> None:
        """
        Plot the mean intensity profile of the cell over time.
        """
        if len(self.intensity_trace) == 0:
            logger.info(f"Cell {self.label} has no intensity data to plot.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.intensity_trace, label=f"Cell {self.label}")
        plt.title(f"Intensity Profile for Cell {self.label}")
        plt.xlabel("Timepoint")
        plt.ylabel("Mean Intensity")
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_processed_trace(self, sigma: float = 2.0, cutoff: float = 0.01, fs: float = 1.0, order: int = 2) -> np.ndarray:
        """
        Return the processed intensity trace:
        1. High-pass filter to remove photobleaching
        2. Gaussian smoothing to reduce noise
        3. Normalization to [0, 1] range with baseline = 0

        Args:
            sigma (float): Standard deviation for Gaussian smoothing.
            cutoff (float): High-pass filter cutoff frequency (Hz).
            fs (float): Sampling frequency (Hz).
            order (int): Filter order.

        Returns:
            np.ndarray: Normalized and denoised trace.
        """
        if not self.intensity_trace:
            return np.array([])

        raw = np.array(self.intensity_trace)

        # Step 1: High-pass filter (remove low-frequency drift)
        nyquist = 0.5 * fs
        norm_cutoff = cutoff / nyquist
        b, a = butter(order, norm_cutoff, btype='high', analog=False)
        detrended = filtfilt(b, a, raw)

        # Step 2: Gaussian smoothing
        smoothed = gaussian_filter1d(detrended, sigma=sigma)

        # Step 3: Normalize to [0, 1]
        baseline = np.percentile(smoothed, 10)
        peak = np.max(smoothed)
        normalized = (smoothed - baseline) / (peak - baseline + 1e-8)

        return normalized
