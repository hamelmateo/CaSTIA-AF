import numpy as np
from src.config.config import SMALL_OBJECT_THRESHOLD
import matplotlib.pyplot as plt


class Cell:
    """
    Represents a nuclei (cell) identified in a segmented image.

    Attributes:
        label (int): Unique label ID assigned to the cell.
        centroid (np.ndarray): Y, X coordinates of the center of mass.
        pixel_coords (np.ndarray): Array of (y, x) pixel coordinates belonging to the cell.
        intensity_traces (np.ndarray): Array of pixel intensities per image in the time series.
        mean_intensity_trace (np.ndarray): Mean intensity over time for this cell.
        is_valid (bool): Whether the cell passes quality control (e.g., pixel count >= 20).
    """

    def __init__(
        self,
        label: int,
        centroid: np.ndarray = np.array([0, 0], dtype=int),  # Initialize as an integer array
        pixel_coords: np.ndarray = np.empty((0, 2), dtype=int),  # Initialize as an empty integer array
    ):
        self.label = label
        self.centroid = centroid
        self.pixel_coords = pixel_coords
        self.intensity_trace = []  # Initialize as an empty list for mean intensity values
        self.is_valid = len(pixel_coords) >= SMALL_OBJECT_THRESHOLD


    def add_mean_intensity(self, image: np.ndarray) -> None:
        """
        Add the mean intensity value from a new image (timepoint).
        """
        intensities = [image[y, x] for y, x in self.pixel_coords]
        mean_intensity = int(np.mean(intensities))

        self.intensity_trace.append(mean_intensity)


    def plot_intensity_profile(self) -> None:
        """
        Plot the mean intensity profile of the cell over time.
        """
        if len(self.intensity_trace) == 0:
            print(f"[INFO] Cell {self.label} has no intensity data to plot.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.intensity_trace, label=f"Cell {self.label}")
        plt.title(f"Intensity Profile for Cell {self.label}")
        plt.xlabel("Timepoint")
        plt.ylabel("Mean Intensity")
        plt.legend()
        plt.grid(True)
        plt.show()


