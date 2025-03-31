from typing import Tuple, List
import numpy as np


class Cell:
    """
    Represents a nuclei (cell) identified in a segmented image.

    Attributes:
        label (int): Unique label ID assigned to the cell.
        centroid (np.ndarray): Y, X coordinates of the center of mass.
        pixel_coords (np.ndarray): Array of (y, x) pixel coordinates belonging to the cell.
        intensity_traces (np.ndarray): Array of pixel intensities per image in the time series.
        is_valid (bool): Whether the cell passes quality control (e.g., pixel count >= 20).
    """

    def __init__(
        self,
        label: int,
        centroid: np.ndarray = np.empty((2,)),  # Initialize as an empty array with shape (2,)
        pixel_coords: np.ndarray = np.empty((0, 2)),  # Initialize as an empty array with shape (0, 2)
        roi_min_pixel_count: int = 20
    ):
        self.label = label
        self.centroid = centroid
        self.pixel_coords = pixel_coords
        self.intensity_traces = np.empty((0,))  # Initialize as an empty array with shape (0,)
        self.is_valid = len(pixel_coords) >= roi_min_pixel_count


    def add_timepoint(self, image: np.ndarray) -> None:
        """
        Add intensity values from a new image (timepoint).

        Args:
            image (np.ndarray): 2D image (H, W) from which to extract pixel intensities.
        """
        intensities = [image[y, x] for y, x in self.pixel_coords]
        self.intensity_traces = np.append(self.intensity_traces, [intensities], axis=0)

    def get_mean_intensity_trace(self) -> List[float]:
        """
        Returns the mean intensity over time for this cell.

        Returns:
            List[float]: One mean value per timepoint.
        """
        return [np.mean(trace) for trace in self.intensity_traces]
        
    def set_inactive(self) -> None:
            """
            Sets the cell as inactive by marking it as invalid.
            """
            self.is_valid = False