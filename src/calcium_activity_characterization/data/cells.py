import numpy as np
import matplotlib.pyplot as plt
import logging
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from calcium_activity_characterization.data.peaks import Peak


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
        centroid: np.ndarray = None,
        pixel_coords: np.ndarray = None,
        small_object_threshold: int = 100,
        big_object_threshold: int = 10000
    ) -> None:
        self.label = label
        self.centroid = centroid if centroid is not None else np.array([0, 0], dtype=int)
        self.pixel_coords = pixel_coords if pixel_coords is not None else np.empty((0, 2), dtype=int)
        self.raw_intensity_trace: list[float] = []
        self.processed_intensity_trace: list[float] = []
        self.binary_trace: list[int] = []
        self.peaks: list["Peak"] = []
        self.is_valid: bool = len(self.pixel_coords) >= small_object_threshold and len(self.pixel_coords) <= big_object_threshold
        self.exclude_from_umap = False



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

    def get_arcos_dataframe(self, pipeline_type: str) -> pd.DataFrame:
            """
            Return a DataFrame formatted for arcos4py binarization and event tracking.

            The DataFrame contains the following columns:
                - frame: Timepoint index.
                - trackID: Unique cell identifier.
                - x: X-coordinate of centroid.
                - y: Y-coordinate of centroid.
                - intensity: Raw intensity trace.

            Returns:
                pd.DataFrame: DataFrame with cell information formatted for arcos4py.
            """
            if not self.raw_intensity_trace:
                logger.warning(f"Cell {self.label} has no intensity data.")
                return pd.DataFrame()

            if pipeline_type == "arcos":
                trace = self.raw_intensity_trace
            elif pipeline_type == "custom":
                trace = self.binary_trace

            data = {
                'frame': range(len(trace)),
                'trackID': [self.label] * len(trace),
                'x': [self.centroid[1]] * len(trace),  # centroid[1] is X-coordinate
                'y': [self.centroid[0]] * len(trace),  # centroid[0] is Y-coordinate
                'intensity': self.raw_intensity_trace,
                'intensity.bin': self.binary_trace
            }

            return pd.DataFrame(data)
    

    def detect_peaks(self, detector) -> None:
        if len(self.processed_intensity_trace) == 0:
            self.peaks = []
            return

        self.peaks = detector.run(self.processed_intensity_trace)

    def binarize_trace_from_peaks(self) -> None:
        """
        Generate binarized trace where values are 1 during peak durations, 0 otherwise.
        """
        if self.processed_intensity_trace is None or len(self.processed_intensity_trace) == 0:
            self.binary_trace = []
            return

        trace_length = len(self.processed_intensity_trace)
        binary = np.zeros(trace_length, dtype=int)

        for peak in self.peaks:
            start = max(0, peak.start_time)
            end = min(trace_length, peak.end_time + 1)
            binary[start:end] = 1

        self.binary_trace = binary.tolist()