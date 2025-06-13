import numpy as np
import matplotlib.pyplot as plt
import logging
import pandas as pd

from calcium_activity_characterization.data.traces import Trace




logger = logging.getLogger(__name__)

class Cell:
    """
    Represents a nucleus (cell) identified in a segmented image.

    Attributes:
        label (int): Unique label ID assigned to the cell.
        centroid (np.ndarray): Y, X coordinates of the center of mass.
        pixel_coords (np.ndarray): Array of (y, x) pixel coordinates belonging to the cell.
        trace (Trace): Associated calcium trace and analysis results.
        is_valid (bool): Whether the cell passes quality control (e.g., pixel count >= threshold).
    """

    def __init__(
        self,
        label: int,
        centroid: np.ndarray = None,
        pixel_coords: np.ndarray = None,
        object_size_thresholds: dict = None,
    ) -> None:
        self.label = label
        self.centroid = centroid if centroid is not None else np.array([0, 0], dtype=int)
        self.pixel_coords = pixel_coords if pixel_coords is not None else np.empty((0, 2), dtype=int)
        self.is_valid: bool = len(self.pixel_coords) >= object_size_thresholds.get("min",200) and len(self.pixel_coords) <= object_size_thresholds.get("max", 10000)
        self.exclude_from_umap = False

        self.trace: Trace = Trace()

    def add_mean_intensity(self, image: np.ndarray) -> None:
        """
        Add the mean intensity value from a new image (timepoint) to the cell.

        Args:
            image (np.ndarray): A 2D grayscale image.
        """
        try:
            intensities = [image[y, x] for y, x in self.pixel_coords]
            mean_intensity = np.mean(intensities)
            self.trace.versions["raw"].append(mean_intensity)
        except Exception as e:
            logger.error(f"Failed to compute mean intensity for cell {self.label}: {e}")

    def plot_raw_intensity_profile(self) -> None:
        """
        Plot the mean intensity profile of the cell over time.
        """
        if len(self.trace.versions["raw"]) == 0:
            logger.info(f"Cell {self.label} has no intensity data to plot.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.trace.versions["raw"], label=f"Cell {self.label}")
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
        active = self.trace.active_trace
        if len(active) == 0:
            logger.info(f"Cell {self.label} has no processed intensity data to plot.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(active, label=f"Cell {self.label}")
        plt.title(f"Processed Intensity Profile for Cell {self.label}")
        plt.xlabel("Timepoint")
        plt.ylabel("Mean Intensity")
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_arcos_dataframe(self) -> pd.DataFrame:
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
        if not self.trace.binary:
            logger.warning(f"Cell {self.label} has no binary trace.")
            return pd.DataFrame()

        data = {
            'frame': range(len(self.trace.binary)),
            'trackID': [self.label] * len(self.trace.binary),
            'x': [self.centroid[1]] * len(self.trace.binary),
            'y': [self.centroid[0]] * len(self.trace.binary),
            'intensity': self.trace.versions["raw"],
            'intensity.bin': self.trace.binary
        }

        return pd.DataFrame(data)
    
    @classmethod
    def from_segmentation_mask(cls, mask: np.ndarray, cell_filtering_parameters: dict) -> list["Cell"]:
        """
        Construct Cell instances from a labeled segmentation mask.

        Args:
            mask (np.ndarray): Labeled mask where each cell is identified by a unique integer.
            cell_filtering_parameters (dict): Parameters for filtering cells, including:
                - object_size_thresholds: dict with 'min' and 'max' pixel count thresholds.
                - border_margin: int, margin to exclude cells near the image border.

        Returns:
            List[Cell]: List of Cell instances parsed from the mask.
        """
        cells = []
        label = 1
        while np.any(mask == label):
            pixel_coords = np.argwhere(mask == label)
            if pixel_coords.size > 0:
                centroid = np.array(np.mean(pixel_coords, axis=0), dtype=int)
                cell = cls(label=label, centroid=centroid, pixel_coords=pixel_coords, object_size_thresholds=cell_filtering_parameters["object_size_thresholds"])

                h, w = mask.shape[:2]
                border_margin = cell_filtering_parameters["border_margin"]
                if (
                    centroid[0] < border_margin or centroid[1] < border_margin or
                    centroid[0] > h - border_margin or centroid[1] > w - border_margin
                ):
                    cell.is_valid = False

                cells.append(cell)
            label += 1
        return cells


