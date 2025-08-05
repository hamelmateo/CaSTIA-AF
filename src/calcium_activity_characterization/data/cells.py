import numpy as np
from calcium_activity_characterization.logger import logger

from calcium_activity_characterization.data.traces import Trace
from calcium_activity_characterization.config.presets import CellFilteringConfig, ObjectSizeThresholds





class Cell:
    """
    Represents a nucleus (cell) identified in a segmented image.

    Attributes:
        label (int): Unique label ID assigned to the cell.
        centroid (np.ndarray): Y, X coordinates of the center of mass.
        pixel_coords (np.ndarray): Array of (y, x) pixel coordinates belonging to the cell.
        trace (Trace): Associated calcium trace and analysis results.
        is_valid (bool): Whether the cell passes quality control (e.g., pixel count >= threshold).
        is_active (bool): Whether the cell is classified as active based on detected peaks.
        occurences_global_events (int): Number of unique global events this cell is involved in.
        occurences_sequential_events (int): Number of unique sequential events this cell is involved in
        occurences_individual_events (int): Number of individual event peaks in this cell.
        occurences_sequential_events_as_origin (int): Number of sequential events where this cell is
            the origin of the event.
    """

    def __init__(
        self,
        label: int,
        centroid: np.ndarray = None,
        pixel_coords: np.ndarray = None,
        object_size_thresholds: ObjectSizeThresholds = None,
    ) -> None:
        self.label = label
        self.centroid = centroid if centroid is not None else np.array([0, 0], dtype=int)
        self.pixel_coords = pixel_coords if pixel_coords is not None else np.empty((0, 2), dtype=int)
        self.is_valid: bool = len(self.pixel_coords) >= object_size_thresholds.min and len(self.pixel_coords) <= object_size_thresholds.max
        self.is_active: bool = False
        self.occurences_global_events: int = 0
        self.occurences_sequential_events: int = 0
        self.occurences_individual_events: int = 0
        self.occurences_sequential_events_as_origin: int = 0

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

    def define_activity(self) -> None:
        """
        Define whether the cell is active based on the presence of detected peaks.
        """
        self.is_active = len(self.trace.peaks) > 0

    def adjust_to_roi(self, start_h: int, start_w: int) -> None:
        """
        Adjust the cell's coordinates and centroid relative to a cropped ROI.

        Args:
            start_h (int): Row offset (start of ROI).
            start_w (int): Column offset (start of ROI).
        """
        self.pixel_coords[:, 0] -= start_h
        self.pixel_coords[:, 1] -= start_w
        cy, cx = self.centroid
        self.centroid = (cy - start_h, cx - start_w)

    def count_occurences_global_events(self) -> int:
        """
        Count the number of unique global events this cell is involved in.

        Returns:
            int: Number of unique global events this cell is involved in.
        """
        unique_event_ids = set(
            p.event_id for p in self.trace.peaks
            if p.in_event == "global"
        )
        self.occurences_global_events = len(unique_event_ids)
        return self.occurences_global_events

    def count_occurences_sequential_events(self) -> int:
        """
        Count the number of unique sequential events this cell is involved in.

        Returns:
            int: Number of unique sequential events this cell is involved in.
        """
        unique_event_ids = set(
            p.event_id for p in self.trace.peaks
            if p.in_event == "sequential"
        )
        self.occurences_sequential_events = len(unique_event_ids)
        return self.occurences_sequential_events
    
    def count_occurences_individual_events(self) -> int:
        """
        Count the number of peaks in this cell that are classified as individual events.

        Returns:
            int: Number of individual event peaks.
        """
        count = sum(1 for p in self.trace.peaks if p.in_event == "individual")
        self.occurences_individual_events = count
        return count

    def count_occurences_sequential_events_as_origin(self) -> int:
        """
        Count the number of peaks in this cell that are origins of sequential events.

        Returns:
            int: Number of origin peaks involved in sequential events.
        """
        unique_event_ids = set(
            p.event_id for p in self.trace.peaks
            if p.origin_type == "origin" and p.in_event == "sequential"
        )
        self.occurences_sequential_events_as_origin = len(unique_event_ids)
        return self.occurences_sequential_events_as_origin

    @classmethod
    def from_segmentation_mask(cls, mask: np.ndarray, cell_filtering_parameters: CellFilteringConfig) -> list["Cell"]:
        """
        Construct Cell instances from a labeled segmentation mask.

        Args:
            mask (np.ndarray): Labeled mask where each cell is identified by a unique integer.
            cell_filtering_parameters (CellFilteringConfig): Parameters for filtering cells, including:
                - object_size_thresholds: ObjectSizeThresholds with 'min' and 'max' pixel count thresholds.
                - border_margin: int, margin to exclude cells near the image border.

        Returns:
            list[Cell]: list of Cell instances parsed from the mask.
        """
        cells = []
        label = 1
        while np.any(mask == label):
            pixel_coords = np.argwhere(mask == label)
            if pixel_coords.size > 0:
                centroid = np.array(np.mean(pixel_coords, axis=0), dtype=int) # TODO tuple for the centroid?
                cell = cls(label=label, centroid=centroid, pixel_coords=pixel_coords, object_size_thresholds=cell_filtering_parameters.object_size_thresholds)

                h, w = mask.shape[:2]
                border_margin = cell_filtering_parameters.border_margin
                if (
                    centroid[0] < border_margin or centroid[1] < border_margin or
                    centroid[0] > h - border_margin or centroid[1] > w - border_margin
                ):
                    cell.is_valid = False

                cells.append(cell)
            label += 1
        return cells


