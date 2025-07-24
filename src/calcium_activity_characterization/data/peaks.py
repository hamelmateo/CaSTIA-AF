"""
Peak class update to include symmetry score at the individual level.

Example:
    >>> peak = Peak(...)
    >>> print(peak.rel_symmetry_score)  # Already computed during creation
"""

from typing import Optional, Literal, List
import numpy as np
from scipy.signal import find_peaks, peak_widths, peak_prominences
from collections import defaultdict, deque
import logging
from calcium_activity_characterization.utilities.peak_utils import find_valley_bounds

from calcium_activity_characterization.config.presets import PeakDetectionConfig

logger = logging.getLogger(__name__)


class Peak:
    """
    Represents a calcium transient peak with timing, intensity, and hierarchical metadata.

    Attributes:
        id (int): Unique identifier for the peak.
        peak_time (int): Time at which the peak reaches its maximum intensity.
        start_time (int): Start time at baseline before the peak rises.
        end_time (int): End time when the peak returns to baseline.
        duration (int): Duration of the peak at baseline (end_time - start_time).
        fhw_start_time (int): Start time at relative height threshold.
        fhw_end_time (int): End time at relative height threshold.
        fhw_duration (int): Duration of the peak at relative height (fhw_end_time - fhw_start_time).
        ref_start_time (int): Reference start time used for event detection.
        ref_end_time (int): Reference end time used for event detection.
        ref_duration (int): Duration between reference start and end times.
        height (float): Absolute height of the peak.
        fhw_height (Optional[float]): Height of the peak relative to a reference value.
        prominence (float): Prominence of the peak compared to surrounding baseline.
        rise_time (int): Time from start_time to peak_time.
        decay_time (int): Time from peak_time to end_time.
        rel_rise_time (int): Time from fhw_start_time to peak_time.
        rel_decay_time (int): Time from peak_time to fhw_end_time.
        rel_symmetry_score (float): Symmetry score of the peak shape (1 = perfect symmetry).
        group_id (Optional[int]): ID of the overlapping group this peak belongs to.
        parent_peak_id (Optional[int]): ID of the parent peak if part of an overlapping group.
        grouping_type (Optional[Literal["individual", "child", "parent"]]): Role in the overlapping group.
        is_analyzed (bool): Flag indicating if this peak has been analyzed.
        in_event (Literal["global", "sequential"]): Type of event this peak is part of, if any.
        event_id (Optional[int]): ID of the event this peak is part of, if any.
        origin_type (Literal["origin", "caused", "individual"]): Type of cause for this peak.
        origin_label (Optional[int]): Label of the origin peak if this is a caused peak.
        in_cluster (bool): Flag indicating if this peak has been added to a cluster (deprecated).
        cluster_id (Optional[int]): ID of the assigned cluster (deprecated).
    """
    def __init__(
        self,
        id: int,
        fhw_start_time: int,
        peak_time: int,
        start_time: int,
        end_time: int,
        fhw_end_time: int,
        height: float,
        prominence: float,
        fhw_height: Optional[float] = None
    ):
        self.id = id

        # Timing metadata
        self.peak_time = peak_time

        self.start_time = start_time # Start time at baseline
        self.end_time = end_time
        self.duration = end_time - start_time

        self.fhw_start_time = fhw_start_time # Start time at relative height
        self.fhw_end_time = fhw_end_time
        self.fhw_duration = fhw_end_time - fhw_start_time

        self.ref_start_time: int # Start time used for reference in event detection
        self.ref_end_time: int
        self.ref_duration: int

        # Peak metadata
        self.height = height
        self.fhw_height = fhw_height
        self.prominence = prominence

        self.rise_time = peak_time - start_time
        self.decay_time = end_time - peak_time
        self.rel_rise_time = peak_time - fhw_start_time
        self.rel_decay_time = fhw_end_time - peak_time
        self.rel_symmetry_score: float = self._compute_symmetry_score()

        # Grouping metadata
        self.group_id: Optional[int] = None  # ID of the overlapping group this peak belongs to
        self.parent_peak_id: Optional[int] = None  # ID of the parent peak if this is part of an overlapping group
        self.grouping_type: Optional[Literal["individual", "child", "parent"]] = None # Role in the overlapping group

        # Event metadata
        self.is_analyzed: bool = False  # Flag to track if this peak has been analyzed
        self.in_event: Literal["global", "sequential"] = None  # Type of event this peak is part of, if any
        self.event_id: Optional[int] = None  # ID of the event this peak is part of, if any

        self.origin_type: Literal["origin", "caused", "individual"] = "individual" # Type of cause for this peak
        self.origin_label: Optional[int] = None # Label of the origin peak if this is a caused peak

        # Deprecated attributes for experimental methods
        self.in_cluster: bool = False  # Set True once this peak is added to a cluster
        self.cluster_id: Optional[int] = None  # ID of the assigned cluster


    def _compute_symmetry_score(self) -> Optional[float]:
        """
        Compute a symmetry score for the peak shape.

        Returns:
            Optional[float]: Symmetry score (1 = perfect symmetry), or None if invalid.
        """
        total_time = self.rel_rise_time + self.rel_decay_time
        if total_time > 0:
            return 1 - abs(self.rel_rise_time - self.rel_decay_time) / total_time
        return None

    def __repr__(self):
        return (
            f"Peak(id={self.id}, time={self.fhw_start_time}, height={self.height:.2f}, in_event={self.in_event})"
        )
    
    def define_ref_times(self, ref_start_time: int, ref_end_time: int):
        """
        Define reference start and end times for this peak.

        Args:
            ref_start_time (int): Reference start time.
            ref_end_time (int): Reference end time.
        """
        self.ref_start_time = ref_start_time
        self.ref_end_time = ref_end_time
        self.ref_duration = ref_end_time - ref_start_time

class PeakDetector:
    """
    Base scaffold for detecting peaks from a calcium trace.
    """
    def __init__(self, config: PeakDetectionConfig):
        """
        Initialize the PeakDetector with parameters.
        
        Args:
            params (PeakDetectionConfig): Parameters for peak detection.
        """
        self.config = config
        self.verbose = config.verbose
        self.method = config.method
        self.detection_params = config.params
        self.grouping_params = config.peak_grouping


    def run(self, trace: list[float]) -> list[Peak]:
        """
        Execute peak detection and processing on the provided trace.
        
        Args:
            trace (list[float]): Calcium intensity trace.
        
        Returns:
            list[Peak]: List of detected peaks.
        """
        peaks = self._detect(trace)
        peaks = self._group_overlapping_peaks(peaks)

        if self.config.refine_durations:
            peaks = self._refine_peak_durations(peaks, trace)

        if self.config.filter_overlapping_peaks:
            peaks = self._filter_children_peaks(peaks)
            peaks = reassign_peak_ids(peaks)

        for peak in peaks:
            peak.define_ref_times(peak.fhw_start_time, peak.fhw_end_time)

        return peaks


    def _detect(self, trace: list[float]) -> list[Peak]:
        """
        Detect peaks in the provided trace using the specified method.

        Args:
            trace (list[float]): Calcium intensity trace.

        Returns:
            list[Peak]: List of detected peaks.
        """

        trace = np.array(trace, dtype=float)

        start_frame = self.config.start_frame or 0
        end_frame = self.config.end_frame or len(trace)

        subtrace = trace[start_frame:end_frame]

        if self.method == "skimage":

            # Extract params safely from structured config
            from dataclasses import asdict

            kwargs = {
                key: value
                for key, value in asdict(self.detection_params).items()
                if key in ["prominence", "distance", "height", "threshold", "width"] and value is not None
            }

            peaks, _ = find_peaks(subtrace, **kwargs)

            # Compute peak prominences parameters
            prominences, _, _ = peak_prominences(subtrace, peaks)

            # Compute relative heights parameters
            _, _, rel_left_ips, rel_right_ips = peak_widths(subtrace, peaks, rel_height=self.detection_params.full_half_width)

            # Compute whole widths
            _, _, left_ips, right_ips = peak_widths(subtrace, peaks, rel_height=self.detection_params.full_duration_threshold)

            peak_list = []
            for i, peak_time in enumerate(peaks):
                fhw_start_time = int(np.floor(rel_left_ips[i])) + start_frame
                fhw_end_time = int(np.ceil(rel_right_ips[i])) + start_frame
                start_time = int(np.floor(left_ips[i])) + start_frame
                end_time = int(np.ceil(right_ips[i])) + start_frame
                prominence = float(prominences[i])
                height = float(subtrace[peak_time])

                peak = Peak(
                    id=i,
                    start_time=start_time,
                    end_time=end_time,
                    fhw_start_time=fhw_start_time,
                    peak_time=peak_time,
                    fhw_end_time=fhw_end_time,
                    height=height,
                    prominence=prominence,
                    fhw_height=self.detection_params.full_half_width
                )
                peak_list.append(peak)

            return peak_list
        else:
            raise NotImplementedError(f"Peak detection method '{self.method}' is not supported yet.")


    def _refine_peak_durations(self, peaks: List[Peak], trace: np.ndarray) -> List[Peak]:
        """
        Refine start/end times of each peak using find_valley_bounds.

        Args:
            peaks (List[Peak]): Detected peaks.
            trace (np.ndarray): Original trace.
        """
        trace = np.array(trace, dtype=float)

        for peak in peaks:
            start, end = find_valley_bounds(trace, peak.fhw_start_time, peak.fhw_end_time)
            peak.start_time = max(start, peak.start_time)
            peak.end_time = min(end, peak.end_time)
            peak.duration = peak.end_time - peak.start_time

        return peaks

    def _group_overlapping_peaks(self, peaks: List[Peak]) -> List[Peak]:
        """
        Identify parent-child peak relationships based on containment and height.

        # There are three categories for peak grouping:
        # - "individual": Peaks that do not overlap with any other peak and are not part of a group.
        # - "parent": Peaks that contain one or more overlapping (child) peaks within their duration and have the highest height in their group.
        # - "child": Peaks that are overlapped by a parent peak and are assigned to the same group as the parent.
        # These categories help distinguish isolated events ("individual"), main events with nested sub-peaks ("parent"), and overlapping sub-events ("child").

        Args:
            peaks (List[Peak]): List of detected peaks.

        Returns:
            List[Peak]: Peaks with group_id, parent_peak_id, and role assigned.
        """
        if not peaks:
            return []

        verbose = self.verbose
        group_id_counter = 0
        grouped_peaks = []
        seen_as_child = set()

        for parent in peaks:
            if parent.id in seen_as_child:
                continue  # already a child → skip

            current_group = []
            for child in peaks:
                if child.id == parent.id or child.id in seen_as_child:
                    continue

                fhw_overlap = max(0, min(parent.fhw_end_time, child.fhw_end_time) - max(parent.fhw_start_time, child.fhw_start_time))
                if fhw_overlap < 2:
                    continue

                if parent.height > child.height or (parent.height == child.height and parent.peak_time < child.peak_time):
                    # Valid child
                    child.group_id = group_id_counter
                    child.parent_peak_id = parent.id
                    child.grouping_type = "child"
                    seen_as_child.add(child.id)
                    current_group.append(child)

            if current_group:
                # Mark parent
                parent.group_id = group_id_counter
                parent.parent_peak_id = None
                parent.grouping_type = "parent"
                grouped_peaks.append(parent)
                grouped_peaks.extend(current_group)
                group_id_counter += 1
            else:
                # No children found → individual
                parent.group_id = None
                parent.parent_peak_id = None
                parent.grouping_type = "individual"
                grouped_peaks.append(parent)

        if verbose:
            total_groups = len(set(p.group_id for p in grouped_peaks if p.group_id is not None))
            logger.info(f"[PeakDetector] Formed {total_groups} containment-based groups.")
            for gid in set(p.group_id for p in grouped_peaks if p.group_id is not None):
                size = sum(p.group_id == gid for p in grouped_peaks)
                logger.info(f" - Group {gid}: {size} peaks")
            num_individuals = sum(p.grouping_type == "individual" for p in grouped_peaks)
            logger.info(f"[PeakDetector] Found {num_individuals} individual (non-nested) peaks.")

        return grouped_peaks

    

    def _filter_children_peaks(self, peaks: List[Peak]) -> List[Peak]:
        """
        Eliminate overlapping peaks, retaining only those with role='parent'.

        Args:
            peaks (List[Peak]): List of grouped peaks.

        Returns:
            List[Peak]: Filtered peaks.
        """
        filtered = [p for p in peaks if p.grouping_type == "individual" or p.grouping_type == "parent"]
        n_removed = len(peaks) - len(filtered)
        if n_removed > 0:
            if self.verbose:
                logger.info(f"[PeakDetector] Eliminated {n_removed} overlapping peaks (non-parents).")
        return filtered


def reassign_peak_ids(peaks: List[Peak]) -> List[Peak]:
        """
        Reassign sequential IDs to all peaks across all cells after overlap removal.
        IDs will start at 0 and increment globally.

        Args:
            peaks (List[Peak]): List of Peak objects to reassign IDs.

        Returns:
            List[Peak]: Peaks with reassigned IDs.
        """
        try:
            new_id = 0
            for peak in peaks:
                peak.id = new_id
                new_id += 1
            return peaks
        except Exception as e:
            logger.error(f"Failed to reassign peak IDs: {e}")
            raise