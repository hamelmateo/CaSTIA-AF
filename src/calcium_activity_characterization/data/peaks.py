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
        id (int): Unique peak identifier.
        peak_time (int): Frame index of the peak maximum.
        start_time (int): Absolute start frame of the peak in the full trace.
        end_time (int): Absolute end frame of the peak in the full trace.
        duration (int): Absolute duration of the peak (end_time - start_time).
        rel_start_time (int): Relative start frame of the peak within the processing window.
        rel_end_time (int): Relative end frame of the peak within the processing window.
        rel_duration (int): Duration of the peak in the relative frame (rel_end_time - rel_start_time).
        height (float): Intensity at the peak maximum.
        prominence (float): Peak prominence from baseline.
        rel_height (Optional[float]): Peak height relative to the baseline in the window.
        rise_time (int): Time from absolute start to peak.
        decay_time (int): Time from peak to absolute end.
        rel_rise_time (int): Time from relative start to peak.
        rel_decay_time (int): Time from peak to relative end.
        rel_symmetry_score (float): Symmetry of the peak shape based on relative rise vs decay.
        group_id (Optional[int]): Overlapping group ID (if any).
        parent_peak_id (Optional[int]): ID of the parent peak if this is part of an overlapping group.
        role (Literal): Role in overlapping group ('individual', 'child', or 'parent').
        scale_class (Optional[str]): Prominence scale class ('minor', 'major', 'super').
        in_cluster (bool): Whether the peak has been assigned to a cluster.
        cluster_id (Optional[int]): ID of the assigned cluster, if any.
        is_analyzed (bool): Whether the peak has been analyzed.
        in_event (Literal): Type of event this peak is part of ('global' or 'sequential').
        origin_type (Literal): Type of cause for this peak ('origin', 'caused', or 'individual').
        origin_label (Optional[int]): Label of the origin peak if this is a caused peak.
    """
    def __init__(
        self,
        id: int,
        rel_start_time: int,
        peak_time: int,
        start_time: int,
        end_time: int,
        rel_end_time: int,
        height: float,
        prominence: float,
        group_id: Optional[int] = None,
        parent_peak_id: Optional[int] = None,
        role: Literal["individual", "child", "parent"] = "individual",
        rel_height: Optional[float] = None
    ):
        self.id = id
        self.peak_time = peak_time
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
        self.rel_height = rel_height
        self.rel_start_time = rel_start_time
        self.rel_end_time = rel_end_time
        self.rel_duration = rel_end_time - rel_start_time
        self.height = height
        self.prominence = prominence

        self.rise_time = peak_time - start_time
        self.decay_time = end_time - peak_time
        self.rel_rise_time = peak_time - rel_start_time
        self.rel_decay_time = rel_end_time - peak_time
        self.rel_symmetry_score: float = self._compute_symmetry_score()

        self.group_id = group_id
        self.parent_peak_id = parent_peak_id
        self.role = role

        self.scale_class: Optional[str] = None  # e.g., 'minor', 'major', 'super'

        self.in_cluster: bool = False  # Set True once this peak is added to a cluster
        self.cluster_id: Optional[int] = None  # ID of the assigned cluster

        self.is_analyzed: bool = False  # Flag to track if this peak has been analyzed
        self.in_event: Literal["global", "sequential"] = None  # Type of event this peak is part of, if any
        self.event_id: Optional[int] = None  # ID of the event this peak is part of, if any

        self.origin_type: Literal["origin", "caused", "individual"] = "individual" # Type of cause for this peak
        self.origin_label: Optional[int] = None # Label of the origin peak if this is a caused peak



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
            f"Peak(id={self.id}, time={self.rel_start_time}, height={self.height:.2f}, in_event={self.in_event})"
        )


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
        self.method = config.method
        self.detection_params = config.params
        self.grouping_params = config.peak_grouping


    def run(self, trace: list[float]) -> list[Peak]:
        """
        Execute peak detection on the provided trace.
        
        Args:
            trace (list[float]): Calcium intensity trace.
        
        Returns:
            list[Peak]: List of detected peaks.
        """
        peaks = self._detect(trace)
        peaks = self._refine_peak_durations(peaks, np.array(trace, dtype=float))
        peaks = self._group_overlapping_peaks(peaks)

        if self.config.filter_overlapping_peaks:
            peaks = self._filter_children_peaks(peaks)
            peaks = reassign_peak_ids(peaks)

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
            _, _, rel_left_ips, rel_right_ips = peak_widths(subtrace, peaks, rel_height=self.detection_params.relative_height)

            # Compute whole widths
            _, _, left_ips, right_ips = peak_widths(subtrace, peaks, rel_height=self.detection_params.full_duration_threshold)

            # Step 2: Classify peaks by scale
            if len(prominences) > 0:
                quantiles = np.quantile(prominences, self.detection_params.scale_class_quantiles)
            else:
                quantiles = [0, 0]

            peak_list = []
            for i, peak_time in enumerate(peaks):
                rel_start_time = int(np.floor(rel_left_ips[i])) + start_frame
                rel_end_time = int(np.ceil(rel_right_ips[i])) + start_frame
                start_time = int(np.floor(left_ips[i])) + start_frame
                end_time = int(np.ceil(right_ips[i])) + start_frame
                prominence = float(prominences[i])
                height = float(subtrace[peak_time])

                # Assign scale class
                if prominence < quantiles[0]:
                    scale_class = "minor"
                elif prominence < quantiles[1]:
                    scale_class = "major"
                else:
                    scale_class = "super"

                peak = Peak(
                    id=i,
                    start_time=start_time,
                    end_time=end_time,
                    rel_start_time=rel_start_time,
                    peak_time=peak_time,
                    rel_end_time=rel_end_time,
                    height=height,
                    prominence=prominence,
                    rel_height=self.detection_params.relative_height
                )
                peak.scale_class = scale_class
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
        for peak in peaks:
            start, end = find_valley_bounds(trace, peak.rel_start_time, peak.rel_end_time)
            peak.start_time = max(start, peak.start_time)
            peak.end_time = min(end, peak.end_time)
            peak.duration = peak.end_time - peak.start_time

        return peaks

    def _group_overlapping_peaks(self, peaks: List[Peak]) -> List[Peak]:
        """
        Identify parent-child peak relationships based on containment and height.

        A peak is a parent of another if:
            - The child peak time is within the parent's start and end
            - The parent height is strictly greater than the child's height

        Once a peak is labeled as a child, it will never be processed again.

        Returns:
            List[Peak]: Peaks with group_id, parent_peak_id, and role assigned.
        """
        if not peaks:
            return []

        verbose = self.grouping_params.verbose
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

                if parent.start_time <= child.peak_time <= parent.end_time and parent.height > child.height:
                    # Valid child
                    child.group_id = group_id_counter
                    child.parent_peak_id = parent.id
                    child.role = "member"
                    seen_as_child.add(child.id)
                    current_group.append(child)

            if current_group:
                # Mark parent
                parent.group_id = group_id_counter
                parent.parent_peak_id = None
                parent.role = "parent"
                grouped_peaks.append(parent)
                grouped_peaks.extend(current_group)
                group_id_counter += 1
            else:
                # No children found → individual
                parent.group_id = None
                parent.parent_peak_id = None
                parent.role = "individual"
                grouped_peaks.append(parent)

        if verbose:
            total_groups = len(set(p.group_id for p in grouped_peaks if p.group_id is not None))
            logger.info(f"[PeakDetector] Formed {total_groups} containment-based groups.")
            for gid in set(p.group_id for p in grouped_peaks if p.group_id is not None):
                size = sum(p.group_id == gid for p in grouped_peaks)
                logger.info(f" - Group {gid}: {size} peaks")
            num_individuals = sum(p.role == "individual" for p in grouped_peaks)
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
        filtered = [p for p in peaks if p.role == "individual" or p.role == "parent"]
        n_removed = len(peaks) - len(filtered)
        if n_removed > 0:
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