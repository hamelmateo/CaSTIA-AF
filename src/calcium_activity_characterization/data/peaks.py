"""
Peak class update to include symmetry score at the individual level.

Example:
    >>> peak = Peak(...)
    >>> print(peak.rel_symmetry_score)  # Already computed during creation
"""

from typing import Optional, Literal, List
import numpy as np
from scipy.signal import find_peaks, peak_widths
from collections import defaultdict, deque
import logging

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
    def __init__(self, params: dict):
        """
        Initialize the PeakDetector with parameters.
        
        Args:
            params (dict): Parameters for peak detection.
        """
        self.params = params
        self.method = params.get("method", "skimage")
        self.method_params = params.get("params", {}).get(self.method, {})
        self.grouping_params = params.get("peak_grouping", {})


    def run(self, trace: list[float]) -> list[Peak]:
        """
        Execute peak detection on the provided trace.
        
        Args:
            trace (list[float]): Calcium intensity trace.
        
        Returns:
            list[Peak]: List of detected peaks.
        """
        peaks = self._detect(trace)
        peaks = self._group_overlapping_peaks(peaks)

        if self.params.get("filter_overlapping_peaks", False):
            peaks = self._filter_non_parent_peaks(peaks)
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

        start_frame = self.method_params.get("start_frame") or 0
        end_frame = self.method_params.get("end_frame") or len(trace)

        subtrace = trace[start_frame:end_frame]

        if self.method == "skimage":



            peaks, properties = find_peaks(
                subtrace,
                prominence=self.method_params.get("prominence", 0.1),
                distance=self.method_params.get("distance", 5),
                height=self.method_params.get("height"),
                threshold=self.method_params.get("threshold"),
                width=self.method_params.get("width")
            )

            prominences = properties["prominences"]

            # Compute relative heights parameters
            rel_peak_metadata = peak_widths(subtrace, peaks, rel_height=self.method_params.get("relative_height", 0.6))
            rel_left_ips = rel_peak_metadata[2]
            rel_right_ips = rel_peak_metadata[3]

            # Compute whole widths
            peak_metadata = peak_widths(subtrace, peaks, rel_height=self.method_params.get("full_duration_threshold", 0.95))
            left_ips = peak_metadata[2]
            right_ips = peak_metadata[3]

            # Step 2: Classify peaks by scale
            if len(prominences) > 0:
                quantiles = np.quantile(prominences, self.params.get("scale_class_quantiles", [0.33, 0.66]))
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
                    rel_height=self.method_params.get("relative_height")
                )
                peak.scale_class = scale_class
                peak_list.append(peak)

            return peak_list
        else:
            raise NotImplementedError(f"Peak detection method '{self.method}' is not supported yet.")


    def _group_overlapping_peaks(self, peaks: List[Peak]) -> List[Peak]:
        """
        Group peaks into overlapping components using transitive overlap logic.

        Args:
            peaks (List[Peak]): List of Peak objects.

        Returns:
            List[Peak]: Peaks with group and role metadata updated.
        """
        if not peaks:
            return []

        overlap_margin = self.grouping_params.get("overlap_margin", 0)
        verbose = self.grouping_params.get("verbose", False)

        # Build adjacency list for overlapping peaks
        graph = defaultdict(set)
        for i, p1 in enumerate(peaks):
            for j, p2 in enumerate(peaks):
                if i >= j:
                    continue
                if p1.rel_start_time <= p2.rel_end_time + overlap_margin and p2.rel_start_time <= p1.rel_end_time + overlap_margin:
                    graph[i].add(j)
                    graph[j].add(i)

        # Find connected components
        visited = set()
        groups = []

        for i in range(len(peaks)):
            if i in visited:
                continue
            queue = deque([i])
            component = []
            while queue:
                idx = queue.popleft()
                if idx in visited:
                    continue
                visited.add(idx)
                component.append(peaks[idx])
                queue.extend(graph[idx])
            groups.append(component)

        # Assign roles and IDs
        grouped_peaks = []
        group_id_counter = 0
        for group in groups:
            if len(group) == 1:
                peak = group[0]
                peak.role = "individual"
                peak.group_id = None
                peak.parent_peak_id = None
                grouped_peaks.append(peak)
            else:
                group_id = group_id_counter
                parent_peak = max(group, key=lambda p: p.prominence)
                parent_peak.role = "parent"
                parent_peak.group_id = group_id
                parent_peak.parent_peak_id = None
                grouped_peaks.append(parent_peak)

                for peak in group:
                    if peak is parent_peak:
                        continue
                    peak.role = "member"
                    peak.group_id = group_id
                    peak.parent_peak_id = parent_peak.id
                    grouped_peaks.append(peak)

                group_id_counter += 1

        if verbose:
            logger.info(f"[PeakDetector] Formed {group_id_counter} overlapping groups.")
            for gid in range(group_id_counter):
                size = sum(p.group_id == gid for p in grouped_peaks)
                logger.info(f" - Group {gid}: {size} peaks")
            num_individuals = sum(p.role == "individual" for p in grouped_peaks)
            logger.info(f"[PeakDetector] Found {num_individuals} individual (non-overlapping) peaks.")

        return grouped_peaks
    

    def _filter_non_parent_peaks(self, peaks: List[Peak]) -> List[Peak]:
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