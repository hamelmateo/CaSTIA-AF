from typing import Optional, Literal, List
import numpy as np
from scipy.signal import find_peaks, peak_widths
from collections import defaultdict, deque


class Peak:
    """
    Represents a calcium transient peak with timing, intensity, and hierarchical metadata.
    """
    def __init__(
        self,
        id: int,
        start_time: int,
        peak_time: int,
        end_time: int,
        height: float,
        prominence: float,
        group_id: Optional[int] = None,
        parent_peak_id: Optional[int] = None,
        role: Literal["individual", "child", "parent"] = "individual"
    ):
        self.id = id
        self.start_time = start_time
        self.peak_time = peak_time
        self.end_time = end_time
        self.duration = end_time - start_time
        self.height = height
        self.prominence = prominence

        self.rise_time = peak_time - start_time
        self.decay_time = end_time - peak_time

        self.group_id = group_id
        self.parent_peak_id = parent_peak_id
        self.role = role

        self.scale_class: Optional[str] = None  # e.g., 'minor', 'major', 'super'

    def __repr__(self):
        return (
            f"Peak(id={self.id}, time={self.peak_time}, height={self.height:.2f}, "
            f"prominence={self.prominence:.2f}, role={self.role}, scale={self.scale_class})"
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

        if self.method == "skimage":
            peaks, properties = find_peaks(
                trace,
                prominence=self.method_params.get("prominence", 0.1),
                distance=self.method_params.get("distance", 5),
                height=self.method_params.get("height"),
                threshold=self.method_params.get("threshold"),
                width=self.method_params.get("width")
            )

            prominences = properties["prominences"]

            # Compute fwhm
            peak_metadata = peak_widths(trace, peaks, rel_height=self.method_params.get("relative_height", 0.6))
            left_ips = peak_metadata[2]
            right_ips = peak_metadata[3]

            # Step 2: Classify peaks by scale
            if len(prominences) > 0:
                quantiles = np.quantile(prominences, self.params.get("scale_class_quantiles", [0.33, 0.66]))
            else:
                quantiles = [0, 0]

            peak_list = []
            for i, peak_time in enumerate(peaks):
                start_time = int(np.floor(left_ips[i]))
                end_time = int(np.ceil(right_ips[i]))
                prominence = float(prominences[i])
                height = float(trace[peak_time])

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
                    peak_time=peak_time,
                    end_time=end_time,
                    height=height,
                    prominence=prominence
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
        verbose = self.grouping_params.get("verbose", True)

        # Build adjacency list for overlapping peaks
        graph = defaultdict(set)
        for i, p1 in enumerate(peaks):
            for j, p2 in enumerate(peaks):
                if i >= j:
                    continue
                if p1.start_time <= p2.end_time + overlap_margin and p2.start_time <= p1.end_time + overlap_margin:
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
            print(f"[PeakDetector] Formed {group_id_counter} overlapping groups.")
            for gid in range(group_id_counter):
                size = sum(p.group_id == gid for p in grouped_peaks)
                print(f" - Group {gid}: {size} peaks")
            num_individuals = sum(p.role == "individual" for p in grouped_peaks)
            print(f"[PeakDetector] Found {num_individuals} individual (non-overlapping) peaks.")

        return grouped_peaks