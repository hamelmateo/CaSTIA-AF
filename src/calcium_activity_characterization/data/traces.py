"""
Example usage:
    >>> t = Trace(raw_trace=[1, 2, 3, 2, 1, 0, 1, 3, 4, 2])
    >>> t.versions["smoothed"] = some_smoothed_version
    >>> t.default_version = "smoothed"
    >>> t.detect_peaks(detector)
    >>> t.binarize_trace_from_peaks()
    >>> print(t.metadata)
"""

import numpy as np
from typing import List, Dict, TYPE_CHECKING


if TYPE_CHECKING:
    from calcium_activity_characterization.data.peaks import Peak

class Trace:
    """
    A container for calcium activity trace and analysis results.

    This class manages multiple versions of the trace (e.g., raw, smoothed, normalized),
    and supports peak detection, binarization, and metadata computation based on a selected version.

    Attributes:
        raw (List[float]): Raw intensity trace.
        versions (Dict[str, List[float]]): Dictionary of named preprocessed trace versions.
        default_version (str): Key name for the version to operate on.
        binary (List[int]): Binarized trace (0/1 based on peak regions).
        peaks (List): List of detected peaks.
        metadata (Dict[str, float]): Computed features from binary + peak data.
    """

    def __init__(self, raw_trace: List[float] = None):
        self.raw: List[float] = raw_trace if raw_trace is not None else []
        self.versions: Dict[str, List[float]] = {}
        self.default_version: str = "smoothed"  # default key to operate on
        self.binary: List[int] = []
        self.peaks: List[Peak] = []
        self.metadata: Dict[str, float] = {}

    @property
    def active_trace(self) -> List[float]:
        """
        Return the currently selected version of the trace for analysis.

        This property behaves like an attribute, allowing you to call `trace.active_trace`
        instead of `trace.get_active_trace()`. It uses the value of `default_version`
        to return the corresponding preprocessed trace.
        """
        return self.versions.get(self.default_version, [])

    def detect_peaks(self, detector) -> None:
        """
        Detect peaks in the active trace version using the provided detector.

        Args:
            detector: An instance of a peak detector with a `.run(trace)` method.
        """
        trace = self.active_trace
        if not trace:
            self.peaks = []
        else:
            self.peaks = detector.run(trace)

    def binarize_trace_from_peaks(self) -> None:
        """
        Generate binary trace (1 during peak intervals, 0 elsewhere) based on peak list.
        Automatically triggers metadata computation.
        """
        trace = self.active_trace
        if not trace:
            self.binary = []
            return

        trace_length = len(trace)
        binary = np.zeros(trace_length, dtype=int)
        for peak in self.peaks:
            start = max(0, peak.start_time)
            end = min(trace_length, peak.end_time + 1)
            binary[start:end] = 1

        self.binary = binary.tolist()
        self.compute_metadata()

    def compute_metadata(self) -> None:
        """
        Compute core activity features from the binary trace and peak list.

        Stored in self.metadata as:
            - fraction_active_time
            - num_peaks
            - burst_frequency
            - mean/std of peak duration
            - mean/std of inter-peak intervals
            - periodicity_score (1 / (1 + CV of inter-peak intervals))
        """
        n = len(self.binary)
        num_peaks = len(self.peaks)

        fraction_active_time = sum(self.binary) / n if n else 0.0
        burst_frequency = num_peaks / n if n else 0.0

        durations = [p.duration for p in self.peaks]
        mean_dur = np.mean(durations) if durations else None
        std_dur = np.std(durations) if durations else None

        start_times = sorted(p.start_time for p in self.peaks)
        intervals = np.diff(start_times)
        mean_ipi = np.mean(intervals) if len(intervals) >= 1 else None
        std_ipi = np.std(intervals) if len(intervals) >= 1 else None

        if len(intervals) >= 2 and np.mean(intervals) > 0:
            cv_ipi = np.std(intervals) / np.mean(intervals)
            periodicity_score = 1 / (1 + cv_ipi)
        else:
            periodicity_score = None

        self.metadata = {
            "fraction_active_time": fraction_active_time,
            "num_peaks": num_peaks,
            "burst_frequency": burst_frequency,
            "mean_peak_duration": mean_dur,
            "std_peak_duration": std_dur,
            "mean_inter_peak_interval": mean_ipi,
            "std_inter_peak_interval": std_ipi,
            "periodicity_score": periodicity_score
        }

    def __repr__(self) -> str:
        """
        Return a summary string for debugging and display.
        """
        return f"<Trace default='{self.default_version}', peaks={len(self.peaks)}, active={self.metadata.get('fraction_active_time', 0):.2f}>"
