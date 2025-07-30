from typing import List, Optional, Tuple
import numpy as np

def analyze_peak_intervals(peak_times: List[int]) -> Tuple[List[int], Optional[float], Optional[float]]:
    """
    Analyze periodicity of global (or trace-level) peak times.

    Computes inter-peak intervals, periodicity score (based on CV),
    and average peak frequency (events per frame).

    Args:
        peak_times (List[int]): Sorted list of peak times (in frames).

    Returns:
        Tuple:
            - intervals (List[int]): List of inter-peak intervals.
            - periodicity_score (Optional[float]): [0, 1] score or None if too few events.
            - average_frequency (Optional[float]): Events per frame, or None if invalid.
    """
    if not peak_times or len(peak_times) < 2:
        return [], None, None

    peak_times = sorted(peak_times)
    intervals = np.diff(peak_times).tolist()

    if len(intervals) < 2:
        return intervals, None, None

    mean_ipi = np.mean(intervals)
    std_ipi = np.std(intervals)
    cv = std_ipi / mean_ipi if mean_ipi > 0 else None
    periodicity_score = 1 / (1 + cv) if cv is not None else None

    total_duration = peak_times[-1] - peak_times[0]
    average_frequency = len(intervals) / total_duration if total_duration > 0 else None

    return intervals, periodicity_score, average_frequency
