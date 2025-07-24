# peak_utils.py
# Usage Example:
# >>> from calcium_activity_characterization.utilities.peak_utils import find_valley_bounds
# >>> left, right = find_valley_bounds(trace, fhw_start_time=100, fhw_end_time=120)

import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def find_valley_bounds(
    trace: np.ndarray,
    fhw_start_time: int,
    fhw_end_time: int,
    max_search: int = 350, #TODO: make this a config parameter
    window: int = 25
) -> Tuple[int, int]:
    """
    Find the left and right valley bounds of a peak in a 1D trace.

    This function searches for local minima around the start and end of a peak window,
    and returns the indices where the valleys (local minima) are located.

    Args:
        trace (np.ndarray): 1D array of the signal.
        fhw_start_time (int): Approximate relative start time of the peak.
        fhw_end_time (int): Approximate relative end time of the peak.
        max_search (int, optional): Maximum number of frames to search left and right. Defaults to 350.
        window (int, optional): Number of neighboring frames to consider when detecting a valley. Defaults to 25.

    Returns:
        Tuple[int, int]: Refined (start_time, end_time) indices of the valley bounds.
    """
    trace = np.asarray(trace, dtype=float)
    n = len(trace)
    left_bound = fhw_start_time
    right_bound = fhw_end_time

    try:
        # --- Left side ---
        for i in range(fhw_start_time - 1, max(fhw_start_time - max_search - 1, 0), -1):
            window_vals = trace[max(i - window, 0): min(i + window + 1, n)]
            if np.all(trace[i] <= window_vals):
                left_bound = i
                break

        # --- Right side ---
        for i in range(fhw_end_time + 1, min(fhw_end_time + max_search + 1, n)):
            window_vals = trace[max(i - window, 0): min(i + window + 1, n)]
            if np.all(trace[i] <= window_vals):
                right_bound = i
                break

    except Exception as e:
        logger.error(f"Failed to find valley bounds: {e}")

    return left_bound, right_bound
