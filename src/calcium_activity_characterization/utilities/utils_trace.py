"""
Utility functions for working with Trace objects.

Example usage:
    from calcium_activity_characterization.utilities.utils_trace import compute_global_trace

    global_trace = compute_global_trace(active_cells, version="smoothed")
    global_trace.detect_peaks(detector)
    global_trace.binarize_trace_from_peaks()
    print(global_trace.metadata)
"""

import numpy as np
from calcium_activity_characterization.data.traces import Trace
from calcium_activity_characterization.data.cells import Cell

def compute_global_trace(cells: list[Cell], version: str = "smoothed", default_version: str = "smoothed") -> Trace:
    """
    Compute the mean trace across all active cells based on the specified version.

    Args:
        cells (list[Cell]): List of Cell instances.
        version (str): The key in trace.versions to average.
        default_version (str): The version to set as default in the returned Trace.

    Returns:
        Trace: A new Trace object containing the averaged trace.

    Raises:
        ValueError: If no cells contain the specified version.
    """
    valid_traces = [
        c.trace.versions[version]
        for c in cells
        if version in c.trace.versions and len(c.trace.versions[version]) > 0
    ]

    if not valid_traces:
        raise ValueError(f"No valid cells found with trace version '{version}'")

    global_array = np.mean(valid_traces, axis=0)

    global_trace = Trace()
    global_trace.versions[default_version] = global_array.tolist()
    global_trace.default_version = default_version

    return global_trace
