# global_event_detection.py
# Usage Example:
# >>> windows = find_significant_activity_peaks(trace, len(cells), threshold_ratio=0.4)
# >>> global_blocks = extract_global_event_blocks(cells, windows, radius=30.0, max_frame_gap=10, min_cell_count=5)

from typing import List, Tuple, Dict, Set
import logging
import numpy as np
from calcium_activity_characterization.data.cells import Cell

logger = logging.getLogger(__name__)

def find_significant_activity_peaks(
    trace,
    total_cells: int,
    threshold_ratio: float = 0.4
) -> List[Tuple[int, int]]:
    """
    Identify time windows from population activity peaks that exceed a threshold percentage of active cells.

    Args:
        trace (Trace): The population activity trace object.
        total_cells (int): Total number of cells in the population.
        threshold_ratio (float): Ratio of active cells at peak to trigger detection.

    Returns:
        List[Tuple[int, int]]: List of (start_frame, end_frame) windows to analyze as global events.
    """
    try:
        peak_windows = []
        for peak in trace.peaks:
            if peak.height >= threshold_ratio * total_cells:
                peak_windows.append((peak.start_time, peak.end_time))
        return peak_windows
    except Exception as e:
        logger.error(f"Error detecting significant peaks: {e}")
        return []

def _get_framewise_active_labels(
    cells: List[Cell],
    start: int,
    end: int
) -> Dict[int, List[int]]:
    """
    Build a dict mapping frame -> list of cell labels whose peak starts at that frame.

    Only includes rel_start_time, not the full duration.

    Args:
        cells (List[Cell]): All cells to consider.
        start (int): Start frame of time window.
        end (int): End frame of time window.

    Returns:
        Dict[int, List[int]]: Frame → list of cell labels whose peak starts at that frame.
    """
    framewise: Dict[int, List[int]] = {}
    for cell in cells:
        for peak in cell.trace.peaks:
            t = peak.rel_start_time
            if start <= t <= end:
                framewise.setdefault(t, []).append(cell.label)
    return framewise


def _get_activated_cells(
    framewise_active: Dict[int, List[int]],
    cells: List[Cell]
) -> Tuple[Dict[int, Cell], Dict[int, int]]:
    """
    Extract active cells and their activation times from framewise active labels.

    Args:
        framewise_active (Dict[int, List[int]]): Framewise active labels.
        cells (List[Cell]): List of all cells in the population.
    
    Returns:
        Tuple[Dict[int, Cell], Dict[int, int]]:
            - active_cells: Mapping of label to Cell object for active cells.
            - cell_activation_time: Mapping of label to activation time.
    """
    label_to_cell = {cell.label: cell for cell in cells}
    active_cells = {
        label: label_to_cell[label]
        for t in framewise_active
        for label in framewise_active[t]
        if label in label_to_cell
    }
    cell_activation_time = {
        label: t for t in framewise_active for label in framewise_active[t]
    }
    return active_cells, cell_activation_time

def _cluster_event_from_origin(
    label: int,
    origin_time: int,
    active_cells: Dict[int, Cell],
    cell_activation_time: Dict[int, int],
    radius: float,
    max_frame_gap: int
) -> Set[int]:
    """
    Cluster cells based on spatial proximity and temporal activation from an origin cell.

    Args:
        label (int): Label of the origin cell.
        origin_time (int): Activation time of the origin cell.
        active_cells (Dict[int, Cell]): Mapping of active cell labels to Cell objects.
        cell_activation_time (Dict[int, int]): Mapping of cell labels to their activation times.
        radius (float): Distance threshold to consider spatial spreading.
        max_frame_gap (int): Maximum number of frames allowed between activations to consider propagation.

    Returns:
        Set[int]: Set of labels in the cluster.
    """
    origin_peak = active_cells[label].trace.get_peak_starting_at(origin_time)
    if origin_peak.in_event:
        return set()
    
    # Only proceed if this origin causes any other peak
    has_valid_neighbor = any(
        0 < (cell_activation_time[other] - origin_time) <= max_frame_gap and
        np.linalg.norm(np.array(active_cells[label].centroid) - np.array(active_cells[other].centroid)) <= radius
        for other in active_cells
        if other != label
    )

    if not has_valid_neighbor:
        return set()

    cluster = {label}
    queue = [(label, origin_time)]
    origin_peak.in_event = "global"
    origin_peak.origin_type = "origin"

    while queue:
        current_label, current_time = queue.pop()
        current_cell = active_cells[current_label]

        for other_label, other_cell in active_cells.items():
            other_time = cell_activation_time[other_label]
            other_peak = other_cell.trace.get_peak_starting_at(other_time)

            if other_label in cluster or other_peak.in_event:
                continue

            if 0 < other_time - current_time <= max_frame_gap:
                dist = np.linalg.norm(np.array(current_cell.centroid) - np.array(other_cell.centroid))
                if dist <= radius:
                    cluster.add(other_label)
                    other_peak.in_event = "global"
                    other_peak.origin_type = "caused"
                    queue.append((other_label, other_time))

    return cluster

def extract_global_event_blocks(
    cells: List[Cell],
    peak_windows: List[Tuple[int, int]],
    radius: float,
    max_frame_gap: int,
    min_cell_count: int = 3
) -> List[Dict[int, List[int]]]:
    """
    From each peak window, extract spatially propagating global events by radius/time criteria.

    Args:
        cells (List[Cell]): List of all cells in the population.
        peak_windows (List[Tuple[int, int]]): Time windows to analyze.
        radius (float): Distance threshold to consider spatial spreading.
        max_frame_gap (int): Maximum number of frames allowed between activations to consider propagation.
        min_cell_count (int): Minimum size to keep the global event.

    Returns:
        List[Dict[int, List[int]]]: List of framewise label dicts (one per GlobalEvent).
    """
    global_event_blocks = []

    try:
        for start, end in peak_windows:
            framewise_active = _get_framewise_active_labels(cells, start, end)
            active_cells, activation_times = _get_activated_cells(framewise_active, cells)

            event_cluster = set()

            for label, cell in active_cells.items():
                if cell.trace.get_peak_starting_at(activation_times[label]).in_event:
                    continue

                origin_time = activation_times[label]
                cluster = _cluster_event_from_origin(
                    label, origin_time, active_cells, activation_times, radius, max_frame_gap)

                event_cluster.update(cluster)

            if len(event_cluster) >= min_cell_count:
                framewise = {
                    t: [l for l in framewise_active.get(t, []) if l in event_cluster]
                    for t in range(start, end + 1)
                }
                global_event_blocks.append(framewise)

        return global_event_blocks

    except Exception as e:
        logger.error(f"Error extracting global event blocks: {e}")
        return []
