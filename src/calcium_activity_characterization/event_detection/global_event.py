# TODO: there are still some detection problem even so i dont know it does come from. needs to be checked and solved asap.
# global_event_detection.py
# Usage Example:
# >>> windows = find_significant_activity_peaks(trace, len(cells), threshold_ratio=0.4)
# >>> global_blocks = extract_global_event_blocks(cells, windows, radius=300, global_max_comm_time=10, min_cell_count=5)

from calcium_activity_characterization.logger import logger
import numpy as np
from calcium_activity_characterization.data.cells import Cell



def find_significant_activity_peaks(
    trace,
    total_cells: int,
    threshold_ratio: float = 0.4
) -> list[tuple[int, int]]:
    """
    Identify time windows from population activity peaks that exceed a threshold percentage of active cells.

    Args:
        trace (Trace): The population activity trace object.
        total_cells (int): Total number of cells in the population.
        threshold_ratio (float): Ratio of active cells at peak to trigger detection.

    Returns:
        list[tuple[int, int]]: list of (start_frame, end_frame) windows to analyze as global events.
    """
    try:
        peak_windows = []
        for peak in trace.peaks:
            if peak.height >= threshold_ratio * total_cells:
                peak_windows.append((peak.start_time, peak.end_time))
                peak.is_global_event = True
        return peak_windows
    except Exception as e:
        logger.error(f"Error detecting significant peaks: {e}")
        return []

def _get_framewise_peaking_labels(
    cells: list[Cell],
    start: int,
    end: int
) -> dict[int, list[tuple[int, int]]]:
    """
    Build a dict mapping frame -> list of (cell label, peak id) tuples for all cells that have a peak in the given time window.
    This is used to determine which cells are peaking at each frame in the specified time window.

    Args:
        cells (list[Cell]): All cells to consider.
        start (int): Start frame of time window.
        end (int): End frame of time window.

    Returns:
        dict[int, list[tuple[int, int]]]: Mapping from frame to list of (cell label, peak id) tuples peaking at this frame.
    """
    framewise: dict[int, list[tuple[int, int]]] = {}
    for cell in cells:
        valid_peaks = [p for p in cell.trace.peaks if start <= p.communication_time <= end]
        if not valid_peaks:
            continue
        keep_only_latest_peak = False  # TODO think about how to handle this
        if keep_only_latest_peak:
            best_peak = max(valid_peaks, key=lambda p: p.communication_time)
            framewise.setdefault(best_peak.communication_time, []).append((cell.label, best_peak.id))
        else: # Accept all peaks in the window
            for peak in valid_peaks:
                framewise.setdefault(peak.communication_time, []).append((cell.label, peak.id))
        framewise_sorted = dict(sorted(framewise.items()))
    return framewise_sorted

def _get_activated_cells(
    framewise_active: dict[int, list[int]],
    cells: list[Cell]
) -> tuple[dict[int, Cell], dict[tuple[int, int], int]]:
    """
    Extract active cells and their activation times from framewise active labels.

    Args:
        framewise_active (dict[int, list[tuple[int, int]]]): Framewise active labels.
        cells (list[Cell]): list of all cells in the population.
    
    Returns:
        tuple[dict[int, Cell], dict[tuple[int, int], int]]:
            - active_cells: Mapping of cell labels to Cell objects that were active.
            - activation_times: Mapping of (cell label, peak id) to activation time.
    """
    label_to_cell = {cell.label: cell for cell in cells}

    active_cells: dict[int, Cell] = {}
    activation_times: dict[tuple[int, int], int] = {}

    for t, entries in framewise_active.items():
        for label, peak_id in entries:
            if label in label_to_cell:
                active_cells[label] = label_to_cell[label]
                activation_times[(label, peak_id)] = t

    return active_cells, activation_times

def _cluster_event_from_origin(
    label: int,
    peak_id: int,
    active_cells: dict[int, Cell],
    cell_activation_time: dict[tuple[int, int], int],
    radius: float,
    global_max_comm_time: int
) -> set[tuple[int, int]]:
    """
    Cluster cells based on spatial proximity and temporal activation from an origin cell.

    Args:
        label (int): Label of the origin cell.
        peak_id (int): Peak ID of the origin cell.
        active_cells (dict[int, Cell]): Mapping of active cell labels to Cell objects.
        cell_activation_time (dict[int, int]): Mapping of cell labels to their activation times.
        radius (float): Distance threshold to consider spatial spreading.
        global_max_comm_time (int): Maximum number of frames allowed between activations to consider propagation.

    Returns:
        set[tuple[int, int]]: set of (cell label, peak id) tuples in the cluster.
    """
    origin_cell = active_cells[label]
    origin_peak = origin_cell.trace.peaks[peak_id]
    origin_time = origin_peak.communication_time
    
    # Only proceed if this origin causes any other peak
    has_valid_neighbor = False
    for (other_label, other_peak_id), other_time in cell_activation_time.items():
        if (other_label, other_peak_id) == (label, peak_id):
            continue

        time_diff = other_time - origin_time
        if 0 < time_diff <= global_max_comm_time:
            dist = np.linalg.norm(np.array(origin_cell.centroid) - np.array(active_cells[other_label].centroid))
            if dist <= radius:
                has_valid_neighbor = True
                break

    if not has_valid_neighbor:
        return set()

    cluster = {(label, peak_id)}
    queue = [(label, peak_id)]
    origin_peak.in_event = "global"
    origin_peak.origin_type = "origin"

    while queue:
        current_label, current_peak_id = queue.pop()
        current_time = cell_activation_time[(current_label, current_peak_id)]
        current_cell = active_cells[current_label]

        for (other_label, other_peak_id), other_time in cell_activation_time.items():
            if (other_label, other_peak_id) == (current_label, current_peak_id):
                continue
            
            other_cell = active_cells[other_label]
            other_peak = other_cell.trace.peaks[other_peak_id]

            if other_peak.in_event:
                continue

            if 0 < other_time - current_time <= global_max_comm_time:
                dist = np.linalg.norm(np.array(current_cell.centroid) - np.array(other_cell.centroid))
                if dist <= radius:
                    cluster.add((other_label, other_peak_id))
                    other_peak.in_event = "global"
                    other_peak.origin_type = "caused"
                    queue.append((other_label, other_peak_id))

    return cluster

def extract_global_event_blocks(
    cells: list[Cell],
    peak_windows: list[tuple[int, int]],
    radius: float,
    global_max_comm_time: int,
    min_cell_count: int = 3
) -> list[dict[int, list[tuple[int, int]]]]:
    """
    From each peak window, extract spatially propagating global events by radius/time criteria.

    Args:
        cells (list[Cell]): list of all cells in the population.
        peak_windows (list[tuple[int, int]]): Time windows to analyze.
        radius (float): Distance threshold to consider spatial spreading.
        global_max_comm_time (int): Maximum number of frames allowed between activations to consider propagation.
        min_cell_count (int): Minimum size to keep the global event.

    Returns:
        list[dict[int, list[tuple[int, int]]]]: list of framewise (cell label, peak id) dicts (one per GlobalEvent).
    """
    global_event_blocks = []

    try:
        for start, end in peak_windows:
            framewise_active = _get_framewise_peaking_labels(cells, start, end)
            active_cells, activation_times = _get_activated_cells(framewise_active, cells)

            event_cluster = set()

            for (label, peak_id), _ in activation_times.items():
                if active_cells[label].trace.peaks[peak_id].in_event:
                    continue

                cluster = _cluster_event_from_origin(
                    label, peak_id, active_cells, activation_times, radius, global_max_comm_time)

                event_cluster.update(cluster)

            # Build a dict: frame -> list of (cell label, peak id) in event_cluster for that frame
            if len({lbl for lbl, _ in event_cluster}) >= min_cell_count:
                framewise = {
                    t: [(lbl, pid) for (lbl, pid) in framewise_active.get(t, []) if (lbl, pid) in event_cluster]
                    for t in range(start, end + 1)
                }
                global_event_blocks.append(framewise)

        return global_event_blocks

    except Exception as e:
        logger.error(f"Error extracting global event blocks: {e}")
        return []
