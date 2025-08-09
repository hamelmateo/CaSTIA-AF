# TODO: Refactor this file by putting out the functions in a separate file just like for global events 
# (potentially add config file parameters)
# cell_to_cell_communication.py
# Usage Example:
# >>> comms = generate_cell_to_cell_communications(cells, graph, copeaking_groups, max_time_gap=10)
# >>> for c in comms: print(c)

import networkx as nx
from scipy.spatial.distance import euclidean

from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.data.copeaking_neighbors import CoPeakingNeighbors
from calcium_activity_characterization.config.presets import EventExtractionConfig

from calcium_activity_characterization.logger import logger


class CellToCellCommunication:
    """
    Represents a causal link between two peaks in neighboring cells.

    Attributes:
        origin (tuple[int, int]): tuple (cell_label, peak.id) for the origin peak.
        cause (tuple[int, int]): tuple (cell_label, peak.id) for the peak that was caused.
        origin_start_time (int): Start time of the origin peak.
        cause_start_time (int): Start time of the caused peak.
    """

    def __init__(
        self,
        origin: tuple[int, int],
        cause: tuple[int, int],
        origin_start_time: int,
        cause_start_time: int,
        origin_centroid: tuple[int, int],
        cause_centroid: tuple[int, int]
    ) -> None:
        self.id: int = id(self)  # Unique identifier for the communication
        self.origin: tuple[int, int] = origin
        self.cause: tuple[int, int] = cause
        self.origin_start_time: int = origin_start_time
        self.cause_start_time: int = cause_start_time

        assert self.cause_start_time >= self.origin_start_time, (
            f"Cause peak must start after origin peak. "
            f"Got origin_start_time={self.origin_start_time}, cause_start_time={self.cause_start_time}"
        )

        self.origin_centroid: tuple[int, int] = origin_centroid
        self.cause_centroid: tuple[int, int] = cause_centroid

        self.duration: int = self.cause_start_time - self.origin_start_time
        self.distance: float = euclidean(origin_centroid, cause_centroid)
        self.speed: float = self.compute_speed()

        self.event_time_phase: float = None  # Placeholder for event time phase, if needed
        self.event_recruitment_phase: float = None  # Placeholder for recruitment phase, if needed

    def compute_speed(self) -> float:
        """Compute speed of communication in units per frame."""
        if self.duration > 0:
            return self.distance / self.duration
        elif self.duration == 0:
            return self.distance / 1.0  # Avoid division by zero, treat as instantaneous
        else:
            raise ValueError("Duration cannot be negative.")

    @property
    def delta_t(self) -> int:
        """Time difference between origin and cause peaks."""
        return self.cause_start_time - self.origin_start_time

    def __repr__(self) -> str:
        return (
            f"<CellToCellCommunication origin={self.origin} cause={self.cause} "
            f"dt={self.delta_t}>"
        )


# ===============================
# COMMUNICATION RESOLUTION LOGIC
# ===============================

def generate_cell_to_cell_communications(
    cells: list[Cell],
    neighbor_graph: nx.Graph,
    copeaking_groups: list[CoPeakingNeighbors],
    max_time_gap: int = 10
) -> list[CellToCellCommunication]:
    """
    Build all causal CellToCellCommunication links from co-peaking groups and independent peaks.

    Args:
        cells (list[Cell]): list of cell objects.
        neighbor_graph (nx.Graph): Spatial graph.
        copeaking_groups (list[CoPeakingNeighbors]): Precomputed co-peaking groups.
        max_time_gap (int): Max delay (in frames) to consider a causal link.

    Returns:
        list[CellToCellCommunication]: All causal communications.
    """
    label_to_cell: dict[int, Cell] = {cell.label: cell for cell in cells}
    communications: list[CellToCellCommunication] = []


    for group in copeaking_groups:
        comms = _resolve_copeaking_group(group, label_to_cell, neighbor_graph, max_time_gap)
        communications.extend(comms)

    comms = _resolve_individual_peaks(cells, neighbor_graph, label_to_cell, max_time_gap)
    communications.extend(comms)
    

    logger.info(f"Generated {len(communications)} cell-to-cell communications.")

    return communications


def _resolve_copeaking_group(
    group: CoPeakingNeighbors,
    label_to_cell: dict[int, Cell],
    neighbor_graph: nx.Graph,
    max_time_gap: int
) -> list[CellToCellCommunication]:
    """Resolve one CoPeaking group, creating communication links and tracking classified peaks."""
    communications: list[CellToCellCommunication] = []
    external_origins = []

    # Check for external origins in the neighbor graph
    for label, cell in label_to_cell.items():
        if label in group.labels: # Skip cells already in the group
            continue
        for _, peak in enumerate(cell.trace.peaks):
            if not peak.in_event:
                if 0 < group.frame - peak.communication_time <= max_time_gap:
                    if any(n in group.labels for n in neighbor_graph.neighbors(label)):
                        external_origins.append((label, peak.id, peak.communication_time))

    # Sort external origins by time (decreasing) and label
    if external_origins:
        external_origins.sort(key=lambda x: (-x[2], x[0]))
        origin_label, origin_id, origin_time = external_origins[0]

        # Create communications from external origin to direct neighbors copeaking cells
        direct_targets = set()
        for target_label, target_id in group.members:
            if target_label in neighbor_graph[origin_label]: 
                cause_time = label_to_cell[target_label].trace.peaks[target_id].communication_time
                communications.append(CellToCellCommunication(
                    origin=(origin_label, origin_id),
                    cause=(target_label, target_id),
                    origin_start_time=origin_time,
                    cause_start_time=cause_time,
                    origin_centroid=label_to_cell[origin_label].centroid,
                    cause_centroid=label_to_cell[target_label].centroid
                ))
                label_to_cell[target_label].trace.peaks[target_id].is_analyzed = True
                direct_targets.add((target_label, target_id))

        # Spatially propagate communications within the group from the external origin using BFS
        communications.extend(_bfs_propagate_within_group(group, label_to_cell, direct_targets))

    else:
        # If no external origins, use the earliest peak in the group as the origin
        sorted_members = sorted(group.members, key=lambda p: label_to_cell[p[0]].trace.peaks[p[1]].start_time)
        origin_label, origin_id = sorted_members[0]

        origin_peak = label_to_cell[origin_label].trace.peaks[origin_id]

        origin_peak.is_analyzed = True
        communications.extend(
            _bfs_propagate_within_group(group, label_to_cell, {(origin_label, origin_id)})
        )

    return communications


def _bfs_propagate_within_group(
    group: CoPeakingNeighbors,
    label_to_cell: dict[int, Cell],
    start_labels: set[tuple[int, int]]
) -> list[CellToCellCommunication]:
    """
    Propagate communication links within a group using BFS from one or more starting nodes.

    Args:
        group (CoPeakingNeighbors): Co-peaking group.
        label_to_cell (dict[int, Cell]): Mapping from label to Cell.
        start_labels (set[tuple[int, int]]): Starting labels for BFS propagation.

    Returns:
        list[CellToCellCommunication]: Communication edges inside the group.
    """
    communications: list[CellToCellCommunication] = []
    start_cells = [label for label, _ in start_labels]
    visited = set(start_cells)
    queue = list(start_cells)

    while queue:
        # For each cell in the queue, find its copeaking neighbors
        current = queue.pop(0)
        for neighbor in group.subgraph.neighbors(current):
            for (cand_label, cand_id) in group.members: # For-loop over neighbor's peaks
                if cand_label == neighbor and not label_to_cell[cand_label].trace.peaks[cand_id].is_analyzed:
                    current_peak_id = next(i for l, i in start_labels if l == current) # Get the index of the peak in the current cell
                    comm = CellToCellCommunication(
                        origin=(current, current_peak_id),  
                        cause=(cand_label, cand_id),
                        origin_start_time=label_to_cell[current].trace.peaks[current_peak_id].communication_time,
                        cause_start_time=label_to_cell[cand_label].trace.peaks[cand_id].communication_time,
                        origin_centroid=label_to_cell[current].centroid,
                        cause_centroid=label_to_cell[cand_label].centroid
                    )
                    communications.append(comm)
                    label_to_cell[cand_label].trace.peaks[cand_id].is_analyzed = True
                    visited.add(cand_label) 
                    queue.append(cand_label) # Add neighbor to queue for further spatial propagation
                    start_labels.add((cand_label, cand_id)) # Add to start labels for next iterations

    return communications



def _resolve_individual_peaks(
    cells: list[Cell],
    neighbor_graph: nx.Graph,
    label_to_cell: dict[int, Cell],
    max_time_gap: int
) -> list[CellToCellCommunication]:
    """Handle communication creation for peaks not part of any group."""
    communications: list[CellToCellCommunication] = []

    for cell in cells:
        for peak in cell.trace.peaks:
            if peak.is_analyzed or peak.in_event:
                continue

            best_candidate = None
            best_dt = float("inf")
            for neighbor in neighbor_graph.neighbors(cell.label):
                for neighbor_peak in label_to_cell[neighbor].trace.peaks:
                    if neighbor_peak.in_event:
                        continue
                    dt = peak.communication_time - neighbor_peak.communication_time
                    if 0 < dt <= max_time_gap and dt < best_dt:
                        best_candidate = (neighbor, neighbor_peak.id, neighbor_peak.communication_time)
                        best_dt = dt

            if best_candidate:
                origin_label, origin_id, origin_time = best_candidate
                comm = CellToCellCommunication(
                    origin=(origin_label, origin_id),
                    cause=(cell.label, peak.id),
                    origin_start_time=origin_time,
                    cause_start_time=peak.communication_time,
                    origin_centroid=label_to_cell[origin_label].centroid,
                    cause_centroid=cell.centroid
                )
                communications.append(comm)
                peak.is_analyzed = True
                
    return communications


def assign_peak_classifications(
    cells: list[Cell],
    communications: list[CellToCellCommunication]
) -> None:
    """
    Assign origin_type and origin_label to each peak based on communication links.

    - If a peak appears at least once as `cause` → "caused"
    - Else if it appears at least once as `origin` and never as `caused` → "origin"
    - Else → "individual"

    Args:
        cells (list[Cell]): list of cell objects containing peaks.
        communications (list[CellToCellCommunication]): Communication links.
    """
    caused_set = {comm.cause for comm in communications}
    origin_set = {comm.origin for comm in communications}

    for cell in cells:
        for _, peak in enumerate(cell.trace.peaks):
            if peak.in_event != "global":
                peak_id = (cell.label, peak.id)

                if peak_id in caused_set:
                    peak.origin_type = "caused"
                    peak.in_event = "sequential"
                elif peak_id in origin_set and peak_id not in caused_set:
                    peak.origin_type = "origin"
                    peak.in_event = "sequential"
                else:
                    peak.in_event = "individual"
                    peak.origin_type = "individual"