# TODO: Refactor this file by putting out the functions in a separate file just like for global events 
# (potentially add config file parameters)
# cell_to_cell_communication.py
# Usage Example:
# >>> comms = generate_cell_to_cell_communications(cells, graph, copeaking_groups, max_time_gap=10)
# >>> for c in comms: print(c)

from typing import Tuple, List, Dict, Set
import networkx as nx
from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.data.copeaking_neighbors import CoPeakingNeighbors

import logging
logger = logging.getLogger(__name__)

class CellToCellCommunication:
    """
    Represents a causal link between two peaks in neighboring cells.

    Attributes:
        origin (Tuple[int, int]): Tuple (cell_label, peak_index) for the origin peak.
        cause (Tuple[int, int]): Tuple (cell_label, peak_index) for the peak that was caused.
        origin_start_time (int): Start time of the origin peak.
        cause_start_time (int): Start time of the caused peak.
    """

    def __init__(
        self,
        origin: Tuple[int, int],
        cause: Tuple[int, int],
        origin_start_time: int,
        cause_start_time: int
    ) -> None:
        self.origin: Tuple[int, int] = origin
        self.cause: Tuple[int, int] = cause
        self.origin_start_time: int = origin_start_time
        self.cause_start_time: int = cause_start_time

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
    cells: List[Cell],
    neighbor_graph: nx.Graph,
    copeaking_groups: List[CoPeakingNeighbors],
    max_time_gap: int = 10
) -> List[CellToCellCommunication]:
    """
    Build all causal CellToCellCommunication links from co-peaking groups and independent peaks.

    Args:
        cells (List[Cell]): List of cell objects.
        neighbor_graph (nx.Graph): Spatial graph.
        copeaking_groups (List[CoPeakingNeighbors]): Precomputed co-peaking groups.
        max_time_gap (int): Max delay (in frames) to consider a causal link.

    Returns:
        List[CellToCellCommunication]: All causal communications.
    """
    label_to_cell: Dict[int, Cell] = {cell.label: cell for cell in cells}
    communications: List[CellToCellCommunication] = []

    for group in copeaking_groups:
        comms = _resolve_copeaking_group(group, label_to_cell, neighbor_graph, max_time_gap)
        communications.extend(comms)

    comms = _resolve_individual_peaks(cells, neighbor_graph, label_to_cell, max_time_gap)
    communications.extend(comms)
    

    logger.info(f"Generated {len(communications)} cell-to-cell communications.")

    return communications


def _resolve_copeaking_group(
    group: CoPeakingNeighbors,
    label_to_cell: Dict[int, Cell],
    neighbor_graph: nx.Graph,
    max_time_gap: int
) -> List[CellToCellCommunication]:
    """Resolve one CoPeaking group, creating communication links and tracking classified peaks."""
    communications: List[CellToCellCommunication] = []
    external_origins = []

    # Check for external origins in the neighbor graph
    for label, cell in label_to_cell.items():
        if label in group.labels: # Skip cells already in the group
            continue
        for _, peak in enumerate(cell.trace.peaks):
            if 0 < group.frame - peak.rel_start_time <= max_time_gap:
                if any(n in group.labels for n in neighbor_graph.neighbors(label)):
                    external_origins.append((label, peak.id, peak.rel_start_time))

    # Sort external origins by time (decreasing) and label
    if external_origins:
        external_origins.sort(key=lambda x: (-x[2], x[0]))
        origin_label, origin_idx, origin_time = external_origins[0]

        # Create communications from external origin to direct neighbors copeaking cells
        direct_targets = set()
        for target_label, target_idx in group.members:
            if target_label in neighbor_graph[origin_label]: 
                cause_time = label_to_cell[target_label].trace.peaks[target_idx].rel_start_time
                communications.append(CellToCellCommunication(
                    origin=(origin_label, origin_idx),
                    cause=(target_label, target_idx),
                    origin_start_time=origin_time,
                    cause_start_time=cause_time
                ))
                label_to_cell[target_label].trace.peaks[target_idx].is_analyzed = True
                direct_targets.add((target_label, target_idx))

        # Spatially propagate communications within the group from the external origin using BFS
        communications.extend(_bfs_propagate_within_group(group, label_to_cell, direct_targets))

    else:
        # If no external origins, use the earliest peak in the group as the origin
        sorted_members = sorted(group.members, key=lambda p: label_to_cell[p[0]].trace.peaks[p[1]].start_time)
        origin_label, origin_idx = sorted_members[0]

        origin_peak = label_to_cell[origin_label].trace.peaks[origin_idx]

        origin_peak.is_analyzed = True
        communications.extend(
            _bfs_propagate_within_group(group, label_to_cell, {(origin_label, origin_idx)})
        )

    return communications


def _bfs_propagate_within_group(
    group: CoPeakingNeighbors,
    label_to_cell: Dict[int, Cell],
    start_labels: Set[Tuple[int, int]]
) -> List[CellToCellCommunication]:
    """
    Propagate communication links within a group using BFS from one or more starting nodes.

    Args:
        group (CoPeakingNeighbors): Co-peaking group.
        label_to_cell (Dict[int, Cell]): Mapping from label to Cell.
        start_labels (Set[int]): Copeaking cells to begin propagation.

    Returns:
        List[CellToCellCommunication]: Communication edges inside the group.
    """
    communications: List[CellToCellCommunication] = []
    start_cells = [label for label, _ in start_labels]
    visited = set(start_cells)
    queue = list(start_cells)

    while queue:
        # For each cell in the queue, find its copeaking neighbors
        current = queue.pop(0)
        for neighbor in group.subgraph.neighbors(current):
            for (cand_label, cand_idx) in group.members: # For-loop over neighbor's peaks
                if cand_label == neighbor and not label_to_cell[cand_label].trace.peaks[cand_idx].is_analyzed:
                    current_peak_index = next(i for l, i in start_labels if l == current) # Get the index of the peak in the current cell
                    comm = CellToCellCommunication(
                        origin=(current, current_peak_index),  
                        cause=(cand_label, cand_idx),
                        origin_start_time=label_to_cell[current].trace.peaks[current_peak_index].rel_start_time,
                        cause_start_time=label_to_cell[cand_label].trace.peaks[cand_idx].rel_start_time
                    )
                    communications.append(comm)
                    label_to_cell[cand_label].trace.peaks[cand_idx].is_analyzed = True
                    visited.add(cand_label) 
                    queue.append(cand_label) # Add neighbor to queue for further spatial propagation
                    start_labels.add((cand_label, cand_idx)) # Add to start labels for next iterations

    return communications



def _resolve_individual_peaks(
    cells: List[Cell],
    neighbor_graph: nx.Graph,
    label_to_cell: Dict[int, Cell],
    max_time_gap: int
) -> List[CellToCellCommunication]:
    """Handle communication creation for peaks not part of any group."""
    communications: List[CellToCellCommunication] = []

    for cell in cells:
        for i, peak in enumerate(cell.trace.peaks):
            if peak.is_analyzed:
                continue

            best_candidate = None
            best_dt = float("inf")
            for neighbor in neighbor_graph.neighbors(cell.label):
                for _, neighbor_peak in enumerate(label_to_cell[neighbor].trace.peaks):
                    dt = peak.rel_start_time - neighbor_peak.rel_start_time
                    if 0 < dt <= max_time_gap and dt < best_dt:
                        best_candidate = (neighbor, neighbor_peak.id, neighbor_peak.rel_start_time)
                        best_dt = dt

            if best_candidate:
                origin_label, origin_idx, origin_time = best_candidate
                comm = CellToCellCommunication(
                    origin=(origin_label, origin_idx),
                    cause=(cell.label, peak.id),
                    origin_start_time=origin_time,
                    cause_start_time=peak.rel_start_time
                )
                communications.append(comm)
                cell.trace.peaks[i].is_analyzed = True
    return communications


def assign_peak_classifications(
    cells: List[Cell],
    communications: List[CellToCellCommunication]
) -> None:
    """
    Assign origin_type and origin_label to each peak based on communication links.

    - If a peak appears at least once as `cause` → "caused"
    - Else if it appears at least once as `origin` and never as `caused` → "origin"
    - Else → "individual"

    The peak.origin_label is always set to the root upstream origin.

    Args:
        cells (List[Cell]): List of cell objects containing peaks.
        communications (List[CellToCellCommunication]): Communication links.
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
                    peak.origin_type = "individual"