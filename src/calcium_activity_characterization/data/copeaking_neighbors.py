# copeaking_neighbors.py
# Usage Example:
# >>> from calcium_activity_characterization.logic.copeaking_neighbors import generate_copeaking_groups
# >>> groups = generate_copeaking_groups(cells, neighbor_graph)

from typing import List, Tuple, Dict
import networkx as nx
from collections import defaultdict
from calcium_activity_characterization.data.cells import Cell


class CoPeakingNeighbors:
    """
    Represents a group of spatially neighboring cells with synchronized calcium peaks.

    Attributes:
        frame (int): Frame index where peaks co-occur.
        members (List[Tuple[int, int]]): List of (cell_label, peak_index) pairs.
        subgraph (nx.Graph): Subgraph connecting spatial neighbors within the group.
    """

    def __init__(self, frame: int, members: List[Tuple[int, int]], full_graph: nx.Graph):
        """
        Initialize a CoPeakingNeighbors group.

        Args:
            frame (int): The frame when the peaks co-occur.
            members (List[Tuple[int, int]]): List of (cell_label, peak_index) pairs.
            full_graph (nx.Graph): Full spatial neighbor graph of the population.
        """
        self.frame: int = frame
        self.members: List[Tuple[int, int]] = members
        self.labels = {label for label, _ in members}
        self.subgraph: nx.Graph = full_graph.subgraph(self.labels).copy()

    def get_labels(self) -> List[int]:
        """Return a list of unique cell labels involved in the co-peaking event."""
        return list(self.labels)

    def __repr__(self) -> str:
        return f"<CoPeakingNeighbors frame={self.frame} labels={sorted(self.labels)}>"


def generate_copeaking_groups(cells: List[Cell], neighbor_graph: nx.Graph) -> List[CoPeakingNeighbors]:
    """
    Generate all co-peaking groups from a list of cells and a spatial neighbor graph.

    Args:
        cells (List[Cell]): List of Cell objects with peak data.
        neighbor_graph (nx.Graph): Undirected graph of spatial neighbors.

    Returns:
        List[CoPeakingNeighbors]: List of detected co-peaking neighbor groups.
    """
    # Map frame -> list of (label, peak_index)
    frame_to_peaks: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    label_to_cell: Dict[int, Cell] = {cell.label: cell for cell in cells}

    for cell in cells:
        for i, peak in enumerate(cell.trace.peaks):
            frame_to_peaks[peak.start_time].append((cell.label, i))

    groups: List[CoPeakingNeighbors] = []

    for frame, peak_entries in frame_to_peaks.items():
        if len(peak_entries) < 2:
            continue  # Only interested in co-peaking

        # Build graph from co-peaking cells
        group_labels = [label for label, _ in peak_entries]
        subgraph = neighbor_graph.subgraph(group_labels)
        connected_components = nx.connected_components(subgraph)

        for component in connected_components:
            component_members = [entry for entry in peak_entries if entry[0] in component]
            if len(component_members) >= 2:
                group = CoPeakingNeighbors(frame=frame, members=component_members, full_graph=neighbor_graph)
                groups.append(group)

    return groups
