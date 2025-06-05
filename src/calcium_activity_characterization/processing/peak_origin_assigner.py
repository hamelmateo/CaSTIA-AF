"""
Module: peak_origin_assigner.py

This module contains a refactored utility class `PeakOriginAssigner` that takes in a list of Cell objects
and a neighbor graph, and assigns origins to calcium peaks using the exact same logic as the original
`Population.assign_peak_origins` method.

Usage Example:
    >>> assigner = PeakOriginAssigner(cells=population.cells, neighbor_graph=population.neighbor_graph)
    >>> assigner.assign_origins(max_time_gap=5)
"""

from typing import List, Dict, Tuple, Optional, Set
import numpy as np
import networkx as nx
from collections import defaultdict, deque
import logging

from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.data.peaks import Peak

logger = logging.getLogger(__name__)


class PeakOriginAssigner:
    """
    Assigns origin labels to calcium peaks based on temporal and spatial constraints.

    Args:
        cells (List[Cell]): List of Cell objects.
        neighbor_graph (nx.Graph): Graph of spatial cell neighbors.
    """

    def __init__(self, cells: List[Cell], neighbor_graph: nx.Graph):
        self.cells = cells
        self.graph = neighbor_graph
        self.label_to_cell = {cell.label: cell for cell in self.cells}
        self.assigned: Set[Tuple[int, int]] = set()

    def run(self, max_time_gap: int = 5) -> None:
        """
        Assigns origins to peaks in the cells based on copeaking activity and spatial relationships.
        This method first assigns origins to peaks that are part of copeaking groups, then assigns origins
        to remaining peaks that were not assigned during copeaking assignment.

        Args:
            max_time_gap (int): Maximum time gap in seconds to consider for assigning origins.
        """
        self._assign_copeaking_peak_origins(max_time_gap)
        self._assign_remaining_peak_origins(max_time_gap)
        self._finalize_cause_types()


    def _assign_copeaking_peak_origins(self, max_time_gap: int) -> None:
        """
        Assigns origins to peaks that are part of copeaking groups.
        This method identifies groups of cells that are copeaking at the same time and assigns
        the origin label based on the closest neighbor's peak within the specified time gap.

        Args:
            max_time_gap (int): Maximum time gap in seconds to consider for assigning origins.
        """
        copeaking_groups_by_time = self._get_copeaking_neighbors()

        for t, components in copeaking_groups_by_time.items():
            for component in components:
                group = [(self.label_to_cell[l], next(p for p in self.label_to_cell[l].trace.peaks if p.rel_start_time == t)) for l in component]

                candidates = []
                for cell, peak in group:
                    if (cell.label, peak.rel_start_time) in self.assigned:
                        continue
                    for neighbor in self.graph.neighbors(cell.label):
                        if neighbor in component:
                            continue
                        neighbor_cell = self.label_to_cell[neighbor]
                        for other_peak in neighbor_cell.trace.peaks:
                            dt = peak.rel_start_time - other_peak.rel_start_time
                            if 0 < dt <= max_time_gap:
                                spatial_dist = np.linalg.norm(cell.centroid - neighbor_cell.centroid)
                                candidates.append((dt, spatial_dist, neighbor))

                if candidates:
                    candidates.sort()
                    origin = candidates[0][2]
                else:
                    group.sort(key=lambda x: (x[1].start_time, x[0].label))
                    origin = group[0][0].label

                origin_map = self._propagate_origin_through_group(origin, component)

                for cell, peak in group:
                    peak.origin_label = origin_map.get(cell.label, cell.label)
                    """
                    if peak.origin_label != cell.label:
                        try:
                            origin_cell = self.label_to_cell[peak.origin_label]
                        except KeyError:
                            logger.warning(f"Origin label {peak.origin_label} not found in label_to_cell mapping.")
                            continue
                        prior_peaks = [p for p in origin_cell.trace.peaks if p.rel_start_time < peak.rel_start_time]
                        copeaks = [p for p in origin_cell.trace.peaks if p.rel_start_time == peak.rel_start_time]
                        if prior_peaks:
                            origin_peak = max(prior_peaks, key=lambda p: p.rel_start_time)
                            peak.origin_time = origin_peak.rel_start_time
                        elif copeaks:
                            origin_peak = max(copeaks, key=lambda p: p.rel_start_time)
                            peak.origin_time = origin_peak.rel_start_time
                            """
                    self.assigned.add((cell.label, peak.rel_start_time))

    def _assign_remaining_peak_origins(self, max_time_gap: int) -> None:
        """
        Assigns origins to remaining peaks that were not assigned during copeaking assignment.
        This method iterates through all cells and their peaks, checking for unassigned peaks.
        If a peak's origin is not yet assigned, it looks for neighboring cells' peaks that are within
        the specified time gap. The closest neighbor's label is assigned as the origin.

        Args:
            max_time_gap (int): Maximum time gap in seconds to consider for assigning origins.
        """
        for cell in self.cells:
            for peak in cell.trace.peaks:
                key = (cell.label, peak.rel_start_time)
                if key in self.assigned:
                    continue

                peak.origin_label = cell.label
                t0 = peak.rel_start_time
                candidates = []

                for neighbor in self.graph.neighbors(cell.label):
                    neighbor_cell = self.label_to_cell[neighbor]
                    for other_peak in neighbor_cell.trace.peaks:
                        t1 = other_peak.rel_start_time
                        if 0 < (t0 - t1) <= max_time_gap:
                            spatial_dist = np.linalg.norm(cell.centroid - neighbor_cell.centroid)
                            candidates.append((t0 - t1, spatial_dist, neighbor))

                if candidates:
                    candidates.sort()
                    origin_label = candidates[0][2]
                    peak.origin_label = origin_label
                    self.assigned.add(key)
                    """
                    origin_cell = self.label_to_cell[origin_label]
                    prior_peaks = [p for p in origin_cell.trace.peaks if p.rel_start_time < peak.rel_start_time]
                    if prior_peaks:
                        origin_peak = max(prior_peaks, key=lambda p: p.rel_start_time)
                        peak.origin_time = origin_peak.rel_start_time
                        """

    def _finalize_cause_types(self) -> None:
        """
        Classifies each peak as 'origin', 'caused', or 'individual' based on its assigned origin.
        - 'caused' if the peak was assigned an origin from a different cell.
        - 'origin' if the peak was used as the origin of at least one other peak.
        - 'individual' if neither.
        Sets the `cause_type` attribute for each peak accordingly.
        """
        peak_index = {
            (cell.label, peak.rel_start_time): peak
            for cell in self.cells
            for peak in cell.trace.peaks
        }

        caused = set()
        origin_peaks = set()

        for (label, time), peak in peak_index.items():
            if peak.origin_label is not None and peak.origin_label != label:
                caused.add((label, peak.rel_start_time))
                
                if peak.origin_time is not None:
                    origin_peaks.add((peak.origin_label, peak.origin_time))


        for (label, time), peak in peak_index.items():
            if (label, time) in caused:
                peak.cause_type = "caused"
            elif (label, time) in origin_peaks:
                peak.cause_type = "origin"
            else:
                peak.cause_type = "individual"

    def _propagate_origin_through_group(self, origin_label: int, component: Set[int]) -> Dict[int, int]:
        """
        Propagates the origin label through the component using BFS.

        Args:
            origin_label (int): The label of the cell that is the origin.
            component (Set[int]): The set of labels in the connected component.
        
        Returns:
            Dict[int, int]: A mapping of cell labels to their assigned origin label.
        """
        queue = deque([origin_label])
        visited = set(queue)
        origin_map = {origin_label: origin_label}

        while queue:
            current = queue.popleft()
            for neighbor in self.graph.neighbors(current):
                if neighbor in component and neighbor not in visited:
                    origin_map[neighbor] = current
                    visited.add(neighbor)
                    queue.append(neighbor)

        return origin_map

    def _get_copeaking_neighbors(self) -> Dict[int, List[Set[int]]]:
        """
        Groups cells by time where they have copeaking activity, returning a dictionary
        mapping relative start times to sets of labels of cells that are copeaking at that time.
        Each set contains labels of cells that are connected in the neighbor graph.
        
        Returns:
            Dict[int, List[Set[int]]]: A dictionary where keys are relative start times
            and values are lists of sets of labels of cells that are copeaking at that time.
            example: {0: [{1, 2}, {3, 6, 7}], 1: [{1, 4}]}
        """
        time_to_labels = defaultdict(list)
        for cell in self.cells:
            for peak in cell.trace.peaks:
                time_to_labels[peak.rel_start_time].append(cell.label)

        result = {}
        for t, labels in time_to_labels.items():
            subgraph = self.graph.subgraph(labels)
            components = [set(comp) for comp in nx.connected_components(subgraph)]
            result[t] = [comp for comp in components if len(comp) > 1]

        return result