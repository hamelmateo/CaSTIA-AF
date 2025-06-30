"""
Population class that encapsulates population-level calcium imaging data and computations.

Usage example:
    >>> population = Population(cells=active_cells)
    >>> population.compute_global_trace()
    >>> population.compute_population_metrics()
    >>> print(population.metadata)
"""

from typing import List, Dict, Optional, Any
import copy
from pathlib import Path
import numpy as np
from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.data.traces import Trace
from calcium_activity_characterization.data.cell_to_cell_communication import CellToCellCommunication, generate_cell_to_cell_communications, assign_peak_classifications
from calcium_activity_characterization.data.copeaking_neighbors import CoPeakingNeighbors, generate_copeaking_groups
from calcium_activity_characterization.data.events import Event, GlobalEvent, SequentialEvent
from calcium_activity_characterization.event_detection.global_event import (
    find_significant_activity_peaks,
    extract_global_event_blocks
)
from calcium_activity_characterization.utilities.spatial import build_spatial_neighbor_graph, filter_graph_by_edge_length_mad, plot_spatial_neighbor_graph
from calcium_activity_characterization.utilities.metrics import Distribution
from calcium_activity_characterization.utilities.loader import get_config_with_fallback
import networkx as nx

import logging
logger = logging.getLogger(__name__)


class Population:
    """
    Represents a population of cells extracted from calcium imaging data,
    along with aggregated metrics, traces, and clustering results.

    Attributes:
        cells (List[Cell]): List of valid Cell objects.
        global_trace (Optional[Trace]): Average trace over all cells.
        metadata (Dict[str, Any]): Dictionary of population-level statistics.
        similarity_matrices (Any): Correlation/similarity matrices.
        peak_clusters (Any): List of temporally clustered peak groups.
        gc_graphs (List[Any]): Granger causality graphs if computed.
        embedding (Any): UMAP or PCA embedding of cells.
    """

    def __init__(self, cells: List[Cell], mask: Optional[np.ndarray], output_path: Optional[Path]) -> None:
        self.cells: List[Cell] = cells
        self.neighbor_graph: nx.Graph = None
        self.copeaking_neighbors: List[CoPeakingNeighbors] = None
        self.cell_to_cell_communication: List[CellToCellCommunication] = None
        self.events: List[Event] = []
        self.activity_trace: Optional[Trace] = None # Sum of raster plot traces over time

        self.metadata: Dict[str, Any] = {}
        self.cell_metrics_distributions: Dict[str, Distribution] = {}
        self.seq_event_metrics_distributions: Dict[str, Distribution] = {}
        self.glob_event_metrics_distributions: Dict[str, Distribution] = {}

        try:
            self.neighbor_graph = build_spatial_neighbor_graph(cells)
            self.neighbor_graph = filter_graph_by_edge_length_mad(self.neighbor_graph, scale=2.0)
            plot_spatial_neighbor_graph(self.neighbor_graph, mask, output_path)
        
        except ValueError as e:
            logger.warning(f"Failed to build spatial neighbor graph: {e}")
            self.neighbor_graph = None


    def _create_trace_object(self, trace: np.ndarray, default_version: str, signal_processing_params: Dict[str, Any] = None, peak_detection_params: Dict[str, Any] = None) -> Trace:
        """
        Internal helper to create a Trace object from a raw array.

        Args:
            trace (np.ndarray): The raw trace array.
            trace_type (str): Name to use for the trace version.
            default_version (str): Which version to default to for downstream use.

        Returns:
            Trace: A Trace object with peak detection and binarization applied.
        """
        t = Trace(trace.tolist())
        t.process_and_plot_trace(input_version="raw", 
                                 output_version=default_version, 
                                 processing_params=signal_processing_params)
        t.default_version = default_version
        t.detect_peaks(peak_detection_params)
        t.binarize_trace_from_peaks()
        return t


    def compute_global_trace(self, version: str = "raw", default_version: str = "raw", signal_processing_params: Dict[str, Any] = None, peak_detection_params: Dict[str, Any] = None) -> None:
        """
        Compute the mean trace across all active cells based on the specified version.

        Args:
            version (str): The key in trace.versions to average.
            default_version (str): The version to set as default in the returned Trace.
            signal_processing_params (Dict[str, Any]): Parameters for signal processing.
            peak_detection_params (Dict[str, Any]): Parameters for peak detection.

        Raises:
            ValueError: If no cells contain the specified version.
        """
        valid_traces = [
            c.trace.versions[version]
            for c in self.cells
            if version in c.trace.versions and len(c.trace.versions[version]) > 0
        ]

        if not valid_traces:
            raise ValueError(f"No valid cells found with trace version '{version}'")

        global_array = np.mean(valid_traces, axis=0)

        self.global_trace = self._create_trace_object(
            global_array,
            default_version=default_version,
            signal_processing_params=signal_processing_params,
            peak_detection_params=peak_detection_params
        )


    def compute_activity_trace(self, default_version: str = "raw", signal_processing_params: Dict[str, Any] = None, peak_detection_params: Dict[str, Any] = None) -> None:
        """
        Compute the activity trace as the sum of binary traces across all cells.

        The activity trace is defined as the sum of binary traces from all cells,
        where each binary trace indicates the presence (1) or absence (0) of activity
        at each time point.

        Args:
            default_version (str): The version to set as default in the returned Trace.
            signal_processing_params (Dict[str, Any]): Parameters for signal processing.
            peak_detection_params (Dict[str, Any]): Parameters for peak detection.

        Raises:
            ValueError: If no binary traces are found in the cells, or if the binary traces are not of uniform length.

        """
        binary_traces = [c.trace.binary for c in self.cells if len(c.trace.binary) > 0]

        if not binary_traces:
            raise ValueError("No binary traces found in cells.")

        trace_lengths = set(len(b) for b in binary_traces)
        if len(trace_lengths) > 1:
            raise ValueError("Binary traces are not of uniform length.")

        activity_trace = np.sum(binary_traces, axis=0)

        self.activity_trace = self._create_trace_object(
            activity_trace,
            default_version=default_version,
            signal_processing_params=signal_processing_params,
            peak_detection_params=peak_detection_params 
        )


    def compute_impulse_trace(self, default_version: str = "raw", signal_processing_params: Dict[str, Any] = None, peak_detection_params: Dict[str, Any] = None) -> None:
        """
        Compute the impulse trace as the sum of rel_start_time occurrences across all cells.

        This trace reflects the number of cell peaks that start at each timepoint.

        Args:
            default_version (str): The version to set as default in the returned Trace.
            signal_processing_params (Dict[str, Any]): Smoothing or filtering params.
            peak_detection_params (Dict[str, Any]): Parameters for peak detection.

        Raises:
            ValueError: If no impulses are found in the cells (empty peak lists) or if the impulse traces are not of uniform length.
        """
        if not any(cell.trace.peaks for cell in self.cells):
            raise ValueError("No impulses found in cells (empty peak lists).")

        impulse_traces = []

        max_time = max(
            (max((p.rel_start_time for p in cell.trace.peaks), default=-1) for cell in self.cells),
            default=-1
        ) + 1

        for cell in self.cells:
            trace = np.zeros(max_time, dtype=int)
            for peak in cell.trace.peaks:
                if 0 <= peak.rel_start_time < max_time:
                    trace[peak.rel_start_time] = 1
            impulse_traces.append(trace)

        trace_lengths = set(len(t) for t in impulse_traces)
        if len(trace_lengths) > 1:
            raise ValueError("Impulse traces are not of uniform length.")

        impulse_trace = np.sum(impulse_traces, axis=0)

        self.impulse_trace = self._create_trace_object(
            impulse_trace,
            default_version=default_version,
            signal_processing_params=signal_processing_params,
            peak_detection_params=peak_detection_params
        )


    def detect_global_events(self, config: Dict[str, Any] = None) -> None:
        """
        Generate global events from the activity trace by detecting significant activity peaks.

        Args:
            config (Dict[str, Any]): Configuration dictionary with parameters for global event detection.
                Expected keys:
                    - "threshold_ratio": Ratio of active cells at peak to trigger detection.
                    - "radius": Radius for peak classification.
                    - "global_max_comm_time": Maximum allowed gap between frames in a global event.
                    - "min_cell_count": Minimum number of cells required to consider a peak significant.
        """
        windows = find_significant_activity_peaks(
            trace=self.activity_trace,
            total_cells=len(self.cells),
            threshold_ratio=get_config_with_fallback(config,"threshold_ratio")
            )

        logger.info("✅ Significant activity windows (start_frame, end_frame):")
        for i, (start, end) in enumerate(windows):
            logger.info(f"  {i:02d}: [{start}, {end}]")

        blocks = extract_global_event_blocks(
            cells=self.cells,
            peak_windows=windows,
            radius=get_config_with_fallback(config,"radius"),
            global_max_comm_time=get_config_with_fallback(config,"global_max_comm_time"),
            min_cell_count=get_config_with_fallback(config,"min_cell_count")
        )

        self.events.extend(GlobalEvent.from_framewise_active_labels(
            framewise_label_blocks=blocks,
            cells=self.cells,
            config=config
        ))


    def detect_sequential_events(self, config: Dict[str, Any] = None) -> None:
        """
        Generate sequential events from cell-to-cell communications.
        This method identifies groups of cells that communicate with each other
        based on their spatial proximity and temporal activity patterns.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary with parameters for sequential event detection.
        """
        clean_cells = self._create_cells_without_global_peaks()

        self.copeaking_neighbors = generate_copeaking_groups(
            cells=clean_cells,
            neighbor_graph=self.neighbor_graph
        )

        self.cell_to_cell_communications = generate_cell_to_cell_communications(
            clean_cells,
            neighbor_graph=self.neighbor_graph,
            copeaking_groups=self.copeaking_neighbors,
            max_time_gap=get_config_with_fallback(config,"seq_max_comm_time")
        )

        del clean_cells  # Free memory

        assign_peak_classifications(self.cells, self.cell_to_cell_communications)

        population_centroids = [np.array(cell.centroid) for cell in self.cells]

        self.events.extend(SequentialEvent.from_communications(
            len(self.events),
            self.cell_to_cell_communications,
            self.cells,
            config=config,
            population_centroids=population_centroids
        ))


    def _create_cells_without_global_peaks(self) -> list[Cell]:
        """
        Returns a deep copy of cells with peaks marked as in_global_event removed.
        Original cells are not affected.
        """
        clean_cells = copy.deepcopy(self.cells)
        for cell in clean_cells:
            cell.trace.peaks = [p for p in cell.trace.peaks if getattr(p, 'in_event', None) != "global"]
            #reassign_peak_ids(cell.trace.peaks)

        return clean_cells


    def assign_peak_event_ids(self) -> None:
        """
        Assigns event_id to each peak that is part of a global or sequential event.
        """
        for event in self.events:
            for cell_label, peak_id in event.peaks_involved:
                cell = next((c for c in self.cells if c.label == cell_label), None)
                if cell is None:
                    continue
                for peak in cell.trace.peaks:
                    if peak.id == peak_id:
                        peak.event_id = event.id
                        break

