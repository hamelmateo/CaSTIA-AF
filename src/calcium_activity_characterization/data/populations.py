"""
Population class that encapsulates population-level calcium imaging data and computations.

Usage example:
    >>> population = Population(cells=active_cells)
    >>> population.compute_global_trace()
    >>> population.compute_population_metrics()
    >>> print(population.metadata)
"""

import numpy as np
import networkx as nx
from scipy.spatial import Voronoi

from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.data.traces import Trace
from calcium_activity_characterization.data.cell_to_cell_communication import CellToCellCommunication, generate_cell_to_cell_communications, assign_peak_classifications
from calcium_activity_characterization.data.copeaking_neighbors import CoPeakingNeighbors, generate_copeaking_groups
from calcium_activity_characterization.data.events import Event, GlobalEvent, SequentialEvent
from calcium_activity_characterization.event_detection.global_event import (
    find_significant_activity_peaks,
    extract_global_event_blocks
)
from calcium_activity_characterization.config.presets import (
    SignalProcessingConfig, 
    PeakDetectionConfig,
    EventExtractionConfig
)

from calcium_activity_characterization.logger import logger



class Population:
    """
    Represents a population of cells extracted from calcium imaging data,
    along with aggregated metrics, traces, and clustering results.

    Attributes:
        cells (list[Cell]): list of valid Cell objects.
        global_trace (Optional[Trace]): Average trace over all cells.
        metadata (dict[str, any]): dictionary of population-level statistics.
        similarity_matrices (any): Correlation/similarity matrices.
        peak_clusters (any): list of temporally clustered peak groups.
        gc_graphs (list[any]): Granger causality graphs if computed.
        embedding (any): UMAP or PCA embedding of cells.
    """

    def __init__(self, nuclei_mask: np.ndarray, cells: list[Cell], neighbor_graph: nx.Graph) -> None:
        self.nuclei_mask: np.ndarray = nuclei_mask
        self.cells: list[Cell] = cells
        self.neighbor_graph: nx.Graph = neighbor_graph
        self.copeaking_neighbors: list[CoPeakingNeighbors] = None
        self.cell_to_cell_communication: list[CellToCellCommunication] = None
        self.events: list[Event] = []
        self.activity_trace: Trace | None = None # Sum of raster plot traces over time

    @classmethod #TODO: refactor
    def from_roi_filtered(
        cls,
        nuclei_mask: np.ndarray,
        cells: list[Cell],
        graph: nx.Graph,
        roi_scale: float,
        img_shape: tuple[int, int],
        border_margin: int
    ) -> "Population":
        """
        Construct a Population by applying an ROI crop and filtering cells and graph.

        Args:
            cells (list[Cell]): Initial list of valid Cell objects.
            graph (nx.Graph): Spatial neighbor graph with cell.label as nodes.
            roi_scale (float): Fraction of image to keep (0 < roi_scale <= 1).
            img_shape (tuple[int, int]): Shape (height, width) of the full image.
            border_margin (int): Margin inside ROI to exclude near-border cells.

        Returns:
            Population: A new Population instance with cropped cells and pruned graph.
        """
        if not (0 < roi_scale <= 1):
            raise ValueError(f"roi_scale must be in (0,1], got {roi_scale}")

        height, width = img_shape

        crop_h = int(height * roi_scale)
        crop_w = int(width * roi_scale)
        start_h = (height - crop_h) // 2
        start_w = (width - crop_w) // 2
        end_h = start_h + crop_h
        end_w = start_w + crop_w

        safe_start_h = start_h + border_margin
        safe_end_h   = end_h   - border_margin
        safe_start_w = start_w + border_margin
        safe_end_w   = end_w   - border_margin

        filtered_cells: list[Cell] = []
        for cell in cells:
            coords = cell.pixel_coords

            ys = coords[:, 0]
            xs = coords[:, 1]
            min_y, max_y = ys.min(), ys.max()
            min_x, max_x = xs.min(), xs.max()

            if (
                min_y >= safe_start_h and max_y < safe_end_h and
                min_x >= safe_start_w and max_x < safe_end_w
            ):
                cell.adjust_to_roi(start_h, start_w)
                filtered_cells.append(cell)

        keep_labels = {c.label for c in filtered_cells}
        pruned_graph = graph.subgraph(keep_labels).copy()

        for cell in filtered_cells:
            if pruned_graph.has_node(cell.label):
                pruned_graph.nodes[cell.label]['pos'] = cell.centroid

        return cls(nuclei_mask,filtered_cells, pruned_graph)


    def _create_trace_object(self, trace: np.ndarray, default_version: str, 
                             signal_processing_params: SignalProcessingConfig = None, 
                             peak_detection_params: PeakDetectionConfig = None) -> Trace:
        """
        Internal helper to create a Trace object from a raw array.

        Args:
            trace (np.ndarray): The raw trace array.
            trace_type (str): Name to use for the trace version.
            default_version (str): Which version to default to for downstream use.
            signal_processing_params (SignalProcessingConfig): Parameters for signal processing.

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


    def compute_global_trace(self, version: str = "raw", default_version: str = "raw", 
                             signal_processing_params: SignalProcessingConfig = None, 
                             peak_detection_params: PeakDetectionConfig = None) -> None:
        """
        Compute the mean trace across all active cells based on the specified version.

        Args:
            version (str): The key in trace.versions to average.
            default_version (str): The version to set as default in the returned Trace.
            signal_processing_params (SignalProcessingConfig): Parameters for signal processing.
            peak_detection_params (PeakDetectionConfig): Parameters for peak detection.

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


    def compute_activity_trace(self, default_version: str = "raw", 
                               signal_processing_params: SignalProcessingConfig = None, 
                               peak_detection_params: PeakDetectionConfig = None) -> None:
        """
        Compute the activity trace as the sum of binary traces across all cells.

        The activity trace is defined as the sum of binary traces from all cells,
        where each binary trace indicates the presence (1) or absence (0) of activity
        at each time point.

        Args:
            default_version (str): The version to set as default in the returned Trace.
            signal_processing_params (SignalProcessingConfig): Parameters for signal processing.
            peak_detection_params (PeakDetectionConfig): Parameters for peak detection.

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


    def compute_impulse_trace(self, default_version: str = "raw", 
                               signal_processing_params: SignalProcessingConfig = None, 
                               peak_detection_params: PeakDetectionConfig = None) -> None:
        """
        Compute the impulse trace as the sum of activation_start_time occurrences across all cells.

        This trace reflects the number of cell peaks that start at each timepoint.

        Args:
            default_version (str): The version to set as default in the returned Trace.
            signal_processing_params (SignalProcessingConfig): Smoothing or filtering params.
            peak_detection_params (PeakDetectionConfig): Parameters for peak detection.

        Raises:
            ValueError: If no impulses are found in the cells (empty peak lists) or if the impulse traces are not of uniform length.
        """
        if not any(cell.trace.peaks for cell in self.cells):
            raise ValueError("No impulses found in cells (empty peak lists).")

        impulse_traces = []

        max_time = max(
            (max((p.activation_start_time for p in cell.trace.peaks), default=-1) for cell in self.cells),
            default=-1
        ) + 1

        for cell in self.cells:
            trace = np.zeros(max_time, dtype=int)
            for peak in cell.trace.peaks:
                if 0 <= peak.activation_start_time < max_time:
                    trace[peak.activation_start_time] = 1
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


    def detect_global_events(self, config: EventExtractionConfig = None) -> None:
        """
        Generate global events from the activity trace by detecting significant activity peaks.

        Args:
            config (EventExtractionConfig): Configuration object with parameters for global event detection.
                Expected keys:
                    - "threshold_ratio": Ratio of active cells at peak to trigger detection.
                    - "radius": Radius for peak classification.
                    - "global_max_comm_time": Maximum allowed gap between frames in a global event.
                    - "min_cell_count": Minimum number of cells required to consider a peak significant.
        """
        windows = find_significant_activity_peaks(
            trace=self.activity_trace,
            total_cells=len(self.cells),
            threshold_ratio=config.threshold_ratio
            )

        logger.info("✅ Significant activity windows (start_frame, end_frame):")
        for i, (start, end) in enumerate(windows):
            logger.info(f"  {i:02d}: [{start}, {end}]")

        blocks = extract_global_event_blocks(
            cells=self.cells,
            peak_windows=windows,
            radius=config.radius,
            global_max_comm_time=config.global_max_comm_time,
            min_cell_count=config.min_cell_count
        )

        # Extract and sort the peak_time of all peaks in the activity trace
        global_event_peak_times = sorted(
            peak.peak_time
            for peak in self.activity_trace.peaks if peak.is_global_event
        )
        self.events.extend(GlobalEvent.from_framewise_peaking_labels(
            events_peak_times=global_event_peak_times,
            framewise_label_blocks=blocks,
            cells=self.cells,
            config=config
        ))


    def detect_sequential_events(self, config: EventExtractionConfig = None) -> None:
        """
        Generate sequential events from cell-to-cell communications.
        This method identifies groups of cells that communicate with each other
        based on their spatial proximity and temporal activity patterns.
        
        Args:
            config (EventExtractionConfig): Configuration object with parameters for sequential event detection.
        """
        self.copeaking_neighbors = generate_copeaking_groups(
            cells=self.cells,
            neighbor_graph=self.neighbor_graph
        )

        self.cell_to_cell_communications = generate_cell_to_cell_communications(
            self.cells,
            neighbor_graph=self.neighbor_graph,
            copeaking_groups=self.copeaking_neighbors,
            max_time_gap=config.seq_max_comm_time
        )

        assign_peak_classifications(self.cells, self.cell_to_cell_communications)

        population_centroids = [np.array(cell.centroid) for cell in self.cells]

        self.events.extend(SequentialEvent.from_communications(
            len(self.events),
            self.cell_to_cell_communications,
            self.cells,
            config=config,
            population_centroids=population_centroids
        ))

    def assign_peak_event_ids(self) -> None:
        """
        Assigns event_id to each peak that is part of a global or sequential event.
        """
        for event in self.events:
            for cell_label, peak_id in event.peaks_involved:
                cell = next(c for c in self.cells if c.label == cell_label)
                cell.trace.peaks[peak_id].event_id = int(event.id)


    def compute_cell_connection_network_graph(self) -> nx.Graph:
        """
        Build an interaction graph based on registered cell-to-cell communications
        from sequential events. For each communication, an edge is added between
        the origin and cause cell labels with weights incremented by 1.

        Returns:
            nx.Graph: Undirected graph where nodes are cell labels and edges indicate
                      communication frequency between cell pairs.
        """
        communication_graph = nx.Graph()

        # Add all known cells as nodes with positions
        for cell in self.cells:
            communication_graph.add_node(cell.label, pos=cell.centroid)

        for event in self.events:
            if not event.__class__.__name__ == "SequentialEvent":
                continue 

            for communication in event.communications:
                origin_label = communication.origin[0]  # (cell_label, peak_id)
                cause_label = communication.cause[0]

                if communication_graph.has_edge(origin_label, cause_label):
                    communication_graph[origin_label][cause_label]['weight'] += 1
                else:
                    communication_graph.add_edge(origin_label, cause_label, weight=1)

        return communication_graph

    def count_cell_occurences_in_events(self) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
        """
        Count the number of occurrences of each cell in different types of events.

        Returns:
            dict[int, int]: Mapping from cell label to number of occurrences in global events.
            dict[int, int]: Mapping from cell label to number of occurrences in sequential events.
            dict[int, int]: Mapping from cell label to number of occurrences in individual events.
            dict[int, int]: Mapping from cell label to number of occurrences as origin in sequential events.
        """
        global_occurences = {cell.label: cell.count_occurences_global_events()
                             for cell in self.cells}
        sequential_occurences = {cell.label: cell.count_occurences_sequential_events()
                                 for cell in self.cells}
        individual_occurences = {cell.label: cell.count_occurences_individual_events()
                                 for cell in self.cells}
        origin_occurences = {cell.label: cell.count_occurences_sequential_events_as_origin()
                             for cell in self.cells}
        return global_occurences, sequential_occurences, individual_occurences, origin_occurences
        
    def get_early_peakers_in_global_event(self, percentile: float, event_id: int) -> dict[int, int]:
        """
        Get a list of cells that are early peakers in a specific global event.

        Args:
            percentile (float): The percentile threshold to determine early peakers.
            event_id (int): The ID of the global event to analyze.

        Returns:
            dict[int, int]: Mapping of cell labels to their early peaker status (1 or 0).
        """
        event = next((e for e in self.events if e.id == event_id), None)
        if not event:
            logger.warning(f"Global event {event_id} not found.")
            return {}

        first_peaking_cells = event.get_first_X_peaking_cells(percentile)

        first_peaking_cells_mapping: dict[int, int] = {}
        for cell in self.cells:
            if cell.label in first_peaking_cells:
                first_peaking_cells_mapping[cell.label] = 1
            else:
                first_peaking_cells_mapping[cell.label] = 0

        return first_peaking_cells_mapping

    def get_pre_event_peakers_of_global_event(self, percentile: float, event_id: int) -> dict[int, int]:
        """
        Get a list of cells that are pre-event peakers in a specific global event.

        Args:
            event_id (int): The ID of the global event to analyze.

        Returns:
            list[int]: List of cell labels that are considered pre-event peakers in the specified global event.
        """
        event = next((e for e in self.events if e.id == event_id), None)
        if not event:
            logger.warning(f"Global event {event_id} not found.")
            return {}

        pre_peakers_time_window = event.event_duration * percentile
        pre_event_peakers_mapping: dict[int, int] = {}
        
        for cell in self.cells:
            for peak in cell.trace.peaks:
                if (event.event_start_time - pre_peakers_time_window) < peak.communication_time < event.event_start_time:
                    pre_event_peakers_mapping[cell.label] = 1
            pre_event_peakers_mapping.setdefault(cell.label, 0)
            
        return pre_event_peakers_mapping

    @staticmethod
    def build_spatial_neighbor_graph(cells: list[Cell]) -> nx.Graph:
        """
        Build a Voronoi-based spatial neighbor graph from cell centroids.

        Args:
            cells (list[Cell]): list of Cell objects with centroid attributes.

        Returns:
            nx.Graph: Undirected graph with one node per cell, and edges for Voronoi neighbors.
        """
        if not cells:
            raise ValueError("Empty cell list passed to spatial neighbor graph builder.")

        label_to_centroid = {cell.label: tuple(cell.centroid) for cell in cells}
        labels = list(label_to_centroid.keys())
        centroids = np.array([label_to_centroid[label] for label in labels])

        if len(centroids) < 3:
            raise ValueError("At least 3 cells are needed to compute a Voronoi diagram.")

        vor = Voronoi(centroids)
        graph = nx.Graph()

        for label in labels:
            graph.add_node(label, pos=label_to_centroid[label])

        for i, j in vor.ridge_points:
            label_i = labels[i]
            label_j = labels[j]
            graph.add_edge(label_i, label_j, method="voronoi")

        return graph