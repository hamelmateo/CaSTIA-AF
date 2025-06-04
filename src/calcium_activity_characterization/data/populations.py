"""
Population class that encapsulates population-level calcium imaging data and computations.

Usage example:
    >>> population = Population(cells=active_cells)
    >>> population.compute_global_trace()
    >>> population.compute_population_metrics()
    >>> print(population.metadata)
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.data.traces import Trace
from calcium_activity_characterization.data.clusters import Cluster

from calcium_activity_characterization.utilities.metrics import compute_histogram, compute_peak_frequency_over_time
from calcium_activity_characterization.utilities.spatial import build_spatial_neighbor_graph, filter_graph_by_edge_length_mad, plot_spatial_neighbor_graph

from calcium_activity_characterization.utilities.loader import get_config_with_fallback

import networkx as nx
from scipy.stats import entropy
from collections import defaultdict, deque

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
        self.global_trace: Optional[Trace] = None   # Mean trace across all cells
        self.activity_trace: Optional[Trace] = None # Sum of raster plot traces over time
        self.impulse_trace: Optional[Trace] = None  # Trace of summed rel_start_time impulses
        self.metadata: Dict[str, Any] = {}
        self.similarity_matrices: Optional[List[np.ndarray]]= None
        self.peak_clusters: Optional[List[Cluster]] = None
        self.event_clusters: Optional[List[List[Cluster]]] = None

        try:
            self.neighbor_graph = build_spatial_neighbor_graph(cells)
            self.neighbor_graph = filter_graph_by_edge_length_mad(self.neighbor_graph, scale=2.0)
            plot_spatial_neighbor_graph(self.neighbor_graph, mask, output_path /"neighbor_graph_filtered.png")
        
        except ValueError as e:
            logger.warning(f"Failed to build spatial neighbor graph: {e}")
            self.neighbor_graph = None



    def assign_peak_origins(self, max_time_gap: int = 5) -> None:
        """
        Assign origin labels to all peaks using neighbor propagation logic.
        Handles co-peaking cells and standard isolated peaks.

        Args:
            max_time_gap (int): Max time allowed between origin and caused peak.
        """
        label_to_cell = {cell.label: cell for cell in self.cells}
        assigned = set()
        graph = self.neighbor_graph
        
        # Step 1: Identify co-peaking subgraphs by time
        copeaking_groups_by_time = self.get_copeaking_neighbors()

        # Step 2: For each timepoint, resolve co-peaking groups
        for t, components in copeaking_groups_by_time.items():
            for component in components:
                group = [(label_to_cell[l], next(p for p in label_to_cell[l].trace.peaks if p.rel_start_time == t)) for l in component]
                
                # Step 1: External candidates
                candidates = []
                for cell, peak in group:
                    if (cell.label, peak.rel_start_time) in assigned:
                        continue
                    peak.origin_label = cell.label
                    for neighbor in graph.neighbors(cell.label):
                        if neighbor in component:
                            continue
                        neighbor_cell = label_to_cell[neighbor]
                        for other_peak in neighbor_cell.trace.peaks:
                            if 0 < (peak.rel_start_time - other_peak.rel_start_time) <= max_time_gap:
                                spatial_dist = np.linalg.norm(cell.centroid - neighbor_cell.centroid)
                                candidates.append((peak.rel_start_time - other_peak.rel_start_time, spatial_dist, neighbor))

                if candidates:
                    candidates.sort()
                    origin = candidates[0][2]
                else:
                    # Step 2: No external origin → find earliest in group
                    group.sort(key=lambda x: (x[1].start_time, x[0].label))
                    origin = group[0][0].label

                # Step 3: Propagate origin through group
                queue = deque([origin])
                visited = set(queue)
                origin_map = {origin: None}

                while queue:
                    current = queue.popleft()
                    for neighbor in graph.neighbors(current):
                        if neighbor in component and neighbor not in visited:
                            origin_map[neighbor] = current
                            visited.add(neighbor)
                            queue.append(neighbor)

                for cell, peak in group:
                    if origin_map.get(cell.label) is None:
                        peak.origin_label = cell.label
                    else:
                        peak.origin_label = origin_map.get(cell.label, cell.label)
                    assigned.add((cell.label, peak.rel_start_time))
        
        # Handle remaining peaks not in any group
        for cell in self.cells: 
            for peak in cell.trace.peaks:
                if (cell.label, peak.rel_start_time) in assigned:
                    continue
                peak.origin_label = cell.label
                t0 = peak.rel_start_time
                candidates = []

                for neighbor in graph.neighbors(cell.label):
                    neighbor_cell = label_to_cell[neighbor]
                    for other_peak in neighbor_cell.trace.peaks:
                        t1 = other_peak.rel_start_time
                        if 0 < (t0 - t1) <= max_time_gap:
                            spatial_dist = np.linalg.norm(cell.centroid - neighbor_cell.centroid)
                            candidates.append((t0 - t1, spatial_dist, neighbor))

                if candidates:
                    candidates.sort(key=lambda x: (x[0], x[1]))  # most recent, then closest
                    peak.origin_label = candidates[0][2]

        self.print_peak_origins()


    def print_peak_origins(self) -> None:
        """
        Print for each peak its cell, start time, and assigned origin peak.
        Also report how many are root peaks vs caused peaks.
        """
        label_to_cell = {cell.label: cell for cell in self.cells}

        root_count = 0
        caused_count = 0

        for cell in self.cells:
            for peak in cell.trace.peaks:
                origin_label = peak.origin_label
                origin_time = None

                if origin_label in label_to_cell:
                    origin_cell = label_to_cell[origin_label]
                    for other_peak in origin_cell.trace.peaks:
                        if other_peak.rel_start_time <= peak.rel_start_time <= other_peak.rel_end_time:
                            origin_time = other_peak.rel_start_time
                            break

                if origin_label == cell.label:
                    root_count += 1
                else:
                    caused_count += 1

                print(f"Cell {cell.label} @ t={peak.rel_start_time} ← Origin: Cell {origin_label} @ t={origin_time}")

        print(f"\n✅ Root peaks (self-origin): {root_count}")
        print(f"✅ Caused peaks (by other cells): {caused_count}")


    def get_copeaking_neighbors(self) -> Dict[int, List[set[int]]]:
        """
        For each rel_start_time, return sets of co-peaking neighbor cell labels.

        Returns:
            Dict[int, List[Set[int]]]: Map from rel_start_time to list of connected cell label sets.
        """
        from collections import defaultdict
        import networkx as nx

        time_to_labels = defaultdict(list)
        for cell in self.cells:
            for peak in cell.trace.peaks:
                time_to_labels[peak.rel_start_time].append(cell.label)

        result = {}
        for t, labels in time_to_labels.items():
            subgraph = self.neighbor_graph.subgraph(labels)
            components = [set(comp) for comp in nx.connected_components(subgraph)]
            # Filter out singletons
            result[t] = [comp for comp in components if len(comp) > 1]

        return result



    def resolve_simultaneous_peaks(self) -> None:
        """
        After initial origin assignment, merge simultaneous peaks from connected cells
        into shared clusters by selecting a canonical root.
        """
        from collections import defaultdict

        label_to_cell = {cell.label: cell for cell in self.cells}
        time_to_cells = defaultdict(list)

        # Index all peaks by rel_start_time
        for cell in self.cells:
            for peak in cell.trace.peaks:
                time_to_cells[peak.rel_start_time].append((cell.label, peak))

        for t, entries in time_to_cells.items():
            # build connected subgraphs of co-peaking neighbors
            labels = [label for label, _ in entries]
            subgraph = self.neighbor_graph.subgraph(labels)

            for component in nx.connected_components(subgraph):
                if len(component) > 1:
                    root_label = min(component)  # or other criteria
                    for label in component:
                        peak = next(p for l, p in entries if l == label)
                        peak.origin_label = root_label



    
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
        t.process_trace("raw", default_version, signal_processing_params)
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



    def compute_population_metrics(self, bin_counts: int = 20, bin_width: int = 1, synchrony_window: int = 1) -> None:
        """
        Compute and store population-level metrics in `self.metadata`.

        Args:
            bin_width (int, optional): Bin width for histograms of durations, start times, and IPIs. Default is 10.
            synchrony_window (int, optional): Maximum time lag (in frames) to consider peaks as synchronous. Default is 1.
        
        Metrics computed and stored in `self.metadata`:
            - fraction_active_cells (float): Proportion of cells with at least one detected peak.
            - fraction_inactive_cells (float): Proportion of cells with no detected peaks.

            - global_peak_duration_hist (dict): Histogram of all peak durations pooled across cells, using `bin_width`.
            - global_peak_prominence_hist (dict): Histogram of all peak prominences pooled across cells (20 bins).
            - periodicity_score_hist (dict): Histogram of periodicity scores per cell (20 bins).
            - peak_frequency_hist (dict): Histogram of per-cell peak frequencies (20 bins).
            - num_peaks_hist (dict): Histogram of number of peaks per cell (20 bins).

            - active_cells_over_time (list of int): Frame-by-frame sum of binary peak traces across all cells.
            - population_peak_frequency_evolution (list of float): Mean peak frequency per frame across all cells, computed in sliding windows (window size 200, step size 50), normalized by (number of cells × window size).
            - proportion_synchronous_peaks (float): Fraction of peaks that co-occur in multiple cells within ±synchrony_window.
            - peak_start_time_entropy (float): Entropy of all peak start times pooled across cells (binned by `bin_width`).
            - ipi_entropy (float): Entropy of inter-peak intervals (IPIs) pooled across all cells (binned by `bin_width`).
            - cell_peak_count_cv (float): Coefficient of variation of peak counts across cells.
        
        Notes:
            - All histograms are returned as dictionaries with "bins" and "counts" keys.
            - Entropy values are computed using the probability distribution of binned values.
            - Synchrony is defined as peaks in different cells occurring within ±synchrony_window frames.
        """
        # 1. Fraction of active vs inactive cells
        n_total = len(self.cells)
        n_active = sum(1 for cell in self.cells if cell.trace.peaks)
        n_inactive = n_total - n_active
        self.metadata["fraction_active_cells"] = n_active / n_total if n_total else 0
        self.metadata["fraction_inactive_cells"] = n_inactive / n_total if n_total else 0

        # 2. Duration, prominence, periodicity score, peak frequencies and number of peaks distributions
        durations = [peak.rel_duration for cell in self.cells for peak in cell.trace.peaks]
        self.metadata["global_peak_duration_hist"] = compute_histogram(durations, bin_width=bin_width)

        prominences = [peak.prominence for cell in self.cells for peak in cell.trace.peaks]
        self.metadata["global_peak_prominence_hist"] = compute_histogram(prominences, bin_width=bin_width)

        periodicity_scores = [cell.trace.metadata.get("periodicity_score", 0) for cell in self.cells]
        self.metadata["periodicity_score_hist"] = compute_histogram(periodicity_scores, bin_count=bin_counts)
        
        peak_frequencies = [cell.trace.metadata.get("peak_frequency", 0) for cell in self.cells]
        self.metadata["peak_frequency_hist"] = compute_histogram(peak_frequencies, bin_count=bin_counts)
        
        num_peaks = [len(cell.trace.peaks) for cell in self.cells]
        self.metadata["num_peaks_hist"] = compute_histogram(num_peaks, bin_width=bin_width)

        # 3. Number of active cells over time
        max_len = max((len(cell.trace.binary) for cell in self.cells if cell.trace.binary), default=0)
        active_cells_over_time = np.zeros(max_len, dtype=int)
        for cell in self.cells:
            if cell.trace.binary:
                active_cells_over_time[:len(cell.trace.binary)] += np.array(cell.trace.binary)
        self.metadata["active_cells_over_time"] = active_cells_over_time.tolist()

        # 5. Peak frequency evolution over time
        window_size = 200
        step_size = 50
        peaks_per_cell_concatenated = [cell.trace.peaks for cell in self.cells]
        self.metadata["population_peak_frequency_evolution"] = compute_peak_frequency_over_time(
            [[peak.rel_start_time for peak in peaks] for peaks in peaks_per_cell_concatenated],
            max_len,
            window_size=window_size,
            step_size=step_size
        )

        # 6. Synchrony between peaks
        peak_times = [(cell_idx, p.rel_start_time) for cell_idx, cell in enumerate(self.cells) for p in cell.trace.peaks]
        peak_times.sort(key=lambda x: x[1])
        total_peaks = len(peak_times)
        count_synch = 0
        for i, (_, t1) in enumerate(peak_times):
            for j in range(i + 1, total_peaks):
                if peak_times[j][1] - t1 > synchrony_window:
                    break
                if peak_times[j][0] != peak_times[i][0]:
                    count_synch += 1
                    break
        self.metadata["proportion_synchronous_peaks"] = count_synch / total_peaks if total_peaks else 0

        # 7. Peak start time entropy
        start_times = [p.rel_start_time for cell in self.cells for p in cell.trace.peaks]
        if start_times:
            hist, _ = np.histogram(start_times, bins=np.arange(0, max(start_times) + bin_width, bin_width))
            probs = hist / np.sum(hist)
            self.metadata["peak_start_time_entropy"] = float(entropy(probs))

        # 8. Inter-peak interval (IPI) entropy
        ipis = []
        for cell in self.cells:
            times = sorted(p.rel_start_time for p in cell.trace.peaks)
            ipis.extend(j - i for i, j in zip(times[:-1], times[1:]) if j > i)
        if ipis:
            hist, _ = np.histogram(ipis, bins=np.arange(0, max(ipis) + bin_width, bin_width))
            probs = hist / np.sum(hist)
            self.metadata["ipi_entropy"] = float(entropy(probs))
        
        # 9. CV of peak count per cell
        peak_counts = [len(cell.trace.peaks) for cell in self.cells]
        mean_count = np.mean(peak_counts)
        std_count = np.std(peak_counts)
        self.metadata["cell_peak_count_cv"] = float(std_count / (mean_count + 1e-8))


    def plot_metadata_summary(self, save_path: Optional[Path] = None) -> None:
        """
        Visualize and optionally save all population-level metadata plots in a single PDF.

        If `save_path` is provided, all plots are saved into one multi-page PDF file.
        If `save_path` is None, plots are shown interactively.

        Each page of the PDF corresponds to a specific population-level metric:

        1. Fraction of Active and Inactive Cells:
            - Bar plot of the fraction of active vs inactive cells.
            - Useful to detect silent or overly active populations.

        2. Peak Rate Evolution:
            - Time series of the number of cells active at each frame.
            - Captures global bursting or wave events.

        3. Peak Duration Histogram:
            - Histogram of peak durations pooled from all cells.
            - Bin width is controlled via compute_population_metrics.

        4. Peak Prominence Histogram:
            - Histogram of peak prominence (baseline-independent strength of peaks).
            - Better suited than amplitude for comparing between cells.

        5. Population Peak Frequency Evolution:
            - Shows mean peak frequency across all cells over time (windowed).
            - Reflects global rhythmicity or activity density.

        6. Periodicity and Peak Frequency Distributions:
            - Two histograms showing how regular and how active cells are.

        7. Synchrony, Entropy, and CV Annotations:
            - Text-based display of:
                - Synchrony (0–1): Temporal co-activation
                - Start Time Entropy (≥ 0): Spread of peaks over time
                - IPI Entropy (≥ 0): Variability in inter-peak intervals
                - Peak Count CV (≥ 0): Heterogeneity in cell activity

        Args:
            save_path (Optional[Path]): Path to save the PDF. If None, plots are shown.

        Raises:
            ValueError: If metadata has not been computed.
        """
        if not self.metadata:
            raise ValueError("No metadata found. Call `compute_population_metrics()` first.")

        pdf = PdfPages(str(save_path)) if save_path else None

        def _handle(fig):
            if pdf:
                pdf.savefig(fig)
                plt.close(fig)
            else:
                plt.show()

        # 1. Active vs Inactive Cells
        fig, ax = plt.subplots(figsize=(6, 4))
        active = self.metadata.get("fraction_active_cells", 0)
        inactive = self.metadata.get("fraction_inactive_cells", 0)
        ax.bar(["Active", "Inactive"], [active, inactive], color=["green", "gray"])
        ax.set_ylim(0, 1)
        ax.set_title("Fraction of Active vs Inactive Cells")
        _handle(fig)

        # 2. Peak Rate Evolution
        if "active_cells_over_time" in self.metadata:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(self.metadata["active_cells_over_time"], color='blue')
            ax.set_title("Peak Rate Evolution Over Time")
            ax.set_xlabel("Time Frame")
            ax.set_ylabel("Number of Active Cells")
            ax.grid(True)
            _handle(fig)

        # 3. Peak Duration Histogram
        if "global_peak_duration_hist" in self.metadata:
            data = self.metadata["global_peak_duration_hist"]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(data["bins"][:-1], data["counts"], width=np.diff(data["bins"]), align="edge", color='purple')
            ax.set_title("Peak Duration Histogram")
            ax.set_xlabel("Duration (frames)")
            ax.set_ylabel("Count")
            ax.grid(True)
            _handle(fig)

        # 4. Peak Prominence Histogram
        if "global_peak_prominence_hist" in self.metadata:
            data = self.metadata["global_peak_prominence_hist"]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(data["bins"][:-1], data["counts"], width=np.diff(data["bins"]), align="edge", color='orange')
            ax.set_title("Peak Prominence Histogram")
            ax.set_xlabel("Prominence")
            ax.set_ylabel("Count")
            ax.grid(True)
            _handle(fig)

        # 5. Population Peak Frequency Evolution
        if "population_peak_frequency_evolution" in self.metadata:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(self.metadata["population_peak_frequency_evolution"], color='darkred')
            ax.set_title("Population Peak Frequency Evolution")
            ax.set_xlabel("Window Index")
            ax.set_ylabel("Frequency per Frame")
            ax.grid(True)
            _handle(fig)

        # 6. Periodicity Score Histogram
        if "periodicity_score_hist" in self.metadata:
            data = self.metadata["periodicity_score_hist"]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(data["bins"][:-1], data["counts"], width=np.diff(data["bins"]), align="edge", color='slateblue')
            ax.set_title("Periodicity Score Histogram")
            ax.set_xlabel("Score (0–1)")
            ax.set_ylabel("Cell Count")
            ax.grid(True)
            _handle(fig)

        # 7. Peak Frequency Histogram
        if "peak_frequency_hist" in self.metadata:
            data = self.metadata["peak_frequency_hist"]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(data["bins"][:-1], data["counts"], width=np.diff(data["bins"]), align="edge", color='seagreen')
            ax.set_title("Peak Frequency Histogram")
            ax.set_xlabel("Peaks / Frame")
            ax.set_ylabel("Cell Count")
            ax.grid(True)
            _handle(fig)

        # 8. Synchrony, Entropy, and CV as text
        synchrony = self.metadata.get("proportion_synchronous_peaks", None)
        entropy_time = self.metadata.get("peak_start_time_entropy", None)
        entropy_ipi = self.metadata.get("ipi_entropy", None)
        cv = self.metadata.get("cell_peak_count_cv", None)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        text = """
        Synchrony: {:.3f} → Higher = more temporal co-activation
        Peak Start Time Entropy: {:.3f} → Higher = more uniformly distributed peak times
        IPI Entropy: {:.3f} → Higher = more irregular peak intervals
        Peak Count CV: {:.3f} → Higher = greater variability across cells
        """.format(
            synchrony or 0, entropy_time or 0, entropy_ipi or 0, cv or 0
        )
        ax.text(0.01, 0.95, text, va="top", ha="left", fontsize=10, wrap=True)
        ax.set_title("Synchrony, Entropy, and Variability Summary", fontsize=12)
        _handle(fig)

        # 8. Number of Peaks per Cell Histogram
        if "num_peaks_hist" in self.metadata:
            data = self.metadata["num_peaks_hist"]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(data["bins"][:-1], data["counts"], width=np.diff(data["bins"]), align="edge", color='darkred')
            ax.set_title("Number of Peaks per Cell")
            ax.set_xlabel("Peak Count")
            ax.set_ylabel("Cell Count")
            ax.grid(True)
            _handle(fig)

        if pdf:
            pdf.close()
