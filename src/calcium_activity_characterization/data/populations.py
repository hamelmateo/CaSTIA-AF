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
from scipy.stats import entropy

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

    def __init__(self, cells: List[Cell]) -> None:
        self.cells: List[Cell] = cells
        self.global_trace: Optional[Trace] = None
        self.metadata: Dict[str, Any] = {}
        self.similarity_matrices: Optional[List[np.ndarray]]= None
        self.peak_clusters: Optional[List[Cluster]] = None

    def compute_global_trace(self, version: str = "raw", default_version: str = "raw") -> None:
        """
        Compute the mean trace across all active cells based on the specified version.

        Args:
            version (str): The key in trace.versions to average.
            default_version (str): The version to set as default in the returned Trace.

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

        self.global_trace = Trace()
        self.global_trace.versions[default_version] = global_array.tolist()
        self.global_trace.default_version = default_version


    def compute_population_metrics(self, bin_width: int = 10, synchrony_window: int = 1) -> None:
        """
        Compute and store population-level metadata into `self.metadata`.

        Args:
            bin_width (int): Width of histogram bins for duration, prominence, and timing metrics.
            synchrony_window (int): Maximum time lag (in frames) to consider peaks as synchronous.

        Metrics computed:
            - fraction_active_cells (float in [0,1]):
                Proportion of cells with ≥1 detected peaks.
            
            - fraction_inactive_cells (float in [0,1]):
                Complement of above. Useful to detect silent populations.

            - global_peak_duration_hist (dict: "bins", "counts"):
                Histogram of all peak durations pooled across all cells using bin width = `bin_width`.

            - global_peak_prominence_hist (dict: "bins", "counts"):
                Histogram of peak prominences (baseline-independent feature) pooled across all cells.
                Prominence is less biased by fluorescence baseline variability than amplitude.

            - periodicity_score_distribution (List[float], range: [0, 1]):
                One value per cell. Higher means more regular inter-peak intervals.
                Computed as: `1 / (1 + CV(IPIs))`.

            - peak_frequency_distribution (List[float], typically ∈ [0, 0.05]):
                Frequency of peaks per cell = num_peaks / trace_length.

            - num_peaks_distribution (List[int]):
                Number of peaks detected per cell.

            - peak_rate_evolution (List[int]):
                Sum of binary traces across cells, frame-by-frame.
                Useful to detect bursts of coordinated population activity.

            - population_peak_frequency_evolution (List[float]):
                Mean peak frequency per frame across all cells, computed in sliding windows.
                Normalized by (number of cells x window size).

            - proportion_synchronous_peaks (float in [0,1]):
                Fraction of peaks that co-occur in multiple cells within ±synchrony_window.
                Higher = more temporal alignment across the population.

            - peak_start_time_entropy (float ≥ 0):
                Entropy of all peak start times pooled across cells (binned).
                Higher = more uniformly distributed peak onsets over time.

            - ipi_entropy (float ≥ 0):
                Entropy of inter-peak intervals (IPIs) pooled across all cells.
                Higher = more heterogeneous pacing of cellular transients.

            - cell_peak_count_cv (float ≥ 0):
                Coefficient of variation of peak counts across cells.
                Higher = more heterogeneity in cell activity level.
        """
        n_total = len(self.cells)
        n_active = sum(1 for cell in self.cells if cell.trace.peaks)
        n_inactive = n_total - n_active
        self.metadata["fraction_active_cells"] = n_active / n_total if n_total else 0
        self.metadata["fraction_inactive_cells"] = n_inactive / n_total if n_total else 0

        # 2. Duration and prominence histograms
        durations = [peak.duration for cell in self.cells for peak in cell.trace.peaks]
        prominences = [peak.prominence for cell in self.cells for peak in cell.trace.peaks]

        if durations:
            dur_bins = np.arange(0, max(durations) + bin_width, bin_width)
            dur_counts, _ = np.histogram(durations, bins=dur_bins)
            self.metadata["global_peak_duration_hist"] = {
                "bins": dur_bins.tolist(),
                "counts": dur_counts.tolist()
    }

        if prominences:
            prom_bins = np.arange(0, max(prominences) + bin_width, bin_width)
            prom_counts, _ = np.histogram(prominences, bins=prom_bins)
            self.metadata["global_peak_prominence_hist"] = {
                "bins": prom_bins.tolist(),
                "counts": prom_counts.tolist()
            }

        # 3. Periodicity & peak frequency distributions
        self.metadata["periodicity_score_distribution"] = [
            cell.trace.metadata.get("periodicity_score")
            for cell in self.cells if "periodicity_score" in cell.trace.metadata
        ]
        self.metadata["peak_frequency_distribution"] = [
            cell.trace.metadata.get("peak_frequency")
            for cell in self.cells if "peak_frequency" in cell.trace.metadata
        ]
        self.metadata["num_peaks_distribution"] = [len(cell.trace.peaks) for cell in self.cells]

        # 4. Raster-like peak count per frame
        max_len = max((len(cell.trace.binary) for cell in self.cells if cell.trace.binary), default=0)
        peak_rate_evolution = np.zeros(max_len, dtype=int)
        for cell in self.cells:
            if cell.trace.binary:
                peak_rate_evolution[:len(cell.trace.binary)] += np.array(cell.trace.binary)
        self.metadata["peak_rate_evolution"] = peak_rate_evolution.tolist()

        # 5. Peak frequency evolution over time (normalized)
        window_size = 200
        step_size = 50
        freq_evo = []
        for start in range(0, max_len - window_size + 1, step_size):
            end = start + window_size
            peak_count = sum(
                sum(start <= p.start_time < end for p in cell.trace.peaks)
                for cell in self.cells
            )
            norm_freq = peak_count / (window_size * n_total) if n_total > 0 else 0
            freq_evo.append(norm_freq)
        self.metadata["population_peak_frequency_evolution"] = freq_evo

        # 6. Synchrony between peaks
        peak_times = [(cell_idx, p.start_time) for cell_idx, cell in enumerate(self.cells) for p in cell.trace.peaks]
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
        start_times = [p.start_time for cell in self.cells for p in cell.trace.peaks]
        if start_times:
            hist, _ = np.histogram(start_times, bins=np.arange(0, max(start_times) + bin_width, bin_width))
            probs = hist / np.sum(hist)
            self.metadata["peak_start_time_entropy"] = float(entropy(probs))

        # 8. Inter-peak interval (IPI) entropy
        ipis = []
        for cell in self.cells:
            times = sorted(p.start_time for p in cell.trace.peaks)
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
        if "peak_rate_evolution" in self.metadata:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(self.metadata["peak_rate_evolution"], color='blue')
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

        # 6. Periodicity Score Distribution
        if "periodicity_score_distribution" in self.metadata:
            values = [v for v in self.metadata["periodicity_score_distribution"] if isinstance(v, (int, float))]
            if values:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(values, bins=20, color='slateblue')
                ax.set_title("Periodicity Score Distribution")
                ax.set_xlabel("Score (0–1)")
                ax.set_ylabel("Cell Count")
                ax.grid(True)
                _handle(fig)


        # 7. Peak Frequency Distribution
        if "peak_frequency_distribution" in self.metadata:
            values = [v for v in self.metadata["peak_frequency_distribution"] if isinstance(v, (int, float))]
            if values:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(values, bins=20, color='seagreen')
                ax.set_title("Peak Frequency per Cell")
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

        # 9. Number of Peaks per Cell
        if "num_peaks_distribution" in self.metadata:
            values = [v for v in self.metadata["num_peaks_distribution"] if isinstance(v, (int, float))]
            if values:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(values, bins=20, color='gray')
                ax.set_title("Number of Peaks per Cell")
                ax.set_xlabel("Num Peaks")
                ax.set_ylabel("Cell Count")
                ax.grid(True)
                _handle(fig)

        if pdf:
            pdf.close()
