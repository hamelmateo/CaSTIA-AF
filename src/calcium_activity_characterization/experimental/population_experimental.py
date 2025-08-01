
from __future__ import annotations
from typing import Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from matplotlib.backends.backend_pdf import PdfPages
from calcium_activity_characterization.utilities.metrics import Distribution, compute_histogram_func, compute_peak_frequency_over_time
from calcium_activity_characterization.data.events import SequentialEvent, GlobalEvent
from calcium_activity_characterization.data.cells import Cell
import copy



def create_cells_without_global_peaks(self) -> list[Cell]:
    """
    Returns a deep copy of cells with peaks marked as in_global_event removed.
    Original cells are not affected.
    """
    clean_cells = copy.deepcopy(self.cells)
    for cell in clean_cells:
        cell.trace.peaks = [p for p in cell.trace.peaks if getattr(p, 'in_event', None) != "global"]
        #reassign_peak_ids(cell.trace.peaks)

    return clean_cells

def compute_population_distributions(self) -> None:
    """
    Compute and store Distribution objects for important cell-wise,
    sequential-event-wise, and global-event-wise metrics.
    """
    # Cell-wise distributions
    durations = []
    prominences = []
    periodicity_scores = []
    num_peaks = []
    in_event_types = []

    for cell in self.cells:
        durations.extend([p.fhw_duration for p in cell.trace.peaks])
        prominences.extend([p.prominence for p in cell.trace.peaks])
        if "periodicity_score" in cell.trace.metadata:
            periodicity_scores.append(cell.trace.metadata["periodicity_score"])
        num_peaks.append(len(cell.trace.peaks))
    in_event_types = [
    str(p.in_event).lower()  # handles "global", "sequential", "none", etc.
    for cell in self.cells
    for p in cell.trace.peaks]

    self.cell_metrics_distributions["peak_duration"] = Distribution.from_values(durations, binning_mode="width", bin_param=1)
    self.cell_metrics_distributions["peak_prominence"] = Distribution.from_values(prominences, binning_mode="width", bin_param=1)
    self.cell_metrics_distributions["periodicity_score"] = Distribution.from_values(periodicity_scores, binning_mode="count", bin_param=50)
    self.cell_metrics_distributions["num_peaks"] = Distribution.from_values(num_peaks, binning_mode="width", bin_param=1)
    label_map = {"global": 1, "sequential": 2, "none": 3}
    encoded = [label_map.get(t, 0) for t in in_event_types]
    self.cell_metrics_distributions["peak_in_event"] = Distribution.from_values(encoded, binning_mode="count", bin_param=3)
    self.cell_metrics_distributions["active_vs_inactive"] = Distribution.from_values([
        1 if len(cell.trace.peaks) > 0 else 0 for cell in self.cells
    ], binning_mode="count", bin_param=2)

    # Sequential event-wise distributions
    comm_times = []
    comm_speeds = []
    avg_comm_times = []
    avg_comm_speeds = []
    elong_scores = []
    radial_scores = []
    dag_depths = []
    n_cells_seq = []
    n_cells_glob = []

    for event in self.events:
        if isinstance(event, SequentialEvent):

            comm_times.extend(event.communication_time_distribution.values)
            comm_speeds.extend(event.communication_speed_distribution.values)
            avg_comm_times.append(event.communication_time_distribution.mean)
            avg_comm_speeds.append(event.communication_speed_distribution.mean)
            elong_scores.append(event.elongation_score)
            radial_scores.append(event.radiality_score)
            dag_depths.append(event.dag_metrics["depth"])
            n_cells_seq.append(event.n_cells_involved)

            self.seq_event_metrics_distributions["communication_time"] = Distribution.from_values(comm_times, binning_mode="count", bin_param=10)
            self.seq_event_metrics_distributions["communication_speed"] = Distribution.from_values(comm_speeds, binning_mode="count", bin_param=50)
            self.seq_event_metrics_distributions["avg_comm_time_per_event"] = Distribution.from_values(avg_comm_times, binning_mode="count", bin_param=10)
            self.seq_event_metrics_distributions["avg_comm_speed_per_event"] = Distribution.from_values(avg_comm_speeds, binning_mode="count", bin_param=50)
            self.seq_event_metrics_distributions["elongation_score"] = Distribution.from_values(elong_scores, binning_mode="count", bin_param=50)
            self.seq_event_metrics_distributions["radiality_score"] = Distribution.from_values(radial_scores, binning_mode="count", bin_param=50)
            self.seq_event_metrics_distributions["dag_max_depth"] = Distribution.from_values(dag_depths, binning_mode="width", bin_param=1)
            self.seq_event_metrics_distributions["n_cells"] = Distribution.from_values(n_cells_seq, binning_mode="width", bin_param=1)

        if isinstance(event, GlobalEvent):
            # Global event-wise distributions
            n_cells_glob.append(event.n_cells_involved)
            self.glob_event_metrics_distributions["n_cells"] = Distribution.from_values(n_cells_glob, binning_mode="width", bin_param=1)


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
    durations = [peak.fhw_duration for cell in self.cells for peak in cell.trace.peaks]
    self.metadata["global_peak_duration_hist"] = compute_histogram_func(durations, bin_width=bin_width)

    prominences = [peak.prominence for cell in self.cells for peak in cell.trace.peaks]
    self.metadata["global_peak_prominence_hist"] = compute_histogram_func(prominences, bin_width=bin_width)

    periodicity_scores = [cell.trace.metadata.get("periodicity_score", 0) for cell in self.cells]
    self.metadata["periodicity_score_hist"] = compute_histogram_func(periodicity_scores, bin_count=bin_counts)
    
    peak_frequencies = [cell.trace.metadata.get("peak_frequency", 0) for cell in self.cells]
    self.metadata["peak_frequency_hist"] = compute_histogram_func(peak_frequencies, bin_count=bin_counts)
    
    num_peaks = [len(cell.trace.peaks) for cell in self.cells]
    self.metadata["num_peaks_hist"] = compute_histogram_func(num_peaks, bin_width=bin_width)

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
        [[peak.fhw_start_time for peak in peaks] for peaks in peaks_per_cell_concatenated],
        max_len,
        window_size=window_size,
        step_size=step_size
    )

    # 6. Synchrony between peaks
    peak_times = [(cell_idx, p.fhw_start_time) for cell_idx, cell in enumerate(self.cells) for p in cell.trace.peaks]
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
    start_times = [p.fhw_start_time for cell in self.cells for p in cell.trace.peaks]
    if start_times:
        hist, _ = np.histogram(start_times, bins=np.arange(0, max(start_times) + bin_width, bin_width))
        probs = hist / np.sum(hist)
        self.metadata["peak_start_time_entropy"] = float(entropy(probs))

    # 8. Inter-peak interval (IPI) entropy
    ipis = []
    for cell in self.cells:
        times = sorted(p.fhw_start_time for p in cell.trace.peaks)
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
