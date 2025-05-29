"""
Example usage:
    >>> t = Trace(raw_trace=[1, 2, 3, 2, 1, 0, 1, 3, 4, 2])
    >>> t.versions["smoothed"] = some_smoothed_version
    >>> t.default_version = "smoothed"
    >>> t.detect_peaks(detector)
    >>> t.binarize_trace_from_peaks()
    >>> t.plot_all_traces(save_path=Path("trace_summary.png"))
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import entropy, skew
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from calcium_activity_characterization.data.peaks import Peak

class Trace:
    """
    A container for calcium activity trace and analysis results.

    This class manages multiple versions of the trace (e.g., raw, smoothed, normalized),
    and supports peak detection, binarization, and metadata computation based on a selected version.

    Attributes:
        raw (List[float]): Raw intensity trace.
        versions (Dict[str, List[float]]): Dictionary of named preprocessed trace versions.
        default_version (str): Key name for the version to operate on.
        binary (List[int]): Binarized trace (0/1 based on peak regions).
        peaks (List): List of detected peaks.
        metadata (Dict[str, float]): Computed features from binary + peak data.
    """

    def __init__(self, raw_trace: List[float] = None):
        self.raw: List[float] = raw_trace if raw_trace is not None else []
        self.versions: Dict[str, List[float]] = {}
        self.default_version: str = "smoothed"  # default key to operate on
        self.binary: List[int] = []
        self.peaks: List[Peak] = []
        self.metadata: Dict[str, float] = {}

    @property
    def active_trace(self) -> List[float]:
        """Returns the currently selected version of the trace."""
        return self.versions.get(self.default_version, [])

    def detect_peaks(self, detector) -> None:
        """Detect peaks in the active trace using the provided detector.
        
        Args:
            detector: An instance of a peak detection algorithm that implements a `run` method.
        """
        trace = self.active_trace
        if trace is None or len(trace) == 0:
            self.peaks = []
        else:
            self.peaks = detector.run(trace)

    def binarize_trace_from_peaks(self) -> None:
        """Convert detected peaks into a binary 0/1 trace and compute metadata."""
        trace = self.active_trace
        if trace is None or len(trace) == 0:
            self.binary = []
            return

        trace_length = len(trace)
        binary = np.zeros(trace_length, dtype=int)
        for peak in self.peaks:
            start = max(0, peak.start_time)
            end = min(trace_length, peak.end_time + 1)
            binary[start:end] = 1

        self.binary = binary.tolist()
        self.compute_metadata()

    def compute_metadata(self) -> None:
        """Compute statistics based on the binary trace and detected peaks.
        
        This includes global statistics, peak feature distributions, periodicity, entropy measures,
        histograms, symmetry, density center, and dynamic evolution features. The results are stored
        in the `metadata` attribute as a dictionary.

        The metadata includes:
            - fraction_active_time: Fraction of time the trace is active (binary = 1).
            - num_peaks: Total number of detected peaks.
            - burst_frequency: Frequency of bursts (peaks) in the trace.
            - total_active_frames: Total number of frames where the trace is active.
            - mean_peak_duration: Mean duration of detected peaks.
            - std_peak_duration: Standard deviation of peak durations.
            - mean_inter_peak_interval: Mean interval between consecutive peaks.
            - std_inter_peak_interval: Standard deviation of inter-peak intervals.
            - mean_peak_amplitude: Mean amplitude of detected peaks.
            - std_peak_amplitude: Standard deviation of peak amplitudes.
            - mean_peak_prominence: Mean prominence of detected peaks.
            - std_peak_prominence: Standard deviation of peak prominences.
            - coefficient_of_variation_prominence: CV of peak prominences.
            - mean_rise_time: Mean rise time of detected peaks.
            - mean_decay_time: Mean decay time of detected peaks.
            - amplitude_skewness: Skewness of peak amplitudes.
            - duration_skewness: Skewness of peak durations.
            - coefficient_of_variation_amplitude: CV of peak amplitudes.
            - coefficient_of_variation_duration: CV of peak durations.
            - periodicity_score: Score indicating periodicity based on inter-peak intervals.
            - inter_peak_interval_entropy: Entropy of inter-peak intervals.
            - peak_time_entropy: Entropy of peak start times normalized by trace length.
            - histogram_peak_amplitude: Histogram of amplitudes (bins and counts).
            - histogram_peak_duration: Histogram of durations (bins and counts).
            - peak_symmetry_score: Mean of symmetry scores (rise vs decay).
            - peak_density_center_of_mass: Center of mass of peak times, normalized.
            - burst_frequency_evolution: Evolution of burst frequency over sliding windows.
            - activity_fraction_evolution: Evolution of active fraction over sliding windows.
        """
        """Compute statistics based on the binary trace and detected peaks."""
        binary = np.array(self.binary)
        peaks = self.peaks
        self.metadata = {}

        # Global statistics
        self.metadata["fraction_active_time"] = float(np.sum(binary)) / len(binary) if len(binary) > 0 else 0.0
        self.metadata["num_peaks"] = len(peaks)
        self.metadata["burst_frequency"] = self.metadata["num_peaks"] / len(binary) if len(binary) > 0 else 0.0
        self.metadata["total_active_frames"] = int(np.sum(binary))

        # Peak feature distributions
        durations = [p.duration for p in peaks]
        amplitudes = [p.height for p in peaks]
        prominences = [p.prominence for p in peaks]
        rise_times = [p.rise_time for p in peaks]
        decay_times = [p.decay_time for p in peaks]
        start_times = [p.start_time for p in peaks]

        intervals = np.diff(start_times)

        self.metadata.update({
            "mean_peak_duration": float(np.mean(durations)) if durations else None,
            "std_peak_duration": float(np.std(durations)) if durations else None,
            "mean_inter_peak_interval": float(np.mean(intervals)) if len(intervals) > 0 else None,
            "std_inter_peak_interval": float(np.std(intervals)) if len(intervals) > 0 else None,
            "mean_peak_amplitude": float(np.mean(amplitudes)) if amplitudes else None,
            "std_peak_amplitude": float(np.std(amplitudes)) if amplitudes else None,
            "mean_peak_prominence": float(np.mean(prominences)) if prominences else None,
            "std_peak_prominence": float(np.std(prominences)) if prominences else None,
            "coefficient_of_variation_prominence": float(np.std(prominences) / np.mean(prominences)) if prominences and np.mean(prominences) != 0 else None,
            "mean_rise_time": float(np.mean(rise_times)) if rise_times else None,
            "mean_decay_time": float(np.mean(decay_times)) if decay_times else None,
            "amplitude_skewness": float(skew(amplitudes)) if len(amplitudes) > 2 else None,
            "duration_skewness": float(skew(durations)) if len(durations) > 2 else None,
            "coefficient_of_variation_amplitude": float(np.std(amplitudes) / np.mean(amplitudes)) if amplitudes and np.mean(amplitudes) != 0 else None,
            "coefficient_of_variation_duration": float(np.std(durations) / np.mean(durations)) if durations and np.mean(durations) != 0 else None
        })

        # Periodicity
        cv_ipi = np.std(intervals) / np.mean(intervals) if len(intervals) > 1 and np.mean(intervals) != 0 else None
        self.metadata["periodicity_score"] = 1 / (1 + cv_ipi) if cv_ipi is not None else None

        # Entropy measures
        def compute_entropy(data, bins=10):
            if len(data) < 2:
                return None
            hist, _ = np.histogram(data, bins=bins, density=True)
            hist = hist[hist > 0]
            return float(entropy(hist)) if len(hist) > 1 else 0.0

        self.metadata["inter_peak_interval_entropy"] = compute_entropy(intervals)

        if start_times:
            norm_times = np.array(start_times) / len(binary)
            self.metadata["peak_time_entropy"] = compute_entropy(norm_times)

        # Histograms
        if amplitudes:
            counts, bins = np.histogram(amplitudes, bins=10)
            self.metadata["histogram_peak_amplitude"] = {"bins": bins.tolist(), "counts": counts.tolist()}

        if durations:
            counts, bins = np.histogram(durations, bins=10)
            self.metadata["histogram_peak_duration"] = {"bins": bins.tolist(), "counts": counts.tolist()}

        # Symmetry
        symmetry_scores = [1 - abs(p.rise_time - p.decay_time) / (p.rise_time + p.decay_time)
                           for p in peaks if (p.rise_time + p.decay_time) > 0]
        self.metadata["peak_symmetry_score"] = float(np.mean(symmetry_scores)) if symmetry_scores else None

        # Center of mass
        if start_times:
            weights = np.ones(len(start_times))
            com = float(np.average(start_times, weights=weights))
            self.metadata["peak_density_center_of_mass"] = com / len(binary)

        # Sliding window dynamic features
        freq_evo = []
        act_evo = []
        window_size = 200
        step_size = 50
        for start in range(0, len(binary) - window_size + 1, step_size):
            end = start + window_size
            bin_window = binary[start:end]
            freq = np.sum([(p.peak_time >= start and p.peak_time < end) for p in peaks])
            freq_evo.append(freq / window_size)
            act_evo.append(np.sum(bin_window) / window_size)

        self.metadata["burst_frequency_evolution"] = freq_evo
        self.metadata["activity_fraction_evolution"] = act_evo


    def plot_metadata(self, save_path: Optional[Path] = None) -> None:
        """
        Visualize scalar metadata, histograms, and time evolution of the trace.

        Args:
            save_path (Optional[Path]): Path to save the figure. If None, shows interactively.
        """
        if not self.metadata:
            raise ValueError("No metadata found. Call `compute_metadata()` first.")

        fig, axs = plt.subplots(3, 2, figsize=(12, 10))
        axs = axs.flatten()

        # Panel 1: Table of scalar metadata
        scalar_keys = [
            "fraction_active_time", "num_peaks", "burst_frequency",
            "mean_peak_duration", "mean_inter_peak_interval",
            "mean_peak_amplitude", "mean_peak_prominence",
            "periodicity_score", "inter_peak_interval_entropy",
            "peak_time_entropy", "peak_density_center_of_mass",
            "peak_symmetry_score"
        ]
        scalars = [(k, f"{self.metadata[k]:.3f}" if self.metadata[k] is not None else "None") for k in scalar_keys if k in self.metadata]
        table = axs[0].table(cellText=scalars, colLabels=["Metric", "Value"], loc="center")
        axs[0].axis("off")
        axs[0].set_title("Global Metadata Summary")

        # Panel 2: Histogram amplitude
        if "histogram_peak_amplitude" in self.metadata:
            data = self.metadata["histogram_peak_amplitude"]
            axs[1].bar(data["bins"][:-1], data["counts"], width=np.diff(data["bins"]), align="edge")
            axs[1].set_title("Peak Amplitude Histogram")
            axs[1].set_xlabel("Amplitude")
            axs[1].set_ylabel("Count")

        # Panel 3: Histogram duration
        if "histogram_peak_duration" in self.metadata:
            data = self.metadata["histogram_peak_duration"]
            axs[2].bar(data["bins"][:-1], data["counts"], width=np.diff(data["bins"]), align="edge")
            axs[2].set_title("Peak Duration Histogram")
            axs[2].set_xlabel("Duration")
            axs[2].set_ylabel("Count")

        # Panel 4: Burst frequency evolution
        if "burst_frequency_evolution" in self.metadata:
            axs[3].plot(self.metadata["burst_frequency_evolution"])
            axs[3].set_title("Burst Frequency Evolution")
            axs[3].set_xlabel("Window index")
            axs[3].set_ylabel("Freq / window")

        # Panel 5: Activity fraction evolution
        if "activity_fraction_evolution" in self.metadata:
            axs[4].plot(self.metadata["activity_fraction_evolution"])
            axs[4].set_title("Active Fraction Evolution")
            axs[4].set_xlabel("Window index")
            axs[4].set_ylabel("Fraction active")

        for ax in axs:
            ax.grid(True)

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()


    def plot_raw_trace(self, save_path: Optional[Path] = None):
        """Plot or save the raw trace.
        
        Args:
            save_path (Optional[Path]): If provided, the figure is saved instead of shown.
        """
        if len(self.raw) > 0:
            plt.figure()
            plt.plot(self.raw)
            plt.title("Raw Trace")
            plt.xlabel("Time")
            plt.ylabel("Intensity")
            if save_path:
                plt.savefig(save_path, dpi=300)
                plt.close()
            else:
                plt.show()

    def plot_version_trace(self, version: str, save_path: Optional[Path] = None):
        """Plot or save a specific version of the trace.

        Args:
            version (str): Name of the version to plot.
            save_path (Optional[Path]): If provided, the figure is saved instead of shown.
        """
        trace = self.versions.get(version, [])
        if len(trace) > 0:
            plt.figure()
            plt.plot(trace)
            plt.title(f"Trace - Version: {version}")
            plt.xlabel("Time")
            plt.ylabel("Intensity")
            if save_path:
                plt.savefig(save_path, dpi=300)
                plt.close()
            else:
                plt.show()

    def plot_binary_trace(self, save_path: Optional[Path] = None):
        """Plot or save the binary trace.
        
        Args:  
            save_path (Optional[Path]): If provided, the figure is saved instead of shown.
        """
        if len(self.binary) > 0:
            plt.figure()
            plt.plot(self.binary, drawstyle='steps-post')
            plt.title("Binary Trace")
            plt.xlabel("Time")
            plt.ylabel("Binary")
            if save_path:
                plt.savefig(save_path, dpi=300)
                plt.close()
            else:
                plt.show()

    def plot_peaks_over_trace(self, save_path: Optional[Path] = None):
        """Plot or save the active trace with detected peaks overlayed.
        
        Args:
            save_path (Optional[Path]): If provided, the figure is saved instead of shown.
        """
        trace = self.active_trace
        if len(trace) == 0:
            return

        plt.figure()
        plt.plot(trace, label=f"Trace: {self.default_version}")
        for peak in self.peaks:
            plt.axvspan(peak.start_time, peak.end_time, color='red', alpha=0.3)
        plt.title("Peaks Over Trace")
        plt.xlabel("Time")
        plt.ylabel("Intensity")
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_all_traces(self, save_path: Optional[Path] = None) -> None:
        """Plot or save all traces: raw, versions, binary, and peak overlay.

        Args:
            save_path (Optional[Path]): If provided, the full trace summary figure is saved.
        """
        num_plots = 0
        if len(self.raw) > 0:
            num_plots += 1
        num_plots += len(self.versions)
        if len(self.binary) > 0:
            num_plots += 1
        if len(self.peaks) > 0:
            num_plots += 1

        fig, axs = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots))

        axs = np.atleast_1d(axs).flatten()

        idx = 0
        if len(self.raw) > 0:
            axs[idx].plot(self.raw)
            axs[idx].set_title("Raw Trace")
            axs[idx].set_ylabel("Intensity")
            idx += 1

        for version_name, trace in self.versions.items():
            axs[idx].plot(trace)
            axs[idx].set_title(f"Trace Version: {version_name}")
            axs[idx].set_ylabel("Intensity")
            idx += 1

        if len(self.binary) > 0:
            axs[idx].step(range(len(self.binary)), self.binary, where='post')
            axs[idx].set_title("Binary Trace")
            axs[idx].set_ylabel("0/1")
            idx += 1

        if len(self.peaks) > 0:
            active_trace = self.active_trace
            axs[idx].plot(active_trace, label="Active Trace")
            for peak in self.peaks:
                axs[idx].axvspan(peak.start_time, peak.end_time, color='red', alpha=0.3)
            axs[idx].set_title("Peaks Overlay")
            axs[idx].legend()
            axs[idx].set_ylabel("Intensity")
            idx += 1

        for ax in axs:
            ax.set_xlabel("Time")
            ax.grid(True)

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    def __repr__(self) -> str:
        return f"<Trace default='{self.default_version}', peaks={len(self.peaks)}, active={self.metadata.get('fraction_active_time', 0):.2f}>"
