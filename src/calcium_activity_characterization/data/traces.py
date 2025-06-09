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
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from calcium_activity_characterization.processing.signal_processing import SignalProcessor
from calcium_activity_characterization.data.peaks import PeakDetector

from calcium_activity_characterization.utilities.metrics import compute_histogram_func, compute_peak_frequency_over_time

if TYPE_CHECKING:
    from calcium_activity_characterization.data.peaks import Peak

import logging
logger = logging.getLogger(__name__)
class Trace:
    """
    A container for calcium activity trace and analysis results.

    This class manages multiple versions of the trace (e.g., raw, smoothed, normalized),
    and supports peak detection, binarization, and metadata computation based on a selected version.

    Attributes:
        versions (Dict[str, List[float]]): Stores multiple processed versions of the trace.
        default_version (str): Key to access the active trace.
        binary (List[int]): Binarized version of the trace based on detected peaks.
        peaks (List[Peak]): Detected peaks in the active trace.
        metadata (Dict[str, Any]): Analysis metadata derived from the trace.
    """

    def __init__(self, raw_trace: Optional[List[float]] = None) -> None:
        """
        Initialize a Trace object to store calcium signal versions and analysis results.

        Args:
            raw_trace (Optional[List[float]]): Optional list of raw intensity values.
        """
        self.versions: Dict[str, List[float]] = {}
        if raw_trace is not None:
            self.versions["raw"] = raw_trace
        
        self.default_version: str = "raw"
        self.binary: List[int] = []
        self.peaks: List["Peak"] = []
        self.metadata: Dict[str, Any] = {}

    @property
    def active_trace(self) -> List[float]:
        """Returns the currently selected version of the trace."""
        return self.versions.get(self.default_version, [])
    
    def add_trace(self, trace: List[float], version_name: str) -> None:
        """
        Add a new trace version to the versions dictionary.

        Args:
            trace (List[float]): The trace data to add.
            version_name (str): The key under which to store the trace.
        """
        if not isinstance(trace, list):
            raise TypeError("trace must be a list of floats.")
        if not isinstance(version_name, str):
            raise TypeError("version_name must be a string.")
        self.versions[version_name] = trace

    def process_trace(self, input_trace_name: str, output_trace_name: str, processing_params: dict) -> None:
        """
        Apply a SignalProcessor to a given trace and store the result as a new version.

        Args:
            input_trace_name (str): Name of the input trace to process.
            output_trace_name (str): Name for the processed trace version.
            config (dict): Configuration dictionary containing processor parameters.

        Returns:
            None
        """
        try:
            if input_trace_name not in self.versions:
                raise ValueError(f"Trace version '{input_trace_name}' not found in self.versions.")

            input_trace = self.versions[input_trace_name]
            processor = SignalProcessor(params=processing_params)
            processed = processor.run(input_trace)

            self.versions[output_trace_name] = processed
        except Exception as e:
            logger.error(f"Failed to process trace from '{input_trace_name}' to '{output_trace_name}': {e}")
            raise

    def detect_peaks(self, detector_params: dict = None) -> None:
        """
        Detect peaks in the active trace using the provided detector parameters.

        Args:
            detector_params (dict): Dictionary of parameters for the peak detection algorithm.
        """
        if detector_params is None:
            logger.error("No detector parameters provided.")
            return
        
        detector = PeakDetector(params=detector_params)
        trace = self.active_trace
        self.peaks = detector.run(trace) if len(trace) > 0 else []
        self._refine_peaks_duration(self.default_version)

    def binarize_trace_from_peaks(self) -> None:
        """Convert detected peaks into a binary 0/1 trace and compute metadata."""
        trace = self.active_trace
        if trace is None or len(trace) == 0:
            self.binary = []
            return

        trace_length = len(trace)
        binary = np.zeros(trace_length, dtype=int)
        for peak in self.peaks:
            start = max(0, peak.rel_start_time)
            end = min(trace_length, peak.rel_end_time + 1)
            binary[start:end] = 1

        self.binary = binary.tolist()
        self.compute_metadata()

    def compute_metadata(self) -> None:
        """Compute statistics based on the binary trace and detected peaks.
        
        This includes global statistics, peak feature distributions, periodicity, entropy measures,
        histograms, symmetry, density center, and dynamic evolution features. The results are stored
        in the `metadata` attribute as a dictionary.

        The metadata includes:
            - num_peaks: Total number of detected peaks.
            - std_peak_duration: Standard deviation of peak durations.
            - std_ipi: Standard deviation of inter-peak intervals (IPIs).
            - std_peak_amplitude: Standard deviation of peak amplitudes.
            - std_peak_prominence: Standard deviation of peak prominences.
            - std_peak_symmetry_score: Standard deviation of symmetry scores (rise vs decay).

            - coefficient_of_variation_prominence: CV of peak prominences.
            - coefficient_of_variation_amplitude: CV of peak amplitudes.
            - coefficient_of_variation_duration: CV of peak durations.

            - periodicity_score: Score indicating periodicity based on inter-peak intervals.

            - entropy_ipi: Entropy of inter-peak intervals. Measures variability in time between peaks.
            - entropy_peak_time: Entropy of peak start times normalized by trace length. Measures how evenly peaks occur over time.
            
            - peak_frequency: Frequency of peaks in the trace.
            - peak_frequency_over_time: Evolution of peak frequency over sliding windows.

            Peak distributions analysis:
            - histogram_peak_amplitude: Histogram of amplitudes (bins and counts).
            - amplitude_skewness: Skewness of peak amplitudes. Skewness quantifies the asymmetry of the distribution.

            - histogram_peak_duration: Histogram of durations (bins and counts).
            - duration_skewness: Skewness of peak durations. Skewness quantifies the asymmetry of the distribution.

        """
        binary = np.array(self.binary)
        peaks = self.peaks
        self.metadata = {}

        # Global statistics
        self.metadata["num_peaks"] = len(peaks)
        self.metadata["peak_frequency"] = self.metadata["num_peaks"] / len(binary) if len(binary) > 0 else 0.0

        # Peak feature distributions
        durations = [p.rel_duration for p in peaks]
        amplitudes = [p.height for p in peaks]
        prominences = [p.prominence for p in peaks]
        start_times = [p.rel_start_time for p in peaks]
        symmetry_scores = [p.rel_symmetry_score for p in peaks]

        intervals = np.diff(start_times)

        self.metadata.update({
            "mean_peak_duration": float(np.mean(durations)) if durations else None,
            "std_peak_duration": float(np.std(durations)) if durations else None,
            "mean_ipi": float(np.mean(intervals)) if len(intervals) > 0 else None,
            "std_ipi": float(np.std(intervals)) if len(intervals) > 0 else None,
            "mean_peak_amplitude": float(np.mean(amplitudes)) if amplitudes else None,
            "std_peak_amplitude": float(np.std(amplitudes)) if amplitudes else None,
            "mean_peak_prominence": float(np.mean(prominences)) if prominences else None,
            "std_peak_prominence": float(np.std(prominences)) if prominences else None,
            "mean_peak_symmetry_score": float(np.mean(symmetry_scores)) if symmetry_scores else None,
            "std_peak_symmetry_score": float(np.std(symmetry_scores)) if symmetry_scores else None,
            "coefficient_of_variation_prominence": float(np.std(prominences) / np.mean(prominences)) if prominences and np.mean(prominences) != 0 else None,
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

        self.metadata["entropy_ipi"] = compute_entropy(intervals)

        if start_times:
            norm_times = np.array(start_times) / len(binary)
            self.metadata["entropy_peak_time"] = compute_entropy(norm_times)

        # Histograms
        if amplitudes:
            self.metadata["histogram_peak_amplitude"] = compute_histogram_func(amplitudes, bin_count=20)

        if durations:
            self.metadata["histogram_peak_amplitude"] = compute_histogram_func(amplitudes, bin_width=10)

        if symmetry_scores:
            self.metadata["histogram_peak_amplitude"] = compute_histogram_func(amplitudes, bin_count=20)

        # Peak frequency over time
        window_size = 200
        step_size = 50
        self.metadata["peak_frequency_over_time"] = compute_peak_frequency_over_time(
            start_times_per_cell=[start_times],
            trace_length=len(binary),
            window_size=window_size,
            step_size=step_size
        )


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
            "fraction_active_time", "num_peaks", "peak_frequency",
            "mean_peak_duration", "mean_ipi",
            "mean_peak_amplitude", "mean_peak_prominence",
            "periodicity_score", "entropy_ipi",
            "entropy_peak_time", "peak_density_center_of_mass",
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

        # Panel 3: Histogram rel_duration
        if "histogram_peak_duration" in self.metadata:
            data = self.metadata["histogram_peak_duration"]
            axs[2].bar(data["bins"][:-1], data["counts"], width=np.diff(data["bins"]), align="edge")
            axs[2].set_title("Peak Duration Histogram")
            axs[2].set_xlabel("Duration")
            axs[2].set_ylabel("Count")

        # Panel 4: peak frequency evolution
        if "peak_frequency_over_time" in self.metadata:
            axs[3].plot(self.metadata["peak_frequency_over_time"])
            axs[3].set_title("peak Frequency Evolution")
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


    def plot_trace(self, version: str, save_path: Optional[Path] = None):
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
            plt.axvspan(peak.rel_start_time, peak.rel_end_time, color='red', alpha=0.3)
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
        num_plots += len(self.versions)
        if len(self.binary) > 0:
            num_plots += 1
        if len(self.peaks) > 0:
            num_plots += 2

        fig, axs = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots))

        axs = np.atleast_1d(axs).flatten()

        idx = 0

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
                axs[idx].axvspan(peak.rel_start_time, peak.rel_end_time, color='red', alpha=0.3)
            axs[idx].set_title("Rel Peaks Overlay")
            axs[idx].legend()
            axs[idx].set_ylabel("Intensity")
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


    def _refine_peaks_duration(self, version: str = "raw") -> None:
        """
        Refine the start and end times of all detected peaks if there is a local minima closest to peak time.    

        Args:
            version (str): The version of the trace to use for
        """
        for peak in self.peaks:
            left, right = find_valley_bounds(self.versions[version], peak.rel_start_time, peak.rel_end_time)
            peak.start_time = max(left, peak.start_time)
            peak.end_time = min(right, peak.end_time)
            peak.duration = peak.end_time - peak.start_time


    def get_peak_starting_at(self, frame: int) -> Optional["Peak"]:
        """
        Return the first peak whose rel_start_time matches the given frame.

        Args:
            frame (int): Frame index to match.

        Returns:
            Optional[Peak]: The matching Peak object, or None if not found.
        """
        for peak in self.peaks:
            if peak.rel_start_time == frame:
                return peak
        return None


def find_valley_bounds(trace: np.ndarray, rel_start_time: int, rel_end_time: int, max_search: int = 100, window: int = 5) -> tuple[int, int]:
    """
    Refine peak boundaries based on valley detection using windowed minima and derivative sign changes.

    Args:
        trace (np.ndarray): Smoothed 1D signal.
        peak_time (int): Index of the peak center.
        max_search (int): Max points to search on each side.
        window (int): Half-window size to validate local minima.

    Returns:
        (start_index, end_index): Refined left/right bounds of the peak.
    """
    trace = np.asarray(trace, dtype=float)

    n = len(trace)
    left_bound = rel_start_time
    right_bound = rel_end_time

    # --- Left side ---
    for i in range(rel_start_time - 1, max(rel_start_time - max_search - 1, 0), -1):
        window_vals = trace[max(i - window,0): min(i + window + 1,n-1)]
        center_val = trace[i]
        if np.all(center_val <= window_vals):
            left_bound = i
            break

    # --- Right side ---
    for i in range(rel_end_time + 1, min(rel_end_time + max_search + 1,n-1), 1):
        window_vals = trace[max(i - window,0): min(i + window + 1,n-1)]
        center_val = trace[i]
        if np.all(center_val <= window_vals):
            right_bound = i
            break

    return left_bound, right_bound