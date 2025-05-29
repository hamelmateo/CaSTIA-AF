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
        
        This includes:
            - Fraction of time the trace is active (binary = 1)
            - Number of detected peaks
            - Burst frequency (peaks per time unit)
            - Mean and standard deviation of peak durations
            - Mean and standard deviation of inter-peak intervals
            - Periodicity score based on coefficient of variation of inter-peak intervals
        """
        n = len(self.binary)
        num_peaks = len(self.peaks)

        fraction_active_time = sum(self.binary) / n if n else 0.0
        burst_frequency = num_peaks / n if n else 0.0

        durations = [p.duration for p in self.peaks]
        mean_dur = np.mean(durations) if durations else None
        std_dur = np.std(durations) if durations else None

        start_times = sorted(p.start_time for p in self.peaks)
        intervals = np.diff(start_times)
        mean_ipi = np.mean(intervals) if len(intervals) >= 1 else None
        std_ipi = np.std(intervals) if len(intervals) >= 1 else None

        if len(intervals) >= 2 and np.mean(intervals) > 0:
            cv_ipi = np.std(intervals) / np.mean(intervals)
            periodicity_score = 1 / (1 + cv_ipi)
        else:
            periodicity_score = None

        self.metadata = {
            "fraction_active_time": fraction_active_time,
            "num_peaks": num_peaks,
            "burst_frequency": burst_frequency,
            "mean_peak_duration": mean_dur,
            "std_peak_duration": std_dur,
            "mean_inter_peak_interval": mean_ipi,
            "std_inter_peak_interval": std_ipi,
            "periodicity_score": periodicity_score
        }

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
        fig, axs = plt.subplots(2 + len(self.versions), 1, figsize=(10, 3 * (2 + len(self.versions))))
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
