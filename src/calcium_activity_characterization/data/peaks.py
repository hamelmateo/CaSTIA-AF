from typing import Optional, Literal
import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths

class Peak:
    """
    Represents a calcium transient peak with timing, intensity, and hierarchical metadata.
    """
    def __init__(
        self,
        id: int,
        start_time: int,
        peak_time: int,
        end_time: int,
        height: float,
        prominence: float,
        width: float,
        group_id: Optional[int] = None,
        parent_peak_id: Optional[int] = None,
        role: Literal["individual", "member", "parent"] = "individual"
    ):
        self.id = id
        self.start_time = start_time
        self.peak_time = peak_time
        self.end_time = end_time
        self.duration = end_time - start_time
        self.height = height
        self.prominence = prominence
        self.width = width

        self.rise_time = peak_time - start_time
        self.decay_time = end_time - peak_time

        self.group_id = group_id
        self.parent_peak_id = parent_peak_id
        self.role = role

        self.scale_class: Optional[str] = None  # e.g., 'minor', 'major', 'super'

    def __repr__(self):
        return (
            f"Peak(id={self.id}, time={self.peak_time}, height={self.height:.2f}, "
            f"prominence={self.prominence:.2f}, role={self.role}, scale={self.scale_class})"
        )


class PeakDetector:
    """
    Base scaffold for detecting peaks from a calcium trace.
    """
    def __init__(self, params: dict):
        """
        Initialize the PeakDetector with parameters.
        
        Args:
            params (dict): Parameters for peak detection.
        """
        self.params = params
        self.method = params.get("method", "skimage")
        self.method_params = params.get("params", {}).get(self.method, {})
        self.scale_class_quantiles = params.get("scale_class_quantiles", [0.33, 0.66])


    def detect(self, trace: list[float]) -> list[Peak]:
        
        trace = np.array(trace, dtype=float)

        if self.method == "skimage":
            peaks, properties = find_peaks(
                trace,
                prominence=self.params.get("prominence", 0.1),
                distance=self.params.get("distance", 5),
                height=self.params.get("height"),
                threshold=self.params.get("threshold"),
                width=self.params.get("width")
            )

            prominences = properties["prominences"]

            # Compute widths
            results_half = peak_widths(trace, peaks, rel_height=0.5)
            widths = results_half[0]
            left_ips = results_half[2]
            right_ips = results_half[3]

            # Step 2: Classify peaks by scale
            if len(prominences) > 0:
                quantiles = np.quantile(prominences, self.params.get("scale_class_quantiles", [0.33, 0.66]))
            else:
                quantiles = [0, 0]

            peak_list = []
            for i, peak_time in enumerate(peaks):
                start_time = int(np.floor(left_ips[i]))
                end_time = int(np.ceil(right_ips[i]))
                prominence = float(prominences[i])
                width = float(widths[i])
                height = float(trace[peak_time])

                # Assign scale class
                if prominence < quantiles[0]:
                    scale_class = "minor"
                elif prominence < quantiles[1]:
                    scale_class = "major"
                else:
                    scale_class = "super"

                peak = Peak(
                    id=i,
                    start_time=start_time,
                    peak_time=peak_time,
                    end_time=end_time,
                    height=height,
                    prominence=prominence,
                    width=width
                )
                peak.scale_class = scale_class
                peak_list.append(peak)

            return peak_list
        else:
            raise NotImplementedError(f"Peak detection method '{self.method}' is not supported yet.")
