# local_minima_detrender.py
# Usage Example:
#   >>> detrender = LocalMinimaDetrender(config, trace)
#   >>> detrended = detrender.run()
#   >>> versions = detrender.get_intermediate_versions()

from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path
import logging
from scipy.signal import argrelmin
from math import atan2, degrees

from calcium_activity_characterization.utilities.plotter import (
    plot_minima_diagnostics,
    plot_final_baseline_fit,
)
from calcium_activity_characterization.config.presets import LocalMinimaParams

logger = logging.getLogger(__name__)


class LocalMinimaDetrender:
    """
    Applies a local minima-based detrending algorithm to a calcium trace using filtered anchor points
    and a piecewise linear baseline fit.

    Args:
        config (LocalMinimaParams): Configuration parameters for minima detection, filtering, and fitting.
        trace (np.ndarray): Smoothed intensity trace.
    """

    def __init__(self, config: LocalMinimaParams, trace: np.ndarray) -> None:
        self.config = config
        self.verbose = config.verbose
        self.trace = np.asarray(trace, dtype=np.float32)
        self.trace_versions: Dict[str, np.ndarray] = {}
        self.anchor_indices: List[int] = []
        self.discarded_minima: Dict[str, List[int]] = {}
        self.inserted_anchors: List[int] = []

    def run(self) -> np.ndarray:
        """
        Execute the full baseline detection and detrending process.

        Returns:
            np.ndarray: The full-length detrended trace.
        """
        self._detect_local_minima()
        self._add_edge_minima()
        self._filter_minima()
        self._correct_crossings()
        self._fit_linear_baseline()
        detrended = self._subtract_and_clip()
        if self.config.diagnostics_enabled:
            self._plot_diagnostics()
        return detrended

    def get_intermediate_versions(self) -> Dict[str, np.ndarray]:
        """
        Return intermediate trace versions for debugging or plotting.

        Returns:
            Dict[str, np.ndarray]: Dictionary of trace versions.
        """
        return self.trace_versions

    def _detect_local_minima(self) -> None:
        """
        Detect local minima in the trace using relative minima detection.
        This method uses the `argrelmin` function from `scipy.signal` to find local minima
        based on the specified order in the configuration.
        """
        try:
            order = self.config.minima_detection_order
            minima = argrelmin(self.trace, order=order, mode="clip")[0]
            self.anchor_indices = sorted(minima.tolist())
            if self.verbose:
                logger.info(f"Detected {len(self.anchor_indices)} local minima with order={order}.")
        except Exception as e:
            logger.error(f"Failed to detect local minima: {e}")
            self.anchor_indices = []

    def _filter_minima(self) -> None:
        """
        Apply filtering steps to discard shoulder-like and angle-defined valley minima.
        This method uses on-peak local minimas rejection and angle-based filtering to refine the anchor points.
        """
        try:
            shoulder_dist = self.config.filtering_shoulder_neighbor_dist
            window = self.config.filtering_shoulder_window
            self.anchor_indices, discarded1 = self._filter_by_shoulder_rejection_iterative(neighbor_dist=shoulder_dist, window=window)
            if self.verbose:
                logger.info(f"After shoulder rejection: {len(self.anchor_indices)} kept, {len(discarded1)} discarded")

            angle_thresh = self.config.filtering_angle_thresh_deg
            self.anchor_indices, discarded2 = self._filter_by_angle_valley(angle_thresh_deg=angle_thresh)
            if self.verbose:
                logger.info(f"After angle filtering: {len(self.anchor_indices)} kept, {len(discarded2)} discarded")

            self.discarded_minima = {
                "shoulder": discarded1,
                "angle": discarded2
            }
        except Exception as e:
            logger.error(f"Failed to filter minima: {e}")
            self.discarded_minima = {"shoulder": [], "angle": []}

    def _add_edge_minima(self) -> None:
        """
        Add anchors near start and end of trace if they are local minima.
        This method checks the first and last segments of the trace to find minima that are lower than
        the existing anchors, ensuring that edge cases are captured.
        """
        try:
            window = self.config.edge_anchors_window
            delta = self.config.edge_anchors_delta

            before = set(sorted(self.anchor_indices))

            # Start
            if self.anchor_indices:
                segment = self.trace[:min(window, self.anchor_indices[0])]
                if len(segment) > 0:
                    idx = int(np.argmin(segment))
                    if self.trace[idx] <= self.trace[self.anchor_indices[0]] * (1 - delta):
                        self.anchor_indices.append(idx)

            # End
            end_start = len(self.trace) - window
            if self.anchor_indices:
                segment = self.trace[max(end_start, self.anchor_indices[-1]+1):]
                if len(segment) > 0:
                    idx = int(np.argmin(segment) + max(end_start, self.anchor_indices[-1]+1))
                    if self.trace[idx] <= self.trace[self.anchor_indices[-1]] * (1 + delta):
                        self.anchor_indices.append(idx)

            after = set(sorted(self.anchor_indices))
            self.inserted_anchors = sorted(list(after - before))
            self.anchor_indices = sorted(self.anchor_indices)

            if self.verbose:
                logger.info(f"Inserted {len(self.inserted_anchors)} edge anchor(s). Total anchors: {len(self.anchor_indices)}")
        except Exception as e:
            logger.error(f"Failed to add edge minima: {e}")

    def _correct_crossings(self) -> None:
        """
        Insert anchor points between overshooting linear segments.
        This method iteratively checks segments between existing anchors and inserts new anchors
        where the linear interpolation overshoots the trace, ensuring that crossings are corrected
        without introducing anchors too close to existing ones.
        """
        try:
            min_dist = self.config.crossing_correction_min_dist
            max_iter = self.config.crossing_correction_max_iterations

            anchor_idx = sorted(self.anchor_indices)
            inserted = []

            for _ in range(max_iter):
                new_points = []
                for i in range(len(anchor_idx) - 1):
                    a, b = anchor_idx[i], anchor_idx[i + 1]
                    seg_x = np.arange(a, b + 1)
                    interp = np.interp(seg_x, [a, b], [self.trace[a], self.trace[b]])
                    residual = interp - self.trace[a:b + 1]
                    if np.any(residual > 0):
                        idx = a + int(np.argmax(residual))
                        if all(abs(idx - x) > min_dist for x in anchor_idx + new_points):
                            new_points.append(idx)

                if not new_points:
                    break
                inserted.extend(new_points)
                anchor_idx.extend(new_points)
                anchor_idx = sorted(set(anchor_idx))

            self.inserted_anchors.extend(inserted)
            self.anchor_indices = sorted(anchor_idx)
            if self.verbose:
                logger.info(f"Corrected crossings. Inserted {len(inserted)} anchor(s). Total anchors: {len(self.anchor_indices)}")
        except Exception as e:
            logger.error(f"Failed to correct baseline crossings: {e}")

    def _fit_linear_baseline(self) -> None:
        """
        Fit a piecewise linear baseline using the anchor points.
        This method uses linear interpolation to create a baseline from the detected anchor points.
        The baseline is stored in the trace_versions dictionary for later use.
        """
        try:
            if len(self.anchor_indices) < 2:
                raise ValueError("At least two anchor points are required to fit a baseline.")

            x = np.array(self.anchor_indices)
            y = self.trace[x]
            t = np.arange(len(self.trace))
            baseline = np.interp(t, x, y)
            self.trace_versions["baseline"] = baseline

            if self.verbose:
                logger.info("Baseline successfully fitted and stored.")
        except Exception as e:
            logger.error(f"Failed to fit linear baseline: {e}")

    def _subtract_and_clip(self) -> np.ndarray:
        """
        Subtract the baseline from the trace and clip negative values to zero.
        This method computes the residual trace by subtracting the baseline from the original trace,
        and then clips any negative values to zero to ensure the detrended trace is non-negative.
        """
        try:
            baseline = self.trace_versions.get("baseline")
            if baseline is None:
                raise ValueError("Baseline has not been computed.")

            residual = self.trace - baseline

            detrended = np.maximum(residual, 0)
            self.trace_versions["detrended_clipped"] = detrended

            if self.verbose:
                logger.info("Detrended trace computed and stored.")
            return detrended
        except Exception as e:
            logger.error(f"Failed to subtract and clip baseline: {e}")
            return np.zeros_like(self.trace)

    def _plot_diagnostics(self) -> None:
        """
        Plot diagnostics if enabled in the configuration.
        This method checks the configuration for diagnostics settings and plots the minima diagnostics
        and final baseline fit if enabled. It saves the plots to the specified output directory.
        """
        try:
            output_dir = Path(self.config.diagnostics_output_dir)
            if output_dir is None:
                logger.warning("Diagnostics enabled but no output_dir provided.")
                return

            discarded = self.discarded_minima
            plot_minima_diagnostics(
                self.trace,
                anchor_idx=self.anchor_indices,
                inserted_idx=self.inserted_anchors,
                discarded1=discarded.get("shoulder", []),
                discarded2=discarded.get("angle", []),
                discarded3=[],
                output_dir=output_dir
            )

            plot_final_baseline_fit(
                trace=self.trace,
                baseline=self.trace_versions["baseline"],
                anchor_idx=self.anchor_indices,
                detrended=self.trace_versions["detrended_clipped"],
                label="full",
                output_dir=output_dir,
                model_name="linear"
            )
        except Exception as e:
            logger.error(f"Failed to plot diagnostics: {e}")

    # Internal helpers (inlined from baseline_utils)
    def _filter_by_shoulder_rejection_iterative(self, neighbor_dist: int, window: int) -> Tuple[List[int], List[int]]:
        """
        Discard minima higher than both neighbors within given distance.

        Returns:
            Tuple[List[int], List[int]]: (retained, discarded)
        """
        minima = sorted(self.anchor_indices)
        discarded_total = []
        while True:
            filtered = []
            discarded = []
            for i, m in enumerate(minima):
                left = minima[i - 1] if i > 0 and abs(minima[i - 1] - m) <= neighbor_dist else None
                right = minima[i + 1] if i < len(minima) - 1 and abs(minima[i + 1] - m) <= neighbor_dist else None
                if left is not None and right is not None and self.trace[m] > self.trace[left] and self.trace[m] > self.trace[right]:
                    discarded.append(m)
                    continue
                if left is not None and self.trace[m] > self.trace[left] and m > (len(self.trace) - window) and i == len(minima) - 1:
                    discarded.append(m)
                    continue

                filtered.append(m)
            if not discarded:
                break
            discarded_total.extend(discarded)
            minima = filtered
        return minima, discarded_total

    def _filter_by_angle_valley(self, angle_thresh_deg: float) -> Tuple[List[int], List[int]]:
        """
        Discard middle minima with sharp valley angle using atan2 geometry.

        Args:
            minima (List[int]): List of anchor indices to filter.
            angle_thresh_deg (float): Angle threshold for filtering.

        Returns:
            Tuple[List[int], List[int]]: (retained, discarded)
        """
        minima = sorted(self.anchor_indices)
        discarded_total = []
        while True:
            filtered = []
            discarded = []
            for i in range(1, len(minima) - 1):
                left, mid, right = minima[i - 1], minima[i], minima[i + 1]
                dx1, dy1 = mid - left, self.trace[mid] - self.trace[left]
                dx2, dy2 = right - mid, self.trace[right] - self.trace[mid]
                angle1 = atan2(dy1, dx1)
                angle2 = atan2(dy2, dx2)
                angle_diff = abs(degrees(angle2 - angle1))
                angle_diff = min(angle_diff, 360 - angle_diff)
                cross = dx1 * dy2 - dy1 * dx2
                if degrees(angle1) > 30 and degrees(angle2) > 30:
                    filtered.append(mid)
                    continue
                if cross < 0 and angle_diff > angle_thresh_deg:
                    discarded.append(mid)
                else:
                    filtered.append(mid)
            if not discarded:
                break
            discarded_total.extend(discarded)
            minima = [minima[0]] + filtered + [minima[-1]]
        return minima, discarded_total