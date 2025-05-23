from typing import List
from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.data.cluster import Cluster


class PeakClusteringEngine:
    """
    Engine for clustering temporally related peaks across multiple cells.
    """

    def __init__(self, config: dict):
        """
        Initialize the clustering engine using a configuration dictionary.

        Args:
            config (dict): Should contain:
                - "time_window_factor": float
                - "score_weights": dict with keys "time" and "duration"
        """
        self.window_size = config.get("window_size", 25)
        self.time_window_factor = config.get("time_window_factor", 0.25)

        weights = config.get("score_weights", {})
        self.time_weight = weights.get("time", 0.7)
        self.duration_weight = weights.get("duration", 0.3)

        self.clusters: List[Cluster] = []
        self.cluster_id_counter = 0

    def run(self, cells: List[Cell]) -> List[Cluster]:
        """
        Run the clustering algorithm on all peaks of all cells.

        Args:
            cells (List[Cell]): List of active Cell objects with .peaks.

        Returns:
            List[Cluster]: List of detected peak clusters.
        """
        n = len(cells)

        for i in range(n):
            origin_cell = cells[i]
            for origin_peak_idx, origin_peak in enumerate(origin_cell.peaks):
                if origin_peak.in_cluster:
                    continue

                #lag_margin = int(self.time_window_factor * origin_peak.duration)
                lag_margin = int(self.window_size)
                window_start = origin_peak.start_time - lag_margin
                window_end = origin_peak.start_time + lag_margin

                cluster_members = [(origin_cell, origin_peak_idx)]

                for j in range(n):
                    if j == i:
                        continue
                    other_cell = cells[j]

                    best_score = -1
                    best_idx = None

                    for other_idx, other_peak in enumerate(other_cell.peaks):
                        if other_peak.in_cluster:
                            continue
                        if window_start <= other_peak.start_time <= window_end:
                            score = self._compute_score(origin_peak, other_peak)
                            if score > best_score:
                                best_score = score
                                best_idx = other_idx

                    if best_idx is not None:
                        cluster_members.append((other_cell, best_idx))

                if len(cluster_members) > 1:
                    cluster = Cluster(self.cluster_id_counter, window_start, window_end)
                    for cell, idx in cluster_members:
                        cluster.add(cell, idx)
                    self.clusters.append(cluster)
                    self.cluster_id_counter += 1

        return self.clusters

    def _compute_score(self, peak1, peak2) -> float:
        """
        Compute a similarity score between two peaks.

        Args:
            peak1 (Peak)
            peak2 (Peak)

        Returns:
            float: Score between 0 and 1.
        """
        time_diff = abs(peak1.start_time - peak2.start_time)
        duration_diff = abs(peak1.duration - peak2.duration)

        max_time_window = self.time_window_factor * peak1.duration
        if max_time_window == 0:
            return 0.0

        score_time = 1 - (time_diff / max_time_window)
        score_duration = 1 - (duration_diff / max(peak1.duration, peak2.duration, 1))

        return self.time_weight * score_time + self.duration_weight * score_duration
