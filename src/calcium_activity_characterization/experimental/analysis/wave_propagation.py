# wave_propagation_analysis.py
# Usage Example:
# >>> analyzer = WavePropagationAnalyzer()
# >>> analyzer.analyze_cluster_propagation(cluster)
# >>> analyzer.plot_cluster_trajectory(cluster, overlay_path, save_path)

import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, List
from calcium_activity_characterization.experimental.analysis.clusters import Cluster
from calcium_activity_characterization.data.cells import Cell
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
from tifffile import imread

logger = logging.getLogger(__name__)

class WavePropagationAnalyzer:
    """
    Analyze direction and speed of wave propagation in a Cluster.
    Stores trajectory metadata in the cluster.metadata dictionary and generates trajectory plots.
    """

    def __init__(self):
        pass

    def analyze_cluster_propagation(self, cluster: Cluster) -> None:
        """
        Analyze centroid-based wave propagation for a spatial cluster.

        Args:
            cluster (Cluster): The spatial cluster to analyze.
        """
        try:
            time_to_centroids: Dict[int, List[np.ndarray]] = defaultdict(list)

            for cell, peak_idx in cluster.members:
                peak = cell.trace.peaks[peak_idx]
                time_to_centroids[peak.start_time].append(cell.centroid)

            sorted_times = sorted(time_to_centroids.keys())
            if len(sorted_times) < 2:
                logger.warning(f"Cluster {cluster.id} has too few timepoints for propagation analysis.")
                return

            time_centroids: Dict[int, np.ndarray] = {}
            for t in sorted_times:
                centroids = np.array(time_to_centroids[t])
                time_centroids[t] = np.mean(centroids, axis=0)

            first_t = sorted_times[0]
            last_t = sorted_times[-1]
            origin = time_centroids[first_t]
            destination = time_centroids[last_t]

            delta = destination - origin
            distance = np.linalg.norm(delta)
            duration = last_t - first_t if last_t != first_t else 1
            direction_vector = delta / distance if distance > 0 else np.array([0.0, 0.0])
            angle_deg = np.degrees(np.arctan2(direction_vector[0], direction_vector[1]))
            speed = distance / duration

            cluster.metadata = {
                "origin_centroid": origin.tolist(),
                "end_centroid": destination.tolist(),
                "direction_vector": direction_vector.tolist(),
                "angle_deg": float(angle_deg),
                "speed": float(speed),
                "duration": int(duration),
                "distance": float(distance),
                "time_centroids": {int(t): c.tolist() for t, c in time_centroids.items()}
            }

        except Exception as e:
            logger.error(f"Propagation analysis failed for cluster {cluster.id}: {e}")
            cluster.metadata = {}

    def plot_cluster_trajectory(self, cluster: Cluster, overlay_path: Path, save_path: Path) -> None:
        """
        Plot and save the centroid trajectory over the overlay image.

        Args:
            cluster (Cluster): The cluster with computed metadata.
            overlay_path (Path): Path to the grayscale image.
            save_path (Path): Path to save the output visualization.
        """
        try:

            metadata = cluster.metadata
            if not metadata:
                logger.warning(f"No metadata available for cluster {cluster.id} to plot.")
                return

            overlay = imread(str(overlay_path))
            img_rgb = np.stack([overlay]*3, axis=-1).astype(np.uint8)

            time_centroids = metadata["time_centroids"]
            times = sorted(map(int, time_centroids.keys()))

            cmap = cm.get_cmap("viridis", len(times))

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img_rgb, cmap="gray")

            # Draw centroid path with time-based coloring and labels
            for idx, t in enumerate(times):
                cy, cx = time_centroids[t]
                ax.plot(cx, cy, 'o', color=cmap(idx), label=f"t={t}")
                ax.text(cx + 2, cy, str(t), color=cmap(idx), fontsize=8)
                if idx > 0:
                    prev_cy, prev_cx = time_centroids[times[idx - 1]]
                    ax.arrow(prev_cx, prev_cy, cx - prev_cx, cy - prev_cy,
                             head_width=1.5, head_length=2.5, color=cmap(idx), alpha=0.8)

            # Mark origin and end
            oc_y, oc_x = metadata["origin_centroid"]
            ec_y, ec_x = metadata["end_centroid"]
            ax.plot(oc_x, oc_y, marker="*", markersize=12, color="red", label="Origin")
            ax.plot(ec_x, ec_y, marker="X", markersize=10, color="blue", label="End")

            ax.set_title(f"Cluster {cluster.id} Trajectory\nSpeed: {metadata['speed']:.2f} px/frame | Direction: {metadata['angle_deg']:.1f}°")
            ax.legend(fontsize=7, loc='lower right')
            ax.axis("off")

            fig.tight_layout()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            logger.info(f"Saved cluster trajectory plot to: {save_path}")

        except Exception as e:
            logger.error(f"Failed to plot cluster trajectory for cluster {cluster.id}: {e}")