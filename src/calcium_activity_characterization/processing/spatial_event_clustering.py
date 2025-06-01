# spatial_event_clustering.py
# Usage Example:
# >>> engine = SpatialEventClusteringEngine()
# >>> event_clusters = engine.run(population)
# >>> engine.plot_clusters_with_graph(population, overlay_path, output_dir, population.activity_trace.peaks)

from typing import List, Tuple, Dict
from collections import deque
import networkx as nx
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_rgb
import random

from calcium_activity_characterization.data.clusters import Cluster
from calcium_activity_characterization.data.populations import Population
from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.data.peaks import Peak

import tifffile
import logging
logger = logging.getLogger(__name__)


class SpatialEventClusteringEngine:
    """
    Identify spatially coherent cell clusters active during each global activity peak.

    Attributes:
        event_clusters (List[List[Cluster]]): Clusters grouped by global peak index.
    """

    def __init__(self):
        self.event_clusters: List[List[Cluster]] = []
        self.cluster_id_counter: int = 0

    def run(self, population: Population) -> List[List[Cluster]]:
        """
        Run spatial event clustering on the given population.

        Args:
            population (Population): The cell population to analyze.

        Returns:
            List[List[Cluster]]: Clusters for each global activity peak.
        """
        neighbor_graph = population.neighbor_graph

        for peak_idx, global_peak in enumerate(population.activity_trace.peaks):
            window_start = global_peak.start_time
            window_end = global_peak.end_time

            # Step 1: Find all (cell, peak_index) where peak.start_time in window and not in another cluster
            active_entries: List[Tuple[Cell, int]] = []
            for cell in population.cells:
                for i, peak in enumerate(cell.trace.peaks):
                    if window_start <= peak.start_time < window_end and not peak.in_cluster:
                        active_entries.append((cell, i))

            # Step 2: Build graph of active nodes
            label_to_entry = {cell.label: (cell, i) for cell, i in active_entries}
            active_labels = set(label_to_entry.keys())

            visited = set()
            clusters_for_peak: List[Cluster] = []

            for label in active_labels:
                if label in visited:
                    continue

                # BFS with 1-hop and shared-neighbor expansion
                cluster_labels = set()
                queue = deque([label])

                while queue:
                    current = queue.popleft()
                    if current in visited:
                        continue
                    visited.add(current)
                    cluster_labels.add(current)

                    # Direct neighbors
                    for neighbor in neighbor_graph.neighbors(current):
                        if neighbor in active_labels and neighbor not in visited:
                            queue.append(neighbor)

                    """
                    # Shared neighbors: anyone sharing at least one direct neighbor
                    for neighbor in neighbor_graph.nodes:
                        if neighbor in active_labels and neighbor not in visited:
                            shared = set(neighbor_graph.neighbors(current)) & set(neighbor_graph.neighbors(neighbor))
                            if shared:
                                queue.append(neighbor)
                    """

                # Build cluster if it has >= 1 cell
                if cluster_labels:
                    cluster = Cluster(
                        id=self.cluster_id_counter,
                        start_time=window_start,
                        end_time=window_end
                    )
                    for label in cluster_labels:
                        cell, idx = label_to_entry[label]
                        cluster.add(cell, idx)
                    clusters_for_peak.append(cluster)
                    self.cluster_id_counter += 1

            self.event_clusters.append(clusters_for_peak)

        return self.event_clusters

    def plot_clusters_with_graph(
        self,
        population: Population,
        overlay_path: Path,
        output_dir: Path,
        global_peaks: List[Peak]
    ) -> None:
        """
        Save overlay images showing clusters and their spatial graph for each global peak.

        Args:
            population (Population): The population (for centroids and graph).
            overlay_path (Path): Path to grayscale overlay image.
            output_dir (Path): Directory to save colored overlays.
            global_peaks (List[Peak]): The global peaks used to generate the event clusters.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        overlay = tifffile.imread(str(overlay_path))
        graph = population.neighbor_graph

        for peak_idx, (cluster_list, global_peak) in enumerate(zip(self.event_clusters, global_peaks)):
            color_overlay = np.stack([overlay] * 3, axis=-1).astype(np.uint8)

            # Generate colors
            cmap = cm.get_cmap('hsv', len(cluster_list) + 1)
            cluster_colors: Dict[int, Tuple[int, int, int]] = {
                cluster.id: tuple((np.array(cmap(i)[:3]) * 255).astype(np.uint8))
                for i, cluster in enumerate(cluster_list)
            }

            # Color cells in clusters
            for cluster in cluster_list:
                color = cluster_colors[cluster.id]
                for cell, _ in cluster.members:
                    for y, x in cell.pixel_coords:
                        color_overlay[y, x] = color

            # Plot with graph overlay
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(color_overlay)
            pos = {node: (xy[1], xy[0]) for node, xy in nx.get_node_attributes(graph, 'pos').items()}

            # Determine node color
            node_colors = []
            for node in graph.nodes:
                found = False
                for cluster in cluster_list:
                    if any(cell.label == node for cell, _ in cluster.members):
                        node_colors.append(to_rgb(np.array(cluster_colors[cluster.id]) / 255))
                        found = True
                        break
                if not found:
                    node_colors.append((0.6, 0.6, 0.6))  # Gray for inactive

            nx.draw(graph, pos=pos, ax=ax, node_size=20, node_color=node_colors, edge_color='cyan', width=0.5)

            window_text = f"Peak Window: {global_peak.start_time} - {global_peak.end_time}"
            ax.set_title(f"Spatial Event Clusters - Global Peak #{peak_idx}\n{window_text}")
            ax.axis("off")

            fig.tight_layout()
            save_path = output_dir / f"event_clusters_overlay_peak_{peak_idx:03d}.png"
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            logger.info(f"Saved: {save_path}")
