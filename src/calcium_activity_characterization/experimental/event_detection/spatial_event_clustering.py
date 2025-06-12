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
from matplotlib import colors, cm
import random

from calcium_activity_characterization.experimental.analysis.clusters import Cluster
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

    def __init__(self, params: Dict = None):
        """
        Initialize the clustering engine.
        
        Args:
            params (Dict): Optional parameters for clustering.
        """
        self.event_clusters: List[List[Cluster]] = []
        self.cluster_id_counter: int = 0
        self.params = params
        self.pipeline = self.params.get("apply", {})


    def run(self, population: Population) -> List[List[Cluster]]:
        """
        Execute spatial clustering by combining direct/indirect neighbor and temporal constraints.

        Args:
            population (Population): The population of cells to cluster.

        Returns:
            List[List[Cluster]]: List of clusters per global peak.
        """
        try:
            self.event_clusters = []
            self.neighbor_graph = population.neighbor_graph
            trace_type = self.params.get("trace", "activity")
            max_comm_time = self.params.get("seq_max_comm_time", 5)
            indirect_hops = self.params.get("indirect_neighbors_num", 1)

            for global_peak in getattr(population, trace_type).peaks:
                window_start = global_peak.start_time
                window_end = global_peak.end_time
                active_entries = self._get_active_entries(population, window_start, window_end)
                active_entries.sort(key=lambda x: x[0].trace.peaks[x[1]].rel_start_time)
                clusters = self._cluster_combined(active_entries, max_comm_time, indirect_hops)
                self.event_clusters.append(clusters)

            return self.event_clusters

        except Exception as e:
            logger.error(f"Failed to run spatial clustering engine: {e}")
            return []

    def _get_active_entries(self, population: Population, window_start: int, window_end: int) -> List[Tuple[Cell, int]]:
        """
        Extract cell/peak index pairs with peaks starting in the specified window.

        Returns:
            List[Tuple[Cell, int]]: Active peaks.
        """
        active_entries = []
        for cell in population.cells:
            for i, peak in enumerate(cell.trace.peaks):
                if window_start <= peak.rel_start_time < window_end and not peak.in_cluster:
                    active_entries.append((cell, i))
        return active_entries

    def _get_neighbors_within_hops(self, label: int, max_hops: int) -> set:
        """
        Return neighbors of a node within a specified number of hops.

        Args:
            label (int): Node label.
            max_hops (int): Max hops to consider.

        Returns:
            set: Set of reachable node labels within hop distance.
        """
        visited = set()
        queue = deque([(label, 0)])
        while queue:
            node, depth = queue.popleft()
            if node in visited or depth > max_hops:
                continue
            visited.add(node)
            for neighbor in self.neighbor_graph.neighbors(node):
                queue.append((neighbor, depth + 1))
        return visited

    def _cluster_combined(self, active_entries: List[Tuple[Cell, int]], max_comm_time: int, max_hops: int) -> List[Cluster]:
        """
        Combine spatial (direct + optional indirect) and optional temporal constraints to form clusters.

        Args:
            active_entries (List[Tuple[Cell, int]]): Active peaks in the current window.
            max_comm_time (int): Maximum allowed frame difference.
            max_hops (int): Max hops to consider for indirect neighbors.
            use_indirect (bool): Whether to include indirect neighbors.
            use_sequential (bool): Whether to apply time constraint.

        Returns:
            List[Cluster]: Formed clusters.
        """
        clusters = []
        visited = set()
        label_to_entry = {cell.label: (cell, idx) for cell, idx in active_entries}

        for cell, idx in active_entries:
            if (cell.label, idx) in visited:
                continue
            peak = cell.trace.peaks[idx]
            cluster = Cluster(self.cluster_id_counter, peak.start_time, peak.end_time)
            cluster.add(cell, idx)
            visited.add((cell.label, idx))

            neighbors = self._get_neighbors_within_hops(cell.label, max_hops if self.pipeline.get("use_indirect_neighbors", False) else 1)
            for other_label in neighbors:
                if other_label == cell.label or other_label not in label_to_entry:
                    continue
                other_cell, other_idx = label_to_entry[other_label]
                other_peak = other_cell.trace.peaks[other_idx]
                if self.pipeline.get("use_sequential", False):
                    time_diff = abs(other_peak.rel_start_time - peak.rel_start_time)
                    if time_diff > max_comm_time:
                        continue
                cluster.add(other_cell, other_idx)
                visited.add((other_cell.label, other_idx))

            if len(cluster) > 1:
                clusters.append(cluster)
                self.cluster_id_counter += 1

        return clusters

    def plot_clusters_with_graph(
        self,
        population: Population,
        overlay_path: Path,
        output_dir: Path
    ) -> None:
        """
        Save overlay images showing clusters and their spatial graph for each global peak.

        Args:
            population (Population): The population (for centroids and graph).
            overlay_path (Path): Path to grayscale overlay image.
            output_dir (Path): Directory to save colored overlays.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        overlay = tifffile.imread(str(overlay_path))
        graph = population.neighbor_graph
        global_peaks = population.impulse_trace.peaks

        for peak_idx, (cluster_list, global_peak) in enumerate(zip(self.event_clusters, global_peaks)):
            color_overlay = np.stack([overlay] * 3, axis=-1).astype(np.uint8)

            # Generate colors
            cluster_colors: Dict[int, Tuple[int, int, int]] = {
                cluster.id: tuple(random.randint(50, 255) for _ in range(3))
                for cluster in cluster_list
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
                        node_colors.append(colors.to_rgb(np.array(cluster_colors[cluster.id]) / 255))
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

    def plot_clustered_raster(self, population: Population, output_dir: Path) -> None:
        """
        Plot a full raster of all cells' binary traces per global peak,
        grouped by clusters with a color and unclustered cells in gray.

        Args:
            population (Population): The full population.
            output_dir (Path): Directory to save raster plots.
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            all_cells = population.cells
            trace_length = max(len(cell.trace.binary) for cell in all_cells)
            cell_label_to_index = {cell.label: i for i, cell in enumerate(all_cells)}

            for peak_idx, clusters in enumerate(self.event_clusters):
                clustered_labels = set()
                cluster_to_cells = {}
                for i, cluster in enumerate(clusters):
                    cluster_to_cells[i] = [cell for cell, _ in cluster.members]
                    clustered_labels.update(cell.label for cell in cluster_to_cells[i])

                unclustered_cells = [cell for cell in all_cells if cell.label not in clustered_labels]

                # Order rows: by cluster then unclustered
                sorted_cells = []
                color_list = []
                cmap = cm.get_cmap("tab20", len(cluster_to_cells))
                for i, cells in cluster_to_cells.items():
                    sorted_cells.extend(cells)
                    color_list.extend([cmap(i)] * len(cells))
                sorted_cells.extend(unclustered_cells)
                color_list.extend([(0.5, 0.5, 0.5, 1.0)] * len(unclustered_cells))  # gray for unclustered

                # Create raster
                n_cells = len(sorted_cells)
                raster = np.zeros((n_cells, trace_length))
                for i, cell in enumerate(sorted_cells):
                    trace = cell.trace.binary
                    raster[i, :len(trace)] = trace

                fig, ax = plt.subplots(figsize=(36, 18))  # 7200x3600 pixels at 200 DPI
                ax.imshow(raster, aspect="auto", cmap="gray_r")

                for i, color in enumerate(color_list):
                    ax.axhline(i - 0.5, color=color, linewidth=2, alpha=0.6)

                ax.set_title(f"Clustered Raster Plot - Global Peak #{peak_idx}")
                ax.set_xlabel("Frame")
                ax.set_ylabel("Cells (grouped by cluster)")
                fig.tight_layout()

                save_path = output_dir / f"clustered_raster_peak_{peak_idx:03d}.png"
                fig.savefig(save_path, dpi=200)
                plt.close(fig)
                logger.info(f"Saved clustered raster plot to: {save_path}")

        except Exception as e:
            logger.error(f"Failed to plot clustered raster: {e}")