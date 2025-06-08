# events.py
# Usage Example:
# >>> events = Event.from_communications(communications, cells, config=EVENT_EXTRACTION_PARAMETERS)
# >>> for ev in events: print(ev)

from typing import List, Tuple, Dict, Optional
import networkx as nx
import numpy as np
from collections import defaultdict

from calcium_activity_characterization.data.cell_to_cell_communication import CellToCellCommunication
from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.utilities.metrics import Distribution
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean, pdist
from sklearn.decomposition import PCA


import logging
logger = logging.getLogger(__name__)

class Event:
    """
    Represents a single calcium propagation event across multiple cells, modeled as a directed graph
    of causally linked cell activations. This class computes spatial, temporal, and structural
    characteristics of the event.

    Attributes:
        id (int): Unique identifier for the event.
        communications (List[CellToCellCommunication]): List of communication links (edges) in the event.
        graph (nx.DiGraph): Directed acyclic graph (DAG) representing causal propagation between cells.
        label_to_cell (Dict[int, Cell]): Mapping from cell labels to Cell instances involved in the event.
        label_to_centroid (Dict[int, np.ndarray]): Mapping from cell labels to their centroids.
        config (Dict): Configuration dictionary for shape, timing, and speed thresholds.

        n_cells_involved (int): Number of unique cells participating in the event.
        event_duration (int): Duration of the event in frames, from first to last activation.

        communication_time_distribution (Distribution): Statistical distribution of time delays (Δt) between cell pairs.
        communication_speed_distribution (Distribution): Statistical distribution of centroid distances divided by Δt.

        framewise_active_labels (Dict[int, List[int]]): Map of each frame to list of active cell labels at that time.
        wavefront (Dict[int, List[np.ndarray]]): Map of each timepoint to cumulative active cell centroids.

        global_propagation_speed (float): Rate of growth of the convex hull area over time.
        dominant_direction_vector (Tuple[float, float]): Principal direction of event propagation (PCA-based).
        directional_propagation_speed (float): Projected speed of furthest peak from the origin in the dominant direction.

        elongation_score (float): Axis ratio of spatial spread (major/minor axis) based on PCA; ≥ 1.
        radiality_score (float): How central the origin cell is within the event hull; ∈ [0, 1], 1 = perfectly centered.
        compactness_score (float): Relative pairwise distance among event cells, normalized by population spacing.

        dag_metrics (Dict[str, float]): Dictionary containing graph structure metrics:
            - n_nodes: number of graph nodes (cells)
            - n_edges: number of graph edges (communications)
            - n_roots: number of source nodes (no incoming edges)
            - depth: max root-to-leaf path length
            - width: max number of nodes at any level
            - avg_out_degree: mean out-degree among non-leaf nodes
            - avg_path_length: mean length of all root-to-leaf paths
    """

    def __init__(self, id: int, communications: List[CellToCellCommunication], label_to_cell: Dict[int, Cell], config: Dict, population_centroids: List[np.ndarray]) -> None:
        """
        Initialize an Event instance.

        Args:
            id (int): Unique event identifier.
            communications (List[CellToCellCommunication]): List of communications in the event.
            label_to_cell (Dict[int, Cell]): Mapping from cell labels to Cell objects.
            config (Dict): Configuration parameters for event classification.
            population_centroids (List[np.ndarray]): List of centroids for all cells in the population.
        """
        self.id = id
        self.communications = communications
        self.label_to_cell = label_to_cell
        self.label_to_centroid = {label: cell.centroid for label, cell in label_to_cell.items()}
        self.config = config

        self.graph = nx.DiGraph()
        self._build_graph()

        self.n_cells_involved = self.graph.number_of_nodes()
        self.event_duration = self._compute_duration()

        self.communication_time_distribution = self._communication_time_distribution()
        self.communication_speed_distribution = self._communication_speed_distribution()

        self.framewise_active_labels = self._compute_framewise_active_labels()
        self.wavefront = self._compute_incremental_wavefront()

        self.global_propagation_speed = self._compute_area_growth_speed()
        self.dominant_direction_vector = self._compute_dominant_direction_vector()
        self.directional_propagation_speed = self._compute_directional_propagation_speed()

        self.elongation_score = self._compute_elongation_score()
        self.radiality_score = self._compute_radiality_score()
        self.compactness_score = self._compute_compactness_score(population_centroids)

        self.dag_metrics = self._compute_dag_metrics()

    def _build_graph(self) -> None:
        """
        Construct the directed graph from communication links.
        """
        for comm in self.communications:
            self.graph.add_edge(comm.origin[0], comm.cause[0], delta_t=comm.delta_t)

    def _compute_duration(self) -> int:
        """
        Compute the duration of the event in frames.

        Returns:
            int: Time span from first to last involved peak.
        """
        times = [c.origin_start_time for c in self.communications] + [c.cause_start_time for c in self.communications]
        return max(times) - min(times) if times else 0

    def _communication_time_distribution(self) -> Distribution:
        """
        Compute the distribution of communication delays (delta_t) across all communications.

        Returns:
            Distribution: Distribution of communication times (delta_t).
        """
        try:
            dt_values = [comm.delta_t for comm in self.communications if comm.delta_t > 0]
            return Distribution.from_values(dt_values)
        except Exception as e:
            logger.error(f"[Event {self.id}] Failed to compute communication times: {e}")
            return Distribution.from_values([])


    def _communication_speed_distribution(self) -> Distribution:
        """
        Compute the distribution of communication speeds (centroid distance / delta_t).

        Returns:
            Distribution: Distribution of communication speeds.
        """
        speeds = []
        try:
            for comm in self.communications:
                if comm.delta_t <= 0:
                    continue

                origin = self.label_to_cell.get(comm.origin[0])
                cause = self.label_to_cell.get(comm.cause[0])
                if origin is None or cause is None:
                    continue

                dist = euclidean(origin.centroid, cause.centroid)
                speed = dist / comm.delta_t
                speeds.append(speed)

            return Distribution.from_values(speeds)

        except Exception as e:
            logger.error(f"[Event {self.id}] Failed to compute communication speeds: {e}")
            return Distribution.from_values([])


    def _compute_framewise_active_labels(self) -> Dict[int, List[int]]:
        """
        Build raw map of which cells are active at each frame.

        Returns:
            Dict[int, List[int]]: Mapping from time t to list of active cell labels.
        """
        active_at_t = defaultdict(list)
        for comm in self.communications:
            for t, label in [(comm.origin_start_time, comm.origin[0]),
                            (comm.cause_start_time, comm.cause[0])]:
                if label not in active_at_t[t]:
                    active_at_t[t].append(label)
        return dict(active_at_t)

    def _compute_incremental_wavefront(self) -> Dict[int, List[np.ndarray]]:
        """
        Accumulate all cells active up to each time t.

        Returns:
            Dict[int, List[np.ndarray]]: Mapping from t to list of centroids.
        """
        wavefront = {}
        activated = set()
        for t in sorted(self.framewise_active_labels.keys()):
            activated.update(self.framewise_active_labels[t])
            centroids = [self.label_to_centroid[l] for l in activated if l in self.label_to_centroid]
            wavefront[t] = centroids
        return wavefront

    def _compute_dominant_direction_vector(self) -> Tuple[float, float]:
        """
        Compute dominant direction of propagation as the unit vector from origin
        to the center of mass (CoM) of all other involved centroids.
        The direction is validated against the average distance from the origin to the directly caused peaks.

        Returns:
            Tuple[float, float]: Dominant direction vector or (0, 0) if radial or undirected.
        """
        try:
            origin_label = self._get_origin_label()
            origin_centroid = np.array(self.label_to_centroid[origin_label])

            # Step 1: Compute CoM of all involved cells except the origin
            other_centroids = [
                np.array(c) for label, c in self.label_to_centroid.items()
                if label != origin_label
            ]

            if not other_centroids:
                logger.warning(f"[Event {self.id}] Not enough centroids for direction computation.")
                return (0.0, 0.0)

            center_of_mass = np.mean(other_centroids, axis=0)
            direction = center_of_mass - origin_centroid
            magnitude = np.linalg.norm(direction)

            # Step 2: Compute CoM of directly caused peaks
            caused_labels = [
                comm.cause[0] for comm in self.communications
                if comm.origin[0] == origin_label
            ]
            caused_centroids = [
                np.array(self.label_to_centroid[label])
                for label in caused_labels if label in self.label_to_centroid
            ]

            if not caused_centroids:
                logger.warning(f"[Event {self.id}] Origin has no caused peaks for threshold computation.")
                return (0.0, 0.0)

            caused_com = np.mean(caused_centroids, axis=0)
            threshold = np.linalg.norm(caused_com - origin_centroid)

            if magnitude < 0.5 * threshold:
                return (0.0, 0.0)

            return tuple(direction / magnitude)

        except Exception as e:
            logger.error(f"[Event {self.id}] Failed to compute CoM direction: {e}")
            return (0.0, 0.0)


    def _compute_directional_propagation_speed(self) -> float:
        """
        Compute the directional propagation speed:
        distance (along dominant direction) from origin to furthest cell,
        divided by their Δt (start time difference).

        Returns:
            float: Directional propagation speed.
        """
        try:
            dir_vec = np.array(self.dominant_direction_vector)
            if np.allclose(dir_vec, 0):
                return 0.0

            # Get origin cell and its activation time
            origin_label = self._get_origin_label()
            origin_cell = self.label_to_cell.get(origin_label)
            origin_centroid = np.array(origin_cell.centroid)

            # Get origin activation time
            t_origin = next(
                (comm.origin_start_time for comm in self.communications if comm.origin[0] == origin_label),
                None)
            if t_origin is None:
                logger.warning(f"[Event {self.id}] Could not determine activation time of origin cell.")
                return 0.0

            # Search for furthest cell in direction
            max_proj = -np.inf
            t_furthest = None

            for comm in self.communications:
                label = comm.cause[0]
                if label == origin_label:
                    continue
                cell = self.label_to_cell.get(label)
                if cell is None:
                    continue
                displacement = np.array(cell.centroid) - origin_centroid
                projection = np.dot(displacement, dir_vec)
                if projection > max_proj:
                    max_proj = projection
                    t_furthest = comm.cause_start_time

            if max_proj <= 0 or t_furthest is None or t_furthest <= t_origin:
                return 0.0

            dt = t_furthest - t_origin
            return max_proj / dt

        except Exception as e:
            logger.error(f"[Event {self.id}] Failed to compute directional propagation speed: {e}")
            return 0.0


    def _compute_area_growth_speed(self) -> float:
        """
        TODO: if going to be used, need to take care of the plateau problem.
        Compute speed as convex hull area increase over time.

        Returns:
            float: Propagation speed in area units per frame.
        """
        areas = []
        times = []
        min_points = self.config["convex_hull"].get("min_points", 3)
        min_dt = self.config["convex_hull"].get("min_duration", 1)

        for t, points in self.wavefront.items():
            if len(points) < min_points:
                continue
            try:
                hull = ConvexHull(np.vstack(points))
                areas.append(hull.area)
                times.append(t)
            except:
                continue

        if len(areas) < 2:
            return 0.0

        d_area = areas[-1] - areas[0]
        d_time = times[-1] - times[0]

        return d_area / d_time if d_time >= min_dt else 0.0


    def _compute_elongation_score(self) -> float:
        """
        Compute the elongation score (PCA axis ratio) of the event shape.
        Ratio of major to minor axis — 1.0 means circular, >1.0 means elongated.

        Returns:
            float: Elongation score ≥ 1.0
        """
        try:
            centroids = list(self.label_to_centroid.values())
            if len(centroids) < 3:
                return 1.0
            X = np.vstack(centroids).astype(float)
            X -= np.mean(X, axis=0)
            _, S, _ = np.linalg.svd(X)
            return S[0] / S[1] if len(S) >= 2 and S[1] > 0 else 1.0
        except Exception as e:
            logger.error(f"[Event {self.id}] Failed to compute elongation score: {e}")
            return 1.0


    def _compute_radiality_score(self) -> float:
        """
        Compute how central the origin is within the convex hull of the event.
        1.0 = perfectly central, 0.0 = origin is on the outer edge.

        Returns:
            float: Radiality score ∈ [0, 1]
        """
        try:
            centroids = list(self.label_to_centroid.values())
            if len(centroids) < 3:
                return 0.0
            points = np.vstack(centroids)
            center = np.mean(points, axis=0)

            origin_label = self._get_origin_label()
            origin = np.array(self.label_to_centroid[origin_label])

            dist_origin_to_center = np.linalg.norm(origin - center)
            hull_radius = max(np.linalg.norm(p - center) for p in points)
            return 1.0 - (dist_origin_to_center / hull_radius) if hull_radius > 0 else 0.0
        except Exception as e:
            logger.error(f"[Event {self.id}] Failed to compute radiality score: {e}")
            return 0.0


    def _compute_compactness_score(self, population_centroids: List[np.ndarray]) -> float:
        """
        TODO: direct neighbors distance instead of pairwise distance.
        Compute how tightly clustered the event cells are compared to the global population.

        Args:
            population_centroids (List[np.ndarray]): List of centroids of all cells in the image.

        Returns:
            float: Compactness score (event spacing / population spacing)
        """
        try:
            event_centroids = list(self.label_to_centroid.values())
            if len(event_centroids) < 2 or len(population_centroids) < 2:
                return 0.0
            event_dist = np.mean(pdist(np.vstack(event_centroids)))
            pop_dist = np.mean(pdist(np.vstack(population_centroids)))
            return event_dist / pop_dist if pop_dist > 0 else 0.0
        except Exception as e:
            logger.error(f"[Event {self.id}] Failed to compute compactness score: {e}")
            return 0.0


    def _compute_dag_metrics(self) -> Dict[str, float]:
        """
        Compute structural metrics describing the DAG topology of the event propagation graph.

        Returns:
            Dict[str, float]: Dictionary of DAG metrics including:
                - n_nodes: Total number of nodes in the graph
                - n_edges: Total number of directed edges
                - n_roots: Nodes with in-degree 0 (initial cells)
                - depth: Maximum path length from any root to leaf
                - width: Maximum number of nodes at any depth level
                - avg_out_degree: Average number of outgoing connections per non-leaf node
                - avg_path_length: Average length of all root-to-leaf paths
        """
        try:
            G = self.graph
            nodes = list(G.nodes)
            if not nodes:
                return {}

            # Basic metrics
            n_nodes = len(nodes)
            n_edges = G.number_of_edges()
            n_roots = sum(1 for n in nodes if G.in_degree(n) == 0)
            n_leaves = [n for n in nodes if G.out_degree(n) == 0]

            # Compute depth (longest root-to-leaf path)
            depth = 0
            root_to_leaf_paths = []
            for root in [n for n in nodes if G.in_degree(n) == 0]:
                for leaf in n_leaves:
                    try:
                        path = nx.shortest_path(G, source=root, target=leaf)
                        root_to_leaf_paths.append(path)
                        depth = max(depth, len(path) - 1)
                    except nx.NetworkXNoPath:
                        continue

            # Width: most nodes at any single level
            level_widths = defaultdict(int)
            for root in [n for n in nodes if G.in_degree(n) == 0]:
                for n in nodes:
                    try:
                        d = nx.shortest_path_length(G, source=root, target=n)
                        level_widths[d] += 1
                    except nx.NetworkXNoPath:
                        continue
            width = max(level_widths.values(), default=1)

            # Average out-degree for non-leaf nodes
            out_degrees = [G.out_degree(n) for n in nodes if G.out_degree(n) > 0]
            avg_out_degree = np.mean(out_degrees) if out_degrees else 0.0

            # Average path length (root to leaf)
            path_lengths = [len(p) - 1 for p in root_to_leaf_paths]
            avg_path_length = float(np.mean(path_lengths)) if path_lengths else 0.0

            return {
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "n_roots": n_roots,
                "depth": depth,
                "width": width,
                "avg_out_degree": round(float(avg_out_degree), 3),
                "avg_path_length": round(float(avg_path_length), 3),
            }

        except Exception as e:
            logger.error(f"[Event {self.id}] Failed to compute DAG metrics: {e}")
            return {}


    def _get_origin_label(self) -> Optional[int]:
        """
        Find the label of the origin cell (the root node in the DAG).
        This is the cell with no incoming edges in the graph.
        If no such node exists, returns the label of the earliest activation.
        If multiple candidates exist, returns the one with the earliest activation time.

        Returns:
            Optional[int]: Label of the origin cell, or None if not found.
        """
        try:
            origin_label = next((n for n in self.graph.nodes if self.graph.in_degree(n) == 0), None)
            if origin_label is not None:
                return origin_label

            # Fallback based on earliest activation
            label_to_first_time = {
                label: min(
                    [c.origin_start_time for c in self.communications if c.origin[0] == label] +
                    [c.cause_start_time for c in self.communications if c.cause[0] == label]
                )
                for label in self.graph.nodes
            }
            fallback_label = min(label_to_first_time, key=label_to_first_time.get)
            logger.warning(f"[Event {self.id}] No root node found; using fallback label {fallback_label}")
            return fallback_label
        except Exception as e:
            logger.error(f"[Event {self.id}] Failed to find origin label: {e}")
            return None



    @classmethod
    def from_communications(
        cls,
        communications: List[CellToCellCommunication],
        cells: List[Cell],
        config: Dict,
        population_centroids: List[np.ndarray] = None
    ) -> List["Event"]:
        """
        Extract events from a list of cell-to-cell communications.

        Args:
            communications (List[CellToCellCommunication]): List of communication links.
            cells (List[Cell]): All valid cells in the population.
            config (Dict): Configuration parameters.

        Returns:
            List[Event]: List of assembled Event objects.
        """
        label_to_cell = {cell.label: cell for cell in cells}
        G = nx.Graph()
        for comm in communications:
            G.add_edge(comm.origin, comm.cause)

        components = list(nx.connected_components(G))
        events = []
        counter = 0
        min_cells = config.get("min_cell_count", 2)

        for component in components:
            if len(component) < min_cells:
                continue
            peak_ids = set(component)
            group_comms = [c for c in communications if c.origin in peak_ids and c.cause in peak_ids]
            
            label_ids = {peak_id[0] for peak_id in peak_ids}
            event_label_to_cell = {label: label_to_cell[label] for label in label_ids if label in label_to_cell}
            event = cls(counter, group_comms, event_label_to_cell, config, population_centroids)
            events.append(event)
            counter += 1

        logger.info(f"Extracted {len(events)} events from {len(communications)} communications.")
        return events