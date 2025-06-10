# event_classes.py
# Usage Example:
# >>> event = SequentialEvent.from_communications(id=0, communications=[...], config={})
# >>> print(event.event_duration, event.n_cells_involved)
# >>> global_event = GlobalEvent.from_framewise_active_labels(id=1, label_to_cell=..., framewise_active_labels=..., config={})

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import euclidean, pdist
from scipy.spatial import ConvexHull
import networkx as nx
from abc import ABC, abstractmethod
from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.data.cell_to_cell_communication import CellToCellCommunication
from calcium_activity_characterization.utilities.metrics import Distribution
import logging

logger = logging.getLogger(__name__)

class Event(ABC):
    """
    Abstract base class for calcium activity events.

    Attributes:
        id (int): Unique identifier.
        peaks_involved (Tuple[int, int]): Indices of the earliest and latest peaks involved in the event.
        label_to_centroid (Dict[int, np.ndarray]): Map of cell labels to centroids.
        n_cells_involved (int): Number of cells in the event.
        event_start_time (int): Earliest peak start time.
        event_end_time (int): Latest peak start time.
        event_duration (int): Duration of the event.
        framewise_active_labels (Dict[int, List[int]]): Active labels per frame.
        wavefront (Dict[int, List[np.ndarray]]): Centroids of active cells per frame.
        config (Dict): Detection parameters used.
        dominant_direction_vector (Tuple[float, float]): Direction of propagation as a unit vector.
        directional_propagation_speed (float): Speed of propagation in the dominant direction.
    """

    def __init__(
        self,
        id: int,
        peaks_involved: List[Tuple[int, int]],
        label_to_centroid: Dict[int, np.ndarray],
        framewise_active_labels: Dict[int, List[int]] = None
    ) -> None:
        self.id = id

        self.peaks_involved = peaks_involved
        self.label_to_centroid = label_to_centroid

        self.n_cells_involved = len(label_to_centroid)
        self.framewise_active_labels = framewise_active_labels

        self.event_start_time, self.event_end_time = self._compute_event_bounds()
        self.event_duration = self.event_end_time - self.event_start_time + 1
        self.wavefront = self._compute_incremental_wavefront()
        
        self.dominant_direction_vector = self._compute_dominant_direction_vector()
        self.directional_propagation_speed = self._compute_directional_propagation_speed()


        if id == 395:
            logger.info(f"[Event {self.id}] Created with {self.n_cells_involved} cells, "
                        f"duration {self.event_duration} frames, "
                        f"start time {self.event_start_time}, end time {self.event_end_time}.")

    def _compute_event_bounds(self) -> Tuple[int, int]:
        """
        Compute the start and end times of the event based on active labels.
        
        Returns:
            Tuple[int, int]: Start and end times of the event.
        """
        times = list(self.framewise_active_labels.keys())
        return min(times), max(times)

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

    @abstractmethod
    def _compute_dominant_direction_vector(self) -> Tuple[float, float]:
        """Compute direction of propagation."""
        pass

    @abstractmethod
    def _compute_directional_propagation_speed(self) -> float:
        """Compute speed in dominant direction."""
        pass


class SequentialEvent(Event):
    """
    Event subclass for sequential neighbor-to-neighbor propagation events.

    Additional Attributes:
        communications (List[CellToCellCommunication]): Directed links between cells.
        graph (nx.DiGraph): Causal propagation graph.
        dag_metrics (Dict): Precomputed DAG metrics.
        communication_speed_distribution (List[float]): Optional.
        communication_time_distribution (List[float]): Optional.
        elongation_score (float): Shape metric.
        compactness_score (float): Shape metric.
        global_propagation_speed (float): Mean propagation speed.
    """

    def __init__(
        self,
        id: int,
        communications: List[CellToCellCommunication],
        label_to_centroid: Dict[int, np.ndarray],
        config_hull: Dict,
        population_centroid: List[np.ndarray]
    ) -> None:
        peak_indices = list({comm.origin for comm in communications}.union({comm.cause for comm in communications}))
        self.communications = communications
        
        if id == 395:
            logger.info(f"[Event {id}] Created wit cells, ")

        self.graph = nx.DiGraph()
        self._build_graph()
        self.dag_metrics = self._compute_dag_metrics()

        super().__init__(id, peak_indices, label_to_centroid, self._compute_framewise_active_labels())

        self.communication_time_distribution: Distribution = self._communication_time_distribution()
        self.communication_speed_distribution: Distribution = self._communication_speed_distribution()
        self.area_propagation_speed: float = self._compute_area_propagation_speed(config_hull)

        self.elongation_score: float = self._compute_elongation_score()
        self.radiality_score: float = self._compute_radiality_score()
        self.compactness_score: float = self._compute_compactness_score(population_centroid)

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


    def _build_graph(self) -> None:
        """
        Construct the directed graph from communication links.
        """
        for comm in self.communications:
            self.graph.add_edge(comm.origin[0], comm.cause[0], delta_t=comm.delta_t)


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

                dist = euclidean(self.label_to_centroid[comm.origin[0]],
                                 self.label_to_centroid[comm.cause[0]])
                speed = dist / comm.delta_t
                speeds.append(speed)

            return Distribution.from_values(speeds)

        except Exception as e:
            logger.error(f"[Event {self.id}] Failed to compute communication speeds: {e}")
            return Distribution.from_values([])


    def _compute_area_propagation_speed(self, config_hull: Dict) -> float:
        """
        TODO: if going to be used, need to take care of the plateau problem.
        Compute speed as convex hull area increase over time.

        Returns:
            float: Propagation speed in area units per frame.
        """
        areas = []
        times = []
        min_points = config_hull.get("min_points", 3)
        min_dt = config_hull.get("min_duration", 1)

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
            origin_centroid = np.array(self.label_to_centroid[origin_label])

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
                displacement = np.array(self.label_to_centroid[label]) - origin_centroid
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
        n_global_events: int,
        communications: List[CellToCellCommunication],
        cells: List[Cell],
        config: Dict,
        population_centroids: List[np.ndarray] = None
    ) -> List["SequentialEvent"]:
        """
        Extract events from a list of cell-to-cell communications.

        Args:
            communications (List[CellToCellCommunication]): List of directed cell-to-cell communication links.
            cells (List[Cell]): List of all cells in the image.
            config (Dict): Configuration parameters for event extraction.
            population_centroids (List[np.ndarray]): Optional list of centroids of all cells in the image.

        Returns:
            List[SequentialEvent]: List of extracted SequentialEvent instances.
        """
        label_to_cell = {cell.label: cell for cell in cells}
        G = nx.Graph()
        for comm in communications:
            G.add_edge(comm.origin, comm.cause)

        components = list(nx.connected_components(G))
        events = []
        counter = n_global_events
        min_cells = config.get("min_cell_count", 2)

        for component in components:
            if len(component) < min_cells:
                continue
            peak_ids = set(component)
            group_comms = [c for c in communications if c.origin in peak_ids and c.cause in peak_ids]
            
            label_ids = {peak_id[0] for peak_id in peak_ids}
            event_label_to_cell = {label: label_to_cell[label] for label in label_ids if label in label_to_cell}
            label_to_centroid = {label: cell.centroid for label, cell in event_label_to_cell.items()}
            event = cls(counter, group_comms, label_to_centroid, config.get("convex_hull"), population_centroids)
            events.append(event)
            counter += 1

        logger.info(f"Extracted {len(events)} events from {len(communications)} communications.")
        return events


class GlobalEvent(Event):
    """
    Event subclass for paracrine/global wave events.

    Constructed from spatial + temporal coactivation without DAG.
    """

    def __init__(
        self,
        id: int,
        peak_indices: Tuple[int, int],
        label_to_centroid: Dict[int, np.ndarray],
        framewise_active_labels: Dict[int, List[int]]
    ) -> None:
        super().__init__(id, peak_indices, label_to_centroid, framewise_active_labels)

        self.dominant_direction_vector = self._compute_dominant_direction_vector()
        self.directional_propagation_speed = self._compute_directional_propagation_speed()

    def _compute_dominant_direction_vector(self) -> Tuple[float, float]:
        """
        TODO: PCA-based analysis on centroids across time.

        Returns:
            Tuple[float, float]: Unit vector (dx, dy) indicating direction.
        """
        return (0.0, 0.0)

    def _compute_directional_propagation_speed(self) -> float:
        """
        TODO: Compute propagation speed across projected direction.

        Returns:
            float: Speed in dominant direction (pixels/frame).
        """
        return 0.0

    @classmethod
    def from_framewise_active_labels(
        cls,
        framewise_label_blocks: List[Dict[int, List[int]]],
        cells: List[Cell],
        config: Dict
    ) -> List["GlobalEvent"]:
        """
        Create multiple GlobalEvent objects from a list of framewise label dictionaries.

        Args:
            framewise_label_blocks (List[Dict[int, List[int]]]): List of frame-label mappings, one per event.
            cells (List[Cell]): All available cell objects.
            config (Dict): Configuration dict. Should contain 'min_cell_count'.

        Returns:
            List[GlobalEvent]: List of created GlobalEvent instances.
        """
        min_cells = config.get("min_cell_count", 3)
        label_to_cell = {cell.label: cell for cell in cells}

        events = []
        counter = 0

        for framewise_labels in framewise_label_blocks:
            involved_labels = {l for labels in framewise_labels.values() for l in labels}
            if len(involved_labels) < min_cells:
                continue

            label_to_centroid = {
                label: label_to_cell[label].centroid
                for label in involved_labels if label in label_to_cell
            }

            peak_indices = [
                (label, peak.id)
                for t, labels in framewise_labels.items()
                for label in labels
                if (peak := label_to_cell[label].trace.get_peak_starting_at(t)) is not None
            ]

            event = cls(
                id=counter,
                label_to_centroid=label_to_centroid,
                framewise_active_labels=framewise_labels,
                peak_indices=peak_indices
            )
            events.append(event)
            counter += 1

        logger.info(f"Created {len(events)} GlobalEvents from framewise label blocks.")
        return events

