# event_classes.py
# Usage Example:
# >>> event = SequentialEvent.from_communications(id=0, communications=[...], config={})
# >>> print(event.event_duration, event.n_cells_involved)
# >>> global_event = GlobalEvent.from_framewise_peaking_labels(id=1, label_to_cell=..., framewise_peaking_labels=..., config={})

from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import euclidean, pdist
from scipy.spatial import ConvexHull
import networkx as nx
from abc import ABC, abstractmethod

from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.data.cell_to_cell_communication import CellToCellCommunication
from calcium_activity_characterization.utilities.metrics import Distribution
from calcium_activity_characterization.config.presets import EventExtractionConfig, ConvexHullParams, DirectionComputationParams

from pathlib import Path
from calcium_activity_characterization.utilities.plotter import plot_event_graph

import logging

logger = logging.getLogger(__name__)

class Event(ABC):
    """
    Abstract base class for calcium activity events.

    Attributes:
        id (int): Unique identifier.
        peaks_involved (List[Tuple[int, int]]): list of cell label and peaks indices involved in the event.
        label_to_centroid (Dict[int, np.ndarray]): Map of cell labels to centroids.
        n_cells_involved (int): Number of cells in the event.
        event_start_time (int): Earliest peak start time.
        event_end_time (int): Latest peak start time.
        event_duration (int): Duration of the event.
        framewise_peaking_labels (Dict[int, List[int]]): Map of cell labels peaking at each frame.
        wavefront (Dict[int, List[np.ndarray]]): Centroids of peaking cells per frame.
        config (Dict): Detection parameters used.
        dominant_direction_vector (Tuple[float, float]): Direction of propagation as a unit vector.
        directional_propagation_speed (float): Speed of propagation in the dominant direction.
    """

    def __init__(
        self,
        id: int,
        peaks_involved: List[Tuple[int, int]],
        label_to_centroid: Dict[int, np.ndarray],
        framewise_peaking_labels: Dict[int, List[int]] = None
    ) -> None:
        self.id = int(id)

        self.peaks_involved = peaks_involved
        self.label_to_centroid = label_to_centroid

        self.n_cells_involved = len(label_to_centroid)
        self.framewise_peaking_labels = framewise_peaking_labels

        self.event_start_time, self.event_end_time = self._compute_event_bounds()
        self.event_duration = self.event_end_time - self.event_start_time + 1
        self.wavefront = self._compute_incremental_wavefront()
        self.growth_curve_distribution = self._compute_growth_curve()
        self.growth_curve_mean = self.growth_curve_distribution.mean
        self.growth_curve_std = self.growth_curve_distribution.std

        self.dominant_direction_vector = None # to be computed in subclasses
        self.directional_propagation_speed = None # to be computed in subclasses


    def _compute_event_bounds(self) -> Tuple[int, int]:
        """
        Compute the start and end times of the event based on active labels.
        
        Returns:
            Tuple[int, int]: Start and end times of the event.
        """
        times = list(self.framewise_peaking_labels.keys())
        return min(times), max(times)

    def _compute_incremental_wavefront(self) -> Dict[int, List[np.ndarray]]:
        """
        Accumulate all cells active up to each time t.

        Returns:
            Dict[int, List[np.ndarray]]: Mapping from t to list of centroids.
        """
        wavefront = {}
        activated = set()
        for t in sorted(self.framewise_peaking_labels.keys()):
            activated.update(self.framewise_peaking_labels[t])
            centroids = [self.label_to_centroid[l] for l in activated if l in self.label_to_centroid]
            wavefront[t] = centroids
        return wavefront

    def _compute_growth_curve(self) -> Distribution:
        """
        Store cumulative number of newly recruited cells per frame as a Distribution.
        """
        try:
            values = [
                len(labels)
                for _, labels in sorted(self.framewise_peaking_labels.items())
            ]
            cumulative = np.cumsum(values).tolist()
            growth_curve = Distribution.from_values(cumulative)
            return growth_curve
        except Exception as e:
            logger.error(f"[Event {self.id}] Failed to compute growth curve: {e}")
            return Distribution.from_values([])    

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
        config_hull: ConvexHullParams,
        population_centroid: List[np.ndarray]
    ) -> None:
        peak_indices = list({comm.origin for comm in communications}.union({comm.cause for comm in communications}))
        self.communications = communications

        self.graph = nx.DiGraph()
        self._build_graph()
        self.dag_metrics = self._compute_dag_metrics()

        super().__init__(id, peak_indices, label_to_centroid, self._compute_framewise_peaking_labels())

        self.dominant_direction_vector = self._compute_dominant_direction_vector()
        self.directional_propagation_speed = self._compute_directional_propagation_speed()

        self.communication_time_distribution: Distribution = self._communication_time_distribution()
        self.communication_time_mean: float = self.communication_time_distribution.mean
        self.communication_time_std: float = self.communication_time_distribution.std
        self.communication_speed_distribution: Distribution = self._communication_speed_distribution()
        self.communication_speed_mean: float = self.communication_speed_distribution.mean
        self.communication_speed_std: float = self.communication_speed_distribution.std
        self.area_propagation_speed: float = self._compute_area_propagation_speed(config_hull)

        self.elongation_score: float = self._compute_elongation_score()
        self.radiality_score: float = self._compute_radiality_score()
        self.compactness_score: float = self._compute_compactness_score(population_centroid)

    def _compute_framewise_peaking_labels(self) -> Dict[int, List[int]]:
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
        If delta_t == 0, it means that we don't have the temporal resolution to measure speed, 
        so we use the maximum possible resolution (1 frame).

        Returns:
            Distribution: Distribution of communication speeds.
        """
        speeds = []
        try:
            for comm in self.communications:
                if comm.delta_t < 0:
                    continue
                elif comm.delta_t == 0:
                    time = 1
                else:
                    time = comm.delta_t

                dist = euclidean(self.label_to_centroid[comm.origin[0]],
                                 self.label_to_centroid[comm.cause[0]])
                speed = dist / time
                speeds.append(speed)

            return Distribution.from_values(speeds)

        except Exception as e:
            logger.error(f"[Event {self.id}] Failed to compute communication speeds: {e}")
            return Distribution.from_values([])


    def _compute_area_propagation_speed(self, config_hull: ConvexHullParams) -> float:
        """
        TODO: if going to be used, need to take care of the plateau problem.
        Compute speed as convex hull area increase over time.

        Returns:
            float: Propagation speed in area units per frame.
        """
        areas = []
        times = []
        min_points = config_hull.min_points
        min_dt = config_hull.min_duration

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
            score = S[0] / S[1] if len(S) >= 2 and S[1] > 0 else 1.0
            return min(score,100.0)  # Cap the score to avoid extreme values
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
        config: EventExtractionConfig,
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
        min_cells = config.min_cell_count

        for component in components:
            if len(component) < min_cells:
                continue
            peak_ids = set(component)
            group_comms = [c for c in communications if c.origin in peak_ids and c.cause in peak_ids]
            
            label_ids = {peak_id[0] for peak_id in peak_ids}
            event_label_to_cell = {label: label_to_cell[label] for label in label_ids if label in label_to_cell}
            label_to_centroid = {label: cell.centroid for label, cell in event_label_to_cell.items()}
            event = cls(counter, group_comms, label_to_centroid, config.convex_hull, population_centroids)
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
        event_peak_time: int,
        peak_indices: Tuple[int, int],
        label_to_centroid: Dict[int, np.ndarray],
        framewise_peaking_labels: Dict[int, List[int]],
        config_direction: DirectionComputationParams
    ) -> None:
        super().__init__(id, peak_indices, label_to_centroid, framewise_peaking_labels)

        self.event_peak_time = int(event_peak_time)

        self.direction_metadata = self._compute_dominant_direction_metadata(config_direction)

        self.dominant_direction_vector = self._compute_dominant_direction_vector()
        self.directional_propagation_speed = self._compute_directional_propagation_speed()

        self.time_to_50, self.normalized_peak_rate_at_50 = self._compute_time_and_peak_rate_at_50()

    def _compute_dominant_direction_metadata(self, config: DirectionComputationParams) -> Dict[str, Any]:
        """
        Compute dominant direction and supporting metadata using trimmed CoM over time bins.

        Returns:
            dict: Includes:
                - direction_vector (tuple)
                - bin_centroids (list[np.ndarray])
                - net_displacement (float)
                - max_event_extent (float)
                - is_directional (bool)
                - bins (list[dict]): per-bin metadata for GUI
        """
        duration = self.event_end_time - self.event_start_time + 1

        bins = [
            (self.event_start_time + i * duration // config.num_time_bins,
            self.event_start_time + (i + 1) * duration // config.num_time_bins - 1)
            for i in range(config.num_time_bins)
        ]
        bins[-1] = (bins[-1][0], self.event_end_time)

        bin_centroids = []
        all_centroids = []
        bin_metadata = []

        for start, end in bins:
            active_labels = set()
            for t in range(start, end + 1):
                active_labels.update(self.framewise_peaking_labels.get(t, []))

            centroids = [self.label_to_centroid[l] for l in active_labels if l in self.label_to_centroid]
            all_centroids.extend(centroids)

            if not centroids:
                centroid = np.array([0.0, 0.0])
                bin_metadata.append({
                    "start": start,
                    "end": end,
                    "centroid": centroid,
                    "raw_centroids": [],
                    "filtered_centroids": [],
                    "radius": 0.0,
                    "label_ids": [],
                    "peak_times": []
                })
                bin_centroids.append(centroid)
                continue

            arr = np.array(centroids)
            med = np.median(arr, axis=0)
            dists = np.linalg.norm(arr - med, axis=1)
            mad = np.median(np.abs(dists - np.median(dists)))
            threshold = np.median(dists) + config.mad_filtering_multiplier * mad
            filtered = arr[dists <= threshold]

            centroid = np.mean(filtered, axis=0) if len(filtered) > 0 else med
            bin_centroids.append(centroid)
            bin_metadata.append({
                "start": start,
                "end": end,
                "centroid": centroid,
                "raw_centroids": arr,
                "filtered_centroids": filtered,
                "radius": threshold,
                "label_ids": list(active_labels),
                "peak_times": list(range(start, end + 1))
            })

        # Exclude first and last bins from directional analysis
        if len(bin_centroids) < 3:
            return {
                "direction_vector": (0.0, 0.0),
                "bin_centroids": bin_centroids,
                "net_displacement": 0.0,
                "max_event_extent": 0.0,
                "is_directional": False,
                "bins": bin_metadata
            }

        inner_centroids = bin_centroids[1:-1]
        centered = np.array(inner_centroids) - np.mean(inner_centroids, axis=0)

        try:
            _, _, Vt = np.linalg.svd(centered)
            vec = Vt[0]
            unit_vec = tuple(vec)
            net_disp = float(np.linalg.norm(inner_centroids[-1] - inner_centroids[0]))
        except Exception:
            logger.warning(f"[Event {self.id}] SVD failed.")
            return {
                "direction_vector": (0.0, 0.0),
                "bin_centroids": bin_centroids,
                "net_displacement": 0.0,
                "max_event_extent": 0.0,
                "is_directional": False,
                "bins": bin_metadata
            }

        all_centroids = np.array(all_centroids)
        max_extent = float(np.max(pdist(all_centroids))) if len(all_centroids) >= 2 else 0.0

        is_directional = net_disp >= config.min_net_displacement_ratio * max_extent if max_extent > 0 else False
        if not is_directional:
            unit_vec = (0.0, 0.0)

        return {
            "direction_vector": unit_vec,
            "bin_centroids": bin_centroids,
            "net_displacement": net_disp,
            "max_event_extent": max_extent,
            "is_directional": is_directional,
            "bins": bin_metadata
        }

    def _compute_dominant_direction_vector(self) -> Tuple[float, float]:
        """
        Compute dominant direction vector of a global wave event.

        Returns:
            tuple[float, float]: Unit vector (dy, dx) or (0.0, 0.0) if isotropic.
        """
        return self.direction_metadata["direction_vector"]


    def _compute_directional_propagation_speed(self) -> float:
        """
        Compute propagation speed across dominant direction based on bin centroids and mean peak times.

        Returns:
            float: Speed in pixels/frame, or 0.0 if not directional or insufficient data.
        """
        metadata = self.direction_metadata
        if not metadata["is_directional"]:
            return 0.0

        bins = metadata["bins"]
        if len(bins) < 3:
            return 0.0

        first_bin = bins[1]
        last_bin = bins[-2]

        centroid_start = np.array(first_bin["centroid"])
        centroid_end = np.array(last_bin["centroid"])

        peak_times_start = first_bin["peak_times"]
        peak_times_end = last_bin["peak_times"]

        if not peak_times_start or not peak_times_end:
            return None

        t_start = np.mean(peak_times_start)
        t_end = np.mean(peak_times_end)
        delta_t = t_end - t_start
        if delta_t <= 0:
            return 0.0

        displacement = np.linalg.norm(centroid_end - centroid_start)
        return displacement / delta_t


    def _compute_time_and_peak_rate_at_50(self, window_size: int = 5) -> Tuple[int, float]:
        """
        Compute the number of frames from event start until cumulative recruited peaks
        reach at least 50% of the total.

        Uses cumulative sum of growth curve to find the earliest frame reaching the threshold.

        Returns:
            float: Frame offset from event start (e.g., 0 means 50% was reached in first frame).
        """
        try:
            cumulative = self.growth_curve_distribution.values
            total = cumulative[-1]
            half = total / 2.0

            # Find the time to reach 50% of total activation
            idx_50 = next((i for i, val in enumerate(cumulative) if val >= half), 0)

            # Compute peak rate at 50% of total activation

            half_window = window_size // 2
            start = max(0, idx_50 - half_window)
            end = min(len(cumulative), idx_50 + half_window + 1)

            x = np.arange(start, end)
            y = cumulative[start:end]

            x_mean, y_mean = np.mean(x), np.mean(y)
            peak_rate_50 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)

            normalized_peak_rate_50 = (peak_rate_50 / total)*100

            return idx_50, normalized_peak_rate_50
        except Exception as e:
            logger.error(f"[Event {self.id}] Failed to compute cumulative growth info: {e}")
            return 0, 0.0

    @classmethod
    def from_framewise_peaking_labels(
        cls,
        events_peak_times: List[int],
        framewise_label_blocks: List[Dict[int, List[Tuple[int, int]]]],
        cells: List[Cell],
        config: EventExtractionConfig
    ) -> List["GlobalEvent"]:
        """
        Create multiple GlobalEvent objects from a list of framewise (cell_label, peak_id) dictionaries.

        Args:
            framewise_label_blocks (List[Dict[int, List[Tuple[int, int]]]]): List of frame -> [(cell_label, peak_id)] mappings, one per event.
            cells (List[Cell]): All available cell objects.
            config (EventExtractionConfig): Configuration object. Should contain 'min_cell_count'.

        Returns:
            List[GlobalEvent]: List of created GlobalEvent instances.
        """
        min_cells = config.min_cell_count
        label_to_cell = {cell.label: cell for cell in cells}

        events = []

        for i, framewise_labels in enumerate(framewise_label_blocks):
            # Gather all unique (label, peak_id) pairs across the event
            involved_label_ids = {label for label, _ in {p for lst in framewise_labels.values() for p in lst}}
            if len(involved_label_ids) < min_cells:
                continue

            # Build label -> centroid mapping for involved cells
            label_to_centroid = {
                label: label_to_cell[label].centroid
                for label in involved_label_ids if label in label_to_cell
            }

            # Collect the (label, peak_id) tuples into a list
            peak_indices = list({(label, pid) for frame in framewise_labels.values() for (label, pid) in frame})

            # Construct the GlobalEvent
            event = cls(
                id=i,
                event_peak_time=events_peak_times[i],
                label_to_centroid=label_to_centroid,
                framewise_peaking_labels={
                    t: [label for (label, _) in entries] for t, entries in framewise_labels.items()
                },
                peak_indices=peak_indices,
                config_direction=config.global_direction_computation
            )
            events.append(event)

        logger.info(f"Created {len(events)} GlobalEvents from framewise label blocks.")
        return events


