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
from scipy.spatial import ConvexHull


class Event:
    """
    Represents a single calcium propagation event across multiple cells.

    Attributes:
        id (int): Unique identifier.
        communications (List[CellToCellCommunication]): Communication links in the event.
        graph (nx.DiGraph): Directed propagation graph.
        n_cells (int): Number of unique cells involved.
        duration (int): Total duration of the event.
        avg_communication_time (float): Average time delay between origin and cause.
        framewise_active_labels (Dict[int, List[int]]): Cells active at each frame.
        wavefront (Dict[int, List[np.ndarray]]): Centroids of all cells active up to time t.
        shape_type (str): 'radial', 'longitudinal', or 'complex'.
        dag_type (str): 'chain', 'branched', or 'multi-origin'.
        propagation_speed (float): Speed of convex hull area growth over time.
        dominant_direction (Tuple[float, float]): Vector from early to late centroid.
    """

    def __init__(self, id: int, communications: List[CellToCellCommunication], label_to_cell: Dict[int, Cell], config: Dict) -> None:
        """
        Initialize an Event instance.

        Args:
            id (int): Unique event identifier.
            communications (List[CellToCellCommunication]): List of communications in the event.
            label_to_cell (Dict[int, Cell]): Mapping from cell labels to Cell objects.
            config (Dict): Configuration parameters for event classification.
        """
        self.id = id
        self.communications = communications
        self.label_to_cell = label_to_cell
        self.config = config

        self.graph = nx.DiGraph()
        self._build_graph()

        self.n_cells = self.graph.number_of_nodes()
        self.duration = self._compute_duration()
        self.avg_communication_time = self._compute_avg_dt()

        self.framewise_active_labels = self._compute_framewise_active_labels()
        self.wavefront = self._compute_incremental_wavefront()

        self.dominant_direction = self._compute_dominant_direction()
        self.propagation_speed = self._compute_area_growth_speed()

        self.shape_type = self._classify_shape_type()
        self.dag_type = self._classify_dag_type()

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

    def _compute_avg_dt(self) -> float:
        """
        Compute average communication delay.

        Returns:
            float: Mean delta_t between origin and cause.
        """
        deltas = [comm.delta_t for comm in self.communications if comm.delta_t > 0]
        return float(np.mean(deltas)) if deltas else 0.0

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
            centroids = [self.label_to_cell[l].centroid for l in activated]
            wavefront[t] = centroids
        return wavefront

    def _compute_dominant_direction(self) -> Tuple[float, float]:
        """
        Compute dominant direction as the mean of all origin-to-cause propagation vectors.

        Returns:
            Tuple[float, float]: Mean direction vector of event propagation.
        """
        vectors = []

        for comm in self.communications:
            origin = self.label_to_cell.get(comm.origin[0])
            cause = self.label_to_cell.get(comm.cause[0])
            if origin is None or cause is None:
                continue

            vec = np.array(cause.centroid) - np.array(origin.centroid)
            vectors.append(vec)

        if not vectors:
            return (0.0, 0.0)

        mean_vector = np.mean(vectors, axis=0)
        return tuple(mean_vector)

    def _compute_area_growth_speed(self) -> float:
        """
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

    def _compute_pca_ratio(self, centroids: List[np.ndarray]) -> float:
        """
        Compute elongation from PCA on centroids.

        Args:
            centroids (List[np.ndarray]): List of cell centroids.

        Returns:
            float: PCA axis ratio (elongation).
        """
        X = np.vstack(centroids)
        X -= np.mean(X, axis=0)
        _, S, _ = np.linalg.svd(X)
        return S[0] / S[1] if len(S) >= 2 and S[1] > 0 else 1.0

    def _classify_shape_type(self) -> str:
        """
        Classify the spatial shape: radial, longitudinal, complex.

        Returns:
            str: Morphological propagation shape.
        """
        thresholds = self.config["shape_classification"]
        radial_th = thresholds.get("pca_ratio_threshold_radial", 1.5)
        long_th = thresholds.get("pca_ratio_threshold_longitudinal", 3.0)

        try:
            all_points = [pt for pts in self.wavefront.values() for pt in pts]
            if len(all_points) < 3:
                return "undefined"
            ratio = self._compute_pca_ratio(all_points)
            if ratio < radial_th:
                return "radial"
            elif ratio >= long_th:
                return "longitudinal"
            else:
                return "complex"
        except:
            return "undefined"

    def _classify_dag_type(self) -> str:
        """
        Classify DAG topology: chain, branched, multi-origin.

        Returns:
            str: DAG type based on roots and leaves.
        """
        roots = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        leaves = [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]
        if len(roots) > 1:
            return "multi-origin"
        elif len(roots) == 1 and len(leaves) == 1:
            return "chain"
        else:
            return "branched"

    def __repr__(self) -> str:
        return (
            f"<Event id={self.id}, cells={self.n_cells}, duration={self.duration}, "
            f"Δt={self.avg_communication_time:.2f}, shape={self.shape_type}, dag={self.dag_type}>"
        )


    @classmethod
    def from_communications(
        cls,
        communications: List[CellToCellCommunication],
        cells: List[Cell],
        config: Dict
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
            event = cls(counter, group_comms, label_to_cell, config)
            events.append(event)
            counter += 1

        return events

