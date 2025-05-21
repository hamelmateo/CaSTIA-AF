"""
Module for clustering calcium cell similarity matrices using various algorithms.

Example usage:
    >>> from calcium_activity_characterization.processing.clustering import ClusteringEngine
    >>> from config import CLUSTERING_PARAMETERS
    >>> engine = ClusteringEngine(CLUSTERING_PARAMETERS)
    >>> labels = engine.run(similarity_matrices)
"""

import numpy as np
from typing import List, Optional
from sklearn.cluster import DBSCAN, AgglomerativeClustering, AffinityPropagation
import hdbscan
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class ClusteringEngine:
    """
    Class to apply clustering algorithms to similarity matrices.

    Supports DBSCAN, HDBSCAN, Agglomerative, AffinityPropagation,
    and Graph-based community clustering.
    """

    def __init__(self, config: dict):
        """
        Initialize the ClusteringEngine with configuration parameters.

        Args:
            config (dict): CLUSTERING_PARAMETERS dictionary.
        """
        self.method = config["method"]
        self.params = config["params"].get(self.method, {})

    def run(self, similarity_matrices: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply clustering on a list of similarity matrices.

        Args:
            similarity_matrices (List[np.ndarray]): List of similarity matrices (NxN).

        Returns:
            List[np.ndarray]: Cluster label arrays, one per matrix.
        """
        all_labels = []

        for idx, sim in enumerate(similarity_matrices):
            dist = 1.0 - sim
            np.fill_diagonal(dist, 0)

            if self.method == "dbscan":
                labels = self._run_dbscan(dist)

            elif self.method == "hdbscan":
                labels = self._run_hdbscan(dist)

            elif self.method == "agglomerative":
                labels = self._run_agglomerative(dist)

            elif self.method == "affinity_propagation":
                labels = self._run_affinity(sim)

            elif self.method == "graph_community":
                labels = self._run_graph_community(sim)

            else:
                raise ValueError(f"Unsupported clustering method: {self.method}")

            all_labels.append(labels)
            self._log_cluster_stats(labels, idx)

        return all_labels

    def _run_dbscan(self, dist: np.ndarray) -> np.ndarray:
        eps = self.params.get("eps", 0.03)
        min_samples = self.params.get("min_samples", 3)
        metric = self.params.get("metric", "precomputed")

        model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        return model.fit_predict(dist)

    def _run_hdbscan(self, dist: np.ndarray) -> np.ndarray:
        min_cluster_size = self.params.get("min_cluster_size", 3)
        min_samples = self.params.get("min_samples", 3)
        metric = self.params.get("metric", "precomputed")
        method = self.params.get("clustering_method", "eom")
        prob_thresh = self.params.get("probability_threshold", 0.85)

        model = hdbscan.HDBSCAN(
            min_samples=min_samples,
            metric=metric,
            min_cluster_size=min_cluster_size,
            cluster_selection_method=method
        )
        labels = model.fit_predict(dist)
        probabilities = model.probabilities_
        labels[probabilities < prob_thresh] = -1
        return labels

    def _run_agglomerative(self, dist: np.ndarray) -> np.ndarray:
        n_clusters = self.params.get("n_clusters")
        distance_threshold = self.params.get("distance_threshold", 0.5)
        metric = self.params.get("metric", "precomputed")
        linkage = self.params.get("linkage", "complete")
        auto_threshold = self.params.get("auto_threshold", False)

        if auto_threshold:
            distance_threshold = self._find_dendrogram_jump_threshold(dist, linkage)

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            metric=metric,
            linkage=linkage
        )
        labels = model.fit_predict(dist)

        label_counts = Counter(labels)
        labels = np.array([label if label_counts[label] > 1 else -1 for label in labels])

        return labels

    def _run_affinity(self, sim: np.ndarray) -> np.ndarray:
        preference = float(self.params.get("preference", np.median(sim)))
        damping = float(self.params.get("damping", 0.9))
        max_iter = int(self.params.get("max_iter", 200))
        convergence_iter = int(self.params.get("convergence_iter", 15))
        affinity = self.params.get("affinity", "precomputed")

        np.fill_diagonal(sim, preference)
        model = AffinityPropagation(affinity=affinity, max_iter=max_iter, convergence_iter=convergence_iter, damping=damping, random_state=42)
        return model.fit_predict(sim)

    def _run_graph_community(self, sim: np.ndarray) -> np.ndarray:
        threshold = float(self.params.get("similarity_threshold", 0.7))
        G = nx.Graph()
        for i in range(len(sim)):
            for j in range(i + 1, len(sim)):
                if sim[i, j] >= threshold:
                    G.add_edge(i, j, weight=sim[i, j])

        communities = list(greedy_modularity_communities(G))
        labels = np.full(len(sim), -1)
        for cluster_id, group in enumerate(communities):
            for idx in group:
                labels[idx] = cluster_id

        return labels

    def _log_cluster_stats(self, labels: np.ndarray, idx: int):
        counts = Counter(labels)
        n_clusters = sum(1 for k in counts if k != -1)
        n_noise = counts.get(-1, 0)
        logger.info(f"[Window {idx}] Clusters: {n_clusters}, Noise: {n_noise}")

    def _find_dendrogram_jump_threshold(self, dist: np.ndarray, linkage_method: str, jump_window: int = 30) -> float:
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform

        condensed = squareform(dist)
        Z = linkage(condensed, method=linkage_method)
        merge_distances = Z[:, 2]

        if len(merge_distances) < jump_window:
            window = merge_distances
        else:
            window = merge_distances[-jump_window:]

        diffs = np.diff(window)
        max_jump_idx = np.argmax(diffs)
        suggested = window[max_jump_idx + 1]

        logger.info(f"[Agglomerative] Dendrogram jump: delta={diffs[max_jump_idx]:.4f} at threshold={suggested:.4f}")
        return suggested
