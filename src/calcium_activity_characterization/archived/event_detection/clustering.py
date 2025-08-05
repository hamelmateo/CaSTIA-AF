"""
Module for clustering calcium cell similarity matrices using various algorithms.

Example usage:
    >>> from calcium_activity_characterization.processing.clustering import ClusteringEngine
    >>> from config import CLUSTERING_PARAMETERS
    >>> engine = ClusteringEngine(CLUSTERING_PARAMETERS)
    >>> labels = engine.run(similarity_matrices)
"""
import numpy as np
from typing import list, tuple
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN, AgglomerativeClustering, AffinityPropagation
import hdbscan
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from collections import Counter
from calcium_activity_characterization.logger import logger
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform




class ClusteringEngine:
    """
    Class to apply clustering algorithms to similarity matrices.

    Stores final labels and provides access via `get_labels()`.
    """

    def __init__(self, config: dict):
        self.method = config["method"]
        self.params = config["params"].get(self.method, {})
        self.min_cluster_size = self.params.get("min_cluster_size", 3)
        self.labels: list[np.ndarray] = []  # list of np.ndarray per matrix

    def run(self, similarity_matrices: list[np.ndarray]):
        """
        Apply clustering to a list of similarity matrices.

        Returns:
            list[np.ndarray]: Final labels (after filtering) for each matrix.
        """

        for idx, sim in enumerate(similarity_matrices):
            dist = 1.0 - sim
            np.fill_diagonal(dist, 0)

            if self.method == "dbscan":
                self._run_dbscan(dist)
            elif self.method == "hdbscan":
                self._run_hdbscan(dist)
            elif self.method == "agglomerative":
                self._run_agglomerative(dist)
            elif self.method == "affinity_propagation":
                self._run_affinity(sim)
            elif self.method == "graph_community":
                self._run_graph_community(sim)
            else:
                raise ValueError(f"Unsupported clustering method: {self.method}")


            self._log_cluster_stats(idx)
            self._filter_small_clusters(idx)
            self._log_cluster_stats(idx)


    def get_labels(self) -> list[np.ndarray]:
        """Return filtered cluster labels from last run()."""
        return self.labels

    def _run_dbscan(self, dist: np.ndarray):
        eps = self.params.get("eps", 0.03)
        min_samples = self.params.get("min_samples", 3)
        metric = self.params.get("metric", "precomputed")
        
        self.labels.append(DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit_predict(dist))
         

    def _run_hdbscan(self, dist: np.ndarray) -> np.ndarray:
        min_cluster_size = self.params.get("min_cluster_size", 3)
        min_samples = self.params.get("min_samples", 3)
        method = self.params.get("clustering_method", "eom")
        prob_thresh = self.params.get("probability_threshold", 0.85)
        model = hdbscan.HDBSCAN(
            min_samples=min_samples,
            metric=self.params.get("metric", "precomputed"),
            min_cluster_size=min_cluster_size,
            cluster_selection_method=method
        )
        labels = model.fit_predict(dist)
        labels[model.probabilities_ < prob_thresh] = -1
        self.labels.append(labels)
        

    def _run_agglomerative(self, dist: np.ndarray):
        n_clusters = self.params.get("n_clusters")
        distance_threshold = self.params.get("distance_threshold", 0.5)
        metric = self.params.get("metric", "precomputed")
        linkage = self.params.get("linkage", "complete")
        auto_threshold = self.params.get("auto_threshold", False)

        if auto_threshold:
            distance_threshold = self.automated_threshold_detection(dist, linkage_method=linkage, verbose=True)
            logger.info(f"[Agglomerative] Auto-detected distance threshold: {distance_threshold:.4f}")

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            metric=metric,
            linkage=linkage
        )
        self.labels.append(model.fit_predict(dist))

    def _run_affinity(self, sim: np.ndarray) :
        preference = float(self.params.get("preference", np.median(sim)))
        damping = float(self.params.get("damping", 0.9))
        np.fill_diagonal(sim, preference)

        labels = AffinityPropagation(
            affinity=self.params.get("affinity", "precomputed"),
            max_iter=int(self.params.get("max_iter", 200)),
            convergence_iter=int(self.params.get("convergence_iter", 15)),
            damping=damping,
            random_state=42
        ).fit_predict(sim)

        self.labels.append(labels)

    def _run_graph_community(self, sim: np.ndarray):
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
        
        self.labels.append(labels)

    def _filter_small_clusters(self, idx: int):
        label_counts = Counter(self.labels[idx])
        filtered = np.array([
            label if label != -1 and label_counts[label] >= self.min_cluster_size else -1
            for label in self.labels[idx]
        ])
        logger.info(f"[Window {idx}] Filtering clusters smaller than {self.min_cluster_size} cells.")
        self.labels[idx] = filtered

    def _log_cluster_stats(self, idx: int):
        counts = Counter(self.labels[idx])
        n_clusters = sum(1 for k in counts if k != -1)
        n_noise = counts.get(-1, 0)
        logger.info(f"[Window {idx}] Clusters: {n_clusters}, Noise: {n_noise}")

    def _find_dendrogram_jump_threshold(self, dist: np.ndarray, linkage_method: str, jump_window: int = 30) -> float:
        condensed = squareform(dist)
        Z = linkage(condensed, method=linkage_method)
        merge_distances = Z[:, 2]
        window = merge_distances[-jump_window:] if len(merge_distances) >= jump_window else merge_distances
        diffs = np.diff(window)
        max_jump_idx = np.argmax(diffs)
        suggested = window[max_jump_idx + 1]
        logger.info(f"[Agglomerative] Dendrogram jump: delta={diffs[max_jump_idx]:.4f} at threshold={suggested:.4f}")
        return suggested
    
    
    def automated_threshold_detection(
        self,
        dist_matrix: np.ndarray,
        linkage_method: str = "average",
        jump_window: int = 30,
        threshold_sweep_range: float = 0.2,
        threshold_sweep_step: float = 0.02,
        verbose: bool = False) -> float:
        """
        Automatically determine best distance threshold using dendrogram jump + silhouette validation.

        Args:
            dist_matrix (np.ndarray): Pairwise distance matrix (square form).
            linkage_method (str): Linkage method for clustering ('average', 'complete', etc.).
            jump_window (int): Number of last merges to consider for jump detection.
            threshold_sweep_range (float): Percent range (+/-) around jump threshold to sweep.
            threshold_sweep_step (float): Step size for threshold sweep.
            min_cluster_size (int): Minimum number of points in a valid cluster.
            verbose (bool): Print debug information.

        Returns:
            float: Best threshold value.
        """
        # Step 1: Linkage
        condensed = squareform(dist_matrix)
        Z = linkage(condensed, method=linkage_method)
        merge_distances = Z[:, 2]
        window = merge_distances[-jump_window:] if len(merge_distances) >= jump_window else merge_distances

        # Step 2: Jump Detection
        diffs = np.diff(window)
        jump_idx = np.argmax(diffs)
        jump_threshold = window[jump_idx + 1]
        if verbose:
            logger.info(f"[Jump Detection] Max jump Δ={diffs[jump_idx]:.4f} at threshold={jump_threshold:.4f}")

        # Step 3: Silhouette Sweep
        sweep_min = max(0.01, jump_threshold * (1 - threshold_sweep_range))
        sweep_max = min(1.0, jump_threshold * (1 + threshold_sweep_range))
        thresholds = np.arange(sweep_min, sweep_max + threshold_sweep_step, threshold_sweep_step)

        best_score = -1.0
        best_threshold = jump_threshold

        for thr in thresholds:
            try:
                model = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=thr,
                    metric="precomputed",
                    linkage=linkage_method
                )
                labels = model.fit_predict(dist_matrix)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                cluster_sizes = [np.sum(labels == l) for l in set(labels) if l != -1]
                #if n_clusters < 2 or any(size < self.min_cluster_size for size in cluster_sizes) or n_clusters > 30:
                #    continue

                score = silhouette_score(dist_matrix, labels, metric='precomputed')

                if verbose:
                    logger.info(f"[Sweep] thr={thr:.3f}, silhouette={score:.3f}, clusters={n_clusters}")

                if score > best_score:
                    best_score = score
                    best_threshold = thr

            except Exception as e:
                if verbose:
                    logger.warning(f"Skipping threshold {thr:.3f}: {e}")

        return best_threshold