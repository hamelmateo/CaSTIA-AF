# experimental_pipeline.py
# Usage: from calcium_activity_characterization.experimental.experimental_pipeline import init_file_paths, run_peak_clustering, run_wave_propagation_analysis

from collections import Counter
import networkx as nx

from calcium_activity_characterization.experimental.event_detection.arcos_event_detection import ArcosEventDetector
from calcium_activity_characterization.experimental.analysis.correlation import CorrelationAnalyzer
from calcium_activity_characterization.experimental.event_detection.clustering import ClusteringEngine
from calcium_activity_characterization.experimental.analysis.causality import GCAnalyzer
from calcium_activity_characterization.experimental.event_detection.peak_clustering import PeakClusteringEngine
from calcium_activity_characterization.experimental.event_detection.spatial_event_clustering import SpatialEventClusteringEngine
from calcium_activity_characterization.experimental.analysis.umap_analysis import run_umap_on_cells
from calcium_activity_characterization.experimental.analysis.wave_propagation import WavePropagationAnalyzer
from calcium_activity_characterization.utilities.loader import (
    save_pickle_file, 
    load_pickle_file, 
    save_clusters_on_overlay, 
    plot_similarity_matrices, 
    get_config_with_fallback,
    plot_raster
)

import logging
logger = logging.getLogger(__name__)


def run_umap(self):
        """
        Run UMAP dimensionality reduction and save the embedding.
        """
        try:
            run_umap_on_cells(
                self.population.cells,
                self.umap_file_path,
                n_neighbors=5,
                min_dist=1,
                n_components=2,
                normalize=False,
            )
        except Exception as e:
            logger.error(f"UMAP processing failed: {e}")


def arcos_event_pipeline(self) -> None:
    """
    Run event detection using arcos4py on the processed DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the binarized data.
    """

    arcos_pipeline = ArcosEventDetector(
        bindata_params=get_config_with_fallback(self.config,"BINDATA_PARAMETERS"),
        tracking_params=get_config_with_fallback(self.config,"TRACKING_PARAMETERS")
    )

    events_df, lineage_tracker = arcos_pipeline.run(active_cells=self.population.cells)

    save_pickle_file(events_df, self.output_path / "tracked_events.pkl")
    save_pickle_file(lineage_tracker, self.output_path / "lineage_tracker.pkl")



def correlation_analysis(self):
    """
    Compute correlation matrices.

    Returns:
        List[np.ndarray]: List of correlation matrices, one per window.
    """
    if self.similarity_matrices_path.exists():
        self.population.similarity_matrices = load_pickle_file(self.similarity_matrices_path)
    
    else:
        analyzer = CorrelationAnalyzer(get_config_with_fallback(self.config,"CORRELATION_PARAMETERS"), self.DEVICE_CORES)
        self.population.similarity_matrices = analyzer.run(self.population.cells, single_window=False)
        
        logger.info(f"Similarity matrices computed for {len(self.population.cells)} active cells.")
        
        # Save the similarity matrices to a file
        save_pickle_file(self.population.similarity_matrices, self.similarity_matrices_path)

    # Save the similarity matrices
    plot_similarity_matrices(self.output_path, self.population.similarity_matrices)


def clustering_cells(self) -> None:
    """
    Cluster cells based on their similarity matrices over a specific time-window.
    """
    engine = ClusteringEngine(get_config_with_fallback(self.config,"CLUSTERING_PARAMETERS"))
    engine.run(self.population.similarity_matrices)
    labels = engine.get_labels()

    logger.info(f"Clustering completed for {len(self.population.cells)} active cells.")


    # Save the clustered labels on the overlay
    plot_raster(output_path=self.output_path, cells=self.population.cells, cluster_labels=labels[0], clustered=True)
    save_clusters_on_overlay(self.overlay_path, self.output_path, labels, self.population.cells)

    return 


def run_peak_clustering(self):
    """
    Run custom peak-based clustering algorithm and store clusters.
    """
    if self.peak_clusters_path.exists():
        self.population.peak_clusters = load_pickle_file(self.peak_clusters_path)

    else:
        clustering_engine = PeakClusteringEngine(get_config_with_fallback(self.config,"PEAK_CLUSTERING_PARAMETERS"))
        self.population.peak_clusters = clustering_engine.run(self.population.cells)

        logger.info(f"Custom peak clustering found {len(self.population.peak_clusters)} clusters.")


        # Log cluster size distribution
        size_counts = Counter(len(c.members) for c in self.population.peak_clusters)
        for size, count in sorted(size_counts.items()):
            logger.info(f"🔹 {count} clusters with {size} peak(s)")

        save_pickle_file(self.population.peak_clusters, self.peak_clusters_path)


def causality_analysis(self):
    """
    For each existing peak cluster, preprocess cell signals, run pairwise GC,
    and build + save a directed GC graph.
    """
    gc_output_dir = self.gc_graph
    gc_output_dir.mkdir(parents=True, exist_ok=True)

    # Analyzer + signal processor
    analyzer = GCAnalyzer(get_config_with_fallback(self.config, "GC_PARAMETERS"), self.DEVICE_CORES)

    for cluster in self.population.peak_clusters:
        logger.info(f"Processing cluster {cluster.id} with {len(cluster.members)} members from {cluster.start_time} to {cluster.end_time}")
        cells = [cell for cell, _ in cluster.members]
        center_time = int((cluster.start_time + cluster.end_time) // 2)

        # Preprocess each cell's trace for GC
        for cell in cells:
            cell.trace.process_trace("raw","gc_trace",get_config_with_fallback(self.config,"GC_PREPROCESSING"))
            cell.trace.default_version = "gc_trace"

        gc_matrix = analyzer.run(cells, center_time)
        if gc_matrix is None:
            continue

        # Build directed graph from GC matrix
        G = nx.DiGraph()
        labels = [cell.label for cell in cells]
        for i, src in enumerate(labels):
            G.add_node(src)
            for j, tgt in enumerate(labels):
                if i != j and gc_matrix[i, j] > 0:
                    G.add_edge(src, tgt, weight=gc_matrix[i, j])

        # Save graph
        save_path = gc_output_dir / f"gc_graph_cluster_{cluster.id:03d}.gpickle"
        save_pickle_file(G, save_path)
        print(f"Saved GC graph for cluster {cluster.id} → {save_path}")


def initialize_population_traces(self):
    """
    Initialize population traces by computing global, activity, and impulse traces.
    """
    self.initialize_global_trace()
    self.initialize_activity_trace()
    self.initialize_impulse_trace()

    logger.info("Population traces initialized successfully.")


def initialize_global_trace(self):
    """
    Compute the global trace from active cells and process it through smoothing,
    peak detection, binarization, and metadata extraction.
    """
    version = "raw"
    default_version = "smoothed"
    self.population.compute_global_trace(version=version , 
                                            default_version=default_version, 
                                            signal_processing_params=get_config_with_fallback(self.config,"POPULATION_TRACES_SIGNAL_PROCESSING_PARAMETERS"), 
                                            peak_detection_params=get_config_with_fallback(self.config, "GLOBAL_PEAK_DETECTION_PARAMETERS"))
    
    self.population.global_trace.plot_all_traces(self.output_path / "global_trace_summary.png")


def initialize_activity_trace(self):
    """
    Compute the global trace from active cells and process it through smoothing,
    peak detection, binarization, and metadata extraction.
    """
    default_version = "smoothed"
    self.population.compute_activity_trace(default_version=default_version,
                                            signal_processing_params=get_config_with_fallback(self.config,"POPULATION_TRACES_SIGNAL_PROCESSING_PARAMETERS"),
                                            peak_detection_params=get_config_with_fallback(self.config, "GLOBAL_PEAK_DETECTION_PARAMETERS"))

    self.population.activity_trace.plot_all_traces(self.output_path / "activity_trace_summary.png")


def initialize_impulse_trace(self):
    """
    Compute the impulse trace from active cells and process it through smoothing,
    peak detection, binarization, and metadata extraction.
    """
    default_version = "smoothed"
    self.population.compute_impulse_trace(default_version=default_version,
                                            signal_processing_params=get_config_with_fallback(self.config,"POPULATION_TRACES_SIGNAL_PROCESSING_PARAMETERS"),
                                            peak_detection_params=get_config_with_fallback(self.config, "IMPULSE_PEAK_DETECTION_PARAMETERS"))

    self.population.impulse_trace.plot_all_traces(self.output_path / "impulse_trace_summary.png")



def run_spatial_event_clustering(self):
    engine = SpatialEventClusteringEngine(get_config_with_fallback(self.config, "SPATIAL_CLUSTERING_PARAMETERS"))
    self.population.event_clusters = engine.run(self.population)
    
    engine.plot_clustered_raster(self.population, self.raster_plots_clustered)

    engine.plot_clusters_with_graph(
        population=self.population,
        overlay_path=self.overlay_path,
        output_dir=self.event_cluster_on_overlay
    )


def run_wave_propagation_analysis(self) -> None:
    """
    TODO: rework the conventions of the method and submethods.
    For each spatial peak cluster, compute wave propagation direction and speed,
    then save a trajectory plot overlaying the segmentation image.
    """

    output_dir = self.output_path / "cluster_trajectories"
    analyzer = WavePropagationAnalyzer()

    if not hasattr(self.population, "event_clusters") or not self.population.event_clusters:
        logger.warning("No event_clusters found in population. Skipping wave propagation analysis.")
        return

    for clusters in self.population.event_clusters:
        for cluster in clusters:
            analyzer.analyze_cluster_propagation(cluster)
            save_path = output_dir / f"cluster_{cluster.id:03d}_trajectory.png"
            analyzer.plot_cluster_trajectory(cluster, self.overlay_path, save_path)

    logger.info(f"✅ Wave propagation analysis completed and saved in {output_dir}")