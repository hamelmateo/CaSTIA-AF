"""Module defining the CalciumPipeline class for processing calcium imaging data.

Example:
    >>> from src.core.pipeline import CalciumPipeline
    >>> pipeline = CalciumPipeline(config)
    >>> pipeline.run(data_path, output_path)
"""

import numpy as np
import tifffile
import os
import logging
from pathlib import Path
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from collections import Counter
import networkx as nx

from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.data.populations import Population
from calcium_activity_characterization.utilities.loader import (
    preprocess_images,
    save_tif_image,
    rename_files_with_padding,
    load_images,
    crop_image,
    save_pickle_file,
    load_pickle_file,
    save_clusters_on_overlay,
    plot_similarity_matrices,
    plot_raster,
    plot_impulse_raster,
    get_config_with_fallback
)
from calcium_activity_characterization.processing.segmentation import segmented
from calcium_activity_characterization.processing.arcos_event_detection import ArcosEventDetector
from calcium_activity_characterization.analysis.umap_analysis import run_umap_on_cells
from calcium_activity_characterization.processing.correlation import CorrelationAnalyzer
from calcium_activity_characterization.processing.clustering import ClusteringEngine
from calcium_activity_characterization.processing.peak_clustering import PeakClusteringEngine
from calcium_activity_characterization.processing.causality import GCAnalyzer
from calcium_activity_characterization.processing.spatial_event_clustering import SpatialEventClusteringEngine
from calcium_activity_characterization.analysis.wave_propagation_analysis import WavePropagationAnalyzer
from calcium_activity_characterization.processing.peak_origin_assigner import PeakOriginAssigner

logger = logging.getLogger(__name__)


class CalciumPipeline:
    """
    Class that runs the calcium imaging analysis pipeline end-to-end.

    Args:
        config (dict): Dictionary of configuration parameters.
    """

    def __init__(self, config: dict):
        self.config = config
        self.DEVICE_CORES = os.cpu_count()
        
        # data
        self.population: Population = None

        # folder paths
        self.data_path: Path = None
        self.output_path: Path = None
        self.directory_name: Path = None

        # file paths
        self.fitc_file_pattern: Path = None
        self.hoechst_file_pattern: Path = None
        self.hoechst_img_path: Path = None
        self.fitc_img_path: Path = None
        self.nuclei_mask_path: Path = None
        self.overlay_path: Path = None
        self.cells_file_path: Path = None
        self.raw_traces_path: Path = None
        self.smoothed_traces_path: Path = None
        self.binary_traces_path: Path = None
        self.sequential_traces_path: Path = None
        self.similarity_matrices_path: Path = None
        self.arcos_input_df: Path = None
        self.peak_clusters_path: Path = None
        self.umap_file_path: Path = None
        self.population_level_metrics_path: Path = None
        self.spatial_neighbor_graph_path: Path = None
        self.gc_graph: Path = None

    def run(self, data_path: Path, output_path: Path) -> None:
        """
        Execute the full calcium imaging pipeline for one ISX folder.

        Args:
            data_path (Path): Path to the ISX folder (containing FITC/HOECHST subfolders).
            output_path (Path): Destination folder for processed results.
        """
        output_path.mkdir(parents=True, exist_ok=True)
        self._init_paths(data_path, output_path)
        self._segment_cells()
        self._compute_intensity()

        if get_config_with_fallback(self.config,"ARCOS_TRACKING"):
            self._arcos_event_pipeline()

        self._signal_processing_pipeline()
        self._binarization_pipeline()
        
        self._assign_peak_origins()

        self._initialize_population_traces()
        self._run_spatial_event_clustering()
        self._save_population_metadata_report()
        
        #self._run_wave_propagation_analysis()
        #self._run_peak_clustering()
        #self._causality_analysis()
        #self._run_umap()
        #self._correlation_analysis()
        #self._clustering_cells()



    def _init_paths(self, data_path: Path, output_path: Path) -> None:
        """
        Initialize and store all input/output paths and file patterns.

        Args:
            data_path (Path): Root path to ISX experiment.
            output_path (Path): Path to save all outputs for this experiment.
        """
        self.data_path = data_path
        self.output_path = output_path
        self.directory_name = data_path.parents[1].name

        self.fitc_file_pattern = rf"{self.directory_name}__w3FITC_t(\d+)\.TIF"
        self.hoechst_file_pattern = rf"{self.directory_name}__w2DAPI_t(\d+)\.TIF"

        self.hoechst_img_path = data_path / "HOECHST"
        self.fitc_img_path = data_path / "FITC"

        self.nuclei_mask_path = output_path / "nuclei_mask.TIF"
        self.overlay_path = output_path / "overlay.TIF"
        self.cells_file_path = output_path / "cells.pkl"
        self.raw_traces_path = output_path / "raw_active_cells.pkl"
        self.smoothed_traces_path = output_path / "smoothed_active_cells.pkl"
        self.binary_traces_path = output_path / "binarized_active_cells.pkl"
        self.sequential_traces_path = output_path / "sequential_active_cells.pkl"
        self.similarity_matrices_path = output_path / "similarity_matrices.pkl"
        self.arcos_input_df = output_path / "arcos_input_df.pkl"
        self.peak_clusters_path = output_path / "peak_clusters.pkl"
        self.umap_file_path = output_path / "umap.npy"
        self.gc_graph = self.output_path / "gc_graphs"
        self.population_level_metrics_path = output_path / "population_metrics.pdf"
        self.spatial_neighbor_graph_path = output_path / "spatial_neighbor_graph.png"
        self.event_cluster_on_overlay = self.output_path / "event_cluster_overlays"
        self.raster_plots_clustered = self.output_path / "cluster_rasters"

    def _segment_cells(self):
        """
        Perform segmentation on DAPI images if needed and convert the mask to Cell objects.
        Reloads from file if available and permitted.
        """
        if not self.cells_file_path.exists():
            if not self.nuclei_mask_path.exists():
                nuclei_mask = segmented(
                    preprocess_images(
                        self.hoechst_img_path,
                        get_config_with_fallback(self.config,"ROI_SCALE"),
                        self.hoechst_file_pattern,
                        get_config_with_fallback(self.config,"PADDING")
                    ),
                    self.overlay_path,
                    get_config_with_fallback(self.config,"SAVE_OVERLAY")
                )
                save_tif_image(nuclei_mask, self.nuclei_mask_path)
            else:
                nuclei_mask = load_images(self.nuclei_mask_path)

            unfiltered_cells = self._convert_mask_to_cells(nuclei_mask)

            cells = [cell for cell in unfiltered_cells if cell.is_valid]

            self.population = Population(cells=cells, mask=nuclei_mask, output_path=self.output_path)
            save_pickle_file(self.population, self.cells_file_path)
            logger.info(f"Kept {len(cells)} active cells out of {len(unfiltered_cells)} total cells.")
        else:
            self.population = load_pickle_file(self.cells_file_path)


    def _convert_mask_to_cells(self, nuclei_mask: np.ndarray) -> List[Cell]:
        """
        Convert a labeled segmentation mask into a list of Cell objects.

        Args:
            nuclei_mask (np.ndarray): Labeled mask of the segmented nuclei.

        Returns:
            List[Cell]: List of Cell objects.
        """
        cells = []
        label = 1
        while np.any(nuclei_mask == label):
            pixel_coords = np.argwhere(nuclei_mask == label)
            if pixel_coords.size > 0:
                centroid = np.array(np.mean(pixel_coords, axis=0), dtype=int)
                cell = Cell(label=label, centroid=centroid, pixel_coords=pixel_coords, small_object_threshold=get_config_with_fallback(self.config,"SMALL_OBJECT_THRESHOLD"), big_object_threshold=get_config_with_fallback(self.config,"BIG_OBJECT_THRESHOLD"))
                if (
                    centroid[0] < 20 or centroid[1] < 20 or
                    centroid[0] > nuclei_mask.shape[0] - 20 or
                    centroid[1] > nuclei_mask.shape[1] - 20
                ):
                    cell.is_valid = False
                cells.append(cell)
            label += 1
        return cells

    def _compute_intensity(self):
        """
        Compute raw intensity traces for all cells, either serially or in parallel.
        Reloads from pickle if permitted.
        """
        if not self.raw_traces_path.exists():
            if get_config_with_fallback(self.config,"PARALLELIZE"):
                self._get_intensity_parallel()
            else:
                self._get_intensity_serial()
            save_pickle_file(self.population, self.raw_traces_path)
        else:
            self.population = load_pickle_file(self.raw_traces_path)

    def _get_intensity_serial(self):
        """
        Compute cell-wise intensity in serial over timepoints.
        """
        calcium_imgs = preprocess_images(
            self.fitc_img_path,
            get_config_with_fallback(self.config,"ROI_SCALE"),
            self.fitc_file_pattern,
            get_config_with_fallback(self.config,"PADDING")
        )
        if calcium_imgs.size > 0:
            for img in calcium_imgs:
                for cell in self.population.cells:
                    cell.add_mean_intensity(img)

    def _get_intensity_parallel(self):
        """
        Compute mean cell intensities in parallel using multiprocessing.
        """
        print("🔄 Computing mean cell intensities in parallel...")
        rename_files_with_padding(self.fitc_img_path, self.fitc_file_pattern, get_config_with_fallback(self.config,"PADDING"))
        print("🔄 Renamed FITC files with padding.")
        image_paths = sorted(self.fitc_img_path.glob("*.TIF"))
        cell_coords = [cell.pixel_coords for cell in self.population.cells]
        func = partial(self._compute_intensity_single, cell_coords=cell_coords, roi_scale=get_config_with_fallback(self.config,"ROI_SCALE"))

        try:
            with ProcessPoolExecutor(max_workers=self.DEVICE_CORES) as executor:
                results = list(tqdm(executor.map(func, image_paths), total=len(image_paths)))
        except Exception as e:
            logger.error(f"Parallel intensity computation failed: {e}")
            raise

        results_per_cell = list(zip(*results))
        for cell, trace in zip(self.population.cells, results_per_cell):
            cell.trace.add_trace(list(map(int, trace)), "raw")

    @staticmethod
    def _compute_intensity_single(image_path: Path, cell_coords: List[np.ndarray], roi_scale: float) -> List[float]:
        """
        Compute mean pixel intensity per cell for a single image.

        Args:
            image_path (Path): Path to image.
            cell_coords (List[np.ndarray]): Coordinates of each cell.
            roi_scale (float): Scaling factor for cropping.

        Returns:
            List[float]: Mean intensities per cell.
        """
        img = tifffile.imread(str(image_path))
        img = crop_image(img, roi_scale)
        return [float(np.mean([img[y, x] for y, x in coords])) for coords in cell_coords]


    def _signal_processing_pipeline(self):
        """
        Run signal processing pipeline on all active cells.
        Reloads from file if permitted.
        """
        if self.smoothed_traces_path.exists():
            self.population = load_pickle_file(self.smoothed_traces_path)
            return

        else:
            for cell in self.population.cells:
                cell.trace.process_trace("raw","smoothed",get_config_with_fallback(self.config,"INDIV_SIGNAL_PROCESSING_PARAMETERS"))
                cell.trace.default_version = "smoothed"
            
            save_pickle_file(self.population, self.smoothed_traces_path)


    def _run_umap(self):
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


    def _arcos_event_pipeline(self) -> None:
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



    def _binarization_pipeline(self): 
        """
        Run peak detection on all active cells using parameters from config and binarize the traces.
        """
        if self.binary_traces_path.exists():
            self.population = load_pickle_file(self.binary_traces_path)
            return
        
        else:
            for cell in self.population.cells:
                cell.trace.detect_peaks(get_config_with_fallback(self.config, "INDIV_PEAK_DETECTION_PARAMETERS"))
                cell.trace.binarize_trace_from_peaks()

            
            logger.info(f"Peaks detected for {len(self.population.cells)} active cells.")
            save_pickle_file(self.population, self.binary_traces_path)

        # Plot the binarized traces
        plot_raster(self.output_path, self.population.cells)
        plot_impulse_raster(self.output_path, self.population.cells)



    def _correlation_analysis(self):
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



    def _clustering_cells(self) -> None:
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
    


    def _run_peak_clustering(self):
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



    def _causality_analysis(self):
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


    def _assign_peak_origins(self) -> None:
        """
        Assign the origin of each peak in the population based on the cell-to-cell communication model.
        """

        self.population.generate_cell_to_cell_communications(get_config_with_fallback(self.config, "MAX_COMMUNICATION_TIME"))
        save_pickle_file(self.population, self.sequential_traces_path)


    def _initialize_population_traces(self):
        """
        Initialize population traces by computing global, activity, and impulse traces.
        """
        self._initialize_global_trace()
        self._initialize_activity_trace()
        self._initialize_impulse_trace()

        logger.info("Population traces initialized successfully.")


    def _initialize_global_trace(self):
        """
        Compute the global trace from active cells and process it through smoothing,
        peak detection, binarization, and metadata extraction.
        """
        version = "raw"
        default_version = "smoothed"
        self.population.compute_global_trace(version=version , 
                                             default_version=default_version, 
                                             signal_processing_params=get_config_with_fallback(self.config,"GLOBAL_SIGNAL_PROCESSING_PARAMETERS"), 
                                             peak_detection_params=get_config_with_fallback(self.config, "GLOBAL_PEAK_DETECTION_PARAMETERS"))
        
        self.population.global_trace.plot_all_traces(self.output_path / "global_trace_summary.png")


    def _initialize_activity_trace(self):
        """
        Compute the global trace from active cells and process it through smoothing,
        peak detection, binarization, and metadata extraction.
        """
        default_version = "smoothed"
        self.population.compute_activity_trace(default_version=default_version,
                                               signal_processing_params=get_config_with_fallback(self.config,"GLOBAL_SIGNAL_PROCESSING_PARAMETERS"),
                                               peak_detection_params=get_config_with_fallback(self.config, "GLOBAL_PEAK_DETECTION_PARAMETERS"))

        self.population.activity_trace.plot_all_traces(self.output_path / "activity_trace_summary.png")


    def _initialize_impulse_trace(self):
        """
        Compute the impulse trace from active cells and process it through smoothing,
        peak detection, binarization, and metadata extraction.
        """
        default_version = "smoothed"
        self.population.compute_impulse_trace(default_version=default_version,
                                              signal_processing_params=get_config_with_fallback(self.config,"GLOBAL_SIGNAL_PROCESSING_PARAMETERS"),
                                              peak_detection_params=get_config_with_fallback(self.config, "IMPULSE_PEAK_DETECTION_PARAMETERS"))

        self.population.impulse_trace.plot_all_traces(self.output_path / "impulse_trace_summary.png")



    def _run_spatial_event_clustering(self):
        engine = SpatialEventClusteringEngine(get_config_with_fallback(self.config, "SPATIAL_CLUSTERING_PARAMETERS"))
        self.population.event_clusters = engine.run(self.population)
        
        engine.plot_clustered_raster(self.population, self.raster_plots_clustered)

        engine.plot_clusters_with_graph(
            population=self.population,
            overlay_path=self.overlay_path,
            output_dir=self.event_cluster_on_overlay
        )


    def _run_wave_propagation_analysis(self) -> None:
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


    def _save_population_metadata_report(self) -> None:
        """
        Compute and save population-level metadata as a multi-page PDF report.
        """
        try:
            self.population.compute_population_metrics()
            self.population.plot_metadata_summary(save_path=self.population_level_metrics_path)
            logger.info(f"✅ Population metadata summary saved to {self.population_level_metrics_path}")
        except Exception as e:
            logger.error(f"Failed to compute or save population metadata: {e}")

