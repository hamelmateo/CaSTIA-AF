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
from calcium_activity_characterization.data.traces import Trace
from calcium_activity_characterization.utilities.loader import (
    preprocess_images,
    save_tif_image,
    rename_files_with_padding,
    load_images,
    crop_image,
    save_pickle_file,
    load_cells_from_pickle,
    load_pickle_file,
    save_clusters_on_overlay,
    plot_similarity_matrices,
    plot_raster,
    get_config_with_fallback
)
from calcium_activity_characterization.processing.segmentation import segmented
from calcium_activity_characterization.processing.signal_processing import SignalProcessor
from calcium_activity_characterization.processing.arcos_event_detection import ArcosEventDetector
from calcium_activity_characterization.analysis.umap_analysis import run_umap_on_cells
from calcium_activity_characterization.data.peaks import PeakDetector
from calcium_activity_characterization.processing.correlation import CorrelationAnalyzer
from calcium_activity_characterization.processing.clustering import ClusteringEngine
from calcium_activity_characterization.processing.peak_clustering import PeakClusteringEngine
from calcium_activity_characterization.data.clusters import Cluster
from calcium_activity_characterization.processing.causality import GCAnalyzer
from calcium_activity_characterization.utilities.utils_trace import compute_global_trace

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
        self.cells: List[Cell] = []
        self.active_cells: List[Cell] = []
        self.global_trace: Trace = None
        #self.similarity_matrices: List[np.ndarray] = []
        #self.peak_clusters: List[Cluster] = []

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
        self.raw_cells_file_path: Path = None
        self.smoothed_cells_file_path: Path = None
        self.binarized_cells_file_path: Path = None
        self.similarity_matrices_path: Path = None
        self.arcos_input_df: Path = None
        self.peak_clusters_path: Path = None
        self.umap_file_path: Path = None
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
        
        self._initialize_and_process_global_trace()

        # Plot metadata and all traces for cell with label 678
        cell_678 = next((cell for cell in self.active_cells if cell.label == 678), None)
        if cell_678 is not None:
            cell_678.trace.plot_metadata(self.output_path / "cell_678_metadata.png")
            cell_678.trace.plot_all_traces(self.output_path / "cell_678_all_traces.png")
        else:
            logger.warning("Cell with label 678 not found among active cells.")
        
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

        self.fitc_file_pattern = rf"{self.directory_name}__w3FITC_t(\\d+).TIF"
        self.hoechst_file_pattern = rf"{self.directory_name}__w2DAPI_t(\\d+).TIF"

        self.hoechst_img_path = data_path / "HOECHST"
        self.fitc_img_path = data_path / "FITC"

        self.nuclei_mask_path = output_path / "nuclei_mask.TIF"
        self.overlay_path = output_path / "overlay.TIF"
        self.cells_file_path = output_path / "cells.pkl"
        self.raw_cells_file_path = output_path / "raw_active_cells.pkl"
        self.smoothed_cells_file_path = output_path / "smoothed_active_cells.pkl"
        self.binarized_cells_file_path = output_path / "binarized_active_cells.pkl"
        self.similarity_matrices_path = output_path / "similarity_matrices.pkl"
        self.arcos_input_df = output_path / "arcos_input_df.pkl"
        self.peak_clusters_path = output_path / "peak_clusters.pkl"
        self.umap_file_path = output_path / "umap.npy"
        self.gc_graph = self.output_path / "gc_graphs"

    def _segment_cells(self):
        """
        Perform segmentation on DAPI images if needed and convert the mask to Cell objects.
        Reloads from file if available and permitted.
        """
        if not get_config_with_fallback(self.config,"EXISTING_CELLS") or not self.cells_file_path.exists():
            if not get_config_with_fallback(self.config,"EXISTING_MASK") or not self.nuclei_mask_path.exists():
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

            self.cells = self._convert_mask_to_cells(nuclei_mask)
            save_pickle_file(self.cells, self.cells_file_path)
        else:
            self.cells = load_cells_from_pickle(self.cells_file_path, True)

        self.active_cells = [cell for cell in self.cells if cell.is_valid]
        logger.info(f"Kept {len(self.active_cells)} active cells out of {len(self.cells)}")

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
        if not get_config_with_fallback(self.config,"EXISTING_RAW_INTENSITY") or not self.raw_cells_file_path.exists():
            if get_config_with_fallback(self.config,"PARALLELELIZE"):
                self._get_intensity_parallel()
            else:
                self._get_intensity_serial()
            save_pickle_file(self.active_cells, self.raw_cells_file_path)
        else:
            self.active_cells = load_cells_from_pickle(self.raw_cells_file_path, True)

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
                for cell in self.active_cells:
                    cell.add_mean_intensity(img)

    def _get_intensity_parallel(self):
        """
        Compute mean cell intensities in parallel using multiprocessing.
        """
        rename_files_with_padding(self.fitc_img_path, self.fitc_file_pattern, get_config_with_fallback(self.config,"PADDING"))
        image_paths = sorted(self.fitc_img_path.glob("*.TIF"))
        cell_coords = [cell.pixel_coords for cell in self.active_cells]
        func = partial(self._compute_intensity_single, cell_coords=cell_coords, roi_scale=get_config_with_fallback(self.config,"ROI_SCALE"))

        try:
            with ProcessPoolExecutor(max_workers=self.DEVICE_CORES) as executor:
                results = list(tqdm(executor.map(func, image_paths), total=len(image_paths)))
        except Exception as e:
            logger.error(f"Parallel intensity computation failed: {e}")
            raise

        results_per_cell = list(zip(*results))
        for cell, trace in zip(self.active_cells, results_per_cell):
            cell.trace.raw = list(map(int, trace))

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
        if get_config_with_fallback(self.config,"EXISTING_PROCESSED_INTENSITY") and self.smoothed_cells_file_path.exists():
            self.active_cells = load_cells_from_pickle(self.smoothed_cells_file_path, True)
            return

        else:
            processor = SignalProcessor(params=get_config_with_fallback(self.config,"SIGNAL_PROCESSING_PARAMETERS"))

            for cell in self.active_cells:
                cell.trace.versions["smoothed"] = processor.run(cell.trace.raw)
                cell.trace.default_version = "smoothed"
            
            save_pickle_file(self.active_cells, self.smoothed_cells_file_path)


    def _run_umap(self):
        """
        Run UMAP dimensionality reduction and save the embedding.
        """
        try:
            run_umap_on_cells(
                self.active_cells,
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

        events_df, lineage_tracker = arcos_pipeline.run(active_cells=self.active_cells)

        save_pickle_file(events_df, self.output_path / "tracked_events.pkl")
        save_pickle_file(lineage_tracker, self.output_path / "lineage_tracker.pkl")



    def _binarization_pipeline(self): 
        """
        Run peak detection on all active cells using parameters from config and binarize the traces.
        """
        if get_config_with_fallback(self.config,"EXISTING_BINARIZED_INTENSITY") and self.binarized_cells_file_path.exists():
            self.active_cells = load_cells_from_pickle(self.binarized_cells_file_path, True)
            return
        
        else:
            detector = PeakDetector(params=get_config_with_fallback(self.config,"PEAK_DETECTION_PARAMETERS"))

            for cell in self.active_cells:
                cell.trace.detect_peaks(detector)
                cell.trace.binarize_trace_from_peaks()

            
            logger.info(f"Peaks detected for {len(self.active_cells)} active cells.")
            save_pickle_file(self.active_cells, self.binarized_cells_file_path)

        # Plot the binarized traces
        plot_raster(self.output_path, self.active_cells)



    def _correlation_analysis(self):
        """
        Compute correlation matrices.

        Returns:
            List[np.ndarray]: List of correlation matrices, one per window.
        """
        if get_config_with_fallback(self.config,"EXISTING_SIMILARITY_MATRICES") and self.similarity_matrices_path.exists():
            self.similarity_matrices = load_pickle_file(self.similarity_matrices_path)
        
        else:
            analyzer = CorrelationAnalyzer(get_config_with_fallback(self.config,"CORRELATION_PARAMETERS"), self.DEVICE_CORES)
            self.similarity_matrices = analyzer.run(self.active_cells, single_window=False)
            
            logger.info(f"Similarity matrices computed for {len(self.active_cells)} active cells.")
            
            # Save the similarity matrices to a file
            save_pickle_file(self.similarity_matrices, self.similarity_matrices_path)

        # Save the similarity matrices
        plot_similarity_matrices(self.output_path, self.similarity_matrices)



    def _clustering_cells(self) -> None:
        """
        Cluster cells based on their similarity matrices over a specific time-window.
        """
        engine = ClusteringEngine(get_config_with_fallback(self.config,"CLUSTERING_PARAMETERS"))
        engine.run(self.similarity_matrices)
        labels = engine.get_labels()

        logger.info(f"Clustering completed for {len(self.active_cells)} active cells.")


        # Save the clustered labels on the overlay
        plot_raster(output_path=self.output_path, cells=self.active_cells, cluster_labels=labels[0], clustered=True)
        save_clusters_on_overlay(self.overlay_path, self.output_path, labels, self.active_cells)

        return 
    


    def _run_peak_clustering(self):
        """
        Run custom peak-based clustering algorithm and store clusters.
        """
        if get_config_with_fallback(self.config,"EXISTING_PEAK_CLUSTERS") and self.peak_clusters_path.exists():
            self.peak_clusters = load_pickle_file(self.peak_clusters_path)

        else:
            clustering_engine = PeakClusteringEngine(get_config_with_fallback(self.config,"PEAK_CLUSTERING_PARAMETERS"))
            self.peak_clusters = clustering_engine.run(self.active_cells)

            logger.info(f"Custom peak clustering found {len(self.peak_clusters)} clusters.")


            # Log cluster size distribution
            size_counts = Counter(len(c.members) for c in self.peak_clusters)
            for size, count in sorted(size_counts.items()):
                logger.info(f"🔹 {count} clusters with {size} peak(s)")

            save_pickle_file(self.peak_clusters, self.peak_clusters_path)



    def _causality_analysis(self):
        """
        For each existing peak cluster, preprocess cell signals, run pairwise GC,
        and build + save a directed GC graph.
        """
        gc_output_dir = self.gc_graph
        gc_output_dir.mkdir(parents=True, exist_ok=True)

        # Analyzer + signal processor
        analyzer = GCAnalyzer(get_config_with_fallback(self.config, "GC_PARAMETERS"), self.DEVICE_CORES)
        processor = SignalProcessor(get_config_with_fallback(self.config, "GC_PREPROCESSING"))

        for cluster in self.peak_clusters:
            logger.info(f"Processing cluster {cluster.id} with {len(cluster.members)} members from {cluster.start_time} to {cluster.end_time}")
            cells = [cell for cell, _ in cluster.members]
            center_time = int((cluster.start_time + cluster.end_time) // 2)

            # Preprocess each cell's trace for GC
            for cell in cells:
                cell.trace.versions["gc_trace"] = processor.run(cell.trace.raw)

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


    def _initialize_and_process_global_trace(self):
        """
        Compute the global trace from active cells and process it through smoothing,
        peak detection, binarization, and metadata extraction.
        """
        version = "smoothed"
        self.global_trace = compute_global_trace(self.active_cells, version=version , default_version=version)

        processor = SignalProcessor(params=get_config_with_fallback(self.config, "SIGNAL_PROCESSING_PARAMETERS"))
        detector = PeakDetector(params=get_config_with_fallback(self.config, "PEAK_DETECTION_PARAMETERS"))

        self.global_trace.versions[version] = processor.run(self.global_trace.versions[version])
        self.global_trace.detect_peaks(detector)
        self.global_trace.binarize_trace_from_peaks()
        self.global_trace.plot_all_traces(self.output_path / "global_trace_summary.png")

