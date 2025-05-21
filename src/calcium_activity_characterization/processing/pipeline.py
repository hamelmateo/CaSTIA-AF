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
from scipy.signal import correlate
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from collections import Counter
import hdbscan
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from calcium_activity_characterization.data.cells import Cell
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
    plot_raster
)
from calcium_activity_characterization.processing.segmentation import segmented
from calcium_activity_characterization.processing.signal_processing import SignalProcessor
from calcium_activity_characterization.processing.event_detection import ArcosEventDetector
from calcium_activity_characterization.analysis.umap_analysis import run_umap_on_cells
from calcium_activity_characterization.data.peaks import PeakDetector


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
        self.similarity_matrices: List[np.ndarray] = []

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
        self.processed_cells_file_path: Path = None
        self.umap_file_path: Path = None

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

        if self.config["ARCOS_TRACKING"]:
            self._arcos_event_pipeline()

        self._signal_processing_pipeline()
        self._binarization_pipeline()
        self._correlation_analysis()
        self._clustering_cells()



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
        self.processed_cells_file_path = output_path / "processed_active_cells.pkl"
        self.binarized_cells_file_path = output_path / "binarized_active_cells.pkl"
        self.similarity_matrices_path = output_path / "similarity_matrices.pkl"
        self.arcos_input_df = output_path / "arcos_input_df.pkl"
        self.umap_file_path = output_path / "umap.npy"

    def _segment_cells(self):
        """
        Perform segmentation on DAPI images if needed and convert the mask to Cell objects.
        Reloads from file if available and permitted.
        """
        if not self.config["EXISTING_CELLS"] or not self.cells_file_path.exists():
            if not self.config["EXISTING_MASK"] or not self.nuclei_mask_path.exists():
                nuclei_mask = segmented(
                    preprocess_images(
                        self.hoechst_img_path,
                        self.config["ROI_SCALE"],
                        self.hoechst_file_pattern,
                        self.config["PADDING"]
                    ),
                    self.overlay_path,
                    self.config["SAVE_OVERLAY"]
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
                cell = Cell(label=label, centroid=centroid, pixel_coords=pixel_coords, small_object_threshold=self.config["SMALL_OBJECT_THRESHOLD"], big_object_threshold=self.config["BIG_OBJECT_THRESHOLD"])
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
        if not self.config["EXISTING_RAW_INTENSITY"] or not self.raw_cells_file_path.exists():
            if self.config["PARALLELELIZE"]:
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
            self.config["ROI_SCALE"],
            self.fitc_file_pattern,
            self.config["PADDING"]
        )
        if calcium_imgs.size > 0:
            for img in calcium_imgs:
                for cell in self.active_cells:
                    cell.add_mean_intensity(img)

    def _get_intensity_parallel(self):
        """
        Compute mean cell intensities in parallel using multiprocessing.
        """
        rename_files_with_padding(self.fitc_img_path, self.fitc_file_pattern, self.config["PADDING"])
        image_paths = sorted(self.fitc_img_path.glob("*.TIF"))
        cell_coords = [cell.pixel_coords for cell in self.active_cells]
        func = partial(self._compute_intensity_single, cell_coords=cell_coords, roi_scale=self.config["ROI_SCALE"])

        try:
            with ProcessPoolExecutor(max_workers=self.DEVICE_CORES) as executor:
                results = list(tqdm(executor.map(func, image_paths), total=len(image_paths)))
        except Exception as e:
            logger.error(f"Parallel intensity computation failed: {e}")
            raise

        results_per_cell = list(zip(*results))
        for cell, trace in zip(self.active_cells, results_per_cell):
            cell.raw_intensity_trace = list(map(int, trace))

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
        if self.config["EXISTING_PROCESSED_INTENSITY"] and self.processed_cells_file_path.exists():
            self.active_cells = load_cells_from_pickle(self.processed_cells_file_path, True)
            return

        else:
            processor = SignalProcessor(params=self.config["SIGNAL_PROCESSING_PARAMETERS"], pipeline=self.config["SIGNAL_PROCESSING"])

            for cell in self.active_cells:
                cell.processed_intensity_trace = processor.run(cell.raw_intensity_trace)
            
            save_pickle_file(self.active_cells, self.processed_cells_file_path)


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
            bindata_params=self.config["BINDATA_PARAMETERS"],
            tracking_params=self.config["TRACKING_PARAMETERS"]
        )

        events_df, lineage_tracker = arcos_pipeline.run(active_cells=self.active_cells)

        save_pickle_file(events_df, self.output_path / "tracked_events.pkl")
        save_pickle_file(lineage_tracker, self.output_path / "lineage_tracker.pkl")



    def _binarization_pipeline(self): 
        """
        Run peak detection on all active cells using parameters from config and binarize the traces.
        """
        if self.config["EXISTING_BINARIZED_INTENSITY"] and self.binarized_cells_file_path.exists():
            self.active_cells = load_cells_from_pickle(self.binarized_cells_file_path, True)
            return
        
        else:
            detector = PeakDetector(params=self.config.get("PEAK_DETECTION", {}))

            for cell in self.active_cells:
                cell.detect_peaks(detector)
                cell.binarize_trace_from_peaks()

            
            logger.info(f"Peaks detected for {len(self.active_cells)} active cells.")
            save_pickle_file(self.active_cells, self.binarized_cells_file_path)

        # Plot the binarized traces
        plot_raster(self.output_path, self.active_cells)


 


    def _compute_cross_corr(self, traces: np.ndarray, lag_range: int) -> np.ndarray:
            """
            Compute the cross-correlation similarity matrix for a given set of traces.

            Args:
                traces (np.ndarray): Array of shape (num_cells, num_timepoints).

            Returns:
                np.ndarray: Similarity matrix of shape (num_cells, num_cells).
            """
            mode = self.config["CORRELATION_PARAMETERS"]["params"]["cross_correlation"]["mode"]
            method = self.config["CORRELATION_PARAMETERS"]["params"]["cross_correlation"]["method"]

            num_cells = len(traces)
            sim_matrix = np.zeros((num_cells, num_cells))

            for i in range(num_cells):
                for j in range(i, num_cells):
                    trace_i = traces[i]
                    trace_j = traces[j]

                    corr = correlate(trace_i, trace_j, mode=mode, method=method)
                    lags = np.arange(-len(trace_j) + 1, len(trace_i))
                    valid = (lags >= -lag_range) & (lags <= lag_range)

                    max_corr = np.max(corr[valid]) / (np.sqrt(np.sum(trace_i)) * np.sqrt(np.sum(trace_j)) + 1e-8)
                    sim_matrix[i, j] = sim_matrix[j, i] = max_corr

            return sim_matrix


    def _compute_jaccard(self, traces: np.ndarray) -> np.ndarray:
        """
        Compute the jaccard similarity matrix for a given set of traces.

        Args:
            traces (np.ndarray): Array of shape (num_cells, num_timepoints).

        Returns:
            np.ndarray: Similarity matrix of shape (num_cells, num_cells).
        """
        n_cells = traces.shape[0]
        sim_matrix = np.zeros((n_cells, n_cells), dtype=float)

        for i in range(n_cells):
            for j in range(i, n_cells):
                a = traces[i]
                b = traces[j]
                intersection = np.logical_and(a, b).sum()
                union = np.logical_or(a, b).sum()
                similarity = intersection / union if union != 0 else 0.0
                sim_matrix[i, j] = similarity
                sim_matrix[j, i] = similarity 

        return sim_matrix


    def _correlation_analysis(self):
        """
        Compute correlation matrices.

        Returns:
            List[np.ndarray]: List of correlation matrices, one per window.
        """
        if self.config["EXISTING_SIMILARITY_MATRICES"] and self.similarity_matrices_path.exists():
            self.similarity_matrices = load_pickle_file(self.similarity_matrices_path)
        
        else:
            method = self.config["CORRELATION_PARAMETERS"]["method"]

            if method == "cross_correlation":
                window_size = self.config["CORRELATION_PARAMETERS"]["params"]["cross_correlation"]["window_size"]
                lag_percent = self.config["CORRELATION_PARAMETERS"]["params"]["cross_correlation"]["lag_percent"]
                step_percent = self.config["CORRELATION_PARAMETERS"]["params"]["cross_correlation"]["step_percent"]

                binary_traces = [np.array(cell.binary_trace, dtype=int) for cell in self.active_cells]
                trace_length = len(binary_traces[0])

                # Pre-slice all windows
                all_window_traces = [
                    [trace[start:start + window_size] for trace in binary_traces]
                    for start in range(0, trace_length - window_size + 1, int(step_percent * window_size))
                ]

                with ProcessPoolExecutor(max_workers=self.DEVICE_CORES) as executor:
                        self.similarity_matrices = list(tqdm(
                            executor.map(partial(self._compute_cross_corr, lag_range=int(lag_percent * window_size)), all_window_traces),
                            total=len(all_window_traces),
                            desc="Computing similarity matrices in parallel"
                        ))

            elif method == "jaccard":
                window_size = self.config["CORRELATION_PARAMETERS"]["params"]["jaccard"]["window_size"]
                lag_percent = self.config["CORRELATION_PARAMETERS"]["params"]["jaccard"]["lag_percent"]
                step_percent = self.config["CORRELATION_PARAMETERS"]["params"]["jaccard"]["step_percent"]

                binary_traces = [np.array(cell.binary_trace, dtype=int) for cell in self.active_cells]
                trace_length = len(binary_traces[0])

                # Pre-slice all windows
                all_window_traces = [
                    [trace[start:start + window_size] for trace in binary_traces]
                    for start in range(0, trace_length - window_size + 1, int(step_percent * window_size))
                ]
                with ProcessPoolExecutor(max_workers=self.DEVICE_CORES) as executor:
                    self.similarity_matrices = list(tqdm(
                        executor.map(self._compute_jaccard, all_window_traces),
                        total=len(all_window_traces),
                        desc="Computing similarity matrices in parallel"
                    ))
        
            else:
                raise ValueError(f"Unsupported similarity method: {method}")
            
            # Save the similarity matrices to a file
            save_pickle_file(self.similarity_matrices, self.similarity_matrices_path)

        # Save the similarity matrices
        plot_similarity_matrices(self.output_path, self.similarity_matrices)



    def _clustering_cells(self) -> None:
        """
        Cluster cells based on their similarity matrices over a specific time-window.
        """
        clustered_labels = []
        method = self.config["CLUSTERING_PARAMETERS"]["method"]

        for sim in self.similarity_matrices:
            if method == "dbscan":
                eps = self.config["CLUSTERING_PARAMETERS"]["params"]["dbscan"]["eps"]
                min_samples = self.config["CLUSTERING_PARAMETERS"]["params"]["dbscan"]["min_samples"]
                metric = self.config["CLUSTERING_PARAMETERS"]["params"]["dbscan"]["metric"]

                dist = 1.0 - sim  # convert similarity to distance
                clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
                labels = clustering.fit_predict(dist)
                clustered_labels.append(labels)

            elif method == "hdbscan":
                min_cluster_size = self.config["CLUSTERING_PARAMETERS"]["params"]["hdbscan"]["min_cluster_size"]
                min_samples = self.config["CLUSTERING_PARAMETERS"]["params"]["hdbscan"]["min_samples"]
                metric = self.config["CLUSTERING_PARAMETERS"]["params"]["hdbscan"]["metric"]
                clustering_method = self.config["CLUSTERING_PARAMETERS"]["params"]["hdbscan"]["clustering_method"]
                probability_threshold = self.config["CLUSTERING_PARAMETERS"]["params"]["hdbscan"]["probability_threshold"]
                metric = self.config["CLUSTERING_PARAMETERS"]["params"]["hdbscan"]["metric"]

                dist = 1.0 - sim
                clustering = hdbscan.HDBSCAN(
                    min_samples=min_samples,
                    metric=metric,
                    min_cluster_size=min_cluster_size,
                    cluster_selection_method=clustering_method
                    )
                labels = clustering.fit_predict(dist)
                probabilities = clustering.probabilities_
                labels[probabilities < probability_threshold] = -1  # Mark low-probability points as noise
                clustered_labels.append(labels)

            elif method == "agglomerative":
                n_clusters = self.config["CLUSTERING_PARAMETERS"]["params"]["agglomerative"]["n_clusters"]
                distance_threshold = self.config["CLUSTERING_PARAMETERS"]["params"]["agglomerative"]["distance_threshold"]
                metric = self.config["CLUSTERING_PARAMETERS"]["params"]["agglomerative"]["metric"]
                linkage = self.config["CLUSTERING_PARAMETERS"]["params"]["agglomerative"]["linkage"]
                auto_threshold = self.config["CLUSTERING_PARAMETERS"]["params"]["agglomerative"]["auto_threshold"]
                
                dist = 1.0 - sim
                np.fill_diagonal(dist, 0)

                if auto_threshold:
                    #best_threshold = find_best_agglomerative_threshold(dist)
                    best_threshold = find_dendrogram_jump_threshold(dist, linkage_method=linkage)                        
                    distance_threshold = best_threshold

                
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    distance_threshold=distance_threshold,
                    metric=metric,
                    linkage=linkage
                )

                labels = clustering.fit_predict(dist)

                # Filter out single cell clusters
                label_counts = Counter(labels)
                labels = np.array([label if label_counts[label] > 1 else -1 for label in labels])

                clustered_labels.append(labels)

            else:
                raise ValueError(f"Unsupported clustering method: {method}")


            for idx, labels in enumerate(clustered_labels):
                label_counts = Counter(labels)
                num_clusters = sum(1 for k in label_counts if k != -1)
                num_noise = label_counts.get(-1, 0)

                print(f"[Clustering Result] Clusters found: {num_clusters}, Noise cells: {num_noise}")
                for cluster_id, count in sorted(label_counts.items()):
                    name = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
                    print(f"    {name}: {count} cells")

        # Save the clustered labels on the overlay
        plot_raster(output_path=self.output_path, cells=self.active_cells, cluster_labels=labels, clustered=True)
        save_clusters_on_overlay(self.overlay_path, self.output_path, clustered_labels, self.active_cells)

        return 
    


def find_best_agglomerative_threshold(dist: np.ndarray) -> float:
    best_score = -1
    best_threshold = None
    for threshold in np.arange(0.05, 1.0, 0.05):
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                metric="precomputed",
                linkage="complete"
            )
            labels = clustering.fit_predict(dist)
            if len(set(labels)) > 1:  # silhouette needs >1 cluster
                score = silhouette_score(X=dist, labels=labels, metric="precomputed")
                print(f"[Threshold {threshold:.2f}] Silhouette Score: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
        except Exception as e:
            print(f"[Threshold {threshold:.2f}] Failed: {e}")

    return best_threshold



def find_dendrogram_jump_threshold(dist: np.ndarray, linkage_method="complete", jump_window=30) -> float:
    """
    Find optimal distance threshold based on the largest jump in the dendrogram merge distances.

    Args:
        dist (np.ndarray): Precomputed square distance matrix (NxN).
        linkage_method (str): Linkage method to use ('complete', 'average', etc.)
        jump_window (int): Number of last merges to inspect for jump

    Returns:
        float: Suggested distance threshold.
    """
    if not (dist.shape[0] == dist.shape[1]):
        raise ValueError("Distance matrix must be square")

    condensed = squareform(dist)
    Z = linkage(condensed, method=linkage_method)
    merge_distances = Z[:, 2]

    # Focus on the last 'jump_window' merges
    if len(merge_distances) < jump_window:
        window = merge_distances
    else:
        window = merge_distances[-jump_window:]

    diffs = np.diff(window)
    max_jump_idx = np.argmax(diffs)
    suggested_threshold = window[max_jump_idx + 1]  # after the jump

    print(f"📈 Dendrogram jump detected: Δ={diffs[max_jump_idx]:.4f} at threshold={suggested_threshold:.4f}")
    return suggested_threshold