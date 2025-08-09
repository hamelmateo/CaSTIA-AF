"""Module defining the CalciumPipeline class for processing calcium imaging data.

Example:
    >>> from src.core.pipeline import CalciumPipeline
    >>> pipeline = CalciumPipeline(config)
    >>> pipeline.run(data_dir, output_dir)
"""

import random
import numpy as np
from pathlib import Path
from collections import Counter
from skimage.segmentation import find_boundaries

from calcium_activity_characterization.logger import logger
from calcium_activity_characterization.config.presets import GlobalConfig, SegmentationConfig, ImageProcessingConfig
from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.data.populations import Population
from calcium_activity_characterization.preprocessing.image_processing import ImageProcessor
from calcium_activity_characterization.preprocessing.segmentation import segmented
from calcium_activity_characterization.preprocessing.trace_extraction import TraceExtractor
from calcium_activity_characterization.io.export import NormalizedDataExporter
from calcium_activity_characterization.io.images_loader import (
    load_existing_img,
    load_pickle_file
)
from calcium_activity_characterization.io.export import (
    save_tif_image,
    save_rgb_tif_image,
    save_pickle_file
)
from calcium_activity_characterization.utilities.plotter import (
    plot_spatial_neighbor_graph,
    plot_event_growth_curve,
    plot_cell_connection_network,
    plot_raster_heatmap,
    plot_raster,
    plot_metric_on_overlay
)
from calcium_activity_characterization.utilities.image_utils import render_cell_outline_overlay


class CalciumPipeline:
    """
    Class to execute the full calcium imaging analysis pipeline.

    Attributes:
        config (GlobalConfig): Configuration parameters for the pipeline.
        population (Population): The cell population being analyzed.
        data_dir (Path): Path to the input data directory.
        output_dir (Path): Path to the output results directory.
        directory_name (str): Name of the data directory.
        fitc_file_pattern (str): Regex pattern for FITC image filenames.
        hoechst_file_pattern (str): Regex pattern for Hoechst image filenames.
        hoechst_img_path (Path): Path to the Hoechst images directory.
        fitc_img_path (Path): Path to the FITC images directory.
        nuclei_mask_path (Path): Path to save/load the nuclei mask image.
        full_nuclei_mask_path (Path): Path to save/load the full nuclei mask image.
        overlay_path (Path): Path to save the cell outline overlay image.
        raw_cells_path (Path): Path to save/load the raw cells pickle file.
        raw_traces_path (Path): Path to save/load the raw traces pickle file.
        processed_traces_path (Path): Path to save/load the processed traces pickle file.
        binary_traces_path (Path): Path to save/load the binary traces pickle file.
        events_path (Path): Path to save/load the detected events pickle file.
        spatial_neighbor_graph_path (Path): Path to save/load the spatial neighbor graph pickle file.
        activity_trace_path (Path): Path to save/load the activity trace pickle file.
    """

    def __init__(self, config: GlobalConfig):
        self.config = config
        
        # data
        self.population: Population = None

        # folder paths
        self.data_dir: Path = None
        self.output_dir: Path = None
        self.directory_name: Path = None

        # file paths
        self.fitc_file_pattern: Path = None
        self.hoechst_file_pattern: Path = None
        self.hoechst_img_path: Path = None
        self.fitc_img_path: Path = None
        self.nuclei_mask_path: Path = None
        self.full_nuclei_mask_path: Path = None
        self.overlay_path: Path = None
        self.raw_cells_path: Path = None
        self.raw_traces_path: Path = None
        self.processed_traces_path: Path = None
        self.binary_traces_path: Path = None
        self.events_path: Path = None
        self.spatial_neighbor_graph_path: Path = None
        self.activity_trace_path: Path = None

    def run(self, data_dir: Path, output_dir: Path) -> None:
        """
        Execute the full calcium imaging pipeline for one ISX folder.

        Args:
            data_dir (Path): Path to the ISX folder (containing FITC/HOECHST subfolders).
            output_dir (Path): Destination folder for processed results.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        self._init_paths(data_dir, output_dir)

        self._segment_cells()
        self._compute_intensity()

        self._signal_processing_pipeline()
        self._binarization_pipeline()
        
        self._initialize_activity_trace()
        self._detect_events()

        self._analyze_population()

        self._export_normalized_datasets()


    def _init_paths(self, data_dir: Path, output_dir: Path) -> None:
        """
        Initialize and store all input/output paths and file patterns.

        Args:
            data_dir (Path): Root path to ISX experiment.
            output_dir (Path): Path to save all outputs for this experiment.
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.directory_name = data_dir.parents[1].name

        # Data paths
        self.fitc_file_pattern = rf"{self.directory_name}__w3FITC_t(\d+)\.TIF"
        self.hoechst_file_pattern = rf"{self.directory_name}__w2DAPI_t(\d+)\.TIF"

        self.hoechst_img_path = data_dir / "HOECHST"
        self.fitc_img_path = data_dir / "FITC"

        # Pick paths for saving snapshot data
        snapshot_dir = output_dir / "population-snapshots"
        self.raw_cells_path = snapshot_dir / "00_raw_cells.pkl"
        self.raw_traces_path = snapshot_dir / "01_raw_traces.pkl"
        self.processed_traces_path = snapshot_dir / "02_smoothed_traces.pkl"
        self.binary_traces_path = snapshot_dir / "03_binarized_traces.pkl"
        self.events_path = snapshot_dir / "04_population_events.pkl"

        # Paths for spatial mapping
        spatial_mapping_dir = output_dir / "cell-mapping"
        self.saved_processing_config_path = spatial_mapping_dir / "hoechst_image_processing_config.json"
        self.saved_seg_config_path = spatial_mapping_dir / "segmentation_config.json"
        self.nuclei_mask_path = spatial_mapping_dir / "nuclei_mask.TIF"
        self.full_nuclei_mask_path = spatial_mapping_dir / "full_nuclei_mask.TIF"
        self.overlay_path = spatial_mapping_dir / "overlay.TIF"
        self.spatial_neighbor_graph_path = spatial_mapping_dir / "neighbors_graph.png"

        # Paths for intermediate results
        processing_dir = output_dir / "signal-processing"
        self.traces_processing_steps = processing_dir / "traces-processing-steps"
        self.activity_trace_path = processing_dir / "activity_trace.pdf"
        self.heatmap_raster_path = processing_dir / "heatmap_raster.png"
        self.raster_path = processing_dir / "raster_plot.png"

        # Path for extracted data
        self.datasets_dir = output_dir / "datasets"


    def _segment_cells(self) -> None:
        """
        #TODO: Complete docstring
        """
        if not self.raw_cells_path.exists():
            nuclei_mask = None

            if self.saved_processing_config_path.exists():
                logger.info(f"Loading image processing config from {self.saved_processing_config_path}")
                proc_config = ImageProcessingConfig.from_json(self.saved_processing_config_path)
            else:
                proc_config = self.config.image_processing_hoechst
                logger.info(f"No saved image processing config found. Using default config.")

            processor = ImageProcessor(config=proc_config)
            if not self.nuclei_mask_path.exists():
                if self.saved_seg_config_path.exists():
                    logger.info(f"Loading segmentation config from {self.saved_seg_config_path}")
                    seg_config = SegmentationConfig.from_json(self.saved_seg_config_path)
                else:
                    seg_config = self.config.segmentation
                    logger.info(f"No saved segmentation config found. Using default config.")
                full_nuclei_mask = segmented(
                    processor.process_all(
                        self.hoechst_img_path,
                        self.hoechst_file_pattern
                    ),
                    seg_config
                )
                save_tif_image(full_nuclei_mask, self.full_nuclei_mask_path)
            else:
                full_nuclei_mask = load_existing_img(self.full_nuclei_mask_path)

            unfiltered_cells = Cell.from_segmentation_mask(full_nuclei_mask, self.config.cell_filtering)
            cells = [cell for cell in unfiltered_cells if cell.is_valid]

            graph = Population.build_spatial_neighbor_graph(cells)
            
            nuclei_mask = processor._crop_image(full_nuclei_mask)
            save_tif_image(nuclei_mask, self.nuclei_mask_path)

            self.population = Population.from_roi_filtered(
                nuclei_mask=nuclei_mask,
                cells=cells,
                graph=graph,
                roi_scale=self.config.image_processing_hoechst.roi_scale,
                img_shape=full_nuclei_mask.shape,
                border_margin=self.config.cell_filtering.border_margin
            )

            processor.config.pipeline.cropping = True
            cropped_hoechst = processor.process_all(self.hoechst_img_path, self.hoechst_file_pattern)[0]

            self.save_cell_outline_overlay(self.overlay_path, self.population.cells, cropped_hoechst)
            plot_spatial_neighbor_graph(self.population.neighbor_graph, cropped_hoechst, self.spatial_neighbor_graph_path)

            save_pickle_file(self.population, self.raw_cells_path)

            logger.info(f"Kept {len(cells)} active cells out of {len(unfiltered_cells)} total cells.")
        else:
            self.population = load_pickle_file(self.raw_cells_path)


    def save_cell_outline_overlay(self, output_path: Path, cells: list[Cell], hoechst_img: np.ndarray) -> None:
        """
        Save an overlay image of cell outlines on the Hoechst image.

        Args:
            output_path (Path): Path to save the overlay image. If None, does not save.
        """
        try:
            outline = self.compute_outline_mask(cells, hoechst_img.shape)
            overlay_img = render_cell_outline_overlay(hoechst_img, outline)
            save_rgb_tif_image(overlay_img, output_path)

        except Exception as e:
            logger.error(f"Failed to save cell outline overlay: {e}")


    def compute_outline_mask(self, cells: list[Cell], hoechst_img_shape: tuple) -> np.ndarray:
        """
        Compute a binary mask where True represents the outline of all valid cells in the population.

        Returns:
            np.ndarray: 2D boolean array with shape (H, W) where True marks cell contours.

        Raises:
            ValueError: If population has no cells or overlay_image is not defined.
        """
        try:
            mask_shape = hoechst_img_shape
            if len(mask_shape) != 2:
                raise ValueError(f"Expected 2D grayscale image, got shape {mask_shape}")

            outline = np.zeros(mask_shape, dtype=bool)

            for cell in cells:
                cell_mask = np.zeros(mask_shape, dtype=bool)
                for y, x in cell.pixel_coords:
                    if 0 <= y < mask_shape[0] and 0 <= x < mask_shape[1]:
                        cell_mask[y, x] = True
                outline |= find_boundaries(cell_mask, mode="inner")

            return outline

        except Exception as e:
            logger.error(f"❌ Failed to compute outline mask: {e}")
            raise


    def _compute_intensity(self) -> None:
        """
        Compute raw calcium traces for all cells in the population.
        If raw traces already exist, load them from file.
        """
        if not self.raw_traces_path.exists():
            extractor = TraceExtractor(
                cells=self.population.cells,
                images_dir=self.fitc_img_path,
                config=self.config.trace_extraction,
                processor=ImageProcessor(self.config.image_processing_fitc)
            )
            extractor.compute(self.fitc_file_pattern)
            save_pickle_file(self.population, self.raw_traces_path)
        else:
            self.population = load_pickle_file(self.raw_traces_path)


    def _signal_processing_pipeline(self) -> None:
        """
        Run signal processing pipeline on all active cells.
        Reloads from file if permitted.
        """
        #if self.processed_traces_path.exists():
        if False:  # Disable reloading for debugging purposes
            self.population = load_pickle_file(self.processed_traces_path)
            return

        else:
            self.output_dir.mkdir(exist_ok=True, parents=True)
            for cell in self.population.cells:
                cell.trace.process_and_plot_trace(
                    input_version="raw",
                    output_version="processed",
                    processing_params=self.config.cell_trace_processing
                )
            save_pickle_file(self.population, self.processed_traces_path)

            plot_raster_heatmap(self.heatmap_raster_path, 
                                [cell.trace.versions["processed"] for cell in self.population.cells], 
                                cut_trace=self.config.cell_trace_processing.detrending.params.cut_trace_num_points
                                )

            # Select 25 random cells (or all if fewer than 25)
            sample_cells = random.sample(self.population.cells, min(25, len(self.population.cells)))
            for cell in sample_cells:
                cell.trace.plot_all_traces(self.traces_processing_steps / f"{cell.label}_all_traces.png")
            

    def _binarization_pipeline(self) -> None: 
        """
        Run peak detection on all active cells using parameters from config and binarize the traces.
        """
        #if self.binary_traces_path.exists():
        if False:  # Disable reloading for debugging purposes
            self.population = load_pickle_file(self.binary_traces_path)
            return
        
        else:
            for cell in self.population.cells:
                cell.trace.detect_peaks(self.config.cell_trace_peak_detection)
                cell.define_activity()
                cell.trace.binarize_trace_from_peaks()

            
            logger.info(f"Peaks detected for {len(self.population.cells)} active cells.")
            save_pickle_file(self.population, self.binary_traces_path)

        plot_raster(self.raster_path, 
                    [cell.trace.binary for cell in self.population.cells], 
                    self.config.cell_trace_processing.detrending.params.cut_trace_num_points
                    )


    def _initialize_activity_trace(self) -> None:
        """
        Compute the global trace from active cells and process it through smoothing,
        peak detection, binarization, and metadata extraction.
        """
        default_version = "processed"
        self.population.compute_activity_trace(default_version=default_version,
                                               signal_processing_params=self.config.activity_trace_processing,
                                               peak_detection_params=self.config.activity_trace_peak_detection)

        self.population.activity_trace.plot_all_traces(self.activity_trace_path)


    def _detect_events(self) -> None:
        """
        Detect events from the population traces and save them.
        This method is a placeholder for future event detection logic.
        """
        #if self.events_path.exists():
        if False:  # Disable reloading for debugging purposes
            self.population = load_pickle_file(self.events_path)
            return

        self.population.detect_global_events(self.config.event_extraction)

        self.population.detect_sequential_events(self.config.event_extraction)

        self.population.assign_peak_event_ids()

        save_pickle_file(self.population, self.events_path)


    def _analyze_population(self) -> None:
        """
        Analyze the population-level metrics and save them.
        This method is a placeholder for future analysis logic.
        """
        # Plot and save growth curves for each global event
        for event in self.population.events:
            if event.__class__.__name__ == "GlobalEvent":
                plot_event_growth_curve(
                    values=event.growth_curve_distribution.values,
                    start=event.event_start_time,
                    time_to_50=event.time_to_50,
                    title=f"Event {event.id} cumulative growth curve",
                    save_path=self.output_dir / "events" / f"event-growth-curve-{event.id}.png"
                )

        cell_pixel_coords = {cell.label: cell.pixel_coords for cell in self.population.cells}

        # Analyze cell occurences in events
        global_counts, seq_counts, individual_counts, origin_counts = self.population.count_cell_occurences_in_events()
        plot_metric_on_overlay(self.population.nuclei_mask,
                               cell_pixel_coords,
                               global_counts,
                               self.output_dir / "cell-mapping" / "cell_occurences_in_global_events_overlay.png",
                               title="Mapping of Cell Occurences in Global Events",
                               colorbar_label="Occurences of Global Events"
                               )
        plot_metric_on_overlay(self.population.nuclei_mask, 
                               cell_pixel_coords, 
                               seq_counts, 
                               self.output_dir / "cell-mapping" / "cell_occurences_in_sequential_events_overlay.png",
                               title="Mapping of Cell Occurences in Sequential Events",
                               colorbar_label="Occurences of Sequential Events"
                               )
        plot_metric_on_overlay(self.population.nuclei_mask, 
                               cell_pixel_coords, 
                               individual_counts, 
                               self.output_dir / "cell-mapping" / "cell_occurences_in_individual_events_overlay.png",
                               title="Mapping of Cell Occurences in Individual Events",
                               colorbar_label="Occurences of Individual Events"
                               )
        plot_metric_on_overlay(self.population.nuclei_mask, 
                               cell_pixel_coords, 
                               origin_counts, 
                               self.output_dir / "cell-mapping" / "cell_occurences_in_origin_seq_events_overlay.png",
                               title="Mapping of Cell Occurences as Origin in Sequential Events",
                               colorbar_label="Occurences as Origin in Sequential Events"
                               )

        large_seq_counts = self.population.filter_large_sequential_events(min_cells=3)
        plot_metric_on_overlay(self.population.nuclei_mask,
                               cell_pixel_coords,
                               large_seq_counts,
                               self.output_dir / "cell-mapping" / "cell_occurences_in_large_sequential_events_overlay.png",
                               title="Mapping of Cell Occurences in Large Sequential Events",
                               colorbar_label="Occurences in Large Sequential Events"
                               )

        # Analyze cell cell interaction
        cell_connection_network = self.population.compute_cell_connection_network_graph()
        plot_cell_connection_network(cell_connection_network, self.population.nuclei_mask, self.output_dir / "cell-mapping" / "cell_connection_network.png")

        # Global events early peakers mapping
        percent = 0.10
        overlap_early_peakers = Counter()
        for event in self.population.events:
            if event.__class__.__name__ == "GlobalEvent":
                early_peakers = self.population.get_early_peakers_in_global_event(percent, event.id)
                plot_metric_on_overlay(self.population.nuclei_mask,
                                    cell_pixel_coords,
                                    early_peakers,
                                    self.output_dir / "cell-mapping" / "global_events" / f"global_event_{event.id}_early_peakers_overlay.png",
                                    title=f"Mapping of the {percent * 100:.1f}% Early Peakers in Global Events {event.id}",
                                    colorbar_label="Early Peakers in Global Events",
                                    show_colorbar=False
                                    )
                overlap_early_peakers.update(early_peakers)
        
        plot_metric_on_overlay(self.population.nuclei_mask,
                               cell_pixel_coords,
                               overlap_early_peakers,
                               self.output_dir / "cell-mapping" / "global_events" / f"global_event_overlap_early_peakers_overlay.png",
                               title=f"Mapping of Overlapping Early Peakers in Global Events",
                               colorbar_label="Overlapping Early Peakers in Global Events",
                               vmin=0,
                               vmax=len([event for event in self.population.events if event.__class__.__name__ == "GlobalEvent"])
                               )

        # Global events pre-event peakers mapping
        percent = 0.10
        overlap_pre_event_peakers = Counter()
        for event in self.population.events:
            if event.__class__.__name__ == "GlobalEvent":
                pre_event_peakers = self.population.get_pre_event_peakers_of_global_event(percent, event.id)
                plot_metric_on_overlay(self.population.nuclei_mask,
                                    cell_pixel_coords,
                                    pre_event_peakers,
                                    self.output_dir / "cell-mapping" / "global_events" / f"global_event_{event.id}_pre_event_peakers_overlay.png",
                                    title=f"Mapping of Pre-Event Peakers in Global Events {event.id}",
                                    colorbar_label="Pre-Event Peakers in Global Events",
                                    show_colorbar=False
                                    )
                overlap_pre_event_peakers.update(pre_event_peakers)
        
        plot_metric_on_overlay(self.population.nuclei_mask,
                                cell_pixel_coords,
                                overlap_pre_event_peakers,
                                self.output_dir / "cell-mapping" / "global_events" / f"global_event_overlap_pre_event_peakers_overlay.png",
                                title=f"Mapping of Overlapping Pre-Event Peakers in Global Events",
                                colorbar_label="Overlapping Pre-Event Peakers in Global Events",
                                vmin=0,
                                vmax=len([event for event in self.population.events if event.__class__.__name__ == "GlobalEvent"])
                                )

        # Analyze multi-modal distribution in cell-cell communications speed
        speed_threshold = 45 # px/frame ~ 15 um/s
        high_speed_cells_all_comms, high_speed_cells = self.population.map_high_cell_communication_speed(speed_threshold)
        plot_metric_on_overlay(self.population.nuclei_mask,
                               cell_pixel_coords,
                               high_speed_cells_all_comms,
                               self.output_dir / "cell-mapping" / "high_speed_cells_overlay.png",
                               title=f"Mapping of Cells with Communication Speed > {speed_threshold} px/frame",
                               colorbar_label="High Speed Cells"
                               )
        plot_metric_on_overlay(self.population.nuclei_mask,
                               cell_pixel_coords,
                               high_speed_cells,
                               self.output_dir / "cell-mapping" / "high_speed_cells_in_large_events_overlay.png",
                               title=f"Mapping of Cells with Communication Speed > {speed_threshold} px/frame in large events",
                               colorbar_label="High Speed Cells"
                               )

    def _export_normalized_datasets(self) -> None:
        """
        Export normalized multi-table datasets: peaks, cells, events, and population-level metrics.

        Args:
            output_dir (Path): Directory where the datasets will be saved.
        """
        try:
            exporter = NormalizedDataExporter(self.population, self.datasets_dir, self.config.export, self.config.cell_trace_processing.detrending.params.cut_trace_num_points)
            exporter.export_all()
            logger.info(f"✅ Normalized datasets exported to {self.datasets_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to export normalized datasets: {e}")