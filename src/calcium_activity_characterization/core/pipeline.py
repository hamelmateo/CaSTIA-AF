"""Module defining the CalciumPipeline class for processing calcium imaging data.

Example:
    >>> from src.core.pipeline import CalciumPipeline
    >>> pipeline = CalciumPipeline(config)
    >>> pipeline.run(data_dir, output_dir)
"""

import numpy as np
import logging
from pathlib import Path

from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.data.populations import Population
from calcium_activity_characterization.utilities.export import MetricExporter, NormalizedDataExporter
from calcium_activity_characterization.utilities.loader import (
    save_tif_image,
    load_images,
    save_pickle_file,
    load_pickle_file,
    plot_raster,
    get_config_with_fallback
)
from calcium_activity_characterization.preprocessing.image_processing import ImageProcessor
from calcium_activity_characterization.preprocessing.segmentation import segmented
from calcium_activity_characterization.preprocessing.trace_extraction import TraceExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)



class CalciumPipeline:
    """
    Main processing pipeline for calcium imaging data.
    This class orchestrates the entire workflow from segmentation to event detection.
    It initializes paths, processes images, computes cell intensities, applies signal processing,
    binarizes traces, detects events, and saves population-level metrics.
    It can run in parallel or serial mode based on configuration settings.

    Attributes:
        config (dict): Configuration dictionary containing parameters for processing.
        population (Population): The population of cells processed in the pipeline.
        nuclei_mask (np.ndarray): Mask of the nuclei used for segmentation.
        data_dir (Path): Path to the input data directory containing ISX folders.
        output_dir (Path): Path to the output directory where results will be saved.
        directory_name (Path): Name of the parent directory containing the ISX folder.
        fitc_file_pattern (Path): Regex pattern to match FITC image files.
        hoechst_file_pattern (Path): Regex pattern to match HOECHST image files.
        hoechst_img_path (Path): Path to the HOECHST images directory.
        fitc_img_path (Path): Path to the FITC images directory.
        nuclei_mask_path (Path): Path to the nuclei mask image file.
        overlay_path (Path): Path to the overlay image file.
        raw_cells_path (Path): Path to save raw cell data.
        raw_traces_path (Path): Path to save raw intensity traces.
        smoothed_traces_path (Path): Path to save smoothed intensity traces.
        binary_traces_path (Path): Path to save binarized intensity traces.
        events_path (Path): Path to save detected events.
        population_level_metrics_path (Path): Path to save population-level metrics report.
        spatial_neighbor_graph_path (Path): Path to save the spatial neighbor graph.
        activity_trace_path (Path): Path to save the activity trace plot.

    Methods:
        run(data_dir: Path, output_dir: Path) -> None:
            Execute the full calcium imaging pipeline for one ISX folder.

    """

    def __init__(self, config: dict):
        self.config = config
        
        # data
        self.population: Population = None
        self.nuclei_mask: np.ndarray = None

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
        self.overlay_path: Path = None
        self.raw_cells_path: Path = None
        self.raw_traces_path: Path = None
        self.smoothed_traces_path: Path = None
        self.binary_traces_path: Path = None
        self.events_path: Path = None
        self.population_level_metrics_path: Path = None
        self.spatial_neighbor_graph_path: Path = None

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

        self.export_normalized_datasets()


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

        # Pick paths for saving results
        self.raw_cells_path = output_dir / "00_raw_cells.pkl"
        self.raw_traces_path = output_dir / "01_raw_traces.pkl"
        self.smoothed_traces_path = output_dir / "02_smoothed_traces.pkl"
        self.binary_traces_path = output_dir / "03_binarized_traces.pkl"
        self.events_path = output_dir / "04_population_events.pkl"

        # Paths for saving images and plots
        self.nuclei_mask_path = output_dir / "nuclei_mask.TIF"
        self.overlay_path = output_dir / "overlay.TIF"
        self.spatial_neighbor_graph_path = output_dir / "neighbors_graph.png"
        self.activity_trace_path = output_dir / "activity_trace.pdf"
        self.population_level_metrics_path = output_dir / "population_metrics.pdf"


    def _segment_cells(self) -> None:
        """
        Perform segmentation on DAPI images if needed and convert the mask to Cell objects.
        Reloads from file if available and permitted.
        """
        logger.info(f"Trying segmentation with parameters: {get_config_with_fallback(self.config, 'SEGMENTATION_PARAMETERS')}")
        if not self.raw_cells_path.exists():
            if not self.nuclei_mask_path.exists():
                processor = ImageProcessor(get_config_with_fallback(self.config, "IMAGE_PROCESSING_PARAMETERS"))
                self.nuclei_mask = segmented(
                    processor.process_all(
                        self.hoechst_img_path,
                        self.hoechst_file_pattern
                    ),
                    self.overlay_path,
                    get_config_with_fallback(self.config, "SEGMENTATION_PARAMETERS")
                )
                save_tif_image(self.nuclei_mask, self.nuclei_mask_path)
            else:
                self.nuclei_mask = load_images(self.nuclei_mask_path)

            unfiltered_cells = Cell.from_segmentation_mask(self.nuclei_mask, get_config_with_fallback(self.config,"CELL_FILTERING_PARAMETERS"))

            cells = [cell for cell in unfiltered_cells if cell.is_valid]

            self.population = Population(cells=cells, mask=self.nuclei_mask, output_dir=self.spatial_neighbor_graph_path)
            save_pickle_file(self.population, self.raw_cells_path)
            logger.info(f"Kept {len(cells)} active cells out of {len(unfiltered_cells)} total cells.")
        else:
            self.population = load_pickle_file(self.raw_cells_path)


    def _compute_intensity(self) -> None:
        """
        Compute raw calcium traces for all cells in the population.
        If raw traces already exist, load them from file.
        """
        if not self.raw_traces_path.exists():
            extractor = TraceExtractor(
                cells=self.population.cells,
                images_dir=self.fitc_img_path,
                config=get_config_with_fallback(self.config, "TRACE_EXTRACTION_PARAMETERS"),
                processor=ImageProcessor(get_config_with_fallback(self.config, "IMAGE_PROCESSING_PARAMETERS"))
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
        if self.smoothed_traces_path.exists():
            self.population = load_pickle_file(self.smoothed_traces_path)
            return

        else:
            for cell in self.population.cells:
                cell.trace.process_trace("raw","smoothed",get_config_with_fallback(self.config,"INDIV_SIGNAL_PROCESSING_PARAMETERS"))
                cell.trace.default_version = "smoothed"
            
            save_pickle_file(self.population, self.smoothed_traces_path)


    def _binarization_pipeline(self) -> None: 
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

        plot_raster(self.output_dir, self.population.cells)


    def _initialize_activity_trace(self) -> None:
        """
        Compute the global trace from active cells and process it through smoothing,
        peak detection, binarization, and metadata extraction.
        """
        default_version = "smoothed"
        self.population.compute_activity_trace(default_version=default_version,
                                               signal_processing_params=get_config_with_fallback(self.config,"POPULATION_TRACES_SIGNAL_PROCESSING_PARAMETERS"),
                                               peak_detection_params=get_config_with_fallback(self.config, "ACTIVITY_TRACE_PEAK_DETECTION_PARAMETERS"))

        self.population.activity_trace.plot_all_traces(self.activity_trace_path)


    def _detect_events(self) -> None:
        """
        Detect events from the population traces and save them.
        This method is a placeholder for future event detection logic.
        """
        if self.events_path.exists():
            self.population = load_pickle_file(self.events_path)
            return

        self.population.detect_global_events(get_config_with_fallback(self.config, "EVENT_EXTRACTION_PARAMETERS"))

        self.population.detect_sequential_events(get_config_with_fallback(self.config, "EVENT_EXTRACTION_PARAMETERS"))

        self.population.assign_peak_event_ids()

        save_pickle_file(self.population, self.events_path)


    def export_normalized_datasets(self) -> None:
        """
        Export normalized multi-table datasets: peaks, cells, events, and population-level metrics.

        Args:
            output_dir (Path): Directory where the datasets will be saved.
        """
        try:
            exporter = NormalizedDataExporter(self.population, self.output_dir)
            exporter.export_all()
            logger.info(f"✅ Normalized datasets exported to {self.output_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to export normalized datasets: {e}")