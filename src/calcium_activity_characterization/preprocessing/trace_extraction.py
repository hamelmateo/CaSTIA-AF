"""
Module containing the TraceExtractor class for computing raw calcium time series
(traces) for segmented cells from FITC image sequences.

Example:
    >>> extractor = TraceExtractor(cells, fitc_dir, config)
    >>> extractor.compute(file_pattern=".*FITC.*\\.TIF")
"""

import tifffile
import numpy as np
import os
from pathlib import Path
from typing import List
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.preprocessing.image_processing import ImageProcessor
from calcium_activity_characterization.utilities.loader import (
    get_config_with_fallback
)

import logging
logger = logging.getLogger(__name__)


class TraceExtractor:
    """
    Extracts raw calcium traces for a list of cells from FITC image sequences.

    Args:
        cells (List[Cell]): List of segmented cells with pixel coordinates.
        images_dir (Path): Path to the directory containing FITC images.
        config (dict): Dictionary of trace extraction configuration parameters.
        processor (ImageProcessor, optional): Image processing utility for preprocessing steps.
    """

    def __init__(self, cells: List[Cell], images_dir: Path, config: dict, processor: any = None) -> None:
        self.cells = cells
        self.images_dir = images_dir
        self.processor = processor
        self.parallelize = get_config_with_fallback(config, "parallelize", False)
        self.trace_version: str = get_config_with_fallback(config, "trace_version_name", "raw")
        self.file_pattern: str = None

    def compute(self, file_pattern: str) -> None:
        """
        Compute traces for all cells across FITC frames.

        Args:
            file_pattern (str): Regex pattern for FITC image filenames.
        """
        self.file_pattern = file_pattern

        if self.parallelize:
            self._compute_traces_parallel()
        else:
            self._compute_traces_serial()

    def _compute_traces_serial(self) -> None:
        """Compute cell-wise intensity traces in serial."""
        images = self.processor.process_all(self.images_dir, self.file_pattern)
        if images.size == 0:
            logger.warning("No FITC images found for trace extraction.")
            return

        for img in images:
            for cell in self.cells:
                cell.add_mean_intensity(img)

    def _compute_traces_parallel(self) -> None:
        """Compute cell-wise intensity traces in parallel using multiprocessing."""
        logger.info("🔄 Computing cell traces in parallel...")

        if get_config_with_fallback(self.processor.apply, "padding", True):
            self.processor.rename_with_padding(self.images_dir, self.file_pattern)

        image_paths = sorted(self.images_dir.glob("*.TIF"))
        cell_coords = [cell.pixel_coords for cell in self.cells]

        func = partial(self._process_single_image, cell_coords=cell_coords, processor=self.processor)

        try:
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                results = list(tqdm(executor.map(func, image_paths), total=len(image_paths)))
        except Exception as e:
            logger.error(f"Parallel trace extraction failed: {e}")
            raise

        # Transpose to cell-major layout
        results_per_cell = list(zip(*results))
        for cell, trace in zip(self.cells, results_per_cell):
            cell.trace.add_trace(trace=list(map(float, trace)), version_name=self.trace_version)

    @staticmethod
    def _process_single_image(image_path: Path, cell_coords: List[np.ndarray], processor: ImageProcessor) -> List[float]:
        """
        Compute the mean intensity for each cell in a single image.

        Args:
            image_path (Path): Path to a single FITC image.
            cell_coords (List[np.ndarray]): Pixel coordinates of each cell.
            roi_scale (float): Scale for cropping around ROI.

        Returns:
            List[float]: List of mean intensities for each cell.
        """
        try:
            img = processor.process_single_image(image_path)
            return [float(np.mean([img[y, x] for y, x in coords])) for coords in cell_coords]
        except Exception as e:
            logger.error(f"Error processing image {image_path.name}: {e}")
            return [0.0] * len(cell_coords)
