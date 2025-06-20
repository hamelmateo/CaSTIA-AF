"""
Module containing the TraceExtractor class for computing raw calcium time series
(traces) for segmented cells from FITC image sequences.

Example:
    >>> extractor = TraceExtractor(cells, fitc_dir, config)
    >>> extractor.compute(file_pattern=".*FITC.*\\.TIF")
"""

import numpy as np
import os
import gc
import math
import cupy as cp
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
        Compute traces for all cells across FITC frames using GPU if enabled.

        Args:
            file_pattern (str): Regex pattern for FITC image filenames.
        """
        self.file_pattern = file_pattern

        USE_GPU = False  # Temporary hardcoded flag
        try:
            if USE_GPU:
                self._compute_traces_gpu()
            elif self.parallelize:
                self._compute_traces_parallel()
            else:
                self._compute_traces_serial()
        except Exception as cpu_error:
            logger.error(f"❌ CPU fallback failed: {cpu_error}")
            raise

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

    def _compute_traces_gpu(self) -> None:
        """
        Compute raw calcium traces using streamed GPU operations with CuPy.
        """
        try:
            logger.info("Starting GPU-based trace extraction...")

            if get_config_with_fallback(self.processor.apply, "padding", True):
                self.processor.rename_with_padding(self.images_dir, self.file_pattern)

            image_paths = sorted(self.images_dir.glob("*.TIF"))
            n_frames = len(image_paths)
            logger.info(f"Found {n_frames} image frames in {self.images_dir}")

            frame_cpu = self.processor.process_single_image(image_paths[0])
            frame_shape = frame_cpu.shape  # (H, W)
            frame_gpu = cp.asarray(frame_cpu)

            mask_stack = cp.zeros((len(self.cells), *frame_shape), dtype=cp.bool_)
            for i, cell in enumerate(self.cells):
                coords = cell.pixel_coords
                ys, xs = coords[:, 0], coords[:, 1]
                mask = (ys < frame_shape[0]) & (xs < frame_shape[1])
                mask_stack[i, ys[mask], xs[mask]] = True

            pixel_counts = cp.sum(mask_stack, axis=(1, 2))
            # Parameters
            batch_size = 100  # hardcoded for now
            n_batches = math.ceil(n_frames / batch_size)
            trace_parts = []

            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, n_frames)
                batch_paths = image_paths[batch_start:batch_end]
                logger.info(f"\n🚀 Batch {batch_idx + 1}/{n_batches}: Processing frames {batch_start} to {batch_end - 1}")

                traces_gpu_batch = cp.zeros((len(self.cells), len(batch_paths)))

                for t, image_path in enumerate(tqdm(batch_paths, desc=f"🔄 [Batch {batch_idx + 1}/{n_batches}] Processing")):
                    try:
                        frame_cpu = self.processor.process_single_image(image_path)
                        frame_gpu = cp.asarray(frame_cpu)
                        traces_gpu_batch[:, t] = self._compute_trace_batch_gpu(frame_gpu, mask_stack, pixel_counts)
                        del frame_gpu
                    except Exception as frame_error:
                        logger.warning(f"⚠️ Failed to process frame {image_path.name}: {frame_error}")

                trace_parts.append(traces_gpu_batch.get())  # Transfer to CPU
                del traces_gpu_batch

                used_bytes = cp.get_default_memory_pool().used_bytes()
                total_bytes = cp.get_default_memory_pool().total_bytes()
                logger.info(f"🧠 GPU Memory Pool: Used = {used_bytes / 1e6:.2f} MB / Total = {total_bytes / 1e6:.2f} MB")

                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()

            trace_matrix = np.concatenate(trace_parts, axis=1)
            logger.info(f"✅ Successfully assembled trace matrix: {trace_matrix.shape}")

            for i, cell in enumerate(self.cells):
                cell.trace.add_trace(trace=trace_matrix[i].tolist(), version_name=self.trace_version)

        except cp.cuda.memory.OutOfMemoryError as gpu_error:
            logger.error(f"GPU out of memory: {gpu_error}. Falling back to CPU.")
            self._compute_traces_parallel()

    @staticmethod
    def _free_gpu_memory():
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.cuda.runtime.deviceSynchronize()  # ⏱️ Forces sync & GC


    @staticmethod
    def _compute_trace_batch_gpu(frame_gpu: cp.ndarray, mask_stack: cp.ndarray, pixel_counts: cp.ndarray) -> cp.ndarray:
        """
        Compute average intensity per cell using precomputed GPU masks.

        Args:
            frame_gpu (cp.ndarray): Single image on GPU.
            mask_stack (cp.ndarray): Boolean masks per cell (N, H, W).
            pixel_counts (cp.ndarray): Number of pixels in each mask.

        Returns:
            cp.ndarray: Mean intensity per cell.
        """
        masked = mask_stack * frame_gpu[None, :, :]
        summed = cp.sum(masked, axis=(1, 2))
        return summed / pixel_counts

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
