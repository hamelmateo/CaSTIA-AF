import time
import logging
from pathlib import Path
from typing import List

from src.io.loader import (
    load_existing_cells,
    load_existing_img,
    save_pickle_file,
)
from src.config.config import (
    ROI_SCALE,
    FITC_FILE_PATTERN,
    HOECHST_FILE_PATTERN,
    PADDING,
    PARALLELELIZE,
    SAVE_OVERLAY,
    EXISTING_CELLS,
    EXISTING_MASK,
    EXISTING_INTENSITY_PROFILE,
    GAUSSIAN_SIGMA,
    HPF_CUTOFF,
    SAMPLING_FREQ,
    ORDER
)
from src.core.pipeline import (
    cells_segmentation,
    convert_mask_to_cells,
    get_cells_intensity_profiles,
    get_cells_intensity_profiles_parallelized,
)
from src.analysis.umap_analysis import run_umap_with_clustering, run_umap_on_cells
from src.analysis.tuning import explore_processing_parameters
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)




def run_pipeline(data_path: Path, output_path: Path) -> None:
    """
    Run the full calcium imaging analysis pipeline on a given ISX folder.

    Args:
        data_path (Path): Path to the ISX data folder (contains HOECHST and FITC).
        output_path (Path): Path to the output folder for this ISX.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    nuclei_mask_path = output_path / "nuclei_mask.TIF"
    overlay_path = output_path / "overlay.TIF"
    temp_overlay_path = output_path / "temp_overlay.TIF"
    cells_file_path = output_path / "cells.pkl"
    active_cells_file_path = output_path / "active_cells.pkl"

    hoechst_img_path = data_path / "HOECHST"
    fitc_img_path = data_path / "FITC"

    start = time.time()
    logger.info(f"Starting pipeline for {data_path.name}...")

    if not EXISTING_CELLS or not cells_file_path.exists():
        if not EXISTING_MASK or not nuclei_mask_path.exists():
            try:
                nuclei_mask = cells_segmentation(
                    hoechst_img_path,
                    ROI_SCALE,
                    HOECHST_FILE_PATTERN,
                    PADDING,
                    overlay_path,
                    SAVE_OVERLAY,
                    nuclei_mask_path,
                )
            except Exception as e:
                logger.error(f"Failed to perform segmentation: {e}")
                return
        else:
            try:
                nuclei_mask = load_existing_img(nuclei_mask_path)
            except Exception as e:
                logger.error(f"Failed to load existing mask: {e}")
                return

        try:
            cells = convert_mask_to_cells(nuclei_mask)
            save_pickle_file(cells, cells_file_path)
        except Exception as e:
            logger.error(f"Error converting/saving cells: {e}")
            return
    else:
        cells = load_existing_cells(cells_file_path, EXISTING_CELLS)

    logger.info(f"Number of cells detected: {len(cells)}")

    active_cells = [cell for cell in cells if cell.is_valid]
    logger.info(f"Active cells: {len(active_cells)} / Total: {len(cells)}")

    if not EXISTING_INTENSITY_PROFILE or not active_cells_file_path.exists():
        try:
            if not PARALLELELIZE:
                get_cells_intensity_profiles(
                    active_cells,
                    fitc_img_path,
                    ROI_SCALE,
                    FITC_FILE_PATTERN,
                    PADDING,
                )
            else:
                logger.debug("Running in parallelized mode for intensity profile computation...")
                get_cells_intensity_profiles_parallelized(
                    active_cells,
                    fitc_img_path,
                    FITC_FILE_PATTERN,
                    PADDING,
                    ROI_SCALE,
                    GAUSSIAN_SIGMA,
                    HPF_CUTOFF,
                    SAMPLING_FREQ,
                    ORDER
                )
            save_pickle_file(active_cells, active_cells_file_path)
        except Exception as e:
            logger.error(f"Failed during intensity profiling: {e}")
            return
    else:
        active_cells = load_existing_cells(active_cells_file_path, EXISTING_CELLS)


    
    # Pick one example cell
    #example_cell = next((cell for cell in active_cells if cell.label == 366), None)

    #if example_cell:
    #    sigmas = [0.1, 0.5, 1.0, 2.0, 5.0]
    #    cutoffs = [0.001, 0.005, 0.01, 0.02, 0.05]
    #    explore_processing_parameters(example_cell, sigmas, cutoffs)


    logger.info("Running UMAP...")
    try:
        run_umap_on_cells(
            active_cells,
            n_neighbors=5,
            min_dist=1,
            n_components=2,
            normalize=True,
            eps=0.5,
            min_samples=5,
        )
    except Exception as e:
        logger.error(f"UMAP analysis failed: {e}")
        return

    logger.info(f"Pipeline for {data_path.name} completed successfully in {time.time() - start:.2f} seconds")

def main() -> None:
    """
    Select root experiment folder and process all ISX folders inside Data/
    """
    app = QApplication(sys.argv)
    root_folder = QFileDialog.getExistingDirectory(None, "Select Root Experiment Folder (e.g., 20250326)")

    if not root_folder:
        logger.info("No folder selected. Exiting.")
        return

    root_path = Path(root_folder)
    data_dir = root_path / "Data"
    output_dir = root_path / "Output"

    if not data_dir.exists():
        logger.error(f"Missing 'Data' directory in {root_path}")
        return

    for isx_folder in sorted(data_dir.glob("IS*")):
        if isx_folder.is_dir():
            logger.info(f"Processing {isx_folder.name}...")
            run_pipeline(isx_folder, output_dir / isx_folder.name)

if __name__ == "__main__":
    main()