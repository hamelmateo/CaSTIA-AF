import time
import logging
from pathlib import Path

from src.io.loader import (
    load_cells_from_pickle,
    load_images,
    save_pickle_file,
)
from src.config.config import (
    ROI_SCALE,
    PADDING,
    PARALLELELIZE,
    SAVE_OVERLAY,
    EXISTING_CELLS,
    EXISTING_MASK,
    EXISTING_INTENSITY_PROFILE,
    GAUSSIAN_SIGMA,
    HPF_CUTOFF,
    SAMPLING_FREQ,
    ORDER,
    BTYPE
)
from src.core.pipeline import (
    cells_segmentation,
    convert_mask_to_cells,
    get_cells_intensity_profiles,
    get_cells_intensity_profiles_parallelized,
)
from src.analysis.umap_analysis import run_umap_on_cells
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

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

    directory_name = data_path.name
    fitc_file_pattern = rf"{directory_name}__w3FITC_t(\d+).TIF"
    hoechst_file_pattern = rf"{directory_name}__w2DAPI_t(\d+).TIF"

    nuclei_mask_path = output_path / "nuclei_mask.TIF"
    overlay_path = output_path / "overlay.TIF"
    cells_file_path = output_path / "cells.pkl"
    active_cells_file_path = output_path / "active_cells.pkl"
    umap_file_path = output_path / "umap.npy"

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
                    hoechst_file_pattern,
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
                nuclei_mask = load_images(nuclei_mask_path)
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
        cells = load_cells_from_pickle(cells_file_path, EXISTING_CELLS)

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
                    fitc_file_pattern, 
                    PADDING,
                )
            else:
                logger.debug("Running in parallelized mode for intensity profile computation...")
                get_cells_intensity_profiles_parallelized(
                    active_cells,
                    fitc_img_path,
                    fitc_file_pattern,
                    PADDING,
                    ROI_SCALE,
                    GAUSSIAN_SIGMA,
                    HPF_CUTOFF,
                    SAMPLING_FREQ,
                    ORDER,
                    BTYPE
                )
            save_pickle_file(active_cells, active_cells_file_path)
        except Exception as e:
            logger.error(f"Failed during intensity profiling: {e}")
            return
    else:
        active_cells = load_cells_from_pickle(active_cells_file_path, EXISTING_CELLS)

    logger.info("Running UMAP...")
    try:
        run_umap_on_cells(
            active_cells, 
            umap_file_path,
            n_neighbors=5, 
            min_dist=1,
            n_components=2,
            normalize=False,
        )
    except Exception as e:
        logger.error(f"UMAP finetuning failed: {e}")
        return

    logger.info(f"Pipeline for {data_path.name} completed successfully in {time.time() - start:.2f} seconds")


def find_isx_folders(folder: Path) -> list[Path]:
    """
    Recursively find all ISX folders under 'Data/' directories.
    Skips any folder paths that include 'Output'.
    """
    isx_folders = []
    for subpath in folder.rglob("*"):
        if "Output" in subpath.parts:
            continue  # skip anything in Output
        if subpath.is_dir() and subpath.name.startswith("IS"):
            isx_folders.append(subpath)
    return isx_folders


def main() -> None:
    """
    Select one or multiple folders (date or ISX) and run the pipeline on all detected ISX folders.
    """
    app = QApplication(sys.argv)
    folder_dialog = QFileDialog()
    folder_dialog.setFileMode(QFileDialog.Directory)
    folder_dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
    folder_dialog.setWindowTitle("Select One or More Folders (Date folders or ISX)")

    if folder_dialog.exec_():
        selected = folder_dialog.selectedFiles()
        all_isx_folders = []

        for folder_str in selected:
            folder = Path(folder_str)
            if folder.name.startswith("IS"):
                all_isx_folders.append(folder)
            else:
                isx_inside = find_isx_folders(folder)
                all_isx_folders.extend(isx_inside)

        if not all_isx_folders:
            logger.warning("No ISX folders found in selected path(s). Exiting.")
            return

        logger.info(f"Found {len(all_isx_folders)} ISX folders.")
        for isx_folder in sorted(all_isx_folders):
            output_folder = isx_folder.parents[1] / "Output" / isx_folder.name
            logger.info(f"Processing {isx_folder}...")
            run_pipeline(isx_folder, output_folder)

    else:
        logger.info("No folder selected. Exiting.")


if __name__ == "__main__":
    main()