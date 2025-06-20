import logging
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

from calcium_activity_characterization.core.pipeline import CalciumPipeline
from calcium_activity_characterization.config.config import (
    DEBUGGING,
    DEBUGGING_FILE_PATH,
    HARDDRIVE_PATH,
    HOECHST_IMAGE_PROCESSING_PARAMETERS,
    FITC_IMAGE_PROCESSING_PARAMETERS,
    SEGMENTATION_PARAMETERS,
    TRACE_EXTRACTION_PARAMETERS,
    CELL_FILTERING_PARAMETERS,
    INDIV_SIGNAL_PROCESSING_PARAMETERS,
    INDIV_PEAK_DETECTION_PARAMETERS,
    POPULATION_TRACES_SIGNAL_PROCESSING_PARAMETERS,
    ACTIVITY_TRACE_PEAK_DETECTION_PARAMETERS,
    EVENT_EXTRACTION_PARAMETERS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG = {
    "HOECHST_IMAGE_PROCESSING_PARAMETERS": HOECHST_IMAGE_PROCESSING_PARAMETERS,
    "FITC_IMAGE_PROCESSING_PARAMETERS": FITC_IMAGE_PROCESSING_PARAMETERS,
    "SEGMENTATION_PARAMETERS": SEGMENTATION_PARAMETERS,
    "TRACE_EXTRACTION_PARAMETERS": TRACE_EXTRACTION_PARAMETERS,
    "CELL_FILTERING_PARAMETERS": CELL_FILTERING_PARAMETERS,
    "INDIV_SIGNAL_PROCESSING_PARAMETERS": INDIV_SIGNAL_PROCESSING_PARAMETERS,
    "INDIV_PEAK_DETECTION_PARAMETERS": INDIV_PEAK_DETECTION_PARAMETERS,
    "POPULATION_TRACES_SIGNAL_PROCESSING_PARAMETERS": POPULATION_TRACES_SIGNAL_PROCESSING_PARAMETERS,
    "ACTIVITY_TRACE_PEAK_DETECTION_PARAMETERS": ACTIVITY_TRACE_PEAK_DETECTION_PARAMETERS,
    "EVENT_EXTRACTION_PARAMETERS": EVENT_EXTRACTION_PARAMETERS,
}

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
    if DEBUGGING:
        logger.info("[DEBUGGING MODE] Using test folder from config.")
        selected = [Path(DEBUGGING_FILE_PATH)]
    else:
        folder_dialog = QFileDialog()
        folder_dialog.setDirectory(HARDDRIVE_PATH)
        folder_dialog.setFileMode(QFileDialog.Directory)
        folder_dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        folder_dialog.setWindowTitle("Select One or More Folders (Date folders or ISX)")

        if not folder_dialog.exec_():
            logger.info("No folder selected. Exiting.")
            return
        
        selected = [Path(folder_str) for folder_str in folder_dialog.selectedFiles()]

    all_isx_folders = []
    for folder in selected:
        if folder.name.startswith("IS"):
            all_isx_folders.append(folder)
        else:
            all_isx_folders.extend(find_isx_folders(folder))

    if not all_isx_folders:
        logger.warning("No ISX folders found in selected path(s). Exiting.")
        return

    pipeline = CalciumPipeline(CONFIG)
    logger.info(f"Found {len(all_isx_folders)} ISX folders.")

    for isx_folder in sorted(all_isx_folders):
        output_folder = isx_folder.parents[1] / "Output" / isx_folder.name
        logger.info(f"Processing {isx_folder}...")
        pipeline.run(isx_folder, output_folder)

if __name__ == "__main__":
    main()
