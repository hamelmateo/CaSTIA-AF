import time
import logging
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

from calcium_activity_characterization.processing.pipeline import CalciumPipeline
from config.config import (
    HARDDRIVE_PATH,
    ROI_SCALE,
    PADDING,
    PARALLELELIZE,
    SAVE_OVERLAY,
    EXISTING_CELLS,
    EXISTING_MASK,
    EXISTING_RAW_INTENSITY,
    EXISTING_PROCESSED_INTENSITY,
    SIGNAL_PROCESSING_PARAMETERS,
    DETRENDING_MODE,
    BINDATA_PARAMETERS,
    TRACKING_PARAMETERS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG = {
    "ROI_SCALE": ROI_SCALE,
    "PADDING": PADDING,
    "PARALLELELIZE": PARALLELELIZE,
    "SAVE_OVERLAY": SAVE_OVERLAY,
    "EXISTING_CELLS": EXISTING_CELLS,
    "EXISTING_MASK": EXISTING_MASK,
    "EXISTING_RAW_INTENSITY": EXISTING_RAW_INTENSITY,
    "EXISTING_PROCESSED_INTENSITY": EXISTING_PROCESSED_INTENSITY,
    "SIGNAL_PROCESSING_PARAMETERS": SIGNAL_PROCESSING_PARAMETERS,
    "DETRENDING_MODE": DETRENDING_MODE,
    "BINDATA_PARAMETERS": BINDATA_PARAMETERS,
    "TRACKING_PARAMETERS": TRACKING_PARAMETERS
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
    folder_dialog = QFileDialog()
    folder_dialog.setDirectory(HARDDRIVE_PATH)
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

        pipeline = CalciumPipeline(CONFIG)
        logger.info(f"Found {len(all_isx_folders)} ISX folders.")

        for isx_folder in sorted(all_isx_folders):
            output_folder = isx_folder.parents[1] / "Output" / isx_folder.name
            logger.info(f"Processing {isx_folder}...")
            pipeline.run(isx_folder, output_folder)

    else:
        logger.info("No folder selected. Exiting.")

if __name__ == "__main__":
    main()
