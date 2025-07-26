import logging
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget
import sys

from calcium_activity_characterization.core.pipeline import CalciumPipeline
from calcium_activity_characterization.config.presets import GLOBAL_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
    Select one or more folders (date folders or ISX) and run the pipeline on all detected ISX folders.
    """
    app = QApplication(sys.argv)
    if GLOBAL_CONFIG.debug.debugging:
        logger.info("[DEBUGGING MODE] Using test folder from config.")
        selected = [Path(GLOBAL_CONFIG.debug.debugging_file_path)]
    else:
        folder_dialog = QFileDialog()
        folder_dialog.setDirectory(str(GLOBAL_CONFIG.debug.harddrive_path))
        folder_dialog.setFileMode(QFileDialog.Directory)
        folder_dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        folder_dialog.setWindowTitle("Select One or More Folders (Date folders or ISX)")

        # Allow selecting multiple directories in non-native dialog
        view = folder_dialog.findChild(QWidget, "listView")
        if view:
            view.setSelectionMode(view.ExtendedSelection)
        f_tree_view = folder_dialog.findChild(QWidget, "treeView")
        if f_tree_view:
            f_tree_view.setSelectionMode(f_tree_view.ExtendedSelection)

        if not folder_dialog.exec_():
            logger.info("No folder selected. Exiting.")
            return

        selected = [Path(folder_str) for folder_str in folder_dialog.selectedFiles()]

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

    pipeline = CalciumPipeline(GLOBAL_CONFIG)
    logger.info(f"Found {len(all_isx_folders)} ISX folders.")

    for isx_folder in sorted(all_isx_folders):
        output_folder = isx_folder.parents[1] / "Output" / isx_folder.name
        logger.info(f"Processing {isx_folder}...")
        pipeline.run(isx_folder, output_folder)

if __name__ == "__main__":
    main()
