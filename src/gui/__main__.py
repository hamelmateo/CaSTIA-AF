# python -m src.gui
from PyQt5.QtWidgets import QApplication, QFileDialog
from pathlib import Path
import sys
import logging

from src.config.config import HARDDRIVE_PATH
from src.gui.umap_viewer import UMAPViewer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main() -> None:
    """
    Launches the PyQt GUI for viewing calcium imaging overlays.
    """
    app = QApplication(sys.argv)

    try:
        folder = QFileDialog.getExistingDirectory(None, "Select Analysis Folder", HARDDRIVE_PATH)
        if not folder:
            logger.info("No folder selected. Exiting.")
            return

        viewer = UMAPViewer(Path(folder))
        viewer.show()
        viewer.raise_()
        sys.exit(app.exec_())

    except Exception as e:
        logger.error(f"Failed to launch viewer: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
