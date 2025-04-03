# python -m src.gui
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys
from src.gui.viewer import OverlayViewer
from pathlib import Path

def main():
    app = QApplication(sys.argv)

    folder = QFileDialog.getExistingDirectory(None, "Select Analysis Folder")
    if not folder:
        print("[INFO] No folder selected. Exiting.")
        return

    viewer = OverlayViewer(Path(folder))
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()