# python -m src.gui
from PyQt5.QtWidgets import QApplication
import sys
from src.gui.viewer import OverlayViewer


def main():
    app = QApplication(sys.argv)
    viewer = OverlayViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()