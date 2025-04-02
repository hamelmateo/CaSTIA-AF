from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QMainWindow
from PyQt5.QtGui import QPixmap, QImage, QPen
from PyQt5.QtCore import Qt, QRectF
from src.io.loader import load_existing_cells
from src.config.config import CELLS_FILE_PATH, OVERLAY_PATH
from src.core.cell import Cell
from src.gui.plotter import show_cell_plot
import tifffile
import numpy as np


class OverlayViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cell Viewer")
        self.setGeometry(100, 100, 1000, 1000)

        self.cells = load_existing_cells(CELLS_FILE_PATH, True)
        self.overlay_img = tifffile.imread(str(OVERLAY_PATH))

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.setCentralWidget(self.view)

        self.pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(self.to_qimage(self.overlay_img)))
        self.scene.addItem(self.pixmap_item)

        self.highlight_pen = QPen(Qt.red)
        self.highlight_pen.setWidth(2)

        self.current_highlight = None
        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)

    def to_qimage(self, img):
        norm_img = ((img / 65535) * 255).astype(np.uint8)
        h, w = norm_img.shape
        return QImage(norm_img.data, w, h, w, QImage.Format_Grayscale8)

    def eventFilter(self, source, event):
        if event.type() == event.MouseMove:
            pos = event.pos()
            img_pos = self.view.mapToScene(pos)
            x, y = int(img_pos.x()), int(img_pos.y())

            hovered_cell = self.find_closest_cell(y, x, radius=8)

            if hovered_cell:
                self.highlight_cell(hovered_cell)
            else:
                self.clear_highlight()

        elif event.type() == event.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                pos = event.pos()
                img_pos = self.view.mapToScene(pos)
                x, y = int(img_pos.x()), int(img_pos.y())
                selected_cell = self.find_closest_cell(y, x, radius=8)
                if selected_cell:
                    show_cell_plot(selected_cell)

        return super().eventFilter(source, event)

    def find_closest_cell(self, y, x, radius=8):
        for cell in self.cells:
            cy, cx = cell.centroid
            if abs(cx - x) <= radius and abs(cy - y) <= radius:
                return cell
        return None

    def highlight_cell(self, cell):
        if self.current_highlight:
            self.scene.removeItem(self.current_highlight)

        min_yx = cell.pixel_coords.min(axis=0)
        max_yx = cell.pixel_coords.max(axis=0)
        top_left = min_yx[::-1]
        bottom_right = max_yx[::-1]
        rect = QRectF(top_left[0], top_left[1], bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
        self.current_highlight = self.scene.addRect(rect, self.highlight_pen)

    def clear_highlight(self):
        if self.current_highlight:
            self.scene.removeItem(self.current_highlight)
            self.current_highlight = None