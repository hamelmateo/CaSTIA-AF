from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QMainWindow, QGraphicsTextItem, QLabel
from PyQt5.QtGui import QPixmap, QImage, QPen, QBrush
from PyQt5.QtCore import Qt, QRectF, QPointF
from src.io.loader import load_existing_cells
from src.gui.plotter import show_cell_plot
import tifffile
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class OverlayViewer(QMainWindow):
    """
    PyQt-based viewer for displaying segmented cell overlays and interacting with cells.
    """

    def __init__(self, folder_path: Path):
        super().__init__()
        self.setWindowTitle("Cell Viewer")
        self.setGeometry(100, 100, 1000, 1000)

        self.folder = folder_path
        self.cells_file = self.folder / "active_cells.pkl"
        self.overlay_file = self.folder / "overlay.TIF"

        try:
            self.cells = load_existing_cells(self.cells_file, True)
            self.overlay_img = tifffile.imread(str(self.overlay_file))
        except Exception as e:
            logger.error(f"Failed to load viewer data: {e}")
            self.cells = []
            self.overlay_img = np.zeros((512, 512), dtype=np.uint16)

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.setCentralWidget(self.view)

        self.pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(self.to_qimage(self.overlay_img)))
        self.scene.addItem(self.pixmap_item)

        self.valid_pen = QPen(Qt.green)
        self.invalid_pen = QPen(Qt.red)
        self.valid_pen.setWidth(2)
        self.invalid_pen.setWidth(2)
        self.centroid_brush = QBrush(Qt.red)

        self.current_highlight = None
        self.current_centroid = None
        self.current_label = None

        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)

    def to_qimage(self, img: np.ndarray) -> QImage:
        """
        Convert a 16-bit grayscale numpy image to an 8-bit QImage.

        Args:
            img (np.ndarray): Image array.

        Returns:
            QImage: Converted image.
        """
        norm_img = ((img / 65535) * 255).astype(np.uint8)
        h, w = norm_img.shape
        return QImage(norm_img.data, w, h, w, QImage.Format_Grayscale8)

    def eventFilter(self, source, event):
        """
        Event filter to handle hover and click interactions with cells.
        """
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
                    try:
                        show_cell_plot(selected_cell)
                    except Exception as e:
                        logger.warning(f"Failed to plot intensity: {e}")

        return super().eventFilter(source, event)

    def find_closest_cell(self, y: int, x: int, radius: int = 8):
        """
        Find the closest cell to the given pixel coordinates.

        Args:
            y (int): Y-coordinate.
            x (int): X-coordinate.
            radius (int): Search radius.

        Returns:
            Cell | None: Closest Cell object if found.
        """
        for cell in self.cells:
            cy, cx = cell.centroid
            if abs(cx - x) <= radius and abs(cy - y) <= radius:
                return cell
        return None

    def highlight_cell(self, cell):
        """
        Draw a rectangle and label to highlight a cell.

        Args:
            cell (Cell): Cell to highlight.
        """
        self.clear_highlight()

        min_yx = cell.pixel_coords.min(axis=0)
        max_yx = cell.pixel_coords.max(axis=0)
        top_left = min_yx[::-1]
        bottom_right = max_yx[::-1]
        rect = QRectF(top_left[0], top_left[1], bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])

        pen = self.valid_pen if cell.is_valid else self.invalid_pen
        self.current_highlight = self.scene.addRect(rect, pen)

        cx, cy = cell.centroid[1], cell.centroid[0]
        self.current_centroid = self.scene.addEllipse(cx - 2, cy - 2, 4, 4, pen, self.centroid_brush)

        label_text = f"ID: {cell.label} ({cx}, {cy})"
        self.current_label = QGraphicsTextItem(label_text)
        self.current_label.setDefaultTextColor(Qt.white)
        self.current_label.setPos(cx + 5, cy - 15)
        self.scene.addItem(self.current_label)

    def clear_highlight(self):
        """
        Clear any active cell highlight.
        """
        for item in [self.current_highlight, self.current_centroid, self.current_label]:
            if item:
                self.scene.removeItem(item)
        self.current_highlight = None
        self.current_centroid = None
        self.current_label = None
