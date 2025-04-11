from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QMainWindow,
    QGraphicsTextItem, QCheckBox, QVBoxLayout, QWidget, QDockWidget
)
from PyQt5.QtGui import QPixmap, QImage, QPen, QBrush, QPolygonF
from PyQt5.QtCore import Qt, QPointF
from src.io.loader import load_cells_from_pickle
import tifffile
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Callable


logger = logging.getLogger(__name__)

class OverlayViewer(QMainWindow):
    def __init__(self, folder_path: Path):
        super().__init__()
        self.setWindowTitle("Cell Viewer")
        self.setGeometry(100, 100, 1000, 1000)

        self.folder = folder_path
        self.cells_file = self.folder / "active_cells.pkl"
        self.overlay_file = self.folder / "overlay.TIF"
        self.cell_click_callback: Optional[Callable[[object], None]] = None

        try:
            self.cells = load_cells_from_pickle(self.cells_file, True)
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

        self.active_cell_items = []
        self.inactive_cell_items = []
        self.current_highlight = []

        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)

        self.checkbox_active = QCheckBox("Show Active Cells")
        self.checkbox_inactive = QCheckBox("Show Inactive Cells")
        self.checkbox_active.stateChanged.connect(self.toggle_overlays)
        self.checkbox_inactive.stateChanged.connect(self.toggle_overlays)

        layout = QVBoxLayout()
        layout.addWidget(self.checkbox_active)
        layout.addWidget(self.checkbox_inactive)

        controls = QWidget()
        controls.setLayout(layout)

        dock = QDockWidget("Overlay Controls", self)
        dock.setWidget(controls)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def to_qimage(self, img: np.ndarray) -> QImage:
        norm_img = ((img / 65535) * 255).astype(np.uint8)
        h, w = norm_img.shape
        return QImage(norm_img.data, w, h, w, QImage.Format_Grayscale8)

    def toggle_overlays(self):
        self.clear_all_overlays()
        if self.checkbox_active.isChecked():
            for cell in self.cells:
                if cell.is_valid:
                    self.draw_outline(cell, self.valid_pen, self.active_cell_items)
        if self.checkbox_inactive.isChecked():
            for cell in self.cells:
                if not cell.is_valid:
                    self.draw_outline(cell, self.invalid_pen, self.inactive_cell_items)

    def draw_outline(self, cell, pen, store_list):
        polygon = QPolygonF([QPointF(x, y) for y, x in cell.pixel_coords])
        item = self.scene.addPolygon(polygon, pen)
        store_list.append(item)

    def clear_all_overlays(self):
        for item in self.active_cell_items + self.inactive_cell_items + self.current_highlight:
            self.scene.removeItem(item)
        self.active_cell_items.clear()
        self.inactive_cell_items.clear()
        self.current_highlight.clear()

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
                if selected_cell and self.cell_click_callback:
                    try:
                        self.cell_click_callback(selected_cell)

                    except Exception as e:
                        logger.warning(f"Failed to plot intensity: {e}")

        return super().eventFilter(source, event)

    def find_closest_cell(self, y: int, x: int, radius: int = 8):
        for cell in self.cells:
            cy, cx = cell.centroid
            if abs(cx - x) <= radius and abs(cy - y) <= radius:
                return cell
        return None

    def highlight_cell(self, cell):
        self.clear_highlight()
        polygon = QPolygonF([QPointF(x, y) for y, x in cell.pixel_coords])
        filled_brush = QBrush(Qt.green)
        item = self.scene.addPolygon(polygon, self.valid_pen, filled_brush)
        self.current_highlight.append(item)

    def clear_highlight(self):
        for item in self.current_highlight:
            self.scene.removeItem(item)
        self.current_highlight.clear()

    def set_callback(self, callback: Callable[[object], None]):
        """
        Register a function to call when a cell is clicked.
        """
        self.cell_click_callback = callback

    def highlight_cell_by_label(self, label: int):
        """
        Highlight a specific cell in the overlay by its label.
        """
        for cell in self.cells:
            if cell.label == label:
                self.clear_highlight()
                self.highlight_cell(cell)
            break
