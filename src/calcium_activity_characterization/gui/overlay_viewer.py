from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QMainWindow,
    QPushButton, QVBoxLayout, QWidget, QDockWidget, QCheckBox, QDialog
)
from PyQt5.QtGui import QPixmap, QImage, QPen, QBrush, QPolygonF
from PyQt5.QtCore import Qt, QPointF
import tifffile
import numpy as np
from pathlib import Path
import logging
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from calcium_activity_characterization.utilities.loader import load_cells_from_pickle, load_pickle_file
from calcium_activity_characterization.processing.signal_processing import SignalProcessor
from config import config

logger = logging.getLogger(__name__)

class OverlayViewer(QMainWindow):
    def __init__(self, folder_path: Path):
        super().__init__()
        self.setWindowTitle("Cell Viewer")
        self.setGeometry(100, 100, 1000, 1000)

        self.folder = folder_path
        self.cells_file = self.folder / "processed_active_cells.pkl"
        self.overlay_file = self.folder / "overlay.TIF"
        self.binarized_data_file = self.folder / "binarized_data.pkl"
        self.selected_cell = None

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

        self.valid_pen = QPen(Qt.green, 2)
        self.highlight_pen = QPen(Qt.yellow, 2)
        self.current_highlight = []

        self.active_cell_items = []

        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)

        # Overlay checkbox
        self.checkbox_active = QCheckBox("Show Cells")
        self.checkbox_active.stateChanged.connect(self.toggle_overlays)

        layout = QVBoxLayout()
        layout.addWidget(self.checkbox_active)

        controls = QWidget()
        controls.setLayout(layout)

        dock = QDockWidget("Controls", self)
        dock.setWidget(controls)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        try:
            self.binarized_df = load_pickle_file(self.binarized_data_file)
        except Exception as e:
            logger.error(f"Could not load binarized data: {e}")
            self.binarized_df = pd.DataFrame()

    def to_qimage(self, img: np.ndarray) -> QImage:
        norm_img = ((img / 65535) * 255).astype(np.uint8)
        h, w = norm_img.shape
        return QImage(norm_img.data, w, h, w, QImage.Format_Grayscale8)

    def toggle_overlays(self):
        self.clear_all_overlays()
        if self.checkbox_active.isChecked():
            for cell in self.cells:
                self.draw_outline(cell, self.valid_pen, self.active_cell_items)

    def draw_outline(self, cell, pen, store_list):
        polygon = QPolygonF([QPointF(x, y) for y, x in cell.pixel_coords])
        item = self.scene.addPolygon(polygon, pen)
        store_list.append(item)

    def clear_all_overlays(self):
        for item in self.active_cell_items + self.current_highlight:
            self.scene.removeItem(item)
        self.active_cell_items.clear()
        self.current_highlight.clear()

    def highlight_cell(self, cell):
        self.clear_highlight()
        polygon = QPolygonF([QPointF(x, y) for y, x in cell.pixel_coords])
        item = self.scene.addPolygon(polygon, self.highlight_pen)
        self.current_highlight.append(item)

    def clear_highlight(self):
        for item in self.current_highlight:
            self.scene.removeItem(item)
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

        elif event.type() == event.MouseButtonPress and event.button() == Qt.LeftButton:
            pos = event.pos()
            img_pos = self.view.mapToScene(pos)
            x, y = int(img_pos.x()), int(img_pos.y())
            selected_cell = self.find_closest_cell(y, x, radius=8)
            if selected_cell:
                self.selected_cell = selected_cell
                self.plot_all_traces()
        return super().eventFilter(source, event)

    def find_closest_cell(self, y: int, x: int, radius: int = 8):
        for cell in self.cells:
            cy, cx = cell.centroid
            if abs(cx - x) <= radius and abs(cy - y) <= radius:
                return cell
        return None

    def plot_all_traces(self):
        cell = self.selected_cell
        if not cell:
            return

        processor = SignalProcessor(config.DETRENDING_MODE, config.SIGNAL_PROCESSING_PARAMETERS[config.DETRENDING_MODE])
        raw = np.array(cell.raw_intensity_trace)
        processed = processor.run(raw)

        bin_row = self.binarized_df[self.binarized_df["trackID"] == cell.label].sort_values("frame")

        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=False)

        axs[0].plot(raw, label="Raw", color="black")
        axs[0].set_title(f"Raw Intensity - Cell {cell.label}")

        axs[1].plot(processed, label="Processed", color="blue")
        axs[1].set_title("Processed with Custom Method")

        if not bin_row.empty:
            axs[2].plot(bin_row["frame"], bin_row["intensity.resc"].values, label="Rescaled", color="blue")
            axs[2].step(
                bin_row["frame"],
                bin_row["intensity.bin"] * bin_row["intensity.resc"].max(),
                label="Binarized",
                color="red",
                where="post"
            )
            axs[2].set_title("ARCOS Binarized")
        else:
            axs[2].text(0.5, 0.5, "No ARCOS data", ha="center", va="center")

        for ax in axs:
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Cell {cell.label} - Trace Viewer")
        layout = QVBoxLayout(dlg)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        dlg.resize(1000, 700)
        dlg.exec_()
