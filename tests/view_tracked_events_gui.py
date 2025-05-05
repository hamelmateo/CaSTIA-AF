from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QDockWidget, QCheckBox, QPushButton
)
from PyQt5.QtGui import QPixmap, QImage, QPen, QPolygonF, QColor
from PyQt5.QtCore import Qt, QPointF
import sys
from pathlib import Path
import numpy as np
import tifffile
import pandas as pd
import random

from calcium_activity_characterization.utilities.loader import load_cells_from_pickle, load_pickle_file


class EventOverlayViewer(QMainWindow):
    def __init__(self, folder_path: Path):
        super().__init__()
        self.setWindowTitle("Event Overlay Viewer")
        self.setGeometry(100, 100, 1200, 1000)

        self.folder = folder_path
        self.overlay_file = self.folder / "overlay.TIF"
        self.cells_file = self.folder / "processed_active_cells.pkl"
        self.events_file = self.folder / "tracked_events.pkl"

        self.overlay_img = tifffile.imread(str(self.overlay_file))
        if self.overlay_img.ndim == 3:
            self.overlay_img = self.overlay_img[0]  # Just use first frame if 3D

        self.cells = load_cells_from_pickle(self.cells_file, load=True)
        self.events_df = load_pickle_file(self.events_file)
        self.max_frame = self.events_df['frame'].max()

        self.frame_events = self._build_frame_event_dict()
        self.event_colors = self._assign_event_colors()

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.frame_label = QLabel("Frame: 0")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.max_frame)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_overlay)

        self.show_overlay_checkbox = QCheckBox("Show Event Overlay")
        self.show_overlay_checkbox.setChecked(True)
        self.show_overlay_checkbox.stateChanged.connect(self.update_overlay)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_frame)

        # Layouts
        control_widget = QWidget()
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.frame_label)
        control_layout.addWidget(self.slider)
        control_layout.addWidget(self.next_button)
        control_layout.addWidget(self.show_overlay_checkbox)
        control_widget.setLayout(control_layout)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.view)
        main_layout.addWidget(control_widget)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.overlay_items = []

        self.update_overlay()

    def _build_frame_event_dict(self):
        frame_events = {}
        for frame, group in self.events_df.groupby("frame"):
            frame_events[frame] = [(row.trackID, row.event_id) for row in group.itertuples()]
        return frame_events

    def _assign_event_colors(self):
        unique_events = self.events_df['event_id'].unique()
        color_map = {
            eid: QColor(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            for eid in unique_events
        }
        return color_map

    def update_overlay(self):
        frame = self.slider.value()
        self.frame_label.setText(f"Frame: {frame}")

        # Draw base image
        base_img = self._to_qimage(self.overlay_img)
        self.pixmap_item.setPixmap(QPixmap.fromImage(base_img))

        # Remove previous overlays
        for item in self.overlay_items:
            self.scene.removeItem(item)
        self.overlay_items.clear()

        if self.show_overlay_checkbox.isChecked():
            for cell_id, event_id in self.frame_events.get(frame, []):
                cell = next((c for c in self.cells if c.label == cell_id), None)
                if not cell:
                    continue
                color = self.event_colors.get(event_id, QColor("red"))
                pen = QPen(color, 2)
                polygon = QPolygonF([QPointF(x, y) for y, x in cell.pixel_coords])
                item = self.scene.addPolygon(polygon, pen)
                self.overlay_items.append(item)

    def next_frame(self):
        current = self.slider.value()
        if current < self.max_frame:
            self.slider.setValue(current + 1)

    def _to_qimage(self, img: np.ndarray) -> QImage:
        norm_img = ((img / 65535) * 255).astype(np.uint8)
        h, w = norm_img.shape
        return QImage(norm_img.data, w, h, w, QImage.Format_Grayscale8)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    from PyQt5.QtWidgets import QFileDialog
    folder = QFileDialog.getExistingDirectory(None, "Select Output Folder")
    if folder:
        viewer = EventOverlayViewer(Path(folder))
        viewer.show()
        sys.exit(app.exec_())
