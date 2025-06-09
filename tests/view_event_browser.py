# view_sequential_selectivity.py
# Modified to visualize calcium events (not individual peaks)
# Usage: Run this script and select an output folder containing overlay.TIF and population_events.pkl

import sys
from pathlib import Path
import numpy as np
import pickle
import tifffile
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGraphicsView, QGraphicsScene, QPushButton,
    QSlider, QFileDialog, QLineEdit, QCheckBox
)
from PyQt5.QtGui import QPixmap, QImage, QPen, QPolygonF
from PyQt5.QtCore import Qt, QTimer, QPointF
from matplotlib import cm
from scipy.spatial import ConvexHull
from math import atan2, cos, sin, radians


class EventViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calcium Event Viewer")
        self.setGeometry(100, 100, 1400, 1000)

        self.population = None
        self.events = []
        self.base_rgb = None
        self.max_frame = 0
        self.label_map = {}
        self.pixel_to_label = {}  # (y, x) -> label
        self.hover_label = QLabel("Hovered Label: None")
        self.last_hovered_label = None

        self.timer = QTimer()
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.next_frame)

        self.hover_timer = QTimer()
        self.hover_timer.setInterval(3000)
        self.hover_timer.timeout.connect(self.refresh_hover_display)
        self.hover_timer.start()

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)
        main_layout.addWidget(self.view, 5)

        controls_layout = QVBoxLayout()
        self.frame_label = QLabel("Frame: 0")
        controls_layout.addWidget(self.frame_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.update_frame)
        controls_layout.addWidget(self.slider)

        self.frame_input = QLineEdit()
        self.goto_btn = QPushButton("Go to frame")
        self.goto_btn.clicked.connect(self.goto_frame)
        controls_layout.addWidget(self.frame_input)
        controls_layout.addWidget(self.goto_btn)

        step_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.prev_frame)
        step_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_frame)
        step_layout.addWidget(self.next_btn)

        controls_layout.addLayout(step_layout)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_animation)
        controls_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_animation)
        controls_layout.addWidget(self.stop_btn)

        self.load_btn = QPushButton("Load Folder")
        self.load_btn.clicked.connect(self.load_folder)
        controls_layout.addWidget(self.load_btn)

        self.show_direction = QCheckBox("Show Propagation Direction")
        self.show_direction.setChecked(True)
        controls_layout.addWidget(self.show_direction)

        self.show_wavefront = QCheckBox("Show Wavefront Hull")
        self.show_wavefront.setChecked(False)
        controls_layout.addWidget(self.show_wavefront)

        controls_layout.addWidget(self.hover_label)
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)
        self.setCentralWidget(main_widget)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        folder = Path(folder)

        with open(folder / "population_events.pkl", 'rb') as f:
            self.population = pickle.load(f)
            self.events = self.population.events

        overlay_path = folder / "nuclei_mask.TIF"
        if not overlay_path.exists():
            overlay_path = folder / "nuclei_mask.tif"
        overlay = tifffile.imread(str(overlay_path))
        base = np.zeros((*overlay.shape, 3), dtype=np.uint8)
        base[overlay > 0] = [128, 128, 128]
        self.base_rgb = base

        self.max_frame = len(self.population.cells[0].trace.binary) - 1
        self.slider.setMaximum(self.max_frame)

        self.label_map = {cell.label: cell for cell in self.population.cells}
        self.pixel_to_label = {(y, x): cell.label for cell in self.population.cells for y, x in cell.pixel_coords}

        self.event_colors = {ev.id: tuple((np.array(cm.tab20(ev.id % 20)[:3]) * 255).astype(int)) for ev in self.events}

        self.update_frame()

    def update_frame(self):
        if self.population is None:
            return

        frame = self.slider.value()
        self.frame_label.setText(f"Frame: {frame}")
        mask = self.base_rgb.copy()

        # Then overwrite with colored peaks from events
        for event in self.events:
            color = self.event_colors[event.id]
            start = event.event_start_time
            end = event.event_end_time
            for label in list({label for label, _ in event.peaks_involved}):
                cell = self.label_map.get(label)
                if cell:
                    for peak in cell.trace.peaks:
                        if start <= peak.rel_start_time <= end:
                            if peak.rel_start_time <= frame <= peak.rel_end_time:
                                for y, x in cell.pixel_coords:
                                    mask[y, x] = color

        qimg = QImage(mask.data, mask.shape[1], mask.shape[0], QImage.Format_RGB888)
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(qimg))

        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def eventFilter(self, obj, event):
        if obj is self.view.viewport() and event.type() == event.MouseMove:
            pos = event.pos()
            img_pos = self.view.mapToScene(pos).toPoint()
            y, x = img_pos.y(), img_pos.x()
            self.last_hovered_label = self.pixel_to_label.get((y, x))
        return super().eventFilter(obj, event)

    def refresh_hover_display(self):
        label = self.last_hovered_label
        if label is None:
            self.hover_label.setText("Hovered Label: None")
            return

        cell = self.label_map.get(label)
        frame = self.slider.value()

        active_event = None
        for event in self.events:
            if label in list({label for label, _ in event.peaks_involved}):
                for peak in cell.trace.peaks:
                    if event.event_start_time <= peak.rel_start_time <= event.event_end_time:
                        if peak.rel_start_time <= frame <= peak.rel_end_time:
                            active_event = event
                            break
            if active_event:
                break

        if active_event:
            info = (f"Hovered Label: {label}\n"
                    f"Event ID: {active_event.id}\n"
                    f"Cells: {active_event.n_cells_involved}\n"
                    f"Duration: {active_event.event_duration} frames\n"
                    f"Directional Speed: {active_event.directional_propagation_speed:.2f}\n")
        else:
            info = f"Hovered Label: {label}\nNo event info"

        self.hover_label.setText(info)

    def next_frame(self):
        frame = self.slider.value()
        if frame < self.max_frame:
            self.slider.setValue(frame + 1)
        else:
            self.stop_animation()

    def prev_frame(self):
        frame = self.slider.value()
        if frame > 0:
            self.slider.setValue(frame - 1)

    def goto_frame(self):
        text = self.frame_input.text()
        if text.isdigit():
            frame = int(text)
            if 0 <= frame <= self.max_frame:
                self.slider.setValue(frame)

    def start_animation(self):
        self.timer.start()

    def stop_animation(self):
        self.timer.stop()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = EventViewer()
    viewer.show()
    sys.exit(app.exec_())
