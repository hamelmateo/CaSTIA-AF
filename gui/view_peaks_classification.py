# view_sequential_selectivity.py
# Usage: Run this script and select an output folder containing overlay.TIF and sequential_active_cells.pkl

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
from PyQt5.QtGui import QPixmap, QImage, QPen
from PyQt5.QtCore import Qt, QTimer, QPoint


class SequentialSelectivityViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sequential Selectivity Viewer")
        self.setGeometry(100, 100, 1400, 1000)

        self.population = None
        self.base_rgb = None
        self.max_frame = 0
        self.label_map = {}
        self.arrow_intervals = []  # (start, end, origin_centroid, cause_centroid)

        self.pixel_to_label = {}  # (y, x) -> label
        self.hover_label = QLabel("Hovered Label: None")
        self.last_hovered_label = None

        self.timer = QTimer()
        self.timer.setInterval(500)  # 0.5 s per frame
        self.timer.timeout.connect(self.next_frame)

        self.hover_timer = QTimer()
        self.hover_timer.setInterval(3000)  # Refresh every 3 seconds
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

        self.show_origin = QCheckBox("Show Origin")
        self.show_origin.setChecked(True)
        controls_layout.addWidget(self.show_origin)

        self.show_caused = QCheckBox("Show Caused")
        self.show_caused.setChecked(True)
        controls_layout.addWidget(self.show_caused)

        self.show_individual = QCheckBox("Show Individual")
        self.show_individual.setChecked(True)
        controls_layout.addWidget(self.show_individual)

        controls_layout.addWidget(self.hover_label)
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)
        self.setCentralWidget(main_widget)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        folder = Path(folder)
        pkl_path = folder / "population-snapshots/04_population_events.pkl"
        if not pkl_path.exists():
            return
        with open(pkl_path, 'rb') as f:
            self.population = pickle.load(f)

        if not self.population.cells:
            return

        overlay_path = folder / "cell-mapping/nuclei_mask.TIF"
        if not overlay_path.exists():
            overlay_path = folder / "cell-mapping/nuclei_mask.tif"
        if overlay_path.exists():
            overlay = tifffile.imread(str(overlay_path))
            base = np.zeros((*overlay.shape, 3), dtype=np.uint8)
            base[overlay > 0] = [128, 128, 128]  # Gray color for overlay
            self.base_rgb = base
        else:
            raise FileNotFoundError(f"Overlay file not found in {folder}")

        self.max_frame = len(self.population.cells[0].trace.binary) - 1
        self.slider.setMaximum(self.max_frame)

        self.label_map = {cell.label: cell for cell in self.population.cells}
        self.pixel_to_label.clear()
        for cell in self.population.cells:
            for y, x in cell.pixel_coords:
                self.pixel_to_label[(y, x)] = cell.label

        self.precompute_arrows()
        self.update_frame()

    def precompute_arrows(self):
        self.arrow_intervals = []
        comms = getattr(self.population, 'cell_to_cell_communications', None)
        if not comms:
            return
        for comm in comms:
            o_label, o_idx = comm.origin
            c_label, c_idx = comm.cause
            origin_cell = self.label_map.get(o_label)
            cause_cell = self.label_map.get(c_label)
            if origin_cell is None or cause_cell is None:
                continue
            try:
                origin_peak = origin_cell.trace.peaks[o_idx]
                cause_peak = cause_cell.trace.peaks[c_idx]
            except IndexError:
                continue
            start = cause_peak.communication_time
            end = min(origin_peak.activation_end_time, cause_peak.activation_end_time)
            self.arrow_intervals.append((start, end, origin_cell.centroid, cause_cell.centroid))

    def update_frame(self):
        if self.population is None:
            return
        frame = self.slider.value()
        self.frame_label.setText(f"Frame: {frame}")

        mask = self.base_rgb.copy()
        for cell in self.population.cells:
            if frame >= len(cell.trace.binary):
                continue
            active_peak = None
            for peak in cell.trace.peaks:
                if peak.communication_time <= frame <= peak.activation_end_time:
                    active_peak = peak
                    break
            if not active_peak:
                continue
            color = None
            if active_peak.origin_type == 'origin' and self.show_origin.isChecked():
                color = [0, 255, 0]
            elif active_peak.origin_type == 'caused' and self.show_caused.isChecked():
                color = [255, 165, 0]
            elif active_peak.origin_type == 'individual' and self.show_individual.isChecked():
                color = [0, 128, 255]
            if color is not None:
                for y, x in cell.pixel_coords:
                    mask[y, x] = color

        qimg = QImage(mask.data, mask.shape[1], mask.shape[0], QImage.Format_RGB888)
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(qimg))

        """
        pen = QPen(Qt.red, 2)
        for start, end, origin_cent, cause_cent in self.arrow_intervals:
            if start <= frame <= end:
                self.scene.addLine(origin_cent[1], origin_cent[0], cause_cent[1], cause_cent[0], pen)
        """
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def eventFilter(self, obj, event):
        if obj is self.view.viewport() and event.type() == event.MouseMove:
            pos = event.pos()
            img_pos = self.view.mapToScene(pos).toPoint()
            y, x = img_pos.y(), img_pos.x()
            label = self.pixel_to_label.get((y, x))
            self.last_hovered_label = label
        return super().eventFilter(obj, event)

    def refresh_hover_display(self):
        label = self.last_hovered_label
        if label is not None:
            self.hover_label.setText(f"Hovered Label: {label}")
        else:
            self.hover_label.setText("Hovered Label: None")

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
    viewer = SequentialSelectivityViewer()
    viewer.show()
    sys.exit(app.exec_())