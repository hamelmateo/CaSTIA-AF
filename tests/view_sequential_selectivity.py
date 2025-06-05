# view_sequential_selectivity.py
# Usage: Run this script and select an output folder containing nuclei_mask.TIF
# and sequential_active_cells.pkl. The nuclei mask will be used as a gray
# background.

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
from PyQt5.QtCore import Qt, QTimer


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

        self.timer = QTimer()
        self.timer.setInterval(500)  # 0.5 s per frame
        self.timer.timeout.connect(self.next_frame)

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
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

        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)
        self.setCentralWidget(main_widget)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        folder = Path(folder)
        pkl_path = folder / "sequential_active_cells.pkl"
        if not pkl_path.exists():
            return
        with open(pkl_path, 'rb') as f:
            self.population = pickle.load(f)

        if not self.population.cells:
            return

        mask_path = folder / "nuclei_mask.TIF"
        if not mask_path.exists():
            mask_path = folder / "nuclei_mask.tiff"
        if mask_path.exists():
            mask = tifffile.imread(str(mask_path))
            base = np.zeros((*mask.shape, 3), dtype=np.uint8)
            base[mask > 0] = [128, 128, 128]
            self.base_rgb = base
        else:
            overlay_path = folder / "overlay.TIF"
            if not overlay_path.exists():
                overlay_path = folder / "overlay.tif"
            if overlay_path.exists():
                overlay = tifffile.imread(str(overlay_path))
                self.base_rgb = np.stack([overlay] * 3, axis=-1).astype(np.uint8)
            else:
                all_coords = np.vstack([cell.pixel_coords for cell in self.population.cells])
                shape = (all_coords[:, 0].max() + 1, all_coords[:, 1].max() + 1)
                self.base_rgb = np.zeros((*shape, 3), dtype=np.uint8)

        self.max_frame = len(self.population.cells[0].trace.binary) - 1
        self.slider.setMaximum(self.max_frame)

        self.label_map = {cell.label: cell for cell in self.population.cells}
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
            start = cause_peak.rel_start_time
            end = origin_peak.rel_end_time
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
                if peak.rel_start_time <= frame <= peak.rel_end_time:
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

        pen = QPen(Qt.red, 2)
        for start, end, origin_cent, cause_cent in self.arrow_intervals:
            if start <= frame <= end:
                self.scene.addLine(origin_cent[1], origin_cent[0], cause_cent[1], cause_cent[0], pen)
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

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
