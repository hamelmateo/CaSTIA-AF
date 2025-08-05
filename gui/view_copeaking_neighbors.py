# view_copeaking_neighbors.py
# Usage: Run this script and select a folder with 'nuclei_mask.TIF' and a population.pkl

import sys
from pathlib import Path
import numpy as np
import pickle
import tifffile
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGraphicsView, QGraphicsScene, QPushButton,
    QSlider, QFileDialog, QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor
from PyQt5.QtCore import Qt, QTimer
import random


class CoPeakingNeighborViewer(QMainWindow):
    """
    GUI to visualize co-peaking spatial neighbor groups over time.
    Co-peaking groups are shown in different colors per frame.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Co-Peaking Neighbors Viewer")
        self.setGeometry(100, 100, 1400, 1000)

        self.population = None
        self.base_rgb = None
        self.max_frame = 0
        self.label_map = {}

        self.timer = QTimer()
        self.timer.setInterval(500)
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

        overlay_path = folder / "nuclei_mask.TIF"
        if not overlay_path.exists():
            overlay_path = folder / "nuclei_mask.tif"
        if not overlay_path.exists():
            return

        overlay = tifffile.imread(str(overlay_path))
        base = np.zeros((*overlay.shape, 3), dtype=np.uint8)
        base[overlay > 0] = [128, 128, 128]
        self.base_rgb = base

        self.max_frame = len(self.population.cells[0].trace.binary) - 1
        self.slider.setMaximum(self.max_frame)

        self.label_map = {cell.label: cell for cell in self.population.cells}

        self.update_frame()

    def update_frame(self):
        if self.population is None:
            return

        frame = self.slider.value()
        self.frame_label.setText(f"Frame: {frame}")
        mask = self.base_rgb.copy()

        frame_groups = [grp for grp in getattr(self.population, 'copeaking_neighbors', []) if grp.frame == frame]
        for group in frame_groups:
            color = self.random_color()
            for label in group.get_labels():
                cell = self.label_map.get(label)
                if cell:
                    for y, x in cell.pixel_coords:
                        mask[y, x] = color

        qimg = QImage(mask.data, mask.shape[1], mask.shape[0], QImage.Format_RGB888)
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(qimg))
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def random_color(self):
        r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        return [r, g, b]

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
    viewer = CoPeakingNeighborViewer()
    viewer.show()
    sys.exit(app.exec_())
