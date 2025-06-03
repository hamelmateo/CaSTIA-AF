"""
Viewer for binarized cell activity signals.
This script provides a GUI to visualize binarized signals of calcium activity from cells.
"""

import sys
from pathlib import Path
import numpy as np
import tifffile
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QPushButton, QSlider, QFileDialog, QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt, QTimer
import pickle

class BinarizedSignalViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Binarized Signal Viewer")
        self.setGeometry(100, 100, 1400, 1000)

        self.image_shape = None
        self.population = []
        self.max_frame = 0
        self.timer = QTimer()
        self.timer.setInterval(20)  # 50 ms per frame
        self.timer.timeout.connect(self.next_frame)

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # Graphics View
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        main_layout.addWidget(self.view, 5)

        # Controls
        controls_layout = QVBoxLayout()

        self.frame_label = QLabel("Frame: 0")
        controls_layout.addWidget(self.frame_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.update_frame)
        controls_layout.addWidget(self.slider)

        frame_input_layout = QHBoxLayout()
        self.frame_input = QLineEdit()
        self.goto_btn = QPushButton("Go to frame")
        self.goto_btn.clicked.connect(self.goto_frame)
        frame_input_layout.addWidget(self.frame_input)
        frame_input_layout.addWidget(self.goto_btn)
        controls_layout.addLayout(frame_input_layout)

        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_frame)
        controls_layout.addWidget(self.next_btn)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_animation)
        controls_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_animation)
        controls_layout.addWidget(self.stop_btn)

        self.load_btn = QPushButton("Load Output Folder")
        self.load_btn.clicked.connect(self.load_folder)
        controls_layout.addWidget(self.load_btn)

        controls_layout.addStretch()
        main_layout.addLayout(controls_layout, 1)

        self.setCentralWidget(main_widget)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            folder = Path(folder)
            cells_path = folder / "binarized_active_cells.pkl"

            with open(cells_path, 'rb') as f:
                self.population = pickle.load(f)

            if not self.population.cells:
                return

            self.max_frame = len(self.population.cells[0].trace.binary) - 1
            self.slider.setMaximum(self.max_frame)

            all_coords = np.vstack([cell.pixel_coords for cell in self.population.cells])
            self.image_shape = (all_coords[:, 0].max() + 1, all_coords[:, 1].max() + 1)

            self.update_frame()

    def update_frame(self):
        frame = self.slider.value()
        self.frame_label.setText(f"Frame: {frame}")

        mask = self.generate_mask(frame)
        qimg = QImage(mask.data, mask.shape[1], mask.shape[0], QImage.Format_RGB888)

        self.scene.clear()
        pixmap = QPixmap.fromImage(qimg)
        self.scene.addPixmap(pixmap)
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def generate_mask(self, frame):
        mask = np.zeros((*self.image_shape, 3), dtype=np.uint8)

        for cell in self.population.cells:
            if frame >= len(cell.trace.binary):
                continue
            state = cell.trace.binary[frame]
            color = [128, 128, 128] if state == 0 else [0, 255, 0]
            for y, x in cell.pixel_coords:
                mask[y, x] = color

        return mask

    def next_frame(self):
        frame = self.slider.value()
        if frame < self.max_frame:
            self.slider.setValue(frame + 1)
        else:
            self.stop_animation()

    def goto_frame(self):
        frame_str = self.frame_input.text()
        if frame_str.isdigit():
            frame = int(frame_str)
            if 0 <= frame <= self.max_frame:
                self.slider.setValue(frame)

    def start_animation(self):
        self.timer.start()

    def stop_animation(self):
        self.timer.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BinarizedSignalViewer()
    window.show()
    sys.exit(app.exec_())
