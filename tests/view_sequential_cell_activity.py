# view_sequential_cell_activity.py
# Usage Example:
# >>> Run this script to explore origin vs caused peaks with arrows between cells

import sys
from pathlib import Path
import numpy as np
import pickle
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGraphicsView, QGraphicsScene, QPushButton,
    QSlider, QFileDialog, QLineEdit, QCheckBox
)
from PyQt5.QtGui import QPixmap, QImage, QPen
from PyQt5.QtCore import Qt, QTimer

class SequentialSignalViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sequential Peak Viewer")
        self.setGeometry(100, 100, 1400, 1000)

        self.population = []
        self.max_frame = 0
        self.image_shape = None
        self.timer = QTimer()
        self.timer.setInterval(20)
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
        if folder:
            with open(Path(folder) / "sequential_active_cells.pkl", 'rb') as f:
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
        mask, arrows = self.generate_mask(frame)

        qimg = QImage(mask.data, mask.shape[1], mask.shape[0], QImage.Format_RGB888)
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(qimg))

        pen = QPen(Qt.red, 2)
        for origin, target in arrows:
            self.scene.addLine(origin[1], origin[0], target[1], target[0], pen)
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def generate_mask(self, frame):
        mask = np.zeros((*self.image_shape, 3), dtype=np.uint8)
        arrows = []
        label_map = {cell.label: cell for cell in self.population.cells}

        for cell in self.population.cells:
            if frame >= len(cell.trace.binary):
                continue

            is_active = cell.trace.binary[frame] == 1
            color = [128, 128, 128]  # Default: gray for inactive

            if is_active:
                peak = next((p for p in cell.trace.peaks if p.rel_start_time <= frame <= p.rel_end_time), None)
                if peak:
                    if peak.cause_type == "origin" and self.show_origin.isChecked():
                        color = [0, 255, 0]
                    elif peak.cause_type == "caused" and self.show_caused.isChecked():
                        color = [255, 165, 0]
                        origin = label_map.get(peak.origin_label)
                        if origin:
                            arrows.append((origin.centroid, cell.centroid))
                    elif peak.cause_type == "individual" and self.show_individual.isChecked():
                        color = [0, 128, 255]
                    else:
                        color = [0, 0, 0]  # Hidden

            for y, x in cell.pixel_coords:
                mask[y, x] = color

        return mask, arrows

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
    viewer = SequentialSignalViewer()
    viewer.show()
    sys.exit(app.exec_())
