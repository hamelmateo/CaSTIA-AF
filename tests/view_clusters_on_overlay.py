import sys
import numpy as np
import pickle
import tifffile
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QLineEdit, QTextEdit, QCheckBox
)
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene
from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.data.clusters import Cluster
from calcium_activity_characterization.utilities.loader import generate_distinct_colors

class HoverableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.hover_pos = QPoint()
        self.label_callback = None

    def mouseMoveEvent(self, event):
        self.hover_pos = event.pos()
        if self.label_callback:
            self.label_callback(event)
        super().mouseMoveEvent(event)

class ClusterOverlayViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Peak Clusters Overlay Viewer")
        self.setGeometry(100, 100, 1400, 1000)

        self.overlay = None
        self.cells = []
        self.clusters = []
        self.max_frame = 0
        self.frame_cluster_map = {}
        self.frame_peak_map = {}
        self.cluster_colors = {}
        self.show_peak_durations = False

        self.hovered_cell_label = QLabel("Hovered Cell: -")
        self.hovered_cell = None

        self.timer = QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.next_frame)

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        self.scene = QGraphicsScene(self)
        self.view = HoverableGraphicsView(self)
        self.view.setScene(self.scene)
        self.view.label_callback = self.update_hover_label
        main_layout.addWidget(self.view, 5)

        controls = QVBoxLayout()

        self.frame_label = QLabel("Frame: 0")
        controls.addWidget(self.frame_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.update_frame)
        controls.addWidget(self.slider)

        frame_input = QHBoxLayout()
        self.frame_input = QLineEdit()
        self.goto_btn = QPushButton("Go to frame")
        self.goto_btn.clicked.connect(self.goto_frame)
        frame_input.addWidget(self.frame_input)
        frame_input.addWidget(self.goto_btn)
        controls.addLayout(frame_input)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.start)
        controls.addWidget(self.play_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop)
        controls.addWidget(self.stop_btn)

        self.next_btn = QPushButton("Next Frame")
        self.next_btn.clicked.connect(self.next_frame)
        controls.addWidget(self.next_btn)

        self.load_btn = QPushButton("Load Folder")
        self.load_btn.clicked.connect(self.load_folder)
        controls.addWidget(self.load_btn)

        self.toggle_checkbox = QCheckBox("Show peak durations instead of cluster intervals")
        self.toggle_checkbox.stateChanged.connect(self.toggle_mode)
        controls.addWidget(self.toggle_checkbox)

        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        controls.addWidget(QLabel("Active Clusters Info:"))
        controls.addWidget(self.info_box)

        controls.addWidget(self.hovered_cell_label)

        controls.addStretch()
        main_layout.addLayout(controls, 1)

        self.setCentralWidget(main_widget)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder:
            return
        folder = Path(folder)

        with open(folder / "binarized_active_cells.pkl", 'rb') as f:
            self.cells = pickle.load(f)

        with open(folder / "peak_clusters.pkl", 'rb') as f:
            self.clusters = pickle.load(f)

        self.overlay = tifffile.imread(str(folder / "overlay.tif"))

        self.max_frame = max(p.end_time for c in self.cells for p in c.peaks)
        self.slider.setMaximum(self.max_frame)
        self.build_frame_maps()
        self.update_frame()

    def build_frame_maps(self):
        self.frame_cluster_map = {}
        self.frame_peak_map = {}
        self.cluster_colors = {}
        colors = generate_distinct_colors(n_colors=len(self.clusters))

        for i, cluster in enumerate(self.clusters):
            color = (np.array(colors[i % len(colors)]) * 255).astype(np.uint8)
            self.cluster_colors[cluster.id] = color

            for t in range(cluster.start_time, cluster.end_time + 1):
                for cell, _ in cluster.members:
                    self.frame_cluster_map.setdefault(t, []).append((cell, color, cluster))

            for cell, peak_idx in cluster.members:
                peak = cell.peaks[peak_idx]
                for t in range(peak.start_time, peak.end_time + 1):
                    self.frame_peak_map.setdefault(t, []).append((cell, color, cluster, peak))

    def update_hover_label(self, event):
        if self.overlay is None or not self.cells:
            return

        pos = self.view.mapToScene(event.pos()).toPoint()
        x, y = pos.x(), pos.y()
        hovered = None

        for cell in self.cells:
            if any(px == x and py == y for py, px in cell.pixel_coords):
                hovered = cell
                break

        if hovered:
            self.hovered_cell = hovered
            self.hovered_cell_label.setText(f"Hovered Cell: {hovered.label}")
        else:
            self.hovered_cell = None
            self.hovered_cell_label.setText("Hovered Cell: -")

        self.update_frame()

    def update_frame(self):
        if self.overlay is None or not self.cells:
            return

        frame = self.slider.value()
        self.frame_label.setText(f"Frame: {frame}")

        rgb = np.stack([self.overlay]*3, axis=-1).astype(np.uint8)
        active_clusters = set()

        frame_data = self.frame_peak_map if self.show_peak_durations else self.frame_cluster_map

        if self.show_peak_durations:
            for cell, color, cluster, _ in frame_data.get(frame, []):
                for y, x in cell.pixel_coords:
                    rgb[y, x] = color
                active_clusters.add(cluster)
        else:
            for cell, color, cluster in frame_data.get(frame, []):
                for y, x in cell.pixel_coords:
                    rgb[y, x] = color
                active_clusters.add(cluster)

        if self.hovered_cell:
            for y, x in self.hovered_cell.pixel_coords:
                rgb[y, x] = [255, 0, 0]

        html_lines = []
        for cluster in sorted(active_clusters, key=lambda c: c.id):
            color = self.cluster_colors[cluster.id]
            hex_color = QColor(*color).name()
            text = f"Cluster {cluster.id}: start={cluster.start_time}, end={cluster.end_time}, n_cells={len(cluster.members)}"
            html_lines.append(f"<span style='color:{hex_color}'>{text}</span>")
        self.info_box.setHtml("<br>".join(html_lines))

        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)

        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def toggle_mode(self):
        self.show_peak_durations = self.toggle_checkbox.isChecked()
        self.update_frame()

    def goto_frame(self):
        text = self.frame_input.text()
        if text.isdigit():
            frame = int(text)
            if 0 <= frame <= self.max_frame:
                self.slider.setValue(frame)

    def next_frame(self):
        frame = self.slider.value()
        if frame < self.max_frame:
            self.slider.setValue(frame + 1)
        else:
            self.stop()

    def start(self):
        self.timer.start()

    def stop(self):
        self.timer.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ClusterOverlayViewer()
    viewer.show()
    sys.exit(app.exec_())
