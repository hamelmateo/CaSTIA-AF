# view_dominant_direction_gui.py
# Usage: Run this script and select a folder containing 'nuclei_mask.TIF' and '04_population_events.pkl'.
# Visualizes the dominant direction of GlobalEvent using temporal bins and centroid evolution.

import sys
from pathlib import Path
import numpy as np
import pickle
import tifffile
from scipy.spatial.distance import pdist

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QGraphicsView, QGraphicsScene, QPushButton, QSlider,
    QFileDialog, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox
)
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor, QBrush
from PyQt5.QtCore import Qt, QTimer, QPointF

from calcium_activity_characterization.data.populations import Population
from calcium_activity_characterization.data.events import GlobalEvent
from calcium_activity_characterization.config.structures import DirectionComputationParams


class DominantDirectionViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GINT - Global Event Direction Inspector")
        self.setGeometry(100, 100, 1400, 1000)

        self.population: Population = None
        self.global_events: list[GlobalEvent] = []
        self.current_event: GlobalEvent = None

        self.base_rgb: np.ndarray = None
        self.label_map = {}  # label -> Cell
        self.overlay_loaded = False

        self.scene = QGraphicsScene(self)
        self.timer = QTimer()
        self.timer.setInterval(800)
        self.timer.timeout.connect(self.next_bin)

        self.current_bin_index = 0
        self.result_metadata = None

        self._setup_ui()

    def _setup_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # Graphics view on left
        self.view = QGraphicsView(self.scene, self)
        main_layout.addWidget(self.view, 5)

        # Control panel on right
        controls = QVBoxLayout()

        self.load_button = QPushButton("Load Folder")
        self.load_button.clicked.connect(self.load_folder)
        controls.addWidget(self.load_button)

        self.event_selector = QComboBox()
        self.event_selector.currentIndexChanged.connect(self.select_event)
        controls.addWidget(self.event_selector)

        controls.addWidget(QLabel("# Time Bins:"))
        self.bin_spin = QSpinBox()
        self.bin_spin.setMinimum(2)
        self.bin_spin.setMaximum(20)
        self.bin_spin.setValue(4)
        controls.addWidget(self.bin_spin)

        controls.addWidget(QLabel("MAD Multiplier:"))
        self.mad_spin = QDoubleSpinBox()
        self.mad_spin.setRange(0.0, 10.0)
        self.mad_spin.setSingleStep(0.1)
        self.mad_spin.setValue(2.0)
        controls.addWidget(self.mad_spin)

        controls.addWidget(QLabel("Min Disp. Ratio:"))
        self.disp_spin = QDoubleSpinBox()
        self.disp_spin.setRange(0.0, 1.0)
        self.disp_spin.setSingleStep(0.05)
        self.disp_spin.setValue(0.1)
        controls.addWidget(self.disp_spin)

        self.recompute_button = QPushButton("Recompute")
        self.recompute_button.clicked.connect(self.select_event)
        controls.addWidget(self.recompute_button)

        controls.addWidget(QLabel("Bin Viewer:"))
        self.bin_slider = QSlider(Qt.Horizontal)
        self.bin_slider.setMinimum(0)
        self.bin_slider.setValue(0)
        self.bin_slider.valueChanged.connect(self.update_bin_display)
        controls.addWidget(self.bin_slider)

        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.prev_bin)
        nav_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_bin)
        nav_layout.addWidget(self.next_button)

        controls.addLayout(nav_layout)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.timer.start)
        controls.addWidget(self.play_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.timer.stop)
        controls.addWidget(self.stop_button)

        self.show_mad = QCheckBox("Show MAD Circle")
        self.show_mad.setChecked(True)
        controls.addWidget(self.show_mad)

        self.show_links = QCheckBox("Show Bin Trajectory")
        self.show_links.setChecked(True)
        controls.addWidget(self.show_links)

        self.show_cells = QCheckBox("Show Peak Cells")
        self.show_cells.setChecked(True)
        controls.addWidget(self.show_cells)

        controls.addStretch()
        main_layout.addLayout(controls, 2)
        self.setCentralWidget(main_widget)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        folder = Path(folder)

        # Load population pickle
        pkl_path = folder / "population-snapshots" / "04_population_events.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Population file not found: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            self.population = pickle.load(f)
            self.global_events = [e for e in self.population.events if isinstance(e, GlobalEvent)]

        # Load nuclei mask (grayscale image)
        mask_path = folder / "cell-mapping" / "nuclei_mask.TIF"
        if not mask_path.exists():
            raise FileNotFoundError(f"Nuclei mask not found: {mask_path}")
        overlay = tifffile.imread(str(mask_path))
        base = np.zeros((*overlay.shape, 3), dtype=np.uint8)
        base[overlay > 0] = [128, 128, 128]
        self.base_rgb = base

        self.label_map = {cell.label: cell for cell in self.population.cells}
        self.event_selector.clear()
        for ev in self.global_events:
            self.event_selector.addItem(f"GlobalEvent {ev.id}", userData=ev)
        self.overlay_loaded = True
        self.select_event()

    def select_event(self):
        if not self.overlay_loaded:
            return

        index = self.event_selector.currentIndex()
        if index < 0:
            return

        self.current_event = self.event_selector.itemData(index)
        config = DirectionComputationParams(
            num_time_bins=self.bin_spin.value(),
            mad_filtering_multiplier=self.mad_spin.value(),
            min_net_displacement_ratio=self.disp_spin.value()
        )
        self.result_metadata = self.current_event._compute_dominant_direction_metadata(config)
        self.bin_slider.setMaximum(len(self.result_metadata.get("bins", [])) - 1)
        self.update_bin_display()

    def update_bin_display(self):
        if not self.overlay_loaded or self.result_metadata is None:
            return

        self.scene.clear()
        frame = self.bin_slider.value()
        mask = self.base_rgb.copy()

        # Show cells in this bin
        if self.show_cells.isChecked():
            bin_info = self.result_metadata['bins'][frame]
            color = (0, 255, 0)  # green for this bin
            for label in bin_info['label_ids']:
                cell = self.label_map.get(label)
                if cell:
                    for y, x in cell.pixel_coords:
                        mask[y, x] = color

        # Draw updated image
        qimg = QImage(mask.data, mask.shape[1], mask.shape[0], QImage.Format_RGB888)
        self.scene.addPixmap(QPixmap.fromImage(qimg))

        # Draw MAD circle and centroid
        if self.show_mad.isChecked():
            centroid = self.result_metadata['bins'][frame]['centroid']
            radius = self.result_metadata['bins'][frame]['radius']
            pen = QPen(Qt.yellow)
            pen.setStyle(Qt.DashLine)
            self.scene.addEllipse(
                centroid[1] - radius, centroid[0] - radius,
                radius * 2, radius * 2, pen
            )
            self.scene.addEllipse(centroid[1]-3, centroid[0]-3, 6, 6, QPen(Qt.yellow), QBrush(Qt.yellow))

        # Draw intra-bin links
        if self.show_links.isChecked():
            coms = self.result_metadata['bin_centroids']
            pen = QPen(Qt.blue)
            pen.setWidth(2)
            for i in range(1, len(coms)):
                prev = coms[i-1]
                curr = coms[i]
                self.scene.addLine(prev[1], prev[0], curr[1], curr[0], pen)

        # Draw final dominant direction
        direction = self.result_metadata['direction_vector']
        if direction != (0.0, 0.0):
            anchor = self.result_metadata['bin_centroids'][0]
            scale = 100.0
            dx, dy = direction[1], direction[0]
            x1, y1 = anchor[1], anchor[0]
            x2, y2 = x1 + dx * scale, y1 + dy * scale
            red_pen = QPen(Qt.red)
            red_pen.setWidth(3)
            self.scene.addLine(x1, y1, x2, y2, red_pen)

        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def prev_bin(self):
        self.bin_slider.setValue(max(0, self.bin_slider.value() - 1))

    def next_bin(self):
        self.bin_slider.setValue(min(self.bin_slider.maximum(), self.bin_slider.value() + 1))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = DominantDirectionViewer()
    viewer.show()
    sys.exit(app.exec_())
