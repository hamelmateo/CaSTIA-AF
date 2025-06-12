"""
Viewer for binary traces of cells in a cluster, showing peaks and allowing navigation through clusters.
"""

import sys
import random
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QLineEdit, QTextEdit
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import pickle

from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.experimental.analysis.clusters import Cluster


class ClusterViewer(QMainWindow):
    def __init__(self, cells, clusters):
        super().__init__()
        self.setWindowTitle("Peak Cluster Viewer (Binarized, Multi-Cell)")
        self.resize(1600, 1000)

        self.cells = cells
        self.clusters = clusters
        self.current_cluster_idx = 0
        self.cluster_cells = []      # list of (cell, peak_index)
        self.selected_cells = []     # 5 cells shown in plot

        self._init_ui()
        self._load_cluster(0)

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # LEFT: Controls
        control_layout = QVBoxLayout()

        self.cluster_label = QLabel("Cluster: 0")
        control_layout.addWidget(self.cluster_label)

        self.cluster_info = QTextEdit()
        self.cluster_info.setReadOnly(True)
        control_layout.addWidget(self.cluster_info, stretch=2)

        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self._prev_cluster)
        nav_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self._next_cluster)
        nav_layout.addWidget(self.next_btn)
        control_layout.addLayout(nav_layout)

        self.goto_input = QLineEdit()
        self.goto_input.setPlaceholderText("Cluster ID")
        control_layout.addWidget(self.goto_input)

        self.goto_btn = QPushButton("Go")
        self.goto_btn.clicked.connect(self._goto_cluster)
        control_layout.addWidget(self.goto_btn)

        self.randomize_btn = QPushButton("Randomize Cells")
        self.randomize_btn.clicked.connect(self._randomize_cells)
        control_layout.addWidget(self.randomize_btn)

        control_layout.addStretch()
        layout.addLayout(control_layout, 2)

        # RIGHT: Plot area
        self.figure, self.axs = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 5)

    def _load_cluster(self, cluster_idx: int):
        if not (0 <= cluster_idx < len(self.clusters)):
            return

        self.current_cluster_idx = cluster_idx
        cluster = self.clusters[cluster_idx]
        self.cluster_label.setText(f"Cluster: {cluster.id}")

        # Always sort members by cell.label
        self.cluster_cells = sorted(cluster.members, key=lambda x: x[0].label)

        # Fix the first cell as the one with the lowest label
        fixed = self.cluster_cells[0]
        rest = self.cluster_cells[1:]
        sampled = random.sample(rest, min(4, len(rest)))
        self.selected_cells = [fixed] + sampled

        self._update_plot(cluster)
        self._update_info()

    def _update_plot(self, cluster):
        for ax in self.axs:
            ax.clear()

        for i, (cell, peak_idx) in enumerate(self.selected_cells):
            if i >= len(self.axs):
                break

            trace = cell.binary_trace
            peak = cell.peaks[peak_idx]

            self.axs[i].plot(trace, color='black')
            self.axs[i].set_ylabel(f"Cell {cell.label}")
            self.axs[i].axvspan(cluster.start_time, cluster.end_time, color='red', alpha=0.2)
            self.axs[i].plot(peak.peak_time, 1, 'ro', markersize=6)

            self.axs[i].set_ylim(-0.1, 1.1)
            self.axs[i].grid(True)

        self.axs[-1].set_xlabel("Time")
        self.figure.tight_layout()
        self.canvas.draw()

    def _update_info(self):
        self.cluster_info.clear()
        sorted_members = sorted(self.cluster_cells, key=lambda x: x[0].label)
        for cell, peak_idx in sorted_members:
            peak = cell.peaks[peak_idx]
            self.cluster_info.append(
                f"Cell {cell.label} - Peak {peak_idx}: "
                f"t={peak.peak_time}, dur={peak.rel_duration}, prom={peak.prominence:.2f}"
            )

    def _randomize_cells(self):
        if not self.cluster_cells:
            return

        fixed = self.cluster_cells[0]
        rest = self.cluster_cells[1:]
        sampled = random.sample(rest, min(4, len(rest)))
        self.selected_cells = [fixed] + sampled
        self._update_plot(self.clusters[self.current_cluster_idx])

    def _prev_cluster(self):
        if self.current_cluster_idx > 0:
            self._load_cluster(self.current_cluster_idx - 1)

    def _next_cluster(self):
        if self.current_cluster_idx < len(self.clusters) - 1:
            self._load_cluster(self.current_cluster_idx + 1)

    def _goto_cluster(self):
        text = self.goto_input.text().strip()
        if text.isdigit():
            cluster_id = int(text)
            for i, c in enumerate(self.clusters):
                if c.id == cluster_id:
                    self._load_cluster(i)
                    return


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    file_dialog = QFileDialog()
    cells_path, _ = file_dialog.getOpenFileName(None, "Select 03_binarized_traces.pkl", "", "*.pkl")
    if not cells_path:
        print("Cancelled.")
        sys.exit()

    clusters_path, _ = file_dialog.getOpenFileName(None, "Select peak_clusters.pkl", "", "*.pkl")
    if not clusters_path:
        print("Cancelled.")
        sys.exit()

    cells = load_pickle(Path(cells_path))
    clusters = load_pickle(Path(clusters_path))

    window = ClusterViewer(cells, clusters)
    window.show()
    sys.exit(app.exec_())
