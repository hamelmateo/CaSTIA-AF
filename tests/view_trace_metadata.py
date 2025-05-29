"""
Metadata Viewer GUI

Usage:
    Run this script and select a folder (e.g., Output/IS1).
    It will load:
        - segmentation_overlay.png (segmentation map image)
        - active_cells.pkl (list of Cell objects with trace + metadata)
    Clicking on a cell will display its metadata (scalar + plots).
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QFileDialog, QPushButton, QMessageBox, QSplitter
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from calcium_activity_characterization.data.cells import Cell

class MetadataViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cell Metadata Viewer")

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignTop)
        self.image_label.mousePressEvent = self.handle_click

        self.canvas = FigureCanvas(Figure(figsize=(6, 8)))
        self.axs = self.canvas.figure.subplots(3, 1)

        self.load_button = QPushButton("Select Output Folder")
        self.load_button.clicked.connect(self.load_data)

        layout = QHBoxLayout()
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.image_label)
        splitter.addWidget(self.canvas)
        splitter.setSizes([600, 600])

        wrapper = QVBoxLayout()
        wrapper.addWidget(self.load_button)
        wrapper.addWidget(splitter)
        self.setLayout(wrapper)

        self.overlay = None
        self.cells = []
        self.centroids = []

    def load_data(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder:
            return

        folder = Path(folder)
        overlay_path = folder / "overlay.TIF"
        pkl_path = folder / "binarized_active_cells.pkl"

        if not overlay_path.exists() or not pkl_path.exists():
            QMessageBox.critical(self, "Error", "Missing overlay or cell data.")
            return

        self.overlay = QPixmap(str(overlay_path))
        self.image_label.setPixmap(self.overlay)

        with open(pkl_path, "rb") as f:
            self.cells = pickle.load(f)

        self.centroids = [tuple(map(int, cell.centroid)) for cell in self.cells]
        self.canvas.figure.suptitle("Click on a cell to view metadata")
        self.canvas.draw()

    def handle_click(self, event):
        if not self.centroids:
            return

        click_pos = event.pos()
        min_dist = float("inf")
        selected_idx = None

        for i, (cx, cy) in enumerate(self.centroids):
            dist = (click_pos.x() - cx) ** 2 + (click_pos.y() - cy) ** 2
            if dist < min_dist:
                min_dist = dist
                selected_idx = i

        if selected_idx is not None:
            cell = self.cells[selected_idx]
            trace = cell.trace
            self.plot_metadata(trace, selected_idx)

    def plot_metadata(self, trace, idx):
        self.canvas.figure.clf()
        axs = self.canvas.figure.subplots(3, 1)

        metadata = trace.metadata
        if not metadata:
            axs[0].text(0.5, 0.5, f"Cell {idx}: No metadata", ha='center')
            self.canvas.draw()
            return

        # Plot 1: Scalar table
        scalar_keys = [
            "fraction_active_time", "burst_frequency", "mean_peak_duration",
            "mean_inter_peak_interval", "mean_peak_amplitude",
            "mean_peak_prominence", "periodicity_score"
        ]
        rows = [(k, f"{metadata[k]:.3f}" if metadata[k] is not None else "None") for k in scalar_keys if k in metadata]
        axs[0].axis('off')
        table = axs[0].table(cellText=rows, colLabels=["Metric", "Value"], loc="center")
        table.scale(1, 1.5)
        axs[0].set_title(f"Cell {idx} Scalar Metadata")

        # Plot 2: Amplitude + duration histograms
        axs[1].set_title("Histograms")
        if "histogram_peak_amplitude" in metadata:
            d = metadata["histogram_peak_amplitude"]
            axs[1].bar(d["bins"][:-1], d["counts"], width=np.diff(d["bins"]), alpha=0.5, label="Amplitude")
        if "histogram_peak_duration" in metadata:
            d = metadata["histogram_peak_duration"]
            axs[1].bar(d["bins"][:-1], d["counts"], width=np.diff(d["bins"]), alpha=0.5, label="Duration")
        axs[1].legend()
        axs[1].set_xlabel("Value")
        axs[1].set_ylabel("Count")

        # Plot 3: Time dynamics
        axs[2].set_title("Dynamics")
        if "burst_frequency_evolution" in metadata:
            axs[2].plot(metadata["burst_frequency_evolution"], label="Burst Freq")
        if "activity_fraction_evolution" in metadata:
            axs[2].plot(metadata["activity_fraction_evolution"], label="Active Frac")
        axs[2].legend()
        axs[2].set_xlabel("Window")
        axs[2].set_ylabel("Value")

        self.canvas.figure.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = MetadataViewer()
    viewer.resize(1200, 800)
    viewer.show()
    sys.exit(app.exec_())
