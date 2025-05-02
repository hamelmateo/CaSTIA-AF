from PyQt5.QtWidgets import (
    QMainWindow, QFileDialog, QVBoxLayout, QPushButton, QWidget, QHBoxLayout
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import pickle
from pathlib import Path
from typing import Optional

from calcium_activity_characterization.gui.overlay_viewer import OverlayViewer


class UMAPViewer(QMainWindow):
    def __init__(self, folder_path: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle("UMAP Viewer")
        self.setGeometry(100, 100, 1200, 800)

        self.embedding = None
        self.cells = None
        self.highlight_dot = None

        self.load_button = QPushButton("Load UMAP from Folder")
        self.load_button.clicked.connect(self.load_umap)

        # Canvas for UMAP and intensity trace plots
        self.umap_canvas = None
        self.raw_trace_canvas = None
        self.processed_trace_canvas = None

        layout = QVBoxLayout()
        layout.addWidget(self.load_button)

        split_layout = QHBoxLayout()
        self.umap_plot_container = QWidget()
        self.trace_plot_container = QWidget()
        self.umap_plot_layout = QVBoxLayout(self.umap_plot_container)
        self.trace_plot_layout = QVBoxLayout(self.trace_plot_container)

        split_layout.addWidget(self.umap_plot_container, 2)
        split_layout.addWidget(self.trace_plot_container, 2)
        layout.addLayout(split_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.overlay_viewer = None

        if folder_path:
            self.load_from_path(folder_path)

    def load_umap(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with UMAP")
        if not folder:
            return
        folder_path = Path(folder)
        self.load_from_path(folder_path)

    def load_from_path(self, folder_path: Path):
        self.folder_path = folder_path
        umap_file = folder_path / "umap.npy"
        cells_file = folder_path / "processed_active_cells.pkl"

        if not umap_file.exists() or not cells_file.exists():
            return

        self.embedding = np.load(umap_file)
        with open(cells_file, "rb") as f:
            self.cells = pickle.load(f)

        self.plot_umap()
        self.overlay_viewer = OverlayViewer(folder_path)
        self.overlay_viewer.set_callback(self.handle_overlay_click)
        self.overlay_viewer.show()

    def plot_umap(self):
        if self.umap_canvas:
            self.umap_plot_layout.removeWidget(self.umap_canvas)
            self.umap_canvas.setParent(None)

        fig, ax = plt.subplots(figsize=(6, 5))
        self.scatter = ax.scatter(self.embedding[:, 0], self.embedding[:, 1], s=10, c='blue', alpha=0.6, picker=True)
        ax.set_title("UMAP Projection")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

        self.umap_canvas = FigureCanvas(fig)
        self.umap_plot_layout.addWidget(self.umap_canvas)
        self.umap_canvas.mpl_connect('pick_event', self.on_pick)
        self.umap_canvas.draw()

    def on_pick(self, event):
        ind = event.ind[0]
        cell = self.cells[ind]
        self.plot_raw_and_processed(cell)
        self.highlight_umap_dot_by_label(cell.label)
        self.overlay_viewer.clear_highlight()
        self.overlay_viewer.highlight_cell(cell)

    def handle_overlay_click(self, cell):
        self.highlight_umap_dot_by_label(cell.label)
        self.plot_raw_and_processed(cell)

    def highlight_umap_dot_by_label(self, label: int):
        if not self.embedding.any():
            return

        if self.highlight_dot:
            self.highlight_dot.remove()
            self.highlight_dot = None

        try:
            idx = next(i for i, cell in enumerate(self.cells) if cell.label == label)
            x, y = self.embedding[idx]
            ax = self.umap_canvas.figure.axes[0]

            self.highlight_dot = ax.plot(x, y, marker='o', markersize=6, markeredgecolor='red',
                                         markerfacecolor='none', markeredgewidth=2)[0]
            self.umap_canvas.draw()
        except StopIteration:
            print(f"Cell label {label} not found in current embedding.")

    def plot_raw_and_processed(self, cell):
        if len(cell.raw_intensity_trace) == 0 or len(cell.processed_intensity_trace) == 0:
            return

        # Clear old canvases
        if self.raw_trace_canvas:
            self.trace_plot_layout.removeWidget(self.raw_trace_canvas)
            self.raw_trace_canvas.setParent(None)
        if self.processed_trace_canvas:
            self.trace_plot_layout.removeWidget(self.processed_trace_canvas)
            self.processed_trace_canvas.setParent(None)

        # Raw trace plot
        raw_fig, raw_ax = plt.subplots(figsize=(8, 2))
        raw_ax.plot(cell.raw_intensity_trace, label="Raw", color="gray", linestyle="--")
        raw_ax.set_title(f"Cell {cell.label} - Raw Intensity")
        raw_ax.set_xlabel("Timepoint")
        raw_ax.set_ylabel("Intensity")
        raw_ax.grid(True)
        self.raw_trace_canvas = FigureCanvas(raw_fig)
        self.trace_plot_layout.addWidget(self.raw_trace_canvas)
        self.raw_trace_canvas.draw()

        # Processed trace plot
        proc_fig, proc_ax = plt.subplots(figsize=(8, 2))
        proc_ax.plot(cell.processed_intensity_trace, label="Processed", color="blue")
        proc_ax.set_title(f"Cell {cell.label} - Processed Intensity")
        proc_ax.set_xlabel("Timepoint")
        proc_ax.set_ylabel("Intensity")
        proc_ax.grid(True)
        self.processed_trace_canvas = FigureCanvas(proc_fig)
        self.trace_plot_layout.addWidget(self.processed_trace_canvas)
        self.processed_trace_canvas.draw()
