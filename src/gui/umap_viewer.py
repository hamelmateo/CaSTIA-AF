from PyQt5.QtWidgets import (
    QMainWindow, QFileDialog, QApplication, QGraphicsScene, QGraphicsView, QVBoxLayout, QPushButton, QWidget, QLabel
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import pickle
import sys
from pathlib import Path
from typing import Optional

from src.gui.viewer import OverlayViewer
from src.gui.plotter import show_cell_plot

class UMAPViewer(QMainWindow):
    def __init__(self, folder_path: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle("UMAP Viewer")
        self.setGeometry(100, 100, 1000, 800)

        self.canvas = None
        self.embedding = None
        self.cells = None

        self.load_button = QPushButton("Load UMAP from Folder")
        self.load_button.clicked.connect(self.load_umap)

        self.info_label = QLabel("Click a point to view the cell label")

        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        layout.addWidget(self.info_label)

        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_widget)

        self.highlight_dot = None

        layout.addWidget(self.plot_widget)

        container = QWidget()
        container.setLayout(layout)
                # Overlay viewer instance (headless)
        self.overlay_viewer = None

        self.setCentralWidget(container)

        if folder_path:
            self.load_from_path(folder_path)

    def load_umap(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with UMAP")
        if not folder:
            return

        folder_path = Path(folder)
        self.load_from_path(folder_path)

    def load_from_path(self, folder_path: Path):
        self.folder_path = folder_path  # Save for overlay link
        umap_file = folder_path / "umap.npy"
        cells_file = folder_path / "active_cells.pkl"

        if not umap_file.exists() or not cells_file.exists():
            self.info_label.setText("Missing 'umap.npy' or 'active_cells.pkl' in folder")
            return

        self.embedding = np.load(umap_file)
        with open(cells_file, "rb") as f:
            self.cells = pickle.load(f)

        self.plot_umap()
        self.overlay_viewer = OverlayViewer(folder_path)
        self.overlay_viewer.set_callback(self.handle_overlay_click)
        self.overlay_viewer.show()



    def plot_umap(self):
        if self.canvas:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.setParent(None)

        fig, ax = plt.subplots(figsize=(8, 6))
        self.scatter = ax.scatter(self.embedding[:, 0], self.embedding[:, 1], s=10, c='blue', alpha=0.6, picker=True)
        ax.set_title("UMAP Projection")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

        self.canvas = FigureCanvas(fig)
        self.plot_layout.addWidget(self.canvas)
        self.canvas.mpl_connect('pick_event', self.on_pick)
        self.canvas.draw()

    def on_pick(self, event):
        ind = event.ind[0]
        cell = self.cells[ind]
        label = cell.label
        cy, cx = cell.centroid
        self.info_label.setText(f"Cell Label: {label}, Centroid: ({cx}, {cy})")

        # Plot intensity trace
        show_cell_plot(cell)

        self.highlight_umap_dot_by_label(label)

        # Launch or update overlay viewer
        #if self.overlay_viewer is None:
        #    self.overlay_viewer = OverlayViewer(self.folder_path)
        #    self.overlay_viewer.show()

        self.overlay_viewer.clear_highlight()
        self.overlay_viewer.highlight_cell(cell)


    def handle_overlay_click(self, cell):
        """
        Called when a cell is clicked in the overlay viewer.
        """
        label = cell.label
        self.highlight_umap_dot_by_label(label)
        show_cell_plot(cell)


    def highlight_umap_dot_by_label(self, label: int):
        """
        Highlight the UMAP dot corresponding to the given cell label.
        """
        if not self.embedding.any():
            return

        # Remove old highlight if it exists
        if self.highlight_dot:
            self.highlight_dot.remove()
            self.highlight_dot = None

        try:
            idx = next(i for i, cell in enumerate(self.cells) if cell.label == label)
            x, y = self.embedding[idx]
            ax = self.canvas.figure.axes[0]

            # Smaller red circle
            self.highlight_dot = ax.plot(x, y, marker='o', markersize=6, markeredgecolor='red',
                                        markerfacecolor='none', markeredgewidth=2)[0]
            self.canvas.draw()
        except StopIteration:
            print(f"Cell label {label} not found in current embedding.")
