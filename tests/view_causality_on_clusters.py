"""
Viewer for Granger causality graphs on clusters of cells.
This script provides a GUI to visualize Granger causality graphs
"""

import sys
import pickle
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget, QGraphicsScene, QGraphicsView
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt
import tifffile
import numpy as np
import networkx as nx

from calcium_activity_characterization.data.cells import Cell

class GCGraphViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Granger Causality Graph Viewer")
        self.setGeometry(100, 100, 1400, 1000)

        self.current_index = 0
        self.graph_paths = []
        self.graphs = []
        self.cells = []
        self.overlay = None
        self.pvalue_threshold = 0.0001  # Can be made configurable

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)

        self.label = QLabel("Cluster: 0")
        self.next_button = QPushButton("Next Cluster")
        self.next_button.clicked.connect(self.next_cluster)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addWidget(self.label)
        layout.addWidget(self.next_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.load_folder()

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder:
            sys.exit(0)

        folder = Path(folder)
        self.overlay = tifffile.imread(str(folder / "overlay.tif"))

        with open(folder / "binarized_active_cells.pkl", "rb") as f:
            self.cells = pickle.load(f)

        graph_dir = folder / "gc_graphs"
        self.graph_paths = sorted(graph_dir.glob("gc_graph_cluster_*.gpickle"))

        if not self.graph_paths:
            self.label.setText("No GC graphs found.")
            return

        for path in self.graph_paths:
            with open(path, "rb") as f:
                self.graphs.append(pickle.load(f))

        self.render_graph(0)

    def next_cluster(self):
        self.current_index = (self.current_index + 1) % len(self.graphs)
        self.render_graph(self.current_index)

    def render_graph(self, index):
        self.scene.clear()
        base_img = np.stack([self.overlay] * 3, axis=-1).astype(np.uint8)

        graph = self.graphs[index]
        involved_labels = set(graph.nodes)
        label_to_cell = {cell.label: cell for cell in self.cells if cell.label in involved_labels}

        # Highlight cluster cells
        for label in involved_labels:
            cell = label_to_cell[label]
            for y, x in cell.pixel_coords:
                base_img[y, x] = [255, 0, 0]  # Red outline

        # Convert image for QPixmap
        h, w, _ = base_img.shape
        image = QImage(base_img.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        # Add to scene
        self.scene.addPixmap(pixmap)

        # Draw arrows with weight-dependent thickness
        for src, tgt, data in graph.edges(data=True):
            weight = data.get("weight", 0)
            if weight >= 1 - self.pvalue_threshold:
                src_pt = label_to_cell[src].centroid[::-1]  # x, y
                tgt_pt = label_to_cell[tgt].centroid[::-1]  # x, y

                pen = QPen(QColor(0, 255, 0), max(1.0, weight * 5))
                self.scene.addLine(src_pt[0], src_pt[1], tgt_pt[0], tgt_pt[1], pen)

        self.label.setText(f"Cluster: {index + 1} / {len(self.graphs)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = GCGraphViewer()
    viewer.show()
    sys.exit(app.exec_())