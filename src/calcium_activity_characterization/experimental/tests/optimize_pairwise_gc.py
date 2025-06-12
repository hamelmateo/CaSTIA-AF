import sys
import numpy as np
import pickle
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QCheckBox, QFileDialog
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LogNorm

from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.processing.signal_processing import SignalProcessor
from calcium_activity_characterization.experimental.analysis.causality import GCAnalyzer
from calcium_activity_characterization.config.config import GC_PREPROCESSING

class GCInteractiveViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GC Interactive Viewer")
        self.resize(1400, 1000)

        self.cells = []
        self.selected_cells = []

        self._build_ui()

    def _build_ui(self):
        container = QWidget()
        layout = QVBoxLayout(container)

        # --- Top Controls ---
        controls = QHBoxLayout()

        self.cell_input = QLineEdit()
        self.cell_input.setPlaceholderText("Enter cell labels (e.g. 1,5,9)")
        controls.addWidget(QLabel("Cells:"))
        controls.addWidget(self.cell_input)

        self.lag_input = QLineEdit("5")
        controls.addWidget(QLabel("Lag:"))
        controls.addWidget(self.lag_input)

        self.start_input = QLineEdit("600")
        controls.addWidget(QLabel("Window Start:"))
        controls.addWidget(self.start_input)

        self.size_input = QLineEdit("400")
        controls.addWidget(QLabel("Window Size:"))
        controls.addWidget(self.size_input)

        self.pval_input = QLineEdit("0.001")
        controls.addWidget(QLabel("P-value Threshold:"))
        controls.addWidget(self.pval_input)

        self.thresh_check = QCheckBox("Threshold Links")
        self.thresh_check.setChecked(True)
        controls.addWidget(self.thresh_check)

        self.min_cells_input = QLineEdit("3")
        controls.addWidget(QLabel("Min Cells:"))
        controls.addWidget(self.min_cells_input)

        load_btn = QPushButton("Load Cells")
        load_btn.clicked.connect(self.load_cells)
        controls.addWidget(load_btn)

        run_btn = QPushButton("Run GC")
        run_btn.clicked.connect(self.run_gc)
        controls.addWidget(run_btn)

        layout.addLayout(controls)

        # --- Bottom Visualization ---
        viz = QHBoxLayout()

        self.trace_fig, self.trace_axs = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
        self.trace_canvas = FigureCanvas(self.trace_fig)
        viz.addWidget(self.trace_canvas, 3)

        self.gc_fig, self.gc_axs = plt.subplots(2, 1, figsize=(6, 10))
        self.gc_canvas = FigureCanvas(self.gc_fig)
        viz.addWidget(self.gc_canvas, 2)

        layout.addLayout(viz)
        self.setCentralWidget(container)

    def load_cells(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select raw_cells.pkl", "", "Pickle Files (*.pkl)")
        if path:
            with open(path, 'rb') as f:
                self.cells = pickle.load(f)
            print(f"Loaded {len(self.cells)} cells.")

    def run_gc(self):
        if not self.cells:
            return

        labels = [int(x.strip()) for x in self.cell_input.text().split(',') if x.strip().isdigit()]
        lag = int(self.lag_input.text())
        start = int(self.start_input.text())
        size = int(self.size_input.text())
        end = start + size
        p_thresh = float(self.pval_input.text())
        thresh_links = self.thresh_check.isChecked()
        min_cells = int(self.min_cells_input.text())

        self.selected_cells = [cell for cell in self.cells if cell.label in labels]
        if len(self.selected_cells) < min_cells:
            print("Too few valid cells selected.")
            return

        # GC preprocessing
        processor = SignalProcessor(GC_PREPROCESSING)
        for cell in self.selected_cells:
            cell.gc_trace = processor.run(cell.raw_intensity_trace)

        # Extract matrix
        traces = [cell.gc_trace for cell in self.selected_cells]
        T = len(traces[0])
        center_time = max(0, min(T - size, start + size // 2))
        window = slice(center_time - size // 2, center_time + size // 2)

        trace_matrix = np.array([t[window] for t in traces]).T  # shape (T, N)

        # GC analysis
        analyzer = GCAnalyzer({
            "mode": "pairwise",
            "parameters": {
                "pairwise": {
                    "window_size": size,
                    "lag_order": lag,
                    "min_cells": min_cells,
                    "pvalue_threshold": p_thresh,
                    "threshold_links": thresh_links,
                    "community_method": "louvain"
                }
            }
        }, DEVICES_CORES=1)

        gc_matrix = analyzer._run_pairwise_gc(trace_matrix)

        # --- Update Plots ---
        self.trace_fig.clf()
        axs = self.trace_fig.subplots(len(self.selected_cells), 1, sharex=True)
        for ax, cell in zip(axs, self.selected_cells):
            trace = cell.gc_trace
            ax.plot(trace, color='gray')
            ax.axvspan(start, end, color='blue', alpha=0.2)
            ax.set_ylabel(f"Cell {cell.label}")
            ax.grid(True)
        axs[-1].set_xlabel("Time")
        self.trace_canvas.draw()

        self.gc_fig.clf()
        ax1, ax2 = self.gc_fig.subplots(2, 1)

        # Prevent log(0) by setting a small epsilon
        eps = 1e-6
        masked_gc_matrix = np.clip(gc_matrix, eps, 1.0)

        # Plot with log color scale
        im = ax1.imshow(masked_gc_matrix, cmap='viridis', norm=LogNorm(vmin=eps, vmax=1.0))
        self.gc_fig.colorbar(im, ax=ax1)
        ax1.set_title("GC Matrix (1 - p-value, log scale)")

        G = nx.DiGraph()
        for i, src in enumerate(labels):
            for j, tgt in enumerate(labels):
                if i != j and gc_matrix[i, j] > 0:
                    G.add_edge(src, tgt, weight=gc_matrix[i, j])

        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', ax=ax2)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()}, ax=ax2)
        ax2.set_title("GC Graph")
        self.gc_canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = GCInteractiveViewer()
    viewer.show()
    sys.exit(app.exec_())
