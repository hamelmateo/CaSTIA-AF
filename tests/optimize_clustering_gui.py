# optimize_clustering_gui.py

import sys
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QFileDialog, QFormLayout, QScrollArea, QLineEdit, QTextEdit
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import tifffile
import pickle
from collections import Counter

from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.processing.clustering import ClusteringEngine
from calcium_activity_characterization.config.config import CLUSTERING_PARAMETERS

class ClusteringGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clustering Optimization GUI")
        self.resize(1800, 1000)

        self.similarity_matrix = None
        self.cells = None
        self.overlay = None
        self.labels = None

        self.method_params = CLUSTERING_PARAMETERS["params"]

        self._init_ui()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        left_panel = QVBoxLayout()

        self.load_folder_btn = QPushButton("Load Analysis Folder")
        self.load_folder_btn.clicked.connect(self.load_analysis_folder)
        left_panel.addWidget(self.load_folder_btn)

        self.method_combo = QComboBox()
        self.method_combo.addItems(list(self.method_params.keys()))
        self.method_combo.currentTextChanged.connect(self.update_param_fields)
        left_panel.addWidget(QLabel("Clustering Method"))
        left_panel.addWidget(self.method_combo)

        self.param_form = QFormLayout()
        self.param_fields = {}
        self.param_widget = QWidget()
        self.param_widget.setLayout(self.param_form)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.param_widget)
        left_panel.addWidget(scroll, stretch=1)

        self.cluster_filter_input = QLineEdit()
        self.cluster_filter_input.setPlaceholderText("e.g. 0,1,4")
        left_panel.addWidget(QLabel("Show Clusters (comma-separated):"))
        left_panel.addWidget(self.cluster_filter_input)

        self.run_btn = QPushButton("Run Clustering")
        self.run_btn.clicked.connect(self.run_clustering)
        left_panel.addWidget(self.run_btn)

        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self.export_results)
        left_panel.addWidget(self.export_btn)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        left_panel.addWidget(self.log_box, stretch=1)

        self.raster_fig, self.raster_ax = plt.subplots(figsize=(6, 4))
        self.raster_canvas = FigureCanvas(self.raster_fig)
        left_panel.addWidget(self.raster_canvas, stretch=3)

        self.overlay_fig, self.overlay_ax = plt.subplots(figsize=(8, 8))
        self.overlay_canvas = FigureCanvas(self.overlay_fig)

        main_layout.addLayout(left_panel, 1)
        main_layout.addWidget(self.overlay_canvas, 1)
        self.update_param_fields(self.method_combo.currentText())

    def update_param_fields(self, method):
        while self.param_form.rowCount():
            self.param_form.removeRow(0)
        self.param_fields.clear()

        for key, val in self.method_params[method].items():
            label = "similarity_threshold" if method == "graph_community" and key == "threshold" else key
            field = QLineEdit(str(val))
            self.param_form.addRow(QLabel(label), field)
            self.param_fields[label] = field

    def get_current_config(self):
        method = self.method_combo.currentText()
        parsed_params = {
            k: self._parse_param_field(v.text())
            for k, v in self.param_fields.items()
        }
        return {
            "method": method,
            "params": { method: parsed_params }
        }

    def _parse_param_field(self, text):
        try:
            return eval(text.strip(), {"__builtins__": {}})
        except:
            return text.strip()

    def load_analysis_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            folder = Path(folder)
            sim_path = folder / "similarity_matrices.pkl"
            cells_path = folder / "binarized_active_cells.pkl"
            overlay_path = folder / "overlay.tif"

            try:
                with open(sim_path, 'rb') as f:
                    loaded = pickle.load(f)
                    self.similarity_matrix = np.array(loaded[0]) if isinstance(loaded, list) and loaded[0].ndim == 2 else np.array(loaded)
                self.log_box.append(f"✅ Loaded similarity matrix: {sim_path}")
            except Exception as e:
                self.log_box.append(f"❌ Failed to load similarity matrix: {e}")

            try:
                with open(cells_path, 'rb') as f:
                    self.cells = pickle.load(f)
                self.log_box.append(f"✅ Loaded cells: {cells_path}")
            except Exception as e:
                self.log_box.append(f"❌ Failed to load cells: {e}")

            try:
                self.overlay = tifffile.imread(str(overlay_path))
                self.log_box.append(f"✅ Loaded overlay: {overlay_path}")
            except Exception as e:
                self.log_box.append(f"❌ Failed to load overlay: {e}")

    def run_clustering(self):
        if self.similarity_matrix is None or self.cells is None:
            self.log_box.append("❌ Please load similarity matrix and cells.")
            return

        try:
            config = self.get_current_config()
            method = config["method"]
            params = config["params"][method]

            matrix = self.similarity_matrix.copy()

            if method == "affinity_propagation" and "preference" in params:
                try:
                    preference = float(params["preference"])
                    np.fill_diagonal(matrix, preference)
                    self.log_box.append(f"ℹ️ Set affinity_propagation diagonal preference to {preference:.3f}")
                except Exception as e:
                    self.log_box.append(f"⚠️ Failed to apply preference: {e}")

            config["params"][method] = params
            engine = ClusteringEngine(config)
            self.labels = engine.run([matrix])[0]

            counts = Counter(self.labels)
            for label, count in counts.items():
                if label != -1 and count <= 3:
                    self.labels[self.labels == label] = -1
            self.log_box.append("⚠️ Removed clusters with 3 or fewer cells.")

            n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
            self.log_box.append(f"🔢 Number of clusters: {n_clusters}")
            self.log_box.append(f"✅ Clustering with {method} completed.")
            self.update_raster_plot()
            self.update_overlay_plot()

        except Exception as e:
            self.log_box.append(f"❌ Error during clustering: {e}")

    def update_raster_plot(self):
        self.raster_ax.clear()
        if self.labels is None or self.cells is None:
            return

        filter_text = self.cluster_filter_input.text().strip()
        selected = set()
        if filter_text:
            try:
                selected = set(int(x.strip()) for x in filter_text.split(",") if x.strip().isdigit())
            except Exception as e:
                self.log_box.append(f"⚠️ Invalid filter input: {e}")

        label_order = [label for label, _ in sorted(zip(self.labels, self.cells), key=lambda x: x[0])]
        sorted_cells = [cell for _, cell in sorted(zip(self.labels, self.cells), key=lambda x: x[0])]
        matrix = np.array([cell.binary_trace for cell in sorted_cells])

        n_clusters = len(set(label_order)) - (1 if -1 in label_order else 0)
        cluster_colors = get_cmap('gist_ncar', n_clusters if n_clusters > 0 else 1)
        color_matrix = np.ones((len(matrix), len(matrix[0]), 3), dtype=float)

        for i, row in enumerate(matrix):
            cluster_label = label_order[i]
            if selected and cluster_label not in selected:
                color = (1, 1, 1)
            else:
                color = cluster_colors(cluster_label % cluster_colors.N)[:3]
            for j, val in enumerate(row):
                color_matrix[i, j] = color if val == 1 else (1, 1, 1)

        self.raster_ax.imshow(color_matrix, aspect='auto')
        self.raster_ax.set_title("Binarized Activity (sorted by cluster)")
        self.raster_canvas.draw()

    def update_overlay_plot(self):
        self.overlay_ax.clear()
        if self.labels is None or self.cells is None or self.overlay is None:
            return

        filter_text = self.cluster_filter_input.text().strip()
        selected_clusters = set()
        if filter_text:
            try:
                selected_clusters = set(int(x.strip()) for x in filter_text.split(",") if x.strip().isdigit())
            except Exception as e:
                self.log_box.append(f"⚠️ Invalid filter input: {e}")

        color_img = np.stack([self.overlay]*3, axis=-1).astype(np.uint8)
        unique_labels = sorted(set(self.labels))
        n_clusters = len([l for l in unique_labels if l != -1])
        cmap = plt.get_cmap("gist_ncar", n_clusters if n_clusters > 0 else 1)
        filtered_colors = [np.array(cmap(i)[:3]) for i in range(cmap.N)
                           if not all(v > 0.85 for v in cmap(i)[:3])]

        label_to_color = {}
        for i, label in enumerate([l for l in unique_labels if l != -1]):
            color = (filtered_colors[i % len(filtered_colors)] * 255).astype(np.uint8)
            label_to_color[label] = color

        for cell, label in zip(self.cells, self.labels):
            if label == -1 or (selected_clusters and label not in selected_clusters):
                continue
            color = label_to_color.get(label, np.array([200, 200, 200], dtype=np.uint8))
            for y, x in cell.pixel_coords:
                color_img[y, x] = color
        self.overlay_ax.imshow(color_img)
        self.overlay_ax.set_title("Cluster Overlay")
        self.overlay_canvas.draw()

    def export_results(self):
        if self.labels is None:
            self.log_box.append("❌ No labels to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save labels", "", "NumPy Files (*.npy)")
        if path:
            np.save(path, self.labels)
            self.log_box.append(f"✅ Labels saved to: {path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ClusteringGUI()
    window.show()
    sys.exit(app.exec_())
