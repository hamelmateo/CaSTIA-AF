# optimize_clustering_gui.py

import sys
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QFileDialog, QFormLayout, QScrollArea, QLineEdit, QTextEdit
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import tifffile
import pickle
import hdbscan
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import (
    DBSCAN, AgglomerativeClustering,
    AffinityPropagation
)
from matplotlib.cm import get_cmap

class ClusteringGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clustering Optimization GUI")
        self.resize(1800, 1000)

        self.similarity_matrix = None
        self.cells = None
        self.overlay = None
        self.labels = None

        self.method_params = {
            "dbscan": {"eps": "0.03", "min_samples": "3"},
            "hdbscan": {
                "min_cluster_size": "3", "min_samples": "3",
                "clustering_method": "eom", "probability_threshold": "0.85",
                "cluster_selection_epsilon": "0.5"
            },
            "agglomerative": {
                "n_clusters": "None", "distance_threshold": "0.5",
                "linkage": "complete"
            },
            "affinity_propagation": {
                "preference": "None", "damping": "0.9"
            },
            "graph_community": {
                "similarity_threshold": "0.7"
            }
        }

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
            if method == "affinity_propagation" and key == "preference":
                # Preference dropdown with computed strategies
                self.pref_combo = QComboBox()
                self.pref_line = QLineEdit()
                self.pref_line.setText(str(val))

                self.pref_combo.addItems([
                    "Select strategy...",
                    "median",
                    "0.5 * median + max",
                    "mean",
                    "min",
                    "max",
                    "custom"
                ])
                self.pref_combo.currentTextChanged.connect(self.set_affinity_preference_value)

                container = QWidget()
                layout = QHBoxLayout(container)
                layout.addWidget(self.pref_combo)
                layout.addWidget(self.pref_line)
                layout.setContentsMargins(0, 0, 0, 0)
                self.param_form.addRow(QLabel("preference"), container)
                self.param_fields["preference"] = self.pref_line
            else:
                field = QLineEdit(str(val))
                self.param_form.addRow(QLabel(key), field)
                self.param_fields[key] = field

    def compute_affinity_preferences(self) -> dict:
        sim = self.similarity_matrix
        return {
            "median": np.median(sim),
            "0.5 * median + max": 0.5 * np.median(sim) + np.max(sim),
            "mean": np.mean(sim),
            "min": np.min(sim),
            "max": np.max(sim),
            "custom": None  # For manual entry
        }

    def set_affinity_preference_value(self, strategy: str):
        if self.similarity_matrix is None:
            self.log_box.append("⚠️ Load similarity matrix before setting preference.")
            return

        prefs = self.compute_affinity_preferences()
        if strategy in prefs and prefs[strategy] is not None:
            self.pref_line.setText(f"{prefs[strategy]:.6f}")
            self.log_box.append(f"ℹ️ Set preference using strategy '{strategy}': {prefs[strategy]:.6f}")


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

        method = self.method_combo.currentText()
        dist = 1.0 - self.similarity_matrix

        try:
            if method == "dbscan":
                eps = float(self.param_fields["eps"].text())
                min_samples = int(self.param_fields["min_samples"].text())
                model = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
                self.labels = model.fit_predict(dist)

            elif method == "hdbscan":
                min_cluster_size = int(self.param_fields["min_cluster_size"].text())
                min_samples = int(self.param_fields["min_samples"].text())
                clustering_method = self.param_fields["clustering_method"].text()
                prob_thresh = float(self.param_fields["probability_threshold"].text())
                cluster_eps = float(self.param_fields["cluster_selection_epsilon"].text())
                model = hdbscan.HDBSCAN(
                    metric='precomputed',
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method=clustering_method,
                    cluster_selection_epsilon=cluster_eps
                )
                self.labels = model.fit_predict(dist)
                self.labels[model.probabilities_ < prob_thresh] = -1

            elif method == "agglomerative":
                n_clusters_text = self.param_fields["n_clusters"].text().strip()
                dist_thresh_text = self.param_fields["distance_threshold"].text().strip()

                n_clusters = None if n_clusters_text.lower() in ["", "none"] else int(n_clusters_text)
                dist_thresh = None if dist_thresh_text.lower() in ["", "none"] else float(dist_thresh_text) if dist_thresh_text.replace('.', '', 1).isdigit() else None

                if (n_clusters is None) == (dist_thresh is None):
                    raise ValueError("Exactly one of n_clusters and distance_threshold must be set.")

                linkage = self.param_fields["linkage"].text()
                model = AgglomerativeClustering(
                    metric='precomputed', linkage=linkage,
                    distance_threshold=dist_thresh if dist_thresh is not None else None,
                    n_clusters=n_clusters
                )
                self.labels = model.fit_predict(dist)

            elif method == "affinity_propagation":
                preference_str = self.param_fields["preference"].text().strip()
                damping_val = self.param_fields["damping"].text().strip()

                try:
                    preference = float(preference_str)
                except ValueError:
                    self.log_box.append(f"⚠️ Invalid preference value: {preference_str}")
                    return

                try:
                    damping = float(damping_val)
                except ValueError:
                    self.log_box.append(f"⚠️ Invalid damping value: {damping_val}")
                    damping = 0.9

                sim_matrix = self.similarity_matrix.copy()
                np.fill_diagonal(sim_matrix, preference)
                model = AffinityPropagation(affinity='precomputed', damping=damping, random_state=42)
                self.labels = model.fit_predict(sim_matrix)

            elif method == "graph_community":
                threshold = float(self.param_fields["similarity_threshold"].text())
                G = nx.Graph()
                for i in range(len(self.similarity_matrix)):
                    for j in range(i+1, len(self.similarity_matrix)):
                        if self.similarity_matrix[i, j] >= threshold:
                            G.add_edge(i, j, weight=self.similarity_matrix[i, j])
                communities = list(greedy_modularity_communities(G))
                self.labels = np.full(len(self.similarity_matrix), -1)
                for cluster_id, group in enumerate(communities):
                    for idx in group:
                        self.labels[idx] = cluster_id


            from collections import Counter

            label_counts = Counter(self.labels)
            for label, count in label_counts.items():
                if label != -1 and count <= 3:
                    self.labels[self.labels == label] = -1

            self.log_box.append("⚠️ Removed clusters with 3 or fewer cells.")

            
            self.log_box.append(f"✅ Clustering with {method} completed.")
            self.update_raster_plot()
            self.update_overlay_plot()
            n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
            self.log_box.append(f"🔢 Number of clusters: {n_clusters}")


        except Exception as e:
            self.log_box.append(f"❌ Error during clustering: {e}")

    def get_param_value(self, key: str) -> str:
        if key == "preference" and hasattr(self, "pref_line") and self.pref_line is not None:
            return self.pref_line.text().strip()
        return self.param_fields[key].text().strip()


    def update_raster_plot(self):
        self.raster_ax.clear()
        self.raster_fig.clf()
        self.raster_ax = self.raster_fig.add_subplot(111)
        if self.labels is None or self.cells is None:
            return
        cells_sorted = [cell for _, cell in sorted(zip(self.labels, self.cells), key=lambda x: x[0])]
        matrix = np.array([cell.binary_trace for cell in cells_sorted])
        label_order = [label for label, _ in sorted(zip(self.labels, self.cells), key=lambda x: x[0])]
        selected = set()
        if self.cluster_filter_input.text().strip():
            try:
                selected = set(int(x.strip()) for x in self.cluster_filter_input.text().split(",") if x.strip().isdigit())
            except Exception as e:
                self.log_box.append(f"⚠️ Invalid filter input: {e}")
        n_clusters = len(set(label_order)) - (1 if -1 in label_order else 0)
        cluster_colors = get_cmap('gist_ncar', n_clusters if n_clusters > 0 else 1)
        color_matrix = np.zeros((len(matrix), len(matrix[0]), 3), dtype=float)
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
        color_img = np.stack([self.overlay]*3, axis=-1).astype(np.uint8)
        unique_labels = sorted(set(self.labels))
        n_clusters = len([l for l in unique_labels if l != -1])
        cmap = plt.get_cmap("gist_ncar", n_clusters if n_clusters > 0 else 1)
        filtered_colors = [np.array(cmap(i)[:3]) for i in range(cmap.N)
                        if not all(v > 0.85 for v in cmap(i)[:3])]  # remove near-white

        label_to_color = {}
        for i, label in enumerate([l for l in unique_labels if l != -1]):
            color = (filtered_colors[i % len(filtered_colors)] * 255).astype(np.uint8)
            label_to_color[label] = color
        selected = set()
        if self.cluster_filter_input.text().strip():
            try:
                selected = set(int(x.strip()) for x in self.cluster_filter_input.text().split(",") if x.strip().isdigit())
            except Exception as e:
                self.log_box.append(f"⚠️ Invalid filter input: {e}")

        for cell, label in zip(self.cells, self.labels):
            if label == -1 or (selected and label not in selected):
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
