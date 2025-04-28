import sys
import random
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QLineEdit, QFormLayout, QScrollArea
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from src.analysis.signal_processing import process_trace
from src.config.config import SIGNAL_PROCESSING_PARAMETERS
from src.config import config

class FineTuneSignalWindow(QMainWindow):
    def __init__(self, cells: list):
        super().__init__()
        self.setWindowTitle("Fine Tune Signal Processing")
        self.resize(1400, 800)

        self.all_cells = cells  # Save full list of loaded cells
        self.cells = cells[:5]  # Start by showing first 5 cells


        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout(self.central_widget)

        # Left panel: controls
        self.controls_layout = QVBoxLayout()

        self.method_label = QLabel("Detrending Method")
        self.method_combo = QComboBox()
        self.method_combo.addItems(SIGNAL_PROCESSING_PARAMETERS.keys())
        self.method_combo.currentTextChanged.connect(self.reset_parameter_fields)

        self.norm_label = QLabel("Normalization Method")
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["deltaf", "minmax", "none", "percentile", "zscore"])

        self.sigma_label = QLabel("Gaussian Sigma")
        self.sigma_input = QLineEdit("2.0")

        # Dynamic parameters area
        self.dynamic_form = QFormLayout()
        self.dynamic_fields = {}

        dynamic_container = QWidget()
        dynamic_container.setLayout(self.dynamic_form)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(dynamic_container)

        # Update button
        self.update_button = QPushButton("Update Plots")
        self.update_button.clicked.connect(self.update_plots)

        self.random_button = QPushButton("Select Random Cells")
        self.random_button.clicked.connect(self.select_random_cells)

        self.controls_layout.addWidget(self.random_button)


        self.controls_layout.addWidget(self.method_label)
        self.controls_layout.addWidget(self.method_combo)
        self.controls_layout.addWidget(self.norm_label)
        self.controls_layout.addWidget(self.norm_combo)
        self.controls_layout.addWidget(self.sigma_label)
        self.controls_layout.addWidget(self.sigma_input)
        self.controls_layout.addWidget(QLabel("Method Parameters:"))
        self.controls_layout.addWidget(scroll)
        self.controls_layout.addWidget(self.update_button)

        # Center panel: plot
        self.plot_layout = QVBoxLayout()
        self.figure, self.axs = plt.subplots(5, 1, figsize=(8, 12))
        self.canvas = FigureCanvas(self.figure)
        self.plot_layout.addWidget(self.canvas)

        # Assemble
        self.layout.addLayout(self.controls_layout, 1)
        self.layout.addLayout(self.plot_layout, 3)

        # Initial setup
        self.reset_parameter_fields(self.method_combo.currentText())
        self.update_plots()



    def select_random_cells(self):
        if len(self.all_cells) < 5:
            print("Not enough cells loaded to select random ones.")
            return

        self.cells = random.sample(self.all_cells, 5)
        self.update_plots()

    def reset_parameter_fields(self, method: str):
        # Clear previous dynamic fields
        for field in self.dynamic_fields.values():
            self.dynamic_form.removeRow(field)
        self.dynamic_fields = {}

        defaults = SIGNAL_PROCESSING_PARAMETERS[method]

        for key, value in defaults.items():
            if key in ["sigma", "normalize_method"]:
                continue  # Already handled globally
            input_field = QLineEdit(str(value))
            self.dynamic_form.addRow(QLabel(key), input_field)
            self.dynamic_fields[key] = input_field

        # Reset sigma and normalization
        self.sigma_input.setText(str(defaults.get("sigma", 2.0)))
        norm_method = defaults.get("normalize_method", "deltaf")
        idx = self.norm_combo.findText(norm_method)
        if idx != -1:
            self.norm_combo.setCurrentIndex(idx)

    def collect_parameters(self) -> dict:
        params = {}
        params["detrending_method"] = self.method_combo.currentText()
        params["sigma"] = float(self.sigma_input.text())
        params["normalize_method"] = self.norm_combo.currentText()
        for key, field in self.dynamic_fields.items():
            value = field.text()
            value = value.strip()
            if value.lower() == "none":
                params[key] = None
            else:
                try:
                    params[key] = int(value)
                except ValueError:
                    try:
                        params[key] = float(value)
                    except ValueError:
                        params[key] = value
        return params

    def update_plots(self):
        params = self.collect_parameters()

        # Recreate the figure and axes cleanly
        self.figure.clf()
        self.figure, self.axs = plt.subplots(5, 2, figsize=(25, 18))
        self.canvas.figure = self.figure  # reconnect canvas to the new figure
        self.figure.subplots_adjust(hspace=0.4, wspace=0.3)

        for idx, cell in enumerate(self.cells):
            raw = np.array(cell.raw_intensity_trace, dtype=float)
            processed = process_trace(raw, params)

            ax_raw = self.axs[idx][0]
            ax_proc = self.axs[idx][1]

            # Plot raw trace
            ax_raw.plot(raw, color="gray")
            ax_raw.set_title(f"Cell {cell.label} - Raw")
            ax_raw.set_xlabel("Timepoint")
            ax_raw.set_ylabel("Intensity")
            ax_raw.grid(True)

            # Plot processed trace
            ax_proc.plot(processed, color="blue")
            ax_proc.set_title(f"Cell {cell.label} - Processed")
            ax_proc.set_xlabel("Timepoint")
            ax_proc.set_ylabel("Intensity")
            ax_proc.grid(True)

        self.canvas.draw()


if __name__ == "__main__":
    from src.io.loader import load_cells_from_pickle
    from pathlib import Path

    app = QApplication(sys.argv)

    # Example load 5 cells manually
    from PyQt5.QtWidgets import QFileDialog

    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(None, 
                                               "Select processed_active_cells.pkl",
                                               "",
                                               "Pickle Files (*.pkl)")
    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(0)

    cells = load_cells_from_pickle(Path(file_path), load=True)

    window = FineTuneSignalWindow(cells)
    window.show()

    sys.exit(app.exec_())
