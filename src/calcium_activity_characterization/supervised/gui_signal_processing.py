# gui_signal_processing.py
# Usage Example:
# >>> gui = SignalProcessingAndPeaksGUI(pipeline, on_validate)
# >>> gui.show()

import random
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFormLayout, QLineEdit, QSplitter, QMessageBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from calcium_activity_characterization.core.pipeline import CalciumPipeline
from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.utilities.loader import get_config_with_fallback
import ast
import logging

logger = logging.getLogger(__name__)


class SignalProcessingAndPeaksGUI(QMainWindow):
    """
    GUI to tune signal processing and peak detection parameters.

    Args:
        pipeline (CalciumPipeline): Pipeline instance with cells loaded.
        on_validate (Callable): Callback when parameters are validated.
    """

    def __init__(self, pipeline: CalciumPipeline, on_validate):
        super().__init__()
        self.setWindowTitle("Signal Processing and Peak Detection")
        self.pipeline = pipeline
        self.on_validate = on_validate

        self.selected_cell: Cell = None
        self.status_label = QLabel("Status: Idle")
        self.param_fields = {}

        self.init_ui()
        self.load_params()
        self.select_random_cell()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left side: trace plots
        self.canvas = FigureCanvas(plt.figure(figsize=(10, 6)))
        splitter.addWidget(self.canvas)

        # Right side: parameter controls
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        self.form_layout = QFormLayout()
        control_layout.addLayout(self.form_layout)

        self.update_btn = QPushButton("Update")
        self.update_btn.clicked.connect(self.update_processing)
        control_layout.addWidget(self.update_btn)

        self.validate_btn = QPushButton("Validate Parameters")
        self.validate_btn.clicked.connect(self.validate_parameters)
        control_layout.addWidget(self.validate_btn)

        self.random_btn = QPushButton("Random Cell")
        self.random_btn.clicked.connect(self.select_random_cell)
        control_layout.addWidget(self.random_btn)

        control_layout.addWidget(self.status_label)
        splitter.addWidget(control_widget)

    def load_params(self):
        for section in ["INDIV_SIGNAL_PROCESSING_PARAMETERS", "INDIV_PEAK_DETECTION_PARAMETERS"]:
            config = get_config_with_fallback(self.pipeline.config, section)
            for key, value in config.items():
                full_key = f"{section}:{key}"
                field = QLineEdit(str(value))
                self.param_fields[full_key] = field
                self.form_layout.addRow(QLabel(full_key), field)

    def safe_eval(self, value: str):
        try:
            return eval(value, {"__builtins__": {}})
        except Exception:
            return value

    def get_updated_config(self):
        config = self.pipeline.config.copy()
        for full_key, field in self.param_fields.items():
            section, key = full_key.split(":")
            val = self.safe_eval(field.text())
            config[section][key] = val
        return config

    def select_random_cell(self):
        try:
            if not self.pipeline.population or not self.pipeline.population.cells:
                raise ValueError("No cells available")
            self.selected_cell = random.choice(self.pipeline.population.cells)
            self.update_processing()
        except Exception as e:
            logger.error(f"Failed to select random cell: {e}")
            QMessageBox.critical(self, "Error", f"No cell could be selected: {e}")

    def update_processing(self):
        self.status_label.setText("Status: Computing...")
        self.repaint()

        try:
            self.pipeline.config = self.get_updated_config()
            self.pipeline._signal_processing_pipeline()
            self.pipeline._binarization_pipeline()
            self.plot_cell()
            self.status_label.setText("Status: Updated ✓")
        except Exception as e:
            logger.error(f"Update failed: {e}")
            self.status_label.setText("Status: Error")
            QMessageBox.critical(self, "Error", f"Failed to update: {e}")

    def plot_cell(self):
        self.canvas.figure.clf()
        ax1 = self.canvas.figure.add_subplot(311)
        ax2 = self.canvas.figure.add_subplot(312)
        ax3 = self.canvas.figure.add_subplot(313)

        raw = self.selected_cell.trace.versions.get("raw")
        smoothed = self.selected_cell.trace.versions.get("processed")
        binary = self.selected_cell.trace.binary
        peaks = self.selected_cell.trace.peaks

        if raw is not None:
            ax1.plot(raw, color='gray')
            ax1.set_title("Raw Trace")

        if smoothed is not None:
            ax2.plot(smoothed, color='blue')
            if peaks:
                peak_times = [p.peak_time for p in peaks]
                peak_heights = [smoothed[p.peak_time] for p in peaks]
                ax2.plot(peak_times, peak_heights, 'ro')
            ax2.set_title("Smoothed Trace with Peaks")

        if binary is not None:
            ax3.plot(binary, color='green')
            ax3.set_title("Binarized Trace")

        self.canvas.draw()

    def validate_parameters(self):
        try:
            self.pipeline.config = self.get_updated_config()
            self.pipeline._signal_processing_pipeline()
            self.pipeline._binarization_pipeline()
            self.on_validate(self.pipeline.config)
            self.close()
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            QMessageBox.critical(self, "Error", f"Validation failed: {e}")
