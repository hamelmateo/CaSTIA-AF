import sys
import random
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QCheckBox, QLineEdit, QFileDialog, QFormLayout, QScrollArea, QTextEdit
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from calcium_activity_characterization.config.config import SIGNAL_PROCESSING_PARAMETERS, PEAK_DETECTION
from calcium_activity_characterization.processing.signal_processing import SignalProcessor
from calcium_activity_characterization.data.peaks import PeakDetector
from calcium_activity_characterization.utilities.loader import load_cells_from_pickle


def _safe_parse(text):
    text = text.strip()
    if text.lower() == "none":
        return None
    try:
        return eval(text, {"__builtins__": {}})
    except Exception:
        return text


class SignalProcessingTestGUI(QMainWindow):
    def __init__(self, cells):
        super().__init__()
        self.setWindowTitle("Signal Processing + Peak Detection GUI")
        self.resize(1600, 900)
        self.cells = cells
        self.random_cells = random.sample(cells, 5)

        # Layouts
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left: Controls
        control_layout = QVBoxLayout()

        self.detrend_checkbox = QCheckBox("Apply Detrending")
        self.detrend_checkbox.setChecked(True)
        control_layout.addWidget(self.detrend_checkbox)

        self.smooth_checkbox = QCheckBox("Apply Smoothing")
        self.smooth_checkbox.setChecked(True)
        control_layout.addWidget(self.smooth_checkbox)

        self.normalize_checkbox = QCheckBox("Apply Normalization")
        self.normalize_checkbox.setChecked(True)
        control_layout.addWidget(self.normalize_checkbox)

        control_layout.addWidget(QLabel("Detrending Method:"))
        self.detrending_combo = QComboBox()
        self.detrending_combo.addItems(SIGNAL_PROCESSING_PARAMETERS["methods"].keys())
        self.detrending_combo.currentTextChanged.connect(self.reset_parameter_fields)
        control_layout.addWidget(self.detrending_combo)

        control_layout.addWidget(QLabel("Normalization Method:"))
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["deltaf", "zscore", "minmax", "percentile", "none"])
        control_layout.addWidget(self.norm_combo)

        control_layout.addWidget(QLabel("Sigma (smoothing):"))
        self.sigma_input = QLineEdit("2.0")
        control_layout.addWidget(self.sigma_input)

        self.dynamic_form = QFormLayout()
        self.dynamic_fields = {}

        dynamic_container = QWidget()
        dynamic_container.setLayout(self.dynamic_form)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(dynamic_container)
        control_layout.addWidget(QLabel("Detrending Parameters:"))
        control_layout.addWidget(scroll)

        self.peak_params = {}
        control_layout.addWidget(QLabel("Peak Detection Parameters:"))
        for key, val in PEAK_DETECTION["params"]["skimage"].items():
            field = QLineEdit(str(val))
            control_layout.addWidget(QLabel(key))
            control_layout.addWidget(field)
            self.peak_params[key] = field

        self.refresh_button = QPushButton("Randomize Cells")
        self.refresh_button.clicked.connect(self.refresh_cells)
        control_layout.addWidget(self.refresh_button)

        self.update_button = QPushButton("Update Plots")
        self.update_button.clicked.connect(self.update_plots)
        control_layout.addWidget(self.update_button)

        self.save_button = QPushButton("Save Figure")
        self.save_button.clicked.connect(self.save_figure)
        control_layout.addWidget(self.save_button)

        self.peak_text = QTextEdit()
        self.peak_text.setReadOnly(True)
        control_layout.addWidget(QLabel("Detected Peaks Summary:"))
        control_layout.addWidget(self.peak_text)

        # Right: Plot area
        self.figure, self.axs = plt.subplots(5, 2, figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)

        main_layout.addLayout(control_layout, 1)
        main_layout.addWidget(self.canvas, 4)

        self.reset_parameter_fields(self.detrending_combo.currentText())
        self.update_plots()

    def reset_parameter_fields(self, method: str):
        while self.dynamic_form.rowCount() > 0:
            self.dynamic_form.removeRow(0)

        self.dynamic_fields = {}
        defaults = SIGNAL_PROCESSING_PARAMETERS["methods"].get(method, {})
        for key, value in defaults.items():
            input_field = QLineEdit(str(value))
            self.dynamic_form.addRow(QLabel(key), input_field)
            self.dynamic_fields[key] = input_field

    def get_processor(self):
        pipeline = {
            "apply": {
                "detrending": self.detrend_checkbox.isChecked(),
                "smoothing": self.smooth_checkbox.isChecked(),
                "normalization": self.normalize_checkbox.isChecked()
            },
            "detrending_mode": self.detrending_combo.currentText(),
            "normalizing_method": self.norm_combo.currentText()
        }

        params = {
            "sigma": float(self.sigma_input.text()),
            "methods": {}
        }

        method_name = self.detrending_combo.currentText()
        method_params = {}

        for key, field in self.dynamic_fields.items():
            method_params[key] = _safe_parse(field.text())

        params["methods"][method_name] = method_params
        return SignalProcessor(params=params, pipeline=pipeline)

    def get_peak_params(self):
        parsed = {"method": "skimage", "params": {"skimage": {}}}
        for key, field in self.peak_params.items():
            parsed["params"]["skimage"][key] = _safe_parse(field.text())
        return parsed

    def refresh_cells(self):
        self.random_cells = random.sample(self.cells, 5)
        self.update_plots()

    def update_plots(self):
        self.figure.clf()
        self.axs = self.figure.subplots(5, 2, squeeze=False)
        self.peak_text.clear()

        colors = plt.cm.tab10.colors

        for i, cell in enumerate(self.random_cells):
            raw = np.array(cell.raw_intensity_trace, dtype=float)
            processor = self.get_processor()
            processed = processor.run(raw)
            cell.processed_intensity_trace = processed.tolist()

            detector = PeakDetector(self.get_peak_params())
            cell.detect_peaks(detector)

            ax_raw = self.axs[i][0]
            ax_proc = self.axs[i][1]
            ax_raw.cla()
            ax_proc.cla()

            ax_raw.plot(raw, color='black')
            ax_raw.set_title(f"Cell {cell.label} - Raw")
            ax_raw.set_xlabel("Time")
            ax_raw.set_ylabel("Intensity")
            ax_raw.grid(True)

            ax_proc.plot(processed, color='blue', label="Processed")
            for peak in cell.peaks:
                ax_proc.plot(peak.peak_time, peak.height, 'r*', markersize=8, label="Peak" if peak.id == 0 else "")
                ax_proc.axvspan(
                    peak.start_time,
                    peak.end_time,
                    color=colors[peak.id % len(colors)],
                    alpha=0.3,
                    label=f"Peak {peak.id}" if peak.id == 0 else None
                )

            ax_proc.set_title("Processed + Peaks")
            ax_proc.set_xlabel("Time")
            ax_proc.set_ylabel("Intensity")
            ax_proc.grid(True)

            peak_lines = [
                f"Cell {cell.label} - Peak {p.id}: t={p.peak_time}, rise={p.rise_time}, prom={p.prominence:.2f}, width={p.width:.2f}, height={p.height:.2f}, class={p.scale_class}"
                for p in cell.peaks
            ]
            self.peak_text.append("\n".join(peak_lines) + "\n")

        handles, labels = self.axs[0][1].get_legend_handles_labels()
        if handles:
            self.axs[0][1].legend(loc='upper right')

        self.figure.tight_layout()
        self.canvas.draw()

    def save_figure(self):
        output_dir = Path("testing/SignalProcessingTest")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"result_plot_{timestamp}.png"
        self.figure.savefig(str(path))
        print(f"Saved plot to {path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(None, "Select raw_active_cells.pkl", "", "Pickle Files (*.pkl)")
    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(0)

    cells = load_cells_from_pickle(Path(file_path), load=True)
    window = SignalProcessingTestGUI(cells)
    window.show()
    sys.exit(app.exec_())
