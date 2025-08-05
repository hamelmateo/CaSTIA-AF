# optimize_signal_processing.py
# Usage example:
#     >>> python optimize_signal_processing.py
#     # GUI will prompt to load a pickle file and visualize raw/processed/bin traces

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
from calcium_activity_characterization.logger import logger

from calcium_activity_characterization.config.structures import (
    SignalProcessingConfig, DetrendingConfig, NormalizationConfig,
    DetrendingMethod, NormalizationMethod,
    LocalMinimaParams, MovingAverageParams, PolynomialParams,
    RobustPolyParams, ButterworthParams, FIRParams, SavgolParams,
    DeltaFParams, ZScoreParams, MinMaxParams, PercentileParams,
    DetrendingParams, NormalizationParams,
    SignalProcessingPipeline,
    PeakDetectionConfig, PeakDetectionMethod, SkimageParams,
    PeakGroupingParams
)
from calcium_activity_characterization.config.presets import STANDARD_ZSCORE_SIGNAL_PROCESSING, CELL_PEAK_DETECTION_CONFIG
from calcium_activity_characterization.preprocessing.signal_processing import SignalProcessor
from calcium_activity_characterization.io.images_loader import load_pickle_file




def _safe_parse(text):
    text = text.strip()
    if text.lower() == "none":
        return None
    try:
        return eval(text, {"__builtins__": {}})
    except Exception:
        return text


def build_detrending_params(method: str, values: dict) -> DetrendingParams:
    if method == "localminima":
        return LocalMinimaParams(**values)
    elif method == "movingaverage":
        return MovingAverageParams(**values)
    elif method == "polynomial":
        return PolynomialParams(**values)
    elif method == "robustpoly":
        return RobustPolyParams(**values)
    elif method == "butterworth":
        return ButterworthParams(**values)
    elif method == "fir":
        return FIRParams(**values)
    elif method == "savgol":
        return SavgolParams(**values)
    else:
        raise ValueError(f"Unsupported detrending method: {method}")


def build_normalization_params(method: str) -> NormalizationParams:
    if method == "zscore":
        return ZScoreParams()
    elif method == "deltaf":
        return DeltaFParams()
    elif method == "minmax":
        return MinMaxParams()
    elif method == "percentile":
        return PercentileParams()
    else:
        raise ValueError(f"Unsupported normalization method: {method}")


class SignalProcessingBinarizedGUI(QMainWindow):
    def __init__(self, cells):
        super().__init__()
        self.setWindowTitle("Signal Processing + Peaks + Binarized GUI")
        self.resize(1800, 900)
        self.cells = cells
        self.random_cells = random.sample(cells, min(5, len(cells)))
        self.selected_cells = []

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        control_layout = QVBoxLayout()

        self.detrend_checkbox = QCheckBox("Apply Detrending")
        control_layout.addWidget(self.detrend_checkbox)

        self.normalize_checkbox = QCheckBox("Apply Normalization")
        control_layout.addWidget(self.normalize_checkbox)

        self.smooth_checkbox = QCheckBox("Apply Smoothing")
        control_layout.addWidget(self.smooth_checkbox)

        control_layout.addWidget(QLabel("Detrending Method:"))
        self.detrending_combo = QComboBox()
        self.detrending_combo.addItems([m.value for m in DetrendingMethod])
        self.detrending_combo.currentTextChanged.connect(self.reset_parameter_fields)
        control_layout.addWidget(self.detrending_combo)

        control_layout.addWidget(QLabel("Normalization Method:"))
        self.norm_combo = QComboBox()
        self.norm_combo.addItems([m.value for m in NormalizationMethod])
        control_layout.addWidget(self.norm_combo)

        control_layout.addWidget(QLabel("Sigma (smoothing):"))
        self.sigma_input = QLineEdit("2.0")
        control_layout.addWidget(self.sigma_input)

        control_layout.addWidget(QLabel("cut_trace_num_points:"))
        self.cut_trace_num_points_input = QLineEdit("100")
        control_layout.addWidget(self.cut_trace_num_points_input)

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
        for key, val in vars(CELL_PEAK_DETECTION_CONFIG.params).items():
            field = QLineEdit(str(val))
            control_layout.addWidget(QLabel(key))
            control_layout.addWidget(field)
            self.peak_params[key] = field


        self.refresh_button = QPushButton("Randomize Cells")
        self.refresh_button.clicked.connect(self.refresh_cells)
        control_layout.addWidget(self.refresh_button)

        self.selection_input = QLineEdit()
        self.selection_input.setPlaceholderText("Enter cell labels (e.g., 12, 25, 100)")
        control_layout.addWidget(self.selection_input)

        self.select_button = QPushButton("Load Selected Cells")
        self.select_button.clicked.connect(self.load_selected_cells)
        control_layout.addWidget(self.select_button)

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

        self.figure, self.axs = plt.subplots(5, 3, figsize=(16, 10))
        self.canvas = FigureCanvas(self.figure)

        main_layout.addLayout(control_layout, 1)
        main_layout.addWidget(self.canvas, 5)

        self.reset_parameter_fields(self.detrending_combo.currentText())
        self.update_plots()

    def reset_parameter_fields(self, method: str):
        while self.dynamic_form.rowCount() > 0:
            self.dynamic_form.removeRow(0)
        self.dynamic_fields = {}
        default_class = build_detrending_params(method, {})
        for field_name, default_val in default_class.__dict__.items():
            field = QLineEdit(str(default_val))
            self.dynamic_form.addRow(QLabel(field_name), field)
            self.dynamic_fields[field_name] = field

    def get_processor(self) -> SignalProcessor:
        try:
            detrend_method = self.detrending_combo.currentText()
            norm_method = self.norm_combo.currentText()
            detrend_values = {k: _safe_parse(f.text()) for k, f in self.dynamic_fields.items()}

            config = SignalProcessingConfig(
                pipeline=SignalProcessingPipeline(
                    detrending = self.detrend_checkbox.isChecked(),
                    normalization = self.normalize_checkbox.isChecked(),
                    smoothing = self.smooth_checkbox.isChecked()
                ),
                smoothing_sigma=float(self.sigma_input.text()),
                normalization=NormalizationConfig(
                    method=NormalizationMethod(norm_method),
                    params=build_normalization_params(norm_method)
                ),
                detrending=DetrendingConfig(
                    method=DetrendingMethod(detrend_method),
                    params=build_detrending_params(detrend_method, detrend_values)
                )
            )
            return SignalProcessor(config)
        except Exception as e:
            logger.error(f"Failed to construct SignalProcessor: {e}")
            raise

    def get_peak_params(self) -> PeakDetectionConfig:
        """
        Construct a PeakDetectionConfig from GUI inputs.

        Returns:
            PeakDetectionConfig: Configuration for peak detection.
        """
        try:
            param_dict = {
                key: _safe_parse(field.text())
                for key, field in self.peak_params.items()
            }

            return PeakDetectionConfig(
                method=PeakDetectionMethod.SKIMAGE,
                params=SkimageParams(**param_dict),
                peak_grouping=PeakGroupingParams(),  # or customize if needed
                start_frame=None,
                end_frame=None,
                filter_overlapping_peaks=False,
            )

        except Exception as e:
            logger.error(f"Failed to construct PeakDetectionConfig: {e}")
            raise

    def refresh_cells(self):
        self.selected_cells = []
        self.random_cells = random.sample(self.cells, min(5, len(self.cells)))
        self.update_plots()

    def load_selected_cells(self):
        try:
            label_ids = [int(val.strip()) for val in self.selection_input.text().split(",") if val.strip().isdigit()]
            label_set = set(label_ids)
            self.selected_cells = [cell for cell in self.cells if cell.label in label_set]
        except Exception as e:
            logger.error(f"Invalid input: {e}")
            self.selected_cells = []
        self.update_plots()

    def update_plots(self):
        cells_to_plot = self.selected_cells if self.selected_cells else self.random_cells

        self.figure.clf()
        self.axs = self.figure.subplots(len(cells_to_plot), 3, squeeze=False)
        self.peak_text.clear()
        colors = plt.cm.tab10.colors

        processor = self.get_processor()
        peak_params = self.get_peak_params()

        for i, cell in enumerate(cells_to_plot):
            raw = np.array(cell.trace.versions["raw"], dtype=float)
            processed = processor.run(raw)
            cell.trace.versions["processed"] = processed.tolist()
            cell.trace.default_version = "processed"
            cell.trace.detect_peaks(peak_params)
            cell.trace.binarize_trace_from_peaks()

            ax_raw, ax_proc, ax_bin = self.axs[i]
            ax_raw.plot(raw, color='black')
            ax_proc.plot(processed, color='blue')
            for peak in cell.trace.peaks:
                ax_proc.plot(peak.peak_time, peak.height, 'r*', markersize=8)
                ax_proc.axvspan(peak.activation_start_time, peak.activation_end_time,
                                color=colors[peak.id % len(colors)], alpha=0.3)
            ax_bin.plot(cell.trace.binary, color='green')

            ax_raw.set_title(f"Cell {cell.label} - Raw")
            ax_proc.set_title("Processed + Peaks")
            ax_bin.set_title("Binarized (from peaks)")
            for ax in (ax_raw, ax_proc, ax_bin):
                ax.set_xlabel("Time")
                ax.grid(True)

            lines = [
                f"Cell {cell.label} - Peak {p.id}: t={p.peak_time}, prom={p.prominence:.2f}, dur={p.fhw_duration:.2f}, role={p.grouping_type}"
                for p in cell.trace.peaks
            ]
            self.peak_text.append("\n".join(lines) + "\n")

        self.figure.tight_layout()
        self.canvas.draw()

    def save_figure(self):
        output_dir = Path("tests/signal_peaks_test/figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"result_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        self.figure.savefig(str(path))
        logger.info(f"Saved plot to {path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(None, "Select 01_raw_traces.pkl", "", "Pickle Files (*.pkl)")
    if not file_path:
        logger.warning("No file selected. Exiting.")
        sys.exit(0)

    population = load_pickle_file(Path(file_path))
    window = SignalProcessingBinarizedGUI(population.cells)
    window.show()
    sys.exit(app.exec_())
