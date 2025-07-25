# gui_signal_processing.py
# Usage Example:
# >>> gui = SignalProcessingAndPeaksGUI(pipeline, on_validate)
# >>> gui.show()

import random
import numpy as np
from dataclasses import asdict
from copy import deepcopy

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFormLayout, QLineEdit, QSplitter, QMessageBox, QComboBox, QCheckBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from calcium_activity_characterization.core.pipeline import CalciumPipeline
from calcium_activity_characterization.data.cells import Cell
from calcium_activity_characterization.config.structures import (
    NormalizationMethod, DetrendingMethod,
    SignalProcessingConfig, NormalizationConfig, DetrendingConfig,
    ZScoreParams, DeltaFParams, PercentileParams, MinMaxParams,
    LocalMinimaParams, PolynomialParams, MovingAverageParams, RobustPolyParams,
    SavgolParams, ButterworthParams, FIRParams, WaveletParams, DoubleCurveFittingParams
)

import logging
logger = logging.getLogger(__name__)

NORMALIZATION_PARAMS = {
    NormalizationMethod.ZSCORE: ZScoreParams,
    NormalizationMethod.MINMAX: MinMaxParams,
    NormalizationMethod.PERCENTILE: PercentileParams,
    NormalizationMethod.DELTAF: DeltaFParams
}

DETRENDING_PARAMS = {
    DetrendingMethod.LOCALMINIMA: LocalMinimaParams,
    DetrendingMethod.MOVINGAVERAGE: MovingAverageParams,
    DetrendingMethod.POLYNOMIAL: PolynomialParams,
    DetrendingMethod.ROBUSTPOLY: RobustPolyParams,
    DetrendingMethod.SAVGOL: SavgolParams,
    DetrendingMethod.BUTTERWORTH: ButterworthParams,
    DetrendingMethod.FIR: FIRParams,
    DetrendingMethod.WAVELET: WaveletParams,
    DetrendingMethod.DOUBLECURVE: DoubleCurveFittingParams
}

class SignalProcessingAndPeaksGUI(QMainWindow):
    def __init__(self, pipeline: CalciumPipeline, on_validate):
        super().__init__()
        self.setWindowTitle("Signal Processing and Peak Detection")
        self.pipeline = pipeline
        self.on_validate = on_validate

        self.selected_cells: list[Cell] = []
        self.status_label = QLabel("Status: Idle")

        self.param_fields = {}
        self.init_ui()
        self.load_params()
        self.select_random_cells()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setSizes([1000, 300])
        layout.addWidget(splitter)

        # Left: trace plot
        self.canvas = FigureCanvas(plt.figure(figsize=(10, 12)))
        splitter.addWidget(self.canvas)

        # Right: config controls
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        self.form_layout = QFormLayout()
        control_layout.addLayout(self.form_layout)

        self.detrend_checkbox = QCheckBox("Enable Detrending")
        self.norm_checkbox = QCheckBox("Enable Normalization")
        self.smooth_checkbox = QCheckBox("Enable Smoothing")
        self.form_layout.addRow(self.detrend_checkbox)
        self.form_layout.addRow(self.norm_checkbox)
        self.form_layout.addRow(self.smooth_checkbox)

        self.norm_dropdown = QComboBox()
        for method in NormalizationMethod:
            self.norm_dropdown.addItem(method.name, method)
        self.norm_dropdown.currentIndexChanged.connect(self.load_normalization_params)
        self.form_layout.addRow(QLabel("Normalization Method"), self.norm_dropdown)

        self.detrend_dropdown = QComboBox()
        for method in DetrendingMethod:
            self.detrend_dropdown.addItem(method.name, method)
        self.detrend_dropdown.currentIndexChanged.connect(self.load_detrending_params)
        self.form_layout.addRow(QLabel("Detrending Method"), self.detrend_dropdown)

        self.update_btn = QPushButton("Update")
        self.update_btn.clicked.connect(self.update_processing)
        control_layout.addWidget(self.update_btn)

        self.validate_btn = QPushButton("Validate Parameters")
        self.validate_btn.clicked.connect(self.validate_parameters)
        control_layout.addWidget(self.validate_btn)

        self.random_btn = QPushButton("Randomize Cells")
        self.random_btn.clicked.connect(self.select_random_cells)
        control_layout.addWidget(self.random_btn)

        control_layout.addWidget(self.status_label)
        splitter.addWidget(control_widget)

    def safe_eval(self, value: str):
        try:
            return eval(value, {"__builtins__": {}})
        except Exception:
            return value

    def load_params(self):
        config = self.pipeline.config.cell_trace_processing
        self.detrend_checkbox.setChecked(config.pipeline.detrending)
        self.norm_checkbox.setChecked(config.pipeline.normalization)
        self.smooth_checkbox.setChecked(config.pipeline.smoothing)

        self.norm_dropdown.setCurrentText(config.normalization.method.name)
        self.detrend_dropdown.setCurrentText(config.detrending.method.name)

        self.load_normalization_params()
        self.load_detrending_params()

    def load_normalization_params(self):
        for key in list(self.param_fields):
            if key.startswith("norm:"):
                widget = self.param_fields.pop(key)
                self.form_layout.removeRow(widget)

        method = self.norm_dropdown.currentData()
        cls = NORMALIZATION_PARAMS[method]
        params = asdict(cls())

        for k, v in params.items():
            field = QLineEdit(str(v))
            key = f"norm:{k}"
            self.param_fields[key] = field
            self.form_layout.addRow(QLabel(key), field)

    def load_detrending_params(self):
        for key in list(self.param_fields):
            if key.startswith("detrend:"):
                widget = self.param_fields.pop(key)
                self.form_layout.removeRow(widget)

        method = self.detrend_dropdown.currentData()
        cls = DETRENDING_PARAMS[method]
        params = asdict(cls())

        for k, v in params.items():
            field = QLineEdit(str(v))
            key = f"detrend:{k}"
            self.param_fields[key] = field
            self.form_layout.addRow(QLabel(key), field)

    def get_updated_config(self):
        signal_config = deepcopy(self.pipeline.config.cell_trace_processing)

        pipeline = signal_config.pipeline
        pipeline.detrending = self.detrend_checkbox.isChecked()
        pipeline.normalization = self.norm_checkbox.isChecked()
        pipeline.smoothing = self.smooth_checkbox.isChecked()

        norm_method = self.norm_dropdown.currentData()
        norm_cls = NORMALIZATION_PARAMS[norm_method]
        norm_kwargs = {
            k.split(":")[1]: self.safe_eval(field.text())
            for k, field in self.param_fields.items() if k.startswith("norm:")
        }
        normalization = NormalizationConfig(method=norm_method, params=norm_cls(**norm_kwargs))

        detrend_method = self.detrend_dropdown.currentData()
        detrend_cls = DETRENDING_PARAMS[detrend_method]
        detrend_kwargs = {
            k.split(":")[1]: self.safe_eval(field.text())
            for k, field in self.param_fields.items() if k.startswith("detrend:")
        }
        detrending = DetrendingConfig(method=detrend_method, params=detrend_cls(**detrend_kwargs))

        return SignalProcessingConfig(
            pipeline=pipeline,
            smoothing_sigma=signal_config.smoothing_sigma,
            normalization=normalization,
            detrending=detrending
        )

    def select_random_cells(self):
        try:
            if not self.pipeline.population or not self.pipeline.population.cells:
                raise ValueError("No cells available")
            self.selected_cells = random.sample(self.pipeline.population.cells, min(5, len(self.pipeline.population.cells)))
            self.update_processing()
        except Exception as e:
            logger.error(f"Failed to select random cells: {e}")
            QMessageBox.critical(self, "Error", f"Failed to select cells: {e}")

    def update_processing(self):
        self.status_label.setText("Status: Computing...")
        self.repaint()

        try:
            self.pipeline.config.cell_trace_processing = self.get_updated_config()
            self.pipeline._signal_processing_pipeline()
            self.pipeline._binarization_pipeline()
            self.plot_cells()
            self.status_label.setText("Status: Updated ✓")
        except Exception as e:
            logger.error(f"Update failed: {e}")
            self.status_label.setText("Status: Error")
            QMessageBox.critical(self, "Error", f"Failed to update: {e}")

    def plot_cells(self):
        self.canvas.figure.clf()
        axs = self.canvas.figure.subplots(len(self.selected_cells), 3, squeeze=False)

        for i, cell in enumerate(self.selected_cells):
            raw = cell.trace.versions.get("raw")
            smoothed = cell.trace.versions.get("processed")
            binary = cell.trace.binary
            peaks = cell.trace.peaks
            colors = plt.cm.tab10.colors
            
            ax1, ax2, ax3 = axs[i]
            if raw is not None:
                ax1.plot(raw, color='gray')
                ax1.set_title(f"Cell {cell.label} - Raw")

            if smoothed is not None:
                ax2.plot(smoothed, color='blue')
                if peaks:
                    for peak in cell.trace.peaks:
                        ax2.plot(peak.peak_time, peak.height, 'r*', markersize=8)
                        ax2.axvspan(peak.ref_start_time, peak.ref_end_time,
                                    color=colors[peak.id % len(colors)], alpha=0.3)
            ax2.set_title("Processed + Peaks")

            if binary is not None:
                ax3.plot(binary, color='green')
                ax3.set_title("Binarized")

            for ax in (ax1, ax2, ax3):
                ax.set_xlabel("Time")
                ax.grid(True)

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def validate_parameters(self):
        try:
            self.pipeline.config.cell_trace_processing = self.get_updated_config()
            self.pipeline._signal_processing_pipeline()
            self.pipeline._binarization_pipeline()
            self.on_validate(self.pipeline.config)
            self.close()
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            QMessageBox.critical(self, "Error", f"Validation failed: {e}")
