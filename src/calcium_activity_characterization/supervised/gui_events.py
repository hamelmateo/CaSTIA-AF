# gui_activity_and_events.py
# Usage Example:
# >>> gui = ActivityAndEventDetectionGUI(pipeline, on_validate)
# >>> gui.show()

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFormLayout, QLineEdit, QMessageBox, QSplitter
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import ast
import logging

from calcium_activity_characterization.core.pipeline import CalciumPipeline
from calcium_activity_characterization.utilities.loader import get_config_with_fallback

logger = logging.getLogger(__name__)


class ActivityAndEventDetectionGUI(QMainWindow):
    """
    GUI to tune parameters for global activity trace and event detection.

    Args:
        pipeline (CalciumPipeline): Pipeline instance with loaded traces.
        on_validate (Callable): Callback triggered when user validates parameters.
    """

    def __init__(self, pipeline: CalciumPipeline, on_validate):
        super().__init__()
        self.setWindowTitle("Activity Trace and Event Detection")
        self.pipeline = pipeline
        self.on_validate = on_validate
        self.param_fields = {}
        self.status_label = QLabel("Status: Idle")

        self.init_ui()
        self.load_params()

        try:
            self.pipeline._initialize_activity_trace()
            self.pipeline._detect_events()
            self.update_visualization()
        except Exception as e:
            logger.error(f"Initial activity/event loading failed: {e}")
            self.status_label.setText("Status: Failed init")

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        self.canvas = FigureCanvas(plt.figure(figsize=(10, 6)))
        splitter.addWidget(self.canvas)

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

        control_layout.addWidget(self.status_label)
        splitter.addWidget(control_widget)

    def load_params(self):
        for section in [
            "POPULATION_TRACES_SIGNAL_PROCESSING_PARAMETERS",
            "ACTIVITY_TRACE_PEAK_DETECTION_PARAMETERS",
            "EVENT_EXTRACTION_PARAMETERS"
        ]:
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

    def update_processing(self):
        self.status_label.setText("Status: Computing...")
        self.repaint()

        try:
            self.pipeline.config = self.get_updated_config()
            self.pipeline._initialize_activity_trace()
            self.pipeline._detect_events()
            self.update_visualization()
            self.status_label.setText("Status: Updated ✓")
        except Exception as e:
            logger.error(f"Failed to update: {e}")
            QMessageBox.critical(self, "Error", f"Failed to update: {e}")
            self.status_label.setText("Status: Error")

    def update_visualization(self):
        self.canvas.figure.clf()

        activity_trace = getattr(self.pipeline.population, "activity_trace", None)
        if not activity_trace or not hasattr(activity_trace, "versions"):
            self.status_label.setText("Status: No activity trace")
            return

        ax1 = self.canvas.figure.add_subplot(411)
        ax1.plot(activity_trace.versions.get("raw", []), color='gray')
        ax1.set_title("Raw Activity Trace")

        ax2 = self.canvas.figure.add_subplot(412)
        ax2.plot(activity_trace.versions.get("processed", []), color='blue')
        ax2.set_title("Smoothed Activity Trace")

        ax3 = self.canvas.figure.add_subplot(413)
        ax3.plot(activity_trace.binary, color='green')
        ax3.set_title("Binarized Activity Trace")

        ax4 = self.canvas.figure.add_subplot(414)
        smoothed = activity_trace.versions.get("processed", [])
        peaks = getattr(activity_trace, "peaks", []) or []
        for peak in peaks:
            ax4.axvspan(peak.start_time, peak.end_time, color='orange', alpha=0.3)
        ax4.set_xlim([0, len(smoothed)])
        ax4.set_title("Detected Global Events (Peak Duration)")

        self.canvas.draw()

    def validate_parameters(self):
        try:
            self.pipeline.config = self.get_updated_config()
            self.pipeline._initialize_activity_trace()
            self.pipeline._detect_events()
            self.on_validate(self.pipeline.config)
            self.close()
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            QMessageBox.critical(self, "Error", f"Validation failed: {e}")
