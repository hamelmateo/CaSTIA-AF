# gui_activity_and_events.py
# Usage Example:
# >>> gui = ActivityAndEventDetectionGUI(pipeline, on_validate)
# >>> gui.show()

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFormLayout, QLineEdit, QMessageBox, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import logging
from dataclasses import asdict, fields, replace
from copy import deepcopy

from calcium_activity_characterization.core.pipeline import CalciumPipeline
from calcium_activity_characterization.config.structures import EventExtractionConfig, ConvexHullParams

logger = logging.getLogger(__name__)

class ActivityAndEventDetectionGUI(QMainWindow):
    """
    GUI for tuning event detection parameters and viewing result.

    Args:
        pipeline (CalciumPipeline): CalciumPipeline instance
        on_validate (Callable): callback when parameters are validated
    """

    def __init__(self, pipeline: CalciumPipeline, on_validate):
        super().__init__()
        self.setWindowTitle("Event Detection Parameters")
        self.pipeline = pipeline
        self.on_validate = on_validate
        self.fields = {}
        self.status_label = QLabel("Status: Idle")

        self.init_ui()
        self.load_params()
        self.update_visualization()

    def init_ui(self):
        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # LEFT: plot area
        self.canvas = FigureCanvas(plt.figure(figsize=(10, 6)))
        layout.addWidget(self.canvas, 2)

        # RIGHT: scrollable parameter panel
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll, 1)

        container = QWidget()
        self.form_layout = QFormLayout(container)
        self.scroll.setWidget(container)

        self.update_button = QPushButton("Update")
        self.update_button.clicked.connect(self.apply_parameters)
        self.validate_button = QPushButton("Validate Parameters")
        self.validate_button.clicked.connect(self.validate_parameters)

        self.form_layout.addRow(self.update_button)
        self.form_layout.addRow(self.validate_button)
        self.form_layout.addRow(self.status_label)

    def load_params(self):
        config: EventExtractionConfig = self.pipeline.config.event_extraction
        flat_config = asdict(config)
        flat_config.update({f"convex_hull.{k}": v for k, v in asdict(config.convex_hull).items()})

        for key, value in flat_config.items():
            if key.startswith("convex_hull."):
                continue
            field = QLineEdit(str(value))
            self.fields[key] = field
            self.form_layout.addRow(QLabel(key), field)

        for key, value in asdict(config.convex_hull).items():
            field = QLineEdit(str(value))
            self.fields[f"convex_hull.{key}"] = field
            self.form_layout.addRow(QLabel(f"convex_hull.{key}"), field)

    def apply_parameters(self):
        self.status_label.setText("Status: Computing...")
        self.repaint()

        try:
            updated_config = self.build_updated_config()
            self.pipeline.config.event_extraction = updated_config
            self.pipeline._initialize_activity_trace()
            self.pipeline._detect_events()
            self.update_visualization()
            self.status_label.setText("Status: Updated ✓")
        except Exception as e:
            logger.error(f"Event detection failed: {e}")
            QMessageBox.critical(self, "Error", f"Failed to detect events: {e}")
            self.status_label.setText("Status: Error")

    def validate_parameters(self):
        try:
            updated_config = self.build_updated_config()
            self.pipeline.config.event_extraction = updated_config
            self.pipeline._initialize_activity_trace()
            self.pipeline._detect_events()
            self.on_validate(self.pipeline.config)
            self.close()
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            QMessageBox.critical(self, "Error", f"Validation failed: {e}")

    def build_updated_config(self) -> EventExtractionConfig:
        kwargs = {}
        hull_kwargs = {}
        for key, field in self.fields.items():
            value = self.safe_eval(field.text())
            if key.startswith("convex_hull."):
                name = key.split("convex_hull.")[1]
                hull_kwargs[name] = value
            else:
                kwargs[key] = value
        kwargs["convex_hull"] = ConvexHullParams(**hull_kwargs)
        return EventExtractionConfig(**kwargs)

    def update_visualization(self):
        self.canvas.figure.clf()
        activity_trace = getattr(self.pipeline.population, "activity_trace", None)
        if not activity_trace:
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
        for peak in getattr(activity_trace, "peaks", []):
            ax4.axvspan(peak.start_time, peak.end_time, color='orange', alpha=0.3)
        ax4.set_xlim([0, len(smoothed)])
        ax4.set_title("Detected Global Events")

        self.canvas.draw()

    @staticmethod
    def safe_eval(value: str):
        try:
            return eval(value, {"__builtins__": {}})
        except Exception:
            return value
