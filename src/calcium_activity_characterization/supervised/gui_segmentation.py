# gui_segmentation_tuner.py
# Usage Example:
# >>> gui = SegmentationTunerGUI(pipeline, on_validate)
# >>> gui.show()

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMainWindow,
    QFormLayout, QLineEdit, QMessageBox, QScrollArea
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from calcium_activity_characterization.core.pipeline import CalciumPipeline
from calcium_activity_characterization.utilities.loader import get_config_with_fallback

import logging

logger = logging.getLogger(__name__)


class SegmentationTunerGUI(QMainWindow):
    """
    GUI for tuning segmentation parameters and viewing result.

    Args:
        pipeline (CalciumPipeline): CalciumPipeline instance
        on_validate (Callable): callback when parameters are validated
    """

    def __init__(self, pipeline: CalciumPipeline, on_validate):
        super().__init__()
        self.setWindowTitle("Segmentation Optimizer")
        self.pipeline = pipeline
        self.on_validate = on_validate

        self.fields = {}
        self.status_label = QLabel("Status: Idle")

        self.init_ui()
        self.load_params()
        self.update_image()

    def init_ui(self):
        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # LEFT: Image display
        self.image_label = QLabel("Image Preview")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label, 2)

        # RIGHT: Control panel
        control_panel = QWidget()
        form_layout = QFormLayout(control_panel)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(control_panel)
        layout.addWidget(scroll, 1)

        form_layout.addRow(QLabel("Segmentation Parameters:"))
        self.form_layout = form_layout

        self.update_button = QPushButton("Update Segmentation")
        self.update_button.clicked.connect(self.apply_parameters)
        form_layout.addRow(self.update_button)

        self.validate_button = QPushButton("Validate Parameters")
        self.validate_button.clicked.connect(self.validate_parameters)
        form_layout.addRow(self.validate_button)

        form_layout.addRow(QLabel(""))
        form_layout.addRow(self.status_label)

    def load_params(self):
        params = get_config_with_fallback(self.pipeline.config, "SEGMENTATION_PARAMETERS")
        self.fields.clear()
        for key, value in params.items():
            field = QLineEdit(str(value))
            self.fields[key] = field
            self.form_layout.addRow(QLabel(key), field)

    def apply_parameters(self):
        self.status_label.setText("Status: Computing...")
        self.repaint()

        try:
            new_params = {key: self.safe_eval(field.text()) for key, field in self.fields.items()}

            self.pipeline.config["SEGMENTATION_PARAMETERS"] = new_params
            self.pipeline._segment_cells()
            self.update_image()
            self.status_label.setText("Status: Updated ✓")
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            QMessageBox.critical(self, "Error", f"Failed to segment: {e}")
            self.status_label.setText("Status: Error")

    @staticmethod
    def safe_eval(value: str):
        try:
            return eval(value, {"__builtins__": {}})
        except Exception:
            return value  # fallback to raw string


    def update_image(self):
        try:
            mask = self.pipeline.nuclei_mask
            if mask is not None:
                normalized = (mask / mask.max() * 255).astype(np.uint8)
                height, width = normalized.shape
                qimg = QImage(normalized.data, width, height, width, QImage.Format_Grayscale8)
                pixmap = QPixmap.fromImage(qimg)
                scaled = pixmap.scaled(
                    self.image_label.width(),
                    self.image_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled)
        except Exception as e:
            logger.warning(f"Failed to update image preview: {e}")

    def validate_parameters(self):
        try:
            new_params = {key: self.safe_eval(field.text()) for key, field in self.fields.items()}
            self.pipeline.config["SEGMENTATION_PARAMETERS"] = new_params
            self.pipeline._segment_cells()
            self.pipeline._compute_intensity()
            self.on_validate(self.pipeline.config)
            self.close()
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            QMessageBox.critical(self, "Error", f"Validation failed: {e}")
