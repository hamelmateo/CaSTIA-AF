# gui_segmentation.py
# Usage Example:
# >>> gui = SegmentationTunerGUI(pipeline, on_validate)
# >>> gui.show()

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMainWindow,
    QFormLayout, QLineEdit, QMessageBox, QScrollArea, QComboBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from dataclasses import asdict, fields
from calcium_activity_characterization.logger import logger

from calcium_activity_characterization.core.pipeline import CalciumPipeline
from calcium_activity_characterization.config.structures import (
    SegmentationConfig,
    SegmentationMethod,
    MesmerParams
)



# Map enum values to parameter dataclass types
PARAM_CLASSES = {
    SegmentationMethod.MESMER: MesmerParams,
    # Add other methods here as needed
}

class SegmentationTunerGUI(QMainWindow):
    """
    GUI for tuning segmentation parameters and viewing the result.

    Args:
        pipeline (CalciumPipeline): CalciumPipeline instance
        on_validate (Callable): callback when parameters are validated
    """

    def __init__(self, pipeline: CalciumPipeline, on_validate):
        super().__init__()
        self.setWindowTitle("Segmentation Optimizer")
        self.pipeline = pipeline
        self.on_validate = on_validate

        self.method_dropdown = None
        self.param_fields = {}
        self.status_label = QLabel("Status: Idle")

        self.init_ui()
        self.load_params()
        self.update_image()

    def init_ui(self):
        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # Left: image preview
        self.image_label = QLabel("Image Preview")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label, 2)

        # Right: scrollable parameter panel
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll, 1)

        container = QWidget()
        self.form_layout = QFormLayout(container)
        self.scroll.setWidget(container)

        # Method dropdown
        self.method_dropdown = QComboBox()
        for method in SegmentationMethod:
            self.method_dropdown.addItem(method.name, method)
        self.method_dropdown.currentIndexChanged.connect(self.on_method_change)
        self.form_layout.addRow(QLabel("Segmentation Method"), self.method_dropdown)

        # Add buttons
        self.update_button = QPushButton("Update Segmentation")
        self.update_button.clicked.connect(self.apply_parameters)
        self.validate_button = QPushButton("Validate Parameters")
        self.validate_button.clicked.connect(self.validate_parameters)

        self.form_layout.addRow(self.update_button)
        self.form_layout.addRow(self.validate_button)
        self.form_layout.addRow(self.status_label)

    def load_params(self):
        """Initializes form fields from current config."""
        config: SegmentationConfig = self.pipeline.config.segmentation

        self.method_dropdown.setCurrentText(config.method.name)
        self.refresh_param_fields(config)

    def refresh_param_fields(self, config: SegmentationConfig):
        """Clears and repopulates parameter fields for current method."""
        for key in list(self.param_fields):
            field_widget = self.param_fields.pop(key)
            self.form_layout.removeRow(field_widget)

        param_cls = PARAM_CLASSES.get(config.method)
        if param_cls is None:
            return

        self.param_dataclass_type: type = param_cls

        # Add save_overlay
        self.save_overlay_field = QLineEdit(str(config.save_overlay))
        self.form_layout.addRow(QLabel("save_overlay"), self.save_overlay_field)

        # Add param fields
        param_values = asdict(config.params)
        for k, v in param_values.items():
            field = QLineEdit(str(v))
            self.param_fields[k] = field
            self.form_layout.addRow(QLabel(k), field)

    def on_method_change(self):
        """Handles switching segmentation method."""
        try:
            method = self.method_dropdown.currentData()
            default_params = PARAM_CLASSES[method]()
            config = SegmentationConfig(
                method=method,
                params=default_params,
                save_overlay=True
            )
            self.refresh_param_fields(config)
        except Exception as e:
            logger.error(f"Failed to switch method: {e}")
            QMessageBox.critical(self, "Error", f"Failed to switch method: {e}")

    def apply_parameters(self):
        """Updates segmentation config and reruns segmentation."""
        self.status_label.setText("Status: Computing...")
        self.repaint()

        try:
            updated_config = self.build_updated_config()
            self.pipeline.config.segmentation = updated_config
            self.pipeline._segment_cells()
            self.update_image()
            self.status_label.setText("Status: Updated ✓")
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            QMessageBox.critical(self, "Error", f"Failed to segment: {e}")
            self.status_label.setText("Status: Error")

    def validate_parameters(self):
        """Applies parameters and calls on_validate callback."""
        try:
            updated_config = self.build_updated_config()
            self.pipeline.config.segmentation = updated_config
            self.pipeline._segment_cells()
            self.pipeline._compute_intensity()
            self.on_validate(self.pipeline.config)
            self.close()
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            QMessageBox.critical(self, "Error", f"Validation failed: {e}")

    def build_updated_config(self) -> SegmentationConfig:
        """Builds SegmentationConfig from form fields."""
        method = self.method_dropdown.currentData()
        param_cls = PARAM_CLASSES[method]
        param_kwargs = {}

        for field in fields(param_cls):
            raw = self.param_fields[field.name].text()
            value = self.safe_eval(raw)
            param_kwargs[field.name] = value

        save_overlay = self.safe_eval(self.save_overlay_field.text())
        return SegmentationConfig(
            method=method,
            params=param_cls(**param_kwargs),
            save_overlay=save_overlay
        )

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

    @staticmethod
    def safe_eval(value: str):
        try:
            return eval(value, {"__builtins__": {}})
        except Exception:
            return value  # fallback to string
