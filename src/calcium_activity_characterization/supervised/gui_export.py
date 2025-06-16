# gui_export.py
# Usage Example:
# >>> gui = FinalExportGUI(pipeline)
# >>> gui.show()

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QLineEdit, QMessageBox
)
from PyQt5.QtCore import Qt
from pathlib import Path

from calcium_activity_characterization.core.pipeline import CalciumPipeline
import logging

logger = logging.getLogger(__name__)


class FinalExportGUI(QMainWindow):
    """
    GUI to export config and population metadata at the end of supervised pipeline.

    Args:
        pipeline (CalciumPipeline): Final pipeline instance.
    """

    def __init__(self, pipeline: CalciumPipeline):
        super().__init__()
        self.setWindowTitle("Final Export")
        self.resize(600, 200)
        self.pipeline = pipeline

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        layout.addWidget(QLabel("Select Output Directory:"))
        self.out_dir_input = QLineEdit(str(self.pipeline.output_dir))
        layout.addWidget(self.out_dir_input)

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_folder)
        layout.addWidget(self.browse_btn)

        self.export_btn = QPushButton("Export Now")
        self.export_btn.clicked.connect(self.run_export)
        layout.addWidget(self.export_btn)

        self.status_label = QLabel("Status: Waiting")
        layout.addWidget(self.status_label)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.out_dir_input.setText(folder)

    def run_export(self):
        output_dir = Path(self.out_dir_input.text())
        if not output_dir.exists():
            QMessageBox.critical(self, "Error", "Invalid output directory.")
            return

        self.status_label.setText("Status: Exporting...")
        self.repaint()

        try:
            self.pipeline.output_dir = output_dir

            # Save pipeline configuration as JSON
            import json
            config_path = output_dir / "config_used.json"
            with open(config_path, 'w') as f:
                json.dump(self.pipeline.config, f, indent=4)
            logger.info(f"✅ Saved config to {config_path}")

            # Export normalized dataset traces, assuming this method is implemented
            if hasattr(self.pipeline, "_export_normalized_datasets"):
                self.pipeline._export_normalized_datasets()
                logger.info("✅ Normalized datasets exported.")
            else:
                logger.warning("⚠️ Method _export_normalized_datasets not found on pipeline.")

            self.status_label.setText("Status: Export completed ✓")
            QMessageBox.information(self, "Done", "Export completed successfully.")
            self.close()

        except Exception as e:
            self.status_label.setText("Status: Error")
            logger.error(f"Export failed: {e}")
            QMessageBox.critical(self, "Error", f"Export failed: {e}")
