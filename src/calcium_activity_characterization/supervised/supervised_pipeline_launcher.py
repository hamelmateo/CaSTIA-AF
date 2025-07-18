# supervised_pipeline_launcher.py
# Usage Example:
# >>> python supervised_pipeline_launcher.py
# GUI will walk through segmentation, signal processing, event detection, and export
# TODO finish the implementation of the new config format

import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

from calcium_activity_characterization.config.presets import GLOBAL_CONFIG

from calcium_activity_characterization.core.pipeline import CalciumPipeline
from calcium_activity_characterization.supervised.gui_segmentation import SegmentationTunerGUI
from calcium_activity_characterization.supervised.gui_signal_processing import SignalProcessingAndPeaksGUI
from calcium_activity_characterization.supervised.gui_events import ActivityAndEventDetectionGUI
from calcium_activity_characterization.supervised.gui_export import FinalExportGUI

import logging

logger = logging.getLogger(__name__)


class SupervisedPipelineLauncher:
    """
    Manages the full supervised pipeline workflow.
    """

    def __init__(self):
        self.config = GLOBAL_CONFIG
        self.pipeline: CalciumPipeline = None
        self.data_path: Path = None
        self.output_path: Path = None

    def start(self):
        self.select_isx_folder()
        if self.data_path is None:
            logger.info("No folder selected. Aborting.")
            return

        self.ask_parameter_loading()
        self.pipeline = CalciumPipeline(self.config)
        self.pipeline._init_paths(self.data_path, self.output_path)
        self.launch_segmentation_gui()

    def select_isx_folder(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setDirectory(str(GLOBAL_CONFIG.debug.harddrive_path))
        dialog.setWindowTitle("Select an ISX folder (e.g., IS1)")
        if dialog.exec_():
            selected = dialog.selectedFiles()
            if selected:
                self.data_path = Path(selected[0])
                self.output_path = self.data_path.parents[1] / "Output" / self.data_path.name
                self.output_path.mkdir(parents=True, exist_ok=True)

    def ask_parameter_loading(self):
        param_path = self.output_path / "config_used.json"
        if param_path.exists():
            reply = QMessageBox.question(
                None,
                "Load Parameters",
                f"Use existing parameters from {param_path.name}?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                import json
                with open(param_path, 'r') as f:
                    self.config.update(json.load(f))
                logger.info("Loaded parameters from config_used.json")

    def launch_segmentation_gui(self):
        self.segmentation_gui = SegmentationTunerGUI(
            pipeline=self.pipeline,
            on_validate=self.handle_segmentation_validated
        )
        self.segmentation_gui.show()

    def handle_segmentation_validated(self, updated_config: dict):
        self.pipeline.config = updated_config
        self.pipeline._segment_cells()
        self.pipeline._compute_intensity()
        self.launch_signal_processing_gui()

    def launch_signal_processing_gui(self):
        self.signal_gui = SignalProcessingAndPeaksGUI(
            pipeline=self.pipeline,
            on_validate=self.handle_signal_processing_validated
        )
        self.signal_gui.show()

    def handle_signal_processing_validated(self, updated_config: dict):
        self.pipeline.config = updated_config
        self.pipeline._signal_processing_pipeline()
        self.pipeline._binarization_pipeline()
        self.launch_activity_event_gui()

    def launch_activity_event_gui(self):
        self.activity_gui = ActivityAndEventDetectionGUI(
            pipeline=self.pipeline,
            on_validate=self.handle_event_detection_validated
        )
        self.activity_gui.show()

    def handle_event_detection_validated(self, updated_config: dict):
        self.pipeline.config = updated_config
        self.pipeline._initialize_activity_trace()
        self.pipeline._detect_events()
        self.launch_export_gui()

    def launch_export_gui(self):
        self.export_gui = FinalExportGUI(self.pipeline)
        self.export_gui.show()



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    launcher = SupervisedPipelineLauncher()
    launcher.start()
    sys.exit(app.exec_())