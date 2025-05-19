import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QWidget, QFormLayout, QLineEdit, QCheckBox, QComboBox, QTextEdit, QScrollArea, QSlider
)
from PyQt5.QtCore import Qt, QTimer
from calcium_activity_characterization.config.config import TRACKING_PARAMETERS
from calcium_activity_characterization.utilities.loader import load_pickle_file, load_existing_img
from arcos4py.tools import track_events_dataframe
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class OptimizeTrackedEventsGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TrackEvent Optimizer GUI")
        self.resize(1600, 900)

        self.df_input = None
        self.events_df = None
        self.overlay_img = None
        self.current_frame = 0
        self.total_frames = 0

        # Layouts
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left: Controls
        control_panel = QVBoxLayout()
        file_button = QPushButton("Load arcos_input_df.pkl")
        file_button.clicked.connect(self.load_arcos_input)
        control_panel.addWidget(file_button)

        overlay_button = QPushButton("Load overlay.tif")
        overlay_button.clicked.connect(self.load_overlay_image)
        control_panel.addWidget(overlay_button)

        self.param_form = QFormLayout()
        self.param_fields = {}
        self._create_param_fields()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        param_widget = QWidget()
        param_widget.setLayout(self.param_form)
        scroll.setWidget(param_widget)
        control_panel.addWidget(QLabel("Tracking Parameters"))
        control_panel.addWidget(scroll)

        self.run_button = QPushButton("Run Event Tracking")
        self.run_button.clicked.connect(self.run_event_tracking)
        control_panel.addWidget(self.run_button)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        control_panel.addWidget(QLabel("Log"))
        control_panel.addWidget(self.log_box)

        main_layout.addLayout(control_panel, 2)

        # Right: Viewer
        viewer_panel = QVBoxLayout()
        self.canvas = FigureCanvas(plt.Figure(figsize=(10, 10)))
        self.ax = self.canvas.figure.subplots()
        viewer_panel.addWidget(self.canvas)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.update_overlay_plot)
        viewer_panel.addWidget(self.slider)

        main_layout.addLayout(viewer_panel, 6)

    def _create_param_fields(self):
        for key, val in TRACKING_PARAMETERS.items():
            if isinstance(val, bool):
                checkbox = QCheckBox()
                checkbox.setChecked(val)
                self.param_form.addRow(QLabel(key), checkbox)
                self.param_fields[key] = checkbox
            elif isinstance(val, str):
                box = QComboBox()
                if key == "clustering_method":
                    box.addItems(["dbscan", "hdbscan"])
                elif key == "linking_method":
                    box.addItems(["nearest", "transportation"])
                else:
                    box.addItem(val)
                box.setCurrentText(val)
                self.param_form.addRow(QLabel(key), box)
                self.param_fields[key] = box
            else:
                line = QLineEdit(str(val))
                self.param_form.addRow(QLabel(key), line)
                self.param_fields[key] = line

    def get_tracking_params(self):
        params = {}
        for key, widget in self.param_fields.items():
            if isinstance(widget, QLineEdit):
                try:
                    params[key] = eval(widget.text(), {"__builtins__": {}})
                except Exception:
                    params[key] = widget.text()
            elif isinstance(widget, QCheckBox):
                params[key] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                params[key] = widget.currentText()
        return params

    def load_arcos_input(self):
        file_dialog = QFileDialog()
        path, _ = file_dialog.getOpenFileName(self, "Select arcos_input_df.pkl", "", "Pickle Files (*.pkl)")
        if path:
            self.df_input = load_pickle_file(Path(path))
            self.log_box.append(f"✅ Loaded: {path}")
            self.total_frames = int(self.df_input['frame'].max())
            self.slider.setMaximum(self.total_frames)

    def load_overlay_image(self):
        file_dialog = QFileDialog()
        path, _ = file_dialog.getOpenFileName(self, "Select overlay.tif", "", "TIF Files (*.tif)")
        if path:
            self.overlay_img = load_existing_img(Path(path))
            self.log_box.append(f"✅ Loaded overlay image: {path}")

    def run_event_tracking(self):
        if self.df_input is None:
            self.log_box.append("❌ Load a file first.")
            return

        self.log_box.append("▶️ Starting event tracking...")
        tracking_params = self.get_tracking_params()
        self.log_box.append(f"Using parameters: {tracking_params}")

        def run():
            start_time = time.time()
            result = track_events_dataframe(self.df_input.copy(), **tracking_params)

            # Handle both single or dual return formats
            if isinstance(result, tuple):
                self.events_df = result[0]
                self.log_box.append("ℹ️ Detected tuple output from ARCOS.")
            else:
                self.events_df = result
                self.log_box.append("ℹ️ Detected single DataFrame output from ARCOS.")
            elapsed = time.time() - start_time
            self.log_box.append(f"✅ Tracking complete in {elapsed:.2f} seconds.")
            self.slider.setMaximum(self.total_frames)
            self.update_overlay_plot()

        QTimer.singleShot(100, run)

    def update_overlay_plot(self):
        frame = self.slider.value()
        self.current_frame = frame

        self.ax.clear()
        if self.overlay_img is not None:
            self.ax.imshow(self.overlay_img, cmap='gray', alpha=0.5)

        df = pd.DataFrame()
        if self.events_df is not None and 'frame' in self.events_df.columns:
            df = self.events_df[self.events_df['frame'] == frame]

        if not df.empty and 'event_id' in df.columns:
            colors = df['event_id'].astype('category').cat.codes
            self.ax.scatter(df['x'], df['y'], c=colors, cmap='tab10', s=60, alpha=0.7, edgecolors='black')
            self.ax.set_title("")
        else:
            self.ax.set_title("")

        self.ax.axis('off')
        self.canvas.draw()

        total_events = self.events_df['event_id'].nunique() if self.events_df is not None and 'event_id' in self.events_df.columns else 0
        frame_events = df['event_id'].nunique() if 'event_id' in df.columns else 0
        self.log_box.append(f"🟢 Frame {frame} | Events: {frame_events} / Total: {total_events}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OptimizeTrackedEventsGUI()
    window.show()
    sys.exit(app.exec_())
