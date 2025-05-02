import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QListWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QWidget
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import tifffile
import pickle
from pathlib import Path


class EventExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Event Frame Explorer")
        self.setGeometry(100, 100, 1200, 800)

        self.folder = None
        self.df_events = None
        self.overlay_img = None
        self.current_event_id = None
        self.current_frame_idx = 0
        self.event_frames = []

        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()

        # Event list
        self.event_list = QListWidget()
        self.event_list.itemClicked.connect(self.load_event)
        layout.addWidget(self.event_list, 2)

        # Control + Plot area
        right_layout = QVBoxLayout()

        self.canvas = FigureCanvas(plt.figure(figsize=(8, 8)))
        right_layout.addWidget(self.canvas)

        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous Frame")
        self.prev_btn.clicked.connect(self.prev_frame)
        self.next_btn = QPushButton("Next Frame")
        self.next_btn.clicked.connect(self.next_frame)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)

        right_layout.addLayout(nav_layout)
        layout.addLayout(right_layout, 5)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.load_folder()

    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder_path:
            sys.exit(0)

        self.folder = Path(folder_path)
        event_file = self.folder / "tracked_events.pkl"
        overlay_file = self.folder / "overlay.TIF"

        with open(event_file, "rb") as f:
            self.df_events = pickle.load(f)

        self.overlay_img = tifffile.imread(str(overlay_file))
        event_ids = sorted(self.df_events["event_id"].dropna().unique())

        for eid in event_ids:
            self.event_list.addItem(f"Event {int(eid)}")

    def load_event(self, item):
        label = item.text()
        self.current_event_id = int(label.split()[-1])
        df = self.df_events[self.df_events["event_id"] == self.current_event_id]
        self.event_frames = sorted(df["frame"].unique())
        self.current_frame_idx = 0
        self.plot_current_frame()

    def plot_current_frame(self):
        if self.current_event_id is None or not self.event_frames:
            return

        frame = self.event_frames[self.current_frame_idx]
        df = self.df_events[(self.df_events["event_id"] == self.current_event_id) &
                            (self.df_events["frame"] == frame)]

        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.imshow(self.overlay_img, cmap="gray")
        scatter = ax.scatter(df["x"], df["y"], c="red", s=50, edgecolors="white")
        ax.set_title(f"Event {self.current_event_id} - Frame {frame}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.invert_yaxis()
        ax.grid(False)
        self.canvas.draw()

    def prev_frame(self):
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.plot_current_frame()

    def next_frame(self):
        if self.current_frame_idx < len(self.event_frames) - 1:
            self.current_frame_idx += 1
            self.plot_current_frame()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EventExplorer()
    window.show()
    sys.exit(app.exec_())