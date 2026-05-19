"""Standalone GUI for visualizing CPR compression depth from the CPR FIFO."""

import argparse
import os
import sys
import time
from collections import deque
from dataclasses import dataclass

try:
    from PyQt5.QtCore import QThread, QTimer, pyqtSignal
    from PyQt5.QtWidgets import QApplication, QFrame, QGridLayout, QLabel, QMainWindow, QWidget
    import pyqtgraph as pg
    GUI_IMPORT_ERROR = None
except ImportError as exc:
    GUI_IMPORT_ERROR = exc
    QApplication = None
    QFrame = object
    QGridLayout = None
    QLabel = object
    QMainWindow = object
    QThread = object
    QTimer = None
    QWidget = object
    pg = None

    class _MissingSignal:
        def connect(self, *_args, **_kwargs):
            pass

        def emit(self, *_args, **_kwargs):
            pass

    def pyqtSignal(*_args, **_kwargs):
        return _MissingSignal()

try:
    from .cpr_depth_reader import PIPE_CPR, setup_cpr_pipes, stat_is_fifo
except ImportError:
    from cpr_depth_reader import PIPE_CPR, setup_cpr_pipes, stat_is_fifo


NORMAL_DISTANCE_BASELINE_MM = 100.0
PLOT_WINDOW_SECONDS = 15.0
RATE_WINDOW_SECONDS = 10.0
MIN_COMPRESSION_DEPTH_MM = 10.0
MIN_PEAK_INTERVAL_SECONDS = 0.25


@dataclass(frozen=True)
class CPRDisplaySample:
    timestamp_s: float
    range_mm: float
    depth_mm: float
    sequence: int


def parse_pipe_line(line, baseline_mm):
    values = line.strip().split(",")
    if len(values) < 3:
        return None

    timestamp_ns = int(values[0])
    range_mm = float(values[1])
    sequence = int(values[2])

    if range_mm < 0:
        return CPRDisplaySample(timestamp_ns / 1e9, range_mm, 0.0, sequence)

    depth_mm = max(0.0, baseline_mm - range_mm)
    return CPRDisplaySample(timestamp_ns / 1e9, range_mm, depth_mm, sequence)


def moving_average(values, window_size=5):
    if len(values) < 3:
        return list(values)

    smoothed = []
    half_window = window_size // 2
    for index in range(len(values)):
        start = max(0, index - half_window)
        stop = min(len(values), index + half_window + 1)
        smoothed.append(sum(values[start:stop]) / (stop - start))
    return smoothed


def detect_compression_peaks(times, depths):
    if len(depths) < 3:
        return []

    smoothed_depths = moving_average(depths)
    peaks = []
    last_peak_time = None

    for index in range(1, len(smoothed_depths) - 1):
        depth = smoothed_depths[index]
        if depth < MIN_COMPRESSION_DEPTH_MM:
            continue
        if depth < smoothed_depths[index - 1] or depth < smoothed_depths[index + 1]:
            continue

        peak_time = times[index]
        if last_peak_time is not None and peak_time - last_peak_time < MIN_PEAK_INTERVAL_SECONDS:
            if peaks and depth > peaks[-1][1]:
                peaks[-1] = (peak_time, depth)
                last_peak_time = peak_time
            continue

        peaks.append((peak_time, depth))
        last_peak_time = peak_time

    return peaks


def estimate_cpr_rate_bpm(peaks):
    if len(peaks) < 2:
        return 0.0

    intervals = [
        peaks[index][0] - peaks[index - 1][0]
        for index in range(1, len(peaks))
        if peaks[index][0] > peaks[index - 1][0]
    ]
    if not intervals:
        return 0.0

    return 60.0 / (sum(intervals) / len(intervals))


class CPRPipeReaderThread(QThread):
    sample_received = pyqtSignal(object)
    status_changed = pyqtSignal(str)

    def __init__(self, pipe_path, baseline_mm, create_pipe=True):
        super().__init__()
        self.pipe_path = pipe_path
        self.baseline_mm = baseline_mm
        self.create_pipe = create_pipe
        self.is_running = True

    def stop(self):
        self.is_running = False
        self.quit()
        self.wait(1500)

    def run(self):
        if self.create_pipe:
            setup_cpr_pipes((self.pipe_path,))

        if os.path.exists(self.pipe_path) and not stat_is_fifo(self.pipe_path):
            self.status_changed.emit(f"{self.pipe_path} exists but is not a FIFO")
            return

        while self.is_running:
            pipe_file = None
            try:
                self.status_changed.emit(f"Waiting for CPR data on {self.pipe_path}")
                fd = os.open(self.pipe_path, os.O_RDONLY | os.O_NONBLOCK)
                pipe_file = os.fdopen(fd, "r", buffering=1)
                self.status_changed.emit(f"Connected to {self.pipe_path}")

                while self.is_running:
                    line = pipe_file.readline()
                    if not line:
                        time.sleep(0.02)
                        continue

                    try:
                        sample = parse_pipe_line(line, self.baseline_mm)
                    except ValueError:
                        continue

                    if sample is not None:
                        self.sample_received.emit(sample)

            except FileNotFoundError:
                time.sleep(0.25)
            except Exception as exc:
                self.status_changed.emit(f"CPR reader error: {exc}")
                time.sleep(0.5)
            finally:
                if pipe_file is not None:
                    pipe_file.close()


class MetricCard(QFrame):
    def __init__(self, title, unit):
        super().__init__()
        self.setObjectName("metricCard")

        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricTitle")
        self.value_label = QLabel("--")
        self.value_label.setObjectName("metricValue")
        self.unit_label = QLabel(unit)
        self.unit_label.setObjectName("metricUnit")

        layout = QGridLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(2)
        layout.addWidget(self.title_label, 0, 0)
        layout.addWidget(self.value_label, 1, 0)
        layout.addWidget(self.unit_label, 2, 0)

    def set_value(self, value):
        self.value_label.setText(value)


class CPRVisualizerWindow(QMainWindow):
    def __init__(self, pipe_path, baseline_mm, create_pipe=True):
        super().__init__()
        self.pipe_path = pipe_path
        self.baseline_mm = baseline_mm
        self.samples = deque()
        self.last_sample = None
        self.last_peaks = []

        self.setWindowTitle("CPR Depth Monitor")
        self.resize(1180, 680)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#101820")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.setLabel("left", "Compression depth", units="mm")
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setYRange(0, max(65, baseline_mm * 0.7), padding=0)
        self.depth_curve = self.plot_widget.plot(
            [],
            [],
            pen=pg.mkPen("#22d3ee", width=3),
            name="Depth",
        )
        self.peak_points = pg.ScatterPlotItem([], [], brush="#f97316", pen=None, size=9)
        self.plot_widget.addItem(self.peak_points)

        self.rate_card = MetricCard("CPR Rate", "compressions/min")
        self.depth_card = MetricCard("Compression Depth", "mm")
        self.range_card = MetricCard("Sensor Range", "mm")
        self.status_label = QLabel("Starting CPR visualizer")
        self.status_label.setObjectName("statusLabel")
        self.baseline_label = QLabel(f"Baseline: {baseline_mm:.1f} mm")
        self.baseline_label.setObjectName("baselineLabel")

        side_panel = QWidget()
        side_panel.setObjectName("sidePanel")
        side_layout = QGridLayout(side_panel)
        side_layout.setContentsMargins(18, 18, 18, 18)
        side_layout.setVerticalSpacing(14)
        side_layout.addWidget(self.rate_card, 0, 0)
        side_layout.addWidget(self.depth_card, 1, 0)
        side_layout.addWidget(self.range_card, 2, 0)
        side_layout.addWidget(self.baseline_label, 3, 0)
        side_layout.addWidget(self.status_label, 4, 0)
        side_layout.setRowStretch(5, 1)

        root = QWidget()
        root.setObjectName("root")
        layout = QGridLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setHorizontalSpacing(18)
        layout.addWidget(self.plot_widget, 0, 0)
        layout.addWidget(side_panel, 0, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnMinimumWidth(1, 285)
        self.setCentralWidget(root)
        self.apply_styles()

        self.reader_thread = CPRPipeReaderThread(pipe_path, baseline_mm, create_pipe=create_pipe)
        self.reader_thread.sample_received.connect(self.add_sample)
        self.reader_thread.status_changed.connect(self.status_label.setText)
        self.reader_thread.start()

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(50)

    def apply_styles(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget#root {
                background: #0b1117;
                color: #e5edf5;
                font-family: Arial, sans-serif;
            }
            QWidget#sidePanel {
                background: #121a22;
                border: 1px solid #263340;
                border-radius: 8px;
            }
            QFrame#metricCard {
                background: #17212b;
                border: 1px solid #2b3b4a;
                border-radius: 8px;
            }
            QLabel#metricTitle {
                color: #91a4b7;
                font-size: 13px;
                font-weight: 700;
            }
            QLabel#metricValue {
                color: #f8fafc;
                font-size: 40px;
                font-weight: 800;
            }
            QLabel#metricUnit {
                color: #7dd3fc;
                font-size: 13px;
            }
            QLabel#statusLabel, QLabel#baselineLabel {
                color: #b8c7d4;
                font-size: 13px;
            }
            """
        )

    def add_sample(self, sample):
        self.samples.append(sample)
        self.last_sample = sample

    def update_display(self):
        if not self.samples:
            return

        latest_time = self.samples[-1].timestamp_s
        cutoff_time = latest_time - PLOT_WINDOW_SECONDS
        while self.samples and self.samples[0].timestamp_s < cutoff_time:
            self.samples.popleft()

        times = [sample.timestamp_s - latest_time for sample in self.samples]
        depths = [sample.depth_mm for sample in self.samples]
        ranges = [sample.range_mm for sample in self.samples]

        self.depth_curve.setData(times, depths)
        self.plot_widget.setXRange(-PLOT_WINDOW_SECONDS, 0, padding=0)

        rate_cutoff = latest_time - RATE_WINDOW_SECONDS
        rate_times = [sample.timestamp_s for sample in self.samples if sample.timestamp_s >= rate_cutoff]
        rate_depths = [sample.depth_mm for sample in self.samples if sample.timestamp_s >= rate_cutoff]
        peaks = detect_compression_peaks(rate_times, rate_depths)
        self.last_peaks = peaks

        peak_x = [peak_time - latest_time for peak_time, _ in peaks]
        peak_y = [peak_depth for _, peak_depth in peaks]
        self.peak_points.setData(peak_x, peak_y)

        rate_bpm = estimate_cpr_rate_bpm(peaks)
        recent_depth = max([depth for depth in depths[-25:] if depth >= 0.0], default=0.0)
        latest_range = ranges[-1]

        self.rate_card.set_value(f"{rate_bpm:.0f}" if rate_bpm > 0 else "--")
        self.depth_card.set_value(f"{recent_depth:.1f}")
        self.range_card.set_value("--" if latest_range < 0 else f"{latest_range:.1f}")

    def closeEvent(self, event):
        self.reader_thread.stop()
        event.accept()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Visualize CPR depth samples from the CPR FIFO.")
    parser.add_argument("--pipe", default=PIPE_CPR, help=f"CPR FIFO path. Default: {PIPE_CPR}")
    parser.add_argument(
        "--baseline-mm",
        type=float,
        default=NORMAL_DISTANCE_BASELINE_MM,
        help="Resting sensor distance in mm. Depth is baseline_mm - range_mm.",
    )
    parser.add_argument("--no-create-pipe", action="store_true", help="Do not create the FIFO if missing.")
    return parser


def main(argv=None):
    if GUI_IMPORT_ERROR is not None:
        raise SystemExit(
            "cpr_data_visualizer.py requires PyQt5 and pyqtgraph. "
            "Install the demo GUI dependencies first."
        )

    args = build_arg_parser().parse_args(argv)
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    window = CPRVisualizerWindow(
        pipe_path=args.pipe,
        baseline_mm=args.baseline_mm,
        create_pipe=not args.no_create_pipe,
    )
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
