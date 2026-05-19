"""Serial reader and FIFO publisher for Arduino VL6180 CPR depth samples."""

import csv
import os
import queue
import stat
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Event, Process
from typing import Optional

PIPE_CPR = "/tmp/emscpr"
PIPE_CPR_ML = "/tmp/emscprml"
CSV_HEADER = ["Timestamp (ns)", "Range (mm)", "Sequence"]


@dataclass(frozen=True)
class CPRDepthSample:
    """One normalized CPR sensor sample."""

    timestamp_ns: int
    range_mm: int
    sequence: int

    def to_csv_row(self):
        return [self.timestamp_ns, self.range_mm, self.sequence]

    def to_csv_line(self):
        return f"{self.timestamp_ns},{self.range_mm},{self.sequence}\n"


def parse_arduino_line(line: str, timestamp_ns: Optional[int] = None) -> Optional[CPRDepthSample]:
    """Parse one line from the Arduino sketch.

    Expected successful sensor line:
        Range:<range_mm>:Seq:<sequence>

    Sensor error/status lines from the sketch include only the sequence number,
    for example:
        No convergence:Seq:<sequence>

    Startup lines such as "Start" return None.
    """
    line = line.strip()
    if not line or line.startswith("Start"):
        return None

    if "Seq" not in line:
        return None

    timestamp_ns = timestamp_ns if timestamp_ns is not None else int(datetime.now().timestamp() * 1e9)
    parts = line.split(":")

    try:
        sequence = int(parts[-1])
    except (IndexError, ValueError):
        return None

    range_mm = -1
    if line.startswith("Range:"):
        try:
            range_mm = int(parts[1])
        except (IndexError, ValueError):
            range_mm = -1

    return CPRDepthSample(timestamp_ns=timestamp_ns, range_mm=range_mm, sequence=sequence)


def setup_cpr_pipes(pipe_paths=(PIPE_CPR, PIPE_CPR_ML)):
    """Create CPR named pipes if they do not already exist."""
    for pipe_path in pipe_paths:
        try:
            if os.path.exists(pipe_path):
                if not stat_is_fifo(pipe_path):
                    raise RuntimeError(f"{pipe_path} exists and is not a FIFO")
                continue
            os.mkfifo(pipe_path)
            print(f"[CPRPipe] Created {pipe_path}")
        except Exception as exc:
            print(f"[CPRPipe] Error creating {pipe_path}: {exc}")


def cleanup_cpr_pipes(pipe_paths=(PIPE_CPR, PIPE_CPR_ML)):
    """Remove CPR named pipes."""
    for pipe_path in pipe_paths:
        try:
            if os.path.exists(pipe_path):
                os.unlink(pipe_path)
        except Exception:
            pass


def stat_is_fifo(path):
    return stat.S_ISFIFO(os.stat(path).st_mode)


class CPRPipePublisher:
    """Non-blocking writer for CPR samples to one or more text FIFOs."""

    def __init__(self, pipe_paths, create_pipes=False):
        self.pipe_paths = tuple(pipe_paths)
        self.create_pipes = create_pipes
        self.sample_queue = queue.Queue(maxsize=64)
        self.opened_pipes = {}
        self.stop_event = threading.Event()
        self.drop_count = 0
        self.threads = []

    def start(self):
        if self.create_pipes:
            setup_cpr_pipes(self.pipe_paths)

        for pipe_path in self.pipe_paths:
            thread = threading.Thread(target=self._open_pipe, args=(pipe_path,), daemon=True)
            thread.start()
            self.threads.append(thread)

        writer_thread = threading.Thread(target=self._write_samples, daemon=True)
        writer_thread.start()
        self.threads.append(writer_thread)

    def publish(self, sample):
        try:
            self.sample_queue.put_nowait(sample)
        except queue.Full:
            self.drop_count += 1

    def stop(self):
        self.stop_event.set()
        for pipe_file in self.opened_pipes.values():
            try:
                pipe_file.close()
            except Exception:
                pass

    def _open_pipe(self, pipe_path):
        try:
            self.opened_pipes[pipe_path] = open(pipe_path, "w", buffering=1)
            print(f"[CPRPipe] Opened pipe {pipe_path}")
        except Exception as exc:
            print(f"[CPRPipe] Failed to open {pipe_path}: {exc}")

    def _write_samples(self):
        while not self.stop_event.is_set() or not self.sample_queue.empty():
            try:
                sample = self.sample_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            line = sample.to_csv_line()
            for pipe_path, pipe_file in list(self.opened_pipes.items()):
                try:
                    pipe_file.write(line)
                    pipe_file.flush()
                except BrokenPipeError:
                    self.opened_pipes.pop(pipe_path, None)
                except Exception as exc:
                    print(f"[CPRPipe] Error writing to {pipe_path}: {exc}")


class ArduinoVL6180Recorder(Process):
    def __init__(
        self,
        port,
        recording_dir,
        thread_stop,
        baudrate=115200,
        pipe_paths=None,
        create_pipes=False,
        save_enabled=True,
    ):
        super().__init__()
        self.serial_port = port
        self.baudrate = baudrate
        self.recording_dir = os.path.join(recording_dir, "VL6180/")
        if save_enabled:
            os.makedirs(self.recording_dir, exist_ok=True)
        self.thread_stop = thread_stop
        self.recording_enabled = Event()
        self.save_enabled = Event()
        if save_enabled:
            self.save_enabled.set()
        self.running = False
        self.ser = None
        self.pipe_paths = tuple(pipe_paths or ())
        self.create_pipes = create_pipes
        self.csv_filename = os.path.join(
            self.recording_dir,
            f"VL6180_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )

    def start_recording(self):
        self.running = True
        self.recording_enabled.set()
        print("Arduino: Recording started.")

    def stop_recording(self):
        self.running = False
        self.recording_enabled.clear()
        print("Arduino: Recording stopped.")

    def set_save_enabled(self, save_enabled):
        if save_enabled:
            os.makedirs(self.recording_dir, exist_ok=True)
            self.save_enabled.set()
        else:
            self.save_enabled.clear()

    def run(self):
        pipe_publisher = None
        csv_file = None
        writer = None
        csv_started = False

        try:
            import serial

            self.ser = serial.Serial(self.serial_port, self.baudrate)
            time.sleep(2)
            print(f"Arduino: Connected to Arduino on port {self.serial_port} at {self.baudrate} baud.")

            if self.pipe_paths:
                pipe_publisher = CPRPipePublisher(self.pipe_paths, create_pipes=self.create_pipes)
                pipe_publisher.start()

            record = False
            if not self.save_enabled.is_set():
                print("Arduino: CSV saving disabled; publishing/reading only.")

            while not self.thread_stop.is_set():
                if not self.recording_enabled.is_set():
                    time.sleep(0.01)
                    continue

                if csv_file is not None and not self.save_enabled.is_set():
                    csv_file.close()
                    csv_file = None
                    writer = None
                    print("Arduino: CSV saving paused.")

                if self.ser.in_waiting <= 0:
                    time.sleep(0.001)
                    continue

                try:
                    line = self.ser.readline().decode("utf-8").strip()
                    if line.startswith("Start"):
                        record = True
                        continue

                    if not record:
                        continue

                    sample = parse_arduino_line(line)
                    if sample is None:
                        continue

                    if self.save_enabled.is_set():
                        if writer is None:
                            os.makedirs(self.recording_dir, exist_ok=True)
                            csv_file = open(self.csv_filename, mode="a" if csv_started else "w", newline="")
                            writer = csv.writer(csv_file)
                            if not csv_started:
                                writer.writerow(CSV_HEADER)
                                csv_started = True
                            print(f"Arduino: Recording started to {self.csv_filename}")

                        writer.writerow(sample.to_csv_row())
                        csv_file.flush()

                    if pipe_publisher is not None:
                        pipe_publisher.publish(sample)

                except Exception as exc:
                    print(f"Arduino: Error parsing serial line: {exc}")

            print("Arduino: recording stopped...")

        except Exception as exc:
            print(f"Arduino: Error: {exc}")

        finally:
            if pipe_publisher is not None:
                pipe_publisher.stop()
            if csv_file is not None:
                csv_file.close()
            self.close()

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Arduino: Serial connection closed.")

    def stop(self):
        self.thread_stop.set()
        self.recording_enabled.clear()
        self.join()
        print("Arduino: Recording process stopped.")


if __name__ == "__main__":
    thread_stop = Event()
    thread_stop.clear()
    recorder = ArduinoVL6180Recorder(
        port="COM5",
        recording_dir="./recordings",
        thread_stop=thread_stop,
        pipe_paths=(PIPE_CPR,),
        create_pipes=True,
    )

    try:
        recorder.start_recording()
        recorder.start()
        time.sleep(10)
        recorder.stop_recording()
        recorder.stop()
    finally:
        recorder.close()
