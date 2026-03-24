# Process for streaming smartwatch/IMU CSV data from pipe to GUI

import os
import time
import csv
from multiprocessing import Process, Queue, Value
import ctypes

from PyQt5.QtCore import QThread, pyqtSignal


def imu_reader_process(pipe_path, data_queue, running_flag, data_path="", save_enabled=False):
    """
    Separate process that reads CSV data from pipe and puts it in queue.
    Args:
        pipe_path: Path to CSV pipe (/tmp/emscsv)
        data_queue: Multiprocessing Queue to send parsed data to GUI
        running_flag: Shared Value to control process lifecycle
        data_path: Path for data collection
        save_enabled: Whether to save data to file
    """
    print(f"[IMUProcess] Starting, PID: {os.getpid()}")

    csv_file = None
    csv_writer = None

    # Setup data collection if enabled
    if save_enabled:
        try:
            os.makedirs(data_path + "smartwatchdata/", exist_ok=True)
            csv_path = data_path + "smartwatchdata/imu_data.csv"
            csv_file = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['timestamp', 'x_01', 'y_01', 'z_01'])
            print(f"[IMUProcess] Data collection enabled: {csv_path}")
        except Exception as e:
            print(f"[IMUProcess] Error setting up data collection: {e}")

    try:
        print(f"[IMUProcess] Opening {pipe_path} for reading...")
        csv_pipe = open(pipe_path, 'r', buffering=1)
        print(f"[IMUProcess] {pipe_path} opened!")

        line_count = 0
        last_debug = time.time()

        while running_flag.value:
            try:
                # Read one line of CSV data
                line = csv_pipe.readline()

                if not line:
                    print("[IMUProcess] CSV pipe closed")
                    break

                line = line.strip()
                if not line:
                    continue

                line_count += 1

                # Parse CSV line
                try:
                    # Expected format: timestamp,x_01,y_01,z_01
                    values = line.split(',')

                    if len(values) >= 4:
                        timestamp = values[0]
                        x_accel = float(values[1])
                        y_accel = float(values[2])
                        z_accel = float(values[3])

                        # Calculate magnitude
                        accel_magnitude = (x_accel**2 + y_accel**2 + z_accel**2)**0.5

                        # Create data dict
                        data = {
                            'timestamp': timestamp,
                            'x': x_accel,
                            'y': y_accel,
                            'z': z_accel,
                            'magnitude': accel_magnitude
                        }

                        # Put in queue (non-blocking, drop if full)
                        try:
                            data_queue.put_nowait(data)
                        except:
                            # Queue full, drop data
                            pass

                        # Save to file if enabled
                        if save_enabled and csv_writer is not None:
                            csv_writer.writerow([timestamp, x_accel, y_accel, z_accel])

                            if line_count % 30 == 0:
                                csv_file.flush()

                        # Debug every 30 lines
                        if line_count % 30 == 0:
                            elapsed = time.time() - last_debug
                            print(f"[IMUProcess] Processed {line_count} lines "
                                  f"(~{30/elapsed:.1f} Hz)")
                            last_debug = time.time()
                    else:
                        print(f"[IMUProcess] Invalid CSV line: {line}")

                except ValueError as e:
                    print(f"[IMUProcess] Error parsing line: {e}")

            except Exception as e:
                if running_flag.value:
                    print(f"[IMUProcess] Error in read loop: {e}")
                break

        csv_pipe.close()
        if csv_file:
            csv_file.close()

    except Exception as e:
        print(f'[IMUProcess] Error: {e}')
        import traceback
        traceback.print_exc()

    print(f"[IMUProcess] Exiting, processed {line_count} lines")


class IMUDisplayThread(QThread):
    """
    Qt thread that reads from multiprocessing queue and emits signals to GUI.
    """
    changeActivityRec = pyqtSignal(str)

    def __init__(self, data_queue):
        super().__init__()
        self.data_queue = data_queue
        self.is_running = True
        print('[IMUDisplayThread] Initialized')

    def stop(self):
        print("[IMUDisplayThread] Stopping...")
        self.is_running = False
        self.quit()
        self.wait()
        print("[IMUDisplayThread] Stopped")

    def run(self):
        """Read data from queue and emit to GUI."""
        print("[IMUDisplayThread] Started")

        while self.is_running:
            try:
                # Try to get data with timeout
                data = self.data_queue.get(timeout=0.1)

                # Format display string
                display_text = (
                    f"Time: {data['timestamp']}\n"
                    f"Linear Accel (m/s²):\n"
                    f"  X: {data['x']:>8.3f}\n"
                    f"  Y: {data['y']:>8.3f}\n"
                    f"  Z: {data['z']:>8.3f}\n"
                    f"  Magnitude: {data['magnitude']:.3f}"
                )

                self.changeActivityRec.emit(display_text)

            except:
                # Timeout or queue empty
                continue

        print("[IMUDisplayThread] Exiting run loop")


class IMUStreamManager:
    """
    Manager class that coordinates the IMU reader process and display thread.
    """
    def __init__(self, data_path="", smartwatch_stream=False):
        self.data_path = data_path
        self.smartwatch_stream = smartwatch_stream
        self.PIPE_CSV = "/tmp/emscsv"

        # Multiprocessing components
        self.data_queue = Queue(maxsize=10)
        self.running_flag = Value(ctypes.c_bool, True)

        # Reader process
        self.reader_process = Process(
            target=imu_reader_process,
            args=(self.PIPE_CSV, self.data_queue, self.running_flag,
                  data_path, smartwatch_stream),
            daemon=True
        )

        # Display thread
        self.display_thread = IMUDisplayThread(self.data_queue)

        print('[IMUStreamManager] Initialized')

    def start(self):
        """Start both the reader process and display thread."""
        print("[IMUStreamManager] Starting...")
        self.running_flag.value = True
        self.reader_process.start()
        self.display_thread.start()
        print(f"[IMUStreamManager] Started (Process PID: {self.reader_process.pid})")

    def stop(self):
        """Stop both the reader process and display thread."""
        print("[IMUStreamManager] Stopping...")

        # Stop process first
        self.running_flag.value = False
        self.reader_process.join(timeout=3)

        if self.reader_process.is_alive():
            print("[IMUStreamManager] Process didn't stop, terminating...")
            self.reader_process.terminate()
            self.reader_process.join(timeout=1)

        # Stop display thread
        self.display_thread.stop()

        print("[IMUStreamManager] Stopped")

    @property
    def changeActivityRec(self):
        """Provide access to display thread's signal for GUI connection."""
        return self.display_thread.changeActivityRec
