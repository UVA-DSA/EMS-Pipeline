# Thread for streaming smartwatch/IMU CSV data from pipe to GUI

import os
import time
import threading
import csv
import io

from PyQt5.QtCore import QThread, pyqtSignal


class IMUThread(QThread):
    """
    Thread that reads CSV data from /tmp/emscsv pipe and emits it to GUI.
    CSV format: timestamp,x_01,y_01,z_01 (linear acceleration only)
    """

    changeActivityRec = pyqtSignal(str)  # Signal to update smartwatch activity display

    def __init__(self, data_path="", smartwatchStream=False):
        super().__init__()

        self.PIPE_CSV = "/tmp/emscsv"
        self.csv_pipe_holder = None
        self.is_running = True
        self.data_path = data_path
        self.smartwatchStream = smartwatchStream

        # For data collection if enabled
        self.csv_writer = None
        self.csv_file = None

        print('[IMUThread] Initialized')

    def stop(self):
        """Stop the IMU streaming thread."""
        print("[INFO] IMU Thread stopping...")
        self.is_running = False

        # Close data collection file if open
        if self.csv_file is not None:
            self.csv_file.close()

        self.quit()
        print("[INFO] IMU Thread Stopped")

    def run(self):
        """Main thread loop - reads CSV data from pipe and emits to GUI."""
        is_connected = False

        # Setup data collection if enabled
        if self.smartwatchStream:
            try:
                os.makedirs(self.data_path + "smartwatchdata/", exist_ok=True)
                csv_path = self.data_path + "smartwatchdata/imu_data.csv"
                self.csv_file = open(csv_path, 'w', newline='')
                self.csv_writer = csv.writer(self.csv_file)
                # Write header
                self.csv_writer.writerow(['timestamp', 'x_01', 'y_01', 'z_01'])
                print(f"[IMUThread] Data collection enabled: {csv_path}")
            except Exception as e:
                print(f"[IMUThread] Error setting up data collection: {e}")

        try:
            print(f"[IMUThread] Opening {self.PIPE_CSV} for reading...")
            self.csv_pipe_holder = open(self.PIPE_CSV, 'r', buffering=1)  # Line buffered
            print(f"[IMUThread] {self.PIPE_CSV} opened!")
            is_connected = True

        except Exception as e:
            print(f'[IMUThread] Error opening CSV pipe: {e}')
            return

        line_count = 0

        while self.is_running and is_connected:
            try:
                # Read one line of CSV data
                line = self.csv_pipe_holder.readline()

                if not line:
                    print("[IMUThread] CSV pipe closed")
                    break

                # Strip whitespace and newlines
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

                        # Calculate magnitude of acceleration vector
                        accel_magnitude = (x_accel**2 + y_accel**2 + z_accel**2)**0.5

                        # Format display string for GUI
                        display_text = (
                            f"Time: {timestamp}\n"
                            f"Linear Accel (m/s²):\n"
                            f"  X: {x_accel:>8.3f}\n"
                            f"  Y: {y_accel:>8.3f}\n"
                            f"  Z: {z_accel:>8.3f}\n"
                            f"  Magnitude: {accel_magnitude:.3f}"
                        )

                        # Emit to GUI
                        self.changeActivityRec.emit(display_text)

                        # Save to file if enabled
                        if self.smartwatchStream and self.csv_writer is not None:
                            self.csv_writer.writerow([timestamp, x_accel, y_accel, z_accel])

                            # Flush periodically
                            if line_count % 30 == 0:
                                self.csv_file.flush()

                        # Debug output every 30 lines (~1 second at 30fps)
                        if line_count % 30 == 0:
                            print(f"[IMUThread] Processed {line_count} CSV lines")
                    else:
                        print(f"[IMUThread] Invalid CSV line (expected 4 values, got {len(values)}): {line}")

                except ValueError as e:
                    print(f"[IMUThread] Error parsing CSV line: {e}")
                    print(f"[IMUThread] Line: {line}")

            except Exception as e:
                print(f'[IMUThread] Error in CSV reading loop: {e}')
                import traceback
                traceback.print_exc()
                break

        print("[IMUThread] Exiting run loop")


class Thread_Watch(QThread):
    """
    Wrapper class for backward compatibility with existing GUI code.
    This maintains the same interface as the original smartwatch_streaming.Thread_Watch
    """

    changeActivityRec = pyqtSignal(str)

    def __init__(self, data_path="", smartwatchStream=False, ip="", port=0):
        super().__init__()

        # Create the actual IMU thread
        self.imu_thread = IMUThread(data_path, smartwatchStream)

        # Connect signals
        self.imu_thread.changeActivityRec.connect(self.changeActivityRec.emit)

        print(f'[Thread_Watch] Wrapper initialized (forwarding to IMUThread)')

    def run(self):
        """Start the IMU thread."""
        self.imu_thread.start()

        # Keep this wrapper thread alive while IMU thread is running
        while self.imu_thread.is_running:
            time.sleep(0.1)

    def stop(self):
        """Stop the IMU thread."""
        self.imu_thread.stop()
        self.quit()