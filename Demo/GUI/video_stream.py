# Process for streaming video to the GUI vision window

import os
import time
import numpy as np
from multiprocessing import Process, Queue, Value
import ctypes

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


def video_reader_process(pipe_path, frame_queue, running_flag):
    """
    Separate process that reads video frames from pipe and puts them in queue.
    Args:
        pipe_path: Path to video pipe (/tmp/emsvid)
        frame_queue: Multiprocessing Queue to send frames to GUI
        running_flag: Shared Value to control process lifecycle
    """
    print(f"[VideoProcess] Starting, PID: {os.getpid()}")

    def read_exactly(pipe, n):
        """Read exactly n bytes from pipe."""
        data = b''
        while len(data) < n:
            chunk = pipe.read(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    try:
        print(f"[VideoProcess] Opening {pipe_path} for reading...")
        video_pipe = open(pipe_path, 'rb', buffering=0)
        print(f"[VideoProcess] {pipe_path} opened!")

        frame_count = 0

        while running_flag.value:
            try:
                # Read video frame
                # Format: 4 bytes length + data
                length_bytes = read_exactly(video_pipe, 4)
                if length_bytes is None:
                    print("[VideoProcess] Video pipe closed")
                    break

                video_length = int.from_bytes(length_bytes, 'big')
                video_data = read_exactly(video_pipe, video_length)

                if video_data is None:
                    print("[VideoProcess] Failed to read video data")
                    break

                # Decode video frame
                try:
                    video_frame = np.frombuffer(video_data, dtype=np.uint8).reshape(
                        (270, 480, 3)
                    )

                    # Put frame in queue (non-blocking, drop if full)
                    try:
                        frame_queue.put_nowait(video_frame.copy())
                        frame_count += 1

                        if frame_count % 30 == 0:
                            print(f"[VideoProcess] Processed {frame_count} frames")
                    except:
                        # Queue full, drop frame
                        pass

                except Exception as e:
                    print(f"[VideoProcess] Failed to decode video frame: {e}")

            except Exception as e:
                if running_flag.value:  # Only print error if we're supposed to be running
                    print(f"[VideoProcess] Error in read loop: {e}")
                break

        video_pipe.close()

    except Exception as e:
        print(f'[VideoProcess] Error opening pipe: {e}')

    print(f"[VideoProcess] Exiting, processed {frame_count} frames")


class VideoDisplayThread(QThread):
    """
    Qt thread that reads from multiprocessing queue and emits signals to GUI.
    This bridges the multiprocessing.Queue to PyQt signals.
    """
    changePixmap = pyqtSignal(QImage)
    changeVisInfo = pyqtSignal(str)

    def __init__(self, frame_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.is_running = True
        print('[VideoDisplayThread] Initialized')

    def stop(self):
        print("[VideoDisplayThread] Stopping...")
        self.is_running = False
        self.quit()
        self.wait()
        print("[VideoDisplayThread] Stopped")

    def run(self):
        """Read frames from queue and emit to GUI."""
        print("[VideoDisplayThread] Started")

        while self.is_running:
            try:
                # Try to get frame with timeout
                frame = self.frame_queue.get(timeout=0.1)

                # Convert BGR to RGB
                RGB_img = frame[:, :, ::-1].copy()

                h, w, ch = RGB_img.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(RGB_img.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)

                self.changePixmap.emit(p)

            except:
                # Timeout or queue empty
                continue

        print("[VideoDisplayThread] Exiting run loop")


class VideoStreamManager:
    """
    Manager class that coordinates the video reader process and display thread.
    Use this in GUI.py instead of VideoThread.
    """
    def __init__(self, data_path, videostream_enabled):
        self.data_path = data_path
        self.videostream_enabled = videostream_enabled
        self.PIPE_VIDEO = "/tmp/emsvid"

        # Multiprocessing components
        self.frame_queue = Queue(maxsize=10)
        self.running_flag = Value(ctypes.c_bool, True)

        # Reader process
        self.reader_process = Process(
            target=video_reader_process,
            args=(self.PIPE_VIDEO, self.frame_queue, self.running_flag),
            daemon=True
        )

        # Display thread (bridges to Qt signals)
        self.display_thread = VideoDisplayThread(self.frame_queue)

        print('[VideoStreamManager] Initialized')

    def start(self):
        """Start both the reader process and display thread."""
        print("[VideoStreamManager] Starting...")
        self.running_flag.value = True
        self.reader_process.start()
        self.display_thread.start()
        print(f"[VideoStreamManager] Started (Process PID: {self.reader_process.pid})")

    def stop(self):
        """Stop both the reader process and display thread."""
        print("[VideoStreamManager] Stopping...")

        # Stop process first
        self.running_flag.value = False
        self.reader_process.join(timeout=3)

        if self.reader_process.is_alive():
            print("[VideoStreamManager] Process didn't stop, terminating...")
            self.reader_process.terminate()
            self.reader_process.join(timeout=1)

        # Stop display thread
        self.display_thread.stop()

        print("[VideoStreamManager] Stopped")

    @property
    def changePixmap(self):
        """Provide access to display thread's signal for GUI connection."""
        return self.display_thread.changePixmap

    @property
    def changeVisInfo(self):
        """Provide access to display thread's signal for GUI connection."""
        return self.display_thread.changeVisInfo
