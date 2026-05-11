import ctypes
import datetime
import os
import select
import time

import numpy as np


PIPE_VIDEO = "/tmp/emsvid"
FRAME_WIDTH = 480
FRAME_HEIGHT = 270
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
PIPE_IDLE_TIMEOUT_SECONDS = 5.0


def _read_exactly(pipe_handle, n_bytes, idle_timeout_seconds=None, on_idle=None):
    data = bytearray()
    while len(data) < n_bytes:
        if idle_timeout_seconds is not None:
            ready, _, _ = select.select([pipe_handle], [], [], idle_timeout_seconds)
            if not ready:
                if on_idle is not None:
                    on_idle()
                continue
        chunk = pipe_handle.read(n_bytes - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def _enqueue_latest(queue_handle, item):
    try:
        queue_handle.put_nowait(item)
    except Exception:
        try:
            queue_handle.get_nowait()
            queue_handle.put_nowait(item)
        except Exception:
            pass


def _build_status_frame(cv2_module, frame_width, frame_height, title, subtitle):
    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    frame[:] = (28, 28, 28)

    cv2_module.rectangle(frame, (0, 0), (frame_width, 60), (12, 12, 12), -1)
    cv2_module.putText(
        frame,
        title,
        (14, 26),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2_module.LINE_AA,
    )
    cv2_module.putText(
        frame,
        subtitle,
        (14, 50),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.48,
        (175, 220, 255),
        1,
        cv2_module.LINE_AA,
    )
    return frame


def vision_reader_process(pipe_path, frame_queue, running_flag, frame_width, frame_height, ml_queue=None):
    """
    Read raw frames from the display video pipe, add a lightweight overlay,
    and forward the annotated frames back to the GUI bridge queue.
    """
    print(f"[VisionProcess] Starting, PID: {os.getpid()}")

    try:
        import cv2
    except ImportError as exc:
        print(f"[VisionProcess] OpenCV is required for overlay rendering: {exc}")
        return

    expected_bytes = frame_width * frame_height * 3
    frame_count = 0
    session_count = 0
    has_received_stream = False
    pipe_stalled = False
    last_pipe_idle_log_at = 0.0

    # ---- Reader-side diagnostic log (mirrors SRT/local sampler logs) -------
    import zlib as _zlib
    _session_ts = time.strftime("%Y%m%d_%H%M%S")
    _log_dir    = "/tmp/ems_session_logs"
    os.makedirs(_log_dir, exist_ok=True)
    _reader_log_path = f"{_log_dir}/vision_reader_{_session_ts}.tsv"
    _reader_log = open(_reader_log_path, "w", buffering=1)
    _reader_log.write("# vision_reader_process diagnostic log\n")
    _reader_log.write(f"# pipe={pipe_path}  expected_frame_bytes={expected_bytes}\n")
    _reader_log.write("event\tframe_count\ttimestamp\tdetail\n")
    print(f"[VisionProcess] Diagnostic log: {_reader_log_path}")

    def _rlog(event, detail=""):
        _reader_log.write(f"{event}\t{frame_count}\t{time.time():.6f}\t{detail}\n")

    while running_flag.value:
        video_pipe = None
        try:
            print(f"[VisionProcess] Opening {pipe_path} for reading...")
            video_pipe = open(pipe_path, "rb", buffering=0)
            session_count += 1
            pipe_stalled = False
            print(f"[VisionProcess] {pipe_path} opened! Session {session_count}")

            while running_flag.value:
                try:
                    def on_pipe_idle():
                        nonlocal pipe_stalled, last_pipe_idle_log_at
                        pipe_stalled = True
                        now = time.monotonic()
                        if now - last_pipe_idle_log_at < PIPE_IDLE_TIMEOUT_SECONDS:
                            return
                        last_pipe_idle_log_at = now
                        print(
                            "[VisionProcess] No video bytes received for "
                            f"{PIPE_IDLE_TIMEOUT_SECONDS:.1f}s"
                        )
                        if has_received_stream:
                            waiting_frame = _build_status_frame(
                                cv2,
                                frame_width,
                                frame_height,
                                "Video stream stalled",
                                "Waiting for new video frames...",
                            )
                            _enqueue_latest(
                                frame_queue,
                                (
                                    waiting_frame,
                                    "Vision stream stalled\n"
                                    "Waiting for new video frames to resume.",
                                ),
                            )

                    length_bytes = _read_exactly(
                        video_pipe,
                        4,
                        idle_timeout_seconds=PIPE_IDLE_TIMEOUT_SECONDS,
                        on_idle=on_pipe_idle,
                    )
                    if length_bytes is None:
                        print("[VisionProcess] Video pipe closed; waiting for stream to resume")
                        if has_received_stream:
                            waiting_frame = _build_status_frame(
                                cv2,
                                frame_width,
                                frame_height,
                                "Waiting for smartglass video...",
                                "Reconnect the app to resume live video.",
                            )
                            _enqueue_latest(
                                frame_queue,
                                (
                                    waiting_frame,
                                    "Vision stream disconnected\n"
                                    "Waiting for smartglass video to resume.",
                                ),
                            )
                        break

                    frame_length = int.from_bytes(length_bytes, "big")

                    # ---- Diagnostic: log every length prefix ------------
                    if frame_length != expected_bytes:
                        # Catch the bug red-handed: log the raw prefix bytes
                        # and the surrounding context.
                        _hex_prefix = length_bytes.hex()
                        _rlog("BAD_LENGTH_PREFIX",
                              f"got={frame_length} expected={expected_bytes} "
                              f"raw_hex={_hex_prefix} after_frames={frame_count}")
                        print(f"[VisionProcess] BAD LENGTH PREFIX after frame "
                              f"{frame_count}: got {frame_length} "
                              f"(0x{_hex_prefix}) expected {expected_bytes}")
                        # Try to drain whatever follows so we can see what's
                        # actually in the pipe.  Read up to 64KB.
                        try:
                            _drained = pipe_handle.read(65536)
                            _rlog("DRAIN_AFTER_BAD_PREFIX",
                                  f"drained_bytes={len(_drained) if _drained else 0}")
                        except Exception as _exc:
                            _rlog("DRAIN_FAILED", str(_exc))
                        # Bail out — outer loop will reopen the pipe and
                        # resync.
                        break

                    frame_bytes = _read_exactly(
                        video_pipe,
                        frame_length,
                        idle_timeout_seconds=PIPE_IDLE_TIMEOUT_SECONDS,
                        on_idle=on_pipe_idle,
                    )
                    if frame_bytes is None:
                        print("[VisionProcess] Incomplete video frame; waiting for reconnection")
                        if has_received_stream:
                            waiting_frame = _build_status_frame(
                                cv2,
                                frame_width,
                                frame_height,
                                "Video stream interrupted",
                                "Waiting for smartglass video to resume.",
                            )
                            _enqueue_latest(
                                frame_queue,
                                (
                                    waiting_frame,
                                    "Vision stream interrupted\n"
                                    "Waiting for smartglass video to resume.",
                                ),
                            )
                        break

                    if frame_length != expected_bytes:
                        print(
                            "[VisionProcess] Unexpected frame size: "
                            f"expected {expected_bytes} bytes, received {frame_length}"
                        )
                        continue

                    # ---- Diagnostic: hash every frame so we can compare to
                    # the SRT receiver's adler32 log for the same frame_count.
                    if frame_count % 30 == 0:
                        _h = _zlib.adler32(frame_bytes) & 0xFFFFFFFF
                        _rlog("FRAME_OK",
                              f"len={frame_length} adler32={_h:08x}")

                    frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
                        (frame_height, frame_width, 3)
                    ).copy()
                    if pipe_stalled:
                        print("[VisionProcess] Video pipe resumed")
                        pipe_stalled = False
                    frame_count += 1
                    has_received_stream = True

                    # Send the clean (pre-overlay) frame to the ML queue so the
                    # inference server receives unmodified image data.
                    if ml_queue is not None:
                        _enqueue_latest(ml_queue, frame.copy())

                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    overlay_text = f"UI overlay | {timestamp}"
                    detail_text = f"Frame {frame_count}"

                    cv2.rectangle(frame, (0, 0), (frame_width, 48), (20, 20, 20), -1)
                    cv2.putText(
                        frame,
                        overlay_text,
                        (12, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        detail_text,
                        (12, 38),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (125, 255, 125),
                        1,
                        cv2.LINE_AA,
                    )

                    info_text = (
                        "Vision UI overlay active\n"
                        f"Timestamp: {timestamp}\n"
                        f"Frame: {frame_count}\n"
                        f"Session: {session_count}\n"
                        "Source: pipe-driven display stream"
                    )
                    _enqueue_latest(frame_queue, (frame, info_text))

                    if frame_count % 30 == 0:
                        print(f"[VisionProcess] Processed {frame_count} frames")
                except Exception as exc:
                    if running_flag.value:
                        print(f"[VisionProcess] Error in read loop: {exc}")
                    break
        except Exception as exc:
            if running_flag.value:
                print(f"[VisionProcess] Error opening pipe: {exc}")
        finally:
            if video_pipe is not None:
                try:
                    video_pipe.close()
                except Exception:
                    pass

    print(f"[VisionProcess] Exiting, processed {frame_count} frames")


def _make_display_thread():
    """
    Lazily create the Qt bridge so the spawned process does not need to import
    PyQt at module import time.
    """
    from PyQt5.QtCore import QThread, Qt, pyqtSignal
    from PyQt5.QtGui import QImage

    class VisionDisplayThread(QThread):
        changePixmap = pyqtSignal(QImage)
        changeVisInfo = pyqtSignal(str)

        def __init__(self, frame_queue):
            super().__init__()
            self.frame_queue = frame_queue
            self.is_running = True
            self._last_info_text = None
            print("[VisionDisplayThread] Initialized")

        def stop(self):
            print("[VisionDisplayThread] Stopping...")
            self.is_running = False
            self.quit()
            self.wait()
            print("[VisionDisplayThread] Stopped")

        def run(self):
            print("[VisionDisplayThread] Started")
            while self.is_running:
                try:
                    frame, info_text = self.frame_queue.get(timeout=0.1)

                    rgb_frame = frame[:, :, ::-1].copy()
                    height, width, channels = rgb_frame.shape
                    bytes_per_line = channels * width
                    image = QImage(
                        rgb_frame.data,
                        width,
                        height,
                        bytes_per_line,
                        QImage.Format_RGB888,
                    ).copy()
                    scaled = image.scaled(
                        DISPLAY_WIDTH,
                        DISPLAY_HEIGHT,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )

                    self.changePixmap.emit(scaled)

                    if info_text and info_text != self._last_info_text:
                        self.changeVisInfo.emit(info_text)
                        self._last_info_text = info_text
                except Exception:
                    continue

            print("[VisionDisplayThread] Exiting run loop")

    return VisionDisplayThread


class VisionStreamManager:
    """
    Manager class that coordinates the vision overlay process and the Qt
    display bridge for the GUI.
    """

    def __init__(self, data_path, videostream_enabled):
        self.data_path = data_path
        self.videostream_enabled = videostream_enabled
        self.pipe_video = PIPE_VIDEO

        import multiprocessing as mp

        self.ctx = mp.get_context("spawn")
        self.frame_queue = self.ctx.Queue(maxsize=10)
        self._ml_queue = self.ctx.Queue(maxsize=5)
        self.running_flag = self.ctx.Value(ctypes.c_bool, True)
        self.reader_process = self.ctx.Process(
            target=vision_reader_process,
            args=(
                self.pipe_video,
                self.frame_queue,
                self.running_flag,
                FRAME_WIDTH,
                FRAME_HEIGHT,
                self._ml_queue,
            ),
            daemon=True,
        )

        display_thread = _make_display_thread()
        self.display_thread = display_thread(self.frame_queue)
        print("[VisionStreamManager] Initialized")

    def start(self):
        print("[VisionStreamManager] Starting...")
        self.running_flag.value = True
        self.reader_process.start()
        self.display_thread.start()
        print(f"[VisionStreamManager] Started (Process PID: {self.reader_process.pid})")

    def stop(self):
        print("[VisionStreamManager] Stopping...")
        self.running_flag.value = False
        self.reader_process.join(timeout=3)

        if self.reader_process.is_alive():
            print("[VisionStreamManager] Process did not stop, terminating...")
            self.reader_process.terminate()
            self.reader_process.join(timeout=1)

        self.display_thread.stop()
        print("[VisionStreamManager] Stopped")

    @property
    def changePixmap(self):
        return self.display_thread.changePixmap

    @property
    def changeVisInfo(self):
        return self.display_thread.changeVisInfo

    @property
    def ml_queue(self):
        return self._ml_queue


# Compatibility alias in case older code still expects the old manager name.
VideoStreamManager = VisionStreamManager