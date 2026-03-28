"""
VideoMLClient — QThread that forwards frames from the vision pipe to the
inference Docker container, draws detection boxes, and emits:
  - annotatedFrame (QImage)  -> connect to setImage to replace the video widget
  - activityResult (str)     -> connect to set VisionInformation box text
"""
import queue
import time

import numpy as np
import requests
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QImage

SERVER_BASE_URL = "http://localhost:8000"
DETR_ENDPOINT = "/infer/detr"
ACTIVITY_ENDPOINT_TEMPLATE = "/infer/activity/{stream_id}"
ACTIVITY_STREAM_ID = "ems_stream"
JPEG_QUALITY = 85
REQUEST_TIMEOUT_SECONDS = 5
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
HEALTH_LOG_INTERVAL_FRAMES = 30
IDLE_LOG_INTERVAL_SECONDS = 2.0


def _opencv():
    # Delay importing OpenCV until after the Qt app is already initialized.
    import cv2

    return cv2


def _draw_detections(frame_bgr, detections):
    """Draw bounding boxes and labels onto frame in-place."""
    cv2 = _opencv()
    for det in detections:
        box = det.get("box_xyxy", [])
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        label = det.get("label", "?")
        score = det.get("score", 0.0)
        color = (0, 220, 0)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        cv2.putText(
            frame_bgr, text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )


def _frame_to_qimage(frame_bgr):
    """Convert a BGR numpy frame to a scaled QImage ready for the video widget."""
    rgb = frame_bgr[:, :, ::-1].copy()
    h, w, ch = rgb.shape
    img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    return img.scaled(
        DISPLAY_WIDTH, DISPLAY_HEIGHT,
        Qt.KeepAspectRatio, Qt.SmoothTransformation,
    )


class VideoMLClient(QThread):
    annotatedFrame = pyqtSignal(QImage)  # -> connect to setImage
    activityResult = pyqtSignal(str)     # -> connect to VisionInformation.setText

    def __init__(self, ml_queue, parent=None):
        super().__init__(parent)
        self.ml_queue = ml_queue
        self.is_running = True
        self._frame_id = 0
        self._last_frame_received_at = None
        self._last_idle_log_at = 0.0
        print("[VideoMLClient] Initialized")

    def stop(self):
        print("[VideoMLClient] Stopping...")
        self.is_running = False
        self.quit()
        self.wait()
        print("[VideoMLClient] Stopped")

    def _encode(self, frame_bgr):
        cv2 = _opencv()
        ok, buf = cv2.imencode(
            ".jpg", frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return buf.tobytes()

    def _post(self, session, url, image_bytes):
        headers = {
            "Content-Type": "application/octet-stream",
            "x-frame-id": str(self._frame_id),
            "x-timestamp": f"{time.time():.6f}",
        }
        resp = session.post(
            url, data=image_bytes, headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json()

    def run(self):
        print("[VideoMLClient] Started")
        session = requests.Session()
        detr_url = f"{SERVER_BASE_URL}{DETR_ENDPOINT}"
        act_url = SERVER_BASE_URL + ACTIVITY_ENDPOINT_TEMPLATE.format(
            stream_id=ACTIVITY_STREAM_ID
        )

        while self.is_running:
            # ml_queue holds raw BGR numpy frames (no overlay drawn yet)
            wait_started = time.monotonic()
            try:
                frame = self.ml_queue.get(timeout=0.2)
            except queue.Empty:
                now = time.monotonic()
                last_frame_age = (
                    now - self._last_frame_received_at
                    if self._last_frame_received_at is not None
                    else now - wait_started
                )
                if now - self._last_idle_log_at >= IDLE_LOG_INTERVAL_SECONDS:
                    print(
                        "[VideoMLClient] Waiting for next frame "
                        f"(last frame {last_frame_age:.1f}s ago)"
                    )
                    self._last_idle_log_at = now
                continue
            except Exception as e:
                print(f"[VideoMLClient] Queue read error: {e}")
                continue

            self._frame_id += 1
            frame_started = time.monotonic()
            queue_wait_ms = (frame_started - wait_started) * 1000.0
            self._last_frame_received_at = frame_started
            annotated = frame.copy()
            detections = []
            act_line = "buffering..."
            det_line = "none"
            encode_ms = 0.0
            detr_ms = 0.0
            act_ms = 0.0
            emit_ms = 0.0

            encode_started = time.monotonic()
            try:
                image_bytes = self._encode(frame)
            except Exception as e:
                print(f"[VideoMLClient] Encode error: {e}")
                continue
            finally:
                encode_ms = (time.monotonic() - encode_started) * 1000.0

            # --- DETR object detection ---
            detr_started = time.monotonic()
            try:
                detr = self._post(session, detr_url, image_bytes)
                detections = detr.get("detections", [])
                det_strs = [
                    f"{d.get('label', '?')} {d.get('score', 0):.2f}"
                    for d in detections[:3]
                ]
                det_line = ", ".join(det_strs) if det_strs else "none"
            except Exception as e:
                det_line = f"DETR unavailable ({e})"
            finally:
                detr_ms = (time.monotonic() - detr_started) * 1000.0

            # --- Activity recognition ---
            act_started = time.monotonic()
            try:
                act = self._post(session, act_url, image_bytes)
                activity = act.get("activity")
                if isinstance(activity, dict):
                    if activity.get('score', 0) > 0.8:
                        act_line = (
                            f"{activity.get('label', '?')} "
                            f"({activity.get('score', 0):.2f})"
                        )
                    else:
                        act_line = (
                            f"Not confident"
                        )
                else:
                    buf = act.get("buffer_size", "?")
                    win = act.get("window_size", "?")
                    act_line = f"buffering {buf}/{win}"
            except Exception as e:
                act_line = f"unavailable ({e})"
            finally:
                act_ms = (time.monotonic() - act_started) * 1000.0

            # Draw boxes on annotated copy then add status bar overlay
            _draw_detections(annotated, detections)
            ts = time.strftime("%H:%M:%S")
            cv2 = _opencv()
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 44), (20, 20, 20), -1)
            cv2.putText(
                annotated, f"Video ML | {ts}", (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                annotated, f"Activity: {act_line}", (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (125, 255, 125), 1, cv2.LINE_AA,
            )

            emit_started = time.monotonic()
            self.annotatedFrame.emit(_frame_to_qimage(annotated))
            self.activityResult.emit(
                f"[Video ML @ {ts}]\nActivity: {act_line}\nDetections: {det_line}"
            )
            emit_ms = (time.monotonic() - emit_started) * 1000.0
            total_ms = (time.monotonic() - frame_started) * 1000.0

            if self._frame_id % HEALTH_LOG_INTERVAL_FRAMES == 0:
                print(
                    f"[VideoMLClient] Frame {self._frame_id} | "
                    f"queue={queue_wait_ms:.1f}ms encode={encode_ms:.1f}ms "
                    f"detr={detr_ms:.1f}ms act={act_ms:.1f}ms "
                    f"emit={emit_ms:.1f}ms total={total_ms:.1f}ms | "
                    f"Activity: {act_line} | Detections: {det_line}"
                )

        session.close()
        print("[VideoMLClient] Exiting")

    # ------------------------------------------------------------------
    # Warmup: called once at app startup (in a daemon thread) so GPU
    # weights are hot before the user presses Start.
    # ------------------------------------------------------------------
    @staticmethod
    def warmup_static():
        """Send a black dummy frame so model weights load into GPU memory."""
        print("[VideoMLClient] Warming up inference server...")
        dummy = np.zeros((270, 480, 3), dtype=np.uint8)
        cv2 = _opencv()
        ok, buf = cv2.imencode(".jpg", dummy, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if not ok:
            print("[VideoMLClient] Warmup encode failed")
            return
        image_bytes = buf.tobytes()
        headers = {
            "Content-Type": "application/octet-stream",
            "x-frame-id": "0",
            "x-timestamp": "0",
        }
        try:
            s = requests.Session()
            s.post(
                f"{SERVER_BASE_URL}{DETR_ENDPOINT}",
                data=image_bytes, headers=headers, timeout=15,
            )
            s.post(
                f"{SERVER_BASE_URL}{ACTIVITY_ENDPOINT_TEMPLATE.format(stream_id=ACTIVITY_STREAM_ID)}",
                data=image_bytes, headers=headers, timeout=15,
            )
            s.close()
            print("[VideoMLClient] Warmup complete")
        except Exception as e:
            print(f"[VideoMLClient] Warmup failed (Docker not running?): {e}")
