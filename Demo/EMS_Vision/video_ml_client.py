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

from IO.bbox_engine import BBoxPublisher, build_bbox_box
from IO.feedback_engine import FeedbackPublisher, format_feedback_text

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
ACTIVITY_FEEDBACK_THRESHOLD = 0.80
BBOX_CONFIDENCE_THRESHOLD = 0.70
MAX_BBOXES_PER_LABEL = 2
REQUIRE_HANDS_FOR_CHEST_COMPRESSIONS_FEEDBACK = False
HANDS_DETECTION_LABEL = "hands"
CHEST_COMPRESSIONS_ACTION_LABEL = "chest_compressions"
NOT_CONFIDENT_ACTION_FEEDBACK = "Not confident"


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
        self._last_published_action_feedback = None
        self._bbox_publisher = BBoxPublisher()
        self._feedback_publisher = FeedbackPublisher()
        print("[VideoMLClient] Initialized")

    def stop(self):
        print("[VideoMLClient] Stopping...")
        self.is_running = False
        self.quit()
        self.wait()
        self._bbox_publisher.close()
        self._feedback_publisher.close()
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

    def _build_bbox_boxes(self, detections, frame_shape):
        if frame_shape is None or len(frame_shape) < 2:
            return []

        frame_height, frame_width = frame_shape[:2]
        label_counts = {}
        boxes = []
        ranked_detections = sorted(
            detections or [],
            key=self._detection_score,
            reverse=True,
        )

        for det in ranked_detections:
            try:
                score_value = float(det.get("score", 0))
            except (TypeError, ValueError):
                continue

            if score_value < BBOX_CONFIDENCE_THRESHOLD:
                continue

            label_text = str(det.get("label") or "").strip()
            if not label_text:
                continue

            label_key = label_text.lower()
            if label_counts.get(label_key, 0) >= MAX_BBOXES_PER_LABEL:
                continue

            bbox_box = build_bbox_box(
                label=label_text,
                score=score_value,
                box_xyxy=det.get("box_xyxy", []),
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if bbox_box is None:
                continue

            boxes.append(bbox_box)
            label_counts[label_key] = label_counts.get(label_key, 0) + 1

        return boxes

    def _detection_score(self, detection):
        try:
            return float((detection or {}).get("score", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            return 0.0

    def _build_action_feedback(self, activity, bbox_boxes):
        if not isinstance(activity, dict):
            return ""

        score = activity.get("score", 0)
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            return ""

        if score_value < ACTIVITY_FEEDBACK_THRESHOLD:
            return NOT_CONFIDENT_ACTION_FEEDBACK

        label_text = str(activity.get("label") or "").strip()
        if not label_text:
            return ""

        if (
            REQUIRE_HANDS_FOR_CHEST_COMPRESSIONS_FEEDBACK
            and label_text.lower() == CHEST_COMPRESSIONS_ACTION_LABEL
            and not any(
                str(box.get("label") or "").strip().lower() == HANDS_DETECTION_LABEL
                for box in bbox_boxes
            )
        ):
            print(
                "[VideoMLClient] Suppressed chest_compressions feedback: "
                "no hands detected above bbox threshold"
            )
            return NOT_CONFIDENT_ACTION_FEEDBACK

        return format_feedback_text(label_text, score_value)

    def _publish_action_feedback(self, action_feedback):
        if action_feedback == self._last_published_action_feedback:
            return

        if self._feedback_publisher.publish_action(action_feedback):
            if action_feedback:
                print(f"[VideoMLClient] Published action feedback: {action_feedback}")
            else:
                print("[VideoMLClient] Cleared action feedback")
            self._last_published_action_feedback = action_feedback

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
            bbox_boxes = []
            act_line = "buffering..."
            action_feedback = ""
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
                bbox_boxes = self._build_bbox_boxes(detections, frame.shape)
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
                    action_feedback = self._build_action_feedback(activity, bbox_boxes)
                    act_line = action_feedback or "Not confident"
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
            if self._bbox_publisher.publish(self._frame_id, bbox_boxes):
                if bbox_boxes or self._frame_id % HEALTH_LOG_INTERVAL_FRAMES == 0:
                    print(
                        f"[VideoMLClient] Published bbox frame: "
                        f"frame_id={self._frame_id} boxes={len(bbox_boxes)}"
                    )
            self._publish_action_feedback(action_feedback)
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
