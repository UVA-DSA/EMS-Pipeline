import asyncio
import json
import logging
import os
import threading
import time


LOGGER = logging.getLogger(__name__)

DEFAULT_BBOX_ENGINE_PATH = "/tmp/ems_bbox.log"
POLL_INTERVAL_SECONDS = 0.1
EXCLUDED_BBOX_LABELS = {
    "bp monitor",
}


def _trace(message):
    #print(f"[BBoxEngine] {message}")
    pass


def get_bbox_engine_path():
    return os.environ.get("BBOX_ENGINE_PATH") or DEFAULT_BBOX_ENGINE_PATH


def reset_bbox_engine(path=None):
    engine_path = path or get_bbox_engine_path()
    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    with open(engine_path, "wb"):
        pass
    _trace(f"reset log at {engine_path}")


def normalize_bbox_label(value):
    if value is None:
        return ""

    label_text = str(value).strip()
    if not label_text:
        return ""
    return label_text


def is_excluded_bbox_label(value):
    label_text = normalize_bbox_label(value)
    if not label_text:
        return False
    return label_text.lower() in EXCLUDED_BBOX_LABELS


def normalize_bbox_score(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_bbox_box(label, score, box_xyxy, frame_width, frame_height):
    label_text = normalize_bbox_label(label)
    if not label_text:
        return None
    if is_excluded_bbox_label(label_text):
        return None

    try:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        frame_width_value = float(frame_width)
        frame_height_value = float(frame_height)
    except (TypeError, ValueError):
        return None

    if frame_width_value <= 0 or frame_height_value <= 0:
        return None

    x1 = max(0.0, min(x1, frame_width_value))
    y1 = max(0.0, min(y1, frame_height_value))
    x2 = max(0.0, min(x2, frame_width_value))
    y2 = max(0.0, min(y2, frame_height_value))
    if x2 <= x1 or y2 <= y1:
        return None

    return {
        "x": round(x1 / frame_width_value, 3),
        "y": round(y1 / frame_height_value, 3),
        "w": round((x2 - x1) / frame_width_value, 3),
        "h": round((y2 - y1) / frame_height_value, 3),
        "label": label_text,
        "score": round(normalize_bbox_score(score), 3),
    }


def build_bbox_payload(frame_id, boxes, timestamp_ms=None):
    try:
        frame_id_value = int(frame_id)
    except (TypeError, ValueError):
        frame_id_value = 0

    if timestamp_ms is None:
        timestamp_ms_value = int(time.time() * 1000)
    else:
        try:
            timestamp_ms_value = int(timestamp_ms)
        except (TypeError, ValueError):
            timestamp_ms_value = int(time.time() * 1000)

    normalized_boxes = []
    for box in boxes or []:
        if not isinstance(box, dict):
            continue
        label_text = normalize_bbox_label(box.get("label"))
        if not label_text:
            continue
        if is_excluded_bbox_label(label_text):
            continue
        try:
            normalized_box = {
                "x": float(box.get("x", 0.0)),
                "y": float(box.get("y", 0.0)),
                "w": float(box.get("w", 0.0)),
                "h": float(box.get("h", 0.0)),
                "label": label_text,
                "score": float(box.get("score", 0.0)),
            }
        except (TypeError, ValueError):
            continue
        normalized_boxes.append(normalized_box)

    return {
        "type": "bbox",
        "frameId": frame_id_value,
        "timestampMs": timestamp_ms_value,
        "boxes": normalized_boxes,
    }


def decode_bbox_payload(raw_payload):
    try:
        message = json.loads(raw_payload.decode("utf-8"))
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    if message.get("type") != "bbox":
        return None

    return build_bbox_payload(
        frame_id=message.get("frameId", 0),
        boxes=message.get("boxes", []),
        timestamp_ms=message.get("timestampMs"),
    )


class BBoxPublisher:
    def __init__(self, path=None):
        self.path = path or get_bbox_engine_path()
        self._fd = None
        self._lock = threading.Lock()
        _trace(f"publisher initialized with path={self.path}")

    def publish(self, frame_id, boxes, timestamp_ms=None):
        payload = build_bbox_payload(
            frame_id=frame_id,
            boxes=boxes,
            timestamp_ms=timestamp_ms,
        )
        encoded = (json.dumps(payload) + "\n").encode("utf-8")

        with self._lock:
            _trace(
                f"publish requested frame_id={payload['frameId']} "
                f"boxes={len(payload['boxes'])}"
            )

            try:
                if self._fd is None:
                    self._fd = os.open(
                        self.path,
                        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                        0o666,
                    )
                    _trace(f"opened publisher fd for {self.path}")
                os.write(self._fd, encoded)
            except OSError as exc:
                LOGGER.debug("BBox publish skipped for %s: %s", self.path, exc)
                _trace(
                    f"publish failed frame_id={payload['frameId']} "
                    f"boxes={len(payload['boxes'])}: {exc}"
                )
                return False

            _trace(
                f"publish wrote frame_id={payload['frameId']} "
                f"boxes={len(payload['boxes'])} to {self.path}"
            )
            return True

    def close(self):
        with self._lock:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                finally:
                    self._fd = None
                _trace(f"publisher closed fd for {self.path}")


class BBoxBroker:
    def __init__(self, path=None, poll_interval_seconds=POLL_INTERVAL_SECONDS):
        self.path = path or get_bbox_engine_path()
        self.poll_interval_seconds = poll_interval_seconds
        self._task = None
        self._running = False
        self._version = 0
        self._offset = 0
        self._buffer = b""
        self._latest_payload = None
        _trace(
            f"broker initialized path={self.path} "
            f"poll_interval={self.poll_interval_seconds}"
        )

    @property
    def current_version(self):
        return self._version

    @property
    def current_payload(self):
        if self._latest_payload is None:
            return None
        return dict(self._latest_payload)

    async def start(self):
        if self._running:
            _trace("broker start requested but already running")
            return

        self._ensure_log_file()
        self._offset = 0
        self._buffer = b""
        self._running = True
        _trace(f"broker starting on {self.path}")
        await self._consume_new_updates()
        self._task = asyncio.create_task(self._receive_loop())
        LOGGER.info("BBox broker watching %s", self.path)
        _trace(
            f"broker started version={self._version} "
            f"current_payload={self._latest_payload}"
        )

    async def stop(self):
        if not self._running and self._task is None:
            _trace("broker stop requested but already stopped")
            return

        self._running = False
        _trace("broker stopping")
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            finally:
                self._task = None

        _trace(
            f"broker stopped version={self._version} "
            f"current_payload={self._latest_payload}"
        )

    async def _receive_loop(self):
        try:
            _trace("broker receive loop started")
            while self._running:
                await self._consume_new_updates()
                await asyncio.sleep(self.poll_interval_seconds)
        except asyncio.CancelledError:
            _trace("broker receive loop cancelled")
            raise
        except Exception as exc:
            LOGGER.info("BBox broker stopped: %s", exc)
            _trace(f"broker receive loop error: {exc}")

    async def _consume_new_updates(self):
        try:
            file_size = os.path.getsize(self.path)
        except OSError:
            _trace(f"broker could not stat log file {self.path}")
            return

        if file_size < self._offset:
            _trace(
                f"log truncated or reset detected: file_size={file_size} "
                f"offset={self._offset}; rewinding"
            )
            self._offset = 0
            self._buffer = b""

        if file_size == self._offset:
            return

        with open(self.path, "rb") as handle:
            handle.seek(self._offset)
            chunk = handle.read()

        self._offset += len(chunk)
        if not chunk:
            return
        _trace(
            f"broker read {len(chunk)} bytes from {self.path}; "
            f"new_offset={self._offset}"
        )

        self._buffer += chunk
        lines = self._buffer.splitlines(keepends=True)
        self._buffer = b""
        if lines and not lines[-1].endswith(b"\n"):
            self._buffer = lines.pop()
            _trace(f"broker buffered partial line of {len(self._buffer)} bytes")

        for line in lines:
            clean_line = line.rstrip(b"\n")
            _trace(f"broker processing raw line={clean_line!r}")
            payload = decode_bbox_payload(clean_line)
            if payload is None:
                _trace("broker ignored invalid bbox line")
                continue

            self._latest_payload = payload
            self._version += 1
            _trace(
                f"broker applied update version={self._version} "
                f"frame_id={payload['frameId']} boxes={len(payload['boxes'])}"
            )

    def _ensure_log_file(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "ab"):
            pass
        _trace(f"ensured log file exists at {self.path}")
