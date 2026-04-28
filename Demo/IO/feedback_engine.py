import asyncio
import json
import logging
import os
import threading


LOGGER = logging.getLogger(__name__)

ACTION_CATEGORY = "action"
PROTOCOL_CATEGORY = "protocol"
VALID_CATEGORIES = {ACTION_CATEGORY, PROTOCOL_CATEGORY}
DEFAULT_FEEDBACK_ENGINE_PATH = "/tmp/ems_feedback.log"
POLL_INTERVAL_SECONDS = 0.1


def _trace(message):
    pass
    # print(f"[FeedbackEngine] {message}")


def get_feedback_engine_path():
    return (
        os.environ.get("FEEDBACK_ENGINE_PATH")
        or os.environ.get("FEEDBACK_SOCKET_PATH")
        or DEFAULT_FEEDBACK_ENGINE_PATH
    )


def reset_feedback_engine(path=None):
    engine_path = path or get_feedback_engine_path()
    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    with open(engine_path, "wb"):
        pass
    _trace(f"reset log at {engine_path}")


def normalize_feedback_value(value):
    if value is None:
        return ""
    return str(value).strip()


def format_feedback_text(label, confidence):
    label_text = normalize_feedback_value(label)
    if not label_text:
        return ""

    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        return label_text

    return f"{label_text} ({confidence_value:.2f})"


def build_feedback_payload(action="", protocol=""):
    feedback = {}
    action_text = normalize_feedback_value(action)
    protocol_text = normalize_feedback_value(protocol)
    if action_text:
        feedback[ACTION_CATEGORY] = action_text
    if protocol_text:
        feedback[PROTOCOL_CATEGORY] = protocol_text

    return {
        "type": "feedback",
        "feedback": feedback,
    }


def initial_feedback_state():
    return {
        ACTION_CATEGORY: "",
        PROTOCOL_CATEGORY: "",
    }


def build_feedback_update(category, value):
    category_text = normalize_feedback_value(category)
    if category_text not in VALID_CATEGORIES:
        raise ValueError(f"Unsupported feedback category: {category}")

    return {
        "category": category_text,
        "value": normalize_feedback_value(value),
    }


def decode_feedback_update(raw_payload):
    try:
        message = json.loads(raw_payload.decode("utf-8"))
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    category = normalize_feedback_value(message.get("category"))
    if category not in VALID_CATEGORIES:
        return None

    return {
        "category": category,
        "value": normalize_feedback_value(message.get("value")),
    }


class FeedbackPublisher:
    def __init__(self, path=None):
        self.path = path or get_feedback_engine_path()
        self._fd = None
        self._lock = threading.Lock()
        _trace(f"publisher initialized with path={self.path}")

    def publish(self, category, value):
        update = build_feedback_update(category, value)
        encoded = (json.dumps(update) + "\n").encode("utf-8")

        with self._lock:
            _trace(
                f"publish requested category={update['category']!r} "
                f"value={update['value']!r}"
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
                LOGGER.debug(
                    "Feedback publish skipped for %s (%s): %s",
                    update["category"],
                    self.path,
                    exc,
                )
                _trace(
                    f"publish failed for category={update['category']!r} "
                    f"value={update['value']!r}: {exc}"
                )
                return False

            _trace(
                f"publish wrote category={update['category']!r} "
                f"value={update['value']!r} to {self.path}"
            )
            return True

    def publish_action(self, value):
        return self.publish(ACTION_CATEGORY, value)

    def publish_protocol(self, value):
        return self.publish(PROTOCOL_CATEGORY, value)

    def close(self):
        with self._lock:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                finally:
                    self._fd = None
                _trace(f"publisher closed fd for {self.path}")


class FeedbackBroker:
    def __init__(self, path=None, poll_interval_seconds=POLL_INTERVAL_SECONDS):
        self.path = path or get_feedback_engine_path()
        self.poll_interval_seconds = poll_interval_seconds
        self._task = None
        self._condition = asyncio.Condition()
        self._running = False
        self._version = 0
        self._offset = 0
        self._buffer = b""
        self._latest_feedback = initial_feedback_state()
        _trace(
            f"broker initialized path={self.path} "
            f"poll_interval={self.poll_interval_seconds}"
        )

    @property
    def current_version(self):
        return self._version

    @property
    def current_feedback(self):
        return dict(self._latest_feedback)

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
        LOGGER.info("Feedback broker watching %s", self.path)
        _trace(
            f"broker started version={self._version} "
            f"current_feedback={self._latest_feedback}"
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

        async with self._condition:
            self._condition.notify_all()
        _trace(
            f"broker stopped version={self._version} "
            f"current_feedback={self._latest_feedback}"
        )

    async def wait_for_update(self, after_version, timeout=None):
        async def _wait():
            async with self._condition:
                await self._condition.wait_for(
                    lambda: self._version > after_version or not self._running
                )
                if self._version <= after_version:
                    return None
                return self._version, dict(self._latest_feedback)

        if timeout is None:
            return await _wait()

        try:
            return await asyncio.wait_for(_wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

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
            LOGGER.info("Feedback broker stopped: %s", exc)
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
            _trace(
                f"broker buffered partial line of {len(self._buffer)} bytes"
            )

        for line in lines:
            clean_line = line.rstrip(b"\n")
            _trace(f"broker processing raw line={clean_line!r}")
            update = decode_feedback_update(clean_line)
            if update is None:
                _trace("broker ignored invalid feedback line")
                continue

            feedback = dict(self._latest_feedback)
            feedback[update["category"]] = update["value"]
            _trace(
                f"broker decoded update category={update['category']!r} "
                f"value={update['value']!r}"
            )

            async with self._condition:
                self._latest_feedback = feedback
                self._version += 1
                self._condition.notify_all()
                _trace(
                    f"broker applied update version={self._version} "
                    f"current_feedback={self._latest_feedback}"
                )

    def _ensure_log_file(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "ab"):
            pass
        _trace(f"ensured log file exists at {self.path}")
