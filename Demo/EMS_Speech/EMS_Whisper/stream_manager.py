import ctypes
import os
import re
import subprocess
import time
from pathlib import Path

from Utils import pipeline_config
from Utils.runtime_paths import demo_path


# ---------------------------------------------------------------------------
# Number normalization
# Whisper output is inconsistent: "5 compressions" vs "five compressions".
# Ground truth writes 1-9 as words and >=10 as digits, so we normalise both
# directions to match that convention before writing to the transcript log.
# ---------------------------------------------------------------------------

_DIGIT_TO_WORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}

_WORD_TO_DIGIT = {v: k for k, v in _DIGIT_TO_WORD.items()}
# Also handle ordinals that Whisper sometimes emits
_ORDINAL_TO_DIGIT = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9",
}

def _normalize_numbers(text: str) -> str:
    """
    Normalise number tokens in *text* to match ground-truth conventions:
      • Standalone digit 0-9  →  word  ("5" → "five")
      • Spoken word for 0-9  →  word  (already correct; kept for safety)
      • Digit ≥ 10           →  kept as digit  ("30" → "30")
      • Ordinals first-ninth →  digit ("third" → "3")
    Punctuation attached to a token is preserved ("five," stays "five,").
    """
    tokens = text.split()
    result = []
    for token in tokens:
        # Strip leading/trailing punctuation for lookup, preserve it
        stripped = token.strip(".,!?;:\"'")
        punct_lead = token[: len(token) - len(token.lstrip(".,!?;:\"'"))]
        punct_trail = token[len(stripped) + len(punct_lead):]

        lower = stripped.lower()

        if re.fullmatch(r'\d+', stripped):
            n = int(stripped)
            if n <= 9:
                # Single digit → word
                replacement = _DIGIT_TO_WORD[stripped]
            else:
                # ≥ 10 → keep as digit
                replacement = stripped
        elif lower in _WORD_TO_DIGIT:
            # Already a word for 0-9 — keep it (ground truth uses words)
            replacement = lower
        elif lower in _ORDINAL_TO_DIGIT:
            # "third" → "3"
            replacement = _ORDINAL_TO_DIGIT[lower]
        else:
            replacement = stripped  # unchanged

        # Re-attach punctuation, preserving original capitalisation for non-number words
        if replacement == stripped.lower() and not replacement.isdigit():
            # Preserve original capitalisation
            if stripped[0].isupper() and len(stripped) > 1:
                replacement = replacement[0].upper() + replacement[1:]
        result.append(punct_lead + replacement + punct_trail)

    return " ".join(result)


def _fmt_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS.mmm to match ground-truth Start/End fields."""
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    secs    = seconds - minutes * 60
    return f"{minutes:02d}:{secs:06.3f}"


def _resolve_int_env(var_names, default_value):
    for var_name in var_names:
        raw_value = os.environ.get(var_name)
        if raw_value in (None, ""):
            continue
        try:
            return int(raw_value)
        except ValueError:
            print(
                f"[WhisperConfig] Invalid integer for {var_name}: {raw_value!r}. "
                f"Using default {default_value}."
            )
    return int(default_value)


def _resolve_whisper_assets():
    """
    Resolve the Whisper stream binary/model and runtime parameters from env
    overrides first, then from repo-local defaults.
    """
    default_binary_path = demo_path(
        "EMS_Speech",
        "EMS_Whisper",
        "whisper.cpp_realtime_stream",
        "build",
        "bin",
        "egosim_stream",
    )
    default_model_path = demo_path(
        "EMS_Speech",
        "EMS_Whisper",
        "whisper.cpp_realtime_stream",
        "models",
        "ggml-finetuned-base-v203.bin",
    )

    binary_path = Path(
        os.environ.get(
            "EMS_WHISPER_STREAM_BIN",
            default_binary_path,
        )
    )
    model_path = Path(
        os.environ.get(
            "EMS_WHISPER_MODEL",
            default_model_path,
        )
    )
    whisper_params = {
        "step": _resolve_int_env(
            ["EMS_WHISPER_STEP"],
            getattr(pipeline_config, "step", 2000),
        ),
        "length": _resolve_int_env(
            ["EMS_WHISPER_MAX_LENGTH", "EMS_WHISPER_LENGTH"],
            getattr(pipeline_config, "length", 4000),
        ),
        "keep_ms": _resolve_int_env(
            ["EMS_WHISPER_KEEP_MS", "EMS_WHISPER_KEEP"],
            getattr(pipeline_config, "keep_ms", 200),
        ),
    }

    return binary_path, model_path, whisper_params


def whisper_speech_process(audio_ml_pipe, transcript_fifo, transcript_queue, running_flag):
    """
    Speech recognition process that runs Whisper C++ streaming.

    Sequence:
      1. Create the transcript FIFO.
      2. Launch `egosim_stream`, which opens `audio_ml_pipe` for reading.
      3. Read transcript lines from the FIFO.
      4. Push transcript lines back to the GUI queue.
    """
    print(f"[WhisperProcess] Starting, PID: {os.getpid()}")

    whisper_proc = None
    transcript_pipe = None
    line_count = 0

    try:
        binary_path, model_path, whisper_params = _resolve_whisper_assets()
        if not binary_path.exists():
            raise FileNotFoundError(f"Whisper stream binary not found: {binary_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Whisper model not found: {model_path}")

        if os.path.exists(transcript_fifo):
            os.unlink(transcript_fifo)
        os.mkfifo(transcript_fifo)
        print(f"[WhisperProcess] Created transcript FIFO: {transcript_fifo}")

        import fcntl

        print("[WhisperProcess] Opening transcript FIFO...")
        fd = os.open(transcript_fifo, os.O_RDONLY | os.O_NONBLOCK)
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
        transcript_pipe = os.fdopen(fd, "r", buffering=1)
        print("[WhisperProcess] Transcript FIFO opened!")

        whisper_cmd = [
            str(binary_path),
            "--model",
            str(model_path),
            "--fifo",
            audio_ml_pipe,
            "--text-fifo",
            transcript_fifo,
            "--step",
            str(whisper_params["step"]),
            "--length",
            str(whisper_params["length"]),
            "--keep",
            str(whisper_params["keep_ms"]),
        ]
        print(f"[WhisperProcess] Starting Whisper: {' '.join(whisper_cmd)}")

        env = os.environ.copy()
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix:
            env["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib:{env.get('LD_LIBRARY_PATH', '')}"
            print(f"[WhisperProcess] Using conda libffi from: {conda_prefix}/lib")

        whisper_proc = subprocess.Popen(
            whisper_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
        )
        print(f"[WhisperProcess] Whisper started (PID: {whisper_proc.pid})")

        time.sleep(1.0)
        if whisper_proc.poll() is not None:
            raise RuntimeError(
                f"egosim_stream exited immediately with code {whisper_proc.returncode}"
            )

        print("[WhisperProcess] egosim_stream still running after 1s - looks good")

        try:
            transcript_queue.put_nowait("__WHISPER_READY__")
        except Exception:
            pass

        # t=0 for transcript timestamps is the moment Whisper signals ready,
        # which is when egosim_stream opens the audio FIFO and starts consuming.
        # All subsequent lines are stamped relative to this anchor.
        t_ready = time.time()

        # Open timestamped transcript log.
        # Format per line:  <MM:SS.mmm>\t<normalized_text>
        # matching the MM:SS.mmm convention used in the ground-truth JSON.
        _log_dir = "/tmp/ems_session_logs"
        os.makedirs(_log_dir, exist_ok=True)
        _session_ts = time.strftime('%Y%m%d_%H%M%S')
        _transcript_path = f"{_log_dir}/transcript_{_session_ts}.tsv"
        _transcript_log = open(_transcript_path, "w", buffering=1)
        # TSV header matches ground-truth field names for easy comparison
        _transcript_log.write("start\tend\tutterance\n")
        print(f"[WhisperProcess] Logging timestamped transcript to: {_transcript_path}")

        # egosim_stream emits one segment per step window.  We record the
        # wall-clock interval [t_line_start, t_line_end] for each segment,
        # where t_line_start = time of previous readline() return (or t_ready)
        # and t_line_end = time this readline() returned.  This gives a coarse
        # but usable alignment with ground-truth Start/End fields.
        t_prev_line = t_ready

        while running_flag.value:
            try:
                if whisper_proc.poll() is not None:
                    print(f"[WhisperProcess] Whisper exited with code {whisper_proc.returncode}")
                    break

                line = transcript_pipe.readline()
                t_line = time.time()

                if not line:
                    continue

                line_count += 1

                raw_text = line.strip()
                if not raw_text:
                    t_prev_line = t_line
                    continue

                # ----------------------------------------------------------
                # Timestamp extraction.
                #
                # NEW binary (egosim_stream patched to emit "T=<ms>\t<text>"):
                #   The prefix is wall-clock ms since egosim_stream's t_start,
                #   which is anchored to just before the main audio loop — very
                #   close to when the audio FIFO is first consumed.  t1 is the
                #   end of the step window; t0 = t1 - step_ms.
                #
                # OLD binary (no prefix):
                #   Fall back to Python wall-clock relative to t_ready, same
                #   as before.  Less accurate but still usable.
                # ----------------------------------------------------------
                if raw_text.startswith("T=") and "\t" in raw_text:
                    ts_part, text_part = raw_text.split("\t", 1)
                    t1_s   = int(ts_part[2:]) / 1000.0
                    t0_s   = max(0.0, t1_s - whisper_params["step"] / 1000.0)
                else:
                    # Fallback: wall-clock interval
                    t0_s   = t_prev_line - t_ready
                    t1_s   = t_line      - t_ready
                    text_part = raw_text

                t0_str = _fmt_timestamp(t0_s)
                t1_str = _fmt_timestamp(t1_s)

                # Normalize number tokens to match ground-truth convention
                normalized = _normalize_numbers(text_part)

                # Write TSV line: start\tend\tutterance
                _transcript_log.write(f"{t0_str}\t{t1_str}\t{normalized}\n")

                t_prev_line = t_line

                # Forward normalized text to GUI queue
                try:
                    transcript_queue.put_nowait(normalized)
                except Exception:
                    try:
                        transcript_queue.get_nowait()
                        transcript_queue.put_nowait(normalized)
                    except Exception:
                        pass

                if line_count % 10 == 0:
                    print(f"[WhisperProcess] Processed {line_count} transcript lines")
            except Exception as exc:
                if running_flag.value:
                    print(f"[WhisperProcess] Error reading transcript: {exc}")
                break

        _transcript_log.close()
    except Exception as exc:
        print(f"[WhisperProcess] Error: {exc}")
    finally:
        print("[WhisperProcess] Cleaning up...")

        if transcript_pipe is not None:
            try:
                transcript_pipe.close()
            except Exception:
                pass

        if whisper_proc is not None:
            try:
                whisper_proc.terminate()
                whisper_proc.wait(timeout=3)
            except Exception:
                try:
                    whisper_proc.kill()
                except Exception:
                    pass

        try:
            if os.path.exists(transcript_fifo):
                os.unlink(transcript_fifo)
        except Exception:
            pass

    print(f"[WhisperProcess] Exiting, processed {line_count} lines")


def _make_transcript_thread():
    """
    Create the Qt transcript bridge lazily so the spawned process never touches
    PyQt imports.
    """
    from PyQt5.QtCore import QThread, pyqtSignal

    class TranscriptDisplayThread(QThread):
        transcript_ready = pyqtSignal(str)
        whisper_ready = pyqtSignal()

        def __init__(self, transcript_queue):
            super().__init__()
            self.transcript_queue = transcript_queue
            self.is_running = True
            print("[TranscriptDisplayThread] Initialized")

        def stop(self):
            print("[TranscriptDisplayThread] Stopping...")
            self.is_running = False
            self.quit()
            self.wait()
            print("[TranscriptDisplayThread] Stopped")

        def run(self):
            print("[TranscriptDisplayThread] Started")
            while self.is_running:
                try:
                    transcript = self.transcript_queue.get(timeout=0.1)
                    if transcript == "__WHISPER_READY__":
                        self.whisper_ready.emit()
                    else:
                        self.transcript_ready.emit(transcript)
                except Exception:
                    continue
            print("[TranscriptDisplayThread] Exiting run loop")

    return TranscriptDisplayThread


class SpeechProcessManager:
    """
    Manager used by the GUI to launch the Whisper pipe reader and surface
    transcripts through Qt signals.
    """

    def __init__(self):
        self.pipe_audio_ml = "/tmp/emsaudml"
        self.transcript_fifo = "/tmp/egosim_transcript"

        import multiprocessing as mp

        self.ctx = mp.get_context("spawn")
        self.transcript_queue = self.ctx.Queue(maxsize=50)
        self.running_flag = self.ctx.Value(ctypes.c_bool, True)
        self.whisper_process = self.ctx.Process(
            target=whisper_speech_process,
            args=(
                self.pipe_audio_ml,
                self.transcript_fifo,
                self.transcript_queue,
                self.running_flag,
            ),
            daemon=True,
        )

        transcript_thread = _make_transcript_thread()
        self.display_thread = transcript_thread(self.transcript_queue)
        print("[SpeechProcessManager] Initialized")

    def start(self):
        print("[SpeechProcessManager] Starting...")
        self.running_flag.value = True
        self.whisper_process.start()
        self.display_thread.start()
        print(f"[SpeechProcessManager] Started (Process PID: {self.whisper_process.pid})")

    def stop(self):
        print("[SpeechProcessManager] Stopping...")

        subprocess.run(["pkill", "-f", "egosim_stream"], stderr=subprocess.DEVNULL)

        self.running_flag.value = False
        self.whisper_process.join(timeout=2)

        if self.whisper_process.is_alive():
            self.whisper_process.terminate()
            self.whisper_process.join(timeout=1)

        if self.whisper_process.is_alive():
            self.whisper_process.kill()

        self.display_thread.stop()
        print("[SpeechProcessManager] Stopped")

    @property
    def transcript_ready(self):
        return self.display_thread.transcript_ready

    @property
    def whisper_ready(self):
        return self.display_thread.whisper_ready


class SpeechProcess:
    """
    Small convenience wrapper preserved for the old GUI interface.
    """

    def __init__(self):
        self.manager = SpeechProcessManager()
        self.transcript_ready = self.manager.transcript_ready
        self.whisper_ready = self.manager.whisper_ready

    def start(self):
        self.manager.start()

    def stop(self):
        self.manager.stop()


__all__ = [
    "SpeechProcess",
    "SpeechProcessManager",
    "_resolve_whisper_assets",
    "whisper_speech_process",
]