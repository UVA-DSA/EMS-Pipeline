# Process for streaming audio from pipe to headphones

import os
import time
from multiprocessing import Process, Queue, Value
import ctypes

from PyQt5.QtCore import QThread, pyqtSignal


def audio_player_process(pipe_path, running_flag):
    """
    Separate process that reads audio from pipe and plays through headphones.
    Args:
        pipe_path: Path to audio pipe (/tmp/emsaud)
        running_flag: Shared Value to control process lifecycle
    """
    # IMPORTANT: Import PyAudio INSIDE the process function, after fork
    import pyaudio

    print(f"[AudioProcess] Starting, PID: {os.getpid()}")

    # Audio configuration
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_SIZE = 1024  # Match your working code

    def read_exactly(pipe, n):
        """Read exactly n bytes from pipe."""
        data = b''
        while len(data) < n:
            chunk = pipe.read(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    frame_count = 0
    p = None
    stream = None
    audio_pipe = None

    try:
        print(f"[AudioProcess] Opening {pipe_path} for reading...")
        audio_pipe = open(pipe_path, 'rb', buffering=0)
        print(f"[AudioProcess] {pipe_path} opened!")

        # Initialize PyAudio (after fork, inside process)
        p = pyaudio.PyAudio()

        print("[AudioProcess] Using system default audio device")

        # Open stream - exactly like your working code
        stream = p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE
        )

        print("[AudioProcess] PyAudio stream opened successfully")
        print("[AudioProcess] Starting read loop...")

        last_debug = time.time()

        while running_flag.value:
            try:
                # Read audio data
                # Format: 4 bytes length + PCM int16 data
                # print("[AudioProcess] Reading length bytes...")  # Uncomment for verbose debug
                length_bytes = read_exactly(audio_pipe, 4)
                if length_bytes is None:
                    print("[AudioProcess] Audio pipe closed (read returned None)")
                    break

                audio_length = int.from_bytes(length_bytes, 'big')
                # print(f"[AudioProcess] Expecting {audio_length} bytes")  # Uncomment for verbose debug
                audio_data = read_exactly(audio_pipe, audio_length)

                if audio_data is None:
                    print("[AudioProcess] Failed to read audio data (read returned None)")
                    break

                # Play audio - exactly like your working code
                stream.write(audio_data, exception_on_underflow=False)

                frame_count += 1

                # Debug first frame
                if frame_count == 1:
                    print(f"[AudioProcess] ✓ First audio frame played ({audio_length} bytes)")

                # Debug every 100 frames (~6 seconds at 16kHz)
                if frame_count % 100 == 0:
                    elapsed = time.time() - last_debug
                    print(f"[AudioProcess] Played {frame_count} audio frames "
                          f"({audio_length} bytes, {elapsed:.1f}s elapsed)")
                    last_debug = time.time()

            except Exception as e:
                if running_flag.value:
                    print(f"[AudioProcess] Error in playback loop: {e}")
                    import traceback
                    traceback.print_exc()
                break

        print(f"[AudioProcess] Exited read loop (running_flag={running_flag.value}, frames={frame_count})")

    except Exception as e:
        print(f'[AudioProcess] Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup - exactly like your working code
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except:
                pass

        if p is not None:
            try:
                p.terminate()
            except:
                pass

        if audio_pipe is not None:
            try:
                audio_pipe.close()
            except:
                pass

    print(f"[AudioProcess] Exiting, played {frame_count} frames")


class AudioStreamManager:
    """
    Manager class for the audio playback process.
    Use this in GUI.py instead of AudioThread.
    """
    def __init__(self):
        self.PIPE_AUDIO = "/tmp/emsaud"

        # Use spawn instead of fork to avoid inherited audio state
        import multiprocessing as mp
        self.ctx = mp.get_context('spawn')

        # IMPORTANT: Create shared Value with spawn context
        self.running_flag = self.ctx.Value(ctypes.c_bool, True)

        # Player process with spawn context
        self.player_process = self.ctx.Process(
            target=audio_player_process,
            args=(self.PIPE_AUDIO, self.running_flag),
            daemon=True
        )

        print('[AudioStreamManager] Initialized (using spawn mode)')

    def start(self):
        """Start the audio player process."""
        print("[AudioStreamManager] Starting...")
        self.running_flag.value = True
        self.player_process.start()
        print(f"[AudioStreamManager] Started (Process PID: {self.player_process.pid})")

    def stop(self):
        """Stop the audio player process."""
        print("[AudioStreamManager] Stopping...")

        self.running_flag.value = False
        self.player_process.join(timeout=3)

        if self.player_process.is_alive():
            print("[AudioStreamManager] Process didn't stop, terminating...")
            self.player_process.terminate()
            self.player_process.join(timeout=1)

        print("[AudioStreamManager] Stopped")



# Process for speech recognition using Whisper streaming
#
# PyQt5 is intentionally NOT imported at module level.
# This file is imported by the spawned process, and PyQt5
# cannot be imported in a process without a display context.

import os
import time
import subprocess
import threading
from multiprocessing import Process, Queue, Value
import ctypes


# ─────────────────────────────────────────────────────────────────────────────
# Runs in spawned process - NO PyQt5 here
# ─────────────────────────────────────────────────────────────────────────────

def whisper_speech_process(audio_ml_pipe, transcript_fifo, transcript_queue, running_flag):
    """
    Speech recognition process that runs Whisper C++ streaming.

    Sequence:
      1. Create transcript FIFO
      2. Launch egosim_stream (it opens audio_ml_pipe for reading → unblocks SRT receiver)
      3. Open transcript FIFO in background thread (blocks until egosim_stream
         opens it for writing, which only happens after it gets audio data)
      4. Read transcript lines and push to queue
    """
    print(f"[WhisperProcess] Starting, PID: {os.getpid()}")

    whisper_proc = None
    line_count = 0

    # Holder for the transcript pipe, opened in a background thread
    transcript_pipe_holder = None
    transcript_open_done = threading.Event()

    try:
        # Create transcript FIFO
        if os.path.exists(transcript_fifo):
            os.unlink(transcript_fifo)
        os.mkfifo(transcript_fifo)
        print(f"[WhisperProcess] Created transcript FIFO: {transcript_fifo}")

        # mythread = threading.Thread(target=open_transcript_pipe, args=(transcript_fifo,), daemon=True)
        # mythread.start()

        # Open transcript FIFO immediately (non-blocking)
        import fcntl
        print(f"[WhisperProcess] Opening transcript FIFO...")
        fd = os.open(transcript_fifo, os.O_RDONLY | os.O_NONBLOCK)
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
        transcript_pipe_holder = os.fdopen(fd, 'r', buffering=1)
        print(f"[WhisperProcess] Transcript FIFO opened!")

        # Start Whisper C++ subprocess
        # egosim_stream will open audio_ml_pipe for reading immediately,
        # which unblocks the SRT receiver's open() on /tmp/emsaudml
        whisper_cmd = [
            './pipe_whisper_stream_2026/build/bin/egosim_stream',
            '--model', 'pipe_whisper_stream_2026/models/ggml-finetuned-base-v203.bin',
            '--fifo', audio_ml_pipe,
            '--text-fifo', transcript_fifo
        ]

        print(f"[WhisperProcess] Starting Whisper: {' '.join(whisper_cmd)}")

        # Set LD_LIBRARY_PATH to use conda's libffi (fixes wayland symbol errors)
        env = os.environ.copy()
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        if conda_prefix:
            env['LD_LIBRARY_PATH'] = f"{conda_prefix}/lib:{env.get('LD_LIBRARY_PATH', '')}"
            print(f"[WhisperProcess] Using conda libffi from: {conda_prefix}/lib")

        whisper_proc = subprocess.Popen(
            whisper_cmd,
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # stdout=None,
            # stderr=None,
            text=True,
            bufsize=1,
            env=env
        )
        print(f"[WhisperProcess] Whisper started (PID: {whisper_proc.pid})")

        # Check egosim_stream didn't crash immediately
        time.sleep(1.0)
        if whisper_proc.poll() is not None:
            stdout, stderr = whisper_proc.communicate()
            print(f"[WhisperProcess] ERROR: egosim_stream crashed immediately!")
            print(f"[WhisperProcess] stdout: {stdout}")
            print(f"[WhisperProcess] stderr: {stderr}")
            return

        # print('running flag value', running_flag.value)
        print(f"[WhisperProcess] egosim_stream still running after 1s - looks good")
        # print('running flag value', running_flag.value)

        # Signal the GUI that Whisper is up and ready
        try:
            transcript_queue.put_nowait("__WHISPER_READY__")
        except:
            pass

        # Open transcript FIFO in background thread so we don't block here.
        # egosim_stream will only open --text-fifo once it has processed
        # enough audio, so we must not block the process on this.


        # Read transcripts and push to GUI queue
        while running_flag.value:
            try:
                # Check if Whisper process is still alive
                if whisper_proc.poll() is not None:
                    print(f"[WhisperProcess] Whisper exited with code {whisper_proc.returncode}")
                    break

                # Wait for transcript pipe to be available
                if transcript_pipe_holder is None:
                    time.sleep(0.1)
                    continue

                line = transcript_pipe_holder.readline()

                if not line:
                    # print('here')
                    continue
                print('heres the line:', line)

                # if not line:
                #     print("[WhisperProcess] Transcript pipe closed")
                #     break

                line = line.strip()
                # if not line:
                #     continue

                line_count += 1

                # Send to GUI queue (non-blocking, drop oldest if full)
                try:
                    print('got this far')
                    transcript_queue.put_nowait(line)
                    print('got down here')
                except:
                    print('statement 1')
                    try:
                        print('statement 2')
                        transcript_queue.get_nowait()
                        print('statement 3')
                        transcript_queue.put_nowait(line)
                        print('statement 4')
                    except:
                        pass

                if line_count % 10 == 0:
                    print(f"[WhisperProcess] Processed {line_count} transcript lines")

            except Exception as e:
                if running_flag.value:
                    print(f"[WhisperProcess] Error reading transcript: {e}")
                    import traceback
                    traceback.print_exc()
                break

    except Exception as e:
        print(f'[WhisperProcess] Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        print("[WhisperProcess] Cleaning up...")

        if transcript_pipe_holder is not None:
            try:
                transcript_pipe_holder.close()
            except:
                pass

        if whisper_proc is not None:
            try:
                whisper_proc.terminate()
                whisper_proc.wait(timeout=3)
            except:
                try:
                    whisper_proc.kill()
                except:
                    pass

        try:
            if os.path.exists(transcript_fifo):
                os.unlink(transcript_fifo)
        except:
            pass

    print(f"[WhisperProcess] Exiting, processed {line_count} lines")


# ─────────────────────────────────────────────────────────────────────────────
# Runs in main process only - PyQt5 imported lazily here
# ─────────────────────────────────────────────────────────────────────────────

def _make_transcript_thread():
    """
    Factory that creates TranscriptDisplayThread with PyQt5.
    Imported lazily so the spawned whisper process never touches Qt.
    """
    from PyQt5.QtCore import QThread, pyqtSignal

    class TranscriptDisplayThread(QThread):
        transcript_ready = pyqtSignal(str)
        whisper_ready = pyqtSignal()

        def __init__(self, transcript_queue):
            super().__init__()
            self.transcript_queue = transcript_queue
            self.is_running = True
            print('[TranscriptDisplayThread] Initialized')

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
                except:
                    continue
            print("[TranscriptDisplayThread] Exiting run loop")

    return TranscriptDisplayThread


class SpeechProcessManager:
    def __init__(self):
        self.PIPE_AUDIO_ML = "/tmp/emsaudml"
        self.TRANSCRIPT_FIFO = "/tmp/egosim_transcript"

        import multiprocessing as mp
        self.ctx = mp.get_context('spawn')

        # Regular Queue/Value (not spawn context - avoids resource tracker blocking)
        self.transcript_queue = self.ctx.Queue(maxsize=50)
        self.running_flag = self.ctx.Value(ctypes.c_bool, True)

        self.whisper_process = self.ctx.Process(
            target=whisper_speech_process,
            args=(self.PIPE_AUDIO_ML, self.TRANSCRIPT_FIFO,
                  self.transcript_queue, self.running_flag),
            daemon=True
        )

        TranscriptDisplayThread = _make_transcript_thread()
        self.display_thread = TranscriptDisplayThread(self.transcript_queue)

        print('[SpeechProcessManager] Initialized')

    def start(self):
        print("[SpeechProcessManager] Starting...")
        self.running_flag.value = True
        self.whisper_process.start()
        self.display_thread.start()
        print(f"[SpeechProcessManager] Started (Process PID: {self.whisper_process.pid})")

    def stop(self):
        print("[SpeechProcessManager] Stopping...")

        # Kill egosim_stream first (nuclear option)
        import subprocess
        subprocess.run(['pkill', '-f', 'egosim_stream'], stderr=subprocess.DEVNULL)

        # Then stop the Python process
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
    Usage in GUI.py:
        self.SpeechProcess = SpeechProcess()
        self.SpeechProcess.transcript_ready.connect(self.TranscriptTextEdit.append)
        self.SpeechProcess.start()
    """
    def __init__(self):
        self.manager = SpeechProcessManager()
        self.transcript_ready = self.manager.transcript_ready

    def start(self):
        self.manager.start()

    def stop(self):
        self.manager.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Google Cloud Speech streaming manager
# Reads raw PCM int16 frames from /tmp/emsaudml (same pipe whisper uses)
# and streams them to Google Cloud Speech-to-Text v1 streaming API.
# Emits transcripts via transcript_ready signal, identical interface to
# SpeechProcessManager so GUI.py can swap between the two transparently.
# ─────────────────────────────────────────────────────────────────────────────

class GoogleSpeechStreamManager:
    """
    Drop-in replacement for SpeechProcessManager that uses Google Cloud STT.

    Prerequisites:
        pip install google-cloud-speech
        GOOGLE_APPLICATION_CREDENTIALS env var set to service-account JSON path.

    Usage in GUI.py:
        self.GoogleSpeechManager = GoogleSpeechStreamManager()
        self.GoogleSpeechManager.transcript_ready.connect(self.update_transcript_widget)
        self.GoogleSpeechManager.start()
        ...
        self.GoogleSpeechManager.stop()
    """

    PIPE_AUDIO_ML = "/tmp/emsaudml"
    SAMPLE_RATE   = 16000
    CHANNELS      = 1
    # Google STT streaming limit is ~5 minutes; we restart automatically
    STREAM_LIMIT_SECS = 240

    def __init__(self):
        from PyQt5.QtCore import QThread, pyqtSignal

        class _GoogleThread(QThread):
            transcript_ready = pyqtSignal(str)
            # No whisper_ready signal — Google is ready immediately
            whisper_ready = pyqtSignal()

            def __init__(self):
                super().__init__()
                self._running = True
                print('[GoogleSpeechThread] Initialized')

            def stop(self):
                print('[GoogleSpeechThread] Stopping...')
                self._running = False
                self.quit()
                self.wait(3000)
                print('[GoogleSpeechThread] Stopped')

            def run(self):
                print('[GoogleSpeechThread] Started')
                try:
                    from google.cloud import speech
                    from google.api_core.exceptions import OutOfRange, ServiceUnavailable
                except ImportError:
                    print('[GoogleSpeechThread] ERROR: google-cloud-speech not installed. '
                          'Run: pip install google-cloud-speech')
                    return

                client = speech.SpeechClient()

                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code='en-US',
                    enable_automatic_punctuation=True,
                    model='latest_long',
                )
                streaming_config = speech.StreamingRecognitionConfig(
                    config=config,
                    interim_results=False,
                )

                # ── Wait for the pipe to exist (SRT receiver creates it on Start) ──
                print('[GoogleSpeechThread] Waiting for audio pipe to become available...')
                while self._running:
                    if os.path.exists(GoogleSpeechStreamManager.PIPE_AUDIO_ML):
                        break
                    time.sleep(0.25)
                if not self._running:
                    print('[GoogleSpeechThread] Stopped before pipe appeared')
                    return

                print('[GoogleSpeechThread] Opening audio pipe...')
                try:
                    pipe = open(GoogleSpeechStreamManager.PIPE_AUDIO_ML, 'rb', buffering=0)
                except Exception as e:
                    print(f'[GoogleSpeechThread] Failed to open pipe: {e}')
                    return
                print('[GoogleSpeechThread] Audio pipe open — Google STT ready')
                self.whisper_ready.emit()

                def _read_exactly(n):
                    buf = b''
                    while len(buf) < n and self._running:
                        chunk = pipe.read(n - len(buf))
                        if not chunk:
                            return None
                        buf += chunk
                    return buf if self._running else None

                def _make_audio_generator():
                    """
                    Returns a fresh generator of StreamingRecognizeRequests.
                    Call this for each new streaming session — a generator can
                    only be iterated once.
                    Frame format: 4-byte big-endian length + PCM int16 data,
                    same as written by the SRT receiver.
                    """
                    while self._running:
                        hdr = _read_exactly(4)
                        if hdr is None:
                            return
                        length = int.from_bytes(hdr, 'big')
                        pcm = _read_exactly(length)
                        if pcm is None:
                            return
                        yield speech.StreamingRecognizeRequest(audio_content=pcm)

                try:
                    while self._running:
                        print('[GoogleSpeechThread] Starting streaming session...')
                        try:
                            responses = client.streaming_recognize(
                                streaming_config,
                                _make_audio_generator(),
                            )
                            for response in responses:
                                if not self._running:
                                    break
                                for result in response.results:
                                    if result.is_final:
                                        text = result.alternatives[0].transcript.strip()
                                        if text:
                                            print(f'[GoogleSpeechThread] Transcript: {text}')
                                            self.transcript_ready.emit(text)

                        except OutOfRange as e:
                            # Google sends this when audio timeout occurs (no audio
                            # received close to real time). Safe to restart session.
                            if self._running:
                                print('[GoogleSpeechThread] Audio timeout from Google — '
                                      'restarting session...')
                            continue
                        except ServiceUnavailable as e:
                            if self._running:
                                print(f'[GoogleSpeechThread] Service unavailable, '
                                      f'retrying in 2s: {e}')
                                time.sleep(2)
                            continue
                        except Exception as e:
                            if self._running:
                                print(f'[GoogleSpeechThread] Streaming error: {e}')
                                import traceback
                                traceback.print_exc()
                                time.sleep(1)
                            continue

                        if self._running:
                            print('[GoogleSpeechThread] Session ended cleanly, restarting...')

                finally:
                    try:
                        pipe.close()
                    except:
                        pass

                print('[GoogleSpeechThread] Exiting')

        # Patch SAMPLE_RATE into the inner class so it can reference it
        _GoogleThread.SAMPLE_RATE = self.SAMPLE_RATE

        self._thread = _GoogleThread()
        print('[GoogleSpeechStreamManager] Initialized')

    # ── public API (mirrors SpeechProcessManager) ──────────────────────────

    def start(self):
        print('[GoogleSpeechStreamManager] Starting...')
        self._thread._running = True
        self._thread.start()
        print('[GoogleSpeechStreamManager] Started')

    def stop(self):
        print('[GoogleSpeechStreamManager] Stopping...')
        self._thread.stop()
        print('[GoogleSpeechStreamManager] Stopped')

    @property
    def transcript_ready(self):
        return self._thread.transcript_ready

    @property
    def whisper_ready(self):
        """Emitted once the pipe is open and Google STT is streaming."""
        return self._thread.whisper_ready