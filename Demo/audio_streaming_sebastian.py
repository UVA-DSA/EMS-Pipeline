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

class SpeechThread(QThread):
    """
    Placeholder for speech recognition.
    Currently does nothing but maintains compatibility.
    """
    def __init__(self):
        super().__init__()
        print('[SpeechThread] Initialized (placeholder)')

    def run(self):
        pass

    def stop(self):
        self.quit()