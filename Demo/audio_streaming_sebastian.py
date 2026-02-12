# Thread for streaming audio from pipe to headphones

import os
import time
import threading
import numpy as np
import pyaudio

from PyQt5.QtCore import QThread, pyqtSignal


class AudioThread(QThread):
    """
    Thread that reads audio data from /tmp/emsaud pipe and plays it through headphones.
    Audio format: raw PCM int16 samples at 16000 Hz, mono
    """

    audio_status = pyqtSignal(str)  # Signal to update status in GUI if needed

    def __init__(self):
        super().__init__()

        self.PIPE_AUDIO = "/tmp/emsaud"
        self.audio_pipe_holder = None
        self.is_running = True

        # Audio configuration (must match server settings)
        self.SAMPLE_RATE = 16000
        self.CHANNELS = 1
        self.CHUNK_SIZE = 1024  # Frames per buffer

        # PyAudio setup
        self.p = pyaudio.PyAudio()
        self.stream = None

        print('[AudioThread] Initialized')

    def read_exactly(self, pipe, n):
        """Read exactly n bytes from pipe."""
        data = b''
        while len(data) < n:
            chunk = pipe.read(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def stop(self):
        """Stop the audio streaming thread."""
        print("[INFO] Audio Thread stopping...")
        self.is_running = False

        # Stop and close PyAudio stream
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()

        if self.p is not None:
            self.p.terminate()

        self.quit()
        print("[INFO] Audio Thread Stopped")

    def run(self):
        """Main thread loop - reads from pipe and plays audio."""
        is_connected = False

        try:
            print(f"[AudioThread] Opening {self.PIPE_AUDIO} for reading...")
            self.audio_pipe_holder = open(self.PIPE_AUDIO, 'rb', buffering=0)
            print(f"[AudioThread] {self.PIPE_AUDIO} opened!")
            is_connected = True

            # Open PyAudio stream for playback
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                output=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            print("[AudioThread] PyAudio stream opened for playback")

        except Exception as e:
            print(f'[AudioThread] Error opening audio pipe or stream: {e}')
            return

        frame_count = 0

        while self.is_running and is_connected:
            try:
                # Read audio data
                # Format: 4 bytes length + PCM int16 data
                length_bytes = self.read_exactly(self.audio_pipe_holder, 4)
                if length_bytes is None:
                    print("[AudioThread] Audio pipe closed")
                    break

                audio_length = int.from_bytes(length_bytes, 'big')
                audio_data = self.read_exactly(self.audio_pipe_holder, audio_length)

                if audio_data is None:
                    print("[AudioThread] Failed to read audio data")
                    break

                # Play audio through speakers/headphones
                # audio_data is already in the correct format (PCM int16)
                self.stream.write(audio_data)

                frame_count += 1

                # Debug output every 100 frames (~6 seconds at 16kHz)
                if frame_count % 100 == 0:
                    print(f"[AudioThread] Played {frame_count} audio frames ({audio_length} bytes)")

            except Exception as e:
                print(f'[AudioThread] Error in audio playback loop: {e}')
                import traceback
                traceback.print_exc()
                break

        print("[AudioThread] Exiting run loop")


class SpeechThread(QThread):
    """
    Placeholder class for compatibility with existing GUI code.
    The actual audio streaming/playback is handled by AudioThread.
    This can be used for speech recognition if needed in the future.
    """

    def __init__(self):
        super().__init__()
        print('[SpeechThread] Initialized (placeholder)')

    def run(self):
        """Placeholder - speech recognition could be added here."""
        pass

    def stop(self):
        """Stop the thread."""
        self.quit()