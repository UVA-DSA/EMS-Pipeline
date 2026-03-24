import ctypes
import os
import time


def audio_player_process(pipe_path, running_flag):
    """
    Separate process that reads audio from a named pipe and plays it.
    """
    import pyaudio

    print(f"[AudioProcess] Starting, PID: {os.getpid()}")

    sample_rate = 16000
    channels = 1
    chunk_size = 1024

    def read_exactly(pipe, n_bytes):
        data = b""
        while len(data) < n_bytes:
            chunk = pipe.read(n_bytes - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    frame_count = 0
    pyaudio_handle = None
    stream = None
    audio_pipe = None

    try:
        print(f"[AudioProcess] Opening {pipe_path} for reading...")
        audio_pipe = open(pipe_path, "rb", buffering=0)
        print(f"[AudioProcess] {pipe_path} opened!")

        pyaudio_handle = pyaudio.PyAudio()
        stream = pyaudio_handle.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            output=True,
            frames_per_buffer=chunk_size,
        )

        print("[AudioProcess] PyAudio stream opened successfully")
        last_debug = time.time()

        while running_flag.value:
            try:
                length_bytes = read_exactly(audio_pipe, 4)
                if length_bytes is None:
                    print("[AudioProcess] Audio pipe closed")
                    break

                audio_length = int.from_bytes(length_bytes, "big")
                audio_data = read_exactly(audio_pipe, audio_length)
                if audio_data is None:
                    print("[AudioProcess] Failed to read audio data")
                    break

                stream.write(audio_data, exception_on_underflow=False)
                frame_count += 1

                if frame_count == 1:
                    print(f"[AudioProcess] Played first audio frame ({audio_length} bytes)")

                if frame_count % 100 == 0:
                    elapsed = time.time() - last_debug
                    print(
                        "[AudioProcess] Played "
                        f"{frame_count} audio frames ({audio_length} bytes, {elapsed:.1f}s elapsed)"
                    )
                    last_debug = time.time()
            except Exception as exc:
                if running_flag.value:
                    print(f"[AudioProcess] Error in playback loop: {exc}")
                break
    except Exception as exc:
        print(f"[AudioProcess] Error: {exc}")
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass

        if pyaudio_handle is not None:
            try:
                pyaudio_handle.terminate()
            except Exception:
                pass

        if audio_pipe is not None:
            try:
                audio_pipe.close()
            except Exception:
                pass

    print(f"[AudioProcess] Exiting, played {frame_count} frames")


class AudioStreamManager:
    """
    Manager class for the audio playback process.
    """

    def __init__(self):
        self.pipe_audio = "/tmp/emsaud"

        import multiprocessing as mp

        self.ctx = mp.get_context("spawn")
        self.running_flag = self.ctx.Value(ctypes.c_bool, True)
        self.player_process = self.ctx.Process(
            target=audio_player_process,
            args=(self.pipe_audio, self.running_flag),
            daemon=True,
        )
        print("[AudioStreamManager] Initialized")

    def start(self):
        print("[AudioStreamManager] Starting...")
        self.running_flag.value = True
        self.player_process.start()
        print(f"[AudioStreamManager] Started (Process PID: {self.player_process.pid})")

    def stop(self):
        print("[AudioStreamManager] Stopping...")
        self.running_flag.value = False
        self.player_process.join(timeout=3)

        if self.player_process.is_alive():
            print("[AudioStreamManager] Process did not stop, terminating...")
            self.player_process.terminate()
            self.player_process.join(timeout=1)

        print("[AudioStreamManager] Stopped")
