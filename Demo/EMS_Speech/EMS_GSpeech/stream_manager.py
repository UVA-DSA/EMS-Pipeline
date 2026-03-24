import os
import time


class GoogleSpeechStreamManager:
    """
    Google Cloud STT manager that reads PCM frames from /tmp/emsaudml.
    """

    pipe_audio_ml = "/tmp/emsaudml"

    def __init__(self):
        from PyQt5.QtCore import QThread, pyqtSignal

        class _GoogleThread(QThread):
            transcript_ready = pyqtSignal(str)
            whisper_ready = pyqtSignal()

            def __init__(self):
                super().__init__()
                self._running = True
                print("[GoogleSpeechThread] Initialized")

            def stop(self):
                print("[GoogleSpeechThread] Stopping...")
                self._running = False
                self.quit()
                self.wait(3000)
                print("[GoogleSpeechThread] Stopped")

            def run(self):
                print("[GoogleSpeechThread] Started")
                try:
                    from google.cloud import speech
                    from google.api_core.exceptions import OutOfRange, ServiceUnavailable
                except ImportError:
                    print(
                        "[GoogleSpeechThread] ERROR: google-cloud-speech not installed. "
                        "Run: pip install google-cloud-speech"
                    )
                    return

                client = speech.SpeechClient()
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code="en-US",
                    enable_automatic_punctuation=True,
                    model="latest_long",
                )
                streaming_config = speech.StreamingRecognitionConfig(
                    config=config,
                    interim_results=False,
                )

                print("[GoogleSpeechThread] Waiting for audio pipe...")
                while self._running:
                    if os.path.exists(GoogleSpeechStreamManager.pipe_audio_ml):
                        break
                    time.sleep(0.25)

                if not self._running:
                    return

                ready_emitted = False
                chunk_size = 3200

                def audio_generator(pipe_handle):
                    while self._running:
                        pcm = pipe_handle.read(chunk_size)
                        if not pcm:
                            return
                        yield speech.StreamingRecognizeRequest(audio_content=pcm)

                while self._running:
                    pipe = None
                    try:
                        pipe = open(GoogleSpeechStreamManager.pipe_audio_ml, "rb", buffering=0)
                        if not ready_emitted:
                            self.whisper_ready.emit()
                            ready_emitted = True
                        print("[GoogleSpeechThread] Audio pipe open - Google STT ready")
                        print("[GoogleSpeechThread] Starting streaming session...")

                        responses = client.streaming_recognize(
                            streaming_config,
                            audio_generator(pipe),
                        )
                        for response in responses:
                            if not self._running:
                                break
                            for result in response.results:
                                if result.is_final:
                                    text = result.alternatives[0].transcript.strip()
                                    if text:
                                        print(f"[GoogleSpeechThread] Transcript: {text}")
                                        self.transcript_ready.emit(text)
                    except OutOfRange:
                        if self._running:
                            print("[GoogleSpeechThread] Audio timeout - restarting session...")
                        continue
                    except ServiceUnavailable as exc:
                        if self._running:
                            print(
                                "[GoogleSpeechThread] Service unavailable, retrying in 2s: "
                                f"{exc}"
                            )
                            time.sleep(2)
                        continue
                    except Exception as exc:
                        if self._running:
                            print(f"[GoogleSpeechThread] Streaming error: {exc}")
                            time.sleep(1)
                        continue
                    finally:
                        if pipe is not None:
                            try:
                                pipe.close()
                            except Exception:
                                pass

                print("[GoogleSpeechThread] Exiting")

        self._thread = _GoogleThread()
        print("[GoogleSpeechStreamManager] Initialized")

    def start(self):
        print("[GoogleSpeechStreamManager] Starting...")
        self._thread._running = True
        self._thread.start()
        print("[GoogleSpeechStreamManager] Started")

    def stop(self):
        print("[GoogleSpeechStreamManager] Stopping...")
        self._thread.stop()
        print("[GoogleSpeechStreamManager] Stopped")

    @property
    def transcript_ready(self):
        return self._thread.transcript_ready

    @property
    def whisper_ready(self):
        return self._thread.whisper_ready
