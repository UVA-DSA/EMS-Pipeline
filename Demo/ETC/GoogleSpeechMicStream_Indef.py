from __future__ import absolute_import, division, print_function
from timeit import default_timer as timer
import sys
import os
import pyaudio
import time
from six.moves import queue
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import numpy as np
from classes import SpeechNLPItem, GUISignal
import wave

import threading
from StoppableThread.StoppableThread import StoppableThread

# Audio recording parameters
STREAMING_LIMIT = 55000
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms

def get_current_time():
    return int(round(time.time() * 1000))

def duration_to_secs(duration):
    return duration.seconds + (duration.nanos / float(1e9))

class MicrophoneStream:
    micSessionCounter = 0
    def __init__(self, Window, rate, chunk_size):
        MicrophoneStream.micSessionCounter += 1
        self.Window = Window
        self._rate = rate
        self._chunk_size = chunk_size
        self._num_channels = 1
        self._max_replay_secs = 5
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = get_current_time()
        self._bytes_per_sample = 2 * self._num_channels
        self._bytes_per_second = self._rate * self._bytes_per_sample
        self._bytes_per_chunk = (self._chunk_size * self._bytes_per_sample)
        self._chunks_per_second = (self._bytes_per_second // self._bytes_per_chunk)

    def __enter__(self):
        self.closed = False
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk_size,
            stream_callback=self._fill_buffer,
        )

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, *args, **kwargs):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):

        # Create Signal Objects
        MsgSignal = customSignalObject()
        MsgSignal.signal.connect(self.Window.updateMsgBox)

        VUSignal = GUISignal()
        VUSignal.signal.connect(self.Window.updateVUBox)

        while not self.closed:
            print("generating")
            if get_current_time() - self.start_time > STREAMING_LIMIT:
                MsgSignal.signal.emit("API's 1 minute limit reached. Restablishing connection!", "", 0, 0)
                self.start_time = get_current_time()
                break

            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Plot in the GUI
            signal = b''.join(data)
            signal = np.fromstring(signal, 'Int16') 
            VUSignal.signal.emit([signal])

            if self.Window.stopped == 1:
                print('Speech Tread Killed')
                self.Window.StartButton.setEnabled(True)
                return

            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)

# Google Cloud Speech API Recognition Thread for Microphone
def GoogleSpeech(Window, SpeechToNLPQueue):

    # Create Signal Object
    SpeechSignal = GUISignal()
    SpeechSignal.signal.connect(Window.UpdateSpeechBox)

    MsgSignal = customSignalObject()
    MsgSignal.signal.connect(Window.updateMsgBox)

    client = speech.SpeechClient()

    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code='en-US',
        #enableWordConfidence = True,
        profanity_filter = True,
        max_alternatives=1,
        enable_word_time_offsets=True)

    streaming_config = speech.types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    mic_manager = MicrophoneStream(Window, SAMPLE_RATE, CHUNK_SIZE)

    with mic_manager as stream:
        while not stream.closed:

            while(Window.stopped == 1):
                pass

            audio_generator = stream.generator()
            requests = (speech.types.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)

            try:
                responses = client.streaming_recognize(streaming_config, requests)
                responses = (r for r in responses if (r.results and r.results[0].alternatives))

                num_chars_printed = 0
                for response in responses:
                    if not response.results:
                        continue
                    result = response.results[0]
                    if not result.alternatives:
                        continue

                    # Display the transcription of the top alternative.
                    transcript = result.alternatives[0].transcript
                    confidence = result.alternatives[0].confidence

                    # Display interim results, but with a carriage return at the end of the
                    # line, so subsequent lines will overwrite them.
                    # If the previous result was longer than this one, we need to print
                    # some extra spaces to overwrite the previous result
                    overwrite_chars = ' ' * (num_chars_printed - len(transcript))

                    if result.is_final:
                        #print(transcript + overwrite_chars)
                        QueueItem = SpeechNLPItem(transcript, result.is_final, confidence, num_chars_printed, 'Speech')
                        SpeechToNLPQueue.put(QueueItem)
                        SpeechSignal.signal.emit([QueueItem])
                        num_chars_printed = 0

                    elif not result.is_final:
                        #sys.stdout.write(transcript + overwrite_chars + '\r')
                        #sys.stdout.flush()
                        QueueItem = SpeechNLPItem(transcript, result.is_final, confidence, num_chars_printed, 'Speech')
                        SpeechSignal.signal.emit([QueueItem])
                        num_chars_printed = len(transcript)

            except Exception as e:
                MsgSignal.signal.emit('Unable to get response from Google! Timeout or network issues. Please Try again!\n Exception: ' + str(e), "", 0, 0)     
                Window.StartButton.setEnabled(True)
                sys.exit()
     
