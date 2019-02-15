from __future__ import division

import re
import sys
import os
import time

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import pyaudio
from six.moves import queue
import threading

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

context_phrases = []


# ============== Custom Thread Class with a Stop Flag ==============

class StoppableThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()
        return self._stop_event.is_set()
    
# ============== Microphone Stream Object ==============

class MicrophoneStream(object):
    micSessionCounter = 0
    def __init__(self, rate, chunk):
        MicrophoneStream.micSessionCounter += 1
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = time.time()
        self.self_time = 0

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
                    
            self.self_time += self._chunk/self._rate  
            
            # Stop streaming after one minute
            if(time.time() > (self.start_time + (60)) or self.self_time > 60):
                #print(time.time() - self.start_time)
                #print(self.self_time)
                #print("\n\nAPI's 1 minute limit reached.\n\n")
                StoppableThread(target = GoogleSpeechStream).start()
                self.closed = 1
                continue
                
            yield b''.join(data)

def GoogleSpeechStream():
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"
    
    client = speech.SpeechClient()
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16, 
        sample_rate_hertz=RATE, 
        language_code='en-US',
        enable_automatic_punctuation=True,
        speech_contexts = [speech.types.SpeechContext(phrases=context_phrases,)]
    )
    streaming_config = types.StreamingRecognitionConfig(config=config, interim_results=True)

    with MicrophoneStream(RATE, CHUNK) as stream:

        audio_generator = stream.generator()
        requests = (types.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)
        responses = client.streaming_recognize(streaming_config, requests)

        num_chars_printed = 0
        for response in responses:
            if not response.results:
                continue
                
            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript
            overwrite_chars = ' ' * (num_chars_printed - len(transcript))

            if not result.is_final:
                sys.stdout.write(transcript + overwrite_chars + '\r')
                sys.stdout.flush()
                #print(response.results[0].alternatives)
                num_chars_printed = len(transcript)

            else:
                print(transcript + overwrite_chars)
                #print(response.results[0].alternatives)
                num_chars_printed = 0

if __name__ == '__main__':
    StoppableThread(target = GoogleSpeechStream).start()
    context_phrases = []


    
