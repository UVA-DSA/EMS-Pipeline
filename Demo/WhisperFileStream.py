from __future__ import absolute_import, division, print_function
import sys
import os
import time
from six.moves import queue
import numpy as np
from classes import SpeechNLPItem, GUISignal
import wave
import pyaudio
import os
import pipeline_config
import re

# Suppress pygame's welcome message
with open(os.devnull, 'w') as f:
    # disable stdout
    oldstdout = sys.stdout
    sys.stdout = f
    import pygame
    # enable stdout
    sys.stdout = oldstdout

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

class FileStream(object):
    # fileSessionCounter = 0
    # position = 0
    """Opens a file stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk, wavefile):
        self._rate = rate
        self._chunk = chunk
        self.filename = wavefile

        self.samplesCounter = 0
        self.start_time = time.time()

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = False

        # initialize pygame and recording end event
        pygame.init()
        pygame.mixer.init(frequency=RATE)  
        self.RECORDING_END = pygame.USEREVENT+1
        pygame.mixer.music.set_endevent(self.RECORDING_END)

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()

        self._audio_stream = self._audio_interface.open(format = pyaudio.paInt16, channels = 1, rate = self._rate, input = True, frames_per_buffer = self._chunk, stream_callback = self._fill_buffer, output=True)

        pygame.mixer.music.load(self.filename)
        pygame.mixer.music.play()

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        while not self.closed:
            for event in pygame.event.get():
                if event.type == self.RECORDING_END: self.closed = True
        return None, pyaudio.paContinue


def process_whisper_response(input_string):
    # used to remove the background noise transcriptions from Whisper output
    # Remove strings enclosed within parentheses
    input_string = re.sub(r'\([^)]*\)', '', input_string)
    # Remove strings enclosed within asterisks
    input_string = re.sub(r'\*[^*]*\*', '', input_string)
    # Remove strings enclosed within brackets
    input_string = re.sub(r'\[[^\]]*\]', '', input_string)
    # Remove null terminator since it does not display properly in Speech Box
    input_string = input_string.replace("\x00", " ")

    # separate transcript and confidence score
    # start = input_string.find('{')
    # end = input_string.find('}')
    # transcript = input_string[:start]
    # isFinal = True if input_string[start+1:start+2] == '1' else False
    # avg_p = float(input_string[start+3:end])

    return input_string, 0, 1#, isFinal, avg_p

def Whisper(SpeechToNLPQueue, EMSAgentSpeechToNLPQueue, wavefile_name):
    # # Create GUI Signal Object
    # SpeechSignal = GUISignal()
    # SpeechSignal.signal.connect(Window.UpdateSpeechBox)

    # MsgSignal = GUISignal()
    # MsgSignal.signal.connect(Window.UpdateMsgBox)

    # ButtonsSignal = GUISignal()
    # ButtonsSignal.signal.connect(Window.ButtonsSetEnabled)
    # num_chars_printed = 0

    with FileStream(RATE, CHUNK, wavefile_name) as fs:
        start = time.perf_counter()
        # print("=============WhisperFileStream.py: Audio file stream started", start)
        
        fifo_path = "/tmp/myfifo"  # Replace with your named pipe path

        while not fs.closed:
            print("==================================")
            try:
                with open(fifo_path, 'r') as fifo:
                    transcript_score = fifo.read().strip()  # Read the message from the named 
                    end = time.perf_counter()
                    # print("===============WhisperFileStream.py: transcript received", transcript_received)
                    transcript, isFinal, avg_p = process_whisper_response(transcript_score)
                    print('[Transcript received by Audiostream', transcript, ']')
                    QueueItem = SpeechNLPItem(transcript, isFinal, avg_p, 0, 'Speech')
                    EMSAgentSpeechToNLPQueue.put(QueueItem)
                    SpeechToNLPQueue.put(QueueItem)

                    pipeline_config.curr_segment += [end-start, transcript, avg_p]
                    start = end 

            except Exception as e:
                print("Exception in Audiostream", e)

    EMSAgentSpeechToNLPQueue.put('Kill')
    return                               
            



            
    






