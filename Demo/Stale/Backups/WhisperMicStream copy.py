from __future__ import absolute_import, division, print_function
import os
from timeit import default_timer as timer
import sys
import pyaudio
import time
from google.cloud import speech_v1 as speech
from six.moves import queue
from classes import SpeechNLPItem, GUISignal
import datetime
import wave
import numpy as np
import threading
import socket
import traceback
from google.api_core import exceptions
from google.api_core.retry import Retry
from copy import copy
import base64
import socketio
# speech to text client
client = None
streaming_config = None
# Audio recording parameters
RATE = 16000
CHUNK = 512  # 100ms #50ms
#CHUNK = int(RATE * 5)  # 100ms

stopped = False


class MicrophoneStream:

    def __init__(self):
        self.is_running = True
        self.audio_buff = queue.Queue()

        self.sio = socketio.Client()

        @self.sio.on('server_audio')
        def on_audio(data):
            # print("Received audio from server", data)
            decoded_audio_data = base64.b64decode(data)
            self.audio_buff.put(decoded_audio_data)

        super().__init__()
        

    def getAudioBuffer(self):  # Get audio buffer
        return self.audio_buff
    
    def run(self):

        while self.is_running:
            try:
                self.sio.connect('http://localhost:9235')  # Connect to the Flask-SocketIO server
                print("Connected to the server! Audio QThread")
                self.sio.emit('message', 'Hello from Audio QThread!')  # Send a message to the server
                self.sio.wait()
                break  # Exit the loop if connected successfully
            except Exception as e:
                # print("Connection failed, retrying...", e)
                time.sleep(5)  # Wait for 5 seconds before retrying






# Google Cloud Speech API Recognition Thread for Microphone
def WhisperMicStream(Window, TranscriptQueue):

    
    #Microphone stream setup
    MicrophoneStreamObj = MicrophoneStream()
    threading.Thread(target=MicrophoneStreamObj.run).start()


    print("api mehe")
    MsgSignal = GUISignal()
    MsgSignal.signal.connect(Window.UpdateMsgBox)

    VUSignal = GUISignal()
    VUSignal.signal.connect(Window.UpdateVUBox)

    # Instantiate PyAudio and initialize PortAudio system resources (1)
    p = pyaudio.PyAudio()
    info = p.get_default_host_api_info()
    # print devices available
    for i in range(info.get('deviceCount')):
        if (p.get_device_info_by_host_api_device_index(info.get('index'), i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(info.get('index'), i).get('name'))
        if (p.get_device_info_by_host_api_device_index(info.get('index'), i).get('maxOutputChannels')) > 0:
            print("Output Device id ", i, " - ", p.get_device_info_by_host_api_device_index(info.get('index'), i).get('name'))
    device_index = info.get('deviceCount') - 1 # get default device as output device

    stream = p.open(format = pyaudio.paInt16, channels = 1, rate = RATE, output = True, frames_per_buffer = CHUNK, output_device_index=device_index)
    
    while True:
        if(MicrophoneStreamObj):
            audio_buff = MicrophoneStreamObj.getAudioBuffer()
            
            try:
                chunk = audio_buff.get()
                if chunk is None:
                    print('### Speech Paused')
                    return
                data = [chunk]
                # print("Writing to stream")

                # VU Meter in the GUI
                signal = b''.join(data)

                stream.write(chunk)

                signal = np.fromstring(signal, 'int16') 
                VUSignal.signal.emit([signal])

            except Exception as e:
                print("Exception in WhisperMicStream: ", e)
                traceback.print_exc()
                # Close stream (4)
                stream.close()


                # Release PortAudio system resources (5)
                p.terminate()

                print("Audio Stream Thread Sent Kill Signal. Bye!")
