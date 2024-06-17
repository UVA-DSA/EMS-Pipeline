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
from copy import copy
from collections import deque

import re

from StreamUtils import UDPStreamReceiver

from audio_streaming import audio_server

# speech to text client
client = None
streaming_config = None
# Audio recording parameters
RATE = 16000
CHUNK_SIZE = 640  # 100ms #50ms
CHANNEL = 1
#CHUNK = int(RATE * 5)  # 100ms

stopped = False

# UDP socket configuration
UDP_IP = "0.0.0.0"  # Listen on all available IPs
UDP_PORT = 8888  # Port to listen on

# PyAudio configuration
CHANNELS = 1  # Mono audio
RATE = 16000  # Sample rate in Hz
CHUNK_SIZE = 640  # Number of audio frames per buffer
FORMAT = pyaudio.paInt16  # 16-bit int

audio_buff = queue.Queue(maxsize=16)
wav_audio_buffer = queue.Queue()


# Jitter buffer configuration
BUFFER_DURATION = 0.5  # Target buffer duration in seconds
MAX_BUFFER_SIZE = int(RATE / CHUNK_SIZE * BUFFER_DURATION)  # Number of chunks to buffer

# Initialize jitter buffer
jitter_buffer = deque(maxlen=MAX_BUFFER_SIZE)

# Flag to control the playback thread
playback_running = True

stop_event = threading.Event()

# Initialize PyAudio


def playback_thread(stop_event):
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True)
    print("Started audio playback")
    sr = UDPStreamReceiver.UDPStreamReceiver(2222)

    q = queue.Queue(50)
    sr.registerQueue(q)

    while not stop_event.is_set():
        if q.empty():
            time.sleep(1e-3)
        else:
            sample = (q.get())
            if(sample is not None):
                stream.write(sample[12:])
                # print("Writing to stream")
            
    print("Audio Server Terminated!")
    sr.unregisterQueue(q)
    sr.close()
    stream.stop_stream()
    p.terminate()
    stop_event.clear()



def receive_and_buffer():
    # Initialize UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(5)  # Set a timeout for the socket operations
    print(f"Listening for audio on UDP port {UDP_PORT}")

    global playback_running
    global stopped
    try:
        while not stopped:
            try:
                # This will now raise a timeout exception if no data is received within the timeout period
                data, addr = sock.recvfrom(CHUNK_SIZE * CHANNELS * 2)  # 2 bytes per sample for 16-bit audio
                samples = np.frombuffer(data, dtype='int16')
                if len(jitter_buffer) < MAX_BUFFER_SIZE:
                    jitter_buffer.append(samples)
            except socket.timeout:
                # Catch the timeout and just continue, this gives us a chance to check if stopped has been set
                continue
    except KeyboardInterrupt:
        print("Keyboard interrupt received, stopping...")
    finally:
        playback_running = False
        sock.close()  # Ensure the socket is closed when exiting
        print("Stopped audio streaming")



def callback(in_data, frame_count, time_info, status):
    try:
        chunk = audio_buff.get(block=False)
    except queue.Empty:
        chunk = '\x00' * frame_count  # Silent bytes if buffer is empty
    return (chunk, pyaudio.paContinue)

def process_whisper_response(response):
    # used to remove the background noise transcriptions from Whisper output
    # Remove strings enclosed within parentheses
    response = re.sub(r'\([^)]*\)', '', response)
    # Remove strings enclosed within asterisks
    response = re.sub(r'\*[^*]*\*', '', response)
    # Remove strings enclosed within brackets
    response = re.sub(r'\[[^\]]*\]', '', response)
    # Remove null terminator since it does not display properly in Speech Box
    response = response.replace("\x00", " ")

    # separate transcript and confidence score
    start = response.find('{')
    end = response.find('}')
    block = response[:start]
    isFinal_str, avg_p_str, latency_str = response[start+1:end].split(",")
    isFinal = int(isFinal_str)
    avg_p = float(avg_p_str)
    latency = int(latency_str)

    return block, isFinal, avg_p, latency


# Google Cloud Speech API Recognition Thread for Microphone
def WhisperMicStream(Window, TranscriptQueue, EMSAgentSpeechToNLPQueue):
    print("Audio streaming app is running in the background.")
    
    global stopped
    global playback_running

    global stop_event
    
    # Start the playback thread
    _playback_thread = threading.Thread(target=playback_thread, args=(stop_event,))
    _playback_thread.start()

    # Start receiving and buffering audio
    # receive_and_buffer_thread = threading.Thread(target=receive_and_buffer)
    # receive_and_buffer_thread.start()

    # Wait for the playback thread to finish

    # playback_thread.join()


    fifo_path = "/tmp/myfifo"
    finalized_blocks = ''

    # MsgSignal = GUISignal()
    # MsgSignal.signal.connect(Window.UpdateMsgBox)

    # VUSignal = GUISignal()
    # VUSignal.signal.connect(Window.UpdateVUBox)

    
    TranscriptSignal = GUISignal()
    TranscriptSignal.signal.connect(Window.UpdateSpeechBox)
    
        
    with open(fifo_path, 'r') as fifo:
        old_response = ""
        while True:

            if(Window.stopped == 1): 
                stopped = True
                playback_running = False
                stop_event.set()
                print("WhisperMicStream received stop signal!")
                break
            
            try:
                response = fifo.read().strip()  # Read the message from the named pipe
            except Exception as e:
                response = ""

            if "You" in response:
                pass
            if response != old_response and response != "":
                block, isFinal, avg_p, latency = process_whisper_response(response) #isFinal = False means block is interim block
                transcript = finalized_blocks + block
                # if received block is finalized, then save to finalized blocks
                if isFinal: finalized_blocks += block
                # send transcript item if its not an empty string or space
                if len(transcript) and not transcript.isspace():
                    # transcriptItem = TranscriptItem(transcript, isFinal, avg_p, latency)
                    transcriptItem = SpeechNLPItem(transcript, isFinal,
                                        avg_p, len(transcript), 'Speech')
                    # EMSAgentQueue.put(transcriptItem)
                    
                    if isFinal:
                        print('WHISPER_OUT',transcriptItem.transcript)
                        TranscriptSignal.signal.emit([transcriptItem])
                        TranscriptQueue.put(transcriptItem) 
                        EMSAgentSpeechToNLPQueue.put(transcriptItem)
                    
                        
                    # intertimTranscriptItem = SpeechNLPItem(block, isFinal,
                    #                   avg_p, len(transcript), 'Speech')

                        
                print("--- Whisper Latency:", latency)
                old_response = response

