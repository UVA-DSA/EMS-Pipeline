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

import re

from audio_streaming import audio_server

from sounddevice_udp_receiver import playback_thread, receive_and_buffer
# speech to text client
client = None
streaming_config = None
# Audio recording parameters
RATE = 16000
CHUNK = 640  # 100ms #50ms
CHANNEL = 1
#CHUNK = int(RATE * 5)  # 100ms

stopped = False


audio_buff = queue.Queue(maxsize=16)
wav_audio_buffer = queue.Queue()

def audio_stream_UDP():
    
    
    # Instantiate PyAudio and initialize PortAudio system resources (1)
    p = pyaudio.PyAudio()
    info = p.get_default_host_api_info()
    device_index = info.get('deviceCount') - 1 # get default device as output device

    stream = p.open(format = pyaudio.paInt16, channels = 1, rate = RATE, output = True, frames_per_buffer = CHUNK, output_device_index=device_index)
    
                
    global audio_buff
    global stopped
    # AUDIO streaming variables and functions
    host_name = socket.gethostname()
    host_ip = '0.0.0.0' #socket.gethostbyname(host_name)
    port = 8888

    client_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    client_socket.bind((host_ip,port))


    print("Audio Thread Created!")

        
    # receive and put audio data in queue
    def getAudioData():
        
        global stopped
        
        while True:
            if stopped == True:
                break
            
            # print("Waiting for Audio client...")
            frame,_= client_socket.recvfrom(CHUNK * CHANNEL * 2)
            stream.write(frame)
            
        client_socket.close()
        stream.close()
        p.terminate()
        print("Audio Server Terminated!")
        
            
    t2 = threading.Thread(target=getAudioData)
    t2.start()

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
def WhisperMicStream(Window, TranscriptQueue):
    print("Audio streaming app is running in the background.")
    
    global stopped
    
    t1 = threading.Thread(target=audio_stream_UDP)
    t1.start()    

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
                break
            
            try:
                response = fifo.read().strip()  # Read the message from the named pipe
            except Exception as e:
                response = ""

            if response == "You":
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
                    
                        
                    # intertimTranscriptItem = SpeechNLPItem(block, isFinal,
                    #                   avg_p, len(transcript), 'Speech')

                        
                print("--- Whisper Latency:", latency)
                old_response = response

