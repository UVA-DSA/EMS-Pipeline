import socket
import sounddevice as sd
import numpy as np
from collections import deque
import threading
import time
import wave

import pyaudio

p = pyaudio.PyAudio()

# UDP socket configuration
UDP_IP = "0.0.0.0"  # Listen on all available IPs
UDP_PORT = 8888  # Port to listen on

# SoundDevice configuration
CHANNELS = 1  # Mono audio
RATE = 16000  # Sample rate in Hz
CHUNK_SIZE = 640  # Number of audio frames per buffer

# Function to continuously receive audio and add it to the jitter buffer
def receive_and_buffer(recording_dir, commandqueue):
    
    # Initialize UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Listening for audio on UDP port {UDP_PORT}")


    sock.setblocking(False)
    
    try:
        
        wf = wave.open(f'{recording_dir}/received_audio.wav', 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(RATE)
        
        while True:
            
            try:   
                data, addr = sock.recvfrom(CHUNK_SIZE * CHANNELS * 2)
                wf.writeframes(data)
                print("Buffer size:", len(data))
            except BlockingIOError:
                # No data received, proceed to check the command queue
                pass
            
            if(not commandqueue.empty() ):
                command = commandqueue.get()
                print("Command received: ", command)
                if(command == "stop"):
                    break
            
        sock.close()
        wf.close()
        print("Stopped audio streaming")
        
    except Exception as e:
        print("Audio streaming exception: ", e)

