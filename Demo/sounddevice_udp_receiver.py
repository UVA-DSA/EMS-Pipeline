import socket
import sounddevice as sd
import numpy as np
from collections import deque
import threading
import time

# UDP socket configuration
UDP_IP = "0.0.0.0"  # Listen on all available IPs
UDP_PORT = 8888  # Port to listen on

# SoundDevice configuration
CHANNELS = 1  # Mono audio
RATE = 16000  # Sample rate in Hz
CHUNK_SIZE = 640  # Number of audio frames per buffer

# Jitter buffer configuration
BUFFER_DURATION = 0.5  # Target buffer duration in seconds
MAX_BUFFER_SIZE = int(RATE / CHUNK_SIZE * BUFFER_DURATION)  # Number of chunks to buffer




# Initialize jitter buffer
jitter_buffer = deque(maxlen=MAX_BUFFER_SIZE)

# Flag to control the playback thread
playback_running = True


def playback_thread():
    with sd.OutputStream(channels=CHANNELS, samplerate=RATE, dtype='int16') as stream:
        while playback_running:
            if len(jitter_buffer) > 0:
                samples = jitter_buffer.popleft()
                stream.write(samples)
                print("Wrote to stream:", len(samples), "samples")
                
            else:
                # Buffer is empty, wait a bit
                time.sleep(CHUNK_SIZE / RATE)
        print("Stopped audio playback")

# Function to continuously receive audio and add it to the jitter buffer
def receive_and_buffer(Window):
    # Initialize UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Listening for audio on UDP port {UDP_PORT}")

    global playback_running
    try:
        while True:
            if(Window.stopped == 1):
                break
            data, addr = sock.recvfrom(CHUNK_SIZE * CHANNELS * 2)  # 2 bytes per sample for 16-bit audio
            samples = np.frombuffer(data, dtype='int16')
            print("Received", len(samples), "samples"   )
            if len(jitter_buffer) < MAX_BUFFER_SIZE:
                jitter_buffer.append(samples)
            print("Buffer size:", len(jitter_buffer))
    except KeyboardInterrupt:
        print("Keyboard interrupt received, stopping...")
    finally:
        playback_running = False
        sock.close()
        print("Stopped audio streaming")

# # Start the playback thread
# playback_thread = threading.Thread(target=playback_thread)
# playback_thread.start()

# # Start receiving and buffering audio
# receive_and_buffer()

# # Wait for the playback thread to finish
# playback_thread.join()
