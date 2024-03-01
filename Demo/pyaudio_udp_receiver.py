import socket
import pyaudio

# UDP socket configuration
UDP_IP = "0.0.0.0"  # Listen on all available IPs
UDP_PORT = 8888  # Port to listen on

# PyAudio configuration
FORMAT = pyaudio.paInt16  # Assuming 16-bit PCM audio
CHANNELS = 1  # Mono audio
RATE = 16000  # Sample rate in Hz
CHUNK_SIZE = 1024  # Number of audio frames per buffer

# Initialize UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print(f"Listening for audio on UDP port {UDP_PORT}")

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open audio stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                frames_per_buffer=CHUNK_SIZE)

# Function to continuously receive audio and play it
def receive_and_play():
    try:
        while True:
            data, addr = sock.recvfrom(CHUNK_SIZE * CHANNELS * 2)  # 2 bytes per sample for 16-bit audio
            stream.write(data)
            print("data received: ",len(data))
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        stream.stop_stream()
        stream.close()
        p.terminate()
        sock.close()
        print("Stopped audio streaming")

# Start receiving and playing audio
receive_and_play()
