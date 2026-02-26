import socket
import pyaudio

# Audio configuration
FORMAT = pyaudio.paInt16  # 16-bit PCM format
CHANNELS = 2  # Mono
RATE = 44100  # Sample rate: 44100 Hz
CHUNK_SIZE = 1024  # Size of each audio chunk (you may need to adjust this)

# UDP socket configuration
UDP_IP = "0.0.0.0"  # IP address to listen on, use "0.0.0.0" to listen on all available IPs
UDP_PORT = 8888  # Port number to listen on

# Initialize UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# Initialize PyAudio
p = pyaudio.PyAudio()

info = p.get_default_host_api_info()
device_index = info.get('deviceCount') - 1 # get default device as output device

# Open a stream for audio playback
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                frames_per_buffer=CHUNK_SIZE,
                output_device_index=device_index)

print("Listening on UDP port %d" % UDP_PORT)

try:
    while True:
        data, addr = sock.recvfrom(CHUNK_SIZE)  # 2 bytes per sample

        print(data)

        print("Received %d bytes from %s" % (len(data), addr))
        stream.write(data)

except KeyboardInterrupt:
    pass
finally:
    print("Shutting down")
    stream.stop_stream()
    stream.close()
    p.terminate()
    sock.close()
