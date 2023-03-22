import socket
import threading, wave, pyaudio, time, queue

host_name = socket.gethostname()
host_ip = '0.0.0.0' #'172.25.149.100'#'192.168.1.102'#  socket.gethostbyname(host_name)
print(host_ip)
port = 50005
# For details visit: www.pyshine.com
q = queue.Queue(maxsize=12800)

def audio_stream_UDP():
    BUFF_SIZE = 1280 #65536
    client_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    # client_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)
    client_socket.bind((host_ip,port))
    p = pyaudio.PyAudio()
    CHUNK = 128
    stream = p.open(format=p.get_format_from_width(2),
                    channels=1,
                    rate=16000,#44100,
                    output=True,
                    frames_per_buffer=CHUNK)
                    
    # create socket
    message = b'Hello'
    # client_socket.sendto(message,(host_ip,port))
    socket_address = (host_ip,port)

    def getAudioData():
        while True:
            print("here")
            frame,_= client_socket.recvfrom(BUFF_SIZE)
            q.put(frame)
            print('Queue size...',q.qsize())
            
    t1 = threading.Thread(target=getAudioData, args=())
    t1.start()
    # time.sleep(5)

    print('Now Playing...')
    while True:
        frame = q.get()
        stream.write(frame)

    client_socket.close()
    print('Audio closed')
    os._exit(1)

t1 = threading.Thread(target=audio_stream_UDP, args=())
t1.start()