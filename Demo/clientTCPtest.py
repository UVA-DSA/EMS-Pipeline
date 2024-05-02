import socket
import pickle
from classes import FeedbackObj

HOST = "127.0.0.1" # The server's hostname or IP address
PORT = 7088  # The port used by the server


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    while(True):
        data = s.recv(1024)
        data_variable = pickle.loads(data)
        print(f"Received {data_variable!r}")
        print("Data: ", data_variable.intervention, data_variable.protocol, data_variable.concept)
        # s.send(b"Hi from client!")