import socket
import pickle

HOST = "127.0.0.1" # The server's hostname or IP address
PORT = 7088  # The port used by the server


class FeedbackObj:
    def __init__(self, intervention, protocol, concept):
        super(FeedbackObj, self).__init__()
        self.intervention = intervention
        self.protocol = protocol
        self.concept = concept


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    while(True):
        data = s.recv(1024)
        data_variable = pickle.loads(data)
        print(f"Received {data_variable!r}")
        print("Data: ", data_variable.intervention, data_variable.protocol, data_variable.concept)
        # s.send(b"Hi from client!")