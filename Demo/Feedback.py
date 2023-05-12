import json
import pickle
import socket
import time
# TCP_IP = '127.0.0.1'
TCP_PORT = 7088

class FeedbackObj:
    def __init__(self, intervention, protocol, concept):
        super(FeedbackObj, self).__init__()
        self.intervention = intervention
        self.protocol = protocol
        self.concept = concept


def sendMessage(feedbackObj, connection):
    count = 0
    data_string = b""
    # data_string = pickle.dumps(feedbackObj)   
    # data_string = json.dumps(feedbackObj)
    
    if feedbackObj.concept != "":
        data_string = b"Concepts: " + feedbackObj.concept.encode('ascii') + b'\0'
    elif feedbackObj.intervention != "":
        data_string = b"Intervention: " + feedbackObj.intervention.encode('ascii') + b'\0'
    elif feedbackObj.protocol != "":
        data_string = b"Protocol: " + feedbackObj.protocol.encode('ascii') + b'\0'
   
    #just for testing, delete later
    # data_string = b"Hello from cogEMS! " +str.encode(str(count)) + b'\0'

    print("data string to send from feedback: ", data_string)   

    sent = connection.send(data_string) #b"hello from server"
    print("sent: ", sent)

    count += 1
    time.sleep(0.2)
            

def Feedback (Window, data_path, FeedbackQueue):
    #initialize tcp connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  

    sock.bind(("0.0.0.0", TCP_PORT))  
    sock.listen(5)  

    print("Waiting for client in feedback...")

    connection,address = sock.accept()  
    print("Client connected for feedback: ",address)

    # while(True):
    #     example = FeedbackObj('intervention', 'protocol', 'concept')
    #     sendMessage(example, connection)
    #     time.sleep(1)

    while True:

        # Get queue item from the Speech-to-Text Module
        received = FeedbackQueue.get()

        if(received == 'Kill'):
            # print("Thread received Kill Signal. Killing Feedback Thread.")
            break

        if(Window.reset == 1):
            # print("Cognitive System Thread Received reset signal. Killing Feedback Thread.")
            return

        # If item received from queue is legitmate
        else:
            print("Received chunk", received)
        
        print("sending message: ", received)
        sendMessage(received, connection)

        try:
            # connection.send("some more data")
            sendMessage(received, connection)
        except:
            print("Reconnecting to a client...")
            connection.close()

            # recreate the socket and reconnect
            # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  

            # sock.bind(("0.0.0.0", TCP_PORT))  
            # sock.listen(5)  

            connection,address = sock.accept()  
            print("Client connected: ",address)
            print("Client connected for feedback: ",address)
            sendMessage(received, connection)
            # connection.send("some more data")






        