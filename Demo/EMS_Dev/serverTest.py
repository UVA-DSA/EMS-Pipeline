import socket
import pickle
import time

HOST = "127.0.0.1" # The server's hostname or IP address
TCP_PORT = 7088

#initialize tcp connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  

sock.bind(("0.0.0.0", TCP_PORT))  
sock.listen(5)  

print("Waiting for android...")



connection,address = sock.accept()  
print("Connected to android: ",address)

# while(True):
#     example = FeedbackObj('intervention', 'protocol', 'concept')
#     sendMessage(example, connection)
#     time.sleep(1)

count = 0

while True:
    # print("data string to send from feedback: ", data_string)   
    data_string = b"Concept: (Wheezing, True, breath sounds, 0.76)" + b'\0' #b"Hello from cogEMS! " +str.encode(str(count)) + b'\0'
    data_string2 = b"Protocol: general - cardiac arrest: 0.919859" + b'\0' #b"Hello from cogEMS! " +str.encode(str(count)) + b'\0'
    data_string3 = b"Intervention: (Transport, 1.0) | (Cardiac monitor, 0.57)" + b'\0' #b"Hello from cogEMS! " +str.encode(str(count)) + b'\0'
    
    num_bytes = len(data_string)
    # connection.send(num_bytes.to_bytes())


    # sent = connection.send(getsizeof(data_string)) #b"hello from server"
    sent = connection.send(data_string) #b"hello from server"
    sent2 = connection.send(data_string2) #b"hello from server"
    sent3 = connection.send(data_string3) #b"hello from server"

    print("sent: ", sent)
    count += 1
    time.sleep(0.2)
    # break
