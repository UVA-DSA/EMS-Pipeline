
import pandas as pd
import socket
import collections
from datetime import datetime
import os
import csv
import time

# variables for getting smartwatch data via udp
localIP     = "0.0.0.0"
localPort   = 7889
bufferSize  = 1024
msgFromServer       = "Hello UDP Client"
bytesToSend         = str.encode(msgFromServer)


# Create a datagram socket
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
# Bind to address and ip
UDPServerSocket.bind((localIP, localPort))
print("UDP server up and listening")
buffer = collections.deque([])
# Listen for incoming datagrams
columns = ['sw_epoch_ms','wrist_position','sensor_type','value_X_Axis','value_Y_Axis','value_Z_Axis','server_epoch_ms']

curr_date = datetime.now()
dt_string = curr_date.strftime("%d-%m-%Y-%H-%M-%S")

newpath = "./smartwatch_data/"+dt_string

if not os.path.exists(newpath):
    os.makedirs(newpath)

    with open('./smartwatch_data/'+dt_string+'/'+dt_string+'data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        
        while(True):
            bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
            message = bytesAddressPair[0]
            address = bytesAddressPair[1]
            # clientMsg = “Message from Client:{}“.format(message.decode())
            clientIP  = "Client IP Address:{}".format(address)
            sw_data = message.decode().split(',')
            
            curr_epoch_time = time.time_ns() // 1e6
            
            # convert to proper types
            sw_data[0] = int (sw_data[0])
            sw_data[3] = float (sw_data[3])
            sw_data[4] = float (sw_data[4])
            sw_data[5] = float (sw_data[5])
            sw_data[6] = curr_epoch_time #check

            writer.writerow(sw_data)
            print(sw_data)