
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import PyQt5.QtWidgets,PyQt5.QtCore
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot

import pandas as pd
import socket
import collections
from cpr_calculation import preprocess_data, find_cpr_rate, find_peaks_valleys

# variables for getting smartwatch data via udp
localIP     = "0.0.0.0"
localPort   = 7889
bufferSize  = 1024
msgFromServer       = "Hello UDP Client"
bytesToSend         = str.encode(msgFromServer)

class Thread_Watch(QThread):
    changeActivityRec = pyqtSignal(str)

    def run(self):
    
        # Create a datagram socket
        UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        # Bind to address and ip
        UDPServerSocket.bind((localIP, localPort))
        print("UDP server up and listening")
        buffer = collections.deque([])
        # Listen for incoming datagrams
        columns = ['EPOCH_Time_ms','Wrist_Position','Sensor_Type','Value_X_Axis','Value_Y_Axis','Value_Z_Axis']
        self.changeActivityRec.emit('Measuring CPR Rate...')

        while(True):
            bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
            message = bytesAddressPair[0]
            address = bytesAddressPair[1]
            # clientMsg = “Message from Client:{}“.format(message.decode())
            clientIP  = "Client IP Address:{}".format(address)
            sw_data = message.decode().split(',')
            
            # convert to proper types
            sw_data[0] = int (sw_data[0])
            sw_data[3] = float (sw_data[3])
            sw_data[4] = float (sw_data[4])
            sw_data[5] = float (sw_data[5])

            # append accelerometer data to buffer
            if(sw_data[2] == 'acc'):
                buffer.append(sw_data)

            current_buffer_size = len(buffer)
            # start = time.time()

            if current_buffer_size%1000 == 0:
                data_frame = pd.DataFrame(list(buffer), columns=columns)

                #preprocess dtaa to get xyz magnitude
                data_frame = preprocess_data(data_frame)

                #calculate cpr rate and avg time
                peaks,valleys,height,min_height = find_peaks_valleys(data_frame,height=32.5,distance=1,prominence=1)
                avg_time,cpr_rate = find_cpr_rate(peaks)
                # print(avg_time,cpr_rate)

                # if cpr rate and avg time are not 0 (i.e. valid), then display in Smartwatch activity box on GUI
                if avg_time != 0 and cpr_rate != 0:
                    str_output = "Average time between peaks in seconds (scipy): " + str(round(avg_time,2)) + " \n" + "CPR Rate Per Minute (scipy): " + str(round(cpr_rate,2))
                    self.changeActivityRec.emit(str(str_output))

                # clear buffer to get next 1000 data points for cpr rate calculation
                buffer.clear()