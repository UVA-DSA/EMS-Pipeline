#Thread for streaming video to the GUI vision window

import csv
import os
import time
import math
import datetime
import time

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
import PyQt5.QtWidgets,PyQt5.QtCore
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

import mediapipe as mp
import socket
import datetime as dt
import numpy as np
import pandas as pd
import cv2
import socket
from PIL import Image
import io


#functions and variables for getting video stream from Android camera on AR glasses
TCP_IP = '127.0.0.1'
TCP_PORT = 9009

seq_all=0
dropped_imgs=0
total_imgs=0


class ThreadPhoneVid(QThread):
    def __init__(self, var, bool):
        self.data_path_str = var + "phonevideodata_fordepth/"
        self.videoStreamBool = bool
        super().__init__()

    def run(self):
        recording_enabled = True
        frame_index = 0

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
        sock.bind(('0.0.0.0', TCP_PORT))  
        sock.listen(5)  

        print("Waiting for client...")

        connection,address = sock.accept()  
        print("Client connected: ",address)


        #array for timestamps when every image is received
        image_times = []

        curr_date = datetime.datetime.now()
        dt_string = curr_date.strftime("%d-%m-%Y-%H-%M-%S")

       
        if self.videoStreamBool == True: #only record data if we are doing data collection
            if not os.path.exists(self.data_path_str + dt_string):
                os.makedirs(self.data_path_str + dt_string)
            with open(self.data_path_str + "phoneviddata.csv", 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["frame", "recieved_ts", "origin_ts"])
                is_video_created = False
            
                while True:
                    if( int.from_bytes(connection.recv(1),"big") == 22):
                        # print("Image received")

                        timestamp = int.from_bytes(connection.recv(8),"big")
                        # print("Got timestamp:" , timestamp)

                        img_byte_length = int.from_bytes(connection.recv(8),"big")
                        # print("Got Num of bytes in the image:" , img_byte_length)

                        # img_buffer_size = math.ceil(img_byte_length/1024)*1024
                        img_buffer_size = img_byte_length
                        # print("Buff Size Should Be: ", img_buffer_size)

                        # if(img_buffer_size > 180000 or img_buffer_size < 10000):
                        #     continue

                        img_bytes = bytearray()

                        while(img_buffer_size>0):
                            read_len=min(img_buffer_size,10240)
                            data = connection.recv(read_len)
                            img_buffer_size -= len(data)
                            img_bytes+=data
                            # print("Remaining buffer size: ",img_buffer_size)

                        #get timestamp when getting the image frame -- for cpr rate detection
                        curr_time = round(time.time()*1000)
                        image_times.append(curr_time)

                        image = Image.open(io.BytesIO(img_bytes))
                        img_ar=np.array(image)
                        image = img_ar

                        #convert image to pixmap format and display on GUI
                        RGB_img = image
                        RGB_img = cv2.rotate(RGB_img, cv2.ROTATE_180)

                        #write to file
                        if(recording_enabled and not is_video_created):

                            height,width,layers=img_ar.shape
                            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
                            video = cv2.VideoWriter(self.data_path_str+'/'+dt_string+'.avi', fourcc, 30, (width, height))
                            is_video_created = True

                        if(recording_enabled):
                            now = time.time()*1e3

                            cv2.imwrite(self.data_path_str+ dt_string + '/img_'+str(frame_index)+'.jpg', img_ar)
                            video.write(RGB_img)
                            writer.writerow([frame_index, now, timestamp])
                            frame_index += 1
                        
                        h, w, ch = RGB_img.shape
                        # print("Image Size",h,w)
                        bytesPerLine = ch * w
                        
                        if(len(image_times) == 10000):
                            #Clear the arrays to get 100 more image frames for cpr rate calculation
                            image_times.clear()

                    else:
                        # continue
                        print("Reconnecting to a client...")
                        connection.close()
                

                        connection,address = sock.accept()  
                        print("Client connected: ",address)

        else: #video stream bool = false --> no data collection
            while True:
                if( int.from_bytes(connection.recv(1),"big") == 22):
                    # print("Image received")

                    timestamp = int.from_bytes(connection.recv(8),"big")
                    # print("Got timestamp:" , timestamp)

                    img_byte_length = int.from_bytes(connection.recv(8),"big")
                    # print("Got Num of bytes in the image:" , img_byte_length)

                    # img_buffer_size = math.ceil(img_byte_length/1024)*1024
                    img_buffer_size = img_byte_length
                    # print("Buff Size Should Be: ", img_buffer_size)

                    # if(img_buffer_size > 180000 or img_buffer_size < 10000):
                    #     continue

                    img_bytes = bytearray()

                    while(img_buffer_size>0):
                        read_len=min(img_buffer_size,10240)
                        data = connection.recv(read_len)
                        img_buffer_size -= len(data)
                        img_bytes+=data
                        # print("Remaining buffer size: ",img_buffer_size)

                    #get timestamp when getting the image frame -- for cpr rate detection
                    curr_time = round(time.time()*1000)
                    image_times.append(curr_time)

                    image = Image.open(io.BytesIO(img_bytes))
                    img_ar=np.array(image)
                    image = img_ar

                    #convert image to pixmap format and display on GUI
                    RGB_img = image
                    RGB_img = cv2.rotate(RGB_img, cv2.ROTATE_180)
                    
                    h, w, ch = RGB_img.shape
                    # print("Image Size",h,w)
                    bytesPerLine = ch * w
                    
                    if(len(image_times) == 10000):
                        #Clear the arrays to get 100 more image frames for cpr rate calculation
                        image_times.clear()

                else:
                    # continue
                    print("Reconnecting to a client...")
                    connection.close()
            

                    connection,address = sock.accept()  
                    print("Client connected: ",address)
