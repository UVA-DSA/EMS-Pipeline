#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============== Imports ==============

from __future__ import absolute_import, division, print_function
from six.moves import queue
import sys
import os
import time
import threading
import math
import datetime
import time

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QLineEdit
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtCore

#from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer

# from mediapipe_thread import MPThread

from PyQt5.QtCore import QCoreApplication, Qt,QBasicTimer, QTimer,QPoint,QSize
import PyQt5.QtWidgets,PyQt5.QtCore

import numpy as np
import pandas as pd

from StoppableThread.StoppableThread import StoppableThread

import py_trees
from py_trees.blackboard import Blackboard

#import GenerateForm
from behaviours_m import *
from DSP.amplitude import Amplitude
from classes import SpeechNLPItem, GUISignal
import GoogleSpeechMicStream
import GoogleSpeechFileStream
#import DeepSpeechMicStream
#import DeepSpeechFileStream
# import WavVecMicStream
# import WavVecFileStream
import TextSpeechStream
import CognitiveSystem

import csv
chunkdata = []

import sys
import cv2
import socket
from PIL import Image
import io
import numpy as np
#import time
#from datetime import datetime
import struct
import os
import traceback
import argparse
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

import mediapipe as mp
import threading

import socket
import threading, wave, pyaudio, time, queue

import matplotlib.pyplot as plt
from scipy.signal import find_peaks,find_peaks_cwt,peak_widths
from scipy import signal
import datetime as dt
import collections
import pylab

# ----------------------------------------------------SMART WATCH--------------------------------------------------------

# functions for smartwatch activity

def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))

def robust_peaks_detection_zscore(df,lag,threshold,influence):
    rate = thresholding_algo(df['Value_Magnitude_XYZ'],lag,threshold,influence)
    indices = np.where(rate['signals'] == 1)[0]
    robust_peaks_time = df.iloc[indices]['EPOCH_Time_ms']
    robust_peaks_value = df.iloc[indices]['Value_Magnitude_XYZ']
    indices = np.where(rate['signals'] == -1)[0]
    robust_valleys_time = df.iloc[indices]['EPOCH_Time_ms']
    robust_valleys_value = df.iloc[indices]['Value_Magnitude_XYZ']
    # # #Plotting
    # fig = plt.figure()
    # ax = fig.subplots()
    # ax.plot(acc_df[‘EPOCH_Time_ms’].tolist(),acc_df[‘Value_Magnitude_XYZ’].tolist())
    # ax.scatter(robust_valleys_time, robust_valleys_value, color = ‘gold’, s = 15, marker = ‘v’, label = ‘Minima’)
    # ax.scatter(robust_peaks_time, robust_peaks_value, color = ‘b’, s = 15, marker = ‘X’, label = ‘Robust Peaks’)
    # ax.legend()
    # ax.grid()
    # plt.show()
    return [robust_peaks_time,robust_peaks_value,robust_valleys_time,robust_valleys_value]

"""
Function to find the peaks and valleys
1. uses scipy find_peaks implementation.
2. paramter tuning is highly important !!!
3. todo
"""
# def find_peaks_valleys_cwt(df):
#     peaks = find_peaks_cwt(df['Value_Magnitude_XYZ'],np.arange(100,2000))
#     print(peaks)
#     height = peaks[1][‘peak_heights’] #list of the heights of the peaks
#     peak_pos = data_frame.iloc[peaks[0]] #list of the peaks positions
#     # #Finding the minima
#     y2 = df[‘Value_Magnitude_XYZ’]*-1
#     minima = find_peaks(y2,height = -5, distance = 1)
#     min_pos = data_frame.iloc[minima[0]] #list of the minima positions
#     min_height = y2.iloc[minima[0]] #list of the mirrored minima heights
    # return peaks
"""
Function to find the peaks and valleys
1. uses scipy find_peaks implementation.
2. paramter tuning is highly important !!!
3. todo
"""
def find_peaks_valleys(df,height,distance,prominence):
    # print("find_peaks",df)
    peaks = find_peaks(df['Value_Magnitude_XYZ'], height = height,  distance = distance,prominence=prominence)
    height = peaks[1]['peak_heights'] #list of the heights of the peaks
    peak_pos = df.iloc[peaks[0]] #list of the peaks positions
    # #Finding the minima
    y2 = df['Value_Magnitude_XYZ']*-1
    minima = find_peaks(y2,height = -5, distance = 1)
    min_pos = df.iloc[minima[0]] #list of the minima positions
    min_height = y2.iloc[minima[0]] #list of the mirrored minima heights
    # print("Peaks",peaks)
    return [peak_pos,min_pos,height,min_height]
"""
Function to calculate the CPR rate
1. finds time differences between peaks and return the average rate per minute
"""
def find_cpr_rate(peaks):
    time_diff_between_peaks=np.diff(peaks['EPOCH_Time_ms'])
    is_not_empty=len(time_diff_between_peaks) > 0
    if is_not_empty:
        avg_time_btwn_peaks_in_seconds_scipy = np.average(time_diff_between_peaks)/1000
        # print ("Average time between peaks in seconds (scipy): ", str(avg_time_btwn_peaks_in_seconds_scipy))
        # print("CPR Rate Per Minute (scipy): ", (1/avg_time_btwn_peaks_in_seconds_scipy)*60)
        return [(avg_time_btwn_peaks_in_seconds_scipy), ((1/avg_time_btwn_peaks_in_seconds_scipy)*60)]
    return [0,0]

""" 
Function to preprocess data
1. calculate magnitude of x,y,z values combined  sqrt(x^2 + y^2 + z^2) * eliminates effects from orientations
2. todo
"""
def preprocess_data(df):
    magnitude_xyz_df = np.sqrt(np.square(df[['Value_X_Axis','Value_Y_Axis','Value_Z_Axis']]).sum(axis=1))
    df['Value_Magnitude_XYZ'] = magnitude_xyz_df
    return df

       
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

# ----------------------------------------------------AUDIO--------------------------------------------------------

# AUDIO streaming variables and functions
host_name = socket.gethostname()
host_ip = '0.0.0.0' #socket.gethostbyname(host_name)
port = 50005
q = queue.Queue(maxsize=12800)
BUFF_SIZE = 1280 #65536

client_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
client_socket.bind((host_ip,port))

p = pyaudio.PyAudio()

RATE = 16000
CHUNK = int (RATE / 10)

def audio_stream_UDP():
    
    #PyAudio Streeam
    stream = p.open(format=p.get_format_from_width(2),
                    channels=1,
                    rate=RATE,#44100,
                    output=True,
                    frames_per_buffer=CHUNK)
                    
    # info = p.get_default_output_device_info()
    # print(info)
            
    # create socket
    message = b'Hello'
    # client_socket.sendto(message,(host_ip,port))
    socket_address = (host_ip,port)

    # receive and put audio data in queue
    def getAudioData():
        while True:
            frame,_= client_socket.recvfrom(BUFF_SIZE)
            q.put(frame)
            # print('Queue size...',q.qsize())
            
    t1 = threading.Thread(target=getAudioData, args=())
    t1.start()

    #write audio data to stream -- dtaa from this stream will be read by GoogleSpeech
    while True:
        frame = q.get()
        stream.write(frame)

# thread for streaming audio to the GUI 
t1 = threading.Thread(target=audio_stream_UDP, args=())
t1.start()

# client_socket.close()
# print('Audio stream socket closed')
# os._exit(1)


# ----------------------------------------------------VIDEO--------------------------------------------------------

#functions and variables for getting video stream from Android camera on AR glasses
TCP_IP = '127.0.0.1'
TCP_PORT = 9600

seq_all=0
dropped_imgs=0
total_imgs=0

if(sys.platform=="win32"):
    adbpath="C:\\Users\\sneha\\workspace\\platform-tools_r33.0.3-windows\\platform-tools\\adb.exe"
#if linux
else:
    adbpath="adb"

def connect():
    read_int=-1
    print("waiting for connection....")
    while(read_int!=100):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((TCP_IP, TCP_PORT))
            read_int=int.from_bytes(s.recv(1),"big")
        except Exception as e:
            print(e)
    print("connected")
    return s

# Media Pipe vars
global mp_drawing, mp_drawing_styles, mp_hands, mp_face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
# mp_pose = mp.solutions.pose

def process_image(image):
    """
    Adds annotations to image for the models you have selected, 
    For now, it just depict results from hand detection
    """
  
    global mp_hands #, mp_face_mesh
    hand_detection_results = None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # hand detection
    with mp_hands.Hands(
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
            hand_detection_results = hands.process(image)

    # annotations of results onto image
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # hand-detection annotations
    if hand_detection_results and hand_detection_results.multi_hand_landmarks:
        for hand_landmarks in hand_detection_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())

    return image


def get_next_image(s):

    global seq_all,dropped_imgs
    try:
        #start = time.time()
        d_type=int.from_bytes(s.recv(1),"big")
        fdist=int.from_bytes(s.recv(4),"big")/100
        if(d_type!=22):
            return -1
        seq=int.from_bytes(s.recv(4),"big")
        height=int.from_bytes(s.recv(4),"big")
        width=int.from_bytes(s.recv(4),"big")
        size=int.from_bytes(s.recv(4),"big")
        if(size>1000000):
            return -1
        img=bytearray()
        #print('received bytes : '+str(size))
        while(size>0):
            read_len=min(size,1024)
            data = s.recv(read_len)
            size -= len(data)
            img+=data
            
        image = Image.open(io.BytesIO(img))
        img_ar=np.array(image)
        if(seq_all==0):
            seq_all=seq
        elif(not(seq==(seq_all+1))):
            dropped_imgs+=1
        seq_all=seq
        #end = time.time()
        #print("time for getting next image: ", end - start)

    except:
        # print(traceback.format_exc())
        return -1
    return img_ar,fdist



# Code for Android App -- currently streams the audio and video data

#make sure the app is installed on the phone
print("port forwarding....")
ret=os.system(adbpath+" forward tcp:9600 tcp:9600")
if(ret==-1):
    print("port forwarding error. Exiting...")
    quit()
print("return value from OS = "+str(ret))

#stopping the app (we need to restart)
ret=os.system(adbpath + " shell am force-stop com.example.camstrm")
if(ret==-1):
    print("error when trying to stop the app. Exiting...")
    quit()

#start the app
print("starting the camstream app...")
ret=os.system(adbpath+" shell am start -n com.example.camstrm/com.example.camstrm.MainActivity --es operation " +str(0)+" --es camid " +str(0) +" --es dynamiclense " +str(1))
if(ret==-1):
    print("Error when starting app with adb. Exiting...")
    quit()
print("return value from OS = "+str(ret))

now = datetime.datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")

s=connect()
p = struct.pack('!i', 23)
s.send(p)


#Thread for streaming video to the GUI vision window
class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    changeVisInfo = pyqtSignal(str)

    def run(self):
        
        image=-1
        #total_imgs=0

        #array for y values for peak detection
        y_vals = []
        #array for timestamps when every image is received
        image_times = []

        self.changeVisInfo.emit(str("Measuring CPR Rate..."))

        with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
            while(True):
                try:
                    #get the next image frame
                    image=get_next_image(s) 
                    
                    # if image is valid
                    if(type(image)==tuple):

                        #get timestamp when getting the image frame -- for cpr rate detection
                        curr_time = round(time.time()*1000)
                        image_times.append(curr_time)

                        height,width,layers=image[0].shape
                        image = image[0]
                        # annotations of results onto image
                        image.flags.writeable = True

                        #process image with mediapipe hand detection
                        hand_detection_results = hands.process(image)

                        # hand-detection annotations
                        if hand_detection_results and hand_detection_results.multi_hand_landmarks:
                            for hand_landmarks in hand_detection_results.multi_hand_landmarks:

                                #append y_val to array for peak detection
                                y_vals.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)
                                
                                #for drawing hand annotations on image
                                mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())
                        else:
                            y_vals.append(0)#(math.nan)

                        #convert image to pixmap format and display on GUI
                        RGB_img = image
                        h, w, ch = RGB_img.shape
                        bytesPerLine = ch * w
                        convertToQtFormat = QImage(RGB_img.data, w, h, bytesPerLine, QImage.Format_RGB888)
                        p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                        self.changePixmap.emit(p)
               
                        #once we have around 100 image frames, calculate cpr rate
                        if(len(image_times) % 100 == 0):
                            #get running mean 
                            mean=np.convolve(y_vals, np.ones(50)/50, mode='valid')
                            mean=np.pad(mean,(len(y_vals)-len(mean),0),'edge')
                            #normalize by removing mean 
                            wrist_data_norm=y_vals-mean
                            #detect peaks for hand detection
                            peaks, _ = find_peaks(wrist_data_norm, height=0.005)
                            peak_times = np.take(image_times, peaks)

                            #find time difference between peaks and calculate cpr rate
                            time_diff_between_peaks=np.diff(peak_times)
                            avg_time_btwn_peaks_in_seconds = np.average(time_diff_between_peaks)/1000
                            avg_time = (avg_time_btwn_peaks_in_seconds)
                            cpr_rate = (1/avg_time_btwn_peaks_in_seconds)*60
                            # print ("Average time between peaks in seconds video): ", str(avg_time))
                            # print("CPR Rate Per Minute (video): ", cpr_rate)

                            #if cpr rate is nan, do not display nan value, instead just display a message like "measuring rate..."
                            if(math.isnan(cpr_rate)):
                                self.changeVisInfo.emit(str("Meauring CPR Rate..."))
                            #if we have non nan data values (i.e. valid data), then display that to Vision Information display box on GUI
                            else:
                                str_output = "Average time between peaks in seconds (video): " + str(round(avg_time,2)) + " \n" + "CPR Rate Per Minute (video): " + str(round(cpr_rate,2))
                                self.changeVisInfo.emit(str(str_output))
                        
                        if(len(image_times) == 10000):
                            #Clear the arrays to get 100 more image frames for cpr rate calculation
                            y_vals.clear()
                            image_times.clear()

                except:
                    print(traceback.format_exc())
                    print("error with video streaming")
        

        #print("done streaming image/video")
        #end            
        
        #display prerecorded video -- for demo purposes
        '''
        cap = cv2.VideoCapture("Sample_Video.mp4")
        while True:
            start = time.time()
            ret, frame = cap.read()
            if ret:
                im = process_image(frame)
                rgbImage = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                end = time.time()
                print("total time to process one frame: ", end - start)
        '''


# ================================================================== GUI ==================================================================
        

# Main Window of the Application

class MainWindow(QWidget):

    def __init__(self, width, height):
        super(MainWindow, self).__init__()
   
        # Fields
        self.width = width
        self.height = height
        self.SpeechThread = None
        self.CognitiveSystemThread = None
        self.stopped = 0
        self.reset = 0
        self.maximal = Amplitude()
        self.finalSpeechSegmentsSpeech = []
        self.finalSpeechSegmentsNLP = []
        self.nonFinalText = ""

        # Set geometry of the window
        self.setWindowTitle('CognitiveEMS Demo')
        self.setWindowIcon(QIcon('./Images/Logos/UVA.png'))
        self.setGeometry(int(width * .065), int(height * .070),
                         int(width * .9), int(height * .85))
        #self.setMinimumSize(1280, 750);
        #self.setFixedSize(width, height)
        # self.showMaximized()

        # Grid Layout to hold all the widgets
        self.Grid_Layout = QGridLayout(self)

        # Font for boxes
        #Box_Font = QFont()
        # Box_Font.setPointSize(BOX_FONT_SIZE)

        # Add disabled buttons to the of GUI to ensure spacing
        R3 = QPushButton(".\n.\n")
        R3.setFont(QFont("Monospace"))
        R3.setEnabled(False)
        R3.setStyleSheet("background-color:transparent;border:0;")
        #self.Grid_Layout.addWidget(R3, 3, 3, 1, 1)

        R6 = QPushButton(".\n.\n")
        R6.setFont(QFont("Monospace"))
        R6.setEnabled(False)
        R6.setStyleSheet("background-color:transparent;border:0;")
        #self.Grid_Layout.addWidget(R6, 6, 3, 1, 1)

        R8 = QPushButton(".\n" * int(self.height/100))
        R8.setFont(QFont("Monospace"))
        R8.setEnabled(False)
        R8.setStyleSheet("background-color:transparent;border:0;")
        #self.Grid_Layout.addWidget(R8, 8, 3, 1, 1)

        C0 = QPushButton("................................")
        C0.setFont(QFont("Monospace"))
        C0.setEnabled(False)
        self.Grid_Layout.addWidget(C0, 13, 0, 1, 1)

        C1 = QPushButton("................................")
        C1.setFont(QFont("Monospace"))
        C1.setEnabled(False)
        self.Grid_Layout.addWidget(C1, 13, 1, 1, 1)

        C2 = QPushButton(
            "....................................................................")
        C2.setFont(QFont("Monospace"))
        C2.setEnabled(False)
        self.Grid_Layout.addWidget(C2, 13, 2, 1, 2)

        # Create main title
        self.MainLabel = QLabel(self)
        self.MainLabel.setText('<font size="6"><b>CognitiveEMS</b></font>')
        self.Grid_Layout.addWidget(self.MainLabel, 0, 0, 1, 1)

        # Data Panel: To hold save state and generate form buttons
        self.DataPanel = QWidget()
        self.DataPanelGridLayout = QGridLayout(self.DataPanel)
        self.Grid_Layout.addWidget(self.DataPanel, 0, 2, 1, 2)

        # Create a save state button in the panel
        self.SaveButton = QPushButton('Save State', self)
        self.SaveButton.clicked.connect(self.SaveButtonClick)
        self.DataPanelGridLayout.addWidget(self.SaveButton, 0, 0, 1, 1)

        # Create a generate form button in the panel
        self.GenerateFormButton = QPushButton('Generate Form', self)
        #self.GenerateFormButton.clicked.connect(self.GenerateFormButtonClick)
        self.DataPanelGridLayout.addWidget(self.GenerateFormButton, 0, 1, 1, 1)

        # Create label and textbox for speech
        self.SpeechLabel = QLabel()
        self.SpeechLabel.setText("<b>Speech Recognition</b>")
        self.Grid_Layout.addWidget(self.SpeechLabel, 1, 0, 1, 1)

        self.SpeechSubLabel = QLabel()
        self.SpeechSubLabel.setText("(Transcript)")
        self.Grid_Layout.addWidget(self.SpeechSubLabel, 2, 0, 1, 1)

        self.SpeechBox = QTextEdit()
        self.SpeechBox.setReadOnly(True)
        # self.SpeechBox.setFont(Box_Font)
        self.SpeechBox.setOverwriteMode(True)
        self.SpeechBox.ensureCursorVisible()
        self.Grid_Layout.addWidget(self.SpeechBox, 3, 0, 2, 1)



        #Create label and media player for videos- - added 3/21/2022
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.video = QVideoWidget()
        
        # Create label and textbox for Vision Information
        self.VisionInformationLabel = QLabel()
        self.VisionInformationLabel.setText("<b>Vision Information</b>")
        self.Grid_Layout.addWidget(self.VisionInformationLabel, 5, 1, 1, 1)
        
        self.VisionInformation = QTextEdit() #QLineEdit()
        self.VisionInformation.setReadOnly(True)
        # self.VisionInformation.setFont(Box_Font)
        self.Grid_Layout.addWidget(self.VisionInformation, 6, 1, 1, 1)

        # Create label and textbox for Vision Information
        self.SmartwatchLabel = QLabel()
        self.SmartwatchLabel.setText("<b>Smartwatch Activity</b>")
        self.Grid_Layout.addWidget(self.SmartwatchLabel, 7, 1, 1, 1)
        
        self.Smartwatch = QTextEdit() #QLineEdit()
        self.Smartwatch.setReadOnly(True)
        # self.VisionInformation.setFont(Box_Font)
        self.Grid_Layout.addWidget(self.Smartwatch, 8, 1, 1, 1)
  
        self.VideoSubLabel = QLabel()
        self.VideoSubLabel.setText("<b>Video Content<b>") #setGeometry(100,100,100,100)
        self.Grid_Layout.addWidget(self.VideoSubLabel, 5, 0, 1, 1)

        self.video = QLabel(self)

        self.Grid_Layout.addWidget(self.video, 6, 0, 3, 1)

        VIDEO_WIDTH = 841
        VIDEO_HEIGHT = 511
        self.video.setGeometry(QtCore.QRect(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT))

        # Threads for video and smartwatch
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.changeVisInfo.connect(self.handle_message2)
        th.start()

        th2 = Thread_Watch(self)
        th2.changeActivityRec.connect(self.handle_message)
        th2.start()

        # th2 = ThreadAudio(self)
        # th2.start()

        # Control Panel: To hold combo box, radio, start, stop, and reset buttons
        self.ControlPanel = QWidget()
        self.ControlPanelGridLayout = QGridLayout(self.ControlPanel)
        self.Grid_Layout.addWidget(self.ControlPanel, 9, 0, 1, 2)

        # Audio Options Menu
        self.ComboBox = QComboBox()
        self.ComboBox.addItems(["Microphone",
                                "000_190105",
                                "001_190105",
                                "002_190105",
                                "003_190105",
                                "004_190105",
                                "005_190105",
                                "006_190105",
                                "007_190105",
                                "008_190105",
                                "009_190105",
                                "010_190105",
                                "011_190105",
                                "NG1",
                                "Other Audio File",
                                "Text File"])

        self.ControlPanelGridLayout.addWidget(self.ComboBox, 0, 0, 1, 1)

        # Radio Buttons Google or Other ML Model
        self.GoogleSpeechRadioButton = QRadioButton("Google Speech API", self)
        self.GoogleSpeechRadioButton.setEnabled(True)
        self.GoogleSpeechRadioButton.setChecked(True)
        self.ControlPanelGridLayout.addWidget(
        self.GoogleSpeechRadioButton, 0, 1, 1, 1)
        
        self.MLSpeechRadioButton = QRadioButton("Wav2Vec2", self)
        self.MLSpeechRadioButton.setEnabled(True) #changed from False to True to enable
        self.MLSpeechRadioButton.setChecked(False)
        self.ControlPanelGridLayout.addWidget(
        self.MLSpeechRadioButton, 0, 2, 1, 1)

        # Create a start button in the Control Panel
        self.StartButton = QPushButton('Start', self)
        self.StartButton.clicked.connect(self.StartButtonClick)
        self.ControlPanelGridLayout.addWidget(self.StartButton, 1, 0, 1, 1)

        # Create a stop button in the Control Panel
        self.StopButton = QPushButton('Stop', self)
        self.StopButton.clicked.connect(self.StopButtonClick)
        self.ControlPanelGridLayout.addWidget(self.StopButton, 1, 1, 1, 1)

        # Create a reset  button in the Control Panel
        self.ResetButton = QPushButton('Reset', self)
        self.ResetButton.clicked.connect(self.ResetButtonClick)
        self.ControlPanelGridLayout.addWidget(self.ResetButton, 1, 2, 1, 1)

        # VU Meter Panel
        self.VUMeterPanel = QWidget()
        self.VUMeterPanelGridLayout = QGridLayout(self.VUMeterPanel)
        self.Grid_Layout.addWidget(self.VUMeterPanel, 10, 0, 1, 2)

        # Add Microphone Logo to the VU Meter Panel
        self.MicPictureBox = QLabel(self)
        self.MicPictureBox.setPixmap(QPixmap('./Images/Logos/mic2.png'))
        self.VUMeterPanelGridLayout.addWidget(self.MicPictureBox, 0, 0, 1, 1)

        # Add the VU Meter progress bar to VU Meter Panel
        self.VUMeter = QProgressBar()
        self.VUMeter.setMaximum(100)
        self.VUMeter.setValue(0)
        self.VUMeter.setTextVisible(False)
        self.VUMeterPanelGridLayout.addWidget(self.VUMeter, 0, 1, 1, 1)

        # Create label and textbox for Concept Extraction
        self.ConceptExtractionLabel = QLabel()
        self.ConceptExtractionLabel.setText("<b>Concept Extraction</b>")
        self.Grid_Layout.addWidget(self.ConceptExtractionLabel, 1, 1, 1, 1)

        self.ConceptExtractionSubLabel = QLabel()
        self.ConceptExtractionSubLabel.setText(
            "(Concept, Presence, Value, Confidence)")
        self.Grid_Layout.addWidget(self.ConceptExtractionSubLabel, 2, 1, 1, 1)

        self.ConceptExtraction = QTextEdit()
        self.ConceptExtraction.setReadOnly(True)
        # self.ConceptExtraction.setFont(Box_Font)
        self.Grid_Layout.addWidget(self.ConceptExtraction, 3, 1, 2, 1)
        

        # Add label, textbox for protcol name
        self.ProtcolLabel = QLabel()
        self.ProtcolLabel.setText("<b>Suggested EMS Protocols</b>")
        self.Grid_Layout.addWidget(self.ProtcolLabel, 1, 2, 1, 2)

        self.ProtcolSubLabel = QLabel()
        self.ProtcolSubLabel.setText("(Protocol, Confidence)")
        self.Grid_Layout.addWidget(self.ProtcolSubLabel, 2, 2, 1, 2)

        self.ProtocolBox = QTextEdit()
        self.ProtocolBox.setReadOnly(True)
        # self.ProtocolBox.setFont(Box_Font)
        self.Grid_Layout.addWidget(self.ProtocolBox, 3, 2, 1, 2)

        # Create label and textbox for interventions
        self.InterventionLabel = QLabel()
        self.InterventionLabel.setText("<b>Suggested Interventions</b>")
        self.Grid_Layout.addWidget(self.InterventionLabel, 4, 2, 1, 2)

        self.InterventionSubLabel = QLabel()
        self.InterventionSubLabel.setText("(Intervention, Confidence)")
        self.Grid_Layout.addWidget(self.InterventionSubLabel, 5, 2, 1, 2)

        self.InterventionBox = QTextEdit()
        self.InterventionBox.setReadOnly(True)
        # self.InterventionBox.setFont(Box_Font)
        self.Grid_Layout.addWidget(self.InterventionBox, 6, 2, 1, 2)

        # Create label and textbox for messages
        self.MsgBoxLabel = QLabel()
        self.MsgBoxLabel.setText("<b>System Messages Log</b>")
        self.Grid_Layout.addWidget(self.MsgBoxLabel, 7, 2, 1, 2)

        self.MsgBox = QTextEdit()
        self.MsgBox.setReadOnly(True)
        self.MsgBox.setFont(QFont("Monospace"))
        # self.MsgBox.setLineWrapMode(QTextEdit.NoWrap)
        self.Grid_Layout.addWidget(self.MsgBox, 8, 2, 1, 2)

        # Populate the Message Box with welcome message
        System_Info_Text_File = open("./ETC/System_Info.txt", "r")
        System_Info_Text = ""
        for line in System_Info_Text_File.readlines():
            System_Info_Text += line
        System_Info_Text_File.close()
        self.MsgBox.setText(System_Info_Text + "\n" + str(datetime.datetime.now().strftime("%c")) + " - Ready to start speech recognition!")
        self.MsgBox.setText(System_Info_Text)
        self.UpdateMsgBox(["Ready to start speech recognition!"])

        # Add Link Lab Logo
        self.PictureBox = QLabel()
        self.PictureBox.setPixmap(QPixmap('./Images/Logos/LinkLabLogo.png'))
        self.PictureBox.setAlignment(Qt.AlignCenter)
        self.Grid_Layout.addWidget(self.PictureBox, 10, 2, 1, 2)

    # ================================================================== GUI Functions ==================================================================
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.video.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(str)
    def handle_message(self, message):
        self.Smartwatch.setText(message)
    
    @pyqtSlot(str)
    def handle_message2(self, message):
        self.VisionInformation.setText(message)

    #video playing -- added 3/21/2022
    def play(self):
        print("hello")
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        if self.mediaPlayer.state() == QMediaPlayer.StoppedState:
            print("video has stopped playing!")
            self.VisionInformation.setPlainText("CPR Done\nAverage Compression Rate: 140 bpm")
        
        else:
            print("testing successful")
            self.mediaPlayer.play()

            
    def mediaStateChanged(self, state):
    	if self.player.state() == QMediaPlayer.StoppedState:
            print("video has stopped playing!")
            self.VisionInformation.setPlainText("CPR Done\nAverage Compression Rate: 140 bpm")
        
    # Called when closing the GUI
    def closeEvent(self, event):
        print('Closing GUI')
        # self.th2.exit()
        self.stopped = 1
        self.reset = 1
        SpeechToNLPQueue.put('Kill')
        event.accept()

    @pyqtSlot()
    def SaveButtonClick(self):
        #name = QFileDialog.getSaveFileName(self, 'Save File')
        name = str(datetime.datetime.now().strftime("%c")) + ".txt"
        #file = open("./Dumps/" + name,'w')
        mode_text = self.ComboBox.currentText()
        speech_text = str(self.SpeechBox.toPlainText())
        concept_extraction_text = str(self.ConceptExtraction.toPlainText())
        protocol_text = str(self.ProtocolBox.toPlainText())
        intervention_text = str(self.InterventionBox.toPlainText())
        msg_text = str(self.MsgBox.toPlainText())
        #text = "Mode:\n\n" + mode_text + "\n\nSpeech: \n\n" + speech_text +"\n\nConcept Extraction:\n\n" + concept_extraction_text + "\n\nProtocol Text:\n\n" + protocol_text + "\n\nIntervention:\n\n" + intervention_text + "\n\nSystem Messages Log:\n\n" + msg_text
        # file.write(text)
        # file.close()
        #self.UpdateMsgBox(["System state dumped to \n/Dumps/" + name])
        self.UpdateMsgBox(["System results saved to results.csv"])
        results = [speech_text, concept_extraction_text,
                   protocol_text, intervention_text]
        with open("results.csv", mode="a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(results)

    @pyqtSlot()
    def StartButtonClick(self):
        print('Start pressed!')
        self.UpdateMsgBox(["Starting!"])
        self.reset = 0
        self.stopped = 0
        self.StartButton.setEnabled(False)
        self.StopButton.setEnabled(True)
        self.ComboBox.setEnabled(False)
        self.ResetButton.setEnabled(False)

        # ==== Start the Speech/Text Thread
        # hacky bypass for demo
        # if True:
        #     self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(
        #             self, SpeechToNLPQueue, './Audio_Scenarios/test5.wav',))
        #     # self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(
        #     #         self, SpeechToNLPQueue, './Audio_Scenarios/2019_Test/002_190105.wav',))
        #     self.SpeechThread.start()
        #     print("Demo Audio Started")

        # If Microphone
        if(self.ComboBox.currentText() == 'Microphone'):
            if(self.GoogleSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(
                    target=GoogleSpeechMicStream.GoogleSpeech, args=(self, SpeechToNLPQueue,))
            elif(self.MLSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(
                    target=WavVecMicStream.WavVec, args=(self, SpeechToNLPQueue,))
            self.SpeechThread.start()
            print('Microphone Speech Thread Started')

        # If Other Audio File
        elif(self.ComboBox.currentText() == 'Other Audio File'):
            audio_fname = QFileDialog.getOpenFileName(
                self, 'Open file', 'c:\\', "Wav files (*.wav)")
            if(self.GoogleSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(
                    self, SpeechToNLPQueue, str(audio_fname),))
            elif(self.MLSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(target=WavVecFileStream.WavVec, args=(self, SpeechToNLPQueue, str(audio_fname),)) #(target=DeepSpeechFileStream.DeepSpeech, args=(self, SpeechToNLPQueue, str(audio_fname),))
            self.otheraudiofilename = str(audio_fname)
            self.SpeechThread.start()
            print("Other Audio File Speech Thread Started")

        # If Text File
        elif(self.ComboBox.currentText() == 'Text File'):
            text_fname = QFileDialog.getOpenFileName(
                self, 'Open file', 'c:\\', "Text files (*.txt)")
            self.SpeechThread = StoppableThread(target=TextSpeechStream.TextSpeech, args=(
                self, SpeechToNLPQueue, str(text_fname),))
            self.SpeechThread.start()
            print("Text File Speech Thread Started")

        # If a Hard-coded Audio test file
        else:
            if(self.GoogleSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(
                    self, SpeechToNLPQueue, './Audio_Scenarios/2019_Test/' + str(self.ComboBox.currentText()) + '.wav',))
            elif(self.MLSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(target=WavVecFileStream.WavVec, args=(
                    self, SpeechToNLPQueue, './Audio_Scenarios/2019_Test/' + str(self.ComboBox.currentText()) + '.wav',)) #(target=DeepSpeechFileStream.DeepSpeech, args=( self, SpeechToNLPQueue, './Audio_Scenarios/2019_Test/' + str(self.ComboBox.currentText()) + '.wav',))
            self.SpeechThread.start()
            print("Hard-coded Audio File Speech Thread Started")

        # ==== Start the Cognitive System Thread
        if(self.CognitiveSystemThread == None):
            print("Cognitive System Thread Started")
            self.CognitiveSystemThread = StoppableThread(
                target=CognitiveSystem.CognitiveSystem, args=(self, SpeechToNLPQueue,))
            self.CognitiveSystemThread.start()
    

    @pyqtSlot()
    def StopButtonClick(self):
        print("Stop pressed!")
        self.UpdateMsgBox(["Stopping!"])
        self.stopped = 1
        time.sleep(.1)
        self.StartButton.setEnabled(True)
        self.ComboBox.setEnabled(True)
        self.ResetButton.setEnabled(True)

    @pyqtSlot()
    def GenerateFormButtonClick(self):
        print('Generate Form pressed!')
        self.UpdateMsgBox(["Form Being Generated!"])
        text = str(self.SpeechBox.toPlainText())
        #StoppableThread(target = GenerateForm.GenerateForm, args=(self, text,)).start()

    @pyqtSlot()
    def ResetButtonClick(self):
        print('Reset pressed!')
        self.UpdateMsgBox(["Resetting!"])
        self.reset = 1
        self.stopped = 1

        if(self.CognitiveSystemThread != None):
            SpeechToNLPQueue.put('Kill')

        self.VUMeter.setValue(0)
        self.finalSpeechSegmentsSpeech = []
        self.finalSpeechSegmentsNLP = []
        self.SpeechBox.clear()
        self.ConceptExtraction.setText('')
        self.ProtocolBox.setText('')
        self.InterventionBox.setText('')
        self.nonFinalText = ""
        self.SpeechThread = None
        self.CognitiveSystemThread = None
        time.sleep(.1)
        self.StartButton.setEnabled(True)

    def ClearWindows(self):
        self.finalSpeechSegmentsSpeech = []
        self.finalSpeechSegmentsNLP = []
        self.nonFinalText = ""
        self.SpeechBox.clear()
        self.ConceptExtraction.setText('')
        self.ProtocolBox.setText('')
        self.InterventionBox.setText('')

    # Update the Speech Box
    #@pyqtSlot
    def UpdateSpeechBox(self, input):
        
        item = input[0]

        if(item.isFinal):
            if(item.origin == 'Speech'):
                self.finalSpeechSegmentsSpeech.append(
                    '<b>' + item.transcript + '</b>')
            elif(item.origin == 'NLP'):
                self.finalSpeechSegmentsNLP.append(
                    '<b>' + item.transcript + '</b>')

            text = ""

            for a in self.finalSpeechSegmentsNLP:
                text += a + "<br>"

            for a in self.finalSpeechSegmentsSpeech[len(self.finalSpeechSegmentsNLP):]:
                text += a + "<br>"

            self.SpeechBox.setText('<b>' + text + '</b>')
            self.SpeechBox.moveCursor(QTextCursor.End)
            self.nonFinalText = ""

        else:
            text = ""

            for a in self.finalSpeechSegmentsNLP:
                text += a + "<br>"

            for a in self.finalSpeechSegmentsSpeech[len(self.finalSpeechSegmentsNLP):]:
                text += a + "<br>"

            if(not len(text)):
                text = "<b></b>"

            previousTextMinusPrinted = self.nonFinalText[:len(
                self.nonFinalText) - item.numPrinted]
            self.nonFinalText = previousTextMinusPrinted + item.transcript
            self.SpeechBox.setText(text + self.nonFinalText)
            self.SpeechBox.moveCursor(QTextCursor.End)
            
            

    # Update the Concept Extraction Box
    def UpdateConceptExtractionBox(self, input):
        global chunkdata
        item = input[0]
        self.ConceptExtraction.setText(item)
        if item != "":
            chunkdata.append(item)
        else:
            chunkdata.append("-")
            
      

    # Update the Protocols and Interventions Boxes
    def UpdateProtocolBoxes(self, input):
        global chunkdata
        protocol_names = input[0]
        interventions = input[1]
        self.ProtocolBox.setText(protocol_names)
        self.InterventionBox.setText(interventions)

        chunkdata.append(protocol_names)
        chunkdata.append(interventions)
        with open("check.csv", mode="a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(chunkdata)
        chunkdata = []

    # Update the System Message Box
    def UpdateMsgBox(self, input):
        item = input[0]
        #previousText = self.MsgBox.toPlainText()
        #self.MsgBox.setText(str(previousText).strip() + "\n" + str(datetime.datetime.now().strftime("%c"))[16:] + " - " + item + "\n")
        self.MsgBox.append(
            "<b>" + str(datetime.datetime.now().strftime("%c"))[16:] + "</b> - " + item)

        self.MsgBox.moveCursor(QTextCursor.End)

    # Update the VU Meter
    def UpdateVUBox(self, input):
        signal = input[0]
        try:
            amp = Amplitude.from_data(signal)
            if amp > self.maximal:
                self.maximal = amp

            value = amp.getValue()
            value = value * 100
            log_value = 50 * math.log10(value + 1)  # Plot on a log scale

            self.VUMeter.setValue(log_value)
        except:
            self.VUMeter.setValue(0)

    # Restarts Google Speech API, called when the API limit is reached
    def StartGoogle(self, input):
        item = input[0]
        if(item == 'Mic'):
            self.SpeechThread = StoppableThread(
                target=GoogleSpeechMicStream.GoogleSpeech, args=(self, SpeechToNLPQueue, ))
            self.SpeechThread.start()
        elif(item == 'File'):
            if self.ComboBox.currentText() == 'Other Audio File':
                print("\n\nStart Again\n\n")
                self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(
                    self, SpeechToNLPQueue, self.otheraudiofilename,))
            else:
                self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(
                    self, SpeechToNLPQueue, './Audio_Scenarios/2019_Test/' + str(self.ComboBox.currentText()) + '.wav',))
            self.SpeechThread.start()

    # Enabled and/or disable given buttons in a tuple (Button Object, True/False)
    def ButtonsSetEnabled(self, input):
        for item in input:
            item[0].setEnabled(item[1])
    

# ================================================================== Main ==================================================================
if __name__ == '__main__':

    # Set the Google Speech API service-account key environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"

    # Create thread-safe queue for communication between Speech and Cognitive System threads
    SpeechToNLPQueue = queue.Queue()

    # GUI: Create the main window, show it, and run the app
    print("Starting GUI")
    app = QApplication(sys.argv)
    screen_resolution = app.desktop().screenGeometry()
    width, height = screen_resolution.width(), screen_resolution.height()

    #width = 1920
    #height = 1080
    #width = 1366
    #height = 768
    print("Screen Resolution\nWidth: %s\nHeight: %s" % (width, height))
    Window = MainWindow(width, height)
    #Window.StartButtonClick()
    Window.show()

    sys.exit(app.exec_())