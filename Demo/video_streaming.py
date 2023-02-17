#Thread for streaming video to the GUI vision window

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

from cpr_calculation import vid_streaming_Cpr

# Media Pipe vars
global mp_drawing, mp_drawing_styles, mp_hands, mp_face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
# mp_pose = mp.solutions.pose

#functions and variables for getting video stream from Android camera on AR glasses
TCP_IP = '127.0.0.1'
TCP_PORT = 9600

seq_all=0
dropped_imgs=0
total_imgs=0

# if(sys.platform=="win32"):
#     adbpath="C:\\Users\\sneha\\workspace\\platform-tools_r33.0.3-windows\\platform-tools\\adb.exe"
# #if linux
# else:
#     adbpath="adb"

# def connect():
#     read_int=-1
#     print("waiting for connection....")
#     while(read_int!=100):
#         try:
#             s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#             s.connect((TCP_IP, TCP_PORT))
#             read_int=int.from_bytes(s.recv(1),"big")
#         except Exception as e:
#             print(e)
#     print("connected")
#     return s




# # Code for Android App -- currently streams the audio and video data

# #make sure the app is installed on the phone
# print("port forwarding....")
# ret=os.system(adbpath+" forward tcp:9600 tcp:9600")
# if(ret==-1):
#     print("port forwarding error. Exiting...")
#     quit()
# print("return value from OS = "+str(ret))

# #stopping the app (we need to restart)
# ret=os.system(adbpath + " shell am force-stop com.example.camstrm")
# if(ret==-1):
#     print("error when trying to stop the app. Exiting...")
#     quit()

# #start the app
# print("starting the camstream app...")
# ret=os.system(adbpath+" shell am start -n com.example.camstrm/com.example.camstrm.MainActivity --es operation " +str(0)+" --es camid " +str(0) +" --es dynamiclense " +str(1))
# if(ret==-1):
#     print("Error when starting app with adb. Exiting...")
#     quit()
# print("return value from OS = "+str(ret))

now = datetime.datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")

# s=connect()
# p = struct.pack('!i', 23)
# s.send(p)

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


# def get_next_image(s):

#     global seq_all,dropped_imgs
#     try:
#         #start = time.time()
#         d_type=int.from_bytes(s.recv(1),"big")
#         fdist=int.from_bytes(s.recv(4),"big")/100
#         if(d_type!=22):
#             return -1
#         seq=int.from_bytes(s.recv(4),"big")
#         height=int.from_bytes(s.recv(4),"big")
#         width=int.from_bytes(s.recv(4),"big")
#         size=int.from_bytes(s.recv(4),"big")
#         if(size>1000000):
#             return -1
#         img=bytearray()
#         #print('received bytes : '+str(size))
#         while(size>0):
#             read_len=min(size,1024)
#             data = s.recv(read_len)
#             size -= len(data)
#             img+=data
            
#         image = Image.open(io.BytesIO(img))
#         img_ar=np.array(image)
#         if(seq_all==0):
#             seq_all=seq
#         elif(not(seq==(seq_all+1))):
#             dropped_imgs+=1
#         seq_all=seq
#         #end = time.time()
#         #print("time for getting next image: ", end - start)

#     except:
#         # print(traceback.format_exc())
#         return -1
#     return img_ar,fdist


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    changeVisInfo = pyqtSignal(str)

    def run(self):

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
        sock.bind(('0.0.0.0', 8899))  
        sock.listen(5)  

        print("Waiting for client...")

        connection,address = sock.accept()  
        print("Client connected: ",address)

        #for peak detection
        y_vals = []

        #array for timestamps when every image is received
        image_times = []

        with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
            while True:
                if( int.from_bytes(connection.recv(1),"big") == 22):
                    print("Image received")

                    timestamp = int.from_bytes(connection.recv(8),"big")
                    print("Got timestamp:" , timestamp)

                    img_byte_length = int.from_bytes(connection.recv(8),"big")
                    print("Got Num of bytes in the image:" , img_byte_length)

                    # img_buffer_size = math.ceil(img_byte_length/1024)*1024
                    img_buffer_size = img_byte_length
                    print("Buff Size Should Be: ", img_buffer_size)

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

                    #CPR CALCULATION

                    #once we have around 100 image frames, calculate cpr rate
                    if(len(image_times) % 100 == 0):
                        #get avg_time_btwn_peaks_in_seconds
                        avg_time_btwn_peaks_in_seconds = vid_streaming_Cpr(y_vals, image_times)
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

                else:
                    print("Garbage")










        # image=-1
        # #total_imgs=0

        # #array for y values for peak detection
        # y_vals = []
        # #array for timestamps when every image is received
        # image_times = []

        # self.changeVisInfo.emit(str("Measuring CPR Rate..."))

        # with mp_hands.Hands(
        # max_num_hands=1,
        # model_complexity=0,
        # min_detection_confidence=0.5,
        # min_tracking_confidence=0.5) as hands:
        #     while(True):
        #         try:
        #             #get the next image frame
        #             image=get_next_image(s) 
                    
        #             # if image is valid
        #             if(type(image)==tuple):

        #                 #get timestamp when getting the image frame -- for cpr rate detection
        #                 curr_time = round(time.time()*1000)
        #                 image_times.append(curr_time)

        #                 height,width,layers=image[0].shape
        #                 image = image[0]
        #                 # annotations of results onto image
        #                 image.flags.writeable = True

        #                 #process image with mediapipe hand detection
        #                 hand_detection_results = hands.process(image)

        #                 # hand-detection annotations
        #                 if hand_detection_results and hand_detection_results.multi_hand_landmarks:
        #                     for hand_landmarks in hand_detection_results.multi_hand_landmarks:

        #                         #append y_val to array for peak detection
        #                         y_vals.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)
                                
        #                         #for drawing hand annotations on image
        #                         mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())
        #                 else:
        #                     y_vals.append(0)#(math.nan)

        #                 #convert image to pixmap format and display on GUI
        #                 RGB_img = image
        #                 h, w, ch = RGB_img.shape
        #                 bytesPerLine = ch * w
        #                 convertToQtFormat = QImage(RGB_img.data, w, h, bytesPerLine, QImage.Format_RGB888)
        #                 p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
        #                 self.changePixmap.emit(p)
               
        #                 #once we have around 100 image frames, calculate cpr rate
        #                 if(len(image_times) % 100 == 0):
        #                     #get running mean 
        #                     mean=np.convolve(y_vals, np.ones(50)/50, mode='valid')
        #                     mean=np.pad(mean,(len(y_vals)-len(mean),0),'edge')
        #                     #normalize by removing mean 
        #                     wrist_data_norm=y_vals-mean
        #                     #detect peaks for hand detection
        #                     peaks, _ = find_peaks(wrist_data_norm, height=0.005)
        #                     peak_times = np.take(image_times, peaks)

        #                     #find time difference between peaks and calculate cpr rate
        #                     time_diff_between_peaks=npgettext.diff(peak_times)
        #                     avg_time_btwn_peaks_in_seconds = np.average(time_diff_between_peaks)/1000
        #                     avg_time = (avg_time_btwn_peaks_in_seconds)
        #                     cpr_rate = (1/avg_time_btwn_peaks_in_seconds)*60
        #                     # print ("Average time between peaks in seconds video): ", str(avg_time))
        #                     # print("CPR Rate Per Minute (video): ", cpr_rate)

        #                     #if cpr rate is nan, do not display nan value, instead just display a message like "measuring rate..."
        #                     if(math.isnan(cpr_rate)):
        #                         self.changeVisInfo.emit(str("Meauring CPR Rate..."))
        #                     #if we have non nan data values (i.e. valid data), then display that to Vision Information display box on GUI
        #                     else:
        #                         str_output = "Average time between peaks in seconds (video): " + str(round(avg_time,2)) + " \n" + "CPR Rate Per Minute (video): " + str(round(cpr_rate,2))
        #                         self.changeVisInfo.emit(str(str_output))
                        
        #                 if(len(image_times) == 10000):
        #                     #Clear the arrays to get 100 more image frames for cpr rate calculation
        #                     y_vals.clear()
        #                     image_times.clear()

        #         except:
        #             print(traceback.format_exc())
        #             print("error with video streaming")
        

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