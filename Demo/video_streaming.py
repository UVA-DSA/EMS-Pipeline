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
TCP_PORT = 8899

seq_all=0
dropped_imgs=0
total_imgs=0


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



class Thread(QThread):
    def __init__(self, var, bool):
        self.data_path_str = var + "videodata/"
        self.videoStreamBool = bool
        super().__init__()

    changePixmap = pyqtSignal(QImage)
    changeVisInfo = pyqtSignal(str)
        
   

    def run(self):
        recording_enabled = True
        frame_index = 0

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
        sock.bind(('0.0.0.0', TCP_PORT))  
        sock.listen(5)  

        print("Waiting for client...")

        connection,address = sock.accept()  
        print("Client connected: ",address)

       
        #for peak detection
        y_vals = []

        #array for timestamps when every image is received
        image_times = []

        curr_date = datetime.datetime.now()
        dt_string = curr_date.strftime("%d-%m-%Y-%H-%M-%S")

        with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
            if self.videoStreamBool == True:
                if not os.path.exists(self.data_path_str + dt_string):
                    os.makedirs(self.data_path_str + dt_string)
                with open(self.data_path_str + "viddata.csv", 'w', newline='') as file:
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
                            original_img = Image.open(io.BytesIO(img_bytes))
                            img_ar=np.array(image)
                            original_img=np.array(original_img)
                            image = img_ar

                            #process image with mediapipe hand detection
                            hand_detection_results = hands.process(image)

                            # hand-detection annotations
                            if hand_detection_results and hand_detection_results.multi_hand_landmarks:
                                self.changeVisInfo.emit(str("Hand Detected! Wrist Position Identified."))
                                for hand_landmarks in hand_detection_results.multi_hand_landmarks:

                                    #append y_val to array for peak detection
                                    y_vals.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)
                                    
                                    #for drawing hand annotations on image
                                    mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())
                            else:
                                self.changeVisInfo.emit(str("Detecting Hands..."))
                                y_vals.append(0)#(math.nan)


                            #convert image to pixmap format and display on GUI
                            RGB_img = image
                            RGB_img = cv2.rotate(RGB_img, cv2.ROTATE_180)
                            vid_img = cv2.cvtColor(RGB_img, cv2.COLOR_BGR2RGB)
                            

                            #write to file
                            if(recording_enabled and not is_video_created):

                                height,width,layers=img_ar.shape
                                fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
                                video = cv2.VideoWriter(self.data_path_str+'/'+dt_string+'.avi', fourcc, 30, (width, height))
                                is_video_created = True

                            if(recording_enabled):
                                now = time.time()*1e3

                                img_ar = cv2.cvtColor(img_ar, cv2.COLOR_BGR2RGB)

                                cv2.imwrite(self.data_path_str+ dt_string + '/img_'+str(frame_index)+'.jpg', img_ar)
                                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

                                cv2.imwrite(self.data_path_str+ dt_string + '/img_'+str(frame_index)+'_original.jpg', original_img)
                                # video.write(vid_img)
                                video.write(original_img)
                                writer.writerow([frame_index, now, timestamp])
                                frame_index += 1
                            
                            h, w, ch = RGB_img.shape
                            # print("Image Size",h,w)
                            bytesPerLine = ch * w
                            convertToQtFormat = QImage(RGB_img.data, w, h, bytesPerLine, QImage.Format_RGB888)
                            p = convertToQtFormat.scaled(2*640, 2*480, Qt.KeepAspectRatio)
                            self.changePixmap.emit(p)
                            
                            if(len(image_times) == 10000):
                                #Clear the arrays to get 100 more image frames for cpr rate calculation
                                y_vals.clear()
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

                        
                        #process image with mediapipe hand detection
                        hand_detection_results = hands.process(image)

                        # hand-detection annotations
                        if hand_detection_results and hand_detection_results.multi_hand_landmarks:
                            self.changeVisInfo.emit(str("Hand Detected! Wrist Position Identified."))
                            for hand_landmarks in hand_detection_results.multi_hand_landmarks:

                                #append y_val to array for peak detection
                                y_vals.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)
                                
                                #for drawing hand annotations on image
                                mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())
                        else:
                            self.changeVisInfo.emit(str("Detecting Hands..."))
                            y_vals.append(0)#(math.nan)


                        #convert image to pixmap format and display on GUI
                        RGB_img = image
                        RGB_img = cv2.rotate(RGB_img, cv2.ROTATE_180)
                        
                        h, w, ch = RGB_img.shape
                        # print("Image Size",h,w)
                        bytesPerLine = ch * w
                        convertToQtFormat = QImage(RGB_img.data, w, h, bytesPerLine, QImage.Format_RGB888)
                        p = convertToQtFormat.scaled(2*640, 2*480, Qt.KeepAspectRatio)
                        self.changePixmap.emit(p)
                        
                        if(len(image_times) == 10000):
                            #Clear the arrays to get 100 more image frames for cpr rate calculation
                            y_vals.clear()
                            image_times.clear()

                    else:
                        # continue
                        print("Reconnecting to a client...")
                        connection.close()
                

                        connection,address = sock.accept()  
                        print("Client connected: ",address)










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