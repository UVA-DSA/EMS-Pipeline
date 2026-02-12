#Thread for streaming video to the GUI vision window

import csv
import os
import scipy
import time
import math
import datetime
import time

import asyncio

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

from socketio import Client

import base64
import io
import threading
import time

from multiprocessing import Process, Queue

from EMS_Vision.ObjectDetector import ObjectDetector
from pipeline_config import socketio_ipaddr

from torch import multiprocessing

import queue
# Media Pipe vars
global mp_drawing, mp_drawing_styles, mp_hands, mp_face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# mp_pose = mp.solutions.pose

seq_all=0
dropped_imgs=0
total_imgs=0

image_queue = Queue(maxsize=1)
display_queue = Queue()
signal_queue = Queue()

#multiprocessing.set_start_method('forkserver')


def process_image(image):
    """
    Adds annotations to image for the models you have selected,
    For now, it just depicts results from hand detection
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



class VideoThread(QThread):


    changePixmap = pyqtSignal(QImage)
    changeVisInfo = pyqtSignal(str)


    def __init__(self, var, bool):
        super().__init__()

        self.data_path_str = var + "videodata/"
        self.videoStreamBool = bool
        self.is_running = True
        self.imagequeue = image_queue

        #self.feedback_client = FeedbackClient(self)

        # self.sio = Client() #TODO: add similar to feedback.py

        self.PIPE_VIDEO = "/tmp/emsvid"
        self.video_pipe_holder = None
        print('DISPLAY INIT!!!!')





        # Create an asyncio event loop for this thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.display_thread = threading.Thread(target=self.display_image, args=(self.changePixmap, display_queue,))

        self.mediapipe_thread = threading.Thread(target=self.process_image)

        self.object_detector = ObjectDetector(image_queue, display_queue, signal_queue)

        self.mp_hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.mp_drawing = mp.solutions.drawing_utils


        # @self.sio.on('server_video')
        def on_message(data):


            # self.imagequeue.put(data)

            # Schedule the asynchronous image processing task
            asyncio.run_coroutine_threadsafe(self.process_image_async(data), self.loop)


            # start = time.time()
            #         # byte_array_string consists a string of base64 encoded bmp image
            # byte_array = base64.b64decode(data)

            # # print("received video: ", (byte_array))

            # # Step 2: Create an image object from the binary data
            # image = Image.open(io.BytesIO(byte_array))

            #     # Convert the PIL image object to an OpenCV image (numpy array)
            # # cv2_img = np.array(image)

            # # cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)

            # # cv2_img = cv2.rotate(cv2_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # # cv2_img = cv2.resize(cv2_img, (640, 480))

            # RGB_img = np.array(image)

            # RGB_img = cv2.rotate(RGB_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # RGB_img = cv2.resize(RGB_img, (640, 480))


            # h, w, ch = RGB_img.shape
            # # print("Image Size",h,w)
            # bytesPerLine = ch * w
            # convertToQtFormat = QImage(RGB_img.data, 640, 480, bytesPerLine, QImage.Format_RGB888)
            # p = convert//ToQtFormat.scaled(w,h, Qt.KeepAspectRatio)

            # print("Time taken to display image: ", time.time()-start)
            # self.changePixmap.emit(p)

            # self.imagequeue.put(data)

        # self.object_detector.start()
        # print("Object Detector Process started")

        # self.mediapipe_thread.start()


    async def process_image_async(self, data):
        """Asynchronous image processing."""
        start = time.time()

        # Decode the image
        byte_array = base64.b64decode(data)
        image = Image.open(io.BytesIO(byte_array))
        RGB_img = np.array(image)


        # # Process the image
        # RGB_img = cv2.rotate(RGB_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # RGB_img = cv2.resize(RGB_img, (640, 480))

        # h, w, ch = RGB_img.shape
        # bytesPerLine = ch * w
        # qImg = QImage(RGB_img.data, 640, 480, bytesPerLine, QImage.Format_RGB888).scaled(w, h, Qt.KeepAspectRatio)

        # # print("Time taken to display image: ", time.time() - start)
        # self.changePixmap.emit(qImg)  # Emit signal to update GUI
        # print("[Video_Streaming] Image received")
        self.imagequeue.put(RGB_img,block=False)



    def stop(self):
        print("[INFO] Video Stream Thread stopping..")
        self.is_running = False
        # self.sio.disconnect()
        # self.sio.eio.disconnect()
        signal_queue.put("stop")
        # print("[INFO] Socketio Disconnected")

        # self.display_thread.join()

        # self.sio = None
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.quit()

        print("[INFO] Video Stream Thread Stopped")




    def display_image(self,changePixmap, display_queue):



        while self.is_running:
            # print('qsize', str(display_queue.qsize()))
            # if display_queue.empty():
            #     continue
            if display_queue.qsize() == 0:
                time.sleep(0.001)
                continue
            else:
                try:
                    frame = display_queue.get_nowait()
                    RGB_img = frame[:, :, ::-1].copy()
                    # print('displaying image')
                except Exception as e:
                    # print(f'[display_image] Error: {type(e).__name__}: {e}')
                    # import traceback
                    # traceback.print_exc()
                    continue
                # print('qsize', str(display_queue.qsize()))


                h, w, ch = RGB_img.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(RGB_img.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640,480, Qt.KeepAspectRatio)
                changePixmap.emit(p)
                # print('got through here')
                # del RGB_img




    def process_image(self, image=None):

        # run below in a another thread
        while self.is_running:
            if not self.imagequeue.empty():
                image = self.imagequeue.get()
                # print("Image received")
                # print("Image shape: ", image.shape)
                # print("Image type: ", type(image))
                # print("Image size: ", image.size)
                # print("Image dtype: ", image.dtype)
                # print("Image len: ", len(image))


                #for peak detection
                y_vals = []

                #array for timestamps when every image is received
                image_times = []

                curr_date = datetime.datetime.now()
                dt_string = curr_date.strftime("%d-%m-%Y-%H-%M-%S")

                # print("Remaining buffer size: ",img_buffer_size)

                #get timestamp when getting the image frame -- for cpr rate detection
                curr_time = round(time.time()*1000)
                image_times.append(curr_time)



                #process image with mediapipe hand detection
                hand_detection_results = self.mp_hands.process(image)

                # hand-detection annotations
                if hand_detection_results and hand_detection_results.multi_hand_landmarks:
                    self.changeVisInfo.emit(str("Hand Detected! Wrist Position Identified."))
                    for hand_landmarks in hand_detection_results.multi_hand_landmarks:

                        #append y_val to array for peak detection
                        y_vals.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)

                        #for drawing hand annotations on image
                        self.mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())
                else:
                    self.changeVisInfo.emit(str("Detecting Hands..."))
                    y_vals.append(0)#(math.nan)


                # cv2_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # cv2_img = cv2.resize(cv2_img, (640, 480))
                cv2_img = image
                h, w, ch = cv2_img.shape
                # print("Image Size",h,w)
                bytesPerLine = ch * w
                convertToQtFormat = QImage(cv2_img.data, 640, 480, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(w,h, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)


                if(len(image_times) == 10000):
                    #Clear the arrays to get 100 more image frames for cpr rate calculation
                    y_vals.clear()
                    image_times.clear()


    def read_exactly(self, pipe, n):
        """Read exactly n bytes from pipe."""
        data = b''
        while len(data) < n:
            chunk = pipe.read(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data


    def run(self):
        count = 0
        is_connected = False
        try:
            print(f"[Reader] Opening {self.PIPE_VIDEO} for reading...")
            self.video_pipe_holder = open(self.PIPE_VIDEO, 'rb', buffering=0)
            print(f"[Reader] {self.PIPE_VIDEO} opened!")
            is_connected = True
            self.display_thread.start()
        except Exception as e:
            print('video pipe exception')

        while self.is_running and is_connected:
            try:
                # print('here')
                # Read video frame
                # Format: 4 bytes length + data
                length_bytes = self.read_exactly(self.video_pipe_holder, 4)
                if length_bytes is None:
                    print("[Reader] Video pipe closed")
                    break

                video_length = int.from_bytes(length_bytes, 'big')
                video_data = self.read_exactly(self.video_pipe_holder, video_length)
                # print(type(video_data))
                if video_data is None:
                    print("[Reader] Failed to read video data")
                    break

                # Decode video frame
                try:
                    video_frame = np.frombuffer(video_data, dtype=np.uint8).reshape(
                        (270, 480, 3)
                    )
                    # self.frame_ready.emit(video_frame.copy())
                    # print('frame count', count)
                    display_queue.put_nowait(video_frame.copy())
                    count += 1
                    # image_queue.put_nowait(video_frame.copy())
                except Exception as e:
                    print(f"[Reader] Failed to decode video frame: {e}")
            except Exception as e:
                print(e)
                print('here exception e')
                print('reader error')


            # try:
            #     # print("[VIDEO STREAMING]: Attempting to connect to the socketio ...")
            #     # self.sio.connect(socketio_ipaddr)  # Connect to the Flask-SocketIO server
            #     # print("Connected to the server!")
            #     # self.sio.emit('message', 'Hello from Video QThread!')  # Send a message to the server
            #
            #
            #
            #     #start object detector engine process
            #     # mp.set_start_method('spawn', force=True)
            #
            #
            #     # image_processor = ImageProcessor(image_queue, display_queue)
            #     # image_processor.start()
            #
            #     # self.sio.wait()
            #
            #     # break  # Exit the loop if connected successfully
            # except Exception as e:
            #     print("Connection failed, retrying...", e)
            #     # time.sleep(5)  # Wait for 5 seconds before retrying