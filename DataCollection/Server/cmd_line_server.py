from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit
import base64
import io
from PIL import Image
import cv2
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from threading import Thread
from gopro import execute_main, main
import asyncio

import multiprocessing

import queue
import time
from datetime import datetime
import os
import wave


imagequeue = multiprocessing.Queue()
commandqueue = multiprocessing.Queue()
video_display_process = None
recording_dir = None
image_index = 0
# Initialize a global audio array to store received audio data
audio_array = np.array([], dtype=np.uint8)

app = Flask(__name__)
socketio = SocketIO(app)



@socketio.on('message')
def handle_message(message):
    # print('Received message:', message)
    pass
    # emit('message', message, broadcast=True)

@socketio.on('video')
def handle_byte_array(byte_array_string):
    # print('Received bytes!')
    
    # # Convert the base64 encoded byte array string to bytes

    video_recording_dir = f"{recording_dir}/video/"
    global image_index
    
    if not os.path.exists(video_recording_dir):
        os.makedirs(video_recording_dir)
        # print("Video recording directory created!")
    
    try:
        # print(video_recording_dir)
        img_data = byte_array_string.split(',')

        byte_array = base64.b64decode(img_data[0])
        # Specify the file path where you want to save the image
        image_path = f'{video_recording_dir}frame_{image_index}_seq-{img_data[1]}_ts-{img_data[2]}.jpeg'  # You can change the file format as needed (e.g., .jpg, .png)
        image_index += 1
        # reconstruct image as an numpy array
        img = imread(io.BytesIO(byte_array))

        # finally convert RGB image to BGR for opencv
        # and save result
        # print('Image save path: ',image_path)
        cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, cv2_img)

        imagequeue.put(cv2_img)
        commandqueue.put("display")

        # print('Image saved successfully.')
    except Exception as e:
        print("EXCEPTION!: ",e)


@socketio.on('audio')
def handle_audio(byte_array):
    audio_recording_dir = f"{recording_dir}/audio/"
    if not os.path.exists(audio_recording_dir):
        os.makedirs(audio_recording_dir, exist_ok=True)
        # print("Audio recording directory created!")
    
    # print('Received audio bytes! ')
        # Convert received data (assuming it's a list of bytes) to a NumPy array
    global audio_array
    
    chunk_audio = np.frombuffer(bytes(byte_array), dtype=np.uint8)
    
    audio_array = np.append(audio_array, chunk_audio)
    
    save_as_wav(audio_data=audio_array, save_path=audio_recording_dir)
    
    
@socketio.on('connect')
def handle_connect():
    print('A user connected')
    

@socketio.on('disconnect')
def handle_disconnect():
    # print('A user disconnected')
    global video_display_process  # Declare the variable as global
    
    global image_index 
    image_index = 0

        

def send_messages():
    print("Type 'quit' to exit.")
    while True:
        message = input("Enter message to send: ")
        if message == 'quit':
            break
        socketio.emit('command', message)  # Send the message to all connected clients


if __name__ == '__main__':

    # Start send_messages in a separate thread
    thread = Thread(target=send_messages)
    thread.start()

    socketio.run(app, host='0.0.0.0', port=5000)
    
