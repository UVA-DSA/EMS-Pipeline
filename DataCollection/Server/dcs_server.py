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

import threading
from collections import deque
from udp_audio_server import receive_and_buffer
from tcp_smartwatch_client import receive_data

imagequeue = multiprocessing.Queue()
commandqueue = multiprocessing.Queue()
audiocommandqueue = queue.Queue()
smartwatchcommandqueue = queue.Queue()

audio_thread = None
sw_thread = None

recording_dir = None
image_index = 0
# Initialize a global audio array to store received audio data
audio_array = np.array([], dtype=np.uint8)




# Replace these values with your smartwatch's IP, port
smartwatch_ip = '172.27.150.154'
smartwatch_port = 7889





app = Flask(__name__)
socketio = SocketIO(app)

class RecordingInfo:
    def __init__(self, subject, intervention):
        self.subject = subject
        self.intervention = intervention
        self.trial = 0
        
def sendCommand(command):
    # print("Command: ")
    socketio.emit('command', command)
    
def init_recording(recording_info:RecordingInfo):
    
    curr_date = datetime.now()
    dt_string = curr_date.strftime("%d-%m-%Y")

    global recording_dir 
    global image_index
    global audio_thread
    global sw_thread
    
    global smartwatchcommandqueue
    global audiocommandqueue
    
    image_index = 0
    recording_dir = f"../DataCollected/{dt_string}/{recording_info.subject}/{recording_info.intervention}/{recording_info.trial}"
    
    trial_num = recording_info.trial

    # Loop to find the next available trial number
    while True:
        recording_dir = f"../DataCollected/{dt_string}/{recording_info.subject}/{recording_info.intervention}/{trial_num}"
        if not os.path.exists(recording_dir):
            os.makedirs(recording_dir)
            recording_info.trial = trial_num  # Update the recording info with the new trial number
            break  # Exit the loop once the folder is created
        trial_num += 1  # Increment the trial number if the folder already exists

    recording_info.trial = trial_num
    recording_dir = f"../DataCollected/{dt_string}/{recording_info.subject}/{recording_info.intervention}/{recording_info.trial}"

    if (audio_thread == None):
        audio_thread = threading.Thread(target=receive_and_buffer, args=(recording_dir, audiocommandqueue))
        audio_thread.start()
        audiocommandqueue.put("start")


    if (sw_thread == None):
        sw_thread = threading.Thread(target=receive_data, args=(smartwatch_ip, smartwatch_port, recording_dir, smartwatchcommandqueue))
        sw_thread.start()
        smartwatchcommandqueue.put("start")

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recording_in_progress')
def recording_in_progress():
    # You can render a template or return a simple message
    return render_template('recording_in_progress.html')  # Assuming you have this template
    # Or return a simple message
    # return "Recording is in progress..."



@app.route('/start', methods = ["POST"])
def start():
    print("Start button clicked!")
    recording_info = None
    if request.method == "POST":
        
        subject = request.form.get("subject")
        intervention = request.form.get("intervention")
        
        recording_info = RecordingInfo(subject=subject, intervention=intervention)
        
        init_recording(recording_info=recording_info)

    sendCommand("start")
    
    global recording_dir

    # GoPro Code
    gopro_process =  multiprocessing.Process(target=execute_main, args=(commandqueue,recording_dir))
    gopro_process.start()
    
    commandqueue.put("start")

    # 'Sent Start Command to CognitiveEMS!'
    return redirect(url_for('recording_in_progress'))


@app.route('/stop')
def stop():
    print("Stop button clicked!")
    
    global commandqueue
    global audiocommandqueue
    global smartwatchcommandqueue
    
    global audio_thread
    global sw_thread
    
    commandqueue.put("stop")
    
    
    if audio_thread != None:
        print("Stopping audio thread")
        audiocommandqueue.put("stop")
        audio_thread = None
        
    
    if sw_thread != None:
        print("Stopping smartwatch thread")
        smartwatchcommandqueue.put("stop")
        sw_thread = None
        
    
    sendCommand("stop")
    
    return redirect(url_for('index'))


@socketio.on('message')
def handle_message(message):
    # print('Received message:', message)
    pass

@socketio.on('video')
def handle_byte_array(byte_array_string):
    
    # # Convert the base64 encoded byte array string to bytes
    video_recording_dir = f"{recording_dir}/video/"
    global image_index
    
    if not os.path.exists(video_recording_dir):
        os.makedirs(video_recording_dir)
        # print("Video recording directory created!")
    
    try:
        # print(video_recording_dir)
        img_data = byte_array_string.split(',')

        #get epoch time in ns
        ts = int(time.time_ns())
        byte_array = base64.b64decode(img_data[0])
        # Specify the file path where you want to save the image
        image_path = f'{video_recording_dir}frame_{image_index}_seq-{img_data[1]}_source_ts-{img_data[2]}_pc_ts{ts}.jpeg'  # You can change the file format as needed (e.g., .jpg, .png)
        image_index += 1
        # reconstruct image as an numpy array
        img = imread(io.BytesIO(byte_array))

        # finally convert RGB image to BGR for opencv
        # and save result
        # print('Image save path: ',image_path)
        cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, cv2_img)

        imagequeue.put(cv2_img)

        # print('Image saved successfully.')
    except Exception as e:
        print("EXCEPTION!: ",e)


# not used
@socketio.on('audio')
def handle_audio(byte_array):
    print("Received audio bytes!", len(byte_array))
    
    
@socketio.on('connect')
def handle_connect():
    print('A user connected')
    

@socketio.on('disconnect')
def handle_disconnect():
    # print('A user disconnected')
    global image_index 
    image_index = 0
    
 

if __name__ == '__main__':

    socketio.run(app, host='0.0.0.0', port=8383)
    
