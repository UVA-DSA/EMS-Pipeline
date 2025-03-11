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

import logging
logging.basicConfig(filename='./log.txt',level=logging.DEBUG)


import multiprocessing
import queue
import time
from datetime import datetime
import os
import sys
import threading
# from udp_audio_server import receive_and_buffer
from tcp_smartwatch_client import receive_smartwatch_data


recording_dir = None

# queues
imagequeue = multiprocessing.Queue()
commandqueue = multiprocessing.Queue()


# Audio related imports and vars
audiocommandqueue = queue.Queue()
audio_thread = None
audio_array = np.array([], dtype=np.uint8)

# thread management
thread_stop = threading.Event()

# Image related vars
image_index = 0


# Smartwatch related vars
smartwatch_ip = '192.168.0.101'
smartwatch_port = 7889
smartwatch_id = 'right'
sw_thread = None


# Kinect related imports and vars
import open3d as o3d
sys.path.append('D:/repos/EMS-Pipeline/DataCollection/Kinect')
from kinect_recorder import RecorderWithCallback
from datetime import datetime

config = o3d.io.AzureKinectSensorConfig()
filename = '{date:%Y-%m-%d-%H-%M-%S}.mkv'.format(date=datetime.now())
device = 0
align_depth_to_color = True
kinect_recorder = None
kinect_thread = None


# Arduino related imports and vars
sys.path.append('D:/repos/EMS-Pipeline/DataCollection/Arduino/VL6180_ToF')
from arduino_serial_reciever import ArduinoVL6180Recorder
port = 'COM5'
arduino_thread = None
process_stop = multiprocessing.Event()


# Main app
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

    global thread_stop
    
    global audiocommandqueue

    global kinect_thread
    global kinect_recorder

    global arduino_thread
    global arduino_recorder 


    global process_stop 

    process_stop.clear()
    
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
    recording_root = "D:/repos/EMS-Pipeline/DataCollection/DataCollected"
    recording_dir = f"{recording_root}/{dt_string}/{recording_info.subject}/{recording_info.intervention}/{recording_info.trial}"
    print("DCS Server Recording Directory: ", recording_dir)

    # Create the necessary recorders
    kinect_recorder = RecorderWithCallback(config, device, filename, align_depth_to_color)
    arduino_recorder = ArduinoVL6180Recorder(port=port, recording_dir=recording_dir, thread_stop=process_stop)  # Replace with your actual port
    

    # Start the recording threads

    if (arduino_thread == None):
        # arduino should be a multiprocessing process 
        arduino_thread = multiprocessing.Process(target=arduino_recorder.run)
        # # arduino_thread = threading.Thread(target=arduino_recorder.run)
        # arduino_recorder.start_recording()
        # arduino_thread.start()


    # if (audio_thread == None):
    #     audio_thread = threading.Thread(target=receive_and_buffer, args=(recording_dir, audiocommandqueue))
    #     audio_thread.start()
    #     audiocommandqueue.put("start")

    if (kinect_thread == None):
        kinect_thread = threading.Thread(target=kinect_recorder.run)
        kinect_thread.start()
        kinect_recorder.start_recording(recording_dir)

    if (sw_thread == None):
        sw_thread = threading.Thread(target=receive_smartwatch_data, args=(smartwatch_ip, smartwatch_port, None, recording_dir,smartwatch_id, thread_stop))
        sw_thread.start()

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

    global process_stop
    global thread_stop

    process_stop.clear()
    thread_stop.clear()
    if request.method == "POST":
        
        subject = request.form.get("subject")
        intervention = request.form.get("intervention")
        
        recording_info = RecordingInfo(subject=subject, intervention=intervention)
        
        init_recording(recording_info=recording_info)

    sendCommand("start")
    
    global recording_dir

    # GoPro Code
    gopro_process =  multiprocessing.Process(target=execute_main, args=(commandqueue,recording_dir))
    # gopro_process.start()
    
    commandqueue.put("start")

    # 'Sent Start Command to CognitiveEMS!'
    return redirect(url_for('recording_in_progress'))


@app.route('/stop')
def stop():
    print("Stop button clicked!")
    
    global commandqueue
    global audiocommandqueue
    
    global audio_thread
    global sw_thread

    global kinect_thread
    global kinect_recorder

    global arduino_thread
    global arduino_recorder

    global thread_stop
    global process_stop
    
    process_stop.set()
    thread_stop.set()

    commandqueue.put("stop")
    
    
    if audio_thread != None:
        print("Stopping audio thread")
        audiocommandqueue.put("stop")
        audio_thread = None
        
    
    if sw_thread != None:
        print("Stopping smartwatch thread")
        sw_thread = None
        

    if kinect_thread != None:
        print("Stopping kinect thread")
        kinect_recorder.stop_recording()
        kinect_recorder = None
        kinect_thread = None

    if arduino_thread != None:
        print("Stopping arduino thread")
        # arduino_recorder.stop_recording()
        arduino_thread = None
    
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
    socketio.run(app, host='0.0.0.0', port=8389, debug=True)
    
