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
    dt_string = curr_date.strftime("%d-%m-%Y-%H-%M-%S")

    global recording_dir 
    global image_index
    
    image_index = 0
    recording_dir = f"../DataCollected/{dt_string}/{recording_info.subject}/{recording_info.intervention}/{recording_info.trial}"
    
    if not os.path.exists(recording_dir):
        os.makedirs(recording_dir)
    # print("Recording Info: ", recording_info)
    
def save_as_wav(audio_data, save_path):
    # Set the parameters for the WAV file
    channels = 1  # Mono audio
    sample_width = 2  # 16-bit audio
    sample_rate = 16000  # Example sample rate (modify according to your data)

    with wave.open(f'{save_path}received_audio.wav', 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)

    # print('Audio data saved as received_audio.wav')
    
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start', methods = ["POST"])
def start():
    recording_info = None
    if request.method == "POST":
        
        subject = request.form.get("subject")
        intervention = request.form.get("intervention")
        
        recording_info = RecordingInfo(subject=subject, intervention=intervention)
        
        init_recording(recording_info=recording_info)
        
        
        # print("Recording Started!")
        
    # emit('command', "start", broadcast=True)
    sendCommand("start")
    
    global recording_dir
    # asyncio.run(main())
    # gopro_process =  multiprocessing.Process(target=execute_main, args=(commandqueue,recording_dir))
    # gopro_process.start()
    
    commandqueue.put("start")
    
    # 'Sent Start Command to CognitiveEMS!'
    return redirect(url_for('index'))


@app.route('/stop')
def stop():
    # emit('command', "stop", broadcast=True)
    sendCommand("stop")
    commandqueue.put("stop")
    return redirect(url_for('index'))


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
        byte_array = base64.b64decode(byte_array_string)
        # Specify the file path where you want to save the image
        image_path = f'{video_recording_dir}frame_{image_index}.jpeg'  # You can change the file format as needed (e.g., .jpg, .png)
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


@socketio.on('audio')
def handle_audio(byte_array):
    audio_recording_dir = f"{recording_dir}/audio/"
    if not os.path.exists(audio_recording_dir):
        os.makedirs(audio_recording_dir)
        # print("Audio recording directory created!")
    
    # print('Received audio bytes! ')
        # Convert received data (assuming it's a list of bytes) to a NumPy array
    global audio_array
    
    chunk_audio = np.frombuffer(bytes(byte_array), dtype=np.uint8)
    
    audio_array = np.append(audio_array, chunk_audio)
    
    save_as_wav(audio_data=audio_array, save_path=audio_recording_dir)
    
    
@socketio.on('connect')
def handle_connect():
    
    global video_display_process  # Declare the variable as global
    
    if video_display_process is None or not video_display_process.is_alive():
        video_display_process = multiprocessing.Process(target=display, args=(imagequeue,commandqueue))
        video_display_process.start()
        
    # print('A user connected')
    

@socketio.on('disconnect')
def handle_disconnect():
    # print('A user disconnected')
    global video_display_process  # Declare the variable as global
    
    global image_index 
    image_index = 0
    video_display_process.terminate()
    

def display(ImageQueue, commandqueue):
    # print("display started:")
    
    while True:
        if not commandqueue.empty():
            command = commandqueue.get()
            if command == "stop":
                break
            
        if not ImageQueue.empty():
            # print("display received:")
            received = ImageQueue.get()
            if (received is None): break
            cv2.imshow("Window", received)
            cv2.waitKey(1)
            
        
        
    cv2.destroyAllWindows()    
        

if __name__ == '__main__':

    socketio.run(app, host='0.0.0.0', port=9235)
    
