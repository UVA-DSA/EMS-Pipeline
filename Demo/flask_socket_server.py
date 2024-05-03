from flask import Flask, request
from flask_socketio import SocketIO


import base64
import io
from PIL import Image
import cv2
from imageio import imread
from PIL import Image
import numpy as np
import multiprocessing

imagequeue = multiprocessing.Queue()

app = Flask(__name__)
socketio = SocketIO(app)

video_display_process = None
image_seq = 0


@socketio.on('connect')
def handle_connect():
    # Accessing the client's address from the request context
    client_address = request.remote_addr
    print(f"Client connected: {client_address}")

    global video_display_process  # Declare the variable as global

    if video_display_process is None or not video_display_process.is_alive():
        video_display_process = multiprocessing.Process(
            target=display_image, args=(imagequeue,))
        video_display_process.start()


@socketio.on('message')
def handleImage(msg):
    print((msg))


@socketio.on('feedback')
def handleFeedback(msg):
    #below is for debugging purposes
    print(msg)

@socketio.on('audio')
def handle_audio(audio_data):
    encoded_audio = base64.b64encode(audio_data).decode('utf-8')
    socketio.emit('server_audio', encoded_audio)


@socketio.on('bytes')
def handle_byte_array(byte_array):
    print('Received video from smartglass')
    byte_array_string = base64.b64encode(byte_array).decode('utf-8')

    socketio.emit('server_video', byte_array_string)

    # global image_seq

    # try:
    #     # byte_array_string consists a string of base64 encoded bmp image
    #     byte_array = base64.b64decode(byte_array_string)

    #     # Step 2: Create an image object from the binary data
    #     image = Image.open(io.BytesIO(byte_array))

    #             # Convert the PIL image object to an OpenCV image (numpy array)
    #     cv2_img = np.array(image)

    #     cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)

    #     cv2_img = cv2.rotate(cv2_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    #     cv2.imwrite(f"./data_collection_folder/img_{image_seq}.jpeg",cv2_img)

    #     image_seq += 1

    #     print('Image saved successfully.')
    # except Exception as e:
    #     print("EXCEPTION!: ",e)


@socketio.on('video')
def handle_base64_img(byte_array_string):
    # print('Received video from smartglass')
    socketio.emit('server_video', byte_array_string)
    # print('Received bytes!')

    # # Convert the base64 encoded byte array string to bytes

    try:
        # byte_array_string consists a string of base64 encoded bmp image
        byte_array = base64.b64decode(byte_array_string)

        # Step 2: Create an image object from the binary data
        image = Image.open(io.BytesIO(byte_array))

        # Convert the PIL image object to an OpenCV image (numpy array)
        cv2_img = np.array(image)

        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)

        cv2_img = cv2.rotate(cv2_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cv2.imwrite(cv2_img, f"./data_collection_folder/img_{image_seq}.jpeg")

        image_seq += 1

        print('Image saved successfully.')
    except Exception as e:
        print("EXCEPTION!: ", e)


def display_image(imagequeue):
    while True:
        if not imagequeue.empty():
            cv2.imshow('image', imagequeue.get())
            cv2.waitKey(1)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
