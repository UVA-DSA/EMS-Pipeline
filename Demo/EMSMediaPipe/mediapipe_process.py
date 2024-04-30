import multiprocessing
import mediapipe as mp
import datetime
from base64 import b64decode
import time
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

# Media Pipe variables
global mp_drawing, mp_drawing_styles, mp_hands, mp_face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh


class MediaPipeProcess(multiprocessing.Process):
    def __init__(self, input_queue, output_queue):
        super(MediaPipeProcess, self).__init__()
        self.mp_hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        if not self.input_queue.empty():
            data = self.image_queue.get()
            image_bytes = b64decode(data)
            image = Image.open(BytesIO(image_bytes))
            RGB_img = np.array(image)
            """ 
            For interacting with video stream, leaving it out while testing.
            RGB_img = cv2.cvtColor(RGB_img, cv2.COLOR_RGB2BGR)
            RGB_img = cv2.rotate(RGB_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            RGB_img = cv2.resize(RGB_img, (640, 480))
            """

    def process_image(self, QueueImage=None):
        # run below in a another thread
        global mp_hands
        hand_detection_results = None

        # color scheme conversions for compatibility
        QueueImage = cv2.cvtcolor(QueueImage.image, cv2.COLOR_BGR2RGB)

        # hand detection
        with mp_hands.Hands(
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            hand_detection_results = hands.process(QueueImage.image)

        QueueImage.image.flags.writeable = True
        QueueImage.image = cv2.cvtColor(QueueImage.image, cv2.COLOR_RGB2BGR)

        # hand detection annotation
        if hand_detection_results and hand_detection_results.multi_hand_landmarks:
            for hand_landmarks in hand_detection_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(QueueImage.image, 
                                          hand_landmarks, 
                                          mp_hands.HAND_CONNECTIONS, 
                                          mp_drawing_styles.get_default_hand_landmarks_style(), 
                                          mp_drawing_styles.get_default_hand_connections_style())
        return QueueImage
