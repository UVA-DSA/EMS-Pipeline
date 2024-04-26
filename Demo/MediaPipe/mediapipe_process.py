import multiprocessing
import mediapipe as mp
import datetime
import time

# Media Pipe vars
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

            self.mp_drawing = mp.solutions.drawing_utils
            self.input_queue = input_queue 
            self.output_queue = output_queue

       def process_image(self, image=None):
            # run below in a another thread
            while True:
                if not self.input_queue.empty():
                    image = self.input_queue.get()
                    
                    #here for debugging purposes, remove when necessary
                    print("Image recieved")
                    continue

                    curr_date = datetime.datetime.now()
                    dt_string = curr_date.strftime("%d-%m-%Y-%H-%M-%S")

                    #process image with mediapipe hand detection
                    hand_detection_results = self.mp_hands.process(image)

                    # hand-detection annotations
                    if hand_detection_results and hand_detection_results.multi_hand_landmarks:
                        for hand_landmarks in hand_detection_results.multi_hand_landmarks:               
                            #for drawing hand annotations on image
                            self.mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())
                        
                        #send image to output queue
                        self.output_queue.put(image)