from multiprocessing import Process
from Feedback import FeedbackClient
from classes import DetectionObj
from .DETR_Engine import DETREngine
import numpy as np
import time
from datetime import datetime

from torch import multiprocessing

from socketio import Client


from pipeline_config import detr_version, socketio_ipaddr

class ObjectDetector(multiprocessing.Process):
    def __init__(self, input_queue, output_queue):
        super(ObjectDetector, self).__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.detr_engine = None
        self.feedback_client = FeedbackClient()
        self.detected_objects = [] #list of identified objects 
        self.action = "Not enough information to determine action." #action for action log
        # self.sio = Client()
        print("ObjectDetector: Initialized")


    def actionRecognition(self, obj):
        self.detected_objects.append(obj)
        if any (o == 'hands' for o in self.detected_objects):
            if any (o == 'bvm' for o in self.detected_objects):
                self.action = 'CPR started' + " " + str(datetime.now())
            elif any (o == 'defibrillator' for o in self.detected_objects):
                self.action = 'Defibrillation started' + " " + str(datetime.now())
        self.feedback_client.send_message(self.action, 'action')


    def run(self):
        # self.sio.connect(socketio_ipaddr)  # Connect to the Flask-SocketIO server
        # print("Connected to the server!")
        # self.sio.emit('message', 'Hello from Object Detector Process!')  # Send a message to the server
        self.detr_engine = DETREngine(detr_version)
        self.feedback_client.start()
        while True:
            print("ObjectDetector: Waiting for frame")
            frame = self.input_queue.get()
            print("ObjectDetector: Got frame")
            # # Do some object detection
            result_image = self.detr_engine.run_workflow(frame)
            
            

            dummy_obj = DetectionObj({'center_point':'(500, 700)', 'width':'400', 'height':'300'}, 'keyboard', '0.255678475236')
            dummy_obj_dict = dummy_obj.__dict__
            self.feedback_client.send_message(dummy_obj_dict, 'feedback')

            self.actionRecognition(dummy_obj)


            # time.sleep(5)
            
            dummy_obj2 = DetectionObj({'center_point':'(800, 500)', 'width':'300', 'height':'200'}, 'computer', '0.455678475236')
            dummy_obj_dict2 = dummy_obj2.__dict__
            self.feedback_client.send_message(dummy_obj_dict2, 'feedback')

            self.actionRecognition(dummy_obj)

            time.sleep(3)

            dummy_obj2 = DetectionObj({'center_point':'(900, 400)', 'width':'300', 'height':'200'}, 'phone', '0.455678475236')
            dummy_obj_dict2 = dummy_obj2.__dict__
            self.feedback_client.send_message(dummy_obj_dict2, 'feedback')

            self.actionRecognition(dummy_obj2)

            time.sleep(3)
    

            dummy_obj2 = DetectionObj({'center_point':'(800, 500)', 'width':'300', 'height':'200'}, 'monitor', '0.455678475236')
            dummy_obj_dict2 = dummy_obj2.__dict__
            self.feedback_client.send_message(dummy_obj_dict2)

            time.sleep(3)

            self.actionRecognition(dummy_obj2)



            # self.feedback_client.send_message(dummy_obj_dict)
            # self.sio.emit('feedback', dummy_obj_dict)
            del frame
            self.output_queue.put(result_image)
            print("ObjectDetector: Put frame")

if __name__ == '__main__':
    print("ObjectDetector: Testing")