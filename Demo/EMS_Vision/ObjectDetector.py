from multiprocessing import Process
from Feedback import FeedbackClient
from classes import DetectionObj
from .DETR_Engine import DETREngine
import numpy as np
import time

from socketio import Client


from pipeline_config import detr_version, socketio_ipaddr

class ObjectDetector(Process):
    def __init__(self, input_queue, output_queue):
        super(ObjectDetector, self).__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.detr_engine = DETREngine(detr_version)
        self.feedback_client = FeedbackClient()
        # self.sio = Client()
        print("ObjectDetector: Initialized")


    def run(self):
        # self.sio.connect(socketio_ipaddr)  # Connect to the Flask-SocketIO server
        # print("Connected to the server!")
        # self.sio.emit('message', 'Hello from Object Detector Process!')  # Send a message to the server
        self.feedback_client.start()
        while True:
            frame = self.input_queue.get()
            print("ObjectDetector: Got frame")
            # # Do some object detection
            result_image = self.detr_engine.run_workflow(frame)
            
                        
            dummy_obj = DetectionObj({'center_point':'(500, 700)', 'width':'400', 'height':'300'}, 'keyboard', '0.255678475236')
            dummy_obj_dict = dummy_obj.__dict__
            self.feedback_client.send_message(dummy_obj_dict)

            # time.sleep(5)
            
            # dummy_obj2 = DetectionObj({'center_point':'(800, 500)', 'width':'300', 'height':'200'}, 'computer', '0.455678475236')
            # dummy_obj_dict2 = dummy_obj2.__dict__
            # self.feedback_client.send_message(dummy_obj_dict2)

            # time.sleep(5)
            # self.feedback_client.send_message(dummy_obj_dict)
            # self.sio.emit('feedback', dummy_obj_dict)
            self.output_queue.put(result_image)
            print("ObjectDetector: Put frame")
