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
        self.action = " " #action for action log\
        # self.sio = Client()
        print("ObjectDetector: Initialized")


    def actionRecognition(self, obj):
        self.detected_objects.append(obj)
        print(self.detected_objects)
        proposedAction = self.action
        #MAY WANT TO CHANGE THIS TO A SWITCH CASE LATER
        #HAD TO USE 'person' instead of hand? keeps calling a hand a person
        if any (o == 'person' for o in self.detected_objects):
            if any (o == 'bvm' for o in self.detected_objects):
                proposedAction = 'CPR started' 
            elif any (o == 'defibrillator' for o in self.detected_objects):
                proposedAction = 'Defibrillation started'
            #TAKE OUT LATER, USING FOR ACTION LOG TESTING
            elif any (o == 'keyboard' for o in self.detected_objects):
                proposedAction = 'typing' 
                self.detected_objects = [];
            elif any (o == 'cell phone' for o in self.detected_objects):
                proposedAction = 'calling' 
                self.detected_objects = [];
        if proposedAction != self.action:
            self.action = proposedAction
            self.feedback_client.send_message(self.action+ " " + str(datetime.now()), 'action')


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
            
        
            (image_array, objectDetected) = result_image
            if str(objectDetected) != '[]':
                for i in range(len(objectDetected)):
                    self.feedback_client.send_message(objectDetected[i], 'objectFeedback')
                    if (objectDetected[i].get('obj_name') != 'null'):
                        self.actionRecognition(objectDetected[i].get('obj_name'))

                



    #   

            # # time.sleep(5)
            
            # dummy_obj2 = DetectionObj({'center_point':'(800, 500)', 'width':'300', 'height':'200'}, 'computer', '0.455678475236')
            # dummy_obj_dict2 = dummy_obj2.__dict__
            # self.feedback_client.send_message(dummy_obj_dict2, 'feedback')

            # self.actionRecognition(dummy_obj)

            # time.sleep(3)

            # dummy_obj2 = DetectionObj({'center_point':'(900, 400)', 'width':'300', 'height':'200'}, 'phone', '0.455678475236')
            # dummy_obj_dict2 = dummy_obj2.__dict__
            # self.feedback_client.send_message(dummy_obj_dict2, 'feedback')

            # self.actionRecognition(dummy_obj2)

            # time.sleep(3)
    

            # dummy_obj2 = DetectionObj({'center_point':'(800, 500)', 'width':'300', 'height':'200'}, 'monitor', '0.455678475236')
            # dummy_obj_dict2 = dummy_obj2.__dict__
            # self.feedback_client.send_message(dummy_obj_dict2, 'feedback')

            # time.sleep(3)

            # self.actionRecognition(dummy_obj2)



            # self.feedback_client.send_message(dummy_obj_dict)
            # self.sio.emit('feedback', dummy_obj_dict)
            del frame
            self.output_queue.put(result_image)
            print("ObjectDetector: Put frame")

if __name__ == '__main__':
    print("ObjectDetector: Testing")