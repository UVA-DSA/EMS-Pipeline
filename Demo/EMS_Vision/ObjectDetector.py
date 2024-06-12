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
        proposedAction = self.action #self.action is the current action being displayed
        #MAY WANT TO CHANGE THIS TO A SWITCH CASE LATER, awk in python though
        #HAD TO USE 'person' instead of hand? keeps calling a hand a person
        if any (o == 'person' for o in self.detected_objects):
            if any (o == 'bvm' for o in self.detected_objects):
                proposedAction = 'CPR started' 
            elif any (o == 'defibrillator' for o in self.detected_objects):
                proposedAction = 'Defibrillation started'
           
           #TAKE OUT LATER, USING FOR ACTION LOG TESTING
            elif any (o == 'keyboard' for o in self.detected_objects):
                proposedAction = 'typing' 
            elif any (o == 'cell phone' for o in self.detected_objects):
                proposedAction = 'calling' 
        
        if proposedAction != self.action: #checks that a new action has began 
            self.action = proposedAction
            self.detected_objects = []; #reset detected objects, such that we don't trigger the same action after the object is no longer onscreen
            self.feedback_client.send_message(self.action+ " " + str(datetime.now().strftime("%H:%M:%S")), 'action')


    def run(self):
        self.detr_engine = DETREngine(detr_version)
        self.feedback_client.start()
        while True:
            print("ObjectDetector: Waiting for frame")
            frame = self.input_queue.get()
            print("ObjectDetector: Got frame")
            # # Do some object detection
            result_image = self.detr_engine.run_workflow(frame)
            
        
            (image_array, objectDetected) = result_image
            if str(objectDetected) != '[]': #if not null, sometimes it identifies a box with no object detection?
                for i in range(len(objectDetected)): # loop through output, as there may be more than one object detected
                    self.feedback_client.send_message(objectDetected[i], 'objectFeedback') #send detected object on objectfeedback channel
                    self.actionRecognition(objectDetected[i].get('obj_name')) 

                

            del frame
            self.output_queue.put(result_image)
            print("ObjectDetector: Put frame")

if __name__ == '__main__':
    print("ObjectDetector: Testing")