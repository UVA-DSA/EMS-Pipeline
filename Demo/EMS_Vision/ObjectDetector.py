from multiprocessing import Process
from Feedback import FeedbackClient
from classes import DetectionObj
from .DETR_Engine import DETREngine
import numpy as np
import time

from pipeline_config import detr_version

class ObjectDetector(Process):
    def __init__(self, input_queue, output_queue):
        super(ObjectDetector, self).__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.detr_engine = DETREngine(detr_version)
        self.feedback_client = FeedbackClient.instance()
        print("ObjectDetector: Initialized")


    def run(self):
        self.feedback_client.start() # start the feedback thread
        while True:
            frame = self.input_queue.get()
            # print("ObjectDetector: Got frame")
            # Do some object detection
            result_image = self.detr_engine.run_workflow(frame)
            
            dummy_obj = DetectionObj('hi', 'hi')
            dummy_obj_dict = dummy_obj.__dict__
            self.feedback_client.sendMessage(dummy_obj_dict)
            self.output_queue.put(result_image)
            # print("ObjectDetector: Put frame")
