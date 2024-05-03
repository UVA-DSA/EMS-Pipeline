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
        self.feedback_client = FeedbackClient()
        print("ObjectDetector: Initialized")


    def run(self):
        self.feedback_client.start() # start the feedback thread
        while True:
            frame = self.input_queue.get()
            # print("ObjectDetector: Got frame")
            # Run object detection engine.
            result_image, result_bbox_objs = self.detr_engine.run_workflow(frame)
            self.feedback_client.sendMessage(result_bbox_objs)
            self.output_queue.put(result_image)