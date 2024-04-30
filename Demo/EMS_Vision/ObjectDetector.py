from multiprocessing import Process

from .DETR_Engine import DETREngine
import time

from pipeline_config import detr_version

class ObjectDetector(Process):
    def __init__(self, input_queue, output_queue):
        super(ObjectDetector, self).__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.detr_engine = DETREngine(detr_version)

        print("ObjectDetector: Initialized")


    def run(self):
        while True:
            frame = self.input_queue.get()
            # print("ObjectDetector: Got frame")
            # Do some object detection
            result_image = self.detr_engine.run_workflow(frame)

            self.output_queue.put(result_image)
            # print("ObjectDetector: Put frame")