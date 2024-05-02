# move this file into EMSDev once done testing

import multiprocessing
import numpy as np
from EMSMediaPipe import MediaPipeProcess
from PIL import Image
from classes import QueueImage
from datetime import datetime
import cv2

test_image = Image.open("/home/cogems_nist/Downloads/mediapipe_landmark.jpg")
cv2image = np.asarray(test_image)

# Create two queues for interprocess communication
image_queue = multiprocessing.Queue(maxsize=1)
results_queue = multiprocessing.Queue()

# Create a MediaPipeWorker object
MediaPipeWorker = MediaPipeProcess(
    input_queue=image_queue, output_queue=results_queue)

# media_pipe_process = multiprocessing.Process(target=MediaPipeWorker.process_image_loop)
# media_pipe_process.start()

MediaPipeWorker.start()
input_image_obj = QueueImage(image=cv2image, timestamp=datetime.now())
image_queue.put(input_image_obj)
# results_queue.get(img).show()

MediaPipeWorker.join()
img = Image.fromarray(results_queue.get().image)
img.show()


"""
Add this code to the end of the run() function in mediapipe_process.py in order to test
the outputs without external scripts.
                test_image = self.output_queue.get()
                test_image = Image.fromarray(test_image.image)
                test_image.show()
"""
