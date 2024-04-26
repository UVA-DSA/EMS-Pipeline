import multiprocessing
import numpy as np
from .mediapipe_process import MediaPipeProcess

image_queue = multiprocessing.Queue()
results_queue = multiprocessing.Queue()

MediaPipeRunner = MediaPipeProcess(input_queue=image_queue, output_queue=results_queue)

media_pipe_process = multiprocessing.Process(target=MediaPipeRunner.process_image)
media_pipe_process.start()

img = np.zeros([100,100,3],dtype=np.uint8)
img.fill(255) # or img[:] = 255

while True:
    image_queue.put(img)