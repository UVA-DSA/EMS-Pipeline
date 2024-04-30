import multiprocessing
import numpy as np
from EMSMediaPipe.mediapipe_process import MediaPipeProcess
from PIL import Image

test_image = Image.open("/home/cogems_nist/Downloads/mediapipe_landmark.jpg")
test_image.show()

image_queue = multiprocessing.Queue(maxsize=1)
results_queue = multiprocessing.Queue()

MediaPipeRunner = MediaPipeProcess(input_queue=image_queue, output_queue=results_queue)

media_pipe_process = multiprocessing.Process(target=MediaPipeRunner.process_image)
media_pipe_process.start()

img = np.zeros([100,100,3],dtype=np.uint8)
img.fill(255) # or img[:] = 255

image_queue.put(img)
#results_queue.get(img).show()