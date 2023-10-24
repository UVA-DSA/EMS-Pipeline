import cv2
import numpy as np
import time
import os

from PIL import Image
from skimage.metrics import structural_similarity


def VideoStream(VisionDataQueue,VideoSignalQueue, video_file_name):
    
    cap = cv2.VideoCapture(video_file_name) # video file path
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
      print("[Video Stream Thread: Error opening video stream or file]")
    
    frame = np.array([])
    fps = (cap.get(cv2.CAP_PROP_FPS))
    print('vid fps',fps)
    idx = 0

    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
      before = frame
      print("Video Stream Alive!")
      ret, frame = cap.read()
      if ret == True:
        
        scale_percent = 50 # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        
          
        # resize image
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        
        
        start_t = time.time_ns()
        
        # Convert images to grayscale
        after_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # # Compute SSIM between two images
        # (score, diff) = structural_similarity(before_gray, after_gray, full=True)
        
        
        # print("latency (ms)", (end_t-start_t)/1e6)
        # print("Image similarity, Latency (ms)", score, (end_t-start_t)/1e6)
      
        # Display the resulting frame
        pil_image = Image.fromarray(after_gray)
        end_t = time.time_ns()
        
        time.sleep(1/fps)
        print("[VISION Stream Sent Image!]", VisionDataQueue.full())
        VisionDataQueue.put({"signal":"Proceed","image":pil_image})
    
      # signal = VideoSignalQueue.get()
      
      # if(signal == "Kill"): 
      #   print("[Video Stream Thread received Kill Signal. Bye!]")
      #   VisionDataQueue.empty()
      #   VisionDataQueue.put({"signal":"Kill","image":None})
      #   break

      else:
        break
    # When everything done, release the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()
    
    print("[Video Stream Thread Died. Sending Kill Signal. Bye!]")
    VisionDataQueue.empty()
    VisionDataQueue.put({"signal":"Kill","image":None})
