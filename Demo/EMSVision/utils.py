import glob
import numpy as np
import time
import cv2
import collections
import os
from PIL import Image
import itertools
from EMSVision.ems_knowledge import *

def generate_labels(protocol):
    if protocol in ems_interventions:
        return list(itertools.chain.from_iterable(list(ems_interventions.get(protocol).values())))
    else:
        return None
    
def classify(image,labels,classifier):
    start_t = time.time_ns()
    scores = classifier(image, 
                        candidate_labels = labels)
    end_t = time.time_ns()
    
    latency = (end_t-start_t)/1e6
    
    return scores,latency