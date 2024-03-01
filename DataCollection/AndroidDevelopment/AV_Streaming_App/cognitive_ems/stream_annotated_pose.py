#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 15:38:43 2022

@author: sleekeagle


this code received the images from ADB connection 
and 
extract the pose keypoints 
and 
display them in real time 
"""
import cv2
import mediapipe as mp
import time
import mediapipe_pose
import client
import struct


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

#connect to server
s=client.connect()

#ask the server to send images
p = struct.pack('!i', 23)
s.send(p)


#processing all the frames in real time makes the code laggish
#so we have to skip (and ignore) some frames

SKIP_FRAMES=5
i=0
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    
    while(True):
        i+=1
        image=client.get_next_image(s)
        if(i<SKIP_FRAMES):
            continue
        i=0
        if(type(image)==tuple):
            image=image[0]
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            
            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            #press q to quit the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
cv2.destroyAllWindows()     
        




