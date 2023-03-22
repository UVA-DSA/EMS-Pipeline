#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:51:50 2021

@author: sleekeagle
"""
import socket
from PIL import Image
import cv2
import io
import numpy as np
import time
from datetime import datetime
import struct
import os

TCP_IP = '127.0.0.1'
TCP_PORT = 8899

def connect():
    read_int=-1
    print("waiting for connection....")
    while(read_int!=100):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((TCP_IP, TCP_PORT))
            read_int=int.from_bytes(s.recv(1),"big")
        except Exception as e:
            print(e)
    print("connected")
    return s
    
    
def get_next_image(s):
    try:
        d_type=int.from_bytes(s.recv(1),"big")      
        if(d_type!=22):
            return -1
        seq=int.from_bytes(s.recv(4),"big")
        height=int.from_bytes(s.recv(4),"big")
        width=int.from_bytes(s.recv(4),"big")
        size=int.from_bytes(s.recv(4),"big")
        if(size>1000000):
            return -1
        img=bytearray()
        print(size)
        while(size>0):
            read_len=min(size,1024)
            data = s.recv(read_len)
            size -= len(data)
            img+=data
            
        image = Image.open(io.BytesIO(img))
        img_ar=np.array(image)
    except:
        return -1
        
    return img_ar,seq
    
#make sure the app is installed on the phone
print("port forwarding....")
ret=os.system("adb forward tcp:9600 tcp:9600")
if(ret==-1):
    print("port forwarding error. Exiting...")
    quit()
print("return value from OS = "+str(ret))
#stopping the app (we need to restart)
ret=os.system("adb shell am force-stop com.example.camstrm")
if(ret==-1):
    print("error when trying to stop the app. Exiting...")
    quit()
#start the app
print("starting the camstream app...")
ret=os.system("adb shell am start -n com.example.camstrm/com.example.camstrm.MainActivity")
if(ret==-1):
    print("Error when starting app with adb. Exiting...")
    quit()
print("return value from OS = "+str(ret))
    
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
s=connect()

p = struct.pack('!i', 23)
s.send(p)

image=-1
while(type(image)!=tuple):
    try:
        image=get_next_image(s)
        height,width,layers=image[0].shape
        #fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        video = cv2.VideoWriter('/home/sleekeagle/vuzix/data/'+dt_string+'.avi', fourcc, 30, (width, height))
    except:
        print('exception')

while(True):
    try:
        image=get_next_image(s)  
        if(type(image)==tuple):
            RGB_img = cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB)
            video.write(RGB_img)
            cv2.imshow('frame',RGB_img)
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
            #cv2.imwrite('/home/sleekeagle/vuzix/data/img.jpg', image[0])
            print(image[0].shape)
    except:
        print("exception..")

print("done")  

video.release() 
cv2.destroyAllWindows()
