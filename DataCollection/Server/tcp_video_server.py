import socket
import socket
from PIL import Image
import cv2
import io
import numpy as np
import time
from datetime import datetime
import struct
import os
import logging
import csv
import math

# Create and configure logger
logging.basicConfig(filename="server.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
 

# Creating an object
logger = logging.getLogger()
 
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)


# Test messages
# logger.debug("Harmless debug Message")
# logger.info("Just an information")
# logger.warning("Its a Warning")
# logger.error("Did you try to divide by zero")
# logger.critical("Internet is down")


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
sock.bind(('0.0.0.0', 8899))  
sock.listen(5)  

print("Waiting for client...")

connection,address = sock.accept()  
print("Client connected: ",address)

# connection.send(b'Oi you sent something to me')




def get_image():
    try:
        buf = connection.recv(4)

        if( not buf):
            return -1

        d_type=int.from_bytes(buf,"big")

        if(d_type!=22):
            return -1

        # print("received: ",d_type)     

        seq=int.from_bytes(connection.recv(4),"big")
        epoch=int.from_bytes(connection.recv(8),"big")
        width=int.from_bytes(connection.recv(4),"big")
        size=int.from_bytes(connection.recv(4),"big")

        img=bytearray()
        print(epoch)

        if(size>1000000):
            return -1

        while(size>0):
            read_len=min(size,1024)
            data = connection.recv(read_len)
            size -= len(data)
            img+=data
            # print("size - data:",size)
            
            
        image = Image.open(io.BytesIO(img))
        img_ar=np.array(image)
    except Exception as e:
        logger.error(e)
        return -1


    return img_ar,seq,epoch

# while True:
#     print(get_image())

recording_enabled = False
frame_index = 0

curr_date = datetime.now()
dt_string = curr_date.strftime("%d-%m-%Y-%H-%M-%S")

newpath = "./video_data/"+dt_string

    
if(recording_enabled):

    if not os.path.exists(newpath):
        os.makedirs(newpath)

    with open('./video_data/'+dt_string+'/'+dt_string+'data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "recieved_ts", "origin_ts"])

        is_video_created = False

        while True:
            if( int.from_bytes(connection.recv(1),"big") == 22):
                print("Image received")

                timestamp = int.from_bytes(connection.recv(8),"big")
                print("Got timestamp:" , timestamp)

                img_byte_length = int.from_bytes(connection.recv(8),"big")
                print("Got Num of bytes in the image:" , img_byte_length)

                # img_buffer_size = math.ceil(img_byte_length/1024)*1024
                img_buffer_size = img_byte_length
                print("Buff Size Should Be: ", img_buffer_size)

                # if(img_buffer_size > 180000 or img_buffer_size < 10000):
                #     continue

                img_bytes = bytearray()

                while(img_buffer_size>0):
                    read_len=min(img_buffer_size,10240)
                    data = connection.recv(read_len)
                    img_buffer_size -= len(data)
                    img_bytes+=data
                    # print("Remaining buffer size: ",img_buffer_size)


                image = Image.open(io.BytesIO(img_bytes))
                img_ar=np.array(image)

                RGB_img = cv2.cvtColor(img_ar, cv2.COLOR_BGR2RGB)
                cv2.imshow('frame',RGB_img)

                if(recording_enabled and not is_video_created):

                    height,width,layers=img_ar.shape
                    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
                    video = cv2.VideoWriter('./'+dt_string+'/'+dt_string+'.avi', fourcc, 30, (width, height))
                    is_video_created = True

                if(recording_enabled):
                    now = time.time()*1e3

                    cv2.imwrite('./'+ dt_string + '/img_'+str(frame_index)+'.jpg', img_ar)
                    video.write(RGB_img)
                    writer.writerow([frame_index, now, timestamp])
                    frame_index += 1


                if (cv2.waitKey(1) & 0xFF == ord('q')):
                    break
            else:
                print("Garbage")

else:

    while True:
        if( int.from_bytes(connection.recv(1),"big") == 22):
            print("Image received")

            timestamp = int.from_bytes(connection.recv(8),"big")
            print("Got timestamp:" , timestamp)

            img_byte_length = int.from_bytes(connection.recv(8),"big")
            print("Got Num of bytes in the image:" , img_byte_length)

            # img_buffer_size = math.ceil(img_byte_length/1024)*1024
            img_buffer_size = img_byte_length
            print("Buff Size Should Be: ", img_buffer_size)

            # if(img_buffer_size > 180000 or img_buffer_size < 10000):
            #     continue

            img_bytes = bytearray()

            while(img_buffer_size>0):
                read_len=min(img_buffer_size,10240)
                data = connection.recv(read_len)
                img_buffer_size -= len(data)
                img_bytes+=data
                # print("Remaining buffer size: ",img_buffer_size)


            image = Image.open(io.BytesIO(img_bytes))
            img_ar=np.array(image)

            RGB_img = cv2.cvtColor(img_ar, cv2.COLOR_BGR2RGB)
            cv2.imshow('frame',RGB_img)

            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
        else:
            print("Garbage")
        


""" LAHIRUS CODE TO RECIEVE IMAGES """
    # while True:  
    #     try: 
            
    #         image=get_image()

    #         if type(image) == tuple :

    #             if(not is_video_created):

    #                 height,width,layers=image[0].shape
    #                 fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    #                 # video = cv2.VideoWriter('./'+dt_string+'/'+dt_string+'.avi', fourcc, 30, (width, height))

    #                 is_video_created = True

    #             now = time.time()*1e3

    #             # cv2.imwrite('./'+ dt_string + '/img_'+str(now)+'.jpg', image[0])
    #             RGB_img = cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB)
    #             video.write(RGB_img)
    #             cv2.imshow('frame',RGB_img)

    #             # writer.writerow([frame_index, now, image[2]])
    #             # frame_index += 1

    #             if (cv2.waitKey(1) & 0xFF == ord('q')):
    #                 break

    #     except Exception as e:
    #         logger.error(e)