import socketio
import asyncio
from time import sleep
import aiohttp
from aiohttp import web
# async def main():
#     sio = socketio.AsyncClient()
#     await sio.connect('http://localhost:8080')
#     sio.wait()
#     audio_or_video = 0
#     while True:
#         

# async def secondary_init():
#     print("Creating asynchronous task")
#     asyncio.create_task(main()) #runs in background
#     print("ASynchronous task created")

# def init():

#     aiohttp.web.run_app(secondary_init(),host="0.0.0.0",port=8080)
from socketIO_client import SocketIO
print("Starting client code")
audio_or_video = 0
with SocketIO('http://localhost:0.0.0.0', 8080) as sio:
    if audio_or_video == 0:
        sio.emit('audio', {'data': 'AUDIO SIGNAL'})
        audio_or_video = 1
        print("[CLIENT][AUDIO] emitted")
    else:
        sio.emit('video',{'data':'VIDEO FRAMES'})
        audio_or_video = 0
        print("[CLIENT][VIDEO] Emitted")
   
    print("Sent")
    sleep(5)

print("Program ended")
