import socketio
import datetime
import asyncio
from aiohttp import web
import threading 
import socket
import aiohttp



class server_network():
    def __init__(self,IP,PORT,WEB_PORT,video_queue,audio_queue):
        self.ip = IP
        self.socket_port = PORT
        self.web_port = WEB_PORT
        self.server = socketio.AsyncServer(async_mode='aiohttp',cors_allowed_origins='*')
        self.application = web.Application()
        self.server.attach(self.application)
        self.audio_queue = audio_queue
        self.video_queue = video_queue
    def set_listener(self,LISTENER):
        self.list = LISTENER
    async def setup(self):
        print("Creating Asyncio Task for Receiving Data Over UDP")
        asyncio.create_task(self.list.listen())
        return self.application
    def event_setup(self):
        @self.server.event
        async def connect(sid,environ):
            print("Connected: ",sid)
        @self.server.event
        async def disconnect(sid):
            print("Disconnected: ", sid)
        @self.server.on("audio")
        async def audio_received(sid,data): #send this to audio thread
            print("[SOCKET][AUDIO] Audio data is received over the network")
            self.audio_queue.put(data)
        @self.server.on("video")
        async def video_received(sid,data): #sends this to the video thread
            print("[SOCKET][VIDEO] Video frames received over the network")
            self.video_queue.put(data)
    def setup_server(self):
        print("Starting the web server")
        aiohttp.web.run_app(self.setup(),host="0.0.0.0",port=self.web_port)
        print("Web server has ended")
    def close(self,s):
        s.close()
    def return_server(self):
        return self.server


class listener():
    def __init__(self,IP,SOCKET_PORT, SERVER):
        self.ip = IP
        self.socket_port = SOCKET_PORT
        self.server = SERVER

    async def listen(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(5)
            connection = sock.bind((self.ip,self.socket_port))
            while True:
                try:
                    data, discard = await asyncio.get_event_loop().sock_recvfrom(sock,65535)
                    print(f"Data Received over UDP at {datetime.now()}: {data}")
                    if 'audio' in data:
                        await self.server.emit("audio",data)
                    elif 'video' in data:
                        await self.server.emit('video',data)
                    else:
                        print(f"[EXTRANEOUS] Data:{data}")
                except Exception as e:
                    print(f"Error Message: {e}")        
    