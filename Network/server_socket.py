import socketio
import datetime
import asyncio
import aiohttp as web
import threading 
import socket




class client_network():
    def __init__(self,IP,PORT,WEB_PORT):
        self.ip = IP
        self.socket_port = PORT
        self.web_port = WEB_PORT
        self.server = socketio.AsyncServer(async_mode='aiohttp',cors_allowed_origins='*')
        self.application = web.Application()
        self.server.attach(self.application)
    def setup(self):
        print("Creating Asyncio Task for Receiving Data Over UDP")
        asyncio.create_task(self.transfer())
        return self.application
    def event_setup(self):
        @self.server.event
        async def connect(sid,environ):
            print("Connected: ",sid)
        @self.server.event
        async def disconnect(sid):
            print("Disconnected: ", sid)
        @self.server.on("audio")
        async def audio_received(sid,data):
            print("[SOCKET][AUDIO] Audio data is received over the network")
        @self.server.on("video")
        async def video_received(sid,data):
            print("[SOCKET][VIDEO] Video frames received over the network")
    def server(self):
        print("Starting the web server")
        aiohttp.web.run_app(self.setip(),host="0.0.0.0",port=self.web_port)
        print("Web server has ended")
    