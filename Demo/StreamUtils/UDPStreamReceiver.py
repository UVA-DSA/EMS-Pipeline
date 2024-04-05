# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 13:52:52 2019

@author: masavoyat
"""
import socket
import threading
import zlib
import struct
import time

class UDPStreamReceiver:
    _BUFFER_SIZE = 1024
    PAYLOAD_TYPES_STR = {127: "RAW_16BIT", 126: "RAW_8BIT", 125: "ZIP_16BIT", 124: "ZIP_8BIT", 0: "UNKNOWN"}
    def __init__(self, udp_port, udp_ip="", name=None):
        if name:
            self.name = name
        else:
            self.name = str(udp_port)
        self._thread = threading.Thread(target=self._main_loop)
        self._registeredQueueList = list()
        self._registeredQueueListLock = threading.Lock()
        self._sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
        self._sock.bind((udp_ip, udp_port))
        self._thread.start()
        self._infos = dict()
        self._infosLock = threading.Lock()
        self._infos["name"] = self.name
        self._infos["port"] = udp_port
        self._infos["ip"] = udp_ip
        self._infos["sampling_frequency"] = 0
        self._infos["payload_type"] = 0
        self._infos["registered_queue"] = 0
        self._infos["last_packet_time"] = 0
        
    def getInfos(self):
        self._infosLock.acquire()
        infos = self._infos.copy()
        self._infosLock.release()
        return infos
        
    def registerQueue(self, q):
        self._registeredQueueListLock.acquire()
        self._registeredQueueList.append(q)
        self._registeredQueueListLock.release()
        
    def unregisterQueue(self, q):
        self._registeredQueueListLock.acquire()
        if q in self._registeredQueueList:
            self._registeredQueueList.remove(q)
        self._registeredQueueListLock.release()
        
    def _main_loop(self):
        while True:
            try:
                data, addr = self._sock.recvfrom(UDPStreamReceiver._BUFFER_SIZE)
                _, payloadType, _, _, samplingFreq = struct.unpack(">BBHII", data[0:12])
                self._infosLock.acquire()
                self._infos["sampling_frequency"] = samplingFreq
                self._infos["payload_type"] = payloadType
                self._infos["registered_queue"] = len(self._registeredQueueList)
                self._infos["last_packet_time"] = time.time()
                self._infosLock.release()
                # Payload ZIP compressed
                if payloadType == 125 or payloadType == 124:
                    header = list(data[:12])
                    header[1] += 2 # payload is now raw
                    data = bytes(header) + zlib.decompress(data[12:])
            except:
                data = None # Send None data to advertise socket is dead
            self._registeredQueueListLock.acquire()
            for q in self._registeredQueueList:
                if q.full():
                    q.get_nowait()
                q.put_nowait(data)
            self._registeredQueueListLock.release()
            if not data:
                return
    
    def close(self):
        self._registeredQueueListLock.acquire()
        self._sock.close()
        self._registeredQueueListLock.release()
    
            
            