# import time
# import socketio
# import threading
# from pipeline_config import socketio_ipaddr, feedback_topic


# class FeedbackClient(threading.Thread):
#     """ Flask client for sending any "Feedback" object to the central server.

#     Attributes:
#         sio (socketio.Client): sets up the socketio connection
#         _sigstop (threading.Event): stop signal
    # """
    
    # # _instance = None
    # # _lock = threading.Lock()
    # # _num_instances = 0

    # # def __init__(self):
    # #     super().__init__()
    # #     self.sio = socketio.Client()
    # #     self._sigstop = threading.Event()
    # #     self.is_connected = False

    # # @classmethod
    # # def instance(cls):
    # #     if cls._instance is None:
    # #         with cls._lock:
    # #             if cls._instance is None:
    # #                 print("New instance of FeedbackClient building...")
    # #                 cls._instance = FeedbackClient()
    # #                 print("Feedback instance created")
    # #     else:
    # #         print("Returning existing Feedback instance")

    # #     cls._num_instances += 1
    # #     print("Feedback Number of instances: ", cls._num_instances)

    # #     return cls._instance

    # def __init__(self):
    #     self.sio = socketio.Client()
    #     self._sigstop = threading.Event()
    

    # def stop(self):
    #     """Call this method to kill the thread."""
    #     self._sigstop.set()

    # def run(self):
    #     """Inherited from threading.Thread. Called in threading.Thread.start()"""
    #     while not self._sigstop.is_set():
    #         try:
    #             self.sio.connect(socketio_ipaddr)
    #             print("Connected to SocketIO server!")
    #             self.is_connected = True
    #             self.sio.wait()
    #         except Exception as e:
    #             print("Connection failed, retrying...", e)
    #             self.is_connected = False
    #             time.sleep(5)
    #     self.sio.disconnect()
    #     print("Successfully exited feedback thread.")

    # def send_message(self, message_obj):
    #     if not self.is_connected:
    #         print("Not connected to server, cannot send message.")
    #         return
    #     self.sio.emit(feedback_topic, message_obj)
    #     print(f"Sent message to feedback {message_obj}")


import time
import socketio
import threading
from pipeline_config import socketio_ipaddr, feedback_topic


class FeedbackClient(threading.Thread):
    """ Flask client for sending any "Feedback" object to the central server.

    Attributes:
        sio (socketio.Client): sets up the socketio connection
        _sigstop (threading.Event): stop signal
    """
    
    # _instance = None
    # _lock = threading.Lock()
    # _num_instances = 0

    def __init__(self):
        super().__init__()
        self.sio = socketio.Client()
        self._sigstop = threading.Event()
        self.is_connected = threading.Event() #changed from threading.Event()

       # Event handlers for socketio
        self.sio.on('connect', self.on_connect)
        self.sio.on('disconnect', self.on_disconnect)
        self.sio.on('connect_error', self.on_connect_error)

    # @classmethod
    # def instance(cls):
    #     if cls._instance is None:
    #         with cls._lock:
    #             if cls._instance is None:
    #                 print("New instance of FeedbackClient building...")
    #                 cls._instance = FeedbackClient()
    #                 print("Feedback instance created")
    #     else:
    #         print("Returning existing Feedback instance")

    #     cls._num_instances += 1
    #     print("Feedback Number of instances: ", cls._num_instances)

    #     return cls._instance

    def stop(self):
        """Call this method to kill the thread."""
        self._sigstop.set() 

    def run(self):
        """Inherited from threading.Thread. Called in threading.Thread.start()"""
        while not self._sigstop.is_set(): 
            try:
                self.sio.connect(socketio_ipaddr)
                print("Connected to SocketIO server!")
                self.sio.emit('message', 'Hello from Feedback Thread!')  # Send a message to the server

                self.is_connected.set()
                self.sio.wait()
            except Exception as e:
                print("Connection failed, retrying...", e)
                self.is_connected.clear()
                time.sleep(5)
        self.sio.disconnect()
        print("Successfully exited feedback thread.")

    def send_message(self, message_obj):
        if not self.is_connected.is_set():
            print("Not connected to server, cannot send message.")
            return
        self.sio.emit(feedback_topic, message_obj)
        #print(f"Sent message to feedback: {message_obj}")

    def on_connect(self):
        print("SocketIO connected")
        self.is_connected.set()

    def on_disconnect(self):
        print("SocketIO disconnected")
        self.is_connected.clear()

    def on_connect_error(self, data):
        print("SocketIO connection error:", data)
        self.is_connected.clear()