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
    
    _instance = None
    _lock = threading.Lock()
    _num_instances = 0


    def __init__(self):
        super().__init__()
        self.is_connected = False
        # raise RuntimeError('use instance()')
        # #print(self._stop)

    @classmethod
    def instance(cls):
        if cls._instance is None:
            with cls._lock:
                if not cls._instance:
                    print("new instance of feedbackClient building...")
                    cls._instance = FeedbackClient()
                    #print('this is called')
                    
                    cls.sio = socketio.Client()
                    cls._sigstop = threading.Event()
                    print("feedback instance created")

        else:
            print("returning feedback instance")

        cls._num_instances += 1
        print("Feedback Number of instances: ", cls._num_instances)

        return cls._instance



    def stop(self):
        """Call this method to kill the thread."""
        self._sigstop.set()

    def run(self):
        """Inherited from threading.thread. Called in threading.thread.start()"""
        while True:
            try:
                # defined 1in pipeline_config.py
                self.sio.connect(socketio_ipaddr)
                print("Connected to SocketIO server!")
                self.sio.emit('message', 'Hello from Feedback Client!')  # Send a message to the server

                self.is_connected = True
                self.sio.wait()
            except Exception as e:
                print("Connection failed, retrying...", e)
                time.sleep(5)
        print("Succesfully exited feedback thread.")

    def sendMessage(self, message_obj):
        if(not self.is_connected):
            print("Not connected to server, cannot send message.")
            return
        #print("Sent message to feedback ", message_obj)
        # if self._sigstop.is_set():
        #     print('Cannot send a message. The connection to the server has been killed.')
        # else:
        self.sio.emit(feedback_topic, message_obj)
        print(f"Sent message to feedback {message_obj}")
