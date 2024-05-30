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

    def __init__(self):
        raise RuntimeError('use instance()')
        # #print(self._stop)

    @classmethod
    def instance(cls):
        if cls._instance is None:
            with cls._lock:
                if not cls._instance:
                    print("new instance of feedbackClient building...")
                    cls._instance = super().__new__(cls)
                    #print('this is called')
                    #super(FeedbackClient, self).__init__()
                    cls.sio = socketio.Client()
                    cls._sigstop = threading.Event()
        return cls._instance



    def stop(self):
        """Call this method to kill the thread."""
        self._sigstop.set()

    def run(self):
        """Inherited from threading.thread. Called in threading.thread.start()"""
        while not self._sigstop.is_set():
            try:
                # defined in pipeline_config.py
                self.sio.connect(socketio_ipaddr)
                self.sio.wait()
            except Exception as e:
                print("Connection failed, retrying...", e)
                time.sleep(5)
        print("Succesfully exited feedback thread.")

    def sendMessage(self, message_obj):
        if self._sigstop.is_set():
            print('Cannot send a message. The connection to the server has been killed.')
        else:
<<<<<<< HEAD
=======
            print(f"Sent message to feedback {message_obj}")
>>>>>>> 672dcf2fdc34eb453f5b92594fd52172060c7938
            self.sio.emit(feedback_topic, message_obj)