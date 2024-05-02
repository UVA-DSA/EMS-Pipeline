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

    def __init__(self):
        print('this is called')
        super(FeedbackClient, self).__init__()
        self.sio = socketio.Client()
        self._sigstop = threading.Event()
        #print(self._stop)

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
            self.sio.emit(feedback_topic, 'teststr')