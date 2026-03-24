
import threading
import socketio
import time

import sys
sys.path.append("../")

from pipeline_config import socketio_ipaddr, feedback_topic


class returnFeedback(threading.Thread):

    def __init__(self):
        super(returnFeedback,self).__init__()
        self.sio = socketio.Client()
        self._sigstop = threading.Event()


        @self.sio.on(feedback_topic)
        def on_message(data):
            print("Message recieved: " , data)
    

    def stop(self):
        self._sigstop.set()

        
    def run(self):
        while not self._sigstop.is_set():
            try:
                self.sio.connect(socketio_ipaddr)
                self.sio.wait()

            except Exception as e:
                print(e)
                print("Connection failed")
                time.sleep(5)


feedbackTest = returnFeedback()

feedbackTest.start()


        
