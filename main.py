import multiprocessing as mp
from multiprocessing import Process, Queue


def spawn_vision():
    return
def spawn_network():
    return




############################################################################################################################################################################################################
"' Code for the GUI"

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGUI import *
from PyQt5 import QTCore
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtMultimedia import *


from multiprocessing import Event

######################
"'Process spawning"
def protocol_process(event):
    while True:
        print("Protocol Process/Thread Running")

def feedback_process(event):
    while True:
        print("Feedback Process Running")

def speech_process(event):
    while True:
        print("Speech Process Running")
    
def vision_process(event):
    while True:
        print("Vision process Running")

def network_process(event):
    while True:
        print("Network Process Running")

######################
###avoid .terminate since it is not a clean shutdown of the process


"'GUI/Window  class"
class gui_window(QWidget):
    def __init__(self,application,queue):
        super(gui_window,self).__init__()
        self.app = application
        self.command_queue = queue
        
        main_layout = QVBoxLayout()
        
        ##setting up window
        self.screen_resolution = self.app.desktop().screenGeometry ##Todo: test to make sure this actually works
        self.width = self.screen_resolution.width()
        self.height = self.screen_resolution.height()
        self.setWindowTitle('CognitiveEMS Debugging Demo')


        ################button creatoin
        self.protocol_button_start = QPushButton('Protocol Start', self)
        self.protocol_button_stop = QPushButton('Protocol Stop',self)
        main_layout.addWidget(self.protocol_button_start)
        main_layout.addWidget(self.protocol_button_stop)

        self.vision_button_start = QPushButton('Vision Start',self)
        self.vision_button_stop = QPushButton('Vision Stop' , self)
        main_layout.addWidget(self.vision_button_start)
        main_layout.addWidget(self.vision_button_stop)
        


        ####event creation for each possible process
        self.protocol_event = Event()
        self.vision_event = Event()
        self.feedback_event = Event()
        self.speech_event = Event()
        self.network_event = Event()
        self.processes = []
        #list of all processes

        ##setting up button 
        self.protocol_button_start.clicked.connect(self.start_protocol)
        self.protocol_button_stop.clicked.connect(self.stop_protocol)
        self.speech_start_button.clicked.connect(self.speech_start)
        self.speech_stop_button.clicked.connect(self.speech_stop)
        self.feedback_start_button.clicked.connect(self.feedback_start)
        self.feedback_stopped_button.clicked.connect(self.feedback_stop)
        self.network_start_button.clicked.connect(self.network_start)
        self.network_end_button.clicked.connect(self.network_stop)
        self.vision_start_button.clicked.connect(self.vision_start)
        self.vision_stop_button.clicked.connect(self.vision_stop)



###function for what happens when a button is clicked
    def start_protocol(self):
        self.protocol_event.clear()
        self.protocol = mp.Process(target=protocol_process,args=(self.protocol_event,))
        print("Protocol start")
        self.processes.append(self.protocol)
        self.protocol.start()

    def stop_protocol(self): 
        self.protocol_event.set()
        print("Protocol stopped")

    def vision_start(self):
        self.vision_event.clear()
        self.vision = mp.Process(target=vision_process,args=(self.vision_event,))
        print("Vision start")
        self.processes.append(self.vision)
        self.vision.start()

    def vision_stop(self):
        self.vision_event.set()
        print("Vision stopped")

    def network_start(self):
        self.network_event.clear()
        self.network = mp.Process(target=network_process,args=(self.network_event,))
        print("Network started")
        self.processes.append(self.network)
        self.network.start()

    def network_stop(self):
        self.network_event.set()
        print("Network stopped")

    def speech_start(self):
        self.speech_event.clear()
        self.speech = mp.Process(target=speech_process,args=(self.speech_event,))
        print("Speech started")
        self.processes.append(self.speech)
        self.speech.start()

    def speech_stop(self):
        self.speech_event.set()
        print("Speech stopped")

    def feedback_start(self):
        self.feedback_event.clear()
        self.feedback = mp.Process(target=feedback_process,args=(self.feedback_event,))
        print("Feedback started")
        self.processes.append(self.feedback)
        self.feedback.start()
        
    def feedback_stop(self):
        self.feedback_event.set()
        print("Feedback stopped")
#################################################




############################################################################################################################################################################################################



"'Code for Multiprocessing set up"
import py_trees 
from py_trees.blackboard import Blackboard


##############
"'Code that runs on start up'"
import sys

if __name__ == "__main__":
    
    print("Starting GUI!")
    application = QApplication(sys.argv)
    command_queue = Queue()
    Window = gui_window(application=application,queue=command_queue)
    
    
    ##Debug information
    print(f"width: {Window.width}")
    print(f"height: {Window.height}")

    Window.show()
