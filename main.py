import multiprocessing as mp
from multiprocessing import Process, Queue
#""event being set corresponds to the start button getting clikced""
############################################################################################################################################################################################################
"' Code for the GUI"

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtMultimedia import *


from multiprocessing import Event
from datetime import datetime

######################
"'Process spawning"
def protocol_process(event):
    while event.is_set() is True:
        print("Protocol Process/Thread Running at ",datetime.now())

def feedback_process(event):
    while event.is_set() is True:
        print("Feedback Process Running at ",datetime.now())

def speech_process(event):
    while event.is_set() is True:
        print("Speech Process Running at ",datetime.now())
    
def vision_process(event):
    while event.is_set() is True:
        print("Vision process Running at ",datetime.now())

def network_process(event):
    while event.is_set() is True:
        print("Network Process Running at ",datetime.now())

######################
###avoid terminate since it is not a clean shutdown of the process


"'GUI/Window  class"
class gui_window(QWidget):
    def __init__(self,application,queue,width,height):
        super(gui_window,self).__init__()
        self.app = application
        self.command_queue = queue
        
        main_layout = QVBoxLayout()
        
        ##setting up window
        self.width = width
        self.height = height
        self.setWindowTitle('CognitiveEMS Debugging Demo')
        self.setLayout(main_layout)
        self.setGeometry(0, 0, self.width, self.height)
        self.setStyleSheet("background-color: #2E2E2E; color: white; font-size: 16px; font-family: Arial;")

        ################button creation
        self.protocol_button_start = QPushButton('Protocol Start', self)
        self.protocol_button_stop = QPushButton('Protocol Stop',self)
        main_layout.addWidget(self.protocol_button_start)
        main_layout.addWidget(self.protocol_button_stop)
        print("Protocol buttons created")

        self.vision_start_button = QPushButton('Vision Start',self)
        self.vision_stop_button = QPushButton('Vision Stop' , self)
        main_layout.addWidget(self.vision_start_button)
        main_layout.addWidget(self.vision_stop_button)
        print("Vision buttons created")

        self.network_start_button = QPushButton('Network Start',self)
        self.network_end_button = QPushButton('Network Stop',self)
        main_layout.addWidget(self.network_start_button)
        main_layout.addWidget(self.network_end_button)  
        print("Network buttons created")

        self.speech_start_button = QPushButton('Speech Start',self)
        self.speech_stop_button = QPushButton('Speech Stop',self)
        main_layout.addWidget(self.speech_start_button)
        main_layout.addWidget(self.speech_stop_button)  
        print("Speech buttons created")

        self.feedback_start_button = QPushButton('Feedback Start',self)
        self.feedback_stopped_button = QPushButton('Feedback Stop',self)
        main_layout.addWidget(self.feedback_start_button)
        main_layout.addWidget(self.feedback_stopped_button)
        print("Feedback buttons created")
        



    #########################################Text box updates and set up
        self.protocol_box = QTextEdit()
        self.protocol_box.setReadOnly(True)
        self.protocol_box.setStyleSheet("background-color: #1E1E1E; color: white; font-size: 14px; font-family: Arial;")    
        self.protocol_box.setOverwriteMode(True)
        main_layout.addWidget(self.protocol_box)
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
        self.protocol_event.set()
        self.protocol = mp.Process(target=protocol_process,args=(self.protocol_event,))
        print("Protocol start")
        self.processes.append(self.protocol)
        self.protocol.start()

    def stop_protocol(self): 
        self.protocol_event.clear()
        self.processes.remove(self.protocol)
        print("Protocol stopped")

    def vision_start(self):
        self.vision_event.set()
        self.vision = mp.Process(target=vision_process,args=(self.vision_event,))
        print("Vision start")
        self.processes.append(self.vision)
        self.vision.start()

    def vision_stop(self):
        self.vision_event.clear()
        print("Vision stopped")

    def network_start(self):
        self.network_event.set()
        self.network = mp.Process(target=network_process,args=(self.network_event,))
        print("Network started")
        self.processes.append(self.network)
        self.network.start()

    def network_stop(self):
        self.network_event.clear()
        print("Network stopped")

    def speech_start(self):
        self.speech_event.set()
        self.speech = mp.Process(target=speech_process,args=(self.speech_event,))
        print("Speech started")
        self.processes.append(self.speech)
        self.speech.start()

    def speech_stop(self):
        self.speech_event.clear()
        print("Speech stopped")

    def feedback_start(self):
        self.feedback_event.set()
        self.feedback = mp.Process(target=feedback_process,args=(self.feedback_event,))
        print("Feedback started")
        self.processes.append(self.feedback)
        self.feedback.start()
        
    def feedback_stop(self):
        self.feedback_event.clear()
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
    width, height = application.desktop().screenGeometry().width(), application.desktop().screenGeometry().height()
    command_queue = Queue()
    Window = gui_window(application=application,queue=command_queue,width=width,height=height)
    
    


    Window.show()
    ##Debug information
    print(f"width: {Window.width}")
    print(f"height: {Window.height}")
    sys.exit(application.exec_()) # Start the event loop
####################################################################################
#if you do not want to load in the process each time the start button is called then you can create the processes outside of the class and just pass htem in as class parameters to the initializatino at which point it will just be equated to the class variable process for each of the 5 process
