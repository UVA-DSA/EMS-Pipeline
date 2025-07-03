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
        message = "Protocol Process/Thread Running at " + str(datetime.now())
        print(message)


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
        #self.setWindowTitle('CognitiveEMS Debugging Demo')
        self.main_title = QLabel(self)
        self.main_title.setText("CognitiveEMS Debugging Demo")
        self.setLayout(main_layout)
        self.setGeometry(0, 0, self.width, self.height)
        self.setStyleSheet("background-color: #2E2E2E; color: white; font-size: 16px; font-family: Arial;")
        main_layout.addWidget(self.main_title)

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
        self.protocol_box.setText("Protocol Log:\n")
        self.protocol_update = pyqtSignal(str)


        self.vision_box = QTextEdit()
        self.vision_box.setReadOnly(True)
        self.vision_box.setStyleSheet("background-color:#1E1E1E; font-size: 14px; font-family: Arial;")
        self.vision_box.setOverwriteMode(True)
        main_layout.addWidget(self.vision_box)
        self.vision_box.setText("Vision Log:\n")

        self.network_box = QTextEdit()
        self.network_box.setReadOnly(True)
        self.network_box.setStyleSheet("background-color:#1E1E1E; font-size: 14px; font-family: Arial;")
        self.network_box.setOverwriteMode(True)
        main_layout.addWidget(self.network_box)
        self.network_box.setText("Network Log:\n")

        self.speech_box = QTextEdit()
        self.speech_box.setReadOnly(True)
        self.speech_box.setStyleSheet("background-color:#1E1E1E; font-size: 14px; font-family: Arial;")
        self.speech_box.setOverwriteMode(True)
        main_layout.addWidget(self.speech_box)
        self.speech_box.setText("Speech Log:\n")

        self.feedback_box = QTextEdit()
        self.feedback_box.setReadOnly(True)
        self.feedback_box.setStyleSheet("background-color:#1E1E1E; font-size: 14px; font-family: Arial;")
        self.feedback_box.setOverwriteMode(True)
        main_layout.addWidget(self.feedback_box)
        self.feedback_box.setText("Feedback Log:\n")    

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background-color:#1E1E1E; font-size: 14px; font-family: Arial;")
        self.log_box.setOverwriteMode(True)
        main_layout.addWidget(self.log_box)
        self.log_box.setText("Log:\n")
        ######################

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
        self.log_box.append("Protocol started at " + str(datetime.now()))
        self.protocol.start()

    def stop_protocol(self): 
        try:
            self.protocol_event.clear()
            if self.protocol is not None or self.protocol in self.processes:
                self.protocol.terminate()  # Ensure the process is terminated
                self.protocol.join()  # Wait for the process to finish
                self.processes.remove(self.protocol)
                self.log_box.append("Protocol stopped at " + str(datetime.now()))
                print("Protocol stopped")
        except:
            pass

    def vision_start(self):
        self.vision_event.set()
        self.vision = mp.Process(target=vision_process,args=(self.vision_event,))
        print("Vision start")
        self.processes.append(self.vision)
        self.log_box.append("Vision started at " + str(datetime.now()))
        self.vision.start()

    def vision_stop(self):
        try: #used to prevent premature exiting
            self.vision_event.clear()
            if self.vision is not None and self.vision in self.processes:# Handles case where the vision process might not be active or a process that did not proper get cleaned up 
                print("Vision stopped")
                self.vision.terminate()  # Ensure the process is terminated
                self.vision.join()  # Wait for the process to finish
                self.processes.remove(self.vision)
                self.log_box.append("Vision stopped at " + str(datetime.now()))
        except:
            pass

    def network_start(self):
        self.network_event.set()
        self.network = mp.Process(target=network_process,args=(self.network_event,))
        print("Network started")
        self.processes.append(self.network)
        self.log_box.append("Network started at " + str(datetime.now()))
        self.network.start()

    def network_stop(self):
        try:
            self.network_event.clear()
            if self.network is not None and self.network in self.processes:
                print("Network stopped")
                self.network.terminate()  # Ensure the process is terminated
                self.network.join()  # Wait for the process to finish
                self.processes.remove(self.network)
                self.log_box.append("Network stopped at " + str(datetime.now()))
        except:
            pass

    def speech_start(self):
        self.speech_event.set()
        self.speech = mp.Process(target=speech_process,args=(self.speech_event,))
        print("Speech started")
        self.processes.append(self.speech)
        self.log_box.append("Speech started at " + str(datetime.now()))
        self.speech.start()

    def speech_stop(self):
        try:
            self.speech_event.clear()
            if (self.speech is not None) and self.speech in self.processes:
                print("Speech stopped")
                self.speech.terminate()  # Ensure the process is terminated
                self.speech.join()  # Wait for the process to finish
                self.processes.remove(self.speech)
                self.log_box.append("Speech stopped at " + str(datetime.now()))
        except:
            pass

    def feedback_start(self):
        self.feedback_event.set()
        self.feedback = mp.Process(target=feedback_process,args=(self.feedback_event,))
        print("Feedback started")
        self.processes.append(self.feedback)
        self.log_box.append("Feedback started at " + str(datetime.now()))
        self.feedback.start()
        
    def feedback_stop(self):
        try:
            self.feedback_event.clear()
            if (self.feedback is not None or self.feedback.is_alive()) and self.feedback in self.processes :
                print("Feedback stopped")
                self.feedback.terminate()  # Ensure the process is terminated
                self.feedback.join()  # Wait for the process to finish
                self.processes.remove(self.feedback)
                self.log_box.append("Feedback stopped at " + str(datetime.now()))
        except:
            pass

#################################################

    @pyqtSlot(str)
    def update_protocol(self, message):
        self.protocol_box.append(message)

    @pyqtSlot(str)
    def update_vision_log(self, message):
        self.vision_box.setText(message)
    
    @pyqtSlot(str)
    def update_network_log(self, message):
        self.network_box.setText(message)
    
    @pyqtSlot(str)
    def update_speech_log(self, message):
        self.speech_box.setText(message)
    
    @pyqtSlot(str)
    def update_feedback_log(self, message):
        self.feedback_box.setText(message)  


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
