import multiprocessing as mp
from multiprocessing import Process, Queue


def spawn_vision():
    return
def spawn_network():
    return




############################################################################################################################################################################################################
"' Code for the GUI"

from PyQt5.QtCore import *
from PyQt.QtWidgets import *
from PyQt5.QtGUI import *
from PyQt5 import QTCore
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtMultimedia import *







"'GUI/Window  class"
class gui_window(QWidget):
    def __init__(self,application):
        super(gui_window,self).init()
        self.app = application
        
        
        
        
        ##setting up window
        self.screen_resolution = app.desktop().screenGeometry ##Todo: test to make sure this actually works
        self.width = screen_resoluton.width()
        self.height = screen_resoluton.height()
        self.setWindowTitle('CognitiveEMS Debugging Demo')



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
        print("Protocol start")

    def stop_protocol(self):
        print("Protocol stopped")

    def vision_start(self):
        print("Vision start")

    def vision_stop(self):
        print("Vision stopped")

    def network_start(self):
        print("Network started")

    def network_stop(self):
        print("Network stopped")

    def speech_start(self):
        print("Speech started")

    def speech_stop(self):
        print("Speech stopped")

    def feedback_start(self):
        print("Feedback started")

    def feedback_stop(self):
        print("Feedback stopped")
#################################################




############################################################################################################################################################################################################



"'Code for Multiprocessing set up"
import py_trees 
from py_trees.blackboard import Blackboard




if __name__ == "__main__":
    
    print("Starting GUI!")
    application = QApplication(sys.argv)
    Window = gui_window(application=application)
    
    
    ##Debug information
    print(f"width: {Window.width}")
    print(f"height: {Window.height}")
    ##debug information



    Window.show()
    #command queue initilaization
    command_queue  = Queue()
    #creating processes
    vision_process = Process(target=spawn_vision)
    network_process = Process(target=spawn_network)
    


    ##starting processes
    vision_process.start()



    #waiting on processes to finish
    vision_process.join()
