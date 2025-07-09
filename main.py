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
from PyQt5.QtCore import pyqtSlot

from multiprocessing import Event
from multiprocessing.shared_memory import SharedMemory
from datetime import datetime
###########################################pyqtSlot Signals
class GUI_signal(QObject):
    signal_protocol = pyqtSignal(str)
    signal_feedback = pyqtSignal(str)
    signal_speech = pyqtSignal(str)
    signal_vision = pyqtSignal(str)
    signal_network = pyqtSignal(str)



from time import sleep


###########################################################################################################################################################################################################
  ### 2 forms of queues == (src, message) or (src,dst,status,reason) or for interprocess communication ==> (src,dst,status signal,message,pipeline number)

  ####when demoing how the processes work, add in additional lines of code
"'Process spawning"
def protocol_process(event,command_queue,process_queue):
    counter = 0
    while event.is_set() is True:
        if process_queue.empty() and counter == 1000000:
            message = "Protocol Process Running in the Background at " + str(datetime.now())
            print(message)
            command_queue.put(('protocol',message))
            counter = 0
        else:
            received_content = process_queue.get()
            command_queue.put(('protocol','feedback',received_content[0]))
        counter+=1

    
    

def feedback_process(event,command_queue,process_queue):
    counter = 0
    while event.is_set() is True:
        print(counter)
        if process_queue.empty() and counter == 1000000:
            message = "Feedback Process Running in the Background at " + str(datetime.now())
            print(message + str(counter))
            counter = 0
            command_queue.put(('feedback',message))
        else:
            received_content = process_queue.get()
            if received_content[0] == 'network':
                command_queue.put(('feedback','network',received_content[1]))
            else:
                data = f"Data at the end of the pipeline is {received_content[1]}"
                command_queue.put('feedback',data)
        counter+=1


def speech_process(event,command_queue,process_queue):
    counter = 0
    while event.is_set() is True:
        if process_queue.empty() and counter == 1000000:
            message = "Speech Process Running in the Background at " + str(datetime.now())
            print(message)
            command_queue.put(('speech',message))
            counter = 0
        else:
            received_content = process_queue.get()
            command_queue.put(('speech','protocol',received_content[0]))
        counter+=1


def vision_process(event,command_queue,process_queue):
    counter = 0
    while event.is_set() is True:
        if process_queue.empty() and counter == 1000000:
            message = "Vision Process Running in the Background at " + str(datetime.now())
            print(message)
            command_queue.put(('vision',message))
            counter = 0
        else:
            received_content = process_queue.get()
            command_queue.put(('vision','feedback',received_content[0]))
        counter+=1

def network_process(event,command_queue,process_queue):
    audio_or_video = 0
    counter = 0
    while event.is_set() is True:
        print(counter)
        if process_queue.empty() :
            message = "Network Process Running in the Background at " + str(datetime.now())
            print(message)
            command_queue.put(('network',message))
            if audio_or_video == 0:
                command_queue.put(('network','speech',counter))
                audio_or_video == 1
            else:
                command_queue.put(('network','vision',counter))
                audio_or_video = 0
            counter = 0
        else:
            received_content = process_queue.get()
            data = f"Data sent over the network is {received_content[0]}"
            command_queue.put(('network',data))
        counter+=1


######################

"'GUI/Window  class"
class gui_window(QWidget):
    def __init__(self,application,queue,width,height):
        super(gui_window,self).__init__()
        self.app = application
        self.command_queue = queue
        


        ##Each process has its own queue to check to ensure that data is not overwritten or lost while still meeting real time constraints:
        self.feedback_queue = Queue()
        self.network_queue = Queue()
        self.vision_queue = Queue()
        self.speech_queue = Queue()
        self.protocol_queue = Queue()


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
        
        self.exit_button = QPushButton('Exit',self)
        main_layout.addWidget(self.exit_button)



    #########################################Text box updates and set up
        self.protocol_box = QTextEdit()
        self.protocol_box.setReadOnly(True)
        self.protocol_box.setStyleSheet("background-color: #1E1E1E; color: white; font-size: 14px; font-family: Arial;")    
        self.protocol_box.setOverwriteMode(True)
        main_layout.addWidget(self.protocol_box)
        self.protocol_box.setText("Protocol Log:\n")


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
        self.exit_button.clicked.connect(self.shared_memory_cleanup)


        #setting up button clickability and color
        self.protocol_button_start.setEnabled(True)
        self.protocol_button_stop.setEnabled(False)
        self.protocol_button_stop.setStyleSheet("color: black")
        self.protocol_button_start.setStyleSheet("color: white")
        self.speech_start_button.setEnabled(True)
        self.speech_start_button.setStyleSheet("color: white")
        self.speech_stop_button.setStyleSheet("color: black")
        self.speech_stop_button.setEnabled(False)
        self.feedback_start_button.setEnabled(True)
        self.feedback_start_button.setStyleSheet("color: white")
        self.feedback_stopped_button.setStyleSheet("color: black")
        self.feedback_stopped_button.setEnabled(False)
        self.network_start_button.setEnabled(True)
        self.network_start_button.setStyleSheet("color: white")
        self.network_end_button.setStyleSheet("color: black")
        self.network_end_button.setEnabled(False)
        self.vision_start_button.setEnabled(True)
        self.vision_start_button.setStyleSheet("color: white")
        self.vision_stop_button.setStyleSheet("color: black")
        self.vision_stop_button.setEnabled(False)

        #signal set up
        self.gui_signal = GUI_signal()
        self.gui_signal.signal_protocol.connect(self.update_protocol)
        self.gui_signal.signal_feedback.connect(self.update_feedback_log)
        self.gui_signal.signal_network.connect(self.update_network_log)
        self.gui_signal.signal_speech.connect(self.update_speech_log)
        self.gui_signal.signal_vision.connect(self.update_vision_log)
      
        #####timer stuff for queue checking
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(10) #delay to prevent it from causing undefined behavior
        self.timer.timeout.connect(self.continuous_queue_check)
        self.timer.start()


    def continuous_queue_check(self):
            ### 2 forms of queues == (src, message) or (src,dst,status,reason) or (type of data sent,dst,contents of message)
            if not command_queue.empty():
                result = command_queue.get()
                if len(result) == 2:
                    src = result[0]
                    message = str(result[1])
                    match src:
                        case 'feedback':
                            self.gui_signal.signal_feedback.emit(message)
                        case 'protocol':
                            self.gui_signal.signal_protocol.emit(message)
                        case 'network':
                            self.gui_signal.signal_network.emit(message)
                        case 'vision':
                            self.gui_signal.signal_vision.emit(message)
                        case 'speech':
                            self.gui_signal.signal_speech.emit(message)
                        case _:
                            self.log_box.append("Source Process is Inputted Incorrectly")
                            print("Invalid Source Process")
                if len(result) == 3: ###one reason for not terminating processs after the thread finishes is due to the fact that there might still be data somewhere in the pipeline
                    src = result[0]
                    dst = result[1]
                    message_contents =  str(result[2])
                    time_stamp = src(datetime.now())
                    match dst:
                        case 'protocol':
                            if self.protocol not in self.processes:
                                self.start_protocol()
                            message = "Data Received from " + str(src) + " at: " + time_stamp
                            self.gui_signal.signal_protocol.emit(message)
                            data = message_contents
                            self.protocol_queue.put((data))

                        case 'feedback':
                            if self.feedback not in self.processes:
                                self.feedback_start()
                            message = "Data Received from " + str(src) + " at: " + time_stamp
                            data = message_contents
                            if src == 'protocol':
                                self.feedback_queue.put(('protocol',data))
                            if src == 'vision': 
                                self.feedback_queue.put(('vision',data))
                            self.gui_signal.signal_feedback.emit(message)

                        case 'network':
                            message = "Data Received from " + str(src) + " at: " + time_stamp
                            update = message_contents
                            self.network_queue.put((update))
                            self.gui_signal.signal_network.emit(message)

                        case 'speech': #signal sent from network to speech thread
                            if self.speech not in self.processes:
                                self.speech_start()
                            message = f"Audio Signal Received from {src} at: "+ time_stamp
                            signal = message_contents
                            self.speech_queue.put((signal))
                            self.gui_signal.signal_speech.emit(message)

                        case 'vision':
                            message = "Video Frames Received from " + str(src) + " at: " + time_stamp
                            frames = message_contents
                            if self.vision not in self.processes:
                                self.vision_start()
                            self.vision_queue.put((frames))
                            self.gui_signal.signal_vision.emit(message)

                        case _:
                            print("INVALID REQUEST")
                        
                        
                if len(result) == 4: 
                    src = result[0] #which process is initiating things
                    dst = result[1] #where the process 
                    status = str(result[2])
                    temp_message = result[3]
                    message = "Process " + str(src) + " due to " + str(temp_message) + "resulting in " + str(status)                     
                    match dst:
                        case 'protocol':
                            self.gui_signal.signal_protocol.emit(message)   
                            if status == 'stop':
                                self.stop_protocol()
                            if status == 'start':
                                self.start_protocol
                        case 'feedback':
                            self.gui_signal.signal_feedback.emit(message)   
                            if status == 'stop':
                                self.feedback_stop()
                            if status == 'start':
                                self.feedback_start()
                        case 'network':
                            self.gui_signal.signal_network.emit(message)
                            if status == 'start':
                                self.network_start()
                            if status == 'stop':
                                self.network_start()
                        case 'vision':
                            self.gui_signal.signal_vision.emit(message)
                            if status == 'start':
                                self.vision_start()
                            if status == 'stop':
                                self.vision_stop()
                        case 'speech':
                            self.gui_signal.signal_speech.emit(message)
                            if status == 'start':
                                self.speech_start()
                            if status == 'stop':
                                self.speech_stop()                   
                        case _:
                            print("Invalid Destination")
                            pass

                    


###function for what happens when a button is clicked
    def start_protocol(self):
        self.protocol_button_start.setEnabled(False)
        self.protocol_button_stop.setEnabled(True)
        self.protocol_button_stop.setStyleSheet("color: white")
        self.protocol_button_start.setStyleSheet("color: black")
        self.protocol_event.set()
        self.protocol = mp.Process(target=protocol_process,args=(self.protocol_event,self.command_queue,self.protocol_queue,))
        print("Protocol start")
        self.processes.append(self.protocol)
        self.log_box.append("Protocol started at " + str(datetime.now()))
        self.protocol.start()

    def stop_protocol(self): 
        try:
            self.protocol_event.clear()
            if self.protocol is not None or self.protocol in self.processes:
                self.protocol_button_start.setEnabled(True)
                self.protocol_button_stop.setEnabled(False)
                self.protocol_button_stop.setStyleSheet("color: black")
                self.protocol_button_start.setStyleSheet("color: white")
                self.protocol.terminate()  # Ensure the process is terminated
                self.protocol.join()  # Wait for the process to finish
                self.processes.remove(self.protocol)
                self.log_box.append("Protocol stopped at " + str(datetime.now()))
                print("Protocol stopped")
        except:
            pass

    def vision_start(self):
        self.vision_event.set()
        self.vision_start_button.setEnabled(False)
        self.vision_stop_button.setEnabled(True)
        self.vision_start_button.setStyleSheet("color: black")
        self.vision_stop_button.setStyleSheet("color: white")
        self.vision = mp.Process(target=vision_process,args=(self.vision_event,self.command_queue,self.vision_queue))
        print("Vision start")
        self.processes.append(self.vision)
        self.log_box.append("Vision started at " + str(datetime.now()))
        self.vision.start()

    def vision_stop(self):
        try: #used to prevent premature exiting
            self.vision_event.clear()
            if self.vision is not None and self.vision in self.processes:# Handles case where the vision process might not be active or a process that did not proper get cleaned up 
                self.vision_start_button.setEnabled(True)
                self.vision_stop_button.setEnabled(False)
                self.vision_start_button.setStyleSheet("color: white")
                self.vision_stop_button.setStyleSheet("color: black")
                print("Vision stopped")
                self.vision.terminate()  # Ensure the process is terminated
                self.vision.join()  # Wait for the process to finish
                self.processes.remove(self.vision)
                self.log_box.append("Vision stopped at " + str(datetime.now()))
        except:
            pass

    def network_start(self):
        self.network_event.set()
        self.network_start_button.setEnabled(False)
        self.network_end_button.setEnabled(True)
        self.network_start_button.setStyleSheet("color: black")
        self.network_end_button.setStyleSheet("color: white")
        self.network = mp.Process(target=network_process,args=(self.network_event,self.command_queue,self.network_queue,))
        print("Network started")
        self.processes.append(self.network)
        self.log_box.append("Network started at " + str(datetime.now()))
        self.network.start()

    def network_stop(self):
        try:
            self.network_event.clear()
            if self.network is not None and self.network in self.processes:
                self.network_start_button.setEnabled(True)
                self.network_end_button.setEnabled(False)
                self.network_start_button.setStyleSheet("color: white")
                self.network_end_button.setStyleSheet("color: black")
                print("Network stopped")
                self.network.terminate()  # Ensure the process is terminated
                self.network.join()  # Wait for the process to finish
                self.processes.remove(self.network)
                self.log_box.append("Network stopped at " + str(datetime.now()))
        except:
            pass

    def speech_start(self):
        self.speech_event.set()
        self.speech_start_button.setEnabled(False)
        self.speech_stop_button.setEnabled(True)
        self.speech_start_button.setStyleSheet("color: black")
        self.speech_stop_button.setStyleSheet("color: white")
        self.speech = mp.Process(target=speech_process,args=(self.speech_event,self.command_queue,self.speech_queue,))
        print("Speech started")
        self.processes.append(self.speech)
        self.log_box.append("Speech started at " + str(datetime.now()))
        self.speech.start()

    def speech_stop(self):
        try:
            self.speech_event.clear()
            if (self.speech is not None) and self.speech in self.processes:
                self.speech_start_button.setEnabled(True)
                self.speech_stop_button.setEnabled(False)
                self.speech_start_button.setStyleSheet("color: white")
                self.speech_stop_button.setStyleSheet("color: black")
                print("Speech stopped")
                self.speech.terminate()  # Ensure the process is terminated
                self.speech.join()  # Wait for the process to finish
                self.processes.remove(self.speech)
                self.log_box.append("Speech stopped at " + str(datetime.now()))
        except:
            pass

    def feedback_start(self):
        self.feedback_event.set()
        self.feedback_start_button.setEnabled(False)
        self.feedback_stopped_button.setEnabled(True)
        self.feedback_start_button.setStyleSheet("color: black")
        self.feedback_stopped_button.setStyleSheet("color: white")
        self.feedback = mp.Process(target=feedback_process,args=(self.feedback_event,self.command_queue,self.feedback_queue,))
        print("Feedback started")
        self.processes.append(self.feedback)
        self.log_box.append("Feedback started at " + str(datetime.now()))
        self.feedback.start()
        
    def feedback_stop(self):
        try:
            self.feedback_event.clear()
            if (self.feedback is not None or self.feedback.is_alive()) and self.feedback in self.processes :
                print("Feedback stopped")
                self.feedback_start_button.setEnabled(True)
                self.feedback_stopped_button.setEnabled(False)
                self.feedback_start_button.setStyleSheet("color: white")
                self.feedback_stopped_button.setStyleSheet("color: black")
                self.feedback.terminate()  # Ensure the process is terminated
                self.feedback.join()  # Wait for the process to finish
                self.processes.remove(self.feedback)
                self.log_box.append("Feedback stopped at " + str(datetime.now()))
        except:
            pass
####################################################3
#Shared Memory Specific Processes:
    def shared_memory_cleanup(self):
        if len(self.processes) > 0:
            for i in range(len(self.processes)):
                self.processes[i].terminate()
                self.processes[i].join()
        print("Program closed")
        self.close()



#################################################

    @pyqtSlot(str)
    def update_protocol(self, message):
        self.protocol_box.append(message)

    @pyqtSlot(str)
    def update_vision_log(self, message):
        self.vision_box.append(message)
    
    @pyqtSlot(str)
    def update_network_log(self, message):
        self.network_box.append(message)
    
    @pyqtSlot(str)
    def update_speech_log(self, message):
        self.speech_box.append(message)
    
    @pyqtSlot(str)
    def update_feedback_log(self, message):
        self.feedback_box.append(message)



"'Code for Multiprocessing set up"
#import py_trees 
#from py_trees.blackboard import Blackboard


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
