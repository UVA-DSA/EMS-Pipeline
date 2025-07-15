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
from threading import Thread
###########################################pyqtSlot Signals
class GUI_signal(QObject):
    signal_protocol = pyqtSignal(str)
    signal_feedback = pyqtSignal(str)
    signal_speech = pyqtSignal(str)
    signal_vision = pyqtSignal(str)
    signal_network = pyqtSignal(str)



from time import sleep


###########################################################################################################################################################################################################
"'Process spawning"
def protocol_process(event,receive_queue, sending_queue, command_queue):
    while event.is_set() is True:
        if not receive_queue.empty():
            received_data = receive_queue.get_nowait()
            print(f"[PROTOCOL] [PROCESS] Received the following message: {received_data}")
            message = f"[PROTOCOL][MESSAGE] Protocol based on received data is sent at {datetime.now()}"
            try:
                sending_queue.put_nowait(message)
                print(message)
            except:
                print("[ERROR][PROTOCOL] Data could not be added to queue to send over to the Feedback Process")




def feedback_process(event,receiving_queue, sending_queue,command_queue):
    while event.is_set() is True:
        if not receiving_queue.empty():
            received_data = receiving_queue.get_nowait()
            print(f"[FEEDBACK] [PROCESS] Received the following message: {received_data}")
            message = f"[FEEDBACK][MESSAGE] Feedback based on received data is sent at {datetime.now()}"
            try:
                sending_queue.put_nowait(message)
                print(message)
            except:
                print("[ERROR][FEEDBACK] Data could not be added to queue to send over to the Network Process")
            


def speech_process(event,received_audio,sent_signal,command_queue):
    while event.is_set() is True:
        if not received_audio.empty():
            received_data = received_audio.get()
            print(f"[SPEECH][PROCESS] Received the following message from network: {received_data}")
            message = f"[SPEECH][MESSAGE] Audio signal received and sent at {datetime.now()}"
            try:
                sent_signal.put_nowait(message)
                print(message)
            except:
                print("[ERROR][SPEECH] Data could not be added to queue to send over to the Protocol Process")


def vision_process(event,video_from_network, frames_to_protocol,command_queue):
    while event.is_set() is True:
        if not video_from_network.empty():
            received_data = video_from_network.get()
            print(f"[VISION][PROCESS] Received the following message from network: {received_data}")
            message = f"[VISION][MESSAGE] Video frames received and sent at {datetime.now()}"
            try:
                frames_to_protocol.put_nowait(message)
                print(message)

            except:
                print("[ERROR][VISION] Data could not be added to queue to send over to the Protocol Process")


def audio_thread(send_audio,send_over_socket):
    while True:
        #to do: see if you can check the message type over the socket to only put that on there
        try:
            message = f"[AUDIO][MESSAGE] Audio Signal Received and Sent at {datetime.now()}"
            send_audio.put_nowait(message)
        except:
            print("[ERROR][AUDIO] Data could not be added to queue to send over to the Speech Process")
        print(message)
        sleep(0.5)
                
def video_thread(send_video,send_over_socket):
    while True:
        try:
            message = f"[VIDEO][MESSAGE] Video frames received and sent at {datetime.now()}"
            send_video.put_nowait(message)
        except:
            print("[ERROR][VIDEO] Data could not be added to queue to send over to the Vision Process")
        print(message)
        sleep(2)

def network_process(event, send_audio, send_video,commands,receiving_queue,response_sending_queue):
    thread_list = []
    ##AUDIO THREAD STUFF
    print("[THREAD FOR AUDIO]")
    thread_for_audio = Thread(target=audio_thread,args=(send_audio,response_sending_queue,))
    thread_list.append(thread_for_audio)
    thread_for_audio.daemon = True #automatically ends the thread when main is killed
    print("Thread for audio is created")

    thread_for_audio.start()

    ##VIDEO THREAD STUFF
    print("[Thread for Video] ")
    thread_for_video = Thread(target=video_thread,args=(send_video,response_sending_queue,))
    thread_list.append(thread_for_video)
    thread_for_video.daemon = True
    print("Thread for Video is Created")
    thread_for_video.start()

    while event.is_set() is True:
        # print(f"[NETWORK][PROCESS] Size of AUdio queue: {send_audio.qsize()}")
        # print(f"[NETWORK][PROCESS] Size of Video queue: {send_video.qsize()} ")
        # sleep(3)
        if not receiving_queue.empty():
            received_data = receiving_queue.get()
            print(f"[NETWORK][PROCESS] Received the following message from feedback: {received_data}")
            message = f"[NETWORK][MESSAGE] Video frames received and sent out {datetime.now()}"
            try:
                response_sending_queue.put_nowait(message)
                print(message)
            except:
                print("[ERROR][NETWORK] Data could not be sent over sockets")


######################

"'GUI/Window  class"
class gui_window(QWidget):
    def __init__(self,application,queue,width,height):
        super(gui_window,self).__init__()
        self.app = application
        self.command_queue = queue
        

        MAXSIZE = 512
        ##Each process has its own queue to check to ensure that data is not overwritten or lost while still meeting real time constraints:
        self.network_send_audio = Queue(maxsize=MAXSIZE)
        self.network_send_video = Queue(maxsize=MAXSIZE)
        self.speech_audio_send_protocol = Queue(maxsize=MAXSIZE)
        self.protocol_send_feedback = Queue(maxsize=MAXSIZE)
        self.feedback_send_network = Queue(maxsize=MAXSIZE)
        self.sendout_network = Queue(maxsize=MAXSIZE)
 


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
      
    #     #####timer stuff for queue checking
    #     self.timer = QtCore.QTimer(self)
    #     self.timer.setInterval(10) #delay to prevent it from causing undefined behavior
    #     self.timer.timeout.connect(self.continuous_queue_check)
    #     self.timer.start()


    # def continuous_queue_check(self):
    #         pass


###function for what happens when a button is clicked
    def start_protocol(self):
        self.protocol_button_start.setEnabled(False)
        self.protocol_button_stop.setEnabled(True)
        self.protocol_button_stop.setStyleSheet("color: white")
        self.protocol_button_start.setStyleSheet("color: black")
        self.protocol_event.set()
        self.protocol = mp.Process(target=protocol_process,args=(self.protocol_event,self.speech_audio_send_protocol,self.protocol_send_feedback,self.command_queue,))
        print("Protocol start")
        self.processes.append(self.protocol)
        self.log_box.append("Protocol started at " + str(datetime.now()))
        print(f"PROCESS: {self.protocol}")
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
        self.vision = mp.Process(target=vision_process,args=(self.vision_event,self.network_send_video,self.speech_audio_send_protocol,self.command_queue,))        
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
        self.network = mp.Process(target=network_process,args=(self.network_event,self.network_send_audio, self.network_send_video,self.command_queue,self.feedback_send_network,self.sendout_network,))
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
        self.speech = mp.Process(target=speech_process,args=(self.speech_event,self.network_send_audio,self.speech_audio_send_protocol,self.command_queue,))
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
        self.feedback = mp.Process(target=feedback_process,args=(self.feedback_event,self.protocol_send_feedback,self.feedback_send_network,self.command_queue))
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