#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============== Imports ==============

from __future__ import absolute_import, division, print_function
from six.moves import queue

import sys
import os
import time
import math
import datetime
import csv
import sys
import os
import time, queue
import subprocess

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QLineEdit
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtCore
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QGuiApplication
from PyQt5.QtCore import QCoreApplication, Qt,QBasicTimer, QTimer,QPoint,QSize
import PyQt5.QtWidgets,PyQt5.QtCore

import py_trees
from py_trees.blackboard import Blackboard

from behaviours_m import *
from DSP.amplitude import Amplitude
from EMSAgent import EMSAgentSystem
import GoogleSpeechMicStream
import GoogleSpeechFileStream
import DeepSpeechMicStream
import DeepSpeechFileStream
import WhisperFileStream
# import WhisperMicStream
import whisper_config

import TextSpeechStream
import CognitiveSystem
import os


import Feedback

#import DeepSpeechMicStream
#import DeepSpeechFileStream
# import WavVecMicStream
# import WavVecFileStream


from StoppableThread.StoppableThread import StoppableThread

from video_streaming import Thread
from smartwatch_streaming import Thread_Watch

chunkdata = []
ConceptDict = dict()

datacollection = False
videostream = False
audiostream = False
smartwatchStream = False
conceptExtractionStream = False
protocolStream = False
interventionStream = False
transcriptStream = False

curr_date = datetime.datetime.now()
dt_string = curr_date.strftime("%d-%m-%Y-%H-%M-%S")

data_path = "./data_collection_folder/"+dt_string+"/"

# ================================================================== GUI ==================================================================


# Main Window of the Application

class MainWindow(QWidget):

    def __init__(self, width, height):
        super(MainWindow, self).__init__()

        # Fields
        self.width = width
        self.height = height
        self.SpeechThread = None
        self.CognitiveSystemThread = None
        self.EMSAgentThread = None
        self.FeedbackThread = None
        self.WhisperSubprocess = None
        self.stopped = 0
        self.reset = 0
        self.maximal = Amplitude()
        self.finalSpeechSegmentsSpeech = []
        self.finalSpeechSegmentsNLP = []
        self.nonFinalText = ""

        # Set geometry of the window
        self.setWindowTitle('CognitiveEMS Demo')
        self.setWindowIcon(QIcon('./Images/Logos/UVA.png'))
        self.setGeometry(int(width * .065), int(height * .070),
                         int(width * .9), int(height * .9))
        
        # Grid Layout to hold all the widgets
        self.Grid_Layout = QGridLayout(self)

        # Add disabled buttons to the of GUI to ensure spacing
        R3 = QPushButton(".\n.\n")
        R3.setFont(QFont("Monospace"))
        R3.setEnabled(False)
        R3.setStyleSheet("background-color:transparent;border:0;")

        R6 = QPushButton(".\n.\n")
        R6.setFont(QFont("Monospace"))
        R6.setEnabled(False)
        R6.setStyleSheet("background-color:transparent;border:0;")

        R8 = QPushButton(".\n" * int(self.height/100))
        R8.setFont(QFont("Monospace"))
        R8.setEnabled(False)
        R8.setStyleSheet("background-color:transparent;border:0;")

        C0 = QPushButton("................................")
        C0.setFont(QFont("Monospace"))
        C0.setEnabled(False)
        self.Grid_Layout.addWidget(C0, 13, 0, 1, 1)

        C1 = QPushButton("................................")
        C1.setFont(QFont("Monospace"))
        C1.setEnabled(False)
        self.Grid_Layout.addWidget(C1, 13, 1, 1, 1)

        C2 = QPushButton(
            "....................................................................")
        C2.setFont(QFont("Monospace"))
        C2.setEnabled(False)
        self.Grid_Layout.addWidget(C2, 13, 2, 1, 2)

        # Create main title
        self.MainLabel = QLabel(self)
        self.MainLabel.setText('<font size="6"><b>CognitiveEMS</b></font>')
        self.Grid_Layout.addWidget(self.MainLabel, 0, 0, 1, 1)

        # Data Panel: To hold save state and generate form buttons
        self.DataPanel = QWidget()
        self.DataPanelGridLayout = QGridLayout(self.DataPanel)
        self.Grid_Layout.addWidget(self.DataPanel, 0, 2, 1, 2)

        # Create a save state button in the panel
        self.SaveButton = QPushButton('Save State', self)
        self.SaveButton.clicked.connect(self.SaveButtonClick)
        self.DataPanelGridLayout.addWidget(self.SaveButton, 0, 0, 1, 1)

        # Create a generate form button in the panel
        self.GenerateFormButton = QPushButton('Generate Form', self)
        self.DataPanelGridLayout.addWidget(self.GenerateFormButton, 0, 1, 1, 1)

        # Create label and textbox for speech
        self.SpeechLabel = QLabel()
        self.SpeechLabel.setText("<b>Speech Recognition</b>")
        self.Grid_Layout.addWidget(self.SpeechLabel, 1, 0, 1, 1)

        self.SpeechSubLabel = QLabel()
        self.SpeechSubLabel.setText("(Transcript)")
        self.Grid_Layout.addWidget(self.SpeechSubLabel, 2, 0, 1, 1)

        self.SpeechBox = QTextEdit()
        self.SpeechBox.setReadOnly(True)
        self.SpeechBox.setOverwriteMode(True)
        self.SpeechBox.ensureCursorVisible()
        QTextCursor(self.SpeechBox.textCursor()).insertBlock()
        self.Grid_Layout.addWidget(self.SpeechBox, 3, 0, 2, 1)

        #Create label and media player for videos- - added 3/21/2022
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.video = QVideoWidget()

        self.Box = QVBoxLayout()
        self.Grid_Layout.addLayout(self.Box, 8, 1, 1, 1)

        # Create label and textbox for Vision Information
        self.VisionInformationLabel = QLabel()
        self.VisionInformationLabel.setText("<b>Vision Information</b>")
        self.Box.addWidget(self.VisionInformationLabel) #7,1,1,1

        self.VisionInformation = QTextEdit() #QLineEdit()
        self.VisionInformation.setReadOnly(True)
        self.Box.addWidget(self.VisionInformation) #8,1,1,1

        # Create label and textbox for Smartwatch
        self.Box2 = QVBoxLayout()
        self.Grid_Layout.addLayout(self.Box2, 6, 1, 2, 1)

        self.SmartwatchLabel = QLabel()
        self.SmartwatchLabel.setText("<b>Smartwatch Activity</b>")
        self.Box2.addWidget(self.SmartwatchLabel) #5,1,1,1

        self.Smartwatch = QTextEdit() #QLineEdit()
        self.Smartwatch.setReadOnly(True)
        self.Smartwatch.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.Box2.addWidget(self.Smartwatch) # 6, 1, 1, 1
 
        self.VideoSubLabel = QLabel()
        self.VideoSubLabel.setText("<b>Video Content<b>") #setGeometry(100,100,100,100)
        self.Grid_Layout.addWidget(self.VideoSubLabel, 5, 0, 1, 1)

        self.video = QLabel(self)

        self.Grid_Layout.addWidget(self.video, 6, 0, 3, 1)

        VIDEO_WIDTH = 640
        VIDEO_HEIGHT = 480
        self.video.setGeometry(QtCore.QRect(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT))

        # Threads for video 
        th = Thread(data_path, videostream)
        th.changePixmap.connect(self.setImage)
        th.changeVisInfo.connect(self.handle_message2)
        th.start()  #Disabled for now

        # Threads for smartwatch
        th2 = Thread_Watch(data_path, smartwatchStream)
        th2.changeActivityRec.connect(self.handle_message)
        th2.start()

        # th2 = ThreadAudio(self)
        # th2.start()

        # Control Panel: To hold combo box, radio, start, stop, and reset buttons
        self.ControlPanel = QWidget()
        self.ControlPanelGridLayout = QGridLayout(self.ControlPanel)
        self.Grid_Layout.addWidget(self.ControlPanel, 9, 0, 1, 2)

        # Audio Options Menu
        self.ComboBox = QComboBox()
        self.ComboBox.addItems(["Microphone",
                                "000_190105",
                                "001_190105",
                                "002_190105",
                                "003_190105",
                                "004_190105",
                                "005_190105",
                                "006_190105",
                                "007_190105",
                                "008_190105",
                                "009_190105",
                                "010_190105",
                                "011_190105",
                                "CPR_transcript1",
                                "CPR_transcript2",
                                "CPR_transcript3",
                                "CPR_Kay",
                                "NG1",
                                "Other Audio File",
                                "Text File"])

        self.ControlPanelGridLayout.addWidget(self.ComboBox, 0, 0, 1, 1)

        # Radio Buttons Google or Other ML Model
        self.GoogleSpeechRadioButton = QRadioButton("Google Speech API", self)
        self.GoogleSpeechRadioButton.setEnabled(True)
        self.GoogleSpeechRadioButton.setChecked(True)
        self.ControlPanelGridLayout.addWidget(
        self.GoogleSpeechRadioButton, 0, 1, 1, 1)


        self.WhisperRadioButton = QRadioButton("OpenAI Whisper", self)
        self.WhisperRadioButton.setEnabled(True)  # Set the initial state
        self.WhisperRadioButton.setChecked(False)  # Set the initial checked state
        self.ControlPanelGridLayout.addWidget(self.WhisperRadioButton, 0, 3, 1, 1)  # Adjust column index as needed

        self.MLSpeechRadioButton = QRadioButton("Wav2Vec2", self)
        self.MLSpeechRadioButton.setEnabled(True) #changed from False to True to enable
        self.MLSpeechRadioButton.setChecked(False)
        self.ControlPanelGridLayout.addWidget(
        self.MLSpeechRadioButton, 0, 2, 1, 1)


        # Create a start button in the Control Panel
        self.StartButton = QPushButton('Start', self)
        self.StartButton.clicked.connect(self.StartButtonClick)
        self.ControlPanelGridLayout.addWidget(self.StartButton, 1, 0, 1, 1)

        # Create a stop button in the Control Panel
        self.StopButton = QPushButton('Stop', self)
        self.StopButton.clicked.connect(self.StopButtonClick)
        self.ControlPanelGridLayout.addWidget(self.StopButton, 1, 1, 1, 1)

        # Create a reset  button in the Control Panel
        self.ResetButton = QPushButton('Reset', self)
        self.ResetButton.clicked.connect(self.ResetButtonClick)
        self.ControlPanelGridLayout.addWidget(self.ResetButton, 1, 2, 1, 1)

        # VU Meter Panel
        self.VUMeterPanel = QWidget()
        self.VUMeterPanelGridLayout = QGridLayout(self.VUMeterPanel)
        self.Grid_Layout.addWidget(self.VUMeterPanel, 10, 0, 1, 2)

        # Add Microphone Logo to the VU Meter Panel
        self.MicPictureBox = QLabel(self)
        self.MicPictureBox.setPixmap(QPixmap('./Images/Logos/mic2.png'))
        self.VUMeterPanelGridLayout.addWidget(self.MicPictureBox, 0, 0, 1, 1)

        # Add the VU Meter progress bar to VU Meter Panel
        self.VUMeter = QProgressBar()
        self.VUMeter.setMaximum(100)
        self.VUMeter.setValue(0)
        self.VUMeter.setTextVisible(False)
        self.VUMeterPanelGridLayout.addWidget(self.VUMeter, 0, 1, 1, 1)

        # Create label and textbox for Concept Extraction
        self.ConceptExtractionLabel = QLabel()
        self.ConceptExtractionLabel.setText("<b>Concept Extraction</b>")
        self.Grid_Layout.addWidget(self.ConceptExtractionLabel, 1, 1, 1, 1)

        self.ConceptExtractionSubLabel = QLabel()
        self.ConceptExtractionSubLabel.setText(
            "(Concept, Presence, Value, Confidence)")
        self.Grid_Layout.addWidget(self.ConceptExtractionSubLabel, 2, 1, 1, 1)

        self.ConceptExtraction = QTextEdit()
        self.ConceptExtraction.setReadOnly(True)
        self.ConceptExtraction.setOverwriteMode(True)
        self.Grid_Layout.addWidget(self.ConceptExtraction, 3, 1, 2, 1)

        # Add label, textbox for protcol name
        self.ProtcolLabel = QLabel()
        self.ProtcolLabel.setText("<b>Suggested EMS Protocols</b>")
        self.Grid_Layout.addWidget(self.ProtcolLabel, 1, 2, 1, 2)

        self.ProtcolSubLabel = QLabel()
        self.ProtcolSubLabel.setText("(Protocol, Confidence)")
        self.Grid_Layout.addWidget(self.ProtcolSubLabel, 2, 2, 1, 2)

        self.ProtocolBox = QTextEdit()
        self.ProtocolBox.setReadOnly(True)
        self.Grid_Layout.addWidget(self.ProtocolBox, 3, 2, 1, 2)

        # Create label and textbox for interventions
        self.InterventionLabel = QLabel()
        self.InterventionLabel.setText("<b>Suggested Interventions</b>")
        self.Grid_Layout.addWidget(self.InterventionLabel, 5, 2, 1, 2)

        self.InterventionSubLabel = QLabel()
        self.InterventionSubLabel.setText("(Intervention, Confidence)")
        self.Grid_Layout.addWidget(self.InterventionSubLabel, 6, 2, 1, 2)

        self.InterventionBox = QTextEdit()
        self.InterventionBox.setReadOnly(True)
        self.Grid_Layout.addWidget(self.InterventionBox, 7, 2, 1, 2)

        self.Box3 = QVBoxLayout()
        self.Grid_Layout.addLayout(self.Box3, 8, 2, 1, 1)

        # Create label and textbox for messages
        self.MsgBoxLabel = QLabel()
        self.MsgBoxLabel.setText("<b>System Messages Log</b>")
        self.Box3.addWidget(self.MsgBoxLabel)#, 7, 2, 1, 2)

        self.MsgBox = QTextEdit()
        self.MsgBox.setReadOnly(True)
        self.MsgBox.setFont(QFont("Monospace"))
        self.Box3.addWidget(self.MsgBox)#, 8, 2, 1, 2)

        # Populate the Message Box with welcome message
        System_Info_Text_File = open("./ETC/System_Info.txt", "r")
        System_Info_Text = ""
        for line in System_Info_Text_File.readlines():
            System_Info_Text += line
        System_Info_Text_File.close()
        self.MsgBox.setText(System_Info_Text + "\n" + str(datetime.datetime.now().strftime("%c")) + " - Ready to start speech recognition!")
        self.MsgBox.setText(System_Info_Text)
        self.UpdateMsgBox(["Ready to start speech recognition!"])

        # Add Link Lab Logo
        self.PictureBox = QLabel()
        self.PictureBox.setPixmap(QPixmap('./Images/Logos/LinkLabLogo.png'))
        self.PictureBox.setAlignment(Qt.AlignCenter)
        self.Grid_Layout.addWidget(self.PictureBox, 10, 2, 1, 2)

        vspacer = PyQt5.QtWidgets.QSpacerItem(
            PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)
        self.Grid_Layout.addItem(vspacer, 11, 0, 1, -1)

        hspacer = PyQt5.QtWidgets.QSpacerItem(
            PyQt5.QtWidgets.QSizePolicy.Expanding, PyQt5.QtWidgets.QSizePolicy.Minimum)
        self.Grid_Layout.addItem(hspacer, 0, 2, -1, 1)

    # ================================================================== GUI Functions ==================================================================
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.video.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(str)
    def handle_message(self, message):
        self.Smartwatch.setText(message)

    @pyqtSlot(str)
    def handle_message2(self, message):
        self.VisionInformation.setText(message)

    #video playing -- added 3/21/2022
    def play(self):
        print("hello")
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        if self.mediaPlayer.state() == QMediaPlayer.StoppedState:
            print("video has stopped playing!")
            self.VisionInformation.setPlainText("CPR Done\nAverage Compression Rate: 140 bpm")

        else:
            print("testing successful")
            self.mediaPlayer.play()


    def mediaStateChanged(self, state):
    	if self.player.state() == QMediaPlayer.StoppedState:
            print("video has stopped playing!")
            self.VisionInformation.setPlainText("CPR Done\nAverage Compression Rate: 140 bpm")

    # Called when closing the GUI
    def closeEvent(self, event):
        print('Closing GUI')
        # self.th2.exit()
        self.stopped = 1
        self.reset = 1
        SpeechToNLPQueue.put('Kill')
        EMSAgentSpeechToNLPQueue.put('Kill')
        FeedbackQueue.put('Kill')
        if self.WhisperSubprocess != None: self.WhisperSubprocess.kill()
        event.accept()

    @pyqtSlot()
    def SaveButtonClick(self):
        #name = QFileDialog.getSaveFileName(self, 'Save File')
        name = str(datetime.datetime.now().strftime("%c")) + ".txt"
        #file = open("./Dumps/" + name,'w')
        mode_text = self.ComboBox.currentText()
        speech_text = str(self.SpeechBox.toPlainText())
        concept_extraction_text = str(self.ConceptExtraction.toPlainText())
        protocol_text = str(self.ProtocolBox.toPlainText())
        intervention_text = str(self.InterventionBox.toPlainText())
        msg_text = str(self.MsgBox.toPlainText())
        #text = "Mode:\n\n" + mode_text + "\n\nSpeech: \n\n" + speech_text +"\n\nConcept Extraction:\n\n" + concept_extraction_text + "\n\nProtocol Text:\n\n" + protocol_text + "\n\nIntervention:\n\n" + intervention_text + "\n\nSystem Messages Log:\n\n" + msg_text
        # file.write(text)
        # file.close()
        #self.UpdateMsgBox(["System state dumped to \n/Dumps/" + name])
        self.UpdateMsgBox(["System results saved to results.csv"])
        results = [speech_text, concept_extraction_text,
                   protocol_text, intervention_text]
        with open("results.csv", mode="a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(results)

    @pyqtSlot()
    def StartButtonClick(self):
        print('Start pressed!')
        self.UpdateMsgBox(["Starting!"])
        self.reset = 0
        self.stopped = 0
        self.StartButton.setEnabled(False)
        self.StopButton.setEnabled(True)
        self.ComboBox.setEnabled(False)
        self.ResetButton.setEnabled(False)

        # ====== Start Whisper 
        if (self.WhisperRadioButton.isChecked()):
            whispercommand = [
                "./stream",
                "-m",                                              # use specific whisper model
                f"models/ggml-{whisper_config.model_size}.bin",    # the whisper model to be used
                "--threads",                                       # use certain number of threads
                str(whisper_config.num_threads),                   # number of threads to use
                "--step",                          
                str(whisper_config.step),
                "--length",
                str(whisper_config.length),
            ]
            # If a Hard-coded Audio test file, use virtual mic to capture the recording
            if(self.ComboBox.currentText() not in {'Microphone', 'Other Audio File', 'Text File'}):
                whispercommand.append("--capture")
            # Start subprocess
            self.WhisperSubprocess = subprocess.Popen(whispercommand, cwd='whisper.cpp/')

        # ==== Start the Speech/Text Thread
        # If Microphone
        if(self.ComboBox.currentText() == 'Microphone'):
            if(self.GoogleSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(
                    target=GoogleSpeechMicStream.GoogleSpeech, args=(self, SpeechToNLPQueue,EMSAgentSpeechToNLPQueue, data_path, audiostream, transcriptStream,))
            
            elif(self.WhisperRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(
                    target=WhisperMicStream.Whisper, args=(self, SpeechToNLPQueue, EMSAgentSpeechToNLPQueue))

            # elif(self.DeepSpeechRadioButton.isChecked()):
            #     self.SpeechThread = StoppableThread(
            #         target=DeepSpeechMicStream.DeepSpeech, args=(self, SpeechToNLPQueue,))
            
            
            elif(self.MLSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(
                    target=WavVecMicStream.WavVec, args=(self, SpeechToNLPQueue,))

            self.SpeechThread.start()
            print('Microphone Speech Thread Started')

        # If Other Audio File
        elif(self.ComboBox.currentText() == 'Other Audio File'):
            audio_fname = QFileDialog.getOpenFileName(
                self, 'Open file', 'c:\\', "Wav files (*.wav)")
            if(self.GoogleSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(self, SpeechToNLPQueue, EMSAgentSpeechToNLPQueue, str(audio_fname), data_path, audiostream, transcriptStream,))

            
            elif(self.WhisperRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(
                    target=WhisperFileStream.Whisper, args=(self, SpeechToNLPQueue, EMSAgentSpeechToNLPQueue, str(audio_fname),))

            elif(self.DeepSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(target=DeepSpeechFileStream.DeepSpeech, args=(
                    self, SpeechToNLPQueue, str(audio_fname),))

                    
            elif(self.MLSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(target=WavVecFileStream.WavVec, args=(self, SpeechToNLPQueue, str(audio_fname),)) #(target=DeepSpeechFileStream.DeepSpeech, args=(self, SpeechToNLPQueue, str(audio_fname),))

            self.otheraudiofilename = str(audio_fname)
            self.SpeechThread.start()
            print("Other Audio File Speech Thread Started")

        # If Text File
        elif(self.ComboBox.currentText() == 'Text File'):
            text_fname = QFileDialog.getOpenFileName(
                self, 'Open file', 'c:\\', "Text files (*.txt)")
            self.SpeechThread = StoppableThread(target=TextSpeechStream.TextSpeech, args=(
                self, SpeechToNLPQueue, str(text_fname),))
            self.SpeechThread.start()
            print("Text File Speech Thread Started")

        # If a Hard-coded Audio test file
        else:
            if(self.GoogleSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(
                    self, SpeechToNLPQueue, EMSAgentSpeechToNLPQueue,'./Audio_Scenarios/2019_Test/' + str(self.ComboBox.currentText()) + '.wav', data_path, audiostream, transcriptStream,))

            elif (self.WhisperRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(
                    target=WhisperFileStream.Whisper, args=(self, SpeechToNLPQueue, EMSAgentSpeechToNLPQueue, './Audio_Scenarios/2019_Test/' + str(self.ComboBox.currentText()) + '.wav',))

            elif(self.DeepSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(target=DeepSpeechFileStream.DeepSpeech, args=(
                    self, SpeechToNLPQueue, './Audio_Scenarios/2019_Test/' + str(self.ComboBox.currentText()) + '.wav',))

            elif(self.MLSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(target=WavVecFileStream.WavVec, args=(
                    self, SpeechToNLPQueue, './Audio_Scenarios/2019_Test/' + str(self.ComboBox.currentText()) + '.wav',)) #(target=DeepSpeechFileStream.DeepSpeech, args=( self, SpeechToNLPQueue, './Audio_Scenarios/2019_Test/' + str(self.ComboBox.currentText()) + '.wav',))

            self.SpeechThread.start()
            print("Hard-coded Audio File Speech Thread Started")


        # ==== Start the Cognitive System Thread
        if(self.CognitiveSystemThread == None):
            print("Cognitive System Thread Started")
            self.CognitiveSystemThread = StoppableThread(
                target=CognitiveSystem.CognitiveSystem, args=(self, SpeechToNLPQueue, FeedbackQueue, data_path, conceptExtractionStream, interventionStream,))
            self.CognitiveSystemThread.start()


        # ==== Start the EMS Agent - Xueren ==== #
        if(self.EMSAgentThread == None):
            print("EMSAgent Thread Started")
            self.EMSAgentThread = StoppableThread(
                target=EMSAgentSystem.EMSAgentSystem, args=(self, EMSAgentSpeechToNLPQueue, FeedbackQueue, data_path, protocolStream))
            self.EMSAgentThread.start()

         # ==== Start the Feedback Thread ==== #
        if(self.FeedbackThread == None):
            print("Feedback Thread Started")
            self.FeedbackThread = StoppableThread(
                target=Feedback.Feedback, args=(self, data_path, FeedbackQueue))
            self.FeedbackThread.start()

    @pyqtSlot()
    def StopButtonClick(self):
        print("Stop pressed!")
        self.UpdateMsgBox(["Stopping!"])
        self.stopped = 1
        time.sleep(.1)
        self.StartButton.setEnabled(True)
        self.ComboBox.setEnabled(True)
        self.ResetButton.setEnabled(True)

    @pyqtSlot()
    def GenerateFormButtonClick(self):
        print('Generate Form pressed!')
        self.UpdateMsgBox(["Form Being Generated!"])
        text = str(self.SpeechBox.toPlainText())
        #StoppableThread(target = GenerateForm.GenerateForm, args=(self, text,)).start()

    @pyqtSlot()
    def ResetButtonClick(self):
        print('Reset pressed!')
        self.UpdateMsgBox(["Resetting!"])
        self.reset = 1
        self.stopped = 1

        if(self.CognitiveSystemThread != None):
            SpeechToNLPQueue.put('Kill')
        #     EMSAgentSpeechToNLPQueue.put('Kill')
        #     FeedbackQueue.put('Kill')
        # SpeechToNLPQueue.put('Kill')
        EMSAgentSpeechToNLPQueue.put('Kill')
        FeedbackQueue.put('Kill')
        self.VUMeter.setValue(0)
        self.finalSpeechSegmentsSpeech = []
        self.finalSpeechSegmentsNLP = []
        self.SpeechBox.clear()
        self.ConceptExtraction.setText('')
        self.ProtocolBox.setText('')
        self.InterventionBox.setText('')
        self.nonFinalText = ""
        self.SpeechThread = None
        self.CognitiveSystemThread = None
        self.EMSAgentThread = None
        self.FeedbackThread = None
        time.sleep(.1)
        self.StartButton.setEnabled(True)

    def ClearWindows(self):
        self.finalSpeechSegmentsSpeech = []
        self.finalSpeechSegmentsNLP = []
        self.nonFinalText = ""
        self.SpeechBox.clear()
        self.ConceptExtraction.setText('')
        self.ProtocolBox.setText('')
        self.InterventionBox.setText('')

    # Update the Speech Box
    # NOTE: we are only updating speech box from raw transcripts.
    # TODO: Update this to show metamap highlights after they return
    #@pyqtSlot
    def UpdateSpeechBox(self, input):
        item = input[0]
        text = ""
        if item.origin == 'Speech': #TODO: Remove this line
            if item.isFinal:
                # if received segment is final, store it first
                # 'Speech' origin is raw transcription text from Google Speech to Text API
                if(item.origin == 'Speech'):
                    self.finalSpeechSegmentsSpeech.append(
                        '<b>' + f'{item.transcript} ({item.confidence:.3f})' + '</b>')
                # 'NLP' origin is highlighted/bolded text with concept extracted using Metamap
                elif(item.origin == 'NLP'):
                    self.finalSpeechSegmentsNLP.append(
                        '<b>' + f'{item.transcript} ({item.confidence:.3f})' + '</b>')
            # make speech text, starting with final segments
            # if highlighted text from Metamap (NLP array) is available, use that
            # else, use raw transcription from Google Speech (Speech array) to Text API
            # num_final_segments = max(len(self.finalSpeechSegmentsNLP), len(self.finalSpeechSegmentsSpeech))
            for i in range(max(len(self.finalSpeechSegmentsNLP), len(self.finalSpeechSegmentsSpeech))):
                if i < len(self.finalSpeechSegmentsNLP):
                    text += self.finalSpeechSegmentsNLP[i] + "<br>"
                else:
                    text += self.finalSpeechSegmentsSpeech[i] + "<br>"

            # If currently received segment is interim result, add it to text
            # Only display intermin results for raw speech text 
            if not item.isFinal: text += f'{item.transcript} ({item.confidence:.3f})'

            # Set SpeechBox text
            self.SpeechBox.setHtml(text)


    # Update the Concept Extraction Box
    def UpdateConceptExtractionBox(self, input):
        global chunkdata
        global ConceptDict
        item = input[0]
        self.ConceptExtraction.setHtml(item)
        if item != "":
            chunkdata.append(item)
        else:
            chunkdata.append("-")



    # Update the Protocols and Interventions Boxes
    def UpdateProtocolBoxes(self, input):
        global chunkdata

        if(len(input) == 1):
            protocol_names = input[0]
            self.ProtocolBox.setText(protocol_names)
            chunkdata.append(protocol_names)

        if(len(input) == 2):
            interventions = input[1]
            self.InterventionBox.setText(interventions)
            chunkdata.append(interventions)

        with open("check.csv", mode="a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(chunkdata)
        chunkdata = []

    # Update the System Message Box
    def UpdateMsgBox(self, input):
        item = input[0]
        #previousText = self.MsgBox.toPlainText()
        #self.MsgBox.setText(str(previousText).strip() + "\n" + str(datetime.datetime.now().strftime("%c"))[16:] + " - " + item + "\n")
        self.MsgBox.append(
            "<b>" + str(datetime.datetime.now().strftime("%c"))[16:] + "</b> - " + item)

        self.MsgBox.moveCursor(QTextCursor.End)

    # Update the VU Meter
    def UpdateVUBox(self, input):
        signal = input[0]
        try:
            amp = Amplitude.from_data(signal)
            if amp > self.maximal:
                self.maximal = amp

            value = amp.getValue()
            value = value * 100
            log_value = 50 * math.log10(value + 1)  # Plot on a log scale

            self.VUMeter.setValue(log_value)
        except:
            self.VUMeter.setValue(0)

    # Restarts Google Speech API, called when the API limit is reached
    def StartGoogle(self, input):
        item = input[0]
        if(item == 'Mic'):
            self.SpeechThread = StoppableThread(
                target=GoogleSpeechMicStream.GoogleSpeech, args=(self, SpeechToNLPQueue,EMSAgentSpeechToNLPQueue, data_path, audiostream, transcriptStream))
            self.SpeechThread.start()
        elif(item == 'File'):
            if self.ComboBox.currentText() == 'Other Audio File':
                print("\n\nStart Again\n\n")
                self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(
                    self, SpeechToNLPQueue, EMSAgentSpeechToNLPQueue, self.otheraudiofilename, data_path, audiostream, transcriptStream))
            else:
                self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(
                    self, SpeechToNLPQueue,EMSAgentSpeechToNLPQueue, './Audio_Scenarios/2019_Test/' + str(self.ComboBox.currentText()) + '.wav', data_path, audiostream, transcriptStream))
            self.SpeechThread.start()

    # Enabled and/or disable given buttons in a tuple (Button Object, True/False)
    def ButtonsSetEnabled(self, input):
        for item in input:
            item[0].setEnabled(item[1])




    # fire up a temporary QApplication

def get_resolution_multiple_screens():

    app = QGuiApplication(sys.argv)
    #QtWidgets.QtGui
    all_screens = app.screens()

    for s in all_screens:

        print()
        print(s.name())
        print(s.availableGeometry())
        print(s.availableGeometry().width())
        print(s.availableGeometry().height())
        print(s.size())
        print(s.size().width())
        print(s.size().height())

    print()
    print('primary:', app.primaryScreen())
    print('primary:', app.primaryScreen().availableGeometry().width())
    print('primary:', app.primaryScreen().availableGeometry().height())

    # now choose one

# ================================================================== Main ==================================================================
if __name__ == '__main__':

    #run with arguments
    #options:
        # --datacollect --> "1" or "0" for collect data or not
        # --streams --> "all" or list of specific streams (options are audio, video, smartwatch, conceptextract, protocol, intervention, transcript)
    #if no arguments given, default options are "0" for data collection and "all" for streams

    arg_count = len(sys.argv)
    
    print(f"Arguments count: {arg_count}")
    for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>6}: {arg}")

    if(arg_count >= 3): 
        if sys.argv[1] == "--datacollect":
            if sys.argv[2] == "1":
                datacollection = True
                videostream = True
                audiostream = True
                smartwatchStream = True
                conceptExtractionStream = True
                protocolStream = True
                interventionStream = True
                transcriptStream = True
            else: #sys.argv[2] == 0
                datacollection = False
                videostream = False
                audiostream = False
                smartwatchStream = False
                conceptExtractionStream = False
                protocolStream = False
                interventionStream = False
                transcriptStream = False
            if (arg_count > 3):
                if sys.argv[3] == "--streams":
                    videostream = False
                    audiostream = False
                    smartwatchStream = False
                    conceptExtractionStream = False
                    protocolStream = False
                    interventionStream = False
                    transcriptStream = False
                    print("User is specifying streams")
                    for i in range(4, len(sys.argv)):
                        if sys.argv[i] == "audio":
                            audiostream = True
                        if sys.argv[i] == "video":
                            videostream = True
                        if sys.argv[i] == "smartwatch":
                            smartwatchStream = True
                        if sys.argv[i] == "conceptextract":
                            conceptExtractionStream = True
                        if sys.argv[i] == "protocol":
                            protocolStream = True
                        if sys.argv[i] == "intervention":
                            interventionStream = True
                        if sys.argv[i] == "transcript":
                            transcriptStream = True
                        if sys.argv[i] == "all":
                            videostream = True
                            audiostream = True
                            smartwatchStream = True
                            conceptExtractionStream = True
                            protocolStream = True
                            interventionStream = True
                            transcriptStream = True

        else: #if arguments specified but --datacollect is not specified then user should enter option for --datacollect
            print("User Error: Please use --datacollect 1 or --datacollect 0 to specifiy if you want data collected with the data collection algorithm.")
            print("Default data collection will collect all streams, but you may also specify streams with the --streams option")
            exit()
    else:
        print("No data collection arguments specified -- defaulting to no data collection")

    print("stream bools: ", audiostream, videostream, smartwatchStream, conceptExtractionStream, protocolStream, interventionStream, transcriptStream)

    smartwatchStream = True
    videostream = True
    audiostream = True
    # audiostream = True # harcode audio saving
    # Set the Google Speech API service-account key environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"

    # Create thread-safe queue for communication between Speech and Cognitive System threads
    SpeechToNLPQueue = queue.Queue()
    EMSAgentSpeechToNLPQueue  = queue.Queue()
    FeedbackQueue = queue.Queue()

    # GUI: Create the main window, show it, and run the app
    print("Starting GUI")
    app = QApplication(sys.argv)
    screen_resolution = app.desktop().screenGeometry()
    width, height = screen_resolution.width(), screen_resolution.height() 

    Window = MainWindow(width, height)
    Window.show()

    sys.exit(app.exec_())