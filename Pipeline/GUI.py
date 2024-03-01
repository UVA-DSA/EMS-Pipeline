#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============== Imports ==============

from __future__ import absolute_import, division, print_function

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
from PyQt5.QtMultimedia import QMediaPlayer
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QGuiApplication
from PyQt5.QtCore import Qt
import PyQt5.QtWidgets,PyQt5.QtCore

import os

# from smartwatch_streaming import Thread_Watch

chunkdata = []
ConceptDict = dict()

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
        self.setupUi()
    
    def setupUi(self):
        # Set geometry of the window
        self.setWindowTitle('CognitiveEMS Demo')
        self.setWindowIcon(QIcon('./Images/Logos/UVA.png'))
        self.setGeometry(int(self.width * .065), int(self.height * .070),
                         int(self.width * .9), int(self.height * .9))
        
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
        # self.SaveButton.clicked.connect(self.SaveButtonClick)
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

        self.TranscriptBox = QTextEdit()
        self.TranscriptBox.setReadOnly(True)
        self.TranscriptBox.setOverwriteMode(True)
        self.TranscriptBox.ensureCursorVisible()
        QTextCursor(self.TranscriptBox.textCursor()).insertBlock()
        self.Grid_Layout.addWidget(self.TranscriptBox, 3, 0, 2, 1)

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
        # th = Thread(data_path, videostream)
        # th.changePixmap.connect(self.setImage)
        # th.changeVisInfo.connect(self.handle_message2)
        # th.start()  #Disabled for now

        # # Threads for smartwatch
        # th2 = Thread_Watch(data_path, smartwatchStream)
        # th2.changeActivityRec.connect(self.handle_message)
        # th2.start()

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
        self.ProtocolBox.setOverwriteMode(True)
        self.Grid_Layout.addWidget(self.ProtocolBox, 3, 2, 1, 2)

        # Create label and textbox for interventions
        self.InterventionLabel = QLabel()
        self.InterventionLabel.setText("<b>Recognized Interventions</b>")
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
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        if self.mediaPlayer.state() == QMediaPlayer.StoppedState:
            self.VisionInformation.setPlainText("CPR Done\nAverage Compression Rate: 140 bpm")
        else:
            self.mediaPlayer.play()


    def mediaStateChanged(self, state):
    	if self.player.state() == QMediaPlayer.StoppedState:
            print("video has stopped playing!")
            self.VisionInformation.setPlainText("CPR Done\nAverage Compression Rate: 140 bpm")

    # @pyqtSlot()
    # def SaveButtonClick(self):
    #     #name = QFileDialog.getSaveFileName(self, 'Save File')
    #     name = str(datetime.datetime.now().strftime("%c")) + ".txt"
    #     #file = open("./Dumps/" + name,'w')
    #     mode_text = self.ComboBox.currentText()
    #     speech_text = str(self.TranscriptBox.toPlainText())
    #     concept_extraction_text = str(self.ConceptExtraction.toPlainText())
    #     protocol_text = str(self.ProtocolBox.toPlainText())
    #     intervention_text = str(self.InterventionBox.toPlainText())
    #     msg_text = str(self.MsgBox.toPlainText())
    #     #text = "Mode:\n\n" + mode_text + "\n\nSpeech: \n\n" + speech_text +"\n\nConcept Extraction:\n\n" + concept_extraction_text + "\n\nProtocol Text:\n\n" + protocol_text + "\n\nIntervention:\n\n" + intervention_text + "\n\nSystem Messages Log:\n\n" + msg_text
    #     # file.write(text)
    #     # file.close()
    #     #self.UpdateMsgBox(["System state dumped to \n/Dumps/" + name])
    #     self.UpdateMsgBox(["System results saved to results.csv"])
    #     results = [speech_text, concept_extraction_text,
    #                protocol_text, intervention_text]
    #     with open("results.csv", mode="a") as csv_file:
    #         writer = csv.writer(csv_file, delimiter=',')
    #         writer.writerow(results)

    @pyqtSlot()
    def StartButtonClick(self):
        pass

    @pyqtSlot()
    def StopButtonClick(self):
        pass

    @pyqtSlot()
    def GenerateFormButtonClick(self):
        print('Generate Form pressed!')
        self.UpdateMsgBox(["Form Being Generated!"])
        text = str(self.TranscriptBox.toPlainText())
        #StoppableThread(target = GenerateForm.GenerateForm, args=(self, text,)).start()

    @pyqtSlot()
    def ResetButtonClick(self):
        pass

    def ClearWindows(self):
        self.finalSpeechSegmentsSpeech = []
        self.finalSpeechSegmentsNLP = []
        self.nonFinalText = ""
        self.TranscriptBox.clear()
        self.ConceptExtraction.setText('')
        self.ProtocolBox.setText('')
        self.InterventionBox.setText('')

    # Update the Speech Box
    # NOTE: we are only updating speech box from raw transcripts.
    # TODO: Update this to show metamap highlights after they return
    #@pyqtSlot
    def UpdateTranscriptBox(self, transcriptItemList):
        self.TranscriptBox.setHtml(transcriptItemList[0].transcript)

    def UpdateProtocolBox(self, protocolItemList):
        self.ProtocolBox.setHtml(f'({protocolItemList[0].protocol}: {protocolItemList[0].protocol_confidence})')

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
                target=GoogleSpeechMicStream.GoogleSpeech, args=(self, TranscriptQueue,EMSAgentSpeechToNLPQueue, data_path, audiostream, transcriptStream))
            self.SpeechThread.start()
        elif(item == 'File'):
            if self.ComboBox.currentText() == 'Other Audio File':
                print("\n\nStart Again\n\n")
                self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(
                    self, TranscriptQueue, EMSAgentSpeechToNLPQueue, self.otheraudiofilename, data_path, audiostream, transcriptStream))
            else:
                self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(
                    self, TranscriptQueue,EMSAgentSpeechToNLPQueue, './Audio_Scenarios/2019_Test/' + str(self.ComboBox.currentText()) + '.wav', data_path, audiostream, transcriptStream))
            self.SpeechThread.start()

    # Enabled and/or disable given buttons in a tuple (Button Object, True/False)
    def ButtonsSetEnabled(self, input):
        for item in input:
            item[0].setEnabled(item[1])


def StartGUI(WindowQueue):
    print("Inside")
    app = QApplication([])
    screen_resolution = app.desktop().screenGeometry()
    width, height = screen_resolution.width(), screen_resolution.height()
    Window = MainWindow(width, height)
    WindowQueue.put(Window)
    Window.show()
    app.exec_()
    print("DONE")