#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============== Imports ==============

from __future__ import absolute_import, division, print_function
from six.moves import queue
import sys
import os
import time
import threading
import math
import datetime

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QLineEdit
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtCore

#from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer

from mediapipe_thread import MPThread

from PyQt5.QtCore import QCoreApplication, Qt,QBasicTimer, QTimer,QPoint,QSize
import PyQt5.QtWidgets,PyQt5.QtCore

import numpy as np
import pandas as pd

from StoppableThread.StoppableThread import StoppableThread

import py_trees
from py_trees.blackboard import Blackboard

#import GenerateForm
from behaviours_m import *
from DSP.amplitude import Amplitude
from classes import SpeechNLPItem, GUISignal
import GoogleSpeechMicStream
import GoogleSpeechFileStream
import DeepSpeechMicStream
import DeepSpeechFileStream
import TextSpeechStream
import CognitiveSystem

import csv
chunkdata = []

# ================================================================== GUI ==================================================================

#BOX_FONT_SIZE = 12

# Main Window of the Application


class MainWindow(QWidget):

    def __init__(self, width, height):
        super(MainWindow, self).__init__()

        # Fields
        self.width = width
        self.height = height
        self.SpeechThread = None
        self.CognitiveSystemThread = None
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
                         int(width * .9), int(height * .85))
        #self.setMinimumSize(1280, 750);
        #self.setFixedSize(width, height)
        # self.showMaximized()

        # Grid Layout to hold all the widgets
        self.Grid_Layout = QGridLayout(self)

        # Font for boxes
        #Box_Font = QFont()
        # Box_Font.setPointSize(BOX_FONT_SIZE)

        # Add disabled buttons to the of GUI to ensure spacing
        R3 = QPushButton(".\n.\n")
        R3.setFont(QFont("Monospace"))
        R3.setEnabled(False)
        R3.setStyleSheet("background-color:transparent;border:0;")
        #self.Grid_Layout.addWidget(R3, 3, 3, 1, 1)

        R6 = QPushButton(".\n.\n")
        R6.setFont(QFont("Monospace"))
        R6.setEnabled(False)
        R6.setStyleSheet("background-color:transparent;border:0;")
        #self.Grid_Layout.addWidget(R6, 6, 3, 1, 1)

        R8 = QPushButton(".\n" * int(self.height/100))
        R8.setFont(QFont("Monospace"))
        R8.setEnabled(False)
        R8.setStyleSheet("background-color:transparent;border:0;")
        #self.Grid_Layout.addWidget(R8, 8, 3, 1, 1)

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
        self.GenerateFormButton.clicked.connect(self.GenerateFormButtonClick)
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
        # self.SpeechBox.setFont(Box_Font)
        self.SpeechBox.setOverwriteMode(True)
        self.SpeechBox.ensureCursorVisible()
        self.Grid_Layout.addWidget(self.SpeechBox, 3, 0, 2, 1)



        #Create label and media player for videos- - added 3/21/2022
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.video = QVideoWidget()
        #self.video.resize(300, 300)
        #self.video.move(0, 0)

        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(directory))) #(QUrl.fromLocalFile("Sample_Video.mp4")))
        #QUrl::fromLocalFile("/home/test/beep.mp3")

        print(self.player.state())
        self.player.play()
        print(QMediaPlayer.PlayingState)

        self.playButton = QPushButton()
        self.playButton.setEnabled(True)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.player.setVideoOutput(self.video)


        
        # Create label and textbox for Vision Information
        self.VisionInformationLabel = QLabel()
        self.VisionInformationLabel.setText("<b>Activity Recognition</b>")
        self.Grid_Layout.addWidget(self.VisionInformationLabel, 5, 1, 1, 1)
        
        self.VisionInformation = QTextEdit() #QLineEdit()
        self.VisionInformation.setReadOnly(True)
        # self.VisionInformation.setFont(Box_Font)
        self.Grid_Layout.addWidget(self.VisionInformation, 6, 1, 2, 1)
  


        self.VideoSubLabel = QLabel()
        self.VideoSubLabel.setText("<b>Video Content<b>") #setGeometry(100,100,100,100)
        self.Grid_Layout.addWidget(self.VideoSubLabel, 5, 0, 1, 1)


        self.video = QLabel(self)

        self.Grid_Layout.addWidget(self.video, 6, 0, 3, 1)

        VIDEO_WIDTH = 841
        VIDEO_HEIGHT = 511
        self.video.setGeometry(QtCore.QRect(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT))

        # self.video.setPixmap(QPixmap("test.jpg"))
        self.video.setScaledContents(True)

        th = MPThread(self)     # mediapipe thread -- see MPThread  in mediapipe_thread file
        th.changePixmap.connect(self.setImage)
        th.start()







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
                                "NG1",
                                "Other Audio File",
                                "Text File"])

        self.ControlPanelGridLayout.addWidget(self.ComboBox, 0, 0, 1, 1)

        # Radio Buttons Google or DeepSpeech
        self.GoogleSpeechRadioButton = QRadioButton("Google Speech API", self)
        self.GoogleSpeechRadioButton.setEnabled(True)
        self.GoogleSpeechRadioButton.setChecked(True)
        self.ControlPanelGridLayout.addWidget(
            self.GoogleSpeechRadioButton, 0, 1, 1, 1)

        self.DeepSpeechRadioButton = QRadioButton("DeepSpeech", self)
        self.DeepSpeechRadioButton.setEnabled(True) #changed from False to True to enable
        self.DeepSpeechRadioButton.setChecked(False)
        self.ControlPanelGridLayout.addWidget(
            self.DeepSpeechRadioButton, 0, 2, 1, 1)

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
        # self.ConceptExtraction.setFont(Box_Font)
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
        # self.ProtocolBox.setFont(Box_Font)
        self.Grid_Layout.addWidget(self.ProtocolBox, 3, 2, 1, 2)

        # Create label and textbox for interventions
        self.InterventionLabel = QLabel()
        self.InterventionLabel.setText("<b>Suggested Interventions</b>")
        self.Grid_Layout.addWidget(self.InterventionLabel, 4, 2, 1, 2)

        self.InterventionSubLabel = QLabel()
        self.InterventionSubLabel.setText("(Intervention, Confidence)")
        self.Grid_Layout.addWidget(self.InterventionSubLabel, 5, 2, 1, 2)

        self.InterventionBox = QTextEdit()
        self.InterventionBox.setReadOnly(True)
        # self.InterventionBox.setFont(Box_Font)
        self.Grid_Layout.addWidget(self.InterventionBox, 6, 2, 1, 2)

        # Create label and textbox for messages
        self.MsgBoxLabel = QLabel()
        self.MsgBoxLabel.setText("<b>System Messages Log</b>")
        self.Grid_Layout.addWidget(self.MsgBoxLabel, 7, 2, 1, 2)

        self.MsgBox = QTextEdit()
        self.MsgBox.setReadOnly(True)
        self.MsgBox.setFont(QFont("Monospace"))
        # self.MsgBox.setLineWrapMode(QTextEdit.NoWrap)
        self.Grid_Layout.addWidget(self.MsgBox, 8, 2, 1, 2)

        # Populate the Message Box with welcome message
        System_Info_Text_File = open("./ETC/System_Info.txt", "r")
        System_Info_Text = ""
        for line in System_Info_Text_File.readlines():
            System_Info_Text += line
        System_Info_Text_File.close()
        #self.MsgBox.setText(System_Info_Text + "\n" + str(datetime.datetime.now().strftime("%c")) + " - Ready to start speech recognition!")
        self.MsgBox.setText(System_Info_Text)
        #self.UpdateMsgBox(["Ready to start speech recognition!"])

        # Add Link Lab Logo
        self.PictureBox = QLabel()
        self.PictureBox.setPixmap(QPixmap('./Images/Logos/LinkLabLogo.png'))
        self.PictureBox.setAlignment(Qt.AlignCenter)
        self.Grid_Layout.addWidget(self.PictureBox, 10, 2, 1, 2)

    # ================================================================== GUI Functions ==================================================================

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
        
        #if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
        #    self.playButton.setIcon(
        #            self.style().standardIcon(QStyle.SP_MediaPause))
        #else:
        #    self.playButton.setIcon(
        #            self.style().standardIcon(QStyle.SP_MediaPlay))
            
    #function to display vision text after delay -- for demo purposes only 4/18/2022
    #def timergo(self):
    #    text = "Start"
    #    try:
    #        self.lst = text #text[self.cnt]
    #        self.line_edit.setText(text)#("".join(str(self.lst[::])))
    #        #self.cnt+=1
    #    except:
    #        #print ("index error in setting text in vision info window")
    #        #or just pass
    #        pass
    #
    #    self.show()
    # Called when closing the GUI
    def closeEvent(self, event):
        print('Closing GUI')
        self.stopped = 1
        self.reset = 1
        SpeechToNLPQueue.put('Kill')
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

        # ==== Start the Speech/Text Thread
        # hacky bypass for demo
        # if True:
        #     self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(
        #             self, SpeechToNLPQueue, './Audio_Scenarios/test5.wav',))
        #     # self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(
        #     #         self, SpeechToNLPQueue, './Audio_Scenarios/2019_Test/002_190105.wav',))
        #     self.SpeechThread.start()
        #     print("Demo Audio Started")

        # If Microphone
        if(self.ComboBox.currentText() == 'Microphone'):
            if(self.GoogleSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(
                    target=GoogleSpeechMicStream.GoogleSpeech, args=(self, SpeechToNLPQueue,))
            elif(self.DeepSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(
                    target=DeepSpeechMicStream.DeepSpeech, args=(self, SpeechToNLPQueue,))
            self.SpeechThread.start()
            print('Microphone Speech Thread Started')

        # If Other Audio File
        elif(self.ComboBox.currentText() == 'Other Audio File'):
            audio_fname = QFileDialog.getOpenFileName(
                self, 'Open file', 'c:\\', "Wav files (*.wav)")
            if(self.GoogleSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(
                    self, SpeechToNLPQueue, str(audio_fname),))
            elif(self.DeepSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(target=DeepSpeechFileStream.DeepSpeech, args=(
                    self, SpeechToNLPQueue, str(audio_fname),))
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
                    self, SpeechToNLPQueue, './Audio_Scenarios/2019_Test/' + str(self.ComboBox.currentText()) + '.wav',))
            elif(self.DeepSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(target=DeepSpeechFileStream.DeepSpeech, args=(
                    self, SpeechToNLPQueue, './Audio_Scenarios/2019_Test/' + str(self.ComboBox.currentText()) + '.wav',))
            self.SpeechThread.start()
            print("Hard-coded Audio File Speech Thread Started")

        # ==== Start the Cognitive System Thread
        if(self.CognitiveSystemThread == None):
            print("Cognitive System Thread Started")
            self.CognitiveSystemThread = StoppableThread(
                target=CognitiveSystem.CognitiveSystem, args=(self, SpeechToNLPQueue,))
            self.CognitiveSystemThread.start()
    
    
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.video.setPixmap(QPixmap.fromImage(image))

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
    def UpdateSpeechBox(self, input):
        
        item = input[0]

        if(item.isFinal):
            if(item.origin == 'Speech'):
                self.finalSpeechSegmentsSpeech.append(
                    '<b>' + item.transcript + '</b>')
            elif(item.origin == 'NLP'):
                self.finalSpeechSegmentsNLP.append(
                    '<b>' + item.transcript + '</b>')

            text = ""

            for a in self.finalSpeechSegmentsNLP:
                text += a + "<br>"

            for a in self.finalSpeechSegmentsSpeech[len(self.finalSpeechSegmentsNLP):]:
                text += a + "<br>"

            self.SpeechBox.setText('<b>' + text + '</b>')
            self.SpeechBox.moveCursor(QTextCursor.End)
            self.nonFinalText = ""

        else:
            text = ""

            for a in self.finalSpeechSegmentsNLP:
                text += a + "<br>"

            for a in self.finalSpeechSegmentsSpeech[len(self.finalSpeechSegmentsNLP):]:
                text += a + "<br>"

            if(not len(text)):
                text = "<b></b>"

            previousTextMinusPrinted = self.nonFinalText[:len(
                self.nonFinalText) - item.numPrinted]
            self.nonFinalText = previousTextMinusPrinted + item.transcript
            self.SpeechBox.setText(text + self.nonFinalText)
            self.SpeechBox.moveCursor(QTextCursor.End)
            
            

    # Update the Concept Extraction Box
    def UpdateConceptExtractionBox(self, input):
        global chunkdata
        item = input[0]
        self.ConceptExtraction.setText(item)
        if item != "":
            chunkdata.append(item)
        else:
            chunkdata.append("-")
            
      

    # Update the Protocols and Interventions Boxes
    def UpdateProtocolBoxes(self, input):
        global chunkdata
        protocol_names = input[0]
        interventions = input[1]
        self.ProtocolBox.setText(protocol_names)
        self.InterventionBox.setText(interventions)

        chunkdata.append(protocol_names)
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
                target=GoogleSpeechMicStream.GoogleSpeech, args=(self, SpeechToNLPQueue, ))
            self.SpeechThread.start()
        elif(item == 'File'):
            if self.ComboBox.currentText() == 'Other Audio File':
                print("\n\nStart Again\n\n")
                self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(
                    self, SpeechToNLPQueue, self.otheraudiofilename,))
            else:
                self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(
                    self, SpeechToNLPQueue, './Audio_Scenarios/2019_Test/' + str(self.ComboBox.currentText()) + '.wav',))
            self.SpeechThread.start()

    # Enabled and/or disable given buttons in a tuple (Button Object, True/False)
    def ButtonsSetEnabled(self, input):
        for item in input:
            item[0].setEnabled(item[1])


# ================================================================== Main ==================================================================
if __name__ == '__main__':

    # Set the Google Speech API service-account key environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"

    # Create thread-safe queue for communication between Speech and Cognitive System threads
    SpeechToNLPQueue = queue.Queue()

    # GUI: Create the main window, show it, and run the app
    print("Starting GUI")
    app = QApplication(sys.argv)
    screen_resolution = app.desktop().screenGeometry()
    width, height = screen_resolution.width(), screen_resolution.height()

    #width = 1920
    #height = 1080
    #width = 1366
    #height = 768
    print("Screen Resolution\nWidth: %s\nHeight: %s" % (width, height))
    Window = MainWindow(width, height)
    Window.show()

    Window.StartButtonClick()

    #v = VideoPlayer()
    #b = QPushButton('start')
    #b.clicked.connect(v.callback)
    #b.show()

    sys.exit(app.exec_())
