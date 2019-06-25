#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ============== Imports ==============

from __future__ import absolute_import, division, print_function
from six.moves import queue
import sys
import os
import time
import threading

from PyQt4.QtCore import *
from PyQt4 import QtSvg
from PyQt4.QtGui import *
import numpy as np
#import io
#from Form_Filling import textParse2
#from Form_Filling import prescription_form2
import GenerateForm
import py_trees
import behaviours_m as be
from py_trees.blackboard import Blackboard
import pandas as pd

#import subprocess
#from operator import itemgetter
from DSP.amplitude import Amplitude
import datetime

from StoppableThread.StoppableThread import StoppableThread
from classes import SpeechNLPItem, GUISignal

import GoogleSpeechMicStream
import GoogleSpeechFileStream
import DeepSpeechMicStream
import DeepSpeechFileStream
import CognitiveSystem

# ============== Parameters of the GUI ==============

# 1 or .7
DIM = 1

# ============== GUI ==============

# Main Window of the Application
class MainWindow(QWidget):

    def __init__(self, width, height):
        super(MainWindow, self).__init__()
        self.setGeometry(100, 100, int(1780), int(950 * DIM))
        self.setWindowTitle('CognitiveEMS Demo')
        self.setWindowIcon(QIcon('./Images/Logos/UVA.png'))
        self.setFixedSize(int(1780 * DIM), int(790 * DIM))
        self.width = width
        self.height = height
        self.stopped = 0
        self.reset = 0
        self.maximal = Amplitude()
        self.finalSpeechSegmentsSpeech = []
        self.finalSpeechSegmentsNLP = []
        self.nonFinalText = ""

        # Create main title
        self.NLPLabel = QLabel(self)
        self.NLPLabel.move(int(20 * DIM), int(20 * DIM))
        self.NLPLabel.setText('<font size="6"><b>CognitiveEMS Demo</b></font>')

        # Create label and textbox for speech 
        self.SpeechLabel = QLabel(self)
        self.SpeechLabel.move(int(20 * DIM), int(70 * DIM))
        self.SpeechLabel.setText('<b>Speech Recognition</b>')

        self.SpeechBox = QTextEdit(self)
        self.SpeechBox.move(int(20 * DIM), int(110 * DIM))
        self.SpeechBox.resize(int(500 * DIM), int(500 * DIM))
        self.SpeechBox.setReadOnly(True)
        self.SpeechBox.setOverwriteMode(True)
        font = QFont("Monospace")
        font = QFont()
        font.setPointSize(12)
        self.SpeechBox.setFont(font)
        self.SpeechBox.ensureCursorVisible()

        # Create label and textbox for NLP 
        self.NLPLabel = QLabel(self)
        self.NLPLabel.move(int(540 * DIM), int(70 * DIM))
        self.NLPLabel.setText('<b>Concept Extraction (Concept, Presence, Value, Confidence)</b>')

        self.NLPBox = QTextEdit(self)
        self.NLPBox.move(int(540 * DIM), int(110 * DIM))
        self.NLPBox.resize(int(600 * DIM), int(650 * DIM))
        self.NLPBox.setLineWrapMode(QTextEdit.NoWrap)
        self.NLPBox.setReadOnly(True)

        # Add label, textbox for protcol name
        self.ProtcolLabel = QLabel(self)
        self.ProtcolLabel.move(int(1160 * DIM), int(70 * DIM))
        self.ProtcolLabel.setText('<b>Suggested EMS Protocols (Protocol, Confidence)</b>')

        self.ProtocolName = QTextEdit(self)
        self.ProtocolName.move(int(1160 * DIM), int(110 * DIM))
        self.ProtocolName.resize(int(600 * DIM), int(110 * DIM))
        self.ProtocolName.setReadOnly(True)

        # Create label and textbox for suggestions 
        self.ActionBoxLabel = QLabel(self)
        self.ActionBoxLabel.move(int(1160 * DIM), int(240 * DIM))
        self.ActionBoxLabel.setText('<b>Suggested Interventions (Action, Confidence)</b>')

        self.ActionBox = QTextEdit(self)
        self.ActionBox.move(int(1160 * DIM), int(280 * DIM))
        self.ActionBox.resize(int(600 * DIM), int(110 * DIM))
        self.ActionBox.setReadOnly(True)

        # Create label and textbox for messages
        self.MsgBoxLabel = QLabel(self)
        self.MsgBoxLabel.move(int(1160 * DIM), int(410 * DIM))
        self.MsgBoxLabel.setText("<b>System Messages Log</b>")

        self.MsgBox = QTextEdit(self)
        self.MsgBox.move(int(1160 * DIM), int(450 * DIM))
        self.MsgBox.resize(int(600 * DIM), int(200 * DIM))
        self.MsgBox.setReadOnly(True)
        self.MsgBox.setFont(QFont("Monospace"))

        System_Info_Text_File = open("./Text_Scenarios/System_Info.txt", "r") 
        System_Info_Text = ""
        for line in System_Info_Text_File.readlines():
            System_Info_Text += line
        System_Info_Text_File.close()
        self.MsgBox.setText(System_Info_Text + "\n" + str(datetime.datetime.now().strftime("%c")) + " - Ready to start speech recognition!")

        # Audio Options Menu
        self.ComboBox = QComboBox(self)
        self.ComboBox.move(int(20 * DIM), int(620 * DIM))
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
                                "Other Audio File"])

        # Radio Buttons Google or DeepSpeech
        self.GoogleSpeechRadioButton = QRadioButton("Google Speech API", self)
        self.GoogleSpeechRadioButton.move(160 + int(20 * DIM), int(625 * DIM))
        self.GoogleSpeechRadioButton.setEnabled(True)
        self.GoogleSpeechRadioButton.setChecked(True)
        #self.GoogleSpeechRadioButton.toggled.connect())
        self.DeepSpeechRadioButton = QRadioButton("DeepSpeech", self)
        self.DeepSpeechRadioButton.move(320 + int(20 * DIM), int(625 * DIM))
        self.DeepSpeechRadioButton.setEnabled(False)
        self.DeepSpeechRadioButton.setChecked(False)
        #self.DeepSpeechRadioButton.toggled.connect())

        # Create a start button in the window
        self.StartButton = QPushButton('   Start   ', self)
        self.StartButton.move(int(20 * DIM), int(660 * DIM))
        self.StartButton.clicked.connect(self.StartButtonClick)

        # Create a stop button in the window
        self.StopButton = QPushButton('   Stop   ', self)
        self.StopButton.move(100 + int(20 * DIM), int(660 * DIM))
        self.StopButton.clicked.connect(self.StopButtonClick)

        # Create a reset  button in the window
        self.ResetButton = QPushButton('Reset', self)
        self.ResetButton.move(200 + int(20 * DIM), int(660 * DIM))
        self.ResetButton.clicked.connect(self.ResetButtonClick)
        self.ResetButton.setEnabled(True)

        # Create a generate report button in the window
        self.GenerateFormButton = QPushButton('   Generate Form   ', self)
        self.GenerateFormButton.move(300 + int(20 * DIM), int(660 * DIM))
        self.GenerateFormButton.clicked.connect(self.GenerateFormButtonClick)
        self.GenerateFormButton.setEnabled(True)

        # Add Microphone Logo
        self.MicPictureBox = QLabel(self)
        #self.MicPictureBox.setGeometry(int(500 * DIM), int(800 * DIM), int(80 * DIM), int(80 * DIM))
        self.MicPictureBox.move(int(20 * DIM), int(710 * DIM))
        self.MicPictureBox.setPixmap(QPixmap('./Images/Logos/mic2.png').scaledToWidth(int(35 * DIM)))

        # Create vu meter
        self.VUMeter = QTextEdit(self)
        self.VUMeter.move(int(60 * DIM), int(715 * DIM))
        self.VUMeter.resize(int(1180 * DIM), int(40 * DIM))
        self.VUMeter.setReadOnly(True)
        self.VUMeter.setOverwriteMode(True)
        self.VUMeter.setStyleSheet("background: rgba(0,0,0,0%)")
        self.VUMeter.setFrameShape(QFrame.NoFrame)

        # Add Link Lab Logo
        PictureBox = QLabel(self)
        PictureBox.setGeometry(int(1241.5 * DIM), int(680 * DIM), int(437 * DIM), int(82 * DIM))
        PictureBox.setPixmap(QPixmap('./Images/Logos/LinkLabLogo.png').scaledToWidth(int(437 * DIM)))

        # Add Credits Label
        #CreditsLabel = QLabel(self)
        #CreditsLabel.move(int(20 * DIM), int(770 * DIM))
        #CreditsLabel.setText('\n<small><center>Mustafa Hotaki and Sile Shu, May 2018</center></small>')

        # Threads
        self.SpeechThread = StoppableThread(target = GoogleSpeechMicStream.GoogleSpeech, args=(self, SpeechToNLPQueue,))
        self.CognitiveSystemThread = StoppableThread(target = CognitiveSystem.CognitiveSystem, args=(self, SpeechToNLPQueue,))

    def closeEvent(self, event):
        print('Closing GUI')
        self.stopped = 1
        SpeechToNLPQueue.put('Kill')
        event.accept()

    @pyqtSlot()
    def StartButtonClick(self):
        print('Start pressed!')
        self.UpdateMsgBox(["Starting!"])
        self.reset = 0
        self.stopped = 0

        self.StartButton.setEnabled(False)
        self.ComboBox.setEnabled(False)
        self.ResetButton.setEnabled(True)
        self.StopButton.setEnabled(True)

        # ==== Start the Speech Thread
        if self.ComboBox.currentText() == 'Microphone':
            if(self.GoogleSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(target = GoogleSpeechMicStream.GoogleSpeech, args=(self, SpeechToNLPQueue,))
            elif(self.DeepSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(target = DeepSpeechMicStream.DeepSpeech, args=(self, SpeechToNLPQueue,))
        elif self.ComboBox.currentText() == 'Other Audio File':
            fname = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\',"Wav files (*.wav)")
            if(self.GoogleSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(target = GoogleSpeechFileStream.GoogleSpeech, args=(self, SpeechToNLPQueue, str(fname),))
            elif(self.DeepSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(target = DeepSpeechFileStream.DeepSpeech, args=(self, SpeechToNLPQueue, str(fname),))
            self.otheraudiofilename = str(fname)
        # Hard-coded test files
        else:
            if(self.GoogleSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(target = GoogleSpeechFileStream.GoogleSpeech, args=(self, SpeechToNLPQueue, './Audio_Scenarios/2019_Test/' + str(self.ComboBox.currentText()) + '.wav',))
            elif(self.DeepSpeechRadioButton.isChecked()):
                self.SpeechThread = StoppableThread(target = DeepSpeechFileStream.DeepSpeech, args=(self, SpeechToNLPQueue, './Audio_Scenarios/2019_Test/' + str(self.ComboBox.currentText()) + '.wav',))
        self.SpeechThread.start()

        # ==== Start the Cognitive System Thread
        self.CognitiveSystemThread = StoppableThread(target = CognitiveSystem.CognitiveSystem, args=(self, SpeechToNLPQueue,))
        self.CognitiveSystemThread.start()

    @pyqtSlot()
    def StopButtonClick(self):
        print('Stop pressed!')
        self.UpdateMsgBox(["Stopping!"])
        self.stopped = 1
        self.SpeechThread.stop()
        self.VUMeter.setText('')
        self.StartButton.setEnabled(True)
        self.ComboBox.setEnabled(True)
        self.GenerateFormButton.setEnabled(True)

    @pyqtSlot()
    def GenerateFormButtonClick(self):
        print('Generate Form pressed!')
        self.UpdateMsgBox(["Form Being Generated!"])
        text = str(self.SpeechBox.toPlainText())
        StoppableThread(target = GenerateForm.GenerateForm, args=(self, text,)).start()

    @pyqtSlot()
    def ResetButtonClick(self):
        print('Reset pressed!')
        self.UpdateMsgBox(["Resetting!"])
        self.reset = 1
        self.SpeechThread.stop()
        SpeechToNLPQueue.put('Kill')
        self.StartButton.setEnabled(True)
        self.VUMeter.setText('')
        self.finalSpeechSegmentsSpeech = []
        self.finalSpeechSegmentsNLP = []
        self.SpeechBox.setText('')
        self.NLPBox.setText('')
        self.ProtocolName.setText('')
        self.ActionBox.setText('')
        self.CognitiveSystemThread = StoppableThread(target = CognitiveSystem, args=(self, SpeechToNLPQueue,))
        self.CognitiveSystemThread.start()
        self.SpeechThread = None
        self.nonFinalText = ""

    def ClearWindows(self):
        self.VUMeter.setText('')
        self.SpeechBox.setText('')
        self.NLPBox.setText('')
        self.ProtocolName.setText('')
        self.ActionBox.setText('')

    def UpdateSpeechBox(self, input):
        item = input[0]
        if(item.isFinal):
            if(item.origin == 'Speech'):
                self.finalSpeechSegmentsSpeech.append('<b>' + item.transcript + '</b>')
            elif(item.origin == 'NLP'):
                self.finalSpeechSegmentsNLP.append('<b>' + item.transcript + '</b>')

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

            previousTextMinusPrinted = self.nonFinalText[:len(self.nonFinalText) - item.numPrinted]
            self.nonFinalText =  previousTextMinusPrinted + item.transcript
            self.SpeechBox.setText(text + self.nonFinalText)
            self.SpeechBox.moveCursor(QTextCursor.End)
            
    def UpdateNLPBox(self, input):
        item = input[0]
        self.NLPBox.setText(item)

    def UpdateProtocolBoxes(self, input):
        protocol_names = input[0]
        interventions = input[1]
        self.ProtocolName.setText(protocol_names)
        self.ActionBox.setText(interventions)

    def UpdateMsgBox(self, input):
        item = input[0]
        previousText = self.MsgBox.toPlainText()
        self.MsgBox.setText(str(previousText).strip() + "\n" + str(datetime.datetime.now().strftime("%c"))[16:] + " - " + item + "\n")
        self.MsgBox.moveCursor(QTextCursor.End)

    def UpdateVUBox(self, input):
        signal = input[0]
        amp = Amplitude.from_data(signal)
        if amp > self.maximal:
            self.maximal = amp
        a = (amp.display(scale=1000, mark=self.maximal))
        self.VUMeter.setText('<b>' + '<font color="green">' + str(a[0])[:60] + '</font>' + '</b>') 
 
    def StartGoogle(self, input):
        item = input[0]
        previousText = self.SpeechBox.toPlainText()
        self.SpeechBox.setText('<b>' + previousText + '</b> <br>')
        if(item == 'Mic'):
            self.SpeechThread = StoppableThread(target = GoogleSpeechMicStream.GoogleSpeech, args=(self, SpeechToNLPQueue, ))
            self.SpeechThread.start()
        elif(item == 'File'):
            if self.ComboBox.currentText() == 'Other Audio File':
                print("\n\nStart Again\n\n")
                self.SpeechThread = StoppableThread(target = GoogleSpeechFileStream.GoogleSpeech, args=(self, SpeechToNLPQueue, self.otheraudiofilename,))
            else:
                self.SpeechThread = StoppableThread(target = GoogleSpeechFileStream.GoogleSpeech, args=(self, SpeechToNLPQueue, './Audio_Scenarios/2019_Test/' + str(self.ComboBox.currentText()) + '.wav',))

            self.SpeechThread.start()

# ============== Main ==============

if __name__ == '__main__':

    # Set environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"

    # Communication: Create thread-safe queues
    SpeechToNLPQueue = queue.Queue()

    # GUI: Create the main window, show it, and run the app
    print("Starting GUI")
    app = QApplication(sys.argv)
    screen_resolution = app.desktop().screenGeometry()
    width, height = screen_resolution.width(), screen_resolution.height()
    #width = 1366
    #height = 768
    #print("Screen Resolution\nWidth: %s\nHeight: %s" % (width, height))

    Window = MainWindow(width, height)
    Window.show()
    sys.exit(app.exec_())

