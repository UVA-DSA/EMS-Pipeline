#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============== Imports ==============

from __future__ import absolute_import, division, print_function

import math
import datetime
import csv
import sys
import os
import queue
import subprocess

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QPushButton, QLineEdit
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaPlayer
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtGui import QImage, QPixmap, QGuiApplication
import PyQt5.QtWidgets,PyQt5.QtCore

# from behaviours_m import *
from DSP.amplitude import Amplitude
# import CognitiveSystem
#import Feedback
import GoogleSpeechMicStream
import GoogleSpeechFileStream
#import DeepSpeechMicStream
#import DeepSpeechFileStream
# import WavVecMicStream
# import WavVecFileStream

import pipeline_config
# import WhisperFileStream
# import WhisperMicStream

from EMS_Agent.Interface import EMSTinyBERTSystem


from StoppableThread.StoppableThread import StoppableThread

from video_streaming_sebastian import VideoStreamManager
from audio_streaming_sebastian import AudioStreamManager, SpeechProcessManager
from imu_streaming_sebastian import IMUStreamManager
# from smartwatch_streaming import Thread_Watch, IMUThread
#from Feedback import FeedbackClient

from GenUtils.genutils import *

import simulator_network_receiver_sebastian as sim_seb

chunkdata = []

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
        #self.FeedbackThread = None
        self.stopped = 0
        self.reset = 0
        self.maximal = Amplitude()
        self.finalSpeechSegmentsSpeech = []
        self.finalSpeechSegmentsNLP = []
        self.nonFinalText = ""

        self.ip_address = get_local_ipv4()

        self.VideoThread = None


        #whisper

        self.WhisperSubprocess = None

        # Set geometry of the window
        self.setWindowTitle('CognitiveEMS Demo')
        self.setWindowIcon(QIcon('./Images/Logos/UVA.png'))
        self.setGeometry(int(width * .065), int(height * .070),
                         int(width * .9), int(height * .9))
        self.Grid_Layout = QGridLayout(self)

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
        #self.GenerateFormButton.clicked.connect(self.GenerateFormButtonClick)
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


        self.Box = QVBoxLayout()
        self.Grid_Layout.addLayout(self.Box, 5, 1, 4, 1)

        # Create label and textbox for Vision Information
        self.VisionInformationLabel = QLabel()
        # self.VisionInformationLabel.move(50,50)
        self.VisionInformationLabel.setText("<b>Vision Information</b>")
        self.Box.addWidget(self.VisionInformationLabel) #7,1,1,1

        self.VisionInformation = QTextEdit() #QLineEdit()
        self.VisionInformation.setReadOnly(True)
        # self.VisionInformation.setFont(Box_Font)
        self.Box.addWidget(self.VisionInformation) #8,1,1,1

        # self.Box.setMargin(0)
        self.Box.setSpacing(0)
        self.Box.setContentsMargins(0,0,0,0)
        self.Box.addStretch()
        # self.Box.sizePolicy.setHorizontalStretch(1)
        # Create label and textbox for Smartwatch
        self.Box2 = QVBoxLayout()
        self.Grid_Layout.addLayout(self.Box2, 2, 1, 2, 1)

        self.SmartwatchLabel = QLabel()
        self.SmartwatchLabel.setText("<b>Smartwatch Activity</b>")
        self.Box2.addWidget(self.SmartwatchLabel) #5,1,1,1

        self.Smartwatch = QTextEdit() #QLineEdit()
        self.Smartwatch.setReadOnly(True)
        self.Smartwatch.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.VisionInformation.setFont(Box_Font)
        self.Box2.addWidget(self.Smartwatch) # 6, 1, 1, 1
        self.Box2.addStretch()
        # self.Box2.sizePolicy.setHorizontalStretch(1)

        self.VideoSubLabel = QLabel()
        self.VideoSubLabel.setText("<b>Video Content<b>") #setGeometry(100,100,100,100)
        self.Grid_Layout.addWidget(self.VideoSubLabel, 5, 0, 1, 1)

        self.video = QLabel(self)

        self.Grid_Layout.addWidget(self.video, 6, 0, 3, 1)

        VIDEO_WIDTH = 640
        VIDEO_HEIGHT = 480
        self.video.setGeometry(QtCore.QRect(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT))

        # Thread for internet check
        # Initialize the internet check thread
        self.internet_check_thread = InternetCheckThread()
        self.internet_check_thread.internet_status_signal.connect(self.update_internet_status)
        self.internet_check_thread.start()


        #self.feedback_client = Feedback.FeedbackClient()
        #self.feedback_client.start()

        self.start_video_audio_imu_processes()




        # ==== Start the EMS Agent - Xueren ==== #
        print("EMSAgent Thread Started")
        self.EMSAgentThread = EMSTinyBERTSystem.EMSAgentInference(self,EMSAgentSpeechToNLPQueue, FeedbackQueue)
        self.EMSAgentThread.start()
        # self.EMSAgentThread = StoppableThread(
        #     # target=EMSAgenSystem.EMSAgentSystem, args=(self, EMSAgentSpeechToNLPQueue, FeedbackQueue, data_path, protocolStream))
        #     target=EMSTinyBERTSystem.EMSTinyBERTSystem, args=(self, EMSAgentSpeechToNLPQueue, FeedbackQueue))
        # self.EMSAgentThread.start()


        # th2 = ThreadAudio(self)
        # th2.start()

        # Control Panel: To hold combo box, radio, start, stop, and reset buttons
        self.ControlPanel = QWidget()
        self.ControlPanelGridLayout = QGridLayout(self.ControlPanel)
        self.Grid_Layout.addWidget(self.ControlPanel, 9, 0, 1, 2)



        self.DataSourceBox = QComboBox()

        self.DataSourceBox.addItems(["simulator", "local files", "Microphone"])

        self.ControlPanelGridLayout.addWidget(self.DataSourceBox, 0, 0, 1, 1)

        # Modality Selection Group
        self.ModalityGroupBox = QGroupBox("Select Modalities")
        self.ModalityLayout = QHBoxLayout()

        self.VideoCheckBox = QCheckBox("Video")
        self.VideoCheckBox.setChecked(True)
        self.ModalityLayout.addWidget(self.VideoCheckBox)

        self.VideoMLCheckBox = QCheckBox("Video ML")
        self.VideoMLCheckBox.setChecked(False)
        self.ModalityLayout.addWidget(self.VideoMLCheckBox)

        self.AudioCheckBox = QCheckBox("Audio")
        self.AudioCheckBox.setChecked(True)
        self.ModalityLayout.addWidget(self.AudioCheckBox)

        self.AudioMLCheckBox = QCheckBox("Audio ML")
        self.AudioMLCheckBox.setChecked(False)
        self.ModalityLayout.addWidget(self.AudioMLCheckBox)

        self.SmartWatchCheckBox = QCheckBox("Smart Watch")
        self.SmartWatchCheckBox.setChecked(True)
        self.ModalityLayout.addWidget(self.SmartWatchCheckBox)

        self.SmartWatchMLCheckBox = QCheckBox("Smart Watch ML")
        self.SmartWatchMLCheckBox.setChecked(False)
        self.ModalityLayout.addWidget(self.SmartWatchMLCheckBox)

        self.ModalityGroupBox.setLayout(self.ModalityLayout)
        self.ControlPanelGridLayout.addWidget(self.ModalityGroupBox, 2, 0, 1, 3)

        # Data Source Configuration Container
        self.ConfigGroupBox = QGroupBox("Data Source Configuration")
        self.ConfigLayout = QVBoxLayout()
        self.ConfigGroupBox.setLayout(self.ConfigLayout)
        self.ControlPanelGridLayout.addWidget(self.ConfigGroupBox, 3, 0, 1, 3)

        # Connect data source change to update configuration options
        self.DataSourceBox.currentTextChanged.connect(self.UpdateDataSourceConfig)

        # Initialize with default configuration
        self.UpdateDataSourceConfig(self.DataSourceBox.currentText())

        # Radio Buttons Google or Other ML Model
        self.GoogleSpeechRadioButton = QRadioButton("Google Speech Cloud Model", self)
        self.GoogleSpeechRadioButton.setEnabled(True)
        self.GoogleSpeechRadioButton.setChecked(True)
        self.ControlPanelGridLayout.addWidget(
            self.GoogleSpeechRadioButton, 0, 1, 1, 1)

        self.MLSpeechRadioButton = QRadioButton("OpenAI Whisper Local Model", self)
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
        # self.ConceptExtractionLabel = QLabel()
        # self.ConceptExtractionLabel.setText("<b>Concept Extraction</b>")
        # self.Grid_Layout.addWidget(self.ConceptExtractionLabel, 1, 1, 1, 1)

        # self.ConceptExtractionSubLabel = QLabel()
        # self.ConceptExtractionSubLabel.setText(
        #     "(Concept, Presence, Value, Confidence)")
        # self.Grid_Layout.addWidget(self.ConceptExtractionSubLabel, 2, 1, 1, 1)

        # self.ConceptExtraction = QTextEdit()
        # self.ConceptExtraction.setReadOnly(True)
        # # self.ConceptExtraction.setFont(Box_Font)
        # self.Grid_Layout.addWidget(self.ConceptExtraction, 3, 1, 2, 1)


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
        # self.InterventionBox.setFont(Box_Font)
        self.Grid_Layout.addWidget(self.InterventionBox, 7, 2, 1, 2)

        self.Box3 = QVBoxLayout()
        self.Grid_Layout.addLayout(self.Box3, 8, 2, 1, 1)

        # Create label and textbox for messages
        self.MsgBoxLabel = QLabel()
        self.MsgBoxLabel.setText("<b>System Messages Log</b>")
        self.Box3.addWidget(self.MsgBoxLabel)#, 7, 2, 1, 2)
        # self.MsgBoxLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.MsgBox = QTextEdit()
        self.MsgBox.setReadOnly(True)
        self.MsgBox.setFont(QFont("Monospace"))
        # self.MsgBox.setLineWrapMode(QTextEdit.NoWrap)
        # self.MsgBox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.Box3.addWidget(self.MsgBox)#, 8, 2, 1, 2)
        # self.Grid_Layout.setRowStretch(7, 0)

        # Populate the Message Box with welcome message
        System_Info_Text_File = open("../ETC/System_Info.txt", "r")
        System_Info_Text = ""
        for line in System_Info_Text_File.readlines():
            System_Info_Text += line
        System_Info_Text_File.close()
        self.MsgBox.setText(System_Info_Text + "\n" + str(datetime.datetime.now().strftime("%c")) + " - Ready to start speech recognition!")
        self.MsgBox.setText(System_Info_Text)
        self.UpdateMsgBox(["Ready to start speech recognition!"])


        self.IpAddressLabel = QLabel("IP: " + self.ip_address)
        self.IpAddressLabel.setFont(QFont("Monospace", 12))
        self.Box3.addWidget(self.IpAddressLabel)  # Add this below the system log text box in the layout

        self.InternetAvailLabel = QLabel("")
        self.InternetAvailLabel.setFont(QFont("Arial", 22))
        # if(self.internet_avail):
        #     self.InternetAvailLabel.setStyleSheet("color: green; font-weight: bold; ")
        # else:
        #     self.InternetAvailLabel.setStyleSheet("color: red; font-weight: bold; ")
        self.Box3.addWidget(self.InternetAvailLabel)  # Add this below the system log text box in the layout

        # Add Link Lab Logo
        self.PictureBox = QLabel()
        self.PictureBox.setPixmap(QPixmap('./Images/Logos/LinkLabLogo.png'))
        self.PictureBox.setAlignment(Qt.AlignCenter)
        self.Grid_Layout.addWidget(self.PictureBox, 10, 2, 1, 2)

        # self.Grid_Layout.addStretch()
        # self.Grid_Layout.setRowStretch(self.Grid_Layout.rowCount(), 2)
        # self.Grid_Layout.setColumnStretch(self.Grid_Layout.columnCount(), 2)
        vspacer = PyQt5.QtWidgets.QSpacerItem(
            PyQt5.QtWidgets.QSizePolicy.Minimum, PyQt5.QtWidgets.QSizePolicy.Expanding)
        self.Grid_Layout.addItem(vspacer, 11, 0, 1, -1)

        hspacer = PyQt5.QtWidgets.QSpacerItem(
            PyQt5.QtWidgets.QSizePolicy.Expanding, PyQt5.QtWidgets.QSizePolicy.Minimum)
        self.Grid_Layout.addItem(hspacer, 0, 2, -1, 1)

        # self.Grid_Layout.setSpacing(0)
        # self.Grid_Layout.setContentsMargins(0, 0, 0, 0)

    def start_video_audio_imu_processes(self):
        # Processes for video
        # self.VideoThread = VideoThread(data_path, videostream)
        # self.VideoThread.changePixmap.connect(self.setImage)
        # self.VideoThread.changeVisInfo.connect(self.handle_message2)
        # sim_seb.setup_pipes()
        # self.VideoThread.start()  #Disabled for now

        self.VideoProcess = VideoStreamManager(data_path, videostream)
        self.VideoProcess.changePixmap.connect(self.setImage)
        self.VideoProcess.changeVisInfo.connect(self.handle_message2)
        sim_seb.setup_pipes()
        self.VideoProcess.start()

        # Processes for audio

        self.AudioProcess = AudioStreamManager()
        self.AudioProcess.start()

        self.SpeechProcess = SpeechProcessManager()
        self.SpeechProcess.transcript_ready.connect(self.update_transcript_widget)
        self.SpeechProcess.start()

        # self.AudioThread = AudioThread()
        # self.AudioThread.start()

        # self.SpeechThread = SpeechThread()

        # Processes for smartwatch

        self.IMUProcess = IMUStreamManager(data_path, smartwatchStream)
        self.IMUProcess.changeActivityRec.connect(self.handle_message)
        self.IMUProcess.start()

        # th2 = Thread_Watch(data_path, smartwatchStream, pipeline_config.smartwatch_ip, pipeline_config.smartwatch_port)
        # th2.changeActivityRec.connect(self.handle_message)
        # th2.start()

    def update_transcript_widget(self, transcript):
        # current_text = self.SpeechBox.toPlainText()
        self.SpeechBox.append(transcript)
        # self.SpeechBox.setPlainText(current_text + transcript + "\n")
        # self.SpeechBox.moveCursor(QTextCursor.End)

    def UpdateDataSourceConfig(self, source):
        """Update configuration options based on selected data source"""
        # Clear existing widgets
        while self.ConfigLayout.count():
            child = self.ConfigLayout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if source == "simulator":
            # Simulator configuration
            simLayout = QGridLayout()

            # Height
            simLayout.addWidget(QLabel("Height:"), 0, 0)
            self.HeightSpinBox = QSpinBox()
            self.HeightSpinBox.setRange(240, 2160)
            self.HeightSpinBox.setValue(270)
            simLayout.addWidget(self.HeightSpinBox, 0, 1)

            # Width
            simLayout.addWidget(QLabel("Width:"), 1, 0)
            self.WidthSpinBox = QSpinBox()
            self.WidthSpinBox.setRange(320, 3840)
            self.WidthSpinBox.setValue(480)
            simLayout.addWidget(self.WidthSpinBox, 1, 1)

            # IP Address
            simLayout.addWidget(QLabel("IP Address:"), 2, 0)
            self.IPAddressLineEdit = QLineEdit()
            self.IPAddressLineEdit.setText("127.0.0.1")
            self.IPAddressLineEdit.setPlaceholderText("e.g., 192.168.1.100")
            simLayout.addWidget(self.IPAddressLineEdit, 2, 1)

            # Port Number
            simLayout.addWidget(QLabel("Port:"), 3, 0)
            self.PortSpinBox = QSpinBox()
            self.PortSpinBox.setRange(1024, 65535)
            self.PortSpinBox.setValue(9000)
            simLayout.addWidget(self.PortSpinBox, 3, 1)

            self.ConfigLayout.addLayout(simLayout)

        elif source == "local files":
            # Local files configuration
            fileLayout = QVBoxLayout()

            # MP4 file selection
            mp4Layout = QHBoxLayout()
            mp4Layout.addWidget(QLabel("Video File (MP4):"))
            self.VideoFileLineEdit = QLineEdit()
            self.VideoFileLineEdit.setReadOnly(True)
            self.VideoFileLineEdit.setPlaceholderText("No file selected")
            mp4Layout.addWidget(self.VideoFileLineEdit)

            self.VideoFileBrowseButton = QPushButton("Browse...")
            self.VideoFileBrowseButton.clicked.connect(self.BrowseVideoFile)
            mp4Layout.addWidget(self.VideoFileBrowseButton)
            fileLayout.addLayout(mp4Layout)

            # CSV file selection
            csvLayout = QHBoxLayout()
            csvLayout.addWidget(QLabel("Data File (CSV):"))
            self.CSVFileLineEdit = QLineEdit()
            self.CSVFileLineEdit.setReadOnly(True)
            self.CSVFileLineEdit.setPlaceholderText("No file selected")
            csvLayout.addWidget(self.CSVFileLineEdit)

            self.CSVFileBrowseButton = QPushButton("Browse...")
            self.CSVFileBrowseButton.clicked.connect(self.BrowseCSVFile)
            csvLayout.addWidget(self.CSVFileBrowseButton)
            fileLayout.addLayout(csvLayout)

            self.ConfigLayout.addLayout(fileLayout)

        elif source == "Microphone":
            # Microphone configuration (if needed)
            micLayout = QVBoxLayout()
            micLayout.addWidget(QLabel("Microphone input will be used for audio capture."))
            self.ConfigLayout.addLayout(micLayout)

    def BrowseVideoFile(self):
        """Open file dialog to select MP4 video file"""
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4);;All Files (*)"
        )
        if fileName:
            self.VideoFileLineEdit.setText(fileName)

    def BrowseCSVFile(self):
        """Open file dialog to select CSV data file"""
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV Data File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if fileName:
            self.CSVFileLineEdit.setText(fileName)

    @pyqtSlot()
    def StartButtonClick(self):

        print('Start pressed!')
        self.UpdateMsgBox(["Starting!"])
        self.reset = 0
        self.stopped = 0
        self.StartButton.setEnabled(False)
        self.StopButton.setEnabled(True)
        self.DataSourceBox.setEnabled(False)
        self.ResetButton.setEnabled(False)

        # Get selected modalities
        modalities = set()
        if self.VideoCheckBox.isChecked():
            modalities.add(1)
        if self.AudioCheckBox.isChecked():
            modalities.add(2)
        if self.SmartWatchCheckBox.isChecked():
            modalities.add(3)
        if self.VideoMLCheckBox.isChecked():
            modalities.add(4)
        if self.AudioMLCheckBox.isChecked():
            modalities.add(5)
        if self.SmartWatchMLCheckBox.isChecked():
            modalities.add(6)

        # Get data source and configuration
        source = self.DataSourceBox.currentText()

        if source == "simulator":
            config = {
                'height': self.HeightSpinBox.value(),
                'width': self.WidthSpinBox.value(),
                'ip': self.IPAddressLineEdit.text(),
                'port': self.PortSpinBox.value()
            }
            # sim_seb.setup_pipes()
            self.srt_client = sim_seb.SRTReceiverProcess(config['ip'], config['port'], config['width'], config['height'], modalities)
            if not self.srt_client.start():
                print("[Main] Failed to connect to SRT server!")
                sys.exit(1)

            # if(self.MLSpeechRadioButton.isChecked()):
            #     print("Starting Whisper for simulator audio")
            #     whispercppcommand = [
            #         "./stream",
            #         "-m", # use specific whisper model
            #         f"models/ggml-{pipeline_config.whisper_model_size}.bin",
            #         "--threads",
            #         str(pipeline_config.num_threads),
            #         "--step",
            #         str(pipeline_config.step),
            #         "--length",
            #         str(pipeline_config.length),
            #         "--keep",
            #         str(pipeline_config.keep_ms)
            #     ]
            #
            #     # Start subprocess
            #     self.WhisperSubprocess = subprocess.Popen(whispercppcommand, cwd='EMS_Whisper/')
            #
            #     # time.sleep(5)
            #
            #     self.SpeechThread = StoppableThread(
            #         target=WhisperMicStream.WhisperMicStream, args=(self, SpeechToNLPQueue,EMSAgentSpeechToNLPQueue,))
            #
            # self.SpeechThread.start()
            # print('started simulator speech thread')



        elif source == "local files":
            config = {
                'video_file': self.VideoFileLineEdit.text(),
                'csv_file': self.CSVFileLineEdit.text()
            }
        elif source == "Microphone":
            config = {}

        # Your start logic here
        print(f"Modalities: {modalities}")
        print(f"Source: {source}")
        print(f"Config: {config}")



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

    @pyqtSlot(bool)
    def update_internet_status(self, is_available):
        self.ip_address = get_local_ipv4()

        if is_available:
            self.InternetAvailLabel.setText("INTERNET AVAILABLE")
            self.InternetAvailLabel.setStyleSheet("color: green; font-weight: bold;")
            self.GoogleSpeechRadioButton.setEnabled(True)
        else:
            self.InternetAvailLabel.setText("NO INTERNET")
            self.InternetAvailLabel.setStyleSheet("color: red; font-weight: bold;")
            self.GoogleSpeechRadioButton.setEnabled(False)

            if(not self.MLSpeechRadioButton.isChecked()):
                #stop everything.
                self.StopButtonClick()

                #reset everything
                # self.ResetButtonClick()

                self.DataSourceBox.setEnabled(True)

                self.MLSpeechRadioButton.click()



    def mediaStateChanged(self, state):
        if self.player.state() == QMediaPlayer.StoppedState:
            print("video has stopped playing!")
            self.VisionInformation.setPlainText("CPR Done\nAverage Compression Rate: 140 bpm")


    def closeEvent(self, event):
        """
        Called when the GUI window is closed.
        Clean up all processes before exiting.
        """
        print("[GUI] Closing, cleaning up processes...")

        # Stop SRT receiver
        if hasattr(self, 'srt_client') and self.srt_client is not None:
            print("[GUI] Stopping SRT receiver...")
            self.srt_client.stop()

        # Stop Video streaming
        if hasattr(self, 'VideoThread') and self.VideoThread is not None:
            print("[GUI] Stopping video stream...")
            self.VideoThread.stop()

        # Stop Audio streaming
        if hasattr(self, 'AudioThread') and self.AudioThread is not None:
            print("[GUI] Stopping audio stream...")
            self.AudioThread.stop()

        # Stop IMU streaming
        if hasattr(self, 'WatchThread') and self.WatchThread is not None:
            print("[GUI] Stopping IMU stream...")
            self.WatchThread.stop()

        # Stop Speech/Whisper process (this kills egosim_stream)
        if hasattr(self, 'SpeechProcess') and self.SpeechProcess is not None:
            print("[GUI] Stopping speech process...")
            self.SpeechProcess.stop()

        print("[GUI] All processes stopped")
        event.accept()

    # Called when closing the GUI
    def closeEvent1(self, event):
        print('Closing GUI')
        #self.feedback_client.send_message("Exiting!", 'exit')
        #self.feedback_client.sio.disconnect()
        #self.feedback_client.stop()
        # self.th2.exit()
        self.internet_check_thread.stop()
        self.stopped = 1
        self.reset = 1
        # self.VideoThread.stop()


        # Stop SRT receiver first (stops data flow)
        if hasattr(self, 'srt_client') and self.srt_client is not None:
            print("[GUI] Stopping SRT receiver...")
            self.srt_client.stop()

        # Stop video
        if hasattr(self, 'VideoProcess') and self.VideoProcess is not None:
            print("[GUI] Stopping VideoProcess...")
            self.VideoProcess.stop()

        # Stop audio
        if hasattr(self, 'AudioProcess') and self.AudioProcess is not None:
            print("[GUI] Stopping AudioProcess...")
            self.AudioProcess.stop()

        # Stop IMU/Watch
        if hasattr(self, 'IMUProcess') and self.IMUProcess is not None:
            print("[GUI] Stopping IMUProcess...")
            self.IMUProcess.stop()

        print("[GUI] All processes stopped")



        SpeechToNLPQueue.put('Kill')
        EMSAgentSpeechToNLPQueue.put('Exit')
        FeedbackQueue.put('Kill')
        event.accept()

        # self.th2.join()



    @pyqtSlot()
    def SaveButtonClick(self):
        #name = QFileDialog.getSaveFileName(self, 'Save File')
        name = str(datetime.datetime.now().strftime("%c")) + ".txt"
        #file = open("./Dumps/" + name,'w')
        mode_text = self.DataSourceBox.currentText()
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

    # @pyqtSlot()
    # def StartButtonClick1(self):
    #     print('Start pressed!')
    #     self.UpdateMsgBox(["Starting!"])
    #     self.reset = 0
    #     self.stopped = 0
    #     self.StartButton.setEnabled(False)
    #     self.StopButton.setEnabled(True)
    #     self.DataSourceBox.setEnabled(False)
    #     self.ResetButton.setEnabled(False)
    #
    #
    #
    #
    #
    #     # ==== Start the Speech/Text Thread
    #     # hacky bypass for demo
    #     # if True:
    #     #     self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(
    #     #             self, SpeechToNLPQueue, './Audio_Scenarios/test5.wav',))
    #     #     # self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(
    #     #     #         self, SpeechToNLPQueue, './Audio_Scenarios/2019_Test/002_190105.wav',))
    #     #     self.SpeechThread.start()
    #     #     print("Demo Audio Started")
    #
    #     # If Microphone
    #     if(self.DataSourceBox.currentText() == 'Microphone'):
    #         if(self.GoogleSpeechRadioButton.isChecked()):
    #             self.SpeechThread = StoppableThread(
    #                 target=GoogleSpeechMicStream.GoogleSpeech, args=(self, SpeechToNLPQueue,EMSAgentSpeechToNLPQueue, data_path, audiostream, transcriptStream,))
    #
    #
    #         elif(self.MLSpeechRadioButton.isChecked()):
    #
    #             print("Starting Whisper for Microphone")
    #             # Start Whisper module
    #             whispercppcommand = [
    #                 "./stream",
    #                 "-m", # use specific whisper model
    #                 f"models/ggml-{pipeline_config.whisper_model_size}.bin",
    #                 "--threads",
    #                 str(pipeline_config.num_threads),
    #                 "--step",
    #                 str(pipeline_config.step),
    #                 "--length",
    #                 str(pipeline_config.length),
    #                 "--keep",
    #                 str(pipeline_config.keep_ms)
    #             ]
    #
    #             # If a Hard-coded Audio test file, use virtual mic to capture the recording
    #             # whispercppcommand.append("--capture")
    #
    #             # Start subprocess
    #             self.WhisperSubprocess = subprocess.Popen(whispercppcommand, cwd='EMS_Whisper/')
    #
    #             # time.sleep(5)
    #
    #             self.SpeechThread = StoppableThread(
    #                 target=WhisperMicStream.WhisperMicStream, args=(self, SpeechToNLPQueue,EMSAgentSpeechToNLPQueue,))
    #
    #
    #         self.SpeechThread.start()
    #         print('Microphone Speech Thread Started')
    #
    #     # If Other Audio File
    #     # elif(self.ComboBox.currentText() == 'Other Audio File'):
    #     #     audio_fname = QFileDialog.getOpenFileName(
    #     #         self, 'Open file', 'c:\\', "Wav files (*.wav)")
    #     #     if(self.GoogleSpeechRadioButton.isChecked()):
    #     #         self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(
    #     #             self, SpeechToNLPQueue, EMSAgentSpeechToNLPQueue, str(audio_fname), data_path, audiostream, transcriptStream,))
    #     #     elif(self.MLSpeechRadioButton.isChecked()):
    #     #         self.SpeechThread = StoppableThread(target=WavVecFileStream.WavVec, args=(self, SpeechToNLPQueue, str(audio_fname),)) #(target=DeepSpeechFileStream.DeepSpeech, args=(self, SpeechToNLPQueue, str(audio_fname),))
    #
    #     #     self.otheraudiofilename = str(audio_fname)
    #     #     self.SpeechThread.start()
    #     #     print("Other Audio File Speech Thread Started")
    #
    #     # If Text File
    #     # elif(self.ComboBox.currentText() == 'Text File'):
    #     #     text_fname = QFileDialog.getOpenFileName(
    #     #         self, 'Open file', 'c:\\', "Text files (*.txt)")
    #     #     self.SpeechThread = StoppableThread(target=TextSpeechStream.TextSpeech, args=(
    #     #         self, SpeechToNLPQueue, str(text_fname),))
    #     #     self.SpeechThread.start()
    #     #     print("Text File Speech Thread Started")
    #
    #     # If a Hard-coded Audio test file
    #     else:
    #         if(self.GoogleSpeechRadioButton.isChecked()):
    #             self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(
    #                 self, SpeechToNLPQueue, EMSAgentSpeechToNLPQueue,'./Audio_Scenarios/2019_Test/' + str(self.DataSourceBox.currentText()) + '.wav', data_path, audiostream, transcriptStream,))
    #         elif(self.MLSpeechRadioButton.isChecked()):
    #
    #
    #             # Start Whisper module
    #             whispercppcommand = [
    #                 "./stream",
    #                 "-m", # use specific whisper model
    #                 f"models/ggml-{pipeline_config.whisper_model_size}.bin",
    #                 "--threads",
    #                 str(pipeline_config.num_threads),
    #                 "--step",
    #                 str(pipeline_config.step),
    #                 "--length",
    #                 str(pipeline_config.length),
    #                 "--keep",
    #                 str(pipeline_config.keep_ms)
    #             ]
    #             # If a Hard-coded Audio test file, use virtual mic to capture the recording
    #             if(pipeline_config.hardcoded):
    #                 whispercppcommand.append("--capture")
    #
    #             # Start subprocess
    #             self.WhisperSubprocess = subprocess.Popen(whispercppcommand, cwd='EMS_Whisper/')
    #
    #             time.sleep(4)
    #
    #             self.SpeechThread = StoppableThread(
    #                 target=WhisperFileStream.Whisper, args=(self, SpeechToNLPQueue, EMSAgentSpeechToNLPQueue, './Audio_Scenarios/2019_Test/' + str(self.DataSourceBox.currentText()) + '.wav'))
    #
    #         self.SpeechThread.start()
    #         print("Hard-coded Audio File Speech Thread Started")
    #
    #     # ==== Start the Cognitive System Thread
    #     # if(self.CognitiveSystemThread == None):
    #     #     print("Cognitive System Thread Started")
    #     #     self.CognitiveSystemThread = StoppableThread(
    #     #         target=CognitiveSystem.CognitiveSystem, args=(self, SpeechToNLPQueue, FeedbackQueue, data_path, conceptExtractionStream, interventionStream,))
    #     #     # self.CognitiveSystemThread.start()
    #
    #
    #
    #
    #     # ==== Start the Feedback Thread ==== #
    #     # if(self.FeedbackThread == None):
    #     #     print("Feedback Thread Started")
    #     #     self.FeedbackThread = StoppableThread(
    #     #         target=Feedback.FeedbackClient, args=(self, data_path, FeedbackQueue))
    #     #     # self.FeedbackThread.start()

    @pyqtSlot()
    def StopButtonClick(self):
        """
        Stop all streaming processes and reset to pre-start state.
        """
        print("Stop pressed!")
        self.UpdateMsgBox(["Stopping!"])

        # Stop SRT receiver if it exists
        if hasattr(self, 'srt_client') and self.srt_client is not None:
            print("[StopButtonClick] Stopping SRT receiver...")
            self.srt_client.stop()
            self.srt_client = None
            print("[StopButtonClick] SRT receiver stopped")

        # Stop and restart SpeechProcess to kill egosim_stream
        if hasattr(self, 'SpeechProcess') and self.SpeechProcess is not None:
            print("[StopButtonClick] Stopping SpeechProcess...")
            self.SpeechProcess.stop()

            # Give it a moment to clean up
            import time
            time.sleep(0.5)

            # Restart it so it's ready for next Start
            print("[StopButtonClick] Restarting SpeechProcess...")
            self.SpeechProcess = SpeechProcessManager()
            self.SpeechProcess.transcript_ready.connect(self.update_transcript_widget)
            self.SpeechProcess.start()
            print("[StopButtonClick] SpeechProcess restarted")

        # Re-enable UI controls
        self.StartButton.setEnabled(True)
        self.DataSourceBox.setEnabled(True)
        self.ResetButton.setEnabled(True)

        # Update state flags
        self.stopped = 1

        print("[StopButtonClick] Stopped successfully")
        self.UpdateMsgBox(["Stopped!"])
        self.start_video_audio_imu_processes()


    @pyqtSlot()
    def StopButtonClick1(self):
        print("Stop pressed!")
        self.UpdateMsgBox(["Stopping!"])
        self.stopped = 1
        time.sleep(.1)
        self.StartButton.setEnabled(True)
        self.DataSourceBox.setEnabled(True)
        self.ResetButton.setEnabled(True)
        self.SpeechThread.stop()

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
        #self.feedback_client.send_message("Resetting", 'reset')

        try:
            self.WhisperSubprocess.kill()
            subprocess.Popen('pkill stream', shell=False)
        except:
            print("Could not kill Whisper!")
        if(self.WhisperSubprocess != None):
            self.WhisperSubprocess.terminate()
        self.WhisperSubprocess = None

        # if(self.CognitiveSystemThread != None):
        #     SpeechToNLPQueue.put('Kill')
        # #     EMSAgentSpeechToNLPQueue.put('Kill')
        # #     FeedbackQueue.put('Kill')
        # # SpeechToNLPQueue.put('Kill')
        EMSAgentSpeechToNLPQueue.put('Kill')
        # FeedbackQueue.put('Kill')
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
    #@pyqtSlot
    def UpdateSpeechBox(self, input):

        item = input[0]

        if(self.GoogleSpeechRadioButton.isChecked()):
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

        if(self.MLSpeechRadioButton.isChecked()):
            if(item.isFinal):
                self.SpeechBox.clear()

                self.finalSpeechSegmentsSpeech.append(item.transcript)


                # for a in self.finalSpeechSegmentsSpeech:


                #     text = sFinal

                text = self.finalSpeechSegmentsSpeech[-1]

                self.SpeechBox.setText('<b>' + text + '</b>')
                self.SpeechBox.moveCursor(QTextCursor.End)
                self.nonFinalText = ""



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

        self.ProtocolBox.setText("")
        for received in input:
            try:
                protocol = received.protocol
                protocol_confidence = received.protocol_confidence
                if(protocol != None and protocol_confidence != None):
                    protocol_display = str(protocol) + " : " +str(round(protocol_confidence,2))
                    self.ProtocolBox.append(protocol_display)
                    chunkdata.append(protocol_display)
            except Exception as e:
                print("Key error!", e)


        try:
            intervention = received.intervention
            intervention_display = intervention

            self.InterventionBox.setText(intervention_display)
            chunkdata.append(intervention_display)
        except Exception as e:
            print("Key error!", e)


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
            if self.DataSourceBox.currentText() == 'Other Audio File':
                print("\n\nStart Again\n\n")
                self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(
                    self, SpeechToNLPQueue, EMSAgentSpeechToNLPQueue, self.otheraudiofilename, data_path, audiostream, transcriptStream))
            else:
                self.SpeechThread = StoppableThread(target=GoogleSpeechFileStream.GoogleSpeech, args=(
                    self, SpeechToNLPQueue, EMSAgentSpeechToNLPQueue, './Audio_Scenarios/2019_Test/' + str(self.DataSourceBox.currentText()) + '.wav', data_path, audiostream, transcriptStream))
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

    # mp.set_start_method('spawn', force=True)


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

    # smartwatchStream = True
    audiostream = True # harcode audio saving
    videostream = True
    # protocolStream = True
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

    #width = 1920
    #height = 1080
    #width = 1366
    #height = 768
    print("Screen Resolution\nWidth: %s\nHeight: %s" % (width, height))
    Window = MainWindow(width, height)
    #Window.StartButtonClick()
    # monitor =  app.desktop().screenGeometry(1)
    # Window.move(monitor.left(), monitor.top())
    Window.show()

    sys.exit(app.exec_())