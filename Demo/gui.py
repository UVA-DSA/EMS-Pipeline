#!/usr/bin/env python

# ============== Imports ==============

from __future__ import absolute_import, division, print_function
from timeit import default_timer as timer
import sys
import os
import pyaudio
import time
import threading
from six.moves import queue
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import urllib
import scipy.io.wavfile as wav
from deepspeech.model import Model
import speech_recognition as sr
import numpy as np
import io
import audioop
import py_trees
import behaviours as be
from py_trees.blackboard import Blackboard
import ConceptExtract as CE
import matplotlib
import wave
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import random
from playsound import playsound
from pygame import mixer

# ============== Dimension of the GUI ==============

DIM = 1

# ============== Custom Speech to NLP Queue Item Class ==============

class SpeechNLPItem:
        def __init__(self, transcript, isFinal):
            self.transcript = transcript
            self.isFinal = isFinal

# ============== Custom Thread Class with a Stop Flag ==============

class StoppableThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()
        return self._stop_event.is_set()

# ============== Google Speech API ==============

# Microphone and Recording Class

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

class MicrophoneStream(object):
    micSessionCounter = 0
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        MicrophoneStream.micSessionCounter += 1
        self._rate = rate
        self._chunk = chunk
        self.samplesCounter = 0
        self.start_time = time.time()

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True
    
    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        # Run the audio stream asynchronously to fill the buffer object.
        # This is necessary so that the input device's buffer doesn't
        # overflow while the calling thread makes network requests, etc.
        self._audio_stream = self._audio_interface.open(format = pyaudio.paInt16, channels = 1, rate = self._rate, input = True, frames_per_buffer = self._chunk, stream_callback = self._fill_buffer,)
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        # Create Signal Objects
        GoogleSignal = customSignalObject()
        GoogleSignal.signal.connect(Window.StartGoogle)

        MsgSignal = customSignalObject()
        MsgSignal.signal.connect(Window.updateMsgBox)

        GraphSignal = GraphSignalObject()
        GraphSignal.signal.connect(Window.updateAudioGraph)

        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Plot
            signal = b''.join(data)
            signal = np.fromstring(signal, 'Int16')
            #GraphSignal.signal.emit(signal)
            Window.GraphBox.plotAudio(signal)

            # Stop streaming after one minute, create new thread that does recognition
            if time.time() > (self.start_time + (60)):
                GoogleSignal.signal.emit('Mic', 0, 0)
                MsgSignal.signal.emit("API's 1 minute limit reached. Restablishing connection!", 0, 0)
                break

            self.samplesCounter += self._chunk

            '''
            if self.samplesCounter/self._rate > 60:
                GoogleSignal.signal.emit(' ', 0, 0)
                print((self.samplesCounter)/self._rate)
                return
            '''

            if Window.stopped == 1:
                print('Speech Tread Killed')
                Window.StartButton.setEnabled(True)
                return

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            
            yield b''.join(data)

# Google Cloud Speech API Recognition Thread for Microphone
def GoogleSpeech():

    # Create Signal Object
    SpeechSignal = customSignalObject()
    SpeechSignal.signal.connect(Window.updateSpeechBox)

    MsgSignal = customSignalObject()
    MsgSignal.signal.connect(Window.updateMsgBox)

    # Set environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"

    language_code = 'en-US'  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = types.RecognitionConfig(encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz = RATE, language_code = language_code)
    streaming_config = types.StreamingRecognitionConfig(config = config, interim_results = True)

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (types.StreamingRecognizeRequest(audio_content = content) for content in audio_generator)

        try:
            responses = client.streaming_recognize(streaming_config, requests)

            # Signal that speech recognition has started
            #print('Started speech recognition via Google Speech API')
            MsgSignal.signal.emit('Started speech recognition on microphone audio via Google Speech API.\nMicrophone Session counter: ' + str(MicrophoneStream.micSessionCounter), 0, 0)
            
            # Now, put the transcription responses to use.
            num_chars_printed = 0
            responseTimeStamp = time.time()

            for response in responses:
                if not response.results:
                    continue
                # The `results` list is DeepSpeechconsecutive. For streaming, we only care about
                # the first result being considered, since once it's `is_final`, it
                # moves on to considering the next utterance.
                result = response.results[0]
                if not result.alternatives:
                    continue

                # Display the transcription of the top alternative.
                transcript = result.alternatives[0].transcript

                # Display interim results, but with a carriage return at the end of the
                # line, so subsequent lines will overwrite them.
                # If the previous result was longer than this one, we need to print
                # some extra spaces to overwrite the previous result
                overwrite_chars = ' ' * (num_chars_printed - len(transcript))

                if result.is_final:
                    #print(transcript + overwrite_chars)
                    SpeechSignal.signal.emit(transcript, num_chars_printed, 1)
                    QueueItem = SpeechNLPItem(transcript, 1)
                    SpeechToNLPQueue.put(QueueItem)
                    responseTimeStamp = time.time()
                    #MsgSignal.signal.emit('Final response put in queue.', 0, 0)
                    num_chars_printed = 0

                elif not result.is_final:
                    #sys.stdout.write(transcript + overwrite_chars + '\r')
                    #sys.stdout.flush()
                    SpeechSignal.signal.emit(transcript, num_chars_printed, 0)
                    num_chars_printed = len(transcript)
                    # If 6 seconds has passed since the last time a response was put in the queue
                    if time.time() - responseTimeStamp > 6:
                        QueueItem = SpeechNLPItem(transcript, 0)
                        SpeechToNLPQueue.put(QueueItem)
                        responseTimeStamp = time.time()

        except Exception as e:
            MsgSignal.signal.emit('Unable to get response from Google! Timeout or network issues. Please Try again!\n Exception: ' + str(e), 0, 0)
            SpeechSignal.signal.emit('\n', 0, 1)
            Window.StartButton.setEnabled(True)
            sys.exit()

class FileStream(object):
    fileSessionCounter = 0
    position = 0
    """Opens a file stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk, wavefile):
        FileStream.fileSessionCounter += 1
        self._rate = rate
        self._chunk = chunk
        self.filename = wavefile
        self.wf = wave.open(wavefile, 'rb')
        self.previousAudio = self.wf.readframes(CHUNK * FileStream.position)

        self.samplesCounter = 0
        self.start_time = time.time()

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True
    
    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        mixer.init(frequency = RATE)
        # Run the audio stream asynchronously to fill the buffer object.
        # This is necessary so that the input device's buffer doesn't
        # overflow while the calling thread makes network requests, etc.
        self._audio_stream = self._audio_interface.open(format = pyaudio.paInt16, channels = 1, rate = self._rate, input = True, frames_per_buffer = self._chunk, stream_callback = self._fill_buffer,)
        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        #self._buff.put(in_data)
        
        data = self.wf.readframes(CHUNK)
        self._buff.put(data)
        return None, pyaudio.paContinue

    def generator(self):
        # Create Signal Objects
        GoogleSignal = customSignalObject()
        GoogleSignal.signal.connect(Window.StartGoogle)

        MsgSignal = customSignalObject()
        MsgSignal.signal.connect(Window.updateMsgBox)
        
        GraphSignal = GraphSignalObject()
        GraphSignal.signal.connect(Window.updateAudioGraph)

        while not self.closed:

            # Stop streaming after one minute, create new thread that does recognition
            #if time.time() > (self.start_time + (60)):
            #    GoogleSignal.signal.emit('File', 0, 0)
            #    MsgSignal.signal.emit("API's 1 minute limit reached. Exiting speech recognition!", 0, 0)
            #    break

            if self.samplesCounter/self._rate > 60:
                FileStream.position -= 3
                GoogleSignal.signal.emit('File', 0, 0)
                print((self.samplesCounter)/self._rate)
                MsgSignal.signal.emit("API's 1 minute limit reached. Restablishing connection!", 0, 0)
                break
           
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()

            if chunk == '':
                MsgSignal.signal.emit('Transcription of audio file complete!', 0, 0)
                FileStream.position = 0
                #mixer.music.stop()
                Window.StartButton.setEnabled(True)
                Window.ComboBox.setEnabled(True)
                return

            if chunk is None:
                return

            data = [chunk]
            self.samplesCounter += self._chunk
            FileStream.position += 1

            # Plot
            signal = b''.join(data)
            signal = np.fromstring(signal, 'Int16')
            #GraphSignal.signal.emit(signal)
            Window.GraphBox.plotAudio(signal)

            if Window.stopped == 1:
                print('Speech Tread Killed')
                Window.StartButton.setEnabled(True)
                mixer.music.stop()
                FileStream.position = 0
                return

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                    self.samplesCounter += self._chunk
                    FileStream.position += 1
                except queue.Empty:
                    break
 
            yield b''.join(data)

# Google Cloud Speech API Recognition Thread for Microphone
def GoogleSpeechFile(wavefile):

    # Create Signal Object
    SpeechSignal = customSignalObject()
    SpeechSignal.signal.connect(Window.updateSpeechBox)

    MsgSignal = customSignalObject()
    MsgSignal.signal.connect(Window.updateMsgBox)

    # Set environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"

    language_code = 'en-US'  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = types.RecognitionConfig(encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz = RATE, language_code = language_code)
    streaming_config = types.StreamingRecognitionConfig(config = config, interim_results = True)

    with FileStream(RATE, CHUNK, wavefile) as stream:
        audio_generator = stream.generator()
        requests = (types.StreamingRecognizeRequest(audio_content = content) for content in audio_generator)

        try:
            responses = client.streaming_recognize(streaming_config, requests)

            # Signal that speech recognition has started
            #print('Started speech recognition via Google Speech API')
            MsgSignal.signal.emit('Started speech recognition on microphone audio via Google Speech API.\nFile Session Counter: ' + str(FileStream.fileSessionCounter), 0, 0)

            
            # Now, put the transcription responses to use.
            num_chars_printed = 0
            responseTimeStamp = time.time()

            for response in responses:
                if not response.results:
                    continue
                # The `results` list is DeepSpeechconsecutive. For streaming, we only care about
                # the first result being considered, since once it's `is_final`, it
                # moves on to considering the next utterance.
                result = response.results[0]
                if not result.alternatives:
                    continue

                # Display the transcription of the top alternative.
                transcript = result.alternatives[0].transcript

                # Display interim results, but with a carriage return at the end of the
                # line, so subsequent lines will overwrite them.
                # If the previous result was longer than this one, we need to print
                # some extra spaces to overwrite the previous result
                overwrite_chars = ' ' * (num_chars_printed - len(transcript))

                if result.is_final:
                    #print(transcript + overwrite_chars)
                    SpeechSignal.signal.emit(transcript, num_chars_printed, 1)
                    QueueItem = SpeechNLPItem(transcript, 1)
                    SpeechToNLPQueue.put(QueueItem)
                    responseTimeStamp = time.time()
                    #MsgSignal.signal.emit('Final response put in queue.', 0, 0)
                    num_chars_printed = 0

                elif not result.is_final:
                    #sys.stdout.write(transcript + overwrite_chars + '\r')
                    #sys.stdout.flush()
                    SpeechSignal.signal.emit(transcript, num_chars_printed, 0)
                    num_chars_printed = len(transcript)
                    # If 6 seconds has passed since the last time a response was put in the queue
                    if time.time() - responseTimeStamp > 6:
                        QueueItem = SpeechNLPItem(transcript, 0)
                        SpeechToNLPQueue.put(QueueItem)
                        responseTimeStamp = time.time()

        except Exception as e:
            MsgSignal.signal.emit('Unable to get response from Google! Timeout or network issues. Please Try again!\n Exception: ' + str(e), 0, 0)
            mixer.music.stop()
            SpeechSignal.signal.emit('\n', 0, 1)
            Window.StartButton.setEnabled(True)
            sys.exit()
            

# ============== Natural Language Processing ==============

# Action leaves
class InformationGathering(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'InformationGathering'):
        super(InformationGathering, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        '''
        create a ConceptExtractor and initialize the patient status
        the list here is the complete list
        '''
        self.ce1_chunk = CE.ConceptExtractor("Concept_List_1.csv")
        self.ce2_chunk = CE.ConceptExtractor("Concept_List_2.csv")
        self.ce1_full = CE.ConceptExtractor("Concept_List_1.csv")
        self.ce2_full = CE.ConceptExtractor("Concept_List_2.csv")
        self.ce1_chunk.StatusInit()
        self.ce2_chunk.StatusInit()
        self.ce1_full.StatusInit()
        self.ce2_full.StatusInit()
        self.CompleteStatusList = []
        return True
    
    def update(self):

        # Create Signal Object
        NLPSignal = customSignalObject()
        NLPSignal.signal.connect(Window.updateNLPBox)

        blackboard = Blackboard()

        self.ce1_chunk.ConceptExtract(blackboard.text)
        blackboard.concepts = self.ce1_chunk.concepts
        self.ce1_chunk.FirstExtract(blackboard.text)
        blackboard.status1 = self.ce1_chunk.Status
        self.ce2_chunk.concepts = blackboard.concepts
        self.ce2_chunk.SecondExtract(blackboard.text)
        blackboard.status2 = self.ce2_chunk.Status

        self.ce1_full.ConceptExtract(blackboard.fullText)
        blackboard.FullConcepts = self.ce1_full.concepts
        self.ce1_full.FirstExtract(blackboard.fullText)
        blackboard.status1_full = self.ce1_full.Status

        self.ce2_full.concepts = blackboard.FullConcepts
        self.ce2_full.SecondExtract(blackboard.fullText)
        blackboard.status2_full = self.ce2_full.Status

        #self.ce1_chunk.DisplayStatus()
        #self.ce2_chunk.DisplayStatus()
        #self.ce1_full.DisplayStatus()
        #self.ce2_full.DisplayStatus()

        newStatusList1_chunk = self.ce1_chunk.ReturnStatusList()
        for item in newStatusList1_chunk:
            if item not in self.CompleteStatusList:
                self.CompleteStatusList.append(item)
                NLPSignal.signal.emit(item + '\n', 1, 0)

        newStatusList2_chunk = self.ce2_chunk.ReturnStatusList()
        for item in newStatusList2_chunk:
            if item not in self.CompleteStatusList:
                self.CompleteStatusList.append(item)
                NLPSignal.signal.emit(item + '\n', 1, 0)

        newStatusList1_full = self.ce1_full.ReturnStatusList()
        for item in newStatusList1_full:
            if item not in self.CompleteStatusList:
                self.CompleteStatusList.append(item)
                NLPSignal.signal.emit(item + '\n', 1, 0)

        newStatusList2_full = self.ce2_full.ReturnStatusList()
        for item in newStatusList2_full:
            if item not in self.CompleteStatusList:
                self.CompleteStatusList.append(item)
                NLPSignal.signal.emit(item + '\n', 1, 0)

        return py_trees.Status.SUCCESS
    
    def terminate(self, new_status):
        pass
    
# Text Collection
class TextCollection(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'TextCollection'):
        super(TextCollection, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
#level = raw_input("Please type in your certification(EMT,A,I/P): \n")
        level = 'I/P'
        blackboard = Blackboard()
        #blackboard.action = []
        blackboard.level = level
        blackboard.tick_num = 0
        blackboard.protocol = "Universal Patient Care"
        self.text = ['']
        self.sent_text = ['']
        self.nonFinalInsertionIndex = 0

        return True

    def update(self):

        self.sent_text = ['']
        while not SpeechToNLPQueue.empty():
            receivedItem = SpeechToNLPQueue.get()

            if receivedItem.isFinal == 1: # If text in queue is final
                currentSentText = self.sent_text[0]
                self.sent_text[0] = currentSentText[:self.nonFinalInsertionIndex]
                self.sent_text[0] += receivedItem.transcript
                self.sent_text[0] += receivedItem.transcript
                self.nonFinalInsertionIndex = len(self.sent_text[0]) - 1

            elif receivedItem.isFinal == 0:
                currentSentText = self.sent_text[0]
                self.sent_text[0] = currentSentText[:self.nonFinalInsertionIndex]
                self.sent_text[0] += receivedItem.transcript

        blackboard = Blackboard()
        blackboard.text = self.sent_text
        blackboard.fullText = self.text
        
        return py_trees.Status.SUCCESS
   
# NLP Thread
def NLP():
    #NLPSignal = customSignalObject()
    #NLPSignal.signal.connect(Window.updateNLPBox)

    while True:
        if threading.current_thread().stopped():
            print('NLP Thread Killed.')
            break

        #received = SpeechToNLPQueue.get()
        #NLPSignal.signal.emit(received.transcript, 0, 0)

        behaviour_tree.tick_tock(sleep_ms = 50, number_of_iterations = 1, pre_tick_handler = None, post_tick_handler = None)

# ============== GUI ==============

# Custom class for plotting audio
class Graph(QWidget):
    def __init__(self, parent = None):
        super(Graph, self).__init__(parent)
        # a figure instance to plot on
        self.figure = Figure()
        self.max = 6000

        # This is the Canvas Widget that displays the `figure
        self.canvas = FigureCanvas(self.figure)

        self.ax = self.figure.add_subplot(111)
        self.ax.grid(True)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.axis([0, 1600, -self.max, self.max])

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plotAudio(self, data):
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
        #newMax = np.amax(np.absolute(data))
        #if newMax > self.max:
        #    self.max = newMax
        self.ax.axis([0, 1600, -self.max, self.max])
        self.ax.plot(data)
        self.canvas.draw()

# Custom objecta for signalling
class customSignalObject(QObject):
    signal = pyqtSignal(str, int, int)

class GraphSignalObject(QObject):
    signal = pyqtSignal(np.ndarray)

# Main Window of the Application
class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setGeometry(100, 100, int(1780 * DIM), int(800 * DIM))
        self.setWindowTitle('CognitiveEMS Demo')
        self.setWindowIcon(QIcon('UVA.png'))
        self.setFixedSize(int(1780 * DIM), int(800 * DIM))
        self.stopped = 0
        self.SpeechText = ''
        self.NLPText1 = ''
        self.NLPText2 = ''

        # Create main title
        NLPLabel = QLabel(self)
        NLPLabel.move(int(20 * DIM), int(20 * DIM))
        NLPLabel.setText('<font size="6"><b>CognitiveEMS Demo</b></font>')

        # Create label and textbox for speech 
        SpeechLabel = QLabel(self)
        SpeechLabel.move(int(20 * DIM), int(70 * DIM))
        SpeechLabel.setText('<b>Speech Recognition via Google Speech API</b>')

        self.SpeechBox = QTextEdit(self)
        self.SpeechBox.move(int(20 * DIM), int(110 * DIM))
        self.SpeechBox.resize(int(500 * DIM), int(300 * DIM))
        self.SpeechBox.setReadOnly(True)
        self.SpeechBox.setOverwriteMode(True)
        self.SpeechBox.ensureCursorVisible()
        
        # Create label and textbox for NLP
        NLPLabel = QLabel(self)
        NLPLabel.move(int(540 * DIM), int(70 * DIM))
        NLPLabel.setText('<b>Natural Language Processing</b>')

        self.NLPBox = QTextEdit(self)
        self.NLPBox.move(int(540 * DIM), int(110 * DIM))
        self.NLPBox.resize(int(500 * DIM), int(300 * DIM))
        self.NLPBox.setReadOnly(True)

        # Add label, textbox for protcol name, and picture box for protocol tree
        ProtcolLabel = QLabel(self)
        ProtcolLabel.move(int(1060 * DIM), int(70 * DIM))
        ProtcolLabel.setText('<b>Protocol Execution</b>')

        ProtocolPictureBoxBackground = QLabel(self)
        ProtocolPictureBoxBackground.setGeometry(int(1060 * DIM), int(110 * DIM), int(700 * DIM), int(300 * DIM))
        ProtocolPictureBoxBackground.setPixmap(QPixmap('white.png'))

        self.ProtocolPictureBox = QLabel(self)
        self.ProtocolPictureBox.setGeometry(int(1060 * DIM), int(110 * DIM), int(700 * DIM), int(300 * DIM))
        self.ProtocolPictureBox.setPixmap(QPixmap('white.png'))

        if DIM == 1:                                             
            self.ProtocolName = QLineEdit(self)
            self.ProtocolName.move(1205, 65 )
            self.ProtocolName.resize(555, 30)
            self.ProtocolName.setReadOnly(True)
        elif DIM == .7:
            self.ProtocolName = QLineEdit(self)
            self.ProtocolName.move(887, 47)
            self.ProtocolName.resize(347, 21)
            self.ProtocolName.setReadOnly(True)

        # Create label and textbox for recommended action
        ActionBoxLabel = QLabel(self)
        ActionBoxLabel.move(int(1060 * DIM), int(425 * DIM))
        ActionBoxLabel.setText('Recommended Action:')

        if DIM == 1:                                             
            self.ActionBox = QLineEdit(self)
            self.ActionBox.move(1225, 420 )
            self.ActionBox.resize(535, 30)
            self.ActionBox.setReadOnly(True)
        elif DIM == .7:
            self.ActionBox = QLineEdit(self)
            self.ActionBox.move(907, int(420 * DIM))
            self.ActionBox.resize(328, 21)
            self.ActionBox.setReadOnly(True)

        # Audio Options Menu
        self.ComboBox = QComboBox(self)
        self.ComboBox.move(int(20 * DIM), int(420 * DIM))
        self.ComboBox.addItems(['Microphone', 'File 1', 'File 2', 'File 3', 'File 4', 'North Garden 1', 'North Garden 2'])

        # Create a start button in the window
        self.StartButton = QPushButton('   Start   ', self)
        self.StartButton.move(140 + int(20 * DIM), int(420 * DIM))
        self.StartButton.clicked.connect(self.StartButtonClick)

        # Create a stop button in the window
        self.StopButton = QPushButton('   Stop   ', self)
        self.StopButton.move(240 + int(20 * DIM), int(420 * DIM))
        self.StopButton.clicked.connect(self.StopButtonClick)

        # Create label and box for audio graph
        AudioLabel = QLabel(self)
        AudioLabel.move(int(20 * DIM), int(470 * DIM))
        AudioLabel.setText('<b>Audio</b>')

        self.GraphBox = Graph(self)
        self.GraphBox.move(int(10 * DIM), int(490 * DIM))
        self.GraphBox.resize(int(1040 * DIM), int(270 * DIM))

        # Create label and textbox for messages
        MsgBoxLabel = QLabel(self)
        MsgBoxLabel.move(int(1060 * DIM), int(470 * DIM))
        MsgBoxLabel.setText('<b>System Messages</b>')

        self.MsgBox = QTextEdit(self)
        self.MsgBox.move(int(1060 * DIM), int(500 * DIM))
        self.MsgBox.resize(int(700 * DIM), int(180 * DIM))
        self.MsgBox.setReadOnly(True)
        self.MsgBox.setText('Ready to start speech recognition')

        # Add UVA Logo
        #PictureBox = QLabel(self)
        #PictureBox.setGeometry(int(20 * DIM), int(698 * DIM), int(217 * DIM), int(82 * DIM))
        #PictureBox.setPixmap(QPixmap('UVAEngLogo.jpg').scaledToWidth(int(217 * DIM)))

        # Add Link Lab Logo
        PictureBox = QLabel(self)
        PictureBox.setGeometry(int(1191.5 * DIM), int(698 * DIM), int(437 * DIM), int(82 * DIM))
        PictureBox.setPixmap(QPixmap('LinkLabLogo.png').scaledToWidth(int(437 * DIM)))

        # Add Credits Label
        CreditsLabel = QLabel(self)
        CreditsLabel.move(int(20 * DIM), int(770 * DIM))
        CreditsLabel.setText('\n<small><center>Mustafa Hotaki and Sile Shu, May 2018</center></small>')

        # Threads
        self.SpeechThread = StoppableThread(target = GoogleSpeech)
        self.NLPThread = StoppableThread(target = NLP)

    def closeEvent(self, event):
        print('Closing GUI')
        self.stopped = 1
        self.SpeechThread.stop()
        self.NLPThread.stop()
        event.accept()

    @pyqtSlot()
    def StartButtonClick(self):
        print('Start pressed')
        self.MsgBox.setText('Started!')
        if not self.NLPThread.is_alive():
            print('NLP Started')
            self.NLPThread = StoppableThread(target = NLP)
            self.NLPThread.start()

        if not self.SpeechThread.is_alive():
            print('Google Started')
            self.stopped = 0
            self.StartButton.setEnabled(False)

            mixer.init(frequency = RATE)
            if self.ComboBox.currentText() == 'Microphone':
                self.SpeechThread = StoppableThread(target = GoogleSpeech)
            elif self.ComboBox.currentText() == 'File 1':
                self.SpeechThread = StoppableThread(target = GoogleSpeechFile, args=('File1.wav',))
                mixer.music.load('File1.wav')
                if FileStream.position == 0:
                   mixer.music.play()
            elif self.ComboBox.currentText() == 'File 2':
                self.SpeechThread = StoppableThread(target = GoogleSpeechFile, args=('File2.wav',))
                mixer.music.load('File2.wav')
                if FileStream.position == 0:
                       mixer.music.play()
            elif self.ComboBox.currentText() == 'File 3':
                self.SpeechThread = StoppableThread(target = GoogleSpeechFile, args=('File3.wav',))
                mixer.music.load('File3.wav')
                if FileStream.position == 0:
                       mixer.music.play()
            elif self.ComboBox.currentText() == 'File 4':
                self.SpeechThread = StoppableThread(target = GoogleSpeechFile, args=('File4.wav',))
                mixer.music.load('File4.wav')
                if FileStream.position == 0:
                       mixer.music.play()
            elif self.ComboBox.currentText() == 'North Garden 1':
                self.SpeechThread = StoppableThread(target = GoogleSpeechFile, args=('NorthGarden1.wav',))
                mixer.music.load('NorthGarden1.wav')
                if FileStream.position == 0:
                       mixer.music.play()
            elif self.ComboBox.currentText() == 'North Garden 2':
                self.SpeechThread = StoppableThread(target = GoogleSpeechFile, args=('NorthGarden2.wav',))
                mixer.music.load('NorthGarden2.wav')
                if FileStream.position == 0:
                       mixer.music.play()

            self.ComboBox.setEnabled(False)
            self.SpeechThread.start()

               
    @pyqtSlot()
    def StopButtonClick(self):
        print('Stop pressed')
        self.MsgBox.setText('Stopped!')
        self.ComboBox.setEnabled(True)
        self.stopped = 1
        self.SpeechThread.stop()
        #self.NLPThread.stop()

    def updateSpeechBox(self, text, numPrinted, isFinal):
        previousText = self.SpeechBox.toPlainText()
        previousTextMinusPrinted = previousText[:len(previousText) - numPrinted]
        self.SpeechBox.clear()
        if isFinal == 1:
            self.SpeechBox.setText('<b>' + previousTextMinusPrinted + text + ' </b>')
            self.SpeechBox.moveCursor(QTextCursor.End)
        else:
            self.SpeechBox.setText('<b>' + previousTextMinusPrinted + '</b>' + text )
            self.SpeechBox.moveCursor(QTextCursor.End)
       
    def updateNLPBox(self, text, option, extra):
        previousText = self.NLPBox.toPlainText()
        self.NLPBox.setText(previousText + text)
        self.NLPBox.moveCursor(QTextCursor.End)

    def updateProtocolBoxes(self, text, option, extra):
        self.ProtocolName.setText(text)
        self.ProtocolPictureBox.setPixmap(QPixmap(text + '.png').scaledToHeight(int(300 * DIM)))

    def updateMsgBox(self, text, option, extra):
        self.MsgBox.setText(text)

    def updateActionBox(self, text, option, extra):
        self.ActionBox.setText(text)

    def StartGoogle(self, text, option, extra):
        previousText = self.SpeechBox.toPlainText()
        self.SpeechBox.setText('<b>' + previousText + '</b> <br>')
        if text == 'Mic':
            self.SpeechThread = StoppableThread(target = GoogleSpeech)
            self.SpeechThread.start()
        elif text == 'File':
            if self.ComboBox.currentText() == 'File 1':
                self.SpeechThread = StoppableThread(target = GoogleSpeechFile, args=('File1.wav',))
            elif self.ComboBox.currentText() == 'File 2':
                self.SpeechThread = StoppableThread(target = GoogleSpeechFile, args=('File2.wav',))
            elif self.ComboBox.currentText() == 'File 3':
                self.SpeechThread = StoppableThread(target = GoogleSpeechFile, args=('File3.wav',))
            elif self.ComboBox.currentText() == 'File 4':
                self.SpeechThread = StoppableThread(target = GoogleSpeechFile, args=('File4.wav',))
            elif self.ComboBox.currentText() == 'North Garden 1':
                self.SpeechThread = StoppableThread(target = GoogleSpeechFile, args=('NorthGarden1.wav',))
            elif self.ComboBox.currentText() == 'North Garden 2':
                self.SpeechThread = StoppableThread(target = GoogleSpeechFile, args=('NorthGarden2.wav',))
            self.SpeechThread.start()

    def updateAudioGraph(self,signal):
        self.GraphBox.plotAudio(signal)
        
def post_tick_handler(behaviour_tree):
    # Create Signal Objects
    ProtocolSignal = customSignalObject()
    ProtocolSignal.signal.connect(Window.updateProtocolBoxes)
    ActionSignal = customSignalObject()
    ActionSignal.signal.connect(Window.updateActionBox)

    blackboard = Blackboard()
    #print(blackboard.protocol)
    ProtocolSignal.signal.emit(blackboard.protocol, 0, 0)
    if hasattr(blackboard, 'action'):
        #print(blackboard.action)
        ActionSignal.signal.emit(blackboard.action, 0, 0)

# ============== Main ==============

if __name__ == '__main__':

    # Communication: Create thread-safe queues
    SpeechToNLPQueue = queue.Queue()
    DeepSpeechQueue = queue.Queue()

    # NLP/Protocols: build the tree
    root_p = py_trees.composites.Sequence("Root_p")
    selector = py_trees.composites.Selector("ProtocolSelector")
    ChestPain = py_trees.composites.Sequence("ChestPain")
    BleedingControl = py_trees.composites.Sequence("BleedingControl")
    BurnInjury = py_trees.composites.Sequence("BurnInjury")
    BLSCPR = py_trees.composites.Sequence("BLSCPR")
    GTM = py_trees.composites.Sequence("GeneralTraumaManagement")
    RES = py_trees.composites.Sequence("RespiratoryDistress")
    BLSCPR_C = be.BLSCPR_Condition()
    ChestPain_C = be.ChestPain_Condition()
    BC_C = be.Bleeding_Control_Condition()
    BI_C = be.Burn_Injury_Condition()
    GTG_C = be.GeneralTraumaGuideline_Condition()
    RD_C = be.RespiratoryDistress_Condition()
    ARB = be.Arbiter()
    IG = InformationGathering()
    TC = TextCollection()
    GTM.add_children([GTG_C, be.GTM_Action])
    ChestPain.add_children([ChestPain_C, be.ChestPain_Action])
    BleedingControl.add_children([BC_C, be.BC_Action])
    BurnInjury.add_children([BI_C, be.Burn_Action])
    BLSCPR.add_children([BLSCPR_C,be.BLSCPR_Action])
    RES.add_children([RD_C,be.Respiratory_Action])
    root_p.add_children([TC,IG,selector,ARB])
    selector.add_children([BLSCPR,BleedingControl,BurnInjury,GTM,RES,ChestPain])
    behaviour_tree = py_trees.trees.BehaviourTree(root_p)
    behaviour_tree.add_post_tick_handler(post_tick_handler)
    behaviour_tree.setup(15)

    # GUI: Create the main window, show it, and run the app
    app = QApplication(sys.argv)
    Window = MainWindow()
    Window.show()
    sys.exit(app.exec_())

