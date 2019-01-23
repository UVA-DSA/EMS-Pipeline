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

# ============== Internet Connection Check ==============

def connected(host='http://google.com'):
    try:
        urllib.urlopen(host)
        return True
    except:
        return False

# ============== DeepSpeech Constants ==============

BEAM_WIDTH = 500
LM_WEIGHT = 1.75
WORD_COUNT_WEIGHT = 1.00
VALID_WORD_COUNT_WEIGHT = 1.00
N_FEATURES = 26
N_CONTEXT = 9

# ==============  DeepSpeech ==============

# Microphone and Recording Thread
def Recording():

    r = sr.Recognizer() # Recognizer object
    m = sr.Microphone(sample_rate = 16000) # Microphone object

    # Adjust level
    with m as source: 
        #r.adjust_for_ambient_noise(source)
        r.energy_threshold = 350

    #print('Ready to recognize audio.\n')

    while True:
        with m as source: 

            duration = 3; # Minimum duration in seconds
            frames = io.BytesIO()
            seconds_per_buffer = (source.CHUNK + 0.0) / source.SAMPLE_RATE
            elapsed_time = 0

            while True:  # loop for the total number of chunks needed
                buffer = source.stream.read(source.CHUNK)
                if len(buffer) == 0: 
                    break
                elapsed_time += seconds_per_buffer

                if elapsed_time > duration: 

                    if elapsed_time > 10:
                        print('Cutoff at max buffer size')
                        break
                    # Detect silence and cutoof recording
                    else:
                        energy = audioop.rms(buffer, source.SAMPLE_WIDTH)  # energy of the audio signal
                        #print(energy)
                        if energy > (r.energy_threshold * .4) and energy < (r.energy_threshold * 2.5): 
                            break

                frames.write(buffer)

            #print('Done Recording\n')

            frame_data = frames.getvalue()
            frames.close()
            recordedAudio = sr.AudioData(frame_data, source.SAMPLE_RATE, source.SAMPLE_WIDTH)

            # Put in queue
            if not DeepSpeechQueue.full():
                DeepSpeechQueue.put(recordedAudio)

# DeepSpeech Thread
def DeepSpeech():

    # Create Signal objects
    SpeechSignal = customSignalObject()
    SpeechSignal.signal.connect(Window.updateSpeechBox)

    MsgSignal = customSignalObject()
    MsgSignal.signal.connect(Window.updateMsgBox)

    # Signal that models are being loaded
    #print('Loading DeepSpeech models. This may take several minutes.')
    MsgSignal.signal.emit('Loading DeepSpeech models. This may take several minutes.', 0, 0)

    # References to DeepSpeech Models:
    model = 'models/output_graph.pb'
    alphabet = 'models/alphabet.txt'
    lm = 'models/lm.binary'
    trie = 'models/trie'

    # Load DeepSpeech Models
    ds = Model(model, N_FEATURES, N_CONTEXT, alphabet, BEAM_WIDTH)
    ds.enableDecoderWithLM(alphabet, lm, trie, LM_WEIGHT, WORD_COUNT_WEIGHT, VALID_WORD_COUNT_WEIGHT)

    # Start recording thread
    RecordingThread = StoppableThread(target = Recording)
    RecordingThread.start()

    # Signal that speech recognition has started
    #print('Started speech recognition via DeepSpeech')
    MsgSignal.signal.emit('Started speech recognition via DeepSpeech.', 0, 0)


    while True:
        if not DeepSpeechQueue.empty():
            recordedAudio = DeepSpeechQueue.get()
            recordedAudioWave = recordedAudio.get_wav_data()
            fs, audio = wav.read(io.BytesIO(recordedAudioWave))
            audio_length = len(audio) * ( 1 / 16000)
            result = ds.stt(audio, 16000)
            if result:
                #print(result)
                SpeechToNLPQueue.put(result + ' ')
                SpeechSignal.signal.emit(result + ' ', 0, 1)

    RecordingThread.stop()

# ============== Google Speech API ==============

# Microphone and Recording Class

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
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

        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Stop streaming after one minute, create new thread that does recognition
            if time.time() > (self.start_time + (60)):
                GoogleSignal.signal.emit(' ', 0, 0)
                break
                # Maybe work with number of chunks rather than actual time, probably more accurate
                #self.closed = True
                #return

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
            MsgSignal.signal.emit('Started speech recognition on microphone audio via Google Speech API.', 0, 0)
            
            # Now, put the transcription responses to use.
            num_chars_printed = 0

            for response in responses:
                if threading.current_thread().stopped():
                    print('Speech Thread Killed.')
                    Window.StartButton.setEnabled(True)
                    break

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

                if not result.is_final:
                    #sys.stdout.write(transcript + overwrite_chars + '\r')
                    #sys.stdout.flush()
                    SpeechSignal.signal.emit(transcript, num_chars_printed, 0)
                    #QueueItem = SpeechNLPItem(transcript, 0)
                    #SpeechToNLPQueue.put(QueueItem)
                    num_chars_printed = len(transcript)

                else:
                    #print(transcript + overwrite_chars)
                    SpeechSignal.signal.emit(transcript, num_chars_printed, 1)
                    QueueItem = SpeechNLPItem(transcript, 1)
                    SpeechToNLPQueue.put(QueueItem)
                    #SpeechToNLPQueue.put(transcript)
                    num_chars_printed = 0

        except:
            #print('Unable to get response from Google!')
            MsgSignal.signal.emit('Unable to get response from Google! Timeout or network issues. Please Try again!', 0, 0)
            SpeechSignal.signal.emit('\n', 0, 1)
            Window.GoogleButton.setEnabled(True)
            Window.DeepSpeechButton.setEnabled(True)
            Window.StartButton.setEnabled(True)
            sys.exit()

# Stream Audio File
def FileStreamGenerator(wavefile):
    wf = wave.open(wavefile, 'rb')
    data = wf.readframes(CHUNK)
    while data != '':
        yield data
        data = wf.readframes(CHUNK)

# Google Cloud Speech API Recognition Thread for local audio file
def GoogleSpeechFile(file):

    # Create Signal Object
    SpeechSignal = customSignalObject()
    SpeechSignal.signal.connect(Window.updateSpeechBox)

    MsgSignal = customSignalObject()
    MsgSignal.signal.connect(Window.updateMsgBox)

    # Set environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"

    from google.cloud import speech
    from google.cloud.speech import enums
    from google.cloud.speech import types
    client = speech.SpeechClient()

    # In practice, stream should be a generator yielding chunks of audio data.
    stream = FileStreamGenerator(file)
    requests = (types.StreamingRecognizeRequest(audio_content=chunk) for chunk in stream)

    config = types.RecognitionConfig(encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz = 16000, language_code = 'en-US')
    streaming_config = types.StreamingRecognitionConfig(config = config)

    # streaming_recognize returns a generator.C
    responses = client.streaming_recognize(streaming_config, requests)
    MsgSignal.signal.emit('Started speech recognition on audio file via Google Speech API.', 0, 0)
    num_chars_printed = 0


    try:
    
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

            #if not result.is_final:
                #SpeechSignal.signal.emit(transcript, num_chars_printed, 0)
                #num_chars_printed = len(transcript)

            if result.is_final:
                #print(transcript + overwrite_chars)
                SpeechSignal.signal.emit(transcript, num_chars_printed, 1)
                QueueItem = SpeechNLPItem(transcript, 1)
                SpeechToNLPQueue.put(QueueItem)
                num_chars_printed = 0

        # Completed Speech Recongition on audio file
        MsgSignal.signal.emit('Audio file speech recognition complete!', 0, 0)
        Window.GoogleButton.setEnabled(True)
        Window.DeepSpeechButton.setEnabled(True)
        Window.StartButton.setEnabled(True)

    except:
        MsgSignal.signal.emit('Unable to get response from Google! Timeout or network issues. Please Try again!', 0, 0)
        SpeechSignal.signal.emit('\n', 0, 1)
        Window.GoogleButton.setEnabled(True)
        Window.DeepSpeechButton.setEnabled(True)
        Window.StartButton.setEnabled(True)
        #sys.exit()

    
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
        self.elapsed_time = 0
        self.start_time = time.time()
        return True


    def update(self):
        #self.sent_text = []
        #if (not SpeechToNLPQueue.empty()) and len(self.text) <= 100:
        #        received = SpeechToNLPQueue.get()
        #        self.text += received.transcript
        #if not len(self.text) == 0 and len(self.text) > 100 or (time.time() - self.start_time) > 2:
        #    self.sent_text.append(self.text)
        #    #self.text = ''
        #    self.start_time = time.time()

        textInQueue = ''
        while not SpeechToNLPQueue.empty():
            receivedItem = SpeechToNLPQueue.get()
            textInQueue += receivedItem.transcript

        self.sent_text[0] = textInQueue
        self.text[0] += textInQueue
      
        blackboard = Blackboard()
        blackboard.text = self.sent_text
        blackboard.fullText = self.text
        return py_trees.Status.SUCCESS
   
# NLP Thread
def NLP():
    #NLPSignal = customSignalObject()
    #NLPSignal.signal.connect(Window.updateNLPBox)
    #while True:
        #received = SpeechToNLPQueue.get()
        #NLPSignal.signal.emit(received, 1, 0)

    while True:
        if threading.current_thread().stopped():
            print('NLP Thread Killed.')
            break
        behaviour_tree.tick_tock(sleep_ms = 50, number_of_iterations = 1, pre_tick_handler = None, post_tick_handler = None)

# ============== GUI ==============

# Custom object for signalling
class customSignalObject(QObject):
    signal = pyqtSignal(str, int, int)

# Main Window of the Application
class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setGeometry(100, 100, 1780, 800)
        self.setWindowTitle('CognitiveEMS Demo')
        self.setWindowIcon(QIcon('UVA.png'))
        self.setFixedSize(1780, 800)
        self.SpeechText = ''
        self.NLPText1 = ''
        self.NLPText2 = ''

        # Create main title
        NLPLabel = QLabel(self)
        NLPLabel.move(20, 20)
        NLPLabel.setText('CognitiveEMS Demo')

        # Create label and textbox for speech 
        SpeechLabel = QLabel(self)
        SpeechLabel.move(20, 80)
        SpeechLabel.setText('Speech Recognition')

        self.SpeechBox = QTextEdit(self)
        self.SpeechBox.move(20, 110)
        self.SpeechBox.resize(500, 300)
        self.SpeechBox.setReadOnly(True)
        self.SpeechBox.setOverwriteMode(True)
        self.SpeechBox.ensureCursorVisible()
        
        # Create label and textbox for NLP
        NLPLabel = QLabel(self)
        NLPLabel.move(540, 80)
        NLPLabel.setText('Natural Language Processing')

        self.NLPBox = QTextEdit(self)
        self.NLPBox.move(540, 110)
        self.NLPBox.resize(500, 300)
        self.NLPBox.setReadOnly(True)

        # Add label, textbox for protcol name, and picture box for protocol tree
        ProtcolLabel = QLabel(self)
        ProtcolLabel.move(1060, 80)
        ProtcolLabel.setText('Protocol Modeling')

        self.ProtocolName = QLineEdit(self)
        self.ProtocolName.move(1200, 80)
        self.ProtocolName.resize(560, 20)
        self.ProtocolName.setReadOnly(True)

        ProtocolPictureBoxBackground = QLabel(self)
        ProtocolPictureBoxBackground.setGeometry(1060, 110, 700, 300)
        ProtocolPictureBoxBackground.setPixmap(QPixmap('white.png'))

        self.ProtocolPictureBox = QLabel(self)
        self.ProtocolPictureBox.setGeometry(1060, 110, 700, 300)
        self.ProtocolPictureBox.setPixmap(QPixmap('white.png'))

        # Create radio buttons for Google and DeepSpeech
        self.GoogleButton = QRadioButton('Google Speech API', self)
        self.GoogleButton.setChecked(True)
        self.GoogleButton.move(20, 430)

        self.DeepSpeechButton = QRadioButton('DeepSpeech', self)
        self.GoogleButton.setChecked(False)
        self.DeepSpeechButton.move(200, 430)

        # Audio Options Menu
        self.ComboBox = QComboBox(self)
        self.ComboBox.move(20, 470)
        self.ComboBox.addItems(['Microphone', 'File 1', 'File 2', 'File 3', 'File 4', 'North Garden'])

        # Create a start button in the window
        self.StartButton = QPushButton('   Start   ', self)
        self.StartButton.move(150, 470)
        self.StartButton.clicked.connect(self.StartButtonClick)

        # Create a stop button in the window
        self.StopButton = QPushButton('   Stop   ', self)
        self.StopButton.move(250, 470)
        self.StopButton.clicked.connect(self.StopButtonClick)

        # Create label and textbox for messages
        MsgBoxLabel = QLabel(self)
        MsgBoxLabel.move(20, 510)
        MsgBoxLabel.setText('Messages')

        self.MsgBox = QTextEdit(self)
        self.MsgBox.move(20, 540)
        self.MsgBox.resize(1020, 30)
        self.MsgBox.setReadOnly(True)
        self.MsgBox.setText('Ready to start speech recognition')

        # Create label and textbox for recommended action
        ActionBoxLabel = QLabel(self)
        ActionBoxLabel.move(20, 580)
        ActionBoxLabel.setText('Recommended Action')

        self.ActionBox = QTextEdit(self)
        self.ActionBox.move(20, 610)
        self.ActionBox.resize(1020, 30)
        self.ActionBox.setReadOnly(True)

        # Add UVA Logo
        PictureBox = QLabel(self)
        PictureBox.setGeometry(20, 698, 217, 82)
        PictureBox.setPixmap(QPixmap('UVAEngLogo.jpg'))

        # Add Link Lab Logo
        PictureBox = QLabel(self)
        PictureBox.setGeometry(257, 698, 437, 82)
        PictureBox.setPixmap(QPixmap('LinkLabLogo.png'))

        # Threads
        self.SpeechThread = StoppableThread(target = GoogleSpeech)
        self.NLPThread = StoppableThread(target = NLP)

    def closeEvent(self, event):
        print('Closing GUI')
        #self.SpeechThread.stop()
        self.NLPThread.stop()
        event.accept()

    @pyqtSlot()
    def StartButtonClick(self):
        print('Start pressed')
        self.MsgBox.setText('Started')
        if not self.NLPThread.is_alive():
            print('NLP Started')
            self.SpeechThread = StoppableThread(target = NLP)
            self.NLPThread.start()

        if self.GoogleButton.isChecked():
            if not self.SpeechThread.is_alive():
                print('Google Started')
                self.DeepSpeechButton.setEnabled(False)
                self.StartButton.setEnabled(False)
                    
                if self.ComboBox.currentText() == 'Microphone':
                    self.SpeechThread = StoppableThread(target = GoogleSpeech)
                elif self.ComboBox.currentText() == 'File 1':
                    self.SpeechThread = StoppableThread(target = GoogleSpeechFile, args=('File1.wav',))
                elif self.ComboBox.currentText() == 'File 2':
                    self.SpeechThread = StoppableThread(target = GoogleSpeechFile, args=('File2.wav',))
                elif self.ComboBox.currentText() == 'File 3':
                    self.SpeechThread = StoppableThread(target = GoogleSpeechFile, args=('File3.wav',))
                elif self.ComboBox.currentText() == 'File 4':
                    self.SpeechThread = StoppableThread(target = GoogleSpeechFile, args=('File4.wav',))
                elif self.ComboBox.currentText() == 'North Garden':
                    self.SpeechThread = StoppableThread(target = GoogleSpeechFile, args=('NorthGarden.wav',))
            
                self.SpeechThread.start()
               
        elif self.DeepSpeechButton.isChecked():
            if not self.SpeechThread.is_alive():
                print('DeepSpeech Started')
                self.GoogleButton.setEnabled(False)
                self.StartButton.setEnabled(False)
                self.SpeechThread = StoppableThread(target = DeepSpeech)
                self.SpeechThread.start()

    @pyqtSlot()
    def StopButtonClick(self):
        print('Stop pressed')
        self.MsgBox.setText('Stopped!')
        self.SpeechThread.stop()
        self.NLPThread.stop()

    def updateSpeechBox(self, text, numPrinted, isFinal):
        previousText = self.SpeechBox.toPlainText()
        previousTextMinusPrinted = previousText[:len(previousText) - numPrinted]
        self.SpeechBox.clear()
        self.SpeechBox.setText(previousTextMinusPrinted + text)
       
    def updateNLPBox(self, text, option, extra):
        previousText = self.NLPBox.toPlainText()
        self.NLPBox.setText(previousText + text)

        '''
        if option == 1:
            self.NLPText1 = text
        elif option == 2:
            self.NLPText2 = text
        self.NLPBox.setPlainText(self.NLPText1 + '\n' + self.NLPText2)
        '''
    def updateProtocolBoxes(self, text, option, extra):
        self.ProtocolName.setText(text)
        self.ProtocolPictureBox.setPixmap(QPixmap(text + '.png'))

    def updateMsgBox(self, text, option, extra):
        self.MsgBox.setText(text)

    def updateActionBox(self, text, option, extra):
        self.ActionBox.setText(text)

    def StartGoogle(self, text, option, extra):
        self.SpeechBox.append('\n')
        self.SpeechThread = StoppableThread(target = GoogleSpeech)
        self.SpeechThread.start()
        
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

