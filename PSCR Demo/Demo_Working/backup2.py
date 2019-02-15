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
from py_trees.blackboard import Blackboard
import py_trees
from pandas import Series, DataFrame
import pandas as pd
import re
import ConceptExtract as CE

# ============== Custom Thread Class ==============

class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

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

# ============== Microphone and Recording Thread for DeepSpeech ==============

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

# ============== DeepSpeech Thread ==============


def DeepSpeech():

    # Create Signal objects
    SpeechSignal = customSignalObject()
    SpeechSignal.signal.connect(Window.updateSpeechBox)

    MsgSignal = customSignalObject()
    MsgSignal.signal.connect(updateMsgBox)

    # Signal that models are being loaded
    #print('Loading DeepSpeech models. This may take several minutes.')
    MsgSignal.signal.emit('Loading DeepSpeech models. This may take several minutes.', 0)

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
    MsgSignal.signal.emit('Started speech recognition via DeepSpeech.', 0)


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
                SpeechSignal.signal.emit(result + ' ', 0)

    RecordingThread.stop()

# ============== Microphone and Recording Class for Google Speech API ==============

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    start_time = None
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
        self._audio_stream = self._audio_interface.open(format=pyaudio.paInt16, channels = 1, rate = self._rate, input = True, frames_per_buffer = self._chunk, stream_callback=self._fill_buffer,)
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

            #if threading.current_thread().stopped:
                #break

            # Stop streaming after one minute, create new thread that does recognition
            if time.time() > (self.start_time + (60)):
                GoogleSignal.signal.emit(' ', 0)
                break
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

# ============== Google Cloud Speech API Recognition Thread ==============

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
            MsgSignal.signal.emit('Started speech recognition via Google Speech API', 0)
            
            # Now, put the transcription responses to use.
            num_chars_printed = 0

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

                if not result.is_final:
                    #sys.stdout.write(transcript + overwrite_chars + '\r')
                    #sys.stdout.flush()
                    SpeechSignal.signal.emit(transcript, num_chars_printed)
                    num_chars_printed = len(transcript)

                else:
                    #print(transcript + overwrite_chars)
                    SpeechSignal.signal.emit(transcript, num_chars_printed)
                    SpeechToNLPQueue.put(transcript)
                    num_chars_printed = 0

        except:
            #print('Unable to get response from Google!')
            MsgSignal.signal.emit('Unable to get response from Google! Timeout or network issues. Please Try again!', 0)
            SpeechSignal.signal.emit('\n', 0)
            Window.GoogleButton.setEnabled(True)
            Window.DeepSpeechButton.setEnabled(True)
            Window.StartButton.setEnabled(True)
            sys.exit()


# ============== Natural Language Processing Thread ==============

def NLP():
    while not threading.current_thread().stopped():
        behaviour_tree.tick_tock(sleep_ms = 50, number_of_iterations = 1, pre_tick_handler = None, post_tick_handler = None)

# ============== GUI ==============

# Custom object for signalling
class customSignalObject(QObject):
    signal = pyqtSignal(str, int)

# Main Window of the Application
class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setGeometry(500, 100, 1060, 800)
        self.setWindowTitle('CognitiveEMS Demo')
        self.setWindowIcon(QIcon('UVA.png'))
        self.setFixedSize(1060, 800)

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

        # Create radio buttons for Google and DeepSpeech
        self.GoogleButton = QRadioButton('Google Speech API', self)
        self.GoogleButton.setChecked(True)
        self.GoogleButton.move(20, 430)

        self.DeepSpeechButton = QRadioButton('DeepSpeech', self)
        self.GoogleButton.setChecked(False)
        self.DeepSpeechButton.move(200, 430)
        #self.DeepSpeechButton.setEnabled(False)

        # Create label and textbox for messages
        MsgBox = QLabel(self)
        MsgBox.move(20, 470)
        MsgBox.setText('Messages')

        self.MsgBox = QTextEdit(self)
        self.MsgBox.move(20, 500)
        self.MsgBox.resize(1020, 30)
        self.MsgBox.setReadOnly(True)
        self.MsgBox.setText('Ready to start speech recognition')

        # Create a start button in the window
        self.StartButton = QPushButton('   Start   ', self)
        self.StartButton.move(20, 550)
        self.StartButton.clicked.connect(self.StartButtonClick)

        # Create a stop button in the window
        self.StopButton = QPushButton('   Stop   ', self)
        self.StopButton.move(110, 550)
        self.StopButton.clicked.connect(self.StopButtonClick)

        # Create a clear button in the window
        self.ClearButton = QPushButton('   Clear   ', self)
        self.ClearButton.move(200, 550)
        self.ClearButton.clicked.connect(self.ClearButtonClick)

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
        #self.NLPThread.stop()
        event.accept()

    @pyqtSlot()
    def StartButtonClick(self):
        #print('Start pressed')
        if self.GoogleButton.isChecked():
            if not self.SpeechThread.is_alive():
                self.DeepSpeechButton.setEnabled(False)
                self.StartButton.setEnabled(False)
                self.SpeechThread = StoppableThread(target = GoogleSpeech)
                self.SpeechThread.start()
        elif self.DeepSpeechButton.isChecked():
            if not self.SpeechThread.is_alive():
                self.GoogleButton.setEnabled(False)
                self.StartButton.setEnabled(False)
                self.SpeechThread = StoppableThread(target = DeepSpeech)
                self.SpeechThread.start()

        if not self.NLPThread.is_alive():
            print('NLP Started')
            self.SpeechThread = StoppableThread(target = NLP)
            self.NLPThread.start()

    @pyqtSlot()
    def StopButtonClick(self):
        #self.SpeechThread.stop()
        self.NLPThread.stop()
        print('Stop pressed')    

    @pyqtSlot()
    def ClearButtonClick(self):
        print('Clear pressed')
        self.SpeechBox.clear()
        self.NLPBox.clear()
        
    def updateSpeechBox(self, text, numPrinted):
        previousText = self.SpeechBox.toPlainText()
        previousText = previousText[:len(previousText) - numPrinted]
        self.SpeechBox.clear()
        self.SpeechBox.setText(previousText + text)

    def updateMsgBox(self, text, option):
        self.MsgBox.setText(text)

    def updateNLPBox(self, text, option):
        self.NLPBox.insertPlainText(text)

    def StartGoogle(self, text, option):
        self.SpeechBox.append(' ')
        self.SpeechThread = StoppableThread(target = GoogleSpeech)
        self.SpeechThread.start()

# ============== Behaviours ==============

# action leaves
class InformationGathering(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'InformationGathering'):
        super(InformationGathering, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        '''
        create a ConceptExtractor and initialize the patient status
        the list here is the complete list
        '''
        self.ce1 = CE.ConceptExtractor("/home/hotaki/PyQt/Concept_List_1.csv")
        self.ce1.StatusInit()
        self.ce2 = CE.ConceptExtractor("/home/hotaki/PyQt/Concept_List_2.csv")
        self.ce2.StatusInit()
        self.StatusOne = ''
        self.StatusTwo = ''
        return True
    
    def initialise(self):
        pass
       # self.StartTime = time.time()
        
    
    def update(self):

        # Create Signal Object
        NLPSignal = customSignalObject()
        NLPSignal.signal.connect(Window.updateNLPBox)

        blackboard = Blackboard()
        self.ce1.ConceptExtract(blackboard.text)
        blackboard.concepts = self.ce1.concepts
        self.ce1.FirstExtract(blackboard.text)
        blackboard.status1 = self.ce1.Status
        self.ce2.concepts = blackboard.concepts
        self.ce2.SecondExtract(blackboard.text)
        blackboard.status2 = self.ce2.Status

        newStatusOne = self.ce1.ReturnStatus()
        newStatusTwo = self.ce2.ReturnStatus()

        if newStatusOne != self.StatusOne:
            NLPSignal.signal.emit(newStatusOne, 1)
            self.StatusOne = newStatusOne

        if newStatusTwo != self.StatusTwo:
            NLPSignal.signal.emit(newStatusTwo, 2)
            self.StatusTwo = newStatusTwo
        
        return py_trees.Status.SUCCESS
    
    def terminate(self, new_status):
        pass
    

class TextCollection(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'TextCollection'):
        super(TextCollection, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        #level = raw_input("Please type in your certification(EMT,A,I/P): \n")
        level = 'I/P'
        blackboard = Blackboard()
        blackboard.action = []
        blackboard.level = level
        blackboard.tick_num = 0
        blackboard.protocol = "Universal Patient Care"
        self.text = ''
        self.sent_text = []
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        self.sent_text = []
        
        if (not SpeechToNLPQueue.empty()) and len(self.text) <= 100:
            self.text += SpeechToNLPQueue.get()
        if not len(self.text) == 0 and len(self.text) > 100 or (time.time() - self.start_time) > 2:
            self.sent_text.append(self.text)
            self.text = ''
            self.start_time = time.time()
        
        blackboard = Blackboard()
        blackboard.text = self.sent_text
        return py_trees.Status.SUCCESS
    
    def terminate(self, new_status):
        pass

# protocols' condition
class BLSCPR_Condition(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'BLSCPR_Condition'):
        super(BLSCPR_Condition, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        self.ce = CE.ConceptExtractor("BLSCPR.csv")
        self.ce.StatusInit()
        blackboard.BLSCPR_Status = self.ce.Status
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        self.ce.concepts = blackboard.concepts
        self.ce.SecondExtract(blackboard.text)
        blackboard.BLSCPR_Status = self.ce.Status
        if blackboard.status1['pulse'].binary == False and \
        blackboard.status1['breath'].binary == False:
            blackboard.Protocol = "BLS CPR"
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE
        
class Bleeding_Control_Condition(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Bleeding_Control_Condition'):
        super(Bleeding_Control_Condition, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        self.ce = CE.ConceptExtractor("BleedingControl.csv")
        self.ce.StatusInit()
        blackboard.BC_Status = self.ce.Status
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        self.ce.concepts = blackboard.concepts
        self.ce.SecondExtract(blackboard.text)
        blackboard.BC_Status = self.ce.Status
        if blackboard.status1['bleed'].binary == True:
            blackboard.Protocol = "Bleeding Control"
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE

'''
class AlteredMentalStatus_Condition(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'AlteredMentalStatus_Condition'):
        super(AlteredMentalStatus_Condition, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        self.ce = CE.ConceptExtractor("AMS.csv")
        self.ce.StatusInit()
        blackboard.AMS_Status = self.ce.Status
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        self.ce.concepts = blackboard.concepts
        self.ce.SecondExtract(blackboard.text)
        blackboard.AMS_Status = self.ce.Status
        if blackboard.status1['breath'].binary == True and blackboard.status1['conscious'].binary == False \
        and blackboard.status1['pulse'].binary == True:
            blackboard.protocol = "Altered Mental Status"
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE
'''

class Burn_Injury_Condition(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Burn_Injury_Condition'):
        super(Burn_Injury_Condition, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        self.ce = CE.ConceptExtractor("Burn.csv")
        self.ce.StatusInit()
        blackboard.Burn_Status = self.ce.Status
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        self.ce.concepts = blackboard.concepts
        if blackboard.fentanyl[0] > 0:
            self.ce.SpecificInit('fentanyl')
        if blackboard.ondansetron[0] > 0:
            self.ce.SpecificInit('ondansetron')
        if blackboard.ketamine[0] > 0:
            self.ce.SpecificInit('ketamine')
        self.ce.SecondExtract(blackboard.text)
        blackboard.Burn_Status = self.ce.Status
        if blackboard.status1['wound'].binary == True and blackboard.status2['burn'].binary == True:
            blackboard.protocol = "Burn Injury"
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE
        
class GeneralTraumaGuideline_Condition(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'GeneralTraumaGuideline_Condition'):
        super(GeneralTraumaGuideline_Condition, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        self.ce = CE.ConceptExtractor("GeneralTrauma.csv")
        self.ce.StatusInit()
        blackboard.GT_Status = self.ce.Status
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        self.ce.concepts = blackboard.concepts
        if blackboard.fentanyl[0] > 0:
            self.ce.SpecificInit('fentanyl')
        if blackboard.ondansetron[0] > 0:
            self.ce.SpecificInit('ondansetron')
        if blackboard.ketamine[0] > 0:
            self.ce.SpecificInit('ketamine')
        self.ce.SecondExtract(blackboard.text)
        blackboard.GT_Status = self.ce.Status
        if blackboard.status1['wound'].binary == True and blackboard.status2['burn'].binary == False:
            blackboard.protocol = "General Trauma Guideline"
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE
        
class RespiratoryDistress_Condition(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'RespiratoryDistress_Condition'):
        super(RespiratoryDistress_Condition, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        self.ce = CE.ConceptExtractor("Respiratory.csv")
        self.ce.StatusInit()
        blackboard.Res_Status = self.ce.Status
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        self.ce.concepts = blackboard.concepts
        self.ce.SecondExtract(blackboard.text)
        blackboard.Res_Status = self.ce.Status
        if blackboard.status1['breath'].binary == False:
            blackboard.protocol = "Respiratory Distress"
            return py_trees.Status.SUCCESS
        if len(blackboard.status1['breath'].value) > 0:
            if int(blackboard.status1['breath'].value) > 30 or int(blackboard.status1['breath'].value) < 10:
                blackboard.protocol = "Respiratory Distress"
                return py_trees.Status.SUCCESS
        if len(blackboard.status1['spo2'].value) > 0:
            if int(blackboard.status1['spo2'].value.replace('%','')) < 70:
                blackboard.protocol = "Respiratory Distress"
                return py_trees.Status.SUCCESS
        if len(blackboard.status1['etco2'].value) >0:
            if int(blackboard.status1['etco2'].value) > 45 or int(blackboard.status1['etco2'].value) < 35:
                blackboard.protocol = "Respiratory Distress"
                return py_trees.Status.SUCCESS
        return py_trees.Status.FAILURE
        
class ChestPain_Condition(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'ChestPain_Condition'):
        super(ChestPain_Condition, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        self.ce = CE.ConceptExtractor("ChestPain.csv")
        self.ce.StatusInit()
        blackboard.CP_Status = self.ce.Status
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        self.ce.concepts = blackboard.concepts
        if blackboard.fentanyl[0] > 0:
            self.ce.SpecificInit('fentanyl')
        if blackboard.ondansetron[0] > 0:
            self.ce.SpecificInit('ondansetron')
        if blackboard.nitroglycerin[0] > 0:
            self.ce.SpecificInit('nitroglycerin')
        self.ce.SecondExtract(blackboard.text)
        blackboard.CP_Status = self.ce.Status
        if blackboard.status1['pain'].binary == True and \
        ('chest' in blackboard.status1['pain'].content or \
         'chest' in blackboard.status2['pain region'].content or\
         'chest' in blackboard.status2['pain region'].value):
            blackboard.protocol = "Chest Pain"
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE
            
# ChestPain
class ECG(py_trees.behaviour.Behaviour):
    def __init__(self, name = '12-Lead ECG'):
        super(ECG, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.CP_Status['12-lead ecg'].binary == True:
            blackboard.action.append(self.name)
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.SUCCESS
    
class MI(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'MI?'):
        super(MI, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.CP_Status['mi'].binary == True:
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE
    
class Transport(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Transport'):
        super(Transport, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        blackboard.action.append(self.name)
        return py_trees.Status.SUCCESS
    
class Aspirin(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Aspirin'):
        super(Aspirin, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.CP_Status['aspirin'].binary == True:
            blackboard.action.append(self.name)
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.SUCCESS

class Advanced(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Advanced EMT?'):
        super(Advanced, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        self.level = blackboard.level
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if self.level == "A" or self.level == "I/P":
            blackboard.action.append('IV access')
            return py_trees.Status.SUCCESS
        else:
            #print "don't have enough certification"
            return py_trees.Status.FAILURE

class IV(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'IV access'):
        super(IV, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.CP_Status['iv/io/vascular access'].binary == True or\
        blackboard.Burn_Status['iv/io/vascular access'].binary == True or\
        blackboard.GT_Status['iv/io/vascular access'].binary == True:
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE
    
class Nausea(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Nausea/Vommiting?'):
        super(Nausea, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.CP_Status['nausea'].binary == True or \
        blackboard.Burn_Status['nausea'].binary == True or \
        blackboard.GT_Status['nausea'].binary == True:
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE

class ondansetron(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'ondansetron'):
        super(ondansetron, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        blackboard.ondansetron = [0,0.]
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        # usage detection
        if blackboard.CP_Status['ondansetron'].binary == True or\
        blackboard.Burn_Status['ondansetron'].binary == True or\
        blackboard.GT_Status['ondansetron'].binary == True:
            blackboard.ondansetron[0] += 1
            blackboard.ondansetron[1] = time.time()
        if blackboard.ondansetron[0] == 0:
            blackboard.action.append('ondansetron 4mg ODT')
            return py_trees.Status.SUCCESS
        elif blackboard.ondansetron[0] == 1 and (time.time() - blackboard.ondansetron[1] > 600):
            blackboard.action.append('ondansetron 4mg ODT')
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.SUCCESS
    
class IP(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'I/P?'):
        super(IP, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        self.level = blackboard.level
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        if self.level == "I/P":
            return py_trees.Status.SUCCESS
        else:
            #print "don't have enough certification"
            return py_trees.Status.FAILURE

    
class Fentanyl_IV(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Fentanyl_IV'):
        super(Fentanyl_IV, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        blackboard.fentanyl = [0,0.]
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        # fentanyl usage detection
        if blackboard.CP_Status['fentanyl'].binary == True or\
        blackboard.Burn_Status['fentanyl'].binary == True or\
        blackboard.GT_Status['fentanyl'].binary == True:
            blackboard.fentanyl[0] += 1
            blackboard.fentanyl[1] = time.time()
        # fentanyl usage under different conditions
        if blackboard.fentanyl[0] == 0:
            if len(blackboard.status1['age'].value) > 0:
                if int(blackboard.status1['age'].value) >= 65:
                    blackboard.action.append('0.5mg/kg Fentanyl IV')
                elif int(blackboard.status1['age'].value) < 65:
                    blackboard.action.append('1mg/kg Fentanyl IV')
            else:
                blackboard.action.append('1mg/kg Fentanyl IV')
            return py_trees.Status.SUCCESS
        elif blackboard.fentanyl[0] == 1 and (time.time() - blackboard.fentanyl[1] > 600):
            if len(blackboard.status1['age'].value) > 0:
                if int(blackboard.status1['age'].value) >= 65:
                    blackboard.action.append('0.5mg/kg Fentanyl IV')
                elif int(blackboard.status1['age'].value) < 65:
                    blackboard.action.append('1mg/kg Fentanyl IV')
            else:
                blackboard.action.append('1mg/kg Fentanyl IV')
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.SUCCESS
        
class Fentanyl_IN(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Fentanyl_IN'):
        super(Fentanyl_IN, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        blackboard.fentanyl = [0,0.]
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        # fentanyl usage detection
        if blackboard.CP_Status['fentanyl'].binary == True or\
        blackboard.Burn_Status['fentanyl'].binary == True or\
        blackboard.GT_Status['fentanyl'].binary == True:
            blackboard.fentanyl[0] += 1
            blackboard.fentanyl[1] = time.time()
        # fentanyl usage under different conditions
        if blackboard.fentanyl[0] == 0 or \
        (blackboard.fentanyl[0] == 1 and (time.time() - blackboard.fentanyl[1] > 600)):
            blackboard.action.append('2mg/kg Fentanyl IN')
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.SUCCESS
    
class Nitro(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'nitroglycerin'):
        super(Nitro, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        blackboard.nitroglycerin = [0,0.]
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        # usage detection
        if blackboard.CP_Status['nitroglycerin'].binary == True:
            blackboard.nitroglycerin[0] += 1
            blackboard.nitroglycerin[1] = time.time()
        if len(blackboard.status1['blood pressure'].value) > 0:
            if int(blackboard.status1['blood pressure'].value.split('/')[0]) < 100:
                if blackboard.nitroglycerin[0] == 0:
                    blackboard.action.append("nitroglycerin 0.4mg")
                    return py_trees.Status.SUCCESS
                elif blackboard.nitroglycerin[0] > 0 and\
                ((time.time() - blackboard.nitroglycerin[1]) > 300):
                    blackboard.action.append("nitroglycerin 0.4mg")
                    return py_trees.Status.SUCCESS
        return py_trees.Status.SUCCESS
    
class NotStarted(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'NS'):
        super(NotStarted, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        return py_trees.Status.SUCCESS
		
# General Trauma
class Evisceration(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Evisceration'):
        super(Evisceration, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.GT_Status['evisceration'].binary == True:
            blackboard.action.append("cover with moist sterile dressing")
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE
        
class OpenChestWound(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'OpenChestWound'):
        super(OpenChestWound, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.GT_Status['open chest wound'].binary == True:
            blackboard.action.append("cover with occlusive dressing")
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE
        
class shock(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'shock'):
        super(shock, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.GT_Status['shock'].binary == True:
            blackboard.action.append("needle chest decompression")
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.SUCCESS
			

# Respiratory
class MDI(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'MDI'):
        super(MDI, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.Res_Status['mdi'].binary == True:
            blackboard.action.append(self.name)
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.SUCCESS
        
class CPAP(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'CPAP'):
        super(CPAP, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.Res_Status['cpap'].binary == True:
            blackboard.action.append(self.name)
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.SUCCESS
        
class AdministerOxygen(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'AdministerOxygen'):
        super(AdministerOxygen, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.Res_Status['administer oxygen'].binary == True:
            blackboard.action.append(self.name)
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.SUCCESS
        
class AlbuterolSulfate(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'AlbuterolSulfate'):
        super(AlbuterolSulfate, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.Res_Status['albuterol sulfate'].binary == True:
            blackboard.action.append(self.name)
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.SUCCESS
        
class Methylprednisolone(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Methylprednisolone'):
        super(Methylprednisolone, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.Res_Status['methylprednisolone'].binary == True:
            blackboard.action.append(self.name)
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.SUCCESS
			
# Burn
class ElectricalBurn(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Electrical Burn?'):
        super(ElectricalBurn, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if 'electric' in blackboard.status2['burn'].value or\
        'electric' in blackboard.status2['burn'].content:
            blackboard.action.append("Search for additional injury")
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE

class ChemicalBurn(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Chemical Burn?'):
        super(ChemicalBurn, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if 'chemical' in blackboard.status2['burn'].value or\
        'chemical' in blackboard.status2['burn'].content:
            blackboard.action.append("Irrigate with water/brush the powder off")
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE

class ThermalBurn(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Thermal Burn?'):
        super(ThermalBurn, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if 'thermal' in blackboard.status2['burn'].value or\
        'thermal' in blackboard.status2['burn'].content:
            blackboard.action.append("Assess for carbon monoxide exposure")
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.SUCCESS
        
        
class SpiRestriction(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Spinal motion restriction'):
        super(SpiRestriction, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.Burn_Status['spinal motion restriction'].binary == True or\
        blackboard.AMS_Status['spinal motion restriction'].binary == True:
            blackboard.action.append(self.name)
        return py_trees.Status.SUCCESS

class Ketamine(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Ketamine'):
        super(Ketamine, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        blackboard.ketamine = [0,0]
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        # usage detection
        if blackboard.Burn_Status['ketamine'].binary == True or\
        blackboard.GT_Status['ketamine'].binary == True:
            blackboard.ketamine[0] += 1
            blackboard.ketamine[1] = time.time()
        if blackboard.fentanyl[0] >= 2 and blackboard.status['pain'].binary == True:
            if blackboard.ketamine[0] == 0:
                blackboard.action.append("ketamine 0.5mg/kg IV")
                return py_trees.Status.SUCCESS
            elif blackboard.ketamine[0] == 1 and (time.time() - blackboard.ketamine[1] > 600):
                blackboard.action.append("ketamine 0.5mg/kg IV")
                return py_trees.Status.SUCCESS
            else:
                return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.SUCCESS
			
# Bleeding Control
class Unstable(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'unstable?'):
        super(Unstable, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.BC_Status['unstable'].binary == True:
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE

class Move(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Move to tourniquet'):
        super(Move, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        blackboard.action.append(self.name)
        return py_trees.Status.SUCCESS
    
class Pressure(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Direct pressure to the bleeding site'):
        super(Pressure, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        blackboard.action.append(self.name)
        return py_trees.Status.SUCCESS
    
class Control(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Controlled?'):
        super(Control, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.status1['bleed'].binary == True:
            return py_trees.Status.FAILURE
        else:
            return py_trees.Status.SUCCESS
			
# BLSCPR
class DNR(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'DNR?'):
        super(DNR, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.BLSCPR_Status['dnr'].binary == True:
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE
        
class CPR(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'CPR'):
        super(CPR, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.BLSCPR_Status['cpr'].binary == True or \
        blackboard.BLSCPR_Status['aed'].binary == True or\
        blackboard.BLSCPR_Status['defibrillation'].binary == True:
            blackboard.action.append(self.name)
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE

class AirwayManagement(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'AirwayManagement'):
        super(AirwayManagement, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.BLSCPR_Status['bvm'].binary == True or\
        blackboard.BLSCPR_Status['opa'].binary == True:
            blackboard.action.append(self.name)
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE
			
# sub-trees
BLSCPR_Action = py_trees.composites.Selector("BLSCPR_Action")
BLSCPR_S1 = py_trees.composites.Sequence("Sequence_1")
BLSCPR_S2 = py_trees.composites.Sequence("Sequence_2")
BLSCPR_NS_1 = NotStarted()
BLSCPR_NS_2 = NotStarted()
BLSCPR_DNR = DNR()
BLSCPR_CPR = CPR()
BLSCPR_AM = AirwayManagement()
BLSCPR_Action.add_children([BLSCPR_S1,BLSCPR_S2,BLSCPR_NS_1])
BLSCPR_S1.add_children([BLSCPR_DNR,BLSCPR_NS_2])
BLSCPR_S2.add_children([BLSCPR_CPR,BLSCPR_AM])

BC_Action = py_trees.composites.Sequence("BleedControl_Action")
BC_S1 = py_trees.composites.Sequence("Sequence_1")
BC_S2 = py_trees.composites.Sequence("Sequence_2")
BC_SE = py_trees.composites.Selector("Selector")
BC_NS = NotStarted()
BC_US = Unstable()
BC_Move_1 = Move()
BC_Move_2 = Move()
BC_Pressure = Pressure()
BC_Control = Control()
BC_Action.add_children([BC_SE,BC_NS])
BC_SE.add_children([BC_S1,BC_S2])
BC_S1.add_children([BC_US,BC_Move_1])
BC_S2.add_children([BC_Pressure,BC_Control,BC_Move_2])

# composite
Burn_Action = py_trees.composites.Sequence("Burn_Action")
B_S1 = py_trees.composites.Sequence("Sequence")
B_S2 = py_trees.composites.Sequence("Sequence")
B_S3 = py_trees.composites.Sequence("Sequence")
B_S4 = py_trees.composites.Sequence("Sequence")
B_SE1 = py_trees.composites.Selector("Selector")
B_SE2 = py_trees.composites.Selector("Selector")
B_SE3 = py_trees.composites.Selector("Selector")
B_SE4 = py_trees.composites.Selector("Selector")
# behaviour
B_NS_1 = NotStarted()
B_NS_2 = NotStarted()
B_NS_3 = NotStarted()
B_SPI = SpiRestriction()
B_EB = ElectricalBurn()
B_CB = ChemicalBurn()
B_TB = ThermalBurn()
B_IV = IV()
B_NAU_1 = Nausea()
B_NAU_2 = Nausea()
B_OD_1 = ondansetron()
B_OD_2 = ondansetron()
B_IP = IP()
B_FEN_IV = Fentanyl_IV()
B_FEN_IN = Fentanyl_IN()
B_A = Advanced()
B_KET = Ketamine()
# tree
Burn_Action.add_children([B_SPI,B_SE1,B_A,B_IP,B_SE2])
B_SE1.add_children([B_EB,B_CB,B_TB,B_NS_1])
B_SE2.add_children([B_S1,B_S2])
B_S1.add_children([B_IV,B_FEN_IV,B_KET,B_SE3])
B_SE3.add_children([B_S3,B_NS_2])
B_S3.add_children([B_NAU_1,B_OD_1])
B_S2.add_children([B_FEN_IN,B_SE4])
B_S4.add_children([B_NAU_2,B_OD_2])
B_SE4.add_children([B_S4,B_NS_3])

Respiratory_Action = py_trees.composites.Sequence("Respiratory_Action")
R_S = py_trees.composites.Sequence("Sequence")
R_SE = py_trees.composites.Selector("Selector")
R_MDI = MDI()
R_CPAP = CPAP()
R_AO = AdministerOxygen()
R_AS = AlbuterolSulfate()
R_A = Advanced()
R_IV = IV()
R_METH = Methylprednisolone()
R_NS = NotStarted()
Respiratory_Action.add_children([R_MDI,R_CPAP,R_AO,R_AS,R_SE])
R_S.add_children([R_A,R_IV,R_METH])
R_SE.add_children([R_S,R_NS])

# composites
GTM_Action = py_trees.composites.Sequence("GeneralTraumaManagement_Action")
GT_S1 = py_trees.composites.Sequence("Sequence_1")
GT_S2 = py_trees.composites.Sequence("Sequence_2")
GT_S3 = py_trees.composites.Sequence("Sequence_3")
GT_S4 = py_trees.composites.Sequence("Sequence_4")
GT_S5 = py_trees.composites.Sequence("Sequence_5")
GT_SE1 = py_trees.composites.Selector("Selector_1")
GT_SE2 = py_trees.composites.Selector("Selector_2")
GT_SE3 = py_trees.composites.Selector("Selector_3")
GT_SE4 = py_trees.composites.Selector("Selector_4")
# behaviours
GT_SPI = SpiRestriction()
GT_Evi = Evisceration()
GT_OCW = OpenChestWound()
GT_Sh = shock()
GT_A = Advanced()
GT_IV = IV()
GT_NAU_1 = Nausea()
GT_NAU_2 = Nausea()
GT_OD_1 = ondansetron()
GT_OD_2 = ondansetron()
GT_IP = IP()
GT_FEN_IV = Fentanyl_IV()
GT_FEN_IN = Fentanyl_IN()
GT_NS_1 = NotStarted()
GT_NS_2 = NotStarted()
GT_NS_3 = NotStarted()
GT_KET = Ketamine()
# tree
GTM_Action.add_children([GT_SPI,GT_SE1,GT_S1])
GT_SE1.add_children([GT_Evi,GT_OCW,GT_NS_1])
GT_S1.add_children([GT_A,GT_Sh,GT_IP,GT_SE2])
GT_SE2.add_children([GT_S2,GT_S3])
GT_S2.add_children([GT_IV,GT_FEN_IV,GT_KET,GT_SE3])
GT_SE3.add_children([GT_S4,GT_NS_2])
GT_S4.add_children([GT_NAU_1,GT_OD_1])
GT_S3.add_children([GT_FEN_IN,GT_SE4])
GT_SE4.add_children([GT_S5,GT_NS_3])
GT_S5.add_children([GT_NAU_2,GT_OD_2])

# chestpain action
# composites
ChestPain_Action = py_trees.composites.Selector("ChestPain_Action")
CP_S1 = py_trees.composites.Sequence("Sequence_1")
CP_S2 = py_trees.composites.Sequence("Sequence_2")
CP_S3 = py_trees.composites.Sequence("Sequence_3")
CP_S4 = py_trees.composites.Sequence("Sequence_4")
CP_S5 = py_trees.composites.Sequence("Sequence_5")
CP_S6 = py_trees.composites.Sequence("Sequence_6")
CP_S7 = py_trees.composites.Sequence("Sequence_7")
CP_SE1 = py_trees.composites.Selector("Selector_1")
CP_SE2 = py_trees.composites.Selector("Selector_2")
CP_SE3 = py_trees.composites.Selector("Selector_3")
CP_SE4 = py_trees.composites.Selector("Selector_4")
# behaviours
CP_ECG = ECG()
CP_MI = MI()
CP_TRANS = Transport()
CP_AS = Aspirin()
CP_NI = Nitro()
CP_A = Advanced()
CP_IV = IV()
CP_NAU_1 = Nausea()
CP_NAU_2 = Nausea()
CP_NAU_3 = Nausea()
CP_OD_1 = ondansetron()
CP_OD_2 = ondansetron()
CP_OD_3 = ondansetron()
CP_IP = IP()
CP_FEN_IV = Fentanyl_IV()
CP_FEN_IN = Fentanyl_IN()
CP_NS_1 = NotStarted()
CP_NS_2 = NotStarted()
CP_NS_3 = NotStarted()
# build sub-tree: chestpain action
ChestPain_Action.add_children([CP_S1,CP_S2])
CP_S1.add_children([CP_ECG,CP_MI,CP_TRANS])
CP_S2.add_children([CP_AS,CP_NI,CP_A,CP_SE1,CP_IP,CP_SE2])
CP_SE1.add_children([CP_S3,CP_NS_1])
CP_S3.add_children([CP_NAU_1,CP_OD_1])
CP_SE2.add_children([CP_S4,CP_S6])
CP_S4.add_children([CP_IV,CP_FEN_IV,CP_SE3])
CP_SE3.add_children([CP_S5,CP_NS_2])
CP_S5.add_children([CP_NAU_2,CP_OD_2])
CP_S6.add_children([CP_FEN_IN,CP_SE4])
CP_SE4.add_children([CP_S7,CP_NS_3])
CP_S7.add_children([CP_NAU_3,CP_OD_3])




# ============== Main ==============

if __name__ == '__main__':

    # Communication: Create thread-safe queues
    SpeechToNLPQueue = queue.Queue()
    DeepSpeechQueue = queue.Queue()

    # NLP/Protocols: build the treeimport behaviours as be
    IG = InformationGathering()
    TC = TextCollection()
    root_p = py_trees.composites.Sequence("Root_p")
    selector = py_trees.composites.Selector("ProtocolSelector")
    ChestPain = py_trees.composites.Sequence("ChestPain")
    BleedingControl = py_trees.composites.Sequence("BleedingControl")
    BurnInjury = py_trees.composites.Sequence("BurnInjury")
    BLSCPR = py_trees.composites.Sequence("BLSCPR")
    GTM = py_trees.composites.Sequence("GeneralTraumaManagement")
    RES = py_trees.composites.Sequence("RespiratoryDistress")
    BLSCPR_C = BLSCPR_Condition()
    ChestPain_C = ChestPain_Condition()
    BC_C = Bleeding_Control_Condition()
    BI_C = Burn_Injury_Condition()
    GTG_C = GeneralTraumaGuideline_Condition()
    RD_C = RespiratoryDistress_Condition()

    GTM.add_children([GTG_C, GTM_Action])
    ChestPain.add_children([ChestPain_C, ChestPain_Action])
    BleedingControl.add_children([BC_C, BC_Action])
    BurnInjury.add_children([BI_C, Burn_Action])
    BLSCPR.add_children([BLSCPR_C,BLSCPR_Action])
    RES.add_children([RD_C,Respiratory_Action])
    root_p.add_children([TC,IG,selector])
    selector.add_children([BLSCPR,BleedingControl,BurnInjury,GTM,RES,ChestPain])
    behaviour_tree = py_trees.trees.BehaviourTree(root_p)
    behaviour_tree.setup(15)
   
    # GUI: Create the main window, show it, and run the app
    app = QApplication(sys.argv)
    Window = MainWindow()
    Window.show()
    sys.exit(app.exec_())

