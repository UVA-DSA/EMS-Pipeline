#!/usr/bin/env python

# ============== Imports ==============

from __future__ import division

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

# ============== Internet Connection Check ==============

def connected(host='http://google.com'):
    try:
        urllib.urlopen(host)
        return True
    except:
        return False

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
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

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
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Stop streaming after one minute
            #if time.time() > (self.start_time + (10)):
                #print('One minute limit reached')
                #Window.StartButton.setEnabled(True)
                #Window.SpeechThread.start()
                #break
    		
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


# ============== GUI ==============

# Custom object for signalling
class customSignalObject(QObject):
    signal = pyqtSignal(str, int)

# Main Window of the Application
class MainWindow(QWidget):

    # Fields
    #SpeechBox = None
    #NLPBox = None
    #GoogleButton = None
    #DeepSpeechButton = None
    #StartButton = None
    #SpeechThread = None
    #NLP_thread = None

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setGeometry(500, 100, 1060, 800)
        self.setWindowTitle('CognitiveEMS Demo')
        self.setWindowIcon(QIcon('UVA.png'))
        self.setFixedSize(1060, 800)

        # Create label and textbox for speech 
        SpeechLabel = QLabel(self)
        SpeechLabel.move(20, 30)
        SpeechLabel.setText('Speech Recognition')

        self.SpeechBox = QTextEdit(self)
        self.SpeechBox.move(20, 60)
        self.SpeechBox.resize(500, 300)
        self.SpeechBox.setReadOnly(True)
        self.SpeechBox.setOverwriteMode(True)
        self.SpeechBox.ensureCursorVisible()
        
        # Create label and textbox for NLP
        NLPLabel = QLabel(self)
        NLPLabel.move(540, 30)
        NLPLabel.setText('Natural Language Processing')

        self.NLPBox = QTextEdit(self)
        self.NLPBox.move(540, 60)
        self.NLPBox.resize(500, 300)
        self.NLPBox.setReadOnly(True)

        # Create radio buttons for Google and DeepSpeech
        self.GoogleButton = QRadioButton('Google Speech API', self)
        self.GoogleButton.setChecked(True)
        self.GoogleButton.move(20, 380)

        self.DeepSpeechButton = QRadioButton('DeepSpeech', self)
        self.GoogleButton.setChecked(False)
        self.DeepSpeechButton.move(200, 380)
        self.DeepSpeechButton.setEnabled(False)


        # Create label and textbox for messages
        MsgBox = QLabel(self)
        MsgBox.move(20, 430)
        MsgBox.setText('Messages')

        self.MsgBox = QTextEdit(self)
        self.MsgBox.move(20, 460)
        self.MsgBox.resize(1020, 30)
        self.MsgBox.setReadOnly(True)
        self.MsgBox.setText('Ready to start speech recognition')


        # Create a button in the window
        self.StartButton = QPushButton('   Start   ', self)
        self.StartButton.move(20, 510)
        self.StartButton.clicked.connect(self.StartButtonClick)

        self.Speech_thread = threading.Thread(target = GoogleSpeech, args = (self.updateSpeechBox, self.updateMsgBox))
        self.NLP_thread = threading.Thread(target = NLP, args = (self.updateNLPBox,))

        # Add a picture
        PictureBox = QLabel(self)
        PictureBox.setGeometry(20, 698, 217, 82)
        #use full ABSOLUTE path to the image, not relative
        PictureBox.setPixmap(QPixmap('UVAEngLogo.jpg'))


    def closeEvent(self, event):
        print('Closing GUI')
        event.accept()
        
    @pyqtSlot()
    def StartButtonClick(self):
        # Threads
        if not self.Speech_thread.is_alive():
            if self.GoogleButton.isChecked():
                self.DeepSpeechButton.setEnabled(False)
                self.StartButton.setEnabled(False)
                self.Speech_thread = threading.Thread(target = GoogleSpeech, args = (self.updateSpeechBox, self.updateMsgBox))
                self.Speech_thread.start()
            else:
                self.GoogleButton.setEnabled(False)

        if not self.NLP_thread.is_alive():
            self.NLP_thread = threading.Thread(target = NLP, args = (self.updateNLPBox,))
            self.NLP_thread.start()

    def updateSpeechBox(self, text, numPrinted):
        previousText = self.SpeechBox.toPlainText()
        previousText = previousText[:len(previousText) - numPrinted]
        self.SpeechBox.clear()
        self.SpeechBox.setText(previousText + text)

    def updateMsgBox(self, text, option):
        self.MsgBox.setText(text)

    def updateNLPBox(self, text, option):
        self.NLPBox.insertPlainText(text)


# ============== Threads ==============

# Google Cloud Speech API Recognition Thread
def GoogleSpeech(updateSpeechFunction, updateMsgFunction):

    # Create Signal Object
    SpeechSignal = customSignalObject()
    SpeechSignal.signal.connect(updateSpeechFunction)

    MsgSignal = customSignalObject()
    MsgSignal.signal.connect(updateMsgFunction)

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
            Window.StartButton.setEnabled(False)
            
            # Now, put the transcription responses to use.
            num_chars_printed = 0

            for response in responses:
                if not response.results:
                    continue

                # The `results` list is consecutive. For streaming, we only care about
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
            Window.StartButton.setEnabled(True)
            sys.exit()

# Natural Language Processing Thread
def NLP(updateFunction):
    
    # Create Signal Object
    NLPSignal = customSignalObject()
    NLPSignal.signal.connect(updateFunction)

    while True:
        if not SpeechToNLPQueue.empty():
            receivedPhrase = SpeechToNLPQueue.get()
            #print 'NLP Thread Received: ', receivedPhrase
            NLPSignal.signal.emit(receivedPhrase, 0)


# ============== Main ==============

if __name__ == '__main__':

    # Set environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"

    # Create a queue
    SpeechToNLPQueue = queue.Queue()

    # Create the main window, show it, and run the app
    app = QApplication(sys.argv)
    Window = MainWindow()
    Window.show()
    sys.exit(app.exec_())




   






   


  





