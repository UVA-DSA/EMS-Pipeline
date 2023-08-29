from __future__ import absolute_import, division, print_function
from timeit import default_timer as timer
import sys
import pyaudio
import time
from six.moves import queue
from classes import SpeechNLPItem, GUISignal
import datetime
import wave
import numpy as np
from WhisperAPI import WhisperAPI
import traceback


# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
#CHUNK = int(RATE * 5)  # 100ms

class MicrophoneStream(object):
    micSessionCounter = 0
    audio_buffer = b""
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, Window, rate, chunk):
        MicrophoneStream.micSessionCounter += 1
        self.Window = Window
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
        # Create GUI Signal Objects
        GoogleSignal = GUISignal()
        GoogleSignal.signal.connect(self.Window.StartGoogle)

        MsgSignal = GUISignal()
        MsgSignal.signal.connect(self.Window.UpdateMsgBox)

        VUSignal = GUISignal()
        VUSignal.signal.connect(self.Window.UpdateVUBox)

        while not self.closed:
            # print("generator debug")
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # VU Meter in the GUI
            signal = b''.join(data)
            signal = np.fromstring(signal, 'int16') 
            VUSignal.signal.emit([signal])

            # Stop streaming after one minute, create new thread that does recognition
            if time.time() > (self.start_time + (60)):
                GoogleSignal.signal.emit(["Mic"])
                MsgSignal.signal.emit(["API's 1 minute limit reached. Restablishing connection!"])
                break

            self.samplesCounter += self._chunk

            if self.Window.stopped == 1:
                print('Speech Tread Killed')
           
                # Dump file to disk
                output_audio = wave.open("./Dumps/" + str(datetime.datetime.now().strftime("%c")) + ".wav",'wb')
                output_audio.setnchannels(1) # mono
                output_audio.setsampwidth(2)
                output_audio.setframerate(RATE)
                output_audio.writeframesraw( MicrophoneStream.audio_buffer )
                output_audio.close()

                MicrophoneStream.audio_buffer = ""
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

            #MicrophoneStream.audio_buffer = np.append(MicrophoneStream.audio_buffer, np.fromstring(b''.join(data), 'Int16'))
            MicrophoneStream.audio_buffer += b''.join(data)
            yield b''.join(data)    # b tells python to treat string as bytes




# Whisper recognition thread for microphone
def Whisper(Window, SpeechToNLPQueue):
    # Create GUI Signal Object
    SpeechSignal = GUISignal()
    SpeechSignal.signal.connect(Window.UpdateSpeechBox)

    MsgSignal = GUISignal()
    MsgSignal.signal.connect(Window.UpdateMsgBox)

    ButtonsSignal = GUISignal()
    ButtonsSignal.signal.connect(Window.ButtonsSetEnabled)

    whisper = WhisperAPI()

    with MicrophoneStream(Window, RATE, CHUNK) as stream:

        audio_generator = stream.generator()

        try:
            responses = whisper.transcribe_stream(audio_generator)
            
            # Signal that streaming has started
            MsgSignal.signal.emit(["Started speech recognition on microphone audio locally with OpenAI Whisper.\nMicrophone Session counter: " + str(MicrophoneStream.micSessionCounter)])

            # Now, put the transcription responses to use.
            num_chars_printed = 0
            responseTimeStamp = time.time()

            for result in responses:
                if not result['transcript']:
                    continue
                
                # transcript = result['transcript']
                transcript = result['latest_finalized_transcript']
                with open('transcript.txt', 'a') as f:
                    f.write(transcript + "\n")

                # unfinalized_transcript = response['unfinalized_transcript']
                # finalized_transcript = response['finalized_transcript']

                QueueItem = SpeechNLPItem(transcript, result['finalized'], -1, num_chars_printed, 'Speech')
                SpeechToNLPQueue.put(QueueItem)
                SpeechSignal.signal.emit([QueueItem])
                num_chars_printed = 0 if result['finalized'] else len(transcript)

            
        except Exception as e:
            print(e)
            exception_string = traceback.format_exc()
            print(exception_string)
            MsgSignal.signal.emit(["Exception with Whisper model! Network or other issues. Please Try again!\n Exception: " + str(e)])     
            ButtonsSignal.signal.emit([(Window.StartButton, True), (Window.ComboBox, True), (Window.ResetButton, True)])
            sys.exit()
