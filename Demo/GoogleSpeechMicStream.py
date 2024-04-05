from __future__ import absolute_import, division, print_function
import os
from timeit import default_timer as timer
import sys
import pyaudio
import time
from google.cloud import speech_v1 as speech
from six.moves import queue
from StreamUtils import UDPStreamReceiver
from classes import SpeechNLPItem, GUISignal
import datetime
import wave
import numpy as np
import threading
import socket
import traceback
from google.api_core import exceptions
from google.api_core.retry import Retry
from copy import copy

# speech to text client
client = None
streaming_config = None
# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 20)  # 100ms #50ms
#CHUNK = int(RATE * 5)  # 100ms

stopped = False

stop_event = threading.Event()


audio_buff = queue.Queue(maxsize=16)
wav_audio_buffer = queue.Queue()

def audio_stream_UDP(stop_event):
    # AUDIO streaming variables and functions

    port = 2222

    sr = UDPStreamReceiver.UDPStreamReceiver(port)
    q = queue.Queue(10)
    sr.registerQueue(q)
    
    # receive and put audio data in queue
    while not stop_event.is_set():
        if q.empty():
            time.sleep(1e-3)
        else:
            payload = (q.get())
            audio = payload[12:]
            audio_buff.put(audio)
            wav_audio_buffer.put(audio)
        # print("Waiting for Audio client...")
        
    sr.unregisterQueue(q)
    sr.close()
    print("Audio Server Terminated!")

                


class MicrophoneStream(object):
    micSessionCounter = 0
    audio_buffer = b""
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, Window, rate, chunk, data_path_str, audioStreamBool, transcriptStreamBool):
        MicrophoneStream.micSessionCounter += 1
        self.Window = Window
        self._rate = rate
        self._chunk = chunk
        self.samplesCounter = 0
        self.start_time = time.time()
        self.client_socket = None
        self.audioStreamBool = audioStreamBool
        # self.data_path_str = data_path_str + "audiodata/"
        
        # if audioStreamBool == True:
        #     if not os.path.exists(self.data_path_str):
        #         os.makedirs(self.data_path_str)
        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True


    def __enter__(self):
        # self._audio_interface = pyaudio.PyAudio()
        # Run the audio stream asynchronously to fill the buffer object.
        # This is necessary so that the input device's buffer doesn't
        # overflow while the calling thread makes network requests, etc.
        # self._audio_stream = self._audio_interface.open(format = pyaudio.paInt16, channels = 1, rate = self._rate, input = True, frames_per_buffer = self._chunk, stream_callback = self._fill_buffer,)
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        # self._audio_stream.stop_stream()
        # self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        # self._buff.put(None)  
        
        print("Closing google")

        # return
        


        # self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        # self._buff.put(in_data)
        # return None, pyaudio.paContinue
        pass



    def generator(self):


        # Create GUI Signal Objects
        GoogleSignal = GUISignal()
        GoogleSignal.signal.connect(self.Window.StartGoogle)

        MsgSignal = GUISignal()
        MsgSignal.signal.connect(self.Window.UpdateMsgBox)

        VUSignal = GUISignal()
        VUSignal.signal.connect(self.Window.UpdateVUBox)

        while not self.closed:

            try:
                # print("generator debug")
                # Use a blocking get() to ensure there's at least one chunk of
                # data, and stop iteration if the chunk is None, indicating the
                # end of the audio stream.
                chunk = audio_buff.get()
                if chunk is None:
                    print('### Speech Paused')
                    return
                data = [chunk]

                # VU Meter in the GUI
                signal = b''.join(data)
                signal = np.fromstring(signal, 'int16') 
                VUSignal.signal.emit([signal])

                # Stop streaming after one minute, create new thread that does recognition
                # if time.time() > (self.start_time + (60)):
                #     GoogleSignal.signal.emit(["Mic"])
                #     MsgSignal.signal.emit(["API's 1 minute limit reached. Restablishing connection!"])
                #     break

                self.samplesCounter += self._chunk

                # Now consume whatever other data's still buffered.
                while True:
                    try:
                        chunk = audio_buff.get(block=False)
                        if chunk is None:
                            print('### Speech Paused')
                            return
                        data.append(chunk)
                    except queue.Empty:
                        break



                if self.Window.stopped == 1:
                    print('Speech Thread Killed')
                    stop_event.set()
                    # Dump file to disk
                    # if self.audioStreamBool == True or True:
                    #     output_audio = wave.open(self.data_path_str+ "audiodata.wav",'wb')
                    #     output_audio.setnchannels(1) # mono
                    #     output_audio.setsampwidth(2)
                    #     output_audio.setframerate(RATE)
                    #     output_audio.writeframesraw( data )
                    #     output_audio.close()
                    #     print(" ####### Audio written")
                    #     wav_audio_buffer = None

                    # client.close()
                    
                    return

                #MicrophoneStream.audio_buffer = np.append(MicrophoneStream.audio_buffer, np.fromstring(b''.join(data), 'Int16'))
                # MicrophoneStream.audio_buffer += b''.join(data)

  
                # if self.audioStreamBool == True:
                #     # Dump file to  if generator closed -- because the if self.Window.stopped == 1: condition is not happening when closing GUI or ctrl C
                #     output_audio = wave.open(self.data_path_str+ "audiodata.wav",'wb')
                #     output_audio.setnchannels(1) # mono
                #     output_audio.setsampwidth(2)
                #     output_audio.setframerate(RATE)
                #     output_audio.writeframesraw( wav_audio_buffer )
                #     output_audio.close()
                #     # print("audio written")

            except:
                  raise Exception("Generator terminated!")
            
            finally:
                # temp_data = copy(data)
                # MicrophoneStream.audio_buffer += b''.join(temp_data) # throws exception none iterating requests
                yield b''.join(data)    # b tells python to treat string as bytes



# Google Cloud Speech API Recognition Thread for Microphone
def GoogleSpeech(Window, SpeechToNLPQueue,EMSAgentSpeechToNLPQueue, data_path_str, audioStreamBool, transcriptStreamBool):

    global stop_event


    t1 = threading.Thread(target=audio_stream_UDP, args=(stop_event,))
    t1.start()

    data_path_str += "audiodata/"
    if audioStreamBool:
        if not os.path.exists(data_path_str):
                    os.makedirs(data_path_str)
    with speech.SpeechClient() as client:
        # Create GUI Signal Object
        SpeechSignal = GUISignal()
        SpeechSignal.signal.connect(Window.UpdateSpeechBox)

        MsgSignal = GUISignal()
        MsgSignal.signal.connect(Window.UpdateMsgBox)

        ButtonsSignal = GUISignal()
        ButtonsSignal.signal.connect(Window.ButtonsSetEnabled)

        language_code = 'en-US'  # a BCP-47 language tag

        # client = speech.SpeechClient()
        # config = speech.RecognitionConfig(encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz = RATE, language_code = language_code,    profanity_filter = True)
        config = speech.RecognitionConfig(encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz = RATE, language_code = language_code,    profanity_filter = True, model= "medical_dictation")
        streaming_config = speech.StreamingRecognitionConfig(config = config, interim_results = True)
        

        
        with MicrophoneStream(Window, RATE, CHUNK, data_path_str, audioStreamBool, transcriptStreamBool) as stream:

            audio_generator = stream.generator() # defined in the MicrophoneStream class above
            requests = (speech.StreamingRecognizeRequest(audio_content = content) for content in audio_generator)
            print("SPEECH-TO-TEXT:", (requests))


            try:
                
                #open file to write speech to text data 
                # data_path_str_transcript = data_path_str + "speechtext_transcript/"
                if transcriptStreamBool == True:
                    if not os.path.exists(data_path_str + "speechtext_transcript/"):
                        os.makedirs(data_path_str+"speechtext_transcript/")
                    speechtext_outputfile = open(data_path_str +"speechtext_transcript/" + "speechtotextdata.txt", 'w')

                responses = client.streaming_recognize(streaming_config, requests)
                # print("responses: ", responses)

                # Signal that streaming has started
                MsgSignal.signal.emit(["Started speech recognition on microphone audio via Google Speech API.\nMicrophone Session counter: " + str(MicrophoneStream.micSessionCounter)])
                
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
                    confidence = result.alternatives[0].confidence

                    # Display interim results, but with a carriage return at the end of the
                    # line, so subsequent lines will overwrite them.
                    # If the previous result was longer than this one, we need to print
                    # some extra spaces to overwrite the previous result
                    overwrite_chars = ' ' * (num_chars_printed - len(transcript))

                    if result.is_final:
                        print(transcript + overwrite_chars)
                        if transcriptStreamBool == True:
                            speechtext_outputfile.write(transcript)

                        QueueItem = SpeechNLPItem(transcript, result.is_final, confidence, num_chars_printed, 'Speech')
                        SpeechToNLPQueue.put(QueueItem)
                        EMSAgentSpeechToNLPQueue.put(QueueItem)
                        SpeechSignal.signal.emit([QueueItem])
                        num_chars_printed = 0

                    elif not result.is_final:
                        #sys.stdout.write(transcript + overwrite_chars + '\r')
                        #sSpeechSignalys.stdout.flush()
                        QueueItem = SpeechNLPItem(transcript, result.is_final, confidence, num_chars_printed, 'Speech')
                        SpeechSignal.signal.emit([QueueItem])
                        num_chars_printed = len(transcript)

            except Exception as e:
                # print(e)
                # print(traceback.format_exc())
                MsgSignal.signal.emit(["Unable to get response from Google! Network or other issues. Please Try again!\n Exception: " + str(e)])     
                ButtonsSignal.signal.emit([(Window.StartButton, True), (Window.ComboBox, True), (Window.ResetButton, True)])
                
                return

            try:
                if audioStreamBool == True:

                    data = []

                    while True:
                            try:
                                chunk = wav_audio_buffer.get(block=False)
                                if chunk is None:
                                    print('### Speech Paused')
                                    return
                                # data += b''.join(chunk)
                                data.append(chunk)
                            except queue.Empty:
                                break

                    data = b''.join(data)

                    print("WAVE AUDIO BUFFER LENGTH: ", len(data))
                    output_audio = wave.open(data_path_str+ "audiodata.wav",'wb')
                    output_audio.setnchannels(1) # mono
                    output_audio.setsampwidth(2)
                    output_audio.setframerate(RATE)
                    output_audio.writeframesraw( data )
                    output_audio.close()

            except Exception as e:
                print("AUDIO SAVING ERROR: ",e)

            client = None
            print("Closing Google-Speech-To-Text!")
            
            
