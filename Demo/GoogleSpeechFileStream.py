from __future__ import absolute_import, division, print_function
from timeit import default_timer as timer
import sys
import os
import time
from six.moves import queue
from google.cloud import speech_v1 as speech
import numpy as np
from classes import SpeechNLPItem, GUISignal
import wave

# Suppress pygame's welcome message
with open(os.devnull, 'w') as f:
    # disable stdout
    oldstdout = sys.stdout
    sys.stdout = f
    from pygame import mixer
    # enable stdout
    sys.stdout = oldstdout

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms


class FileStream(object):
    fileSessionCounter = 0
    position = 0
    """Opens a file stream as a generator yielding the audio chunks."""

    def __init__(self, Window, rate, chunk, wavefile):
        FileStream.fileSessionCounter += 1
        self.Window = Window
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
        #self._audio_interface = pyaudio.PyAudio()
        mixer.init(frequency=RATE)
        #self._audio_stream = self._audio_interface.open(format = pyaudio.paInt16, channels = 1, rate = self._rate, input = True, frames_per_buffer = self._chunk, stream_callback = self._fill_buffer,)
        self.closed = False

        if(FileStream.position == 0):
            mixer.music.load(self.filename)
            mixer.music.play()

        return self

    def __exit__(self, type, value, traceback):
        # self._audio_stream.stop_stream()
        # self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        # self._buff.put(None)
        # self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        # self._buff.put(in_data)
        pass
        #data = self.wf.readframes(CHUNK)

        # self._buff.put(data)
        # return None, pyaudio.paContinue

    def generator(self):
        # Create GUI Signal Objects
        GoogleSignal = GUISignal()
        GoogleSignal.signal.connect(self.Window.StartGoogle)

        MsgSignal = GUISignal()
        MsgSignal.signal.connect(self.Window.UpdateMsgBox)

        VUSignal = GUISignal()
        VUSignal.signal.connect(self.Window.UpdateVUBox)

        ButtonsSignal = GUISignal()
        ButtonsSignal.signal.connect(self.Window.ButtonsSetEnabled)

        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            #chunk = self._buff.get()
            time.sleep(.1)
            chunk = self.wf.readframes(CHUNK)
            # print("chunk: ", chunk)

            if chunk == '':
                FileStream.position = 0
                MsgSignal.signal.emit(["Transcription of audio file complete!"])
                ButtonsSignal.signal.emit(
                    [(self.Window.StartButton, True), (self.Window.ComboBox, True), (self.Window.ResetButton, True)])
                return

            if chunk is None:
                return

            if self.samplesCounter/self._rate > 60:
                #FileStream.position -= 3
                GoogleSignal.signal.emit(["File"])
                print((self.samplesCounter)/self._rate)
                MsgSignal.signal.emit(["API's 1 minute limit reached. Restablishing connection!"])
                break

            data = [chunk]

            signal = b''.join(data)
            # print(signal)
            try:
                signal = np.fromstring(signal, 'int16')
                VUSignal.signal.emit([signal])

            except Exception as e:
                print("exception occurred")
                print(e)

            self.samplesCounter += self._chunk
            FileStream.position += 1

            if self.Window.stopped == 1:
                print('File Speech Tread Killed')
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


def GoogleSpeech(Window, SpeechToNLPQueue,EMSAgentSpeechToNLPQueue, wavefile_name):
    # Create GUI Signal Object
    SpeechSignal = GUISignal()
    SpeechSignal.signal.connect(Window.UpdateSpeechBox)

    MsgSignal = GUISignal()
    MsgSignal.signal.connect(Window.UpdateMsgBox)

    ButtonsSignal = GUISignal()
    ButtonsSignal.signal.connect(Window.ButtonsSetEnabled)

    language_code = 'en-US'  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                                      sample_rate_hertz=RATE, language_code=language_code, profanity_filter=True)  # ,model='video')
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

    with FileStream(Window, RATE, CHUNK, wavefile_name) as stream:
        audio_generator = stream.generator()
        requests = (speech.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)

        try:
            responses = client.streaming_recognize(streaming_config, requests)
            # Signal that streaming has started
            print("Started speech recognition on file audio via Google Speech API.\nFile Session Counter: " +
                  str(FileStream.fileSessionCounter))
            MsgSignal.signal.emit(
                ["Started speech recognition on file audio via Google Speech API.\nFile Session Counter: " + str(FileStream.fileSessionCounter)])

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
                    #print(transcript + overwrite_chars)
                    QueueItem = SpeechNLPItem(transcript, result.is_final,
                                              confidence, num_chars_printed, 'Speech')
                    SpeechToNLPQueue.put(QueueItem)
                    EMSAgentSpeechToNLPQueue.put(QueueItem)

                    SpeechSignal.signal.emit([QueueItem])
                    num_chars_printed = 0

                elif not result.is_final:
                    #sys.stdout.write(transcript + overwrite_chars + '\r')
                    # sys.stdout.flush()
                    QueueItem = SpeechNLPItem(transcript, result.is_final,
                                              confidence, num_chars_printed, 'Speech')
                    SpeechSignal.signal.emit([QueueItem])
                    num_chars_printed = len(transcript)

        except Exception as e:
            # print(e)
            MsgSignal.signal.emit(
                ["Unable to get response from Google! Network or other issues. Please Try again!\n Exception: " + str(e)])
            ButtonsSignal.signal.emit(
                [(Window.StartButton, True), (Window.ComboBox, True), (Window.ResetButton, True)])
            sys.exit()
