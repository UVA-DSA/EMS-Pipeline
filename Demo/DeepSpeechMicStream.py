from __future__ import absolute_import, division, print_function
from timeit import default_timer as timer
import sys
import os
import pyaudio
import time
from six.moves import queue
import numpy as np
from classes import SpeechNLPItem, GUISignal


import argparse
import shlex
import subprocess
import sys
import wave
import audioop
import scipy.io.wavfile as wav
import io


from deepspeech import Model, printVersions
from timeit import default_timer as timer

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

class MicrophoneStream(object):
    micSessionCounter = 0
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
        # Create Signal Objects
        MsgSignal = GUISignal()
        MsgSignal.signal.connect(self.Window.UpdateMsgBox)

        # Create Signal Objects
        VUSignal = GUISignal()
        VUSignal.signal.connect(self.Window.UpdateVUBox)

        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Plot in the GUI
            signal = b''.join(data)
            signal = np.fromstring(signal, 'Int16') 
            VUSignal.signal.emit([signal])

            # Stop streaming after one minute, create new thread that does recognition
            if time.time() > (self.start_time + (10)):
                MsgSignal.signal.emit(["Recorded 10 seconds"])
                self.Window.StartButton.setEnabled(True)
                break

            self.samplesCounter += self._chunk

            if self.Window.stopped == 1:
                print('Speech Tread Killed')
                self.Window.StartButton.setEnabled(True)
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
            
            yield signal


try:
    from shhlex import quote
except ImportError:
    from pipes import quote

# These constants control the beam search decoder

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_ALPHA = 0.75

# The beta hyperparameter of the CTC decoder. Word insertion bonus.
LM_BETA = 1.85 #

def convert_samplerate(audio_path):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate 16000 --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path))
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use 16kHz files or install it: {}'.format(e.strerror))

    return 16000, np.frombuffer(output, np.int16)


def metadata_to_string(metadata):
    return ''.join(item.character for item in metadata.items)


class VersionAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        super(VersionAction, self).__init__(nargs=0, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        printVersions()
        exit(0)

# DeepSpeech Recognition Thread for Microphone
def DeepSpeech(Window, SpeechToNLPQueue):

    # Create Signal Object
    SpeechSignal = GUISignal()
    SpeechSignal.signal.connect(Window.UpdateSpeechBox)

    MsgSignal = GUISignal()
    MsgSignal.signal.connect(Window.UpdateMsgBox)

     # References to models:
    model_path = 'DeepSpeech_Models/deepspeech-0.9.3-models.pbmm'
    scorer_path = 'DeepSpeech_Models/deepspeech-0.9.3-models.scorer'

    print('Loading model from file {}'.format(model_path), file=sys.stderr)
    model_load_start = timer()
    ds = Model(model_path) # setting the model to be used in deepspeech
    model_load_end = timer() - model_load_start
    print('Loaded model in {:.3}s.'.format(model_load_end), file=sys.stderr)

    print('Loading language model from files {} {}'.format(scorer_path), file=sys.stderr)
    lm_load_start = timer()
    ds.enableExternalScorer(scorer_path)
    lm_load_end = timer() - lm_load_start
    print('Loaded language model in {:.3}s.'.format(lm_load_end), file=sys.stderr)

    # these lines are used to help the decoder in deepspeech
    # values are recommended by the developers
    ds.setScorerAlphaBeta(LM_ALPHA, LM_BETA)
    ds.setBeamWidth(BEAM_WIDTH)

    audio = []
    with MicrophoneStream(Window, RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        for content in audio_generator:
            for sample in content:
                audio.append(sample)
    
    result = ds.stt(audio, 16000)

    QueueItem = SpeechNLPItem(result, True, 0, 0, 'Speech')
    SpeechToNLPQueue.put(QueueItem)
    SpeechSignal.signal.emit([QueueItem])


