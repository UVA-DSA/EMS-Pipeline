from __future__ import absolute_import, division, print_function
from timeit import default_timer as timer
import sys
from six.moves import queue
import numpy as np
from classes import SpeechNLPItem, GUISignal

import argparse
import shlex
import subprocess
import sys
import wave

#from deepspeech import Model, printVersions
from deepspeech import Model

from timeit import default_timer as timer

try:
    from shhlex import quote
except ImportError:
    from pipes import quote


def convert_samplerate(audio_path):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate 16000 --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path))
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use 16kHz files or install it: {}'.format(e.strerror))

    return 16000, np.frombuffer(output, dtype=np.int16)


def metadata_to_string(metadata):
    return ''.join(item.character for item in metadata.items)


class VersionAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        super(VersionAction, self).__init__(nargs=0, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        printVersions()
        exit(0)


# These constants control the beam search decoder

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_ALPHA = 0.75

# The beta hyperparameter of the CTC decoder. Word insertion bonus.
LM_BETA = 1.85


def DeepSpeech(Window, SpeechToNLPQueue, wavefile):

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

    audio = wavefile

    fin = wave.open(audio, 'rb') # open wave in read only mode
    fs = fin.getframerate() # obtaining the frame rate
    # DeepSpeech has to use 16kHz sample rate
    # DeepSpeech also needs the type to be of type 16 bit array
    if fs != 16000:
        print('Warning: original sample rate ({}) is different than 16kHz. Resampling might produce erratic speech recognition.'.format(fs), file=sys.stderr)
        fs, audio = convert_samplerate(audio)
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), dtype=np.int16)

    audio_length = fin.getnframes() * (1/16000)
    fin.close()

    print('Running inference.', file=sys.stderr)
    inference_start = timer()
    output = (ds.stt(audio, fs))
    print(output)
    inference_end = timer() - inference_start
    print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)

    QueueItem = SpeechNLPItem(output, True, 0, 0, 'Speech')
    SpeechToNLPQueue.put(QueueItem)
    SpeechSignal.signal.emit([QueueItem])


