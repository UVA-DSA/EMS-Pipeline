# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import tensorflow as tf

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

# See the License for the specific language governing permissions and
# limitations under the License.

import operator
import os
import io
import queue

import pipeline_config

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)

from tqdm import tqdm

#import tensorflow as tf
from EMSConformer.inference.tensorflow_asr.utils.file_util import read_file
import json
from pydub import AudioSegment

from EMSConformer.inference.tensorflow_asr.featurizers.speech_featurizers import read_raw_audio

from datetime import datetime
import argparse

import time

import pyaudio
import wave
import numpy as np

import sounddevice as sd
from scipy.io.wavfile import write

from classes import TranscriptItem

import tensorflow as tf
logger = tf.get_logger()

def capture_audio(model, SpeechToNLPQueue, ConformerSignalQueue):
    
    full_transcription = ""
    while True:
        
        signal = ConformerSignalQueue.get()
        print("EMSConformer Signal: ",signal)
        if(signal == "Kill"): 
            SpeechToNLPQueue.put("Kill")
            break
        
        print('EMSConformer: Capturing audio!')
        fs = 16000  # Sample rate
        seconds = 4 # Duration of recording

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished
        
        print(type(myrecording))
        print(myrecording.shape)
        write('output.wav', fs, myrecording)  # Save as WAV f
            
            
        # print('Sounddevice :',myrecording)
        print('EMSConformer: Transcribing audio!')
        
        myrecording = myrecording.ravel()
        audio_buffer = read_raw_audio(myrecording)

        transcript, latency = trainscribe(model,audio_buffer)
        full_transcription += transcript
        
        transcriptItem = TranscriptItem(full_transcription, True, 0, latency)
        SpeechToNLPQueue.put(transcriptItem)  
        
        if pipeline_config.speech_standalone:
            pipeline_config.trial_data['speech latency (ms)'].append(latency)
            pipeline_config.trial_data['transcript'].append(full_transcription)
            pipeline_config.trial_data['whisper confidence'].append(0)
        
        print('EMSConformer: Transcription Done!')
        
        


def trainscribe(model,audio_buffer):
    
    signal = audio_buffer # audio buffer
    start_t = time.perf_counter()

    transcript = model.pred(signal)
    transcript = "".join([chr(u) for u in transcript])
    
    end_t = time.perf_counter()

    latency = (end_t-start_t)*1e3
    
    print("\ntranscription:",transcript)
    print("latency:",latency)
    print("")
    
    return transcript, latency


def main(
    SpeechToNLPQueue: queue,
    ConformerSignalQueue: queue,
):

    tf.keras.backend.clear_session()

    module = tf.saved_model.load(export_dir='/home/kesharaw/Desktop/repos/EMS-Pipeline/Demo/EMSConformer/speech_models')
    
    capture_audio(module, SpeechToNLPQueue, ConformerSignalQueue)


