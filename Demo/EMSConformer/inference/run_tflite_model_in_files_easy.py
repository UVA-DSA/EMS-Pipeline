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

import operator
import os
import io
import queue

import pipeline_config

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)

from tqdm import tqdm

from EMSConformer.inference.tensorflow_asr.metrics.error_rates import ErrorRate
from EMSConformer.inference.tensorflow_asr.utils.file_util import read_file
from EMSConformer.inference.tensorflow_asr.utils.metric_util import cer, wer


import fire
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

def capture_audio(tflitemodel,input_details,output_details,blank,num_rnns,nstates,statesize, SpeechToNLPQueue, ConformerSignalQueue):
    
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
        # write('output.wav', fs, myrecording)  # Save as WAV f
            
            
        # print('Sounddevice :',myrecording)
        print('EMSConformer: Transcribing audio!')
        
        myrecording = myrecording.ravel()
        audio_buffer = read_raw_audio(myrecording)

        transcript, latency = trainscribe(tflitemodel,input_details,output_details,audio_buffer,blank,num_rnns,nstates,statesize)
        full_transcription += transcript
        
        transcriptItem = TranscriptItem(full_transcription, True, 0, latency)
        SpeechToNLPQueue.put(transcriptItem)  

        if pipeline_config.speech_standalone:
            pipeline_config.trial_data['speech latency (ms)'].append(latency)
            pipeline_config.trial_data['transcript'].append(full_transcription)
            pipeline_config.trial_data['whisper confidence'].append(0)

        print('EMSConformer: Transcription Done!')
        
        
        
        

def writeListFile(file_path, output_list, encoding = None):
    f = open(file_path, mode = "w")
    output_str = "\n".join(output_list)
    f.write(output_str)
    f.close()

def readFile(file_path, encoding = None):
    f = open(file_path, 'r')
    lines = f.read().splitlines()
#    print(lines[0])
    res = []
    for line in lines:
        line = line.strip()
        res.append(line)
    f.close()
    return res

"""
compare_normal_output is used to evaluate wer and cer given a normal text file 
with only transcribed text and ground true text.

Input: the path of text file with only transcribed text and ground true text.
Output: the WER and CER for the given text file
"""   
def compare_normal_output(filepath: str):

    logger.info(f"Evaluating result from {filepath} ...")
    metrics = {
        "wer": ErrorRate(wer, name="wer", dtype=tf.float32),
        "cer": ErrorRate(cer, name="cer", dtype=tf.float32),
    }
    with read_file(filepath) as path:
        with open(path, "r", encoding="utf-8") as openfile:
            lines = openfile.read().splitlines()
    for eachline in tqdm(lines):
        groundtruth, asr_output = eachline.split("\t")
        groundtruth = tf.convert_to_tensor([groundtruth], dtype=tf.string)
        asr_output = tf.convert_to_tensor([asr_output], dtype=tf.string)
        metrics["wer"].update_state(decode=asr_output, target=groundtruth)
        metrics["cer"].update_state(decode=asr_output, target=groundtruth)
    for key, value in metrics.items():
        logger.info(f"{key}: {value.result().numpy()}")

"""
The main function to evaluate the WER and CER of EMSConformer tflite on our EMS audio dataset. 
It uses EMSConformer tflite model to transcribe EMS audio dataset and correspondinly compute WER and CER.

Input:  1) tflite_model: the conformer tflite model used to transcribe the EMS audio files
        2) data_path: the tsv file that contains the testset audio paths
Output: 1) the WER and CER for both greedy search and beam search.
        2) the intermediate transcription text output.
"""


def trainscribe(tflitemodel,input_details,output_details, audio_buffer,blank,num_rnns,nstates,statesize):
    
    signal = audio_buffer # audio buffer
    start_t = time.perf_counter()
    tflitemodel.resize_tensor_input(input_details[0]["index"], signal.shape)
    tflitemodel.allocate_tensors()
    tflitemodel.set_tensor(input_details[0]["index"], signal)
    tflitemodel.set_tensor(input_details[1]["index"], tf.constant(blank, dtype=tf.int32))
    #print('input_details[2]["index"] =', input_details[2]["index"])
    #print('tf.zeros([num_rnns, nstates, 1, statesize] =', tf.zeros([num_rnns, nstates, 1, statesize]))
    tflitemodel.set_tensor(input_details[2]["index"], tf.zeros([num_rnns, nstates, 1, statesize], dtype=tf.float32))
    tflitemodel.invoke()
    hyp0 = tflitemodel.get_tensor(output_details[0]["index"])
    hyp1 = tflitemodel.get_tensor(output_details[1]["index"])
    hyp2 = tflitemodel.get_tensor(output_details[2]["index"])

    end_t = time.perf_counter()

    latency = (end_t-start_t)*1e3

    pred_line = "".join([chr(u) for u in hyp0])
    print("\ntranscription:",pred_line)
    print("latency:",latency)
    print("")
    
    return pred_line, latency


def main(
    SpeechToNLPQueue: queue,
    ConformerSignalQueue: queue,
    tflite: str,
    blank: int = 0,
    num_rnns: int = 1,
    nstates: int = 2,
    statesize: int = 320,
):


    tflitemodel = tf.lite.Interpreter(model_path=tflite)    
    input_details = tflitemodel.get_input_details()

    # print("input_details: ")
    # print(input_details)

    output_details = tflitemodel.get_output_details()
    #print("output_details: ")                                                                                                                                                                           
    #print(output_details)


    capture_audio(tflitemodel,input_details,output_details,blank,num_rnns,nstates,statesize, SpeechToNLPQueue, ConformerSignalQueue)


"""
The main function to evaluate the WER and CER of EMSConformer tflite on our EMS audio dataset. 
It uses EMSConformer tflite model to transcribe EMS audio dataset and correspondinly compute WER and CER.

Input:  1) tflite_model: the conformer tflite model used to transcribe the EMS audio files
        2) data_path: the tsv file that contains the testset audio paths
Output: the WER and CER for both greedy search and beam search.
"""
# if __name__ == "__main__":

#     time_s = datetime.now()

#     parser = argparse.ArgumentParser(description = "control the functions for evaluating tflite with files")
#     parser.add_argument("--tflite_model", action='store', type=str, default = "./tflite_models/pretrained_librispeech_train_ss_test_concatenated_epoch50_noOptimize.tflite", required=True)
#     args = parser.parse_args()

#     print()
#     print("args.tflite_model =", args.tflite_model)

#     main(tflite=args.tflite_model)

