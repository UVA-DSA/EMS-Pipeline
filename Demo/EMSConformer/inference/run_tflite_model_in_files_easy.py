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

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)

import tensorflow as tf
from tqdm import tqdm

from EMSConformer.inference.tensorflow_asr.metrics.error_rates import ErrorRate
from EMSConformer.inference.tensorflow_asr.utils.file_util import read_file
from EMSConformer.inference.tensorflow_asr.utils.metric_util import cer, wer

logger = tf.get_logger()

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


def capture_audio(tflitemodel,input_details,output_details,blank,num_rnns,nstates,statesize, SpeechToNLPQueue, ConformerSignalQueue):
    
    while True:
        
        signal = ConformerSignalQueue.get()
        print("EMSConformer Signal: ",signal)
        if(signal == "Kill"): break
        
        print('EMSConformer: Capturing audio!')
        fs = 16000  # Sample rate
        seconds = 3 # Duration of recording

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
        
        transcriptItem = TranscriptItem(transcript, True, 0, latency)
        # EMSAgentQueue.put(transcriptItem)
        SpeechToNLPQueue.put(transcriptItem)  
        
        pipeline_config.curr_segment += [transcriptItem.transcriptionDuration, transcriptItem.transcript, transcriptItem.confidence, 'wer', 'cer']
        # if we made a protocol prediction
            # if no suggesion, save one hot vector of all 0's
        pipeline_config.curr_segment += [-1, "NULL", -1, -1, -1, -1, -1, -1, -1, -1]
        pipeline_config.rows_trial.append(pipeline_config.curr_segment)
        pipeline_config.curr_segment = []
        
        
        print('EMSConformer: Transcription Done!')
        
        
        
        
    
    
    
    
    
    
#     chunk = 1024  # Record in chunks of 1024 samples
#     sample_format = pyaudio.paInt16  # 16 bits per sample
#     channels = 1
#     fs = 16000  # Record at 16000 samples per second
#     filename = "output.wav"

#     p = pyaudio.PyAudio()  # Create an interface to PortAudio

#     print('Recording')

#     stream = p.open(format=sample_format,
#                     channels=channels,
#                     rate=fs,
#                     frames_per_buffer=chunk,
#                     input=True)

#     frames = []  # Initialize array to store frames

# # Store data in chunks for 3 seconds
#     # Store data in chunks for 3 seconds
#     seconds = 10
#     num_chunks = int(fs / chunk * seconds)
#     print(num_chunks)
#     i = 1
#     while True:
        
#         data = stream.read(chunk)
        
#         frames.append(data)
#         if(i%num_chunks == 0):
#             # call transcription
            
#             audio_data = b"".join(frames)
#                         # Convert the audio data to WAV format
#             with io.BytesIO() as wav_buffer:
#                 with wave.open(wav_buffer, 'wb') as wf:
#                     wf.setnchannels(channels)
#                     wf.setsampwidth(sample_format // 8)
#                     wf.setframerate(fs)
#                     wf.writeframes(audio_data)

#                 # Read the WAV data from the buffer
#                 wav_buffer.seek(0)
#                 wav_data = wav_buffer.read()

            
#             audio_buffer = read_raw_audio(wav_data)
#             audio_buffer = audio_buffer.astype(np.float32)
#             print(audio_buffer.shape)
            
#             frames = []
            
#             trainscribe(tflitemodel,input_details,output_details,audio_buffer,blank,num_rnns,nstates,statesize)
            
        
#         i = i+1

#     # Stop and close the stream 
#     stream.stop_stream()
#     stream.close()
#     # Terminate the PortAudio interface
#     p.terminate()

#     print('Finished recording')

#     # Save the recorded data as a WAV file
#     wf = wave.open(filename, 'wb')
#     wf.setnchannels(channels)
#     wf.setsampwidth(p.get_sample_size(sample_format))
#     wf.setframerate(fs)
#     wf.writeframes(b''.join(frames))
#     wf.close()










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

