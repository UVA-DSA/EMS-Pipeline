
from datetime import datetime
import argparse

import time

import wave
import numpy as np

import sounddevice as sd
from scipy.io.wavfile import write

from classes import TranscriptItem

from transformers import pipeline
import queue
import time

import multiprocessing
from threading import Thread
model_type = "tiny" # or "base"
model_type = "base"
version = 3


tiny_version_to_name_dict = {
    2: "saahith/tiny.en-combined_v4-1-0.1-32-1e-05-fragrant-sweep-3",
    3: "saahith/tiny.en-combined_v4-1-0.1-32-1e-05-revived-sweep-7",
    4: "saahith/tiny.en-combined_v4-1-0-32-1e-05-glamorous-sweep-1",
    5: "saahith/tiny.en-tiny.en-combined_v4-1-0.1-8-1e-06-sunny-sweep-55",
}


base_version_to_name_dict = {
    3: "saahith/base.en-combined_v4-2-0-8-1e-05-deft-sweep-10",
    4: "saahith/base.en-combined_v4-1-0-8-1e-05-elated-sweep-2",
    5: "saahith/base.en-combined_v4-1-0.1-8-1e-05-charmed-sweep-25",
    6: "saahith/base.en-combined_v4-1-0-16-1e-05-swept-sweep-3",
}

def transcribe(SpeechToNLPQueue,pipe, myrecording):
    # Code for the second part of the task
    # Send data to process1 through the queue
                # run inference
    pipeaudio = myrecording.ravel()
        
    start_t = time.perf_counter()
    output = pipe(pipeaudio)
    end_t = time.perf_counter()

    latency = (end_t-start_t)*1e3


    transcript = output["text"]
    print('EMSWhisper -- ',transcript, latency)

    transcriptItem = TranscriptItem(transcript, True, 1, latency)

    SpeechToNLPQueue.put(transcriptItem)
    



def WhisperPipeline( SpeechToNLPQueue, WhisperSignalQueue):
    
    # read in audio file
    # audio_file = "jfk.wav"
    # audio, rate = librosa.load(audio_file, sr=16000)


    model_repo = tiny_version_to_name_dict[version] if model_type == "tiny" else base_version_to_name_dict[version]
    pipe = pipeline(model=model_repo, batch_size=1, device=0, chunk_length_s=5, stride_length_s=(2, 2)) # set device to 0 to run on cuda

    data_queue = queue.Queue()

    audio_buffer = np.zeros((1600,1))
    while True:
        
        print("SignalQueue",WhisperSignalQueue, WhisperSignalQueue.qsize())
        signal = WhisperSignalQueue.get()
        WhisperSignalQueue.empty()
        print("EMSWhisper Signal: ",signal)
        if(signal == "Kill"): 
            audio_buffer = np.zeros((1600,1))
            print("EMSWhisper Killed!")
            break
        
        print('EMSWhisper: Capturing audio!')
        fs = 16000  # Sample rate
        seconds = 5 # Duration of recording

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished
        
        print(myrecording.shape)
        
            # run inference
        audio_buffer = np.concatenate((audio_buffer,myrecording),axis=0)
        
        process1 = Thread(target=transcribe, args=(SpeechToNLPQueue,pipe,audio_buffer))
        process1.start()
        
        # print(audio_buffer.shape)
        # print(pipeaudio.shape)
        
        # start_t = time.perf_counter()
        # output = pipe(pipeaudio)
        # end_t = time.perf_counter()
        
        # latency = (end_t-start_t)*1e3
        
        
        # transcript = output["text"]
        # print(transcript, latency)
        
        # transcriptItem = TranscriptItem(transcript, True, 1, latency)
        
        # SpeechToNLPQueue.put(transcriptItem)
        
        # if(audio_buffer.shape[0] == fs*70):
        #     audio_buffer = np.zeros((48000,1))



if __name__ == '__main__':
    q = queue.Queue()
    r = queue.Queue()
    WhisperPipeline(q,r)
    