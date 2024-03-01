
import os
import queue
import subprocess
import pipeline_config #rename pipeline_config
import WhisperFileStream
from EMSAgent import EMSAgentSystem
from threading import Thread
from time import sleep


import multiprocessing



def SpeechModule(recording=pipeline_config.recording_name, whisper_model="test"):
# Set the Google Speech API service-account key environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"

# ===== Create thread-safe queues to pass on transcript and communication between modules ============
    SpeechToNLPQueue = queue.Queue()
    EMSAgentQueue  = queue.Queue()
    FeedbackQueue = queue.Queue()
    SignalQueue = queue.Queue()
    
    # ===== Start Speech Recognition module =================================================
    
    
# Start Whisper module
    whispercppcommand = [
    "./stream",
    "-m", # use specific whisper model
    f"models/ggml-{whisper_model}.bin", 
    "--threads",
    str(pipeline_config.num_threads),
    "--step",                          
    str(pipeline_config.step),
    "--length",
    str(pipeline_config.length),
    "--keep",
    str(pipeline_config.keep_ms)
    ]
    # If a Hard-coded Audio test file, use virtual mic to capture the recording
    if(pipeline_config.hardcoded):
        whispercppcommand.append("--capture")

    # Start subprocess
    WhisperSubprocess = subprocess.Popen(whispercppcommand, cwd='whisper.cpp/')
    
    sleep(10)
    
    # ===== Start Whisper Audiostream and Read module =========================================
    recording_dir = f'./Audio_Scenarios/2019_Test/{recording}.wav'
        
    Audiostream = Thread(
                    target=WhisperFileStream.Whisper, args=(SpeechToNLPQueue, EMSAgentQueue, recording_dir, True, SignalQueue))
    Audiostream.start()
    
            
    Audiostream.join()
    WhisperSubprocess.terminate()
    
    
    

    
    
    

if __name__ == '__main__':
    SpeechModule()