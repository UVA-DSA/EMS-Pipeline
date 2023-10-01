
import os
import queue
import subprocess
import pipeline_config #rename pipeline_config
import WhisperFileStream
from EMSAgent import EMSAgentSystem
from threading import Thread

def Pipeline(recording=pipeline_config.recording_name):
# Set the Google Speech API service-account key environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"

# ===== Create thread-safe queues for communication between modules ============
    SpeechToNLPQueue = queue.Queue()
    EMSAgentQueue  = queue.Queue()
    FeedbackQueue = queue.Queue()

# ===== Start Whisper module ===================================================
    whispercppcommand = [
    "./stream",
    "-m", # use specific whisper model
    f"models/ggml-{pipeline_config.model_size}-finetuned.bin", 
    "--threads",
    str(pipeline_config.num_threads),
    "--step",                          
    str(pipeline_config.step),
    "--length",
    str(pipeline_config.length)
    ]
    # If a Hard-coded Audio test file, use virtual mic to capture the recording
    if(pipeline_config.hardcoded):
        whispercppcommand.append("--capture")
    # Start subprocess
    WhisperSubprocess = subprocess.Popen(whispercppcommand, cwd='whisper.cpp/')

# ===== Start Audiostream module =========================================
    Audiostream = Thread(
                    target=WhisperFileStream.Whisper, args=(SpeechToNLPQueue, EMSAgentQueue, './Audio_Scenarios/2019_Test/' + str(recording)))
    Audiostream.start()

# ===== Start EMSAgent module =================================================
    EMSAgent = Thread(target=EMSAgentSystem.EMSAgentSystem, args=(EMSAgentQueue, FeedbackQueue))
    EMSAgent.start()

# ===== Exiting Program ====================================================
    '''
    When Whisper() of WhisperFileStream finishes running, that means recording is finished
    Then WhisperFileStream will queue 'Kill' to EMSAgent, which will break EMSAgent loop. This is a temporary hack
    TODO: Make a Event() from Threading module to use as flag signal instead of queueing the string 'Kill'
    '''
    Audiostream.join()
    EMSAgent.join()
    WhisperSubprocess.terminate()


if __name__ == '__main__':
    Pipeline()