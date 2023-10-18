
import os
import queue
import subprocess
import pipeline_config #rename pipeline_config
import WhisperFileStream
import FIFOStreamClient
from EMSAgent import EMSAgentSystem
from threading import Thread
from time import sleep,time


def Pipeline(recording=pipeline_config.recording_name):
# Set the Google Speech API service-account key environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"

# ===== Create thread-safe queues to pass on transcript and communication between modules ============
    SpeechToNLPQueue = queue.Queue()
    EMSAgentQueue  = queue.Queue()
    FeedbackQueue = queue.Queue()
    SignalQueue = queue.Queue()

# ===== Start Whisper module ===================================================
    whispercppcommand = [
    "./stream",
    "-m", # use specific whisper model
    f"models/ggml-{pipeline_config.whisper_model_size}.bin", 
    "--threads",
    str(pipeline_config.num_threads),
    "--step",                          
    str(pipeline_config.step),
    "--length",
    str(pipeline_config.length),
    "--keep",
    str(pipeline_config.keep_ms),
    # "-vth",
    # str(pipeline_config.vth),
    # "-ac",
    # str(pipeline_config.audio_ctx)
    "-mt",
    str(pipeline_config.max_tokens)
    ]
    # If a Hard-coded Audio test file, use virtual mic to capture the recording
    if(pipeline_config.hardcoded):
        whispercppcommand.append("--capture")

# ===== Start EMSAgent module =================================================
    EMSAgent = Thread(target=EMSAgentSystem.EMSAgentSystem, args=(EMSAgentQueue, FeedbackQueue))
    EMSAgent.start()

# ===== Sleep for 3 seconds to finish starting Whisper module and EMSAgent module =====
    print("======================= Warmup Phase ======================")
    signal = FeedbackQueue.get()
    while (signal.protocol != 'wd'):
        print('.',end="")
        sleep(0.1)
    print("======================= Warmup Done ======================")







# ===== Start Whisper Subprocess module =================================================


    # Start subprocess
    print("======================= Whisper Warmup ======================")
 
    if pipeline_config.whisper_model_size == 'tiny-finetuned-q5':

        is_streaming = True

    # # ===== Start Audio Streaming Subprocess module =================================================

        # audiostreamcommand = [
        # "./stream-file",
        # "-f", # uspecify which wavefile
        # f"{recording}.wav", 
        # ]
        
        # AudioStreamSubprocess = subprocess.Popen(audiostreamcommand, cwd='whisper.cpp/')


    # ===== Start Audiostream module =========================================
        Audiostream = Thread(
                        target=WhisperFileStream.Whisper, args=(SpeechToNLPQueue, EMSAgentQueue, f'./Audio_Scenarios/2019_Test/{recording}.wav', is_streaming, SignalQueue))
        Audiostream.start()

        sleep(3)

        WhisperSubprocess = subprocess.Popen(whispercppcommand, cwd='whisper.cpp/')


    # ===== Start FIFO reading module =========================================
        FIFOStream = Thread(
                        target=FIFOStreamClient.Fifo, args=(SpeechToNLPQueue, EMSAgentQueue, SignalQueue))
        FIFOStream.start()

        


    else:

        is_streaming = True

   
        WhisperSubprocess = subprocess.Popen(whispercppcommand, cwd='whisper.cpp/')

        sleep(5)

  
     # ===== Start Audiostream module =========================================
        Audiostream = Thread(
                        target=WhisperFileStream.Whisper, args=(SpeechToNLPQueue, EMSAgentQueue, f'./Audio_Scenarios/2019_Test/{recording}.wav', is_streaming, SignalQueue))
        Audiostream.start()


    # # ===== Start Audio Streaming Subprocess module =================================================

        # audiostreamcommand = [
        # "./stream-file",
        # "-f", # uspecify which wavefile
        # f"{recording}.wav", 
        # ]
        
        # AudioStreamSubprocess = subprocess.Popen(audiostreamcommand, cwd='whisper.cpp/')







        # ===== Start FIFO reading module =========================================
        FIFOStream = Thread(
                        target=FIFOStreamClient.Fifo, args=(SpeechToNLPQueue, EMSAgentQueue, SignalQueue))
        FIFOStream.start()

        



    print("======================= Whisper Warmup Done ======================")






# ===== Exiting Program ====================================================
    '''
    When Whisper() of WhisperFileStream finishes running, that means recording is finished
    Then WhisperFileStream will queue 'Kill' to EMSAgent, which will break EMSAgent loop. This is a temporary hack
    TODO: Make a Event() from Threading module to use as flag signal instead of queueing the string 'Kill'
    '''
    Audiostream.join()
    FIFOStream.join()
    EMSAgent.join()
    WhisperSubprocess.terminate()
    # AudioStreamSubprocess.terminate()


if __name__ == '__main__':
    Pipeline()
