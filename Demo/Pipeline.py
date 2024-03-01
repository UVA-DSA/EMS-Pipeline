
import os
import queue
import subprocess
import pipeline_config #rename pipeline_config
import WhisperFileStream
import FIFOStreamClient
from EMSAgent import EMSAgentSystem
from threading import Thread
from time import sleep,time

from multiprocessing import Process

def Pipeline(recording=pipeline_config.recording_name, whisper_model=pipeline_config.whisper_model_sizes[0], trial=0):
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
    f"models/ggml-{whisper_model}.bin", 
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
    # "-mt",
    # str(pipeline_config.max_tokens)
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
 

    is_streaming = True

# # ===== Start Audio Streaming Subprocess module =================================================

    # audiostreamcommand = [
    # "./stream-file",
    # "-f", # uspecify which wavefile
    # f"{recording}.wav", 
    # ]
    

    WhisperSubprocess = subprocess.Popen(whispercppcommand, cwd='whisper.cpp/')


    sleep(6)

 
    print("======================= Whisper Warmup Done ======================")
 
 
    # ===== Start Audiostream module =========================================
    if(pipeline_config.action_recognition):
        audio_dir = f'./Audio_Scenarios/2023_Test/{recording}.wav'
    else:
        audio_dir = f'./Audio_Scenarios/2019_Test/{recording}.wav'
        
    Audiostream = Thread(
                    target=WhisperFileStream.Whisper, args=(SpeechToNLPQueue, EMSAgentQueue, audio_dir, is_streaming, SignalQueue))
    Audiostream.start()


    if(pipeline_config.action_recognition):
        clipcppcommand = [
            "./zsl",
            "-m", # use specific whisper model
            f"../models/{pipeline_config.clip_model}", 
            "-t", # threads
            str(pipeline_config.clip_threads),
            "--text",
            str("dummy1"),
            "--text",
            str("dummy2"),
            "--image",
            str(f"../../.././Video_Scenarios/2023_Test/{recording}.avi"),
            "--trial",
            str(trial)
            ]
            
        print("======================= Clip CPP Starting... ======================")
        sleep(5)
        ClipSubprocess = subprocess.Popen(clipcppcommand, cwd='clip.cpp/build/bin/')



# # ===== Start FIFO reading module =========================================
#     FIFOStream = Thread(
#                     target=FIFOStreamClient.Fifo, args=(SpeechToNLPQueue, EMSAgentQueue, SignalQueue))
#     FIFOStream.start()




    # audiostreamcommand = [
    # "python",
    # "AudioStream.py",
    # "-f", # use specific whisper model
    # f"./Audio_Scenarios/2019_Test/{recording}.wav", 
    # ]        
    
    # AudioStreamSubProcess = subprocess.Popen(audiostreamcommand, cwd='./')
    


# # ===== Start FIFO reading module =========================================
#     SignalFIFOStream = Thread(
#                     target=FIFOStreamClient.SignalFifo, args=(SignalQueue,))
#     SignalFIFOStream.start()

  # p.subprocess is alive
    #                # ===== Start Whisper Audiostream and Read module =========================================
    # Audiostream = Thread(
    #                 target=WhisperFileStream.SDStream, args=(SpeechSignalQueue, f'./Audio_Scenarios/2019_Test/{recording}.wav'))
    # Audiostream.start()
    
        
    #                      # ===== Start Whisper Audiostream and Read module =========================================
    # FIFOStream = Thread(
    #                 target=WhisperFileStream.ReadPipe, args=(SpeechToNLPQueue,VideoSignalQueue,SpeechSignalQueue))
    # FIFOStream.start()
    


    # # ===== Start Audio Streaming Subprocess module =================================================

        # audiostreamcommand = [
        # "./stream-file",
        # "-f", # uspecify which wavefile
        # f"{recording}.wav", 
        # ]
        
        # AudioStreamSubprocess = subprocess.Popen(audiostreamcommand, cwd='whisper.cpp/')





        





 
    

# ===== Exiting Program ====================================================
    '''
    When Whisper() of WhisperFileStream finishes running, that means recording is finished
    Then WhisperFileStream will queue 'Kill' to EMSAgent, which will break EMSAgent loop. This is a temporary hack
    TODO: Make a Event() from Threading module to use as flag signal instead of queueing the string 'Kill'
    '''
    Audiostream.join()
    # SignalFIFOStream.join()
    # FIFOStream.join()
    EMSAgent.join()
    
    
    # AudioStreamSubProcess.terminate()

    WhisperSubprocess.terminate()
    
    if(pipeline_config.action_recognition):
        ClipSubprocess.terminate()

    sleep(5)

if __name__ == '__main__':
    Pipeline()
