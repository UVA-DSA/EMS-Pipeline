
import os
import queue
import subprocess
import pipeline_config #rename pipeline_config
import WhisperFileStream
from EMSAgent import EMSAgentSystem
from threading import Thread
from time import sleep

from EMSVision import EMSVisionSystem
import VideoFileStream
from EMSWhisper import WhisperAgent

import multiprocessing


if pipeline_config.speech_model != 'whisper':
    import EMSConformerFileStream
    from EMSConformer.inference import run_tflite_model_in_files_easy
    from EMSConformer.inference import run_saved_model

def Pipeline(recording=pipeline_config.recording_name, videofile=pipeline_config.video_name, whisper_model=pipeline_config.whisper_model_size):
# Set the Google Speech API service-account key environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"

# ===== Create thread-safe queues to pass on transcript and communication between modules ============
    SpeechToNLPQueue = queue.Queue()
    FeedbackQueue = queue.Queue()
    VideoDataQueue = queue.Queue()
    # SpeechSignalQueue = Queue()
    VideoSignalQueue = queue.Queue()
    ConformerSignalQueue = queue.Queue()

    
    # ===== Start Speech Recognition module =================================================
    
    # Check speech model 
    if(pipeline_config.speech_model == 'whisper'):
        # ===== Start EMSAgent module =================================================
        if(not pipeline_config.speech_standalone):
            EMSAgent = Thread(target=EMSAgentSystem.EMSAgentSystem, args=(SpeechToNLPQueue, FeedbackQueue))
            EMSAgent.start()
            
            sleep(3)
        
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
        
        sleep(2)
        
        # ===== Start Whisper Audiostream and Read module =========================================
        if pipeline_config.endtoendspv:
            recording_dir = f'./Audio_Scenarios/2023_Test/{recording}.wav'
        else:
            recording_dir = f'./Audio_Scenarios/2019_Test/{recording}.wav'
            
        Audiostream = Thread(
                        target=WhisperFileStream.Whisper, args=(SpeechToNLPQueue,VideoSignalQueue, recording_dir))
        Audiostream.start()
        
        
        # #                # ===== Start Whisper Audiostream and Read module =========================================
        # Audiostream = Thread(
        #                 target=WhisperFileStream.PyAudioStream, args=(SpeechSignalQueue, f'./Audio_Scenarios/2019_Test/{recording}.wav'))
        # Audiostream.start()
        
        #         #  # ===== Start Whisper HuggingFace Pipeline module =========================================
        # WhisperPipeline = Thread(
        #                 target=WhisperAgent.WhisperPipeline, args=(SpeechToNLPQueue,SpeechSignalQueue))
        # WhisperPipeline.start()
        
           
        #                      # ===== Start Whisper Audiostream and Read module =========================================
        # FIFOStream = Thread(
        #                 target=WhisperFileStream.ReadPipe, args=(SpeechToNLPQueue,VideoSignalQueue,SpeechSignalQueue))
        # FIFOStream.start()
          
        if pipeline_config.endtoendspv:

            
        # ===== Start Video Action Recognition module =========================================
            EMSVision = Thread(
                            target=EMSVisionSystem.EMSVision, args=(FeedbackQueue, VideoDataQueue))
            EMSVision.start()
               
            sleep(10)
                           # # ===== Start Video Streaming module =========================================
            Videostream = Thread(
                            target=VideoFileStream.VideoStream, args=(VideoDataQueue, f'./Video_Scenarios/2023_Test/{recording}.avi'))
            Videostream.start()
            
    else:
                    
      # ===== Start EMSAgent module =================================================
        EMSAgent = Thread(target=EMSAgentSystem.EMSAgentSystem, args=(SpeechToNLPQueue, FeedbackQueue))
        EMSAgent.start()
        
        sleep(3)
    
        # tflite
        if(pipeline_config.conformer_model_type.endswith(".tflite")):
            EMSConformer = Thread(target=run_tflite_model_in_files_easy.main, args=(SpeechToNLPQueue,ConformerSignalQueue, f"./EMSConformer/speech_models/{pipeline_config.conformer_model_type}"))
            EMSConformer.start()

        else:
            # full model
            EMSConformer = Thread(target=run_saved_model.main, args=(SpeechToNLPQueue,ConformerSignalQueue))
            EMSConformer.start()
            
        sleep(25)

        # ===== Start Conformer Audiostream module =========================================
        Audiostream = Thread(
                        target=EMSConformerFileStream.ConformerStream, args=(SpeechToNLPQueue,VideoSignalQueue,ConformerSignalQueue, f'./Audio_Scenarios/2019_Test/{recording}.wav'))
        Audiostream.start()


# ===== Exiting Program ====================================================
    '''
    When Whisper() of WhisperFileStream finishes running, that means recording is finished
    Then WhisperFileStream will queue 'Kill' to EMSAgent, which will break EMSAgent loop. This is a temporary hack
    TODO: Make a Event() from Threading module to use as flag signal instead of queueing the string 'Kill'
    '''

    if(pipeline_config.speech_model == 'whisper'):
        # WhisperPipeline.join()
        if(pipeline_config.endtoendspv):
            Audiostream.join()
            Videostream.join()
            EMSVision.join()
            EMSAgent.join()
            WhisperSubprocess.kill()
        else:
            Audiostream.join()
            if(not pipeline_config.speech_standalone):
                EMSAgent.join()
            # EMSAgent.join()
            WhisperSubprocess.terminate()
    else: 
        EMSAgent.join()
        EMSConformer.join()
        Audiostream.join()

if __name__ == '__main__':
    Pipeline()