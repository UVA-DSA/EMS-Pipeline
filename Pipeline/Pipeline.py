
import os
import queue
import subprocess
import pipeline_config #rename pipeline_config
import WhisperFileStream
from EMS_TinyBERT import EMSTinyBERTSystem
from EMS_Vision import EMSVisionSystem
import VideoFileStream
from threading import Thread
from time import sleep
from PyQt5.QtWidgets import QApplication
from GUI import MainWindow, StartGUI
from multiprocessing import Process

if pipeline_config.speech_model != 'whisper':
    import EMSConformerFileStream
    from EMSConformer.inference import run_tflite_model_in_files_easy
    from EMSConformer.inference import run_saved_model



def Pipeline(recording=pipeline_config.recording_name, videofile=pipeline_config.video_name, whisper_model=pipeline_config.whisper_model_size):
    # Set the Google Speech API service-account key environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"
    
    # ===== Create thread-safe queues to pass on transcript and communication between modules ============
    TranscriptQueue = queue.Queue()
    ProtocolQueue = queue.Queue()
    VideoDataQueue = queue.Queue()
    # SpeechSignalQueue = Queue()
    VideoSignalQueue = queue.Queue()
    ConformerSignalQueue = queue.Queue()
    WindowQueue = queue.Queue()


    # ===== Create GUI ===================================================================================
    # GUI: Create the main window, show it, and run the app
    # GUIThread = Thread(target=StartGUI, args=(WindowQueue,))
    # GUIThread.start()
    # Window = WindowQueue.get()
    Window = None
    
    # ===== Start Speech Recognition module =================================================
    
    # Check speech model 
    if(pipeline_config.speech_model == 'whisper'):
        # ===== Start EMS_TinyBERT module =================================================
        if(not pipeline_config.speech_standalone):
            EMS_TinyBERT = Thread(target=EMSTinyBERTSystem.EMSTinyBERTSystem, args=(Window, TranscriptQueue, ProtocolQueue))
            EMS_TinyBERT.start()
            
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
        WhisperSubprocess = subprocess.Popen(whispercppcommand, cwd='EMS_Whisper/')
        
        sleep(2)
        
        # ===== Start Whisper Audiostream and Read module =========================================
        if pipeline_config.endtoendspv:
            recording_dir = f'./Scenarios/Audio_Scenarios/2023_Test/{recording}.wav'
        else:
            recording_dir = f'./Scenarios/Audio_Scenarios/2019_Test/{recording}.wav'
        
        Audiostream = Thread(
                        target=WhisperFileStream.Whisper, args=(Window, TranscriptQueue,VideoSignalQueue, recording_dir))
        Audiostream.start()
                  
        if pipeline_config.endtoendspv:
        # ===== Start Video Action Recognition module =========================================
            EMSVision = Thread(
                            target=EMSVisionSystem.EMSVision, args=(ProtocolQueue, VideoDataQueue))
            EMSVision.start()
               
            sleep(10)
                           # # ===== Start Video Streaming module =========================================
            Videostream = Thread(
                            target=VideoFileStream.VideoStream, args=(VideoDataQueue, f'./Scenarios/Video_Scenarios/2023_Test/{recording}.avi'))
            Videostream.start()
            
    else:
                    
      # ===== Start EMSTinyBERT module =================================================
        EMSTinyBERT = Thread(target=EMSTinyBERTSystem.EMSTinyBERTSystem, args=(TranscriptQueue, ProtocolQueue))
        EMSTinyBERT.start()
        
        sleep(3)
    
        # tflite
        if(pipeline_config.conformer_model_type.endswith(".tflite")):
            EMSConformer = Thread(target=run_tflite_model_in_files_easy.main, args=(TranscriptQueue,ConformerSignalQueue, f"./EMSConformer/speech_models/{pipeline_config.conformer_model_type}"))
            EMSConformer.start()

        else:
            # full model
            EMSConformer = Thread(target=run_saved_model.main, args=(TranscriptQueue,ConformerSignalQueue))
            EMSConformer.start()
            
        sleep(25)

        # ===== Start Conformer Audiostream module =========================================
        Audiostream = Thread(
                        target=EMSConformerFileStream.ConformerStream, args=(TranscriptQueue,VideoSignalQueue,ConformerSignalQueue, f'./Scenarios/Audio_Scenarios/2019_Test/{recording}.wav'))
        Audiostream.start()


# ===== Exiting Program ====================================================
    '''
    When Whisper() of WhisperFileStream finishes running, that means recording is finished
    Then WhisperFileStream will queue 'Kill' to EMS_TinyBERT, which will break EMS_TinyBERT loop. This is a temporary hack
    TODO: Make a Event() from Threading module to use as flag signal instead of queueing the string 'Kill'
    '''

    if(pipeline_config.speech_model == 'whisper'):
        # WhisperPipeline.join()
        if(pipeline_config.endtoendspv):
            Audiostream.join()
            Videostream.join()
            EMSVision.join()
            EMS_TinyBERT.join()
            WhisperSubprocess.kill()
        else:
            Audiostream.join()
            if(not pipeline_config.speech_standalone):
                EMS_TinyBERT.join()
            # EMS_TinyBERT.join()
            WhisperSubprocess.terminate()
    else: 
        EMSTinyBERT.join()
        EMSConformer.join()
        Audiostream.join()
        
    # GUIThread.join()

if __name__ == '__main__':
    Pipeline()