import os
from StoppableThread.StoppableThread import StoppableThread
import numpy as np
import re
import threading    # for Lock() object
import time
from transformers import pipeline, WhisperProcessor
import torch
import whisper_config
import multiprocessing
import soundfile as sf
import subprocess
import sys

queue = multiprocessing.Queue()    # queue to transfer audio frames to WhisperCore


class naiveWhisperAPI():
    # make sure to start all processes as stoppable threads
    def __init__(self):
        self.transcription_output = []
        # self.transcription_output_lock = threading.Lock()
        

    def read_pipe(self, pipe_conn):
        # responsible for reading pipe for transcription outputs and updated the transcription_output variable
        while True:
            transcription = pipe_conn.recv()
            if transcription == "STOP":
                break
            # with self.transcription_output_lock:
            self.transcription_output.append(transcription)


    def populate_audio_queue(self, audio_generator):
        # responsible for populating the audio queue with audio data from audio stream
        for audio_data in audio_generator:
            queue.put(audio_data)


    def transcribe_stream(self, audio_stream):
        # populate audio queue read by WhisperCore
        audio_population_thread = StoppableThread(target=self.populate_audio_queue, args=(audio_stream,))
        audio_population_thread.start()

        print("started audio population thread")
        # create a pipe to transfer transcription outputs from WhisperCore to this thread
        parent_conn, child_conn = multiprocessing.Pipe()


        # create a process to run WhisperCore
        whisper_core_process = multiprocessing.Process(target=self.run_whisper_core, args=(queue, child_conn))
        whisper_core_process.start()


        # create a thread to read from the pipe and update the transcription
        transcription_results_thread = StoppableThread(target=self.read_pipe, args=(parent_conn,))
        transcription_results_thread.start()


        # whisper_core_process.join()

        # audio_population_thread.stop()
        # audio_population_thread.join()

        # transcription_results_thread.stop()
        # transcription_results_thread.join()
        print("started whisper core process")



        # input a stream of audio
        # create a subprocess that runs silero vad and whispercpp
        # output transcription (if needed)
        while True:
            time.sleep(0.4)
            yield self.generate_output_to_yield()


    def generate_output_to_yield(self):
        if len(self.transcription_output) > 0:
            # with self.transcription_output_lock:
            transcription = " ".join(self.transcription_output)
            self.transcription_output = []

            return {
                "transcript": transcription,
                "finalized": True
            }
        
        return {
            "transcript": None,
            "finalized": False
        }


    def run_whisper_core(self, audio_queue, pipe_conn):

        # responsible for running WhisperCore as  
        print("running whisper core from api")
        whisper_core = WhisperCore(audio_queue, pipe_conn)
        whisper_core.run()



################################################################################################################################################################################



class WhisperCore():
    def __init__(self, audio_queue, pipe_conn):
        # set variables read from whisper_config.py
        self.model_name = whisper_config.model_name
        # self.VAD_THRESHOLD = whisper_config.VAD_threshold
        self.CHUNK_DURATION_LIMIT = whisper_config.AUDIO_SEGMENT_DURATION_LIMIT
        self.INTERMEDIATE_AUDIO_FILE = "intermediate_audio.wav"
        self.new_audio = []
        self.new_audio_duration = 0
        self.transcription = ""

        self.queue = audio_queue
        self.pipe_conn = pipe_conn


    def write_pipe(self, pipe_conn):
        # responsible for writing to the pipe
        while True:
            time.sleep(0.4)
            if len(self.transcription) > 0:
                # print("hello from write_pipe")
                print(self.transcription, end=" ")
                pipe_conn.send(self.transcription)
                self.transcription = ""


    def run(self):
        # read audio frames from the queue and process them
        # create a thread to write to the pipe and update the transcription
        transcription_results_thread = StoppableThread(target=self.write_pipe, args=(self.pipe_conn,))
        transcription_results_thread.start()

        # print("started transcription results thread")
        while True:
            try:
                audio_frame = self.queue.get(timeout=1)
                self.ingest_audio(audio_frame)   
            except Exception as e:
                print(e)
                break
        
        transcription_results_thread.stop()
        transcription_results_thread.join()
        

    def ingest_audio(self, audio_frame):
        # normalize the audio
        # print("ingesting audio")
        audio_frame = self.int2float(np.frombuffer(audio_frame,np.int16))
        # speech_prob = self.vad_model(audio_frame, 16000)
        self.new_audio.extend(audio_frame)
        self.new_audio_duration += self.calculate_duration_of_buffer(audio_frame)

        # print("new audio duration: ", self.new_audio_duration)
        # if the new audio is long enough, add it to the chunks_dict
        
        if self.new_audio_duration >= self.CHUNK_DURATION_LIMIT:
            # result_queue = multiprocessing.Queue()
            # # transcription = transcribe_with_whispercpp(self.new_audio, model_name=self.model_name)
            # process = multiprocessing.Process(target=transcribe_with_whispercpp, args=(self.new_audio, self.model_name, result_queue))
            # process.start()
            # process.join()
            # transcription = result_queue.get()
            transcription = self.transcribe_with_whispercpp(self.new_audio, model_name=self.model_name)
            self.transcription += transcription + " "

            self.new_audio = []
            self.new_audio_duration = 0


    # audio normalization code provided by Alexander Veysov in silero-vad repository
    def int2float(self, sound):
        _sound = np.copy(sound)  #
        abs_max = np.abs(_sound).max()
        _sound = _sound.astype('float32')
        if abs_max > 0:
            _sound *= 1/abs_max
        audio_float32 = torch.from_numpy(_sound.squeeze())
        return audio_float32


    def remove_strings_in_parentheses_and_asterisks(self, transcription):
        # used to remove the background noise transcriptions from Whisper output
        # Remove strings enclosed within parentheses
        transcription = re.sub(r'\([^)]*\)', '', transcription)
        # Remove strings enclosed within asterisks
        transcription = re.sub(r'\*[^*]*\*', '', transcription)
        # Remove strings enclosed within brackets
        transcription = re.sub(r'\[[^\]]*\]', '', transcription)
        return transcription


    def calculate_duration_of_buffer(self, buffer):   
            return len(buffer) * (1 / 16000)    # multiply number of blocks by seconds per block


    def write_audio_to_file(self, audio_data, file_path):
            # Write the audio to a file - Helper method for interacting with whispercpp
            with open(file_path, "w") as file:  # first clear the file of any existing data
                file.truncate()
            sf.write(file_path, audio_data, 16000)


    def read_from_file(self, file_path):
            # read the audio data from a file - helper method for interacting with whispercpp
            with open(file_path, "r") as file:
                return file.read()


    def transcribe_with_whispercpp(self, audio_data, model_name="whisper-tiny.en", num_threads=4):
            # write audio to file
            # run whisper subprocess
            # read transcript from file and return
            current_dir = os.getcwd()
            INTERMEDIATE_AUDIO_FILE = "intermediate_audio.wav"
            # INTERMEDIATE_TRANSCRIPTION_TEXT_FILE = os.path.join(current_dir, "intermediate_transcription.txt")
            INTERMEDIATE_TRANSCRIPTION_TEXT_FILE = "intermediate_transcription.txt"
            ggml_name = model_name.replace("whisper-", "")
            
            os.chdir(whisper_config.PATH_TO_WHISPER_CPP_FOLDER)
            # write audio to file
            self.write_audio_to_file(audio_data, INTERMEDIATE_AUDIO_FILE)
            try:
                # ./main -f  samples/jfk.wav -m models/ggml-tiny.en.bin -otxt -of hellothere
                commands = [
                    "./main",
                    "-m",   # specify the whisper model to be used
                    f"models/ggml-{ggml_name}.bin",    # specify the whisper model to be used
                    "-otxt",   # output to text file
                    "-of",    # output file name
                    INTERMEDIATE_TRANSCRIPTION_TEXT_FILE[:-4], # remove .txt from the end of the file name
                    "--threads",   # number of threads
                    str(num_threads),
                    "-f",
                    INTERMEDIATE_AUDIO_FILE] # output file name
                
                # print("Running the following command: ", " ".join(commands))
                subprocess.run(commands, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)   # run cpp executable with the commands
            finally:
                transcription = self.read_from_file(INTERMEDIATE_TRANSCRIPTION_TEXT_FILE)

                os.chdir(current_dir)   # change back to original directory
            
            # print("hello")
            # print("transcription: ", transcription)
            
            return transcription


