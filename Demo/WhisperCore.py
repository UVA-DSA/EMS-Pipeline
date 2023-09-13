import os
from StoppableThread.StoppableThread import StoppableThread
import numpy as np
import re
import time
from transformers import pipeline, WhisperProcessor
import torch
import whisper_config
import soundfile as sf
import subprocess
import queue

class WhisperCore():
    def __init__(self, audio_queue, pipe_conn):
        # set variables read from whisper_config.py
        self.model_name = whisper_config.model_name
        self.VAD_THRESHOLD = whisper_config.VAD_threshold
        self.CHUNK_DURATION_LIMIT = whisper_config.AUDIO_SEGMENT_DURATION_LIMIT

        self.new_audio = []
        self.new_audio_duration = 0
        self.transcription = ""

        self.queue = audio_queue
        self.pipe_conn = pipe_conn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set device to cuda if available, otherwise cpu
        self.vad_model, self.vad_model_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                    model='silero_vad',
                                    force_reload=True)

        self.transcription_queue = queue.Queue()  # queue to transfer transcriptions to WhisperAPI
        self.current_chunk = []
        self.new_audio = []
        self.current_chunk_duration = 0

        # load huggingface models
        if whisper_config.mode == "huggingface":
            hf_user = "saahith/" if 'combined' in self.model_name else "openai/"
            model_repo = hf_user + self.model_name
            self.processor = WhisperProcessor.from_pretrained(model_repo)
            self.model_pipe = pipeline(model=model_repo)


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

        transcription_thread = StoppableThread(target=self.transcription_worker, args=())
        transcription_thread.start()

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
        speech_prob = self.vad_model(audio_frame, 16000)
        self.new_audio.extend(audio_frame)

        # if the new audio is long enough, add it to the transcription queue
        if self.current_chunk_duration >= self.CHUNK_DURATION_LIMIT:
            self.transcription_queue.put(self.current_chunk)
            self.current_chunk = self.new_audio
            self.current_chunk_duration = self.calculate_duration_of_buffer(self.current_chunk)
            self.new_audio = []

        # if the frame is classified as non-speech, try adding saved audio to the current chunk
        elif speech_prob < self.VAD_THRESHOLD: # if silence detected
            # if adding new_audio to current_chunk results in chunk duration < 30 seconds, create new chunk
            # otherwise, add it to the current chunk

            if self.calculate_duration_of_buffer(self.new_audio) + self.current_chunk_duration < self.CHUNK_DURATION_LIMIT:
                self.current_chunk.extend(self.new_audio)
                self.current_chunk_duration += self.calculate_duration_of_buffer(self.new_audio)

            else:
                self.transcription_queue.put(self.current_chunk)
                self.current_chunk = self.new_audio
                self.current_chunk_duration = self.calculate_duration_of_buffer(self.current_chunk)

            self.new_audio = []

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


    def get_longest_audio_chunk_to_transcribe(self):
        count = 0
        audio = []
        for i in range(30 // whisper_config.AUDIO_SEGMENT_DURATION_LIMIT):
            if self.transcription_queue.empty():
                # self.transcription += " Running WhisperCPP on " + str(count) + " chunks of audio. \n\n"
                return audio
            else:
                count += 1
                audio += self.transcription_queue.get()
        # self.transcription += " Running WhisperCPP on " + str(count) + " chunks of audio. \n\n"
        return audio
        

    def transcription_worker(self):
        while True:
            time.sleep(0.1)
            if self.transcription_queue.empty():
                continue
            
            if whisper_config.mode == "whispercpp":
                transcription = self.transcribe_with_whispercpp(self.get_longest_audio_chunk_to_transcribe())
                self.transcription += transcription + " "

            elif whisper_config.mode == "huggingface":
                transcription = self.transcribe_with_huggingface(self.get_longest_audio_chunk_to_transcribe())
                self.transcription += transcription + " "
        

    def transcribe_with_huggingface(self, audio_data):
        audio_data = np.array(audio_data)
        # normalize the text
        normalized_text = self.processor.tokenizer._normalize(self.model_pipe(audio_data)['text'])
        return normalized_text


    def transcribe_with_whispercpp(self, audio_data, num_threads=4):
            # write audio to file
            # run whisper subprocess
            # read transcript from file and return
            while True:
                time.sleep(0.1)
                current_dir = os.getcwd()
                INTERMEDIATE_AUDIO_FILE = "intermediate_audio.wav"
                # INTERMEDIATE_TRANSCRIPTION_TEXT_FILE = os.path.join(current_dir, "intermediate_transcription.txt")
                INTERMEDIATE_TRANSCRIPTION_TEXT_FILE = "intermediate_transcription.txt"
                ggml_name = self.model_name.replace("whisper-", "")
                
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
                                
                return transcription


