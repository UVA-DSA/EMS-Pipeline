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
import threading

class WhisperCore():
    def __init__(self, audio_queue, pipe_conn, stop_event):
        self.previous_segments = []
        self.previous_segments_lock = threading.Lock()
        self.stop_event = stop_event

        # set variables read from whisper_config.py
        self.model_name = whisper_config.model_name
        self.VAD_THRESHOLD = whisper_config.VAD_threshold
        self.USE_VAD = whisper_config.use_vad
        self.AUDIO_SEGMENT_LENGTH = whisper_config.AUDIO_SEGMENT_DURATION_LIMIT
        self.FINALIZATION_LIMIT = whisper_config.finalization_limit

        self.new_chunk = []
        self.new_chunk_duration = 0
        self.transcription = ""

        self.queue = audio_queue
        self.pipe_conn = pipe_conn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set device to cuda if available, otherwise cpu
        self.vad_model, self.vad_model_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                    model='silero_vad',
                                    force_reload=True)

        self.transcription_queue = queue.Queue()  # queue to transfer transcriptions to WhisperAPI
        self.current_segment = []
        self.new_chunk = []
        self.current_segment_duration = 0

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
            if not self.transcription_queue.empty():
                # print("hello from write_pipe")
                # print(self.transcription, end=" ")
                pipe_conn.send(self.transcription_queue.get())


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
        self.new_chunk.extend(audio_frame)

        # if the frame is classified as silence (meaning chunk ends on silence), try adding audio chunk to current segment.
        if not self.USE_VAD or speech_prob <= self.VAD_THRESHOLD: # if silence detected
            # if adding new_chunk to current_segment results in the current segment length > SEGMENT_LENGTH_LIMIT, start new segment
            # otherwise, append new_chunk to the current segment

            if self.calculate_duration_of_buffer(self.new_chunk) + self.current_segment_duration <= self.AUDIO_SEGMENT_LENGTH:
                self.current_segment.extend(self.new_chunk)
                self.current_segment_duration += self.calculate_duration_of_buffer(self.new_chunk)

            else:
                with self.previous_segments_lock:
                    self.previous_segments.append(self.current_segment)
                self.current_segment = self.new_chunk
                self.current_segment_duration = self.calculate_duration_of_buffer(self.current_segment)

            self.new_chunk = []
    

    def get_audio_input_to_transcribe(self):
        # this method is used to generate a segment of audio to transcribe.
        # it is called in transcription_worker
        audio = []
        segment_index = 0
        input_duration = 0
        with self.previous_segments_lock:

            is_final = False    # variable storing whether this will be the last time we are seeing the audio samples involved

            while segment_index < len(self.previous_segments):
                curr_segment = self.previous_segments[segment_index]
                current_segment_duration = self.calculate_duration_of_buffer(curr_segment)

                if self.FINALIZATION_LIMIT == -1:   # immediate finalization
                    audio = curr_segment
                    self.previous_segments = self.previous_segments[segment_index+1:]
                    return curr_segment, True

                elif current_segment_duration + input_duration < self.FINALIZATION_LIMIT:
                    audio += curr_segment
                    input_duration += current_segment_duration
                    segment_index += 1

                else:
                    # we've reached the finaliaztion limit, so we need to cut out the corresponding audio segments from prev_segments
                    self.previous_segments = self.previous_segments[segment_index:]
                    is_final = True
                    break

        return audio, is_final


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


    def transcription_worker(self):
        while not self.stop_event.is_set():
            time.sleep(0.1)
            if len(self.previous_segments) > 0:
                audio, finalized_status = self.get_audio_input_to_transcribe()
                if whisper_config.mode == "whispercpp":
                    transcription = self.transcribe_with_whispercpp(audio)

                elif whisper_config.mode == "huggingface":
                    transcription = self.transcribe_with_huggingface(audio)
            
                self.transcription_queue.put((transcription, finalized_status))

        

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
                        "--print-colors", # display color of confidence value for word
                        # "--log-score", # write confidence value for each token in score.txt file
                        "-f",
                        INTERMEDIATE_AUDIO_FILE] # output file name
                    
                    # print("Running the following command: ", " ".join(commands))
                    subprocess.run(commands)   # run cpp executable with the commands
                finally:
                    transcription = self.read_from_file(INTERMEDIATE_TRANSCRIPTION_TEXT_FILE)

                    os.chdir(current_dir)   # change back to original directory
                                
                return transcription


