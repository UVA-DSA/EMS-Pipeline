import os
from StoppableThread.StoppableThread import StoppableThread
import numpy as np
import re
import threading    # for Lock() object
import time
from transformers import pipeline, WhisperProcessor
import torch
import whisper_config

class WhisperAPI():
    # make sure to start all processes as stoppable threads

    def __init__(self):
        # set variables read from whisper_config.py
        self.model_name = whisper_config.model_name
        self.VAD_THRESHOLD = whisper_config.VAD_threshold
        self.CHUNK_DURATION_LIMIT = whisper_config.AUDIO_SEGMENT_DURATION_LIMIT

    """
    transcribe_stream is the outward facing method that will input an audio generator
    and output a transcription generator

    initialize the chunk dict, new_audio array, current_chunk array, and chunk_transcription array

    start a parallel transcription thread

    once a frame is ingested, run it through voice activity detection and add it the relevant buffers
    """


    #### START OF STREAMING RELATED METHODS ####
    def initialize_variables_for_streaming(self):
        self.chunks_dict = {}   # dict of chunks to be transcribed, with key = chunk index, value = (chunk_version, chunk)
        self.chunk_transcripts = []  # list of transcripts for each chunk

        self.finalized_transcript = "" # portion of the transcript that has been finalized
        self.unfinalized_transcript = "" # portion of the transcript that has not been finalized

        self.need_to_yield = False  # flag to indicate whether we need to yield the current transcript (if there have been changes since the last yield)
        
        self.current_chunk = [] # the current chunk of audio, to which we are adding new audio
        self.new_audio = [] # new audio that has been added to the end of current chunk
        
        self.chunk_lock = threading.Lock()  # lock to ensure that only one thread is accessing the chunks_dict at a time

        self.chunk_idx = 0 # index of the current chunk being ingested
        self.chunk_version = 0 # version of the current chunk being ingested
        self.index_to_yield = -1 # index of the chunk to yield
        self.finalized_chunk_indices_to_be_yielded = []
        self.chunk_indices_lock = threading.Lock() # lock to ensure that only one thread is accessing the finalized_chunk_indices_to_be_yielded at a time

        # load vad model
        # import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set device to cuda if available, otherwise cpu
        self.vad_model, self.vad_model_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                    model='silero_vad',
                                    force_reload=True)
        

        # load huggingface whisper model
        # from transformers import pipeline, WhisperProcessor
        hf_user = "saahith/" if 'combined' in self.model_name else "openai/"
        model_repo = hf_user + self.model_name
        self.whisper_processor = WhisperProcessor.from_pretrained(model_repo)
        self.whisper_pipe = pipeline(model=model_repo)
        # print("variables set up")
        

    def generate_output_to_yield(self):
        if len(self.finalized_chunk_indices_to_be_yielded) > 0:
                # with self.chunk_indices_lock:
                with open('transcript.txt', 'a') as f:
                    f.write("Reading from generate_output_to_yield: " + str(self.finalized_chunk_indices_to_be_yielded) + "\n")

                chunk_index_to_yield = self.finalized_chunk_indices_to_be_yielded[0]
                
                with open('transcript.txt', 'a') as f:
                    f.write("Yielding chunk index: " + str(chunk_index_to_yield) + "\n")

                self.finalized_chunk_indices_to_be_yielded = self.finalized_chunk_indices_to_be_yielded[1:]

                transcript = self.chunk_transcripts[chunk_index_to_yield]
                transcript = self.remove_strings_in_parentheses_and_asterisks(transcript)


                return {
                    "transcript": transcript,
                    "finalized": True
                }
                # return {
                #     "transcript": transcript,
                #     "finalized_transcript": self.finalized_transcript,
                #     "unfinalized_transcript": self.unfinalized_transcript,
                #     "finalized": len(self.chunks_dict) == 0,
                #     "latest_finalized_transcript": latest_finalized_transcript
                # }

        return {
            "transcript": None,
            "finalized_transcript": None,
            "unfinalized_transcript": None,
            "finalized": False
        }
    

    def ingest_audio(self, audio_frame):
        # normalize the audio
        audio_frame = self.int2float(np.frombuffer(audio_frame,np.int16))
        speech_prob = self.vad_model(audio_frame, 16000)
        self.new_audio.extend(audio_frame)
        # print(f"self.chunk_dict size: {len(self.chunks_dict.keys())}")
        if speech_prob < self.VAD_THRESHOLD: # if silence detected
            # if adding new_audio to current_chunk results in chunk duration < 30 seconds, create new chunk
            # otherwise, add it to the current chunk
            if self.calculate_duration_of_buffer(self.current_chunk) + self.calculate_duration_of_buffer(self.new_audio) < self.CHUNK_DURATION_LIMIT:
                self.current_chunk.extend(self.new_audio)
                self.chunk_version += 1

            else:
                self.chunk_idx += 1
                self.chunk_version = 0
                self.current_chunk = self.new_audio
            
            self.new_audio = []
            with self.chunk_lock:
                self.chunks_dict[self.chunk_idx] = (self.chunk_version, self.current_chunk)


            with open('transcript.txt', 'a') as f:
                f.write(str(self.chunks_dict.keys())  + "\n")


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


    # inputs a genereator/stream of audio data and returns a generator of transcriptions
    # to be used in the WhisperMicStream and WhisperFileStream classes
    def transcribe_stream(self, audio_generator):
        # requests is a generator of audio data
        self.initialize_variables_for_streaming()

        # start a parallel transcription thread
        transcription_thread = StoppableThread(target=self.transcription_worker)
        transcription_thread.start()

        for audio_data in audio_generator:
            # transcribe the audio data
            self.ingest_audio(audio_data)
            yield self.generate_output_to_yield()
        
        # keep yielding if we're not done transcribing the audio
        while len(self.chunks_dict) > 0:
            time.sleep(0.5)
            yield self.generate_output_to_yield()


    def transcription_worker(self):
        # read current transcription buffer to file
        while True:
            # run whisper as subprocess (with multithreading) and write transcript to file
            # read from transcript file and update current chunk
            time.sleep(0.5)

            with open('transcript.txt', 'a') as f:
                f.write("transcription worker: " + str(self.chunks_dict.keys()) + "\n")
            # if there is a chunk to read
            if self.chunks_dict.keys():

                with open('transcript.txt', 'a') as f:
                    f.write(str(self.chunks_dict.keys()) + "\n")
                

                chunk_idx_to_transcribe = min(self.chunks_dict.keys())
                with open('transcript.txt', 'a') as f:
                    f.write("chunk_idx_to_transcribe: " + str(chunk_idx_to_transcribe) + "\n")
                    f.write("self.chunk_idx: " + str(self.chunk_idx) + "\n")

                if chunk_idx_to_transcribe == self.chunk_idx:
                    continue
                
                with self.chunk_lock:
                    chunk_version, chunk = self.chunks_dict[chunk_idx_to_transcribe]
                    chunk_copy = chunk.copy()   # in case other threads modify the chunk while we're transcribing it
                
                # make sure we have a spot in chunk_transcripts to store the transcription
                while len(self.chunk_transcripts) < chunk_idx_to_transcribe + 1:
                    self.chunk_transcripts.append("")

                self.chunk_transcripts[chunk_idx_to_transcribe] = self.transcribe_with_huggingface(chunk_copy)
                
                # with open('transcript.txt', 'a') as f:
                #     f.write(f"Transcribed {chunk_idx_to_transcribe}\n")
                #     f.write(f"transcription: {self.chunk_transcripts[chunk_idx_to_transcribe]}\n")
                #     f.write(f"chunk_idx_to_transcribe: {chunk_idx_to_transcribe}")
                #     f.write(f"chunk_version: {chunk_version}")
                #     f.write(f"self.chunk_idx: {self.chunk_idx}")
                #     f.write("-----------------------------------------\n\n\n\n")


                # with self.chunk_indices_lock:
                self.finalized_chunk_indices_to_be_yielded.append(chunk_idx_to_transcribe)
                
                # with open('transcript.txt', 'a'):
                #     f.write(str(self.finalized_chunk_indices_to_be_yielded) + "\n")
                
                # with open('transcript.txt', 'a') as f:
                #     f.write("about to delete chunk\n")
                
                # with chunk_lock:
                del self.chunks_dict[chunk_idx_to_transcribe]
                
                # with open('transcript.txt', 'a') as f:
                #     f.write(f"Deleted chunk {chunk_idx_to_transcribe} from chunks_dict\n")

                # self.need_to_yield = True   # set flag to true to indicate that we need to yield the new transcript
                # self.index_to_yield = chunk_idx_to_transcribe

                # remove the chunk from chunks_dict if it hasn't been modified since we started transcribing it


    def transcribe_with_huggingface(self, audio_data):
        # run whisper pipe and save the output
        audio_data = np.array(audio_data)
        output = self.whisper_pipe(audio_data)['text']
        # normalize the text
        normalized_text = self.whisper_processor.tokenizer._normalize(output)
        return normalized_text


    def transcribe_with_whispercpp(self, audio_data, num_threads=4):
        global MODEL_NAME
        # write audio to file
        # run whisper subprocess
        # read transcript from file and return
        current_dir = os.getcwd()

        # write audio to file
        write_audio_to_file(audio_data, INTERMEDIATE_AUDIO_FILE)
        try:
            # ./main -f  samples/jfk.wav -m models/ggml-tiny.en.bin -otxt -of hellothere
            os.chdir(PATH_TO_WHISPER_CPP_FOLDER)
            commands = [
                "./main",
                "-m",   # specify the whisper model to be used
                f"models/ggml-{MODEL_NAME}.bin",    # specify the whisper model to be used
                "-otxt",   # output to text file
                "-of",    # output file name
                INTERMEDIATE_TRANSCRIPTION_TEXT_FILE[:-4], # remove .txt from the end of the file name
                "--threads",   # number of threads
                str(num_threads),
                "-f",
                INTERMEDIATE_AUDIO_FILE] # output file name
            subprocess.run(commands, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)   # run cpp executable with the commands
        finally:
            os.chdir(current_dir)   # change back to original directory
        return read_from_file(INTERMEDIATE_TRANSCRIPTION_TEXT_FILE)


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