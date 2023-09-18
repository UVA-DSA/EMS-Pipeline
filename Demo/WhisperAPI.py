from StoppableThread.StoppableThread import StoppableThread
import time
import multiprocessing
from WhisperCore import WhisperCore

audio_frame_queue = multiprocessing.Queue()    # queue to transfer audio frames to WhisperCore

class WhisperAPI():
    # make sure to start all processes as stoppable threads
    def __init__(self):
        self.transcription_output_queue = []
        self.stop_event = multiprocessing.Event()


        
    def read_pipe(self, pipe_conn):
        # responsible for reading pipe for transcription outputs and updated the transcription_output variable
        while True:
            transcription_data = pipe_conn.recv()
            self.transcription_output_queue.append(transcription_data)


    def populate_audio_queue(self, audio_generator):
        # responsible for populating the audio queue with audio data from audio stream
        for audio_data in audio_generator:
            audio_frame_queue.put(audio_data)


    def transcribe_stream(self, audio_stream):
        # populate audio queue read by WhisperCore
        audio_population_thread = StoppableThread(target=self.populate_audio_queue, args=(audio_stream,))
        audio_population_thread.start()

        print("started audio population thread")
        # create a pipe to transfer transcription outputs from WhisperCore to this thread
        parent_conn, child_conn = multiprocessing.Pipe()

        # create a process to run WhisperCore
        whisper_core_process = multiprocessing.Process(target=self.run_whisper_core, args=(audio_frame_queue, child_conn, self.stop_event))
        whisper_core_process.start()

        # create a thread to read from the pipe and update the transcription
        transcription_results_thread = StoppableThread(target=self.read_pipe, args=(parent_conn,))
        transcription_results_thread.start()

        print("started whisper core process")

        # input a stream of audio
        # create a subprocess that runs silero vad and whispercpp
        # output transcription (if needed)
        while True:
            time.sleep(0.1)
            yield self.generate_output_to_yield()


    def generate_output_to_yield(self):
        if len(self.transcription_output_queue) == 0:
            return {
                "transcript": None,
                "finalized": False
            }
        
        transcription, finalized = self.transcription_output_queue[0]
        self.transcription_output_queue = self.transcription_output_queue[1:]
        return {
            "transcript": transcription,
            "finalized": finalized
        }
        
    def run_whisper_core(self, audio_frame_queue, pipe_conn, stop_event):
        # responsible for running WhisperCore as  
        print("running whisper core from api")
        whisper_core = WhisperCore(audio_frame_queue, pipe_conn, stop_event)
        whisper_core.run()