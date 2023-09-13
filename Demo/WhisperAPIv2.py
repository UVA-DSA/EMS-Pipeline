from StoppableThread.StoppableThread import StoppableThread
import time
import multiprocessing
from WhisperCore import WhisperCore

audio_frame_queue = multiprocessing.Queue()    # queue to transfer audio frames to WhisperCore

class WhisperAPIv2():
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
            audio_frame_queue.put(audio_data)


    def transcribe_stream(self, audio_stream):
        # populate audio queue read by WhisperCore
        audio_population_thread = StoppableThread(target=self.populate_audio_queue, args=(audio_stream,))
        audio_population_thread.start()

        print("started audio population thread")
        # create a pipe to transfer transcription outputs from WhisperCore to this thread
        parent_conn, child_conn = multiprocessing.Pipe()

        # create a process to run WhisperCore
        whisper_core_process = multiprocessing.Process(target=self.run_whisper_core, args=(audio_frame_queue, child_conn))
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


    def run_whisper_core(self, audio_frame_queue, pipe_conn):

        # responsible for running WhisperCore as  
        print("running whisper core from api")
        whisper_core = WhisperCore(audio_frame_queue, pipe_conn)
        whisper_core.run()