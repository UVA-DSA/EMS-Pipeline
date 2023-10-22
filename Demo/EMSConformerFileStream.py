import queue
import wave
import pyaudio
import time
import re
from classes import TranscriptItem
import traceback

# Wavefile recording parameters
RATE = 16000
CHUNK = RATE // 10  # 100ms

def ConformerStream(SpeechToNLPQueue,VideoSignalQueue, ConformerSignalQueue, wavefile_name):
    print('Conformer started!')
    with wave.open(wavefile_name, 'rb') as wf:
        try:
            # Instantiate PyAudio and initialize PortAudio system resources (1)
            p = pyaudio.PyAudio()
            info = p.get_default_host_api_info()
            device_index = info.get('deviceCount') - 1 # get default device as output device
                
            stream = p.open(format = pyaudio.paInt16, channels = 1, rate = RATE, output = True, frames_per_buffer = CHUNK, output_device_index=device_index)
            
            # Play samples from the wave file (3)
            while len(data:=wf.readframes(CHUNK)):  # Requires Python 3.8+ for :=
                stream.write(data)
                VideoSignalQueue.put('Proceed')
                ConformerSignalQueue.put('Proceed')

            # Close stream (4)
            stream.close()

            SpeechToNLPQueue.put('Kill')
            VideoSignalQueue.put('Kill')
            ConformerSignalQueue.queue.clear()
            ConformerSignalQueue.put('Kill')

            # Release PortAudio system resources (5)
            p.terminate()

            print("EMS Conformer Audio Stream Thread Sent Kill Signal. Bye!")
            
            return

        except Exception as e:
            print("EXCEPTION: ", e)
            traceback.print_exception(e)            



        







