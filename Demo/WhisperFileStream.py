import pipeline_config
import wave
import pyaudio
import time
import re
from classes import TranscriptItem
import traceback
# import sounddevice as sd
# import soundfile as sf

import queue

import sys
import threading


# Wavefile recording parameters
RATE = 16000
CHUNK = 1600  # 100ms

                
if(pipeline_config.endtoendspv):
    RATE = 44100
    CHUNK = RATE//10
    
                    
SIGNAL = False

'''
This method processes the whisper response we receive from fifo pipe 
The response is type string, and is in the format 'block{isFinal,avg_p,latency}'
We remove background noise strings from the block (such as *crash* and [DOOR OPENS]) 
And separate out the parts of response into block, isFinal, avg_p, and latency

block : speech converted into text
isFinal : '1' = the block is a final block. '0' = the block is a interim block
avg_p : 'float' of inclusive range [0,1]. the average confidence probability of all the tokens in the block
latency : latency of whisper streaming per block in milliseconds
'''
def process_whisper_response(response):
    # used to remove the background noise transcriptions from Whisper output
    # Remove strings enclosed within parentheses
    response = re.sub(r'\([^)]*\)', '', response)
    # Remove strings enclosed within asterisks
    response = re.sub(r'\*[^*]*\*', '', response)
    # Remove strings enclosed within brackets
    response = re.sub(r'\[[^\]]*\]', '', response)
    # Remove null terminator since it does not display properly in Speech Box
    response = response.replace("\x00", " ")

    # separate transcript and confidence score
    start = response.find('{')
    end = response.find('}')
    block = response[:start]
    isFinal_str, avg_p_str, latency_str = response[start+1:end].split(",")
    isFinal = int(isFinal_str)
    avg_p = float(avg_p_str)
    latency = int(latency_str)

    return block, isFinal, avg_p, latency


def SDStream(SpeechSignalQueue,wavefile_name):

    blocksize = 1024
    buffersize = 20
    device = len(sd.query_devices()) -1

    q = queue.Queue(maxsize=20)
    event = threading.Event()
    print("Audio Streaming Started! ",wavefile_name)
    
    SpeechSignalQueue.put("Proceed")

    def callback(outdata, frames, time, status):
        assert frames == blocksize
        if status.output_underflow:
            print('Output underflow: increase blocksize?', file=sys.stderr)
            raise sd.CallbackAbort
        assert not status
        try:
            data = q.get_nowait()
        except queue.Empty:
            print('Buffer is empty: increase buffersize?', file=sys.stderr)
            raise sd.CallbackAbort
        if len(data) < len(outdata):
            outdata[:len(data)] = data
            outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
            raise sd.CallbackStop
        else:
            outdata[:] = data

        SpeechSignalQueue.put("Proceed")

    try:
        with sf.SoundFile(wavefile_name) as f:
            for _ in range(buffersize):
                data = f.buffer_read(blocksize, dtype='float32')
                if not data:
                    break
                q.put_nowait(data)  # Pre-fill queue
                
            print(f.samplerate)
            stream = sd.RawOutputStream(
                samplerate=f.samplerate, blocksize=blocksize,
                device=device, channels=f.channels, dtype='float32',
                callback=callback, finished_callback=event.set)
            with stream:
                timeout = blocksize * buffersize / f.samplerate
                while data:
                    data = f.buffer_read(blocksize, dtype='float32')
                    q.put(data, timeout=timeout)
                event.wait()  # Wait until playback is finished
                
                print("Audio streaming over!")
                SpeechSignalQueue.empty()
                SpeechSignalQueue.put("Kill")
                
    except KeyboardInterrupt:
        print('\nInterrupted by user')
    except queue.Full:
        # A timeout occurred, i.e. there was an error in the callback
        print('\nQueue Full')
    except Exception as e:
        print('\nException ',e)
        
        
    
    # # Extract data and sampling rate from file
    # data, fs = sf.read(wavefile_name, dtype='float32')  
    # sd.play(data, fs)
    # SpeechSignalQueue.put('Proceed')
    
    # print("Audio Stream started playing :",wavefile_name)
    
    # status = sd.wait()  # Wait until file is done playings
    
    # SpeechSignalQueue.put('Kill')
    
    # print("Audio Stream Done Playing. Sent Kill Signal. Bye!")
    

def PyAudioStream(SpeechSignalQueue,wavefile_name):
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
                # VideoSignalQueue.put('Proceed')
                SpeechSignalQueue.put("Proceed")

            # Close stream (4)
            stream.close()

            SpeechSignalQueue.empty()
            SpeechSignalQueue.put('Kill')
            # VideoSignalQueue.put('Kill')

            # Release PortAudio system resources (5)
            p.terminate()

            print("Audio Stream Thread Sent Kill Signal. Bye!")

        except Exception as e:
            print("EXCEPTION: ", e)
            traceback.print_exception(e)            



            
    




   
    
    
    
def ReadPipe(SpeechToNLPQueue,VideoSignalQueue, SpeechSignalQueue):
    
    fifo_path = "/tmp/myfifo"
    finalized_blocks = ''
        
    with open(fifo_path, 'r') as fifo:
        try:
            
            old_response = ""
            # Play samples from the wave file (3)
            
            if(not SpeechSignalQueue.empty()): signal = SpeechSignalQueue.get(block=False)
            else: signal = "Proceed"
                
            while signal != 'Kill':  # Requires Python 3.8+ for :=
                response = fifo.read().strip()  # Read the message from the named pipe
                VideoSignalQueue.put('Proceed')
                    
                if(not SpeechSignalQueue.empty()): signal = SpeechSignalQueue.get(block=False)
                else: signal = "Proceed"
                    
                    
                if response != old_response and response != "":
                    if(response.count("}")> 2 or response.count("{") > 2):
                        continue
                    block, isFinal, avg_p, latency = process_whisper_response(response) #isFinal = False means block is interim block
                    transcript = finalized_blocks + block
                    # if received block is finalized, then save to finalized blocks
                    
                    if isFinal: finalized_blocks += block
                    transcriptItem = TranscriptItem(transcript, isFinal, avg_p, latency)
                    # EMSAgentQueue.put(transcriptItem)
                    SpeechToNLPQueue.put(transcriptItem)  
                    
                    print("--- Whisper Latency:", latency)
                    old_response = response                

            SpeechToNLPQueue.put('Kill')
            VideoSignalQueue.put('Kill')
            
            

            print("FIFO Read Thread Sent Kill Signal. Bye!")

        except Exception as e:
            print("EXCEPTION: ", e)

def Whisper(SpeechToNLPQueue,VideoSignalQueue, wavefile_name):
    fifo_path = "/tmp/myfifo"
    finalized_blocks = ''
    VideoSignalQueue.put('Proceed')
        
    with open(fifo_path, 'r') as fifo:
        with wave.open(wavefile_name, 'rb') as wf:
            try:
                # Instantiate PyAudio and initialize PortAudio system resources (1)
                p = pyaudio.PyAudio()
                info = p.get_default_host_api_info()
                device_index = info.get('deviceCount') - 1 # get default device as output device

                stream = p.open(format = pyaudio.paInt16, channels = 1, rate = RATE, output = True, frames_per_buffer = CHUNK, output_device_index=device_index)
                
                old_response = ""
                # Play samples from the wave file (3)
                while len(data:=wf.readframes(CHUNK)):  # Requires Python 3.8+ for :=
                    stream.write(data)
                    try:
                        response = fifo.read().strip()  # Read the message from the named pipe
                    except Exception as e:
                        response = ""
                    VideoSignalQueue.put('Proceed')

                    if response != old_response and response != "":
                        block, isFinal, avg_p, latency = process_whisper_response(response) #isFinal = False means block is interim block
                        transcript = finalized_blocks + block
                        # if received block is finalized, then save to finalized blocks
                        if isFinal: finalized_blocks += block
                        # send transcript item if its not an empty string or space
                        if len(transcript) and not transcript.isspace():
                            transcriptItem = TranscriptItem(transcript, isFinal, avg_p, latency)
                            # EMSAgentQueue.put(transcriptItem)
                            SpeechToNLPQueue.put(transcriptItem)  
                        print("--- Whisper Latency:", latency)
                        old_response = response
                # Close stream (4)
                stream.close()

                SpeechToNLPQueue.put('Kill')
                VideoSignalQueue.put('Kill')

                # Release PortAudio system resources (5)
                p.terminate()

                print("Audio Stream Thread Sent Kill Signal. Bye!")

            except Exception as e:
                print("EXCEPTION: ", e)
                traceback.print_exception(e)            



            
    






