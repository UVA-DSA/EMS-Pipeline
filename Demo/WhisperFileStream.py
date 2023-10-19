import wave
import pyaudio
import time
import re
from classes import TranscriptItem
import traceback

# Wavefile recording parameters
RATE = 16000
CHUNK = RATE // 10  # 100ms

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

def Whisper(SpeechToNLPQueue, EMSAgentQueue, wavefile_name):
    fifo_path = "/tmp/myfifo"
    finalized_blocks = ''
        
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
                    response = fifo.read().strip()  # Read the message from the named pipe

                    if response != old_response and response != "":
                        block, isFinal, avg_p, latency = process_whisper_response(response) #isFinal = False means block is interim block
                        transcript = finalized_blocks + block
                        # if received block is finalized, then save to finalized blocks
                        if isFinal: finalized_blocks += block
                        transcriptItem = TranscriptItem(transcript, isFinal, avg_p, latency)
                        EMSAgentQueue.put(transcriptItem)
                        SpeechToNLPQueue.put(transcriptItem)  
                        print("--- Whisper Latency:", latency)
                        old_response = response
                # Close stream (4)
                stream.close()

                EMSAgentQueue.put('Kill')

                # Release PortAudio system resources (5)
                p.terminate()

            except Exception as e:
                print("EXCEPTION: ", e)
                traceback.print_exception(e)            



            
    






