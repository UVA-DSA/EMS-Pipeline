import pipeline_config
import wave
import pyaudio
import time
import re
from classes import TranscriptItem, SpeechNLPItem
import traceback
# import sounddevice as sd
# import soundfile as sf

import queue

import sys
import threading
from classes import GUISignal


# Wavefile recording parameters
RATE = 16000
CHUNK = 1024  # 100ms

                
if(pipeline_config.endtoendspv):
    RATE = 44100
    CHUNK = 1024
    
                    
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

def Whisper(Window, TranscriptQueue,EMSAgentSpeechQueue, wavefile_name):
    fifo_path = "/tmp/myfifo"
    finalized_blocks = ''
    
    if Window:
        TranscriptSignal = GUISignal()
        TranscriptSignal.signal.connect(Window.UpdateSpeechBox)
        
    if("scenario" in wavefile_name):
        RATE = 44100
        CHUNK = 1024
        
    else:
        RATE = 16000
        CHUNK = 1024

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
                    
                    if(Window.stopped == 1): 
                        break
                    stream.write(data)
                    try:
                        response = fifo.read().strip()  # Read the message from the named pipe
                    except Exception as e:
                        response = ""

                    if response != old_response and response != "":
                        block, isFinal, avg_p, latency = process_whisper_response(response) #isFinal = False means block is interim block
                        transcript = finalized_blocks + block
                        # if received block is finalized, then save to finalized blocks
                        if isFinal: finalized_blocks += block
                        # send transcript item if its not an empty string or space
                        if len(transcript) and not transcript.isspace():
                            # transcriptItem = TranscriptItem(transcript, isFinal, avg_p, latency)
                            transcriptItem = SpeechNLPItem(transcript, isFinal,
                                              avg_p, len(transcript), 'Speech')
                            # EMSAgentQueue.put(transcriptItem)
                            
                            if isFinal:
                                print('WHISPER_OUT',transcriptItem.transcript)
                                TranscriptSignal.signal.emit([transcriptItem])
                                TranscriptQueue.put(transcriptItem) 
                                EMSAgentSpeechQueue.put(transcriptItem)
                                
                            # intertimTranscriptItem = SpeechNLPItem(block, isFinal,
                            #                   avg_p, len(transcript), 'Speech')

                                
                        print("--- Whisper Latency:", latency)
                        old_response = response
                # Close stream (4)
                stream.close()

                TranscriptQueue.put('Kill')
                EMSAgentSpeechQueue.put('Kill')

                # Release PortAudio system resources (5)
                p.terminate()

                print("Audio Stream Thread Sent Kill Signal. Bye!")

            except Exception as e:
                print("EXCEPTION: ", e)
                traceback.print_exception(e)            



            
    






