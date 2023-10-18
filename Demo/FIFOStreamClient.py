import time
from classes import TranscriptItem
import os
import re

'''
This method processes the whisper response we receive from fifo pipe 
The response is type string, and is in the format 'block{isFinal,avg_p}'
We remove background noise strings from the block (such as *crash* and [DOOR OPENS]) 
And separate out the parts of response into block, isFinal, and avg_p

block : speech converted into text
isFinal : '1' = the block is a final block. '0' = the block is a interim block
avg_p : 'float' of inclusive range [0,1]. the average confidence probability of all the tokens in the block
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
    stats = response[start+1:end]

    stats = stats.split(',')
    isFinal = True if stats[0] == '1' else False
    p_string=stats[1]
    latency=stats[2]
    avg_p = float(p_string)
    # print(p_string)

    return block, isFinal, avg_p, latency


def Fifo(SpeechToNLPQueue, EMSAgentQueue, SignalQueue):
    
    fifo_path = "/tmp/myfifo"
    finalized_blocks = ''
    
    print("here1")
    with open(fifo_path, 'r', os.O_NONBLOCK) as fifo:
        print("here2")
        try:
            print('Speech pipe opened!')
            SignalQueue.put("ready")

            old_response = ""
            while True:  # Requires Python 3.8+ for :=
                signal = SignalQueue.get()
                if(signal == "Done"):
                    break
                response = fifo.read().strip()  # Read the message from the named pipe
                if(response != old_response):
                    if(response != ""):
                        if("does not include pressure;" in response):
                            continue
                        block, isFinal, avg_p, latency = process_whisper_response(response) #isFinal = False means block is interim block
                        transcript = finalized_blocks + block
                        # if received block is finalized, then save to finalized blocks
                        if isFinal: finalized_blocks += block
                        transcriptItem = TranscriptItem(transcript, isFinal, avg_p, latency)
                        EMSAgentQueue.put(transcriptItem)
                        SpeechToNLPQueue.put(transcriptItem)  
                        print("--- Speech Latency:", latency,'ms')
                        old_response = response                
            # Close stream (4)
            EMSAgentQueue.put('Kill')

        except Exception as e:
            print("EXCEPTION: ", e)
            # traceback.print_exception(e)

