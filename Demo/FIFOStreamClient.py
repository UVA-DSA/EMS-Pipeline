import time
from classes import TranscriptItem
import os
import re
from time import sleep
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
    isFinal_str, avg_p_str, latency_str = response[start+1:end].split(",")
    isFinal = True if isFinal_str == '1' else False
    avg_p = float(avg_p_str)
    latency = int(latency_str)
    # print(p_string)

    return block, isFinal, avg_p, latency

def SignalFifo(SignalQueue):
    fifo_path = "/tmp/signalfifo"
    
    with open(fifo_path, 'r', os.O_NONBLOCK) as fifo:
        while True:
            sleep(1)
            
            data = fifo.readline()
            
            if not data:
                SignalQueue.put("Proceed")
                continue
            
            signal = data.strip()
            if signal == "Kill":
                print("Received Kill Signal from Signal FIFO", signal)
                SignalQueue.put("Kill")
                break
                
                
                

def Fifo(SpeechToNLPQueue, EMSAgentQueue, SignalQueue):
    
    fifo_path = "/tmp/myfifo"
    finalized_blocks = ''
    
    with open(fifo_path, 'r', os.O_NONBLOCK) as fifo:
        try:
            print('Speech pipe opened!')
            old_response = ""
            while True:  # Requires Python 3.8+ for :=
                if(not SignalQueue.empty()):
                    signal = SignalQueue.get()
                    if(signal == "Kill"):
                        break
                response = fifo.read().strip()  # Read the message from the named pipe
                if(response != old_response):
                    if(response != ""):

                        if(response.count("{") > 2): continue
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

