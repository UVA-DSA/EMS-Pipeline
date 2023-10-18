import wave
import pyaudio
import time
import re
from classes import TranscriptItem
import traceback
import os
# Wavefile recording parameters
RATE = 16000
# CHUNK = RATE // 50  # 320 samples
CHUNK = 2048  # 320 samples

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

# def Whisper(SpeechToNLPQueue, EMSAgentQueue, wavefile_name):
#     fifo_path = "/tmp/myfifo"
#     finalized_blocks = ''
        
#     with open(fifo_path, 'r') as fifo:
#         try:
#             print('Speech pipe opened!')

#             old_response = ""
#             # Play samples from the wave file (3)
#             while True:  # Requires Python 3.8+ for :=
#                 response = fifo.read().strip()  # Read the message from the named pipe
#                 if(response != old_response):
#                     if(response != ""):
#                         if("does not include pressure;" in response):
#                             continue
#                         block, isFinal, avg_p, latency = process_whisper_response(response) #isFinal = False means block is interim block
#                         transcript = finalized_blocks + block
#                         # if received block is finalized, then save to finalized blocks
#                         if isFinal: finalized_blocks += block
#                         transcriptItem = TranscriptItem(transcript, isFinal, avg_p, latency)
#                         EMSAgentQueue.put(transcriptItem)
#                         SpeechToNLPQueue.put(transcriptItem)  
#                         print("--- Speech Latency:", latency,'ms')
#                         old_response = response                


#             EMSAgentQueue.put('Kill')


#         except Exception as e:
#             print("EXCEPTION: ", e)
#             # traceback.print_exception(e)




# backup


def Whisper(SpeechToNLPQueue, EMSAgentQueue, wavefile_name, is_streaming, SignalQueue):
    fifo_path = "/tmp/myfifo"
    finalized_blocks = ''

    
    print("here1")
    if(is_streaming):        
            with wave.open(wavefile_name, 'rb') as wf:
                print("here3")

                try:
                    # Instantiate PyAudio and initialize PortAudio system resources (1)
                    p = pyaudio.PyAudio()
                    info = p.get_default_host_api_info()
                    device_index = info.get('deviceCount') - 1 # get default device as output device
                        
                    stream = p.open(format = pyaudio.paInt16, channels = 1, rate = RATE, output = True, frames_per_buffer = CHUNK, output_device_index=device_index)

                    # Play samples from the wave file (3)
                    data = wf.readframes(CHUNK)
                    while len(data):  # Requires Python 3.8+ for :=
                        stream.write(data)
                        data = wf.readframes(CHUNK)
                        SignalQueue.put('Progress')
                        

                    stream.close()
                    
                    SignalQueue.put('Done')
                    EMSAgentQueue.put('Kill')

                    # Release PortAudio system resources (5)
                    p.terminate()

                except Exception as e:
                    print("EXCEPTION: ", e)
                    # traceback.print_exception(e)


    else:

        with open(fifo_path, 'r') as fifo:
            try:
                print('Speech pipe opened!')

                old_response = ""
                # Play samples from the wave file (3)
                while True:  # Requires Python 3.8+ for :=
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


                EMSAgentQueue.put('Kill')


            except Exception as e:
                print("EXCEPTION: ", e)
                # traceback.print_exception(e)

















    # # Create GUI Signal Object
    # SpeechSignal = GUISignal()
    # SpeechSignal.signal.connect(Window.UpdateSpeechBox)

    # MsgSignal = GUISignal()
    # MsgSignal.signal.connect(Window.UpdateMsgBox)

    # ButtonsSignal = GUISignal()
    # ButtonsSignal.signal.connect(Window.ButtonsSetEnabled)
    # num_chars_printed = 0
    

    # with FileStream(RATE, CHUNK, wavefile_name) as fs:
    #     start = time.perf_counter()
    #     # print("=============WhisperFileStream.py: Audio file stream started", start)
        
    #     fifo_path = "/tmp/myfifo"  # Replace with your named pipe path

    #     while not fs.closed:
    #         try:
    #             with open(fifo_path, 'r') as fifo:
    #                 response = fifo.read().strip()  # Read the message from the named 
    #                 end = time.perf_counter()
    #                 block, isFinal, avg_p = process_whisper_response(response) #isFinal = False means block is interim block
    #                 transcript = finalized_blocks + block
    #                 # if received block is finalized, then save to finalized blocks
    #                 if isFinal: finalized_blocks += block
    #                 transcriptItem = TranscriptItem(transcript, isFinal, avg_p, end-start)
    #                 EMSAgentQueue.put(transcriptItem)
    #                 SpeechToNLPQueue.put(transcriptItem)
    #                 start = end 

    #         except Exception as e:
    #             print("Exception in Audiostream", e)
    #             # print(traceback.format_exc())

    # EMSAgentQueue.put('Kill')
    # return                               
            



            
    






