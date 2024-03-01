import wave
import pyaudio
import time
import re
from classes import TranscriptItem
import traceback
import os
import pipeline_config




# Wavefile recording parameters
RATE = 16000
# CHUNK = RATE // 50  # 320 samples
CHUNK = 512  # 320 samples

if(pipeline_config.action_recognition):
    RATE = 44100
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



def SDStream(SpeechSignalQueue,wavefile_name):


    
    blocksize = 2048
    buffersize = 20
    device = len(sd.query_devices()) -1

    q = queue.Queue(maxsize=20)
    event = threading.Event()
    print("Audio Streaming Started! ",wavefile_name)
    

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
    

def ReadPipe(SpeechToNLPQueue, SpeechSignalQueue):
    
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
                    
                if(not SpeechSignalQueue.empty()): signal = SpeechSignalQueue.get(block=False)
                else: signal = "Proceed"
                    
                    
                if response != old_response and response != "":
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
            

            print("FIFO Read Thread Sent Kill Signal. Bye!")

        except Exception as e:
            print("EXCEPTION: ", e)





# backup



def Whisper(SpeechToNLPQueue, EMSAgentQueue, wavefile_name, is_streaming, SignalQueue):
    fifo_path = "/tmp/myfifo"
    finalized_blocks = ''
    
    if "scenario" in wavefile_name:
        RATE = 44100
    else:
        RATE = 16000
        
    if(is_streaming):        
        with open(fifo_path, 'r') as fifo:
            with wave.open(wavefile_name, 'rb') as wf:
                try:
                    # Instantiate PyAudio and initialize PortAudio system resources (1)
                    p = pyaudio.PyAudio()
                    info = p.get_default_host_api_info()
                    device_index = info.get('deviceCount') - 1 # get default device as output device
                        
                    old_response = ""
                    stream = p.open(format = pyaudio.paInt16, channels = 1, rate = RATE, output = True, frames_per_buffer = CHUNK, output_device_index=device_index)
                    data = wf.readframes(CHUNK)
                    
                    while len(data):  # Requires Python 3.8+ for :=
                        stream.write(data)
                        response = fifo.read().strip()  # Read the message from the named pipe
                        data = wf.readframes(CHUNK)
                        if(response != old_response):
                            if(response != ""):
                                block, isFinal, avg_p, latency = process_whisper_response(response) #isFinal = False means block is interim block
                                transcript = finalized_blocks + block
                                # if received block is finalized, then save to finalized blocks
                                if isFinal: finalized_blocks += block
                                # send transcript item if its not an empty string or space
                                if len(transcript) and not transcript.isspace():
                                    transcriptItem = TranscriptItem(transcript, isFinal, avg_p, latency)
                                    EMSAgentQueue.put(transcriptItem)
                                    SpeechToNLPQueue.put(transcriptItem)  
                                print("--- Speech Latency:", latency,'ms')
                                                            
                                if pipeline_config.speech_standalone:
                                    pipeline_config.trial_data['speech latency (ms)'].append(latency)
                                    pipeline_config.trial_data['transcript'].append(transcript)
                                    pipeline_config.trial_data['whisper confidence'].append(avg_p)
                    
                                old_response = response                
                    # Close stream (4)
                    stream.close()
                    
                    SignalQueue.put('Done')
                    EMSAgentQueue.put('Kill')
                    SpeechToNLPQueue.put('Kill')

                    # Release PortAudio system resources (5)
                    p.terminate()

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
            



            
    
# q = queue.Queue()
# SDStream(q,"./Audio_Scenarios/2019_Test/000_190105.wav")




