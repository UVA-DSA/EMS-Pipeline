import sounddevice as sd
import soundfile as sf

import queue
import threading
import argparse

import os

def SDStream(SpeechSignalQueue,wavefile_name):


    fifo_path = "/tmp/signalfifo"
    finalized_blocks = ''
    # Check if the named pipe exists, and create it if it doesn't
    if not os.path.exists(fifo_path):
        os.mkfifo(fifo_path)
        
    with open(fifo_path, 'w', os.O_NONBLOCK) as fifo:
            
        # Extract data and sampling rate from file
        SpeechSignalQueue.put("Proceed")
        
        data, fs = sf.read(wavefile_name, dtype='float32')  
        sd.play(data, fs)
        status = sd.wait()  # Wait until file is done playings
        SpeechSignalQueue.put("Kill")
        
        fifo.write("Kill")
        
    # blocksize = 1024
    # buffersize = 20
    # device = len(sd.query_devices()) -1

    # q = queue.Queue(maxsize=100)
    # event = threading.Event()
    # print("Audio Streaming Started! ",wavefile_name)
    
        
    # def callback(outdata, frames, time, status):
    #     assert frames == blocksize
    #     if status.output_underflow:
    #         print('Output underflow: increase blocksize?', file=sys.stderr)
    #         raise sd.CallbackAbort
    #     assert not status
    #     try:
    #         data = q.get_nowait()
    #     except queue.Empty:
    #         print('Buffer is empty: increase buffersize?', file=sys.stderr)
    #         raise sd.CallbackAbort
    #     if len(data) < len(outdata):
    #         outdata[:len(data)] = data
    #         outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
    #         raise sd.CallbackStop
    #     else:
    #         outdata[:] = data

    #     SpeechSignalQueue.put("Proceed")

    # try:
    #     with sf.SoundFile(wavefile_name) as f:
    #         for _ in range(buffersize):
    #             data = f.buffer_read(blocksize, dtype='float32')
    #             if not data:
    #                 break
    #             q.put_nowait(data)  # Pre-fill queue
                
    #         print(f.samplerate)
    #         stream = sd.RawOutputStream(
    #             samplerate=f.samplerate, blocksize=blocksize,
    #             device=device, channels=f.channels, dtype='float32',
    #             callback=callback, finished_callback=event.set)
    #         with stream:
    #             timeout = blocksize * buffersize / f.samplerate
    #             while data:
    #                 data = f.buffer_read(blocksize, dtype='float32')
    #                 q.put(data, timeout=timeout)
    #             event.wait()  # Wait until playback is finished
                
    #             print("Audio streaming over!")
    #             SpeechSignalQueue.put("Kill")
    # except KeyboardInterrupt:
    #     print('\nInterrupted by user')
    # except queue.Full:
    #     # A timeout occurred, i.e. there was an error in the callback
    #     print('\nQueue Full')
    # except Exception as e:
    #     print('\nException ',e)
        
        
if __name__ == '__main__':
    
    # parser = argparse.ArgumentParser(
    #                 prog='ProgramName',
    #                 description='What the program does',
    #                 epilog='Text at the bottom of help')
    
    # parser.add_argument('-f', '--file')      # option that takes a value
    # args = parser.parse_args()
    
    # input_file = args.file
    

    input_file = "./Audio_Scenarios/2019_Test/000_190105.wav"
    SpeechSignalQueue = queue.Queue()
    
    SDStream(SpeechSignalQueue,input_file)
