import socket
import threading
import sys
import time

import sounddevice

s=0 #s socket
# packet_size in bytes for incoming udp sockets
mic_udp_packet_size=2304 #same UDP packet size as android app !!!!!
mic_udp_data = bytearray()
mic_cyclic_buffer = bytearray() #cycle buffer
event_obj_MicSoundRender = threading.Event() # conditional object on thread ::  .wait to wait on it , .set to signal it
current_mic_channel = sounddevice.default.device[1] #DEFULT output channel
cur_index = 0#used by callback to traverse cyclic buffer
cyclic_buffer_size = 16
recorded_data = bytearray()
recording = False


def createsock():
    """ Listen FOR UDP packets on PORT 6666 """
    print("Creating socket")
    global s
    # Create a UDP socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Bind the socket to the port
    server_address = ('0.0.0.0', 8888)
    s.bind(server_address)

def callback(outdata, frames, time, status):
    """callback for output device that is ready to consume and render auido data.
        output buffer is gathered from circular buffer(mic_cyclic_buffer)
    """
    global cur_index
    if status:
        print(status)

    # print(len(outdata))

    next_index = cur_index + len(outdata)
    if (next_index > len(mic_cyclic_buffer)):
        # print("Buffer overflow returning to begining of cyclic buffer")
        overflow = next_index - len(mic_cyclic_buffer) #number of bytes to be written to second part of outdata
        # print("overflow is : " + str(overflow))
        # print("current index is : " + str(cur_index))
        # print("lenghts of mic_cyclic buffer is : " +str(len(mic_cyclic_buffer)))
        intermediate_index = len(outdata) - overflow #number of bytes written to first part of outdata
        # print("intermediate_index is : " + str(intermediate_index))
        outdata[0 : intermediate_index] = mic_cyclic_buffer[cur_index:len(mic_cyclic_buffer)] #first part (end of cyclic buffer)
        outdata[intermediate_index : len(outdata)]  = mic_cyclic_buffer[0:overflow] #second part ( start of cyclic buffer )
        cur_index = overflow #update cur_index for next callback
        # print("Current index is now:"+str(cur_index))
    else:
        outdata[:] = mic_cyclic_buffer[cur_index:next_index]
        cur_index = next_index
        # print("Current index is now:"+str(cur_index))

class MicSoundRender(threading.Thread): ##Renderer thread
    """Renderer Thread that renders auido data with callback()"""
    def __init__(self):
        super(MicSoundRender, self).__init__()
    def run(self):
        with sounddevice.RawOutputStream(dtype='int16',samplerate=44100,channels=2, callback=callback,latency='low',device=current_mic_channel):
            event_obj_MicSoundRender.wait()
        print("MicSoundRender done")

class MicSoundUDPrecv(threading.Thread):
    """Receiver thread that is responsible for capturing streaming UDP packets and writing them
    to circular buffer(mic_cyclic_buffer)
    """
    def __init__(self):
        super(MicSoundUDPrecv, self).__init__()
        self.mstart = False
    def run(self):
        times = 0
        global recorded_data
        try:
            while (True):
                data, address = s.recvfrom(mic_udp_packet_size)
                if(recording):
                    recorded_data = recorded_data + data
                mic_cyclic_buffer[times * mic_udp_packet_size  :  (times+1) * mic_udp_packet_size ] = data
                times = times + 1
                times = times % cyclic_buffer_size
                if (not self.mstart):
                    playa.start()
                    self.mstart = True
                    print("MicSoundUDPrecv Started once")
        except Exception as e:
            print("MicSoundUDPrecv Socket closed thread done", e)

        self.mstart = False
        # print("MicSoundUDPrecv thread done ")

def toggle_state():  # on main thread close open socket and restart threads
    """Open/Close UDP socket and restarts threads"""
    global runing_state
    global mic_recv
    global playa
    global s
    global cur_index
    if runing_state:
        event_obj_MicSoundRender.set()
        s.close()
        mic_recv.join()
        print("paused everything")
    else:
        print("####### RESTARTING AGAIN #######")
        cur_index = 0
        createsock()
        event_obj_MicSoundRender.clear()
        mic_recv = MicSoundUDPrecv()
        playa = MicSoundRender()
        mic_recv.start()

    runing_state=not runing_state

def audio_server():

    global runing_state
    global mic_recv
    global playa
    global s


    createsock()
    mic_recv = MicSoundUDPrecv()
    playa = MicSoundRender()
    mic_recv.start()

    runing_state=True
    print(threading.currentThread().getName() + " Main Thread : ")
    print("####### Server is listening #######")

    while(True):
        in_str = input("Commands : help,toggle,exit,restart,devices,mic #,stop,record,write\n")
        if(in_str.lower() =="toggle"):
            toggle_state()
        elif(in_str.lower() =="exit"):
            sys.exit(0)
        elif (in_str.lower() == "record"):
            recording = True
            print("Recording to memory use write command to stop recording and write to disk...")
        elif(in_str.lower() == "write"):
            recording = False
            f = open("data.pcm", "wb")
            f.write(recorded_data)
            f.close()
            recorded_data.clear()
            print("Done recording saved as data.pcm")
        elif(in_str.lower() =="help"):
            print("  toggle: Start/Stop listening\n")
            print("  exit: Exit the Program\n")
            print("  restart: Reset and start listening\n")
            print("  devices: Show the audio input/output devices id's. These ids can be used with mic "
                "command to write to an output devices. (Example usage:mic 5)\n")
            print("  mic #: Change output channel to a device with given id (Example usage:mic 5).Id's can be queried with devices command")
            print("  mic: If no id is present show the current output channel\n")
            print("  stop: Stop listening\n")
            print("  record: Start recording and save raw auido data to memory. To write from memory to persistent storage and stop recording"
                "use write command\n")
            print("  write: Stop recording to memory and write to persistent storage\n")
        elif(in_str.lower() =="restart"):
            if runing_state:
                toggle_state()
                time.sleep(0.5)
                toggle_state()
            else:
                toggle_state()
        elif(in_str.lower() =="devices"):
            print(sounddevice.query_devices())

        elif(in_str.startswith('mic')):
            splited = in_str.split(' ')
            try:
                mint = splited[1]
                if(not mint.isdecimal()):
                    raise Exception
                current_mic_channel = int(mint)
                print("changed mic channel to "+ str(mint)+" restarting")
                if runing_state:
                    toggle_state()
                    time.sleep(0.5)
                    toggle_state()
                else:
                    toggle_state()
            except  :
                print("current mic channel is : " + str(current_mic_channel))
        elif (in_str.lower() == "stop"):
            if(runing_state):
                toggle_state()



if __name__ == "__main__":
    audio_server()