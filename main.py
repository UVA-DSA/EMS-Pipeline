import multiprocessing as mp
from multiprocessing import Process, Queue

def spawn_vision():
    return
def spawn_network():
    return





if __name__ == "__main__":


    #command queue initilaization
    command_queue  = Queue()
    #creating processes
    vision_process = Process(target=spawn_vision)
    network_process = Process(target=spawn_network)
    


    ##starting processes
    vision_process.start()



    #waiting on processes to finish
    vision_process.join()
