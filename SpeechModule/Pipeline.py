# Pipeline of CognitiveEMS
import sys
import os
from six.moves import queue

import StoppableThread
import SpeechModule

# Global variables
AccumulatedText = []

# Interconnecting queues
SpeechOutQueue = queue.Queue()

if __name__ == '__main__':

    def SpeechDisplay():
        global AccumulatedText
        while(True):
            result = SpeechOutQueue.get()
            #print(result)
            
            num_chars_printed = 0
            top_alternative = result.alternatives[0]
            transcript = top_alternative.transcript
            overwrite_chars = ' ' * (num_chars_printed - len(transcript))

            if not result.is_final:
                sys.stdout.write(transcript + overwrite_chars + '\r')
                sys.stdout.flush()
                num_chars_printed = len(transcript)
            else:
                print(transcript + overwrite_chars)
                AccumulatedText.append((transcript, top_alternative.confidence))
                num_chars_printed = 0
            

    # Speech Module
    SpeechThread = StoppableThread.StoppableThread(target = SpeechModule.GoogleSpeechIndefiniteStream, args=(SpeechOutQueue,))
    SpeechThread.start()

    DisplayThread = StoppableThread.StoppableThread(target = SpeechDisplay)
    DisplayThread.start()

