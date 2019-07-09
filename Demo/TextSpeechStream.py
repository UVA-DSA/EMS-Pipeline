from __future__ import absolute_import, division, print_function
from timeit import default_timer as timer
import sys
import os
import time
from six.moves import queue
import numpy as np
from classes import SpeechNLPItem, GUISignal
from Form_Filling import textParse2
import commands

BETWEEN_CHARACTERS_PAUSE = .06
BETWEEN_WORDS_PAUSE = .1
BETWEEN_SENETENCES_PAUSE = 1
COMMA_PAUSE = .3

def TextSpeech(Window, SpeechToNLPQueue, textfile_name):
    print("Entered TextSpeech")

    # Create GUI Signal Object
    SpeechSignal = GUISignal()
    SpeechSignal.signal.connect(Window.UpdateSpeechBox)

    MsgSignal = GUISignal()
    MsgSignal.signal.connect(Window.UpdateMsgBox)

    ButtonsSignal = GUISignal()
    ButtonsSignal.signal.connect(Window.ButtonsSetEnabled)

    Text_File = open(textfile_name, "rb") 

    text = ""
    for line in Text_File.readlines():
        text += line
    counter = 0
    
    # Break into sentences
    dummy12= text
    dummy12 = dummy12.replace('\r', '').replace('\n', '')
    dummyP2=dummy12.replace(' ','%20')
    dummyP3=dummyP2.replace('\'','%27')
    dummyP=dummyP3.replace('&','%26')
    part1='curl -d text='+dummyP+' http://bark.phon.ioc.ee/punctuator'
    op = commands.getstatusoutput(part1)
    output = op[1].rsplit('\n', 1)[1]
    sentsList = textParse2.sent_tokenize(output) #final sentences

    # Stream text
    num_chars_printed = 0
    for sentence in sentsList:
        for i, character in enumerate(sentence):
            QueueItem = SpeechNLPItem(sentence[ : i + 1], False, 0, num_chars_printed, 'Speech')
            SpeechSignal.signal.emit([QueueItem])
            if(character == " "):
                time.sleep(BETWEEN_WORDS_PAUSE)
            elif(character == ","):
                time.sleep(COMMA_PAUSE)
            elif(character == "."):
                time.sleep(BETWEEN_SENETENCES_PAUSE)
            else:
                time.sleep(BETWEEN_CHARACTERS_PAUSE)
            num_chars_printed = len(sentence[ : i + 1])

            if(Window.stopped == 1):
                print('Text Speech Tread Killed')
                QueueItem = SpeechNLPItem(sentence[ : i + 1], True, 0, num_chars_printed, 'Speech')
                SpeechSignal.signal.emit([QueueItem])
                SpeechToNLPQueue.put(QueueItem)
                return

        QueueItem = SpeechNLPItem(sentence, True, 0, num_chars_printed, 'Speech')
        SpeechSignal.signal.emit([QueueItem])
        SpeechToNLPQueue.put(QueueItem)
        num_chars_printed = 0

    # Clean up and end thread
    MsgSignal.signal.emit(["Transcription of text file complete!"])
    ButtonsSignal.signal.emit([(Window.StartButton, True), (Window.ComboBox, True), (Window.ResetButton, True)])

