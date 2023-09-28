from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

# ============== Custom Speech to NLP Queue Item Class ==============

class SpeechNLPItem:
        def __init__(self, transcript, isFinal, confidence, numPrinted, origin):
            self.transcript = transcript
            self.isFinal = isFinal
            self.confidence = confidence
            self.origin = origin
            self.numPrinted = numPrinted

#======================= To be replaced with better designed classes:

class QueueItem:
    def __init__(self, transcript, isFinal, confidence):
            self.transcript = transcript
            self.isFinal = isFinal
            self.confidence = 0

# Custom object for signalling
class GUISignal(QObject):
    signal = pyqtSignal(list)


