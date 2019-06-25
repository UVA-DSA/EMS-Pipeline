from PyQt4.QtCore import *
from PyQt4 import QtSvg
from PyQt4.QtGui import *
import numpy as np

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


