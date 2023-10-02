from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

class TranscriptItem:
      def __init__(self, transcript, isFinal, confidence, transcriptionDuration):
            self.transcript = transcript
            self.isFinal = isFinal
            self.confidence = confidence
            self.transcriptionDuration = transcriptionDuration

# ------------ For Feedback ------------
class FeedbackObj:
    def __init__(self, intervention, protocol, concept):
        super(FeedbackObj, self).__init__()
        self.intervention = intervention
        self.protocol = protocol
        self.concept = concept

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


