from PyQt5.QtCore import pyqtSignal, QObject


# ------------ For MediaPipe Interprocess Communication ------------

class QueueImage:
    """
    Custom image object for MediaPipeProcess processor.

    Attributes:
        timestamp (datetime): unique ID for each image
        image (numpy.array): numpy containing image RGB data
    """
    def __init__(self, timestamp, image):
        self.timestamp = timestamp
        self.image = image


# ------------ For Transcription ------------


class TranscriptItem:
    def __init__(self, transcript, isFinal, confidence, transcriptionDuration):
        self.transcript = transcript
        self.isFinal = isFinal
        self.confidence = confidence
        self.transcriptionDuration = transcriptionDuration

# ------------ For Feedback ------------


class FeedbackObj:
    def __init__(self, intervention, protocol, protocol_confidence, concept):
        super(FeedbackObj, self).__init__()
        self.intervention = intervention
        self.protocol = protocol
        self.protocol_confidence = protocol_confidence
        self.concept = concept

# ------------ Custom Speech to NLP QueueItem Class ------------


class SpeechNLPItem:
    def __init__(self, transcript, isFinal, confidence, numPrinted, origin):
        self.transcript = transcript
        self.isFinal = isFinal
        self.confidence = confidence
        self.origin = origin
        self.numPrinted = numPrinted

# To be replaced with better designed classes:

# ------------ QueueItem class containing transcripts (add more detailed description) ------------
class QueueItem:
    def __init__(self, transcript, isFinal, confidence):
        self.transcript = transcript
        self.isFinal = isFinal
        self.confidence = 0


# ------------ Custom Class for PyQt signalling ------------

class GUISignal(QObject):
    signal = pyqtSignal(list)
