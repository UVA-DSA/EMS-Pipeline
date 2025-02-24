from PyQt5.QtCore import pyqtSignal, QObject
PROTOCOL = "protocol"

# ------------ For MediaPipe Interprocess Communication ------------

class MPQueueImage:
    """
    Custom image object for MediaPipeProcess processor.

    Attributes:
        timestamp (datetime): unique ID for each image
        image (numpy.array): numpy containing image RGB data
    """
    def __init__(self, timestamp, image):
        self.timestamp = timestamp
        self.image = image


# ------------ Object Detection Results  ------------
class DetectionObj:
    """
    Object to return Detection results from the Action Recognition model.
    {'type':'detection', 'box_coords':[(0, 102), (242, 511)], 'obj_name': 'keyboard', 'confidence':'0.95'}

    Attributes:
        box_coords (dictionary): box coordinates 
        obj_name (str) : name of the object

    """
    def __init__(self, box_coords, name, confidence):
        self.type = 'detection'
        self.box_coords = box_coords
        self.obj_name = name
        self.confidence = confidence


# ------------ Object Detection Results  ------------
class ProtocolObj:
    """
    Object to return Protocol results from the Protocol Recognition model.

    Attributes:
    protocol (str): protocol name
    protocol_confidence (str): confidence in protocol 
        

    """
    def __init__(self, protocol, protocol_confidence):
        self.type = 'Protocol'
        self.protocol = protocol
        self.protocol_confidence = protocol_confidence

# ------------ For Transcription ------------


class TranscriptItem:
    def __init__(self, transcript, isFinal, confidence, transcriptionDuration):
        self.type = 'transcript'
        self.transcript = transcript
        self.isFinal = isFinal
        self.confidence = confidence
        self.transcriptionDuration = transcriptionDuration

# ------------ For Feedback ------------


class FeedbackObj:
    def __init__(self, intervention, protocol, protocol_confidence, concept): #add objdet result class
        super(FeedbackObj, self).__init__()
        self.type = 'feedback'
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
