

import sys
import time
sys.path.append("../EMS-Pipeline/Demo")
from Feedback import FeedbackClient
from classes import DetectionObj, ProtocolObj

detectObj = DetectionObj({"center_point" : (50, 50), "width": 100, "height": 75}, "hand", "0.85")
protocolObj1 = ProtocolObj("CPR", "0.85")
protocolObj2 = ProtocolObj("BVM", "0.85")
fc1 = FeedbackClient.instance()
fc2 = FeedbackClient.instance()
fc3 = FeedbackClient.instance()

fc1.start()
#fc.sendMessage(detectObj.__dict__)

time.sleep(4)

fc1.sendMessage(detectObj.__dict__)
fc2.sendMessage(protocolObj1.__dict__)
fc3.sendMessage(protocolObj2.__dict__)



