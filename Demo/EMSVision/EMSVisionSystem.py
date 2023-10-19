import pipeline_config
from transformers import pipeline

from EMSVision.utils import *

def EMSVision(FeedbackQueue, VisionDataQueue):
    
    # More models in the model hub.
    model_name = pipeline_config.vision_model_type

    classifier = pipeline("zero-shot-image-classification", model = model_name, device=0)

    while True:
        try:
            
            protocol_msg = FeedbackQueue.get()
            message = VisionDataQueue.get()
                    
            if(message["signal"] == "Kill" or protocol_msg == "Kill"): 
                print("[EMS Vision Thread received Kill Signal. Bye!]")
                break
            
            protocol = protocol_msg.protocol
            print("[EMS Vision Thread: received-protocol:",protocol,']\n')
            
            print('\n\n=============================================================')
            
            labels_for_classification = generate_labels(protocol)

            if(labels_for_classification is None): continue
                
            print("[EMS Vision Thread: generated-labels:",labels_for_classification,']')

            labels_for_classification = generate_labels(protocol)
            
            pil_image = message["image"]
            model_scores,model_latency = classify(pil_image,labels_for_classification,classifier)

            print("[EMS Vision Thread: action-recognition-results:",model_scores,model_latency,']')

        except Exception as e:
            print('[EMS Vision Thread: EXCEPTION!]',e)