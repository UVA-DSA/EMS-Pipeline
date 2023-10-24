import pipeline_config
from transformers import pipeline
import traceback
from EMSVision.utils import *

def EMSVision(FeedbackQueue, VideoDataQueue):
    
    # More models in the model hub.
    model_name = pipeline_config.vision_model_type
    classifier = pipeline("zero-shot-image-classification", model = model_name, device=0)
    index = 0

    sucessful_protocol_found = False
    sucessful_protocol = ""
    while True:
        try:           
            if(not FeedbackQueue.empty()):
                protocol_msg = FeedbackQueue.get(block=False)
                protocol = protocol_msg.protocol
            else:
                protocol = sucessful_protocol if(sucessful_protocol_found) else "None"

            image_message = VideoDataQueue.get()
            
            print(FeedbackQueue.qsize(), VideoDataQueue.qsize())
            if(image_message["signal"] == "Kill"): 
                print("[EMS Vision Thread received Kill Signal. Bye!]")
                break
            
            print("[EMS Vision Thread: received-protocol:",protocol,']\n')
            
            print('\n\n=============================================================')
            

            labels_for_classification = generate_labels(protocol)
            # if labels_for_classification is None, that means the protocol does not initiate action recogn
            if(labels_for_classification is None): 
                sucessful_protocol_found = False
                model_scores,model_latency = "None", None
            # if labels_for_classification is not None, that means a protocol that initiates action recogn was found
            else:
                print("[EMS Vision Thread: generated-labels:",labels_for_classification,']')
                sucessful_protocol_found = True
                sucessful_protocol = protocol 
                print("[EMSVISIONTHREAD:]",protocol,sucessful_protocol,sucessful_protocol_found)

                pil_image = image_message["image"]
                model_scores,model_latency = classify(pil_image,labels_for_classification,classifier)
                pil_image.save(f'{pipeline_config.directory}T{pipeline_config.trial_num}_{pipeline_config.curr_recording}_{index}_pil.jpg')
            index += 1

            print(f'[EMS Vision Thread: model scores-{model_scores}, latency-{model_latency}]')
            pipeline_config.vision_data['protocol'].append(protocol)
            pipeline_config.vision_data['intervention recognition'].append(model_scores)
            pipeline_config.vision_data['intervention latency'].append(model_latency)

        except Exception as e:
            print('[EMS Vision Thread: EXCEPTION!]',e)
            print(traceback.format_exc())
            break