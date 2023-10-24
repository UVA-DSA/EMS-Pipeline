import pipeline_config
from transformers import pipeline
import traceback
from EMSVision.utils import *

model_name = pipeline_config.vision_model_type
print("Vision Here 1")
classifier = pipeline("zero-shot-image-classification", model = model_name, device=0)
print("Vision Here 2")

def EMSVision(FeedbackQueue, VisionDataQueue):
    
    # More models in the model hub.
    model_name = pipeline_config.vision_model_type
    print("Vision Here 1")
    classifier = pipeline("zero-shot-image-classification", model = model_name, device=0)
    print("Vision Here 2")
    index = 0

    protocol_success = False
    protocol_previous = ""
    while True:
        try:
            
            if(not FeedbackQueue.empty()):
                protocol_msg = FeedbackQueue.get(block=False)
                protocol = protocol_msg.protocol
            else:
                if(not protocol_success): protocol = "None"

            print("Raw protocol",protocol_msg)
            message = VisionDataQueue.get()
            


            print(FeedbackQueue.qsize(), VisionDataQueue.qsize())
            if(message["signal"] == "Kill"): 
                print("[EMS Vision Thread received Kill Signal. Bye!]")
                break
            
            print("[EMS Vision Thread: received-protocol:",protocol,']\n')
            
            print('\n\n=============================================================')
            

            labels_for_classification = generate_labels(protocol)
            if(labels_for_classification is None and not protocol_success): 
                model_scores,model_latency = -1, -1
            else:
                print("[EMS Vision Thread: generated-labels:",labels_for_classification,']')
                
                if not protocol_success:  protocol_previous = protocol

                print("[EMSVISIONTHREAD:]",protocol,protocol_previous,protocol_success)
                protocol_success = True

                labels_for_classification = generate_labels(protocol_previous)

                pil_image = message["image"]
                model_scores,model_latency = classify(pil_image,labels_for_classification,classifier)
                pil_image.save(f'{pipeline_config.directory}T{pipeline_config.trial_num}_{pipeline_config.curr_recording}_{index}_pil.jpg')
            index += 1

            print(f'[EMS Vision Thread: model scores-{model_scores}, latency-{model_latency}]')
            pipeline_config.curr_segment += [model_scores, model_latency]
            pipeline_config.rows_trial.append(pipeline_config.curr_segment)
            pipeline_config.curr_segment = []
            
                
            print("[EMS Vision Thread: action-recognition-results:",model_scores,model_latency,']')

        except Exception as e:
            print('[EMS Vision Thread: EXCEPTION!]',e)
            print(traceback.format_exc())
            break