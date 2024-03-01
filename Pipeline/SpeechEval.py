# python script to run end to end evaluations

from Pipeline import Pipeline
import pipeline_config
from transformers import WhisperProcessor
from evaluate import load
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from EMSAgent.default_sets import ungroup_p_node
import os
from time import sleep
import datetime
from EMSAgent.utils import get_precision_at_k, get_recall_at_k, get_r_precision_at_k, get_ndcg_at_k
# -- static helper variables ---------------------------------------
# initialize processor depending on whisper model size 
if pipeline_config.whisper_model_size == 'base-finetuned':
    Processor = WhisperProcessor.from_pretrained("saahith/whisper-base.en-combined-v10")
elif pipeline_config.whisper_model_size == 'base.en':
    Processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
elif pipeline_config.whisper_model_size in {'tiny-finetuned', 'tiny-finetuned-v2'}:
    Processor = WhisperProcessor.from_pretrained("saahith/whisper-tiny.en-combined-v10")
elif pipeline_config.whisper_model_size == 'tiny.en':
    Processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
wer_metric = load("wer")
cer_metric = load("cer")


# --- helper methods -----------------
def get_ground_truth_transcript(recording):
    with open(f'Audio_Scenarios/2019_Test_Ground_Truth/{recording}-transcript.txt') as f:
        ground_truth = f.read()
    return ground_truth

def get_ground_truth_protocol(recording):
    with open(f'Audio_Scenarios/2019_Test_Ground_Truth/{recording}-protocol.txt') as f:
        ground_truth = f.read()
    return ground_truth

def get_ground_truth_one_hot_vector(recording):
    ground_truth = get_ground_truth_protocol(recording)
    one_hot_vector = [1 if label.lower() == ground_truth.lower() else 0 for label in ungroup_p_node]
    return np.array(one_hot_vector)

# def get_wer_and_cer(recording, transcript):
#     # tokenized_reference_text = Processor.tokenizer._normalize(get_ground_truth_transcript(recording))
#     # tokenized_prediction_text = Processor.tokenizer._normalize(transcript)
#     # wer = wer_metric.compute(references=[tokenized_reference_text], predictions=[tokenized_prediction_text])
#     # cer = cer_metric.compute(references=[tokenized_reference_text], predictions=[tokenized_prediction_text])
#     return 0, 0

def check_protocol_correct(recording, protocol):
    if protocol == -1: return -1
    ground_truth = get_ground_truth_protocol(recording)
    return int(protocol.lower() == ground_truth.lower())

def is_singleton_array(arr):
    shape = arr.shape
    # Check if all dimensions have size 1
    return all(dim == 1 for dim in shape)

# --- main ---------------------------
if __name__ == '__main__':

    if pipeline_config.speech_model == 'whisper': 
        speech_models = pipeline_config.whisper_model_sizes
    else:
        speech_models = ['conformer']

    # get timestamp of experiment run
    pipeline_config.time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    for trial in range(pipeline_config.num_trials):
        pipeline_config.trial_num = trial
        for speech_model in speech_models:
            # set directory and make directory
            pipeline_config.directory = f"Evaluation_Results/{pipeline_config.time_stamp}/{pipeline_config.protocol_model_type}/{pipeline_config.protocol_model_device}/{speech_model}/"
            if not os.path.exists(pipeline_config.directory):
                os.makedirs(pipeline_config.directory)
            # evaluate recordings
            for recording in pipeline_config.recordings_to_test:
                pipeline_config.image_directory = f"{pipeline_config.directory}/T{trial}_{recording}_images/"
                if not os.path.exists(pipeline_config.image_directory):
                    os.makedirs(pipeline_config.image_directory)
                pipeline_config.curr_recording = recording
                # set field names
                pipeline_config.trial_data['speech latency (ms)'] = []
                pipeline_config.trial_data['transcript'] = []
                pipeline_config.trial_data['whisper confidence'] = []
                pipeline_config.trial_data['WER'] = []
                pipeline_config.trial_data['CER'] = []
                pipeline_config.trial_data['protocol latency (ms)'] = []
                pipeline_config.trial_data['protocol prediction'] = []
                pipeline_config.trial_data['protocol confidence'] = []
                pipeline_config.trial_data['protocol correct?'] = []
                pipeline_config.trial_data['one hot prediction'] = []
                pipeline_config.trial_data['one hot gt'] = []
                pipeline_config.trial_data['tn'] = []
                pipeline_config.trial_data['fp'] = []
                pipeline_config.trial_data['fn'] = []
                pipeline_config.trial_data['tp'] = []
                pipeline_config.trial_data['logits'] = []
                if pipeline_config.endtoendspv:
                    pipeline_config.vision_data['protocol'] = []
                    pipeline_config.vision_data['intervention recognition'] = []
                    pipeline_config.vision_data['intervention latency'] = []
                    
                # run pipeline
                # Pipeline(recording=recording, video_file=video, whisper_model=whisper_model)
                print("Running E2E Evaluation for Recording: ",recording)
                print("\n\n\n ********************** CONFIGURATION **************************** \n\n")

                print("speech-model:",speech_model)
                print("protocol-model:",pipeline_config.protocol_model_type)
                print("protocol-device:",pipeline_config.protocol_model_device)
                print("isEnd-to-End:",pipeline_config.endtoendspv)
                print("Trial:",trial)
                print("Recording:",recording)

                print("\n\n\n ******************************************************** \n\n")

                Pipeline(recording=recording, whisper_model=speech_model)

                # print("SIZE OF DICT")
                # for key, value in pipeline_config.trial_data.items():
                #     print(f"{key}: {len(value)}")

                # remove keys from pipeline_config.trial_data that don't correspond to speech stuff
                relevant_keys = set(['speech latency (ms)', 'transcript', 'whisper confidence'])

                dictionary_keys = pipeline_config.trial_data.keys()
                for key in dictionary_keys:
                    if key not in relevant_keys:
                        del pipeline_config.trial_data[key]
                


                pipeline_config.trial_data['speech latency (ms)'] = []
                pipeline_config.trial_data['transcript'] = []
                pipeline_config.trial_data['whisper confidence'] = []

                # write out data
                df = pd.DataFrame(pipeline_config.trial_data)
                df.to_csv(f'{pipeline_config.directory}T{trial}_SPEECH+PROTOCOL_{recording}.csv')

                if pipeline_config.endtoendspv:
                    df = pd.DataFrame(pipeline_config.vision_data)
                    df.to_csv(f'{pipeline_config.directory}T{trial}_VISION_{recording}.csv')
            
                break #for test   
            if(pipeline_config.speech_model == 'conformer'): break