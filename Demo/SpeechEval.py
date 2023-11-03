# python script to run end to end evaluations

from Pipeline import Pipeline
import pipeline_config
import numpy as np
import pandas as pd
from EMSAgent.default_sets import ungroup_p_node
import os
from time import sleep
import datetime
from EMSAgent.utils import get_precision_at_k, get_recall_at_k, get_r_precision_at_k, get_ndcg_at_k
from SpeechModule import SpeechModule
# -- static helper variables ---------------------------------------


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

    speech_models = pipeline_config.whisper_model_sizes

    # get timestamp of experiment run
    pipeline_config.time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    for trial in range(pipeline_config.num_trials_per_recording):
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
                    
                # run pipeline
                # Pipeline(recording=recording, video_file=video, whisper_model=whisper_model)
                print("Running E2E Evaluation for Recording: ",recording)
                print("\n\n\n ********************** CONFIGURATION **************************** \n\n")

                print("speech-model:",speech_model)
                print("protocol-model:",pipeline_config.protocol_model_type)
                print("protocol-device:",pipeline_config.protocol_model_device)
                print("Trial:",trial)
                print("Recording:",recording)

                print("\n\n\n ******************************************************** \n\n")

                SpeechModule(recording=recording, whisper_model=speech_model)

                # print("SIZE OF DICT")
                # for key, value in pipeline_config.trial_data.items():
                #     print(f"{key}: {len(value)}")

                # remove keys from pipeline_config.trial_data that don't correspond to speech stuff
                relevant_keys = set(['speech latency (ms)', 'transcript', 'whisper confidence'])

                del pipeline_config.trial_data['WER']
                del pipeline_config.trial_data['CER']
                del pipeline_config.trial_data['protocol latency (ms)']
                del pipeline_config.trial_data['protocol prediction']
                del pipeline_config.trial_data['protocol confidence']
                del pipeline_config.trial_data['protocol correct?']
                del pipeline_config.trial_data['one hot prediction']
                del pipeline_config.trial_data['one hot gt']
                del pipeline_config.trial_data['tn']
                del pipeline_config.trial_data['fp']
                del pipeline_config.trial_data['fn']
                del pipeline_config.trial_data['tp']
                del pipeline_config.trial_data['logits']


                # dictionary_keys = pipeline_config.trial_data.keys()
                # for key in dictionary_keys:
                #     if key not in relevant_keys:
                #         del pipeline_config.trial_data[key]
                



                # write out data
                df = pd.DataFrame(pipeline_config.trial_data)
                df.to_csv(f'{pipeline_config.directory}T{trial}_SPEECH_{recording}.csv')
                
                pipeline_config.trial_data['speech latency (ms)'] = []
                pipeline_config.trial_data['transcript'] = []
                pipeline_config.trial_data['whisper confidence'] = []
