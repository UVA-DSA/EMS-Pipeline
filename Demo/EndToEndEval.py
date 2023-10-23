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

def get_wer_and_cer(recording, transcript):
    # tokenized_reference_text = Processor.tokenizer._normalize(get_ground_truth_transcript(recording))
    # tokenized_prediction_text = Processor.tokenizer._normalize(transcript)
    # wer = wer_metric.compute(references=[tokenized_reference_text], predictions=[tokenized_prediction_text])
    # cer = cer_metric.compute(references=[tokenized_reference_text], predictions=[tokenized_prediction_text])
    return 0, 0

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

    one_hot_pred_all_recordings = []
    one_hot_gt_all_recordings = []
    logits_all_recordings = []
    df_all_recordings = []

    if pipeline_config.speech_model == 'whisper': 
        speech_models = pipeline_config.whisper_model_sizes
    else:
        speech_models = ['conformer']

    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    for trial in range(pipeline_config.num_trials):
        for speech_model in speech_models:
            for recording in pipeline_config.recordings_to_test:


                # field names
                fields = ['time audio->transcript (s)', 'transcript', 'whisper confidence', 'WER', 'CER', #4
                            'time protocol input->output (s)', 'protocol prediction', 'protocol confidence', #7
                            'protocol correct? (1 = True, 0 = False, -1=None given)', #8
                            'one hot prediction', 'one hot GT', 'tn', 'fp', 'fn', 'tp', 'logits'] 
                if pipeline_config.endtoendspv:
                    fields += ['intervention recognition','intervention latency']

                # data rows of csv file for this run
                pipeline_config.curr_segment = []
                pipeline_config.rows_trial = []

                # run pipeline
                # Pipeline(recording=recording, video_file=video, whisper_model=whisper_model)
                print("Running E2E Evaluation for Recording: ",recording)
                print("\n\n\n ********************** CONFIGURATION **************************** \n\n")

                print("speech-model:",speech_model)
                print("protocol-model:",pipeline_config.protocol_model_type)
                print("protocol-device:",pipeline_config.protocol_model_device)
                print("isEnd-to-End:",pipeline_config.endtoendspv)
                print("isData-Save:",pipeline_config.data_save)
                print("Trial:",trial)
                print("Recording:",recording)


                print("\n\n\n ******************************************************** \n\n")

                Pipeline(recording=recording, whisper_model=speech_model)
                
                # get data
                rows_trial = pipeline_config.rows_trial
                # get ground truth one hot vector
                if(not pipeline_config.endtoendspv): gt = get_ground_truth_one_hot_vector(recording)
                else: gt = get_ground_truth_one_hot_vector("000_190105") # placeholder
                # evaluate metrics
                for i in range(len(rows_trial)):
                    row = rows_trial[i]
                    # WER, CER
                    if(not pipeline_config.endtoendspv):
                        row[3], row[4] = get_wer_and_cer(recording, row[1])
                        row[8] = check_protocol_correct(recording, row[6])
                        
                    else: 
                        row[3], row[4] = get_wer_and_cer("000_190105", row[1]) # placeholder
                        row[8] = check_protocol_correct("000_190105", row[6]) # placeholder
                        
                    # protocol correct? (1 = True, 0 = False, -1=None given) 
                    # if protocol given, evaluate protocol model
                    if row[7] > 0:
                        pred = np.array(row[9])
                        logits = np.array(row[15])
                        # one hot prediction
                        row[9] = str(pred)
                        # one hot GT
                        row[10] = str(gt)
                        # tn, fp, fn, tp
                        row[11], row[12], row[13], row[14] = confusion_matrix(gt, pred).ravel()
                
                # save last one hot vectors and logit
                one_hot_pred_all_recordings.append(pred)
                one_hot_gt_all_recordings.append(gt)
                logits_all_recordings.append(logits)

                # save dataframe
                df = pd.DataFrame(rows_trial, columns=fields)
                df_all_recordings.append(df)
                
                # Write to csv
                if(pipeline_config.data_save):
                    directory = f"Evaluation_Results/{time_stamp}/{pipeline_config.protocol_model_type}/{pipeline_config.protocol_model_device}/{speech_model}/"
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    df.to_csv(f'{directory}T{trial}_{recording}.csv')
                         
            # protocol model report for ALL recordings
            if(pipeline_config.data_save and pipeline_config.speech_model == "whisper" and not pipeline_config.endtoendspv):
                report = classification_report(one_hot_gt_all_recordings, one_hot_pred_all_recordings, target_names=ungroup_p_node, output_dict=True)
                with open(f'Evaluation_Results/{time_stamp}/{pipeline_config.protocol_model_type}/{pipeline_config.protocol_model_device}/{speech_model}/protocol-model-evaluation-report.txt', 'w') as f:
                    f.write(str(report))
                    
            # protocol model report
            report = classification_report(one_hot_gt_all_recordings, one_hot_pred_all_recordings, target_names=ungroup_p_node, output_dict=True)
            p1 = get_precision_at_k(np.array(one_hot_gt_all_recordings), np.array(logits_all_recordings), k=1)
            r1 = get_recall_at_k(np.array(one_hot_gt_all_recordings), np.array(logits_all_recordings), k=1)
            dcg1 = get_ndcg_at_k(np.array(one_hot_gt_all_recordings), np.array(logits_all_recordings), k=1)
            rprecision1 = get_r_precision_at_k(np.array(one_hot_gt_all_recordings), np.array(logits_all_recordings), k=1)
            report['P@1'] = p1
            report['R@1'] = r1
            report['nDCG@1'] = dcg1
            report['R-Precision@1'] = rprecision1

            df = pd.DataFrame(report).T
            df.to_csv(f'{directory}T{trial}_protocol-model-evaluation-report.csv')
            
            if(pipeline_config.speech_model == 'conformer'): break
