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
    tokenized_reference_text = Processor.tokenizer._normalize(get_ground_truth_transcript(recording))
    tokenized_prediction_text = Processor.tokenizer._normalize(transcript)
    wer = wer_metric.compute(references=[tokenized_reference_text], predictions=[tokenized_prediction_text])
    cer = cer_metric.compute(references=[tokenized_reference_text], predictions=[tokenized_prediction_text])
    return wer, cer

def check_protocol_correct(recording, protocol):
    print('check protocol correct:',protocol)
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
    
    for recording in pipeline_config.recordings_to_test:
        # field names
        fields = ['speech latency (ms)', 'transcript', 'whisper confidence', 'WER', 'CER', #4
                    'protocol latency (ms)', 'protocol prediction', 'protocol confidence', #7
                    'protocol correct? (1 = True, 0 = False, -1=None given)', #8
                    'one hot prediction', 'one hot GT', 'tn', 'fp', 'fn', 'tp', 'logits'] 

        # data rows of csv file for this run
        pipeline_config.curr_segment = []
        pipeline_config.rows_trial = []

        # run pipeline
        Pipeline(recording=recording)
        
        # get data
        rows_trial = pipeline_config.rows_trial
        # get ground truth one hot vector
        gt = get_ground_truth_one_hot_vector(recording)
        # evaluate metrics
        for i in range(len(rows_trial)):
            row = rows_trial[i]
            # WER, CER
            row[3], row[4] = get_wer_and_cer(recording, row[1])
            # protocol correct? (1 = True, 0 = False, -1=None given) 
            row[8] = check_protocol_correct(recording, row[6])
            # if protocol given, evaluate protocol model
            if row[7] > 0:
                pred = np.array(row[9])
                logits = np.array(row[15])
                # one hot prediction
                row[9] = str(pred)
                # one hot GT
                row[10] = str(gt)
                # logits
                row[15] = str(logits)
                # tn, fp, fn, tp
                row[11], row[12], row[13], row[14] = confusion_matrix(gt, pred).ravel()
                
        
        # save last one hot vectors and logit
        one_hot_pred_all_recordings.append(pred)
        one_hot_gt_all_recordings.append(gt)
        logits_all_recordings.append(logits)

        # save dataframe
        df = pd.DataFrame(rows_trial, columns=fields)
        df_all_recordings.append(df)

        # Write data to csv
        df.to_csv(f'Evaluation_Results/{pipeline_config.protocol_model_type}/{pipeline_config.protocol_model_device}/{pipeline_config.whisper_model_size}/{recording}.csv')

    # evalation for ALL RECORDINGS

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
    df.to_csv(f'Evaluation_Results/{pipeline_config.protocol_model_type}/{pipeline_config.protocol_model_device}/{pipeline_config.whisper_model_size}/protocol-model-evaluation-report.csv')

    # end to end report
    fields = [
        'avg speech latency',  # average of segment latencies for each recording
        'whisper confidence', # LAST CONFIDENCE for each recording
        'WER', # LAST WER for each recording
        'CER', # LAST CER for each recording
        'avg protocol latency (ms)', # average of segment latencies for each recroding
        'protocol confidence',  # LAST CONFIDENCE for each recording
        'protocol correct? (1 = True, 0 = False, -1=None given)', # LAST prediction correct?
        'tn', 'fp', 'fn', 'tp' # LAST values
    ]
    rows = []

    for df in df_all_recordings:
        avg_lat_whisper = df['speech latency (ms)'].mean()
        whisper_confidence = df['whisper confidence'].iloc[-1]
        WER = df['WER'].iloc[-1]
        CER = df['CER'].iloc[-1]
        avg_lat_protocol = df['protocol latency (ms)'].mean()
        protocol_confidence = df['protocol confidence'].iloc[-1]
        protocol_correct = df['protocol correct? (1 = True, 0 = False, -1=None given)'].iloc[-1]
        tn = df['tn'].iloc[-1]
        fp = df['fp'].iloc[-1]
        fn = df['fn'].iloc[-1]
        tp = df['tp'].iloc[-1]
        rows.append([avg_lat_whisper, whisper_confidence, WER, CER, avg_lat_protocol, protocol_confidence, protocol_correct, tn, fp, fn, tp])
    
    df = pd.DataFrame(rows, columns=fields)
    df.loc[len(df.index)] = df.mean().T
    df.index = pipeline_config.recordings_to_test + ["overall average"]

    # write to csv
    df.to_csv(f'Evaluation_Results/{pipeline_config.protocol_model_type}/{pipeline_config.protocol_model_device}/{pipeline_config.whisper_model_size}/end-to-end-evaluation-report.csv')



        

    
