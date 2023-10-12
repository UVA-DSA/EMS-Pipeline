# python script to run end to end evaluations

from Pipeline import Pipeline
import pipeline_config
import csv
from transformers import WhisperProcessor
from evaluate import load
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from EMSAgent.default_sets import ungroup_p_node

# -- static helper variables ---------------------------------------
# initialize processor depending on whisper model size 
if pipeline_config.whisper_model_size == 'base-finetuned':
    Processor = WhisperProcessor.from_pretrained("saahith/whisper-base.en-combined-v10")
elif pipeline_config.whisper_model_size == 'base.en':
    Processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
elif pipeline_config.whisper_model_size == 'tiny-finetuned':
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
    if protocol == -1: return -1
    ground_truth = get_ground_truth_protocol(recording)
    return int(protocol.lower() == ground_truth.lower())

# --- main ---------------------------
if __name__ == '__main__':

    one_hot_pred_all_recordings = []
    one_hot_gt_all_recordings = []
    
    for recording in pipeline_config.recordings_to_test:
        # run one trial of the pipeline 
        for trial_num in range(1,pipeline_config.num_trials_per_recording+1):
            # field names
            fields = ['time audio->transcript (s)', 'transcript', 'whisper confidence', 'WER', 'CER', #4
                      'time protocol input->output (s)', 'protocol prediction', 'protocol confidence', #7
                      'protocol correct? (1 = True, 0 = False, -1=None given)', #8
                      'one hot prediction', 'one hot GT', 'tn', 'fp', 'fn', 'tp'] 

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
                if row[8] != -1:
                    pred = np.array(row[9])
                    print('gt', gt)
                    print('pred', pred)
                    # one hot prediction
                    row[9] = str(pred)
                    # one hot GT
                    row[10] = str(gt)
                    # tn, fp, fn, tp
                    row[11], row[12], row[13], row[14] = confusion_matrix(gt, pred).ravel()
            
            # save last one hot vectors
            one_hot_pred_all_recordings.append(pred)
            one_hot_gt_all_recordings.append(gt)
            
            # Write data to csv
            with open (f'Evaluation_Results/{pipeline_config.protocol_model_type}/{pipeline_config.protocol_model_device}/{pipeline_config.whisper_model_size}/{recording}-trial-{trial_num}.csv', 'w') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(fields)
                writer.writerows(rows_trial)
        # TODO
    #     # find average of all trials for the recording
    #     with open (f'Evaluation_Results/{device}/{recording}-average-{i}.csv', 'w') as csvFile:
    #         fields = ['time audio->transcript (s)', 'whisper confidence', 'WER', 'CER', 'time protocol input->output (s)', 'protocol prediction', 'protocol confidence', 'protocol accuracy'] 
    #         pass
    # # TODO
    # evalation for ALL RECORDINGS
    report = classification_report(one_hot_gt_all_recordings, one_hot_pred_all_recordings, target_names=ungroup_p_node, output_dict=True)
    with open(f'Evaluation_Results/{pipeline_config.protocol_model_type}/{pipeline_config.protocol_model_device}/{pipeline_config.whisper_model_size}/protocol-model-evaluation-report.txt', 'w') as f:
        f.write(str(report))


    # with open (f'Evaluation_Results/{device}/all-recordings-average-{i}.csv', 'w') as csvFile:
    #     fields = ['time audio->transcript (s)', 'whisper confidence', 'WER', 'CER', 'time protocol input->output (s)', 'protocol prediction', 'protocol confidence', 'protocol accuracy'] 
    #     pass

    

    
