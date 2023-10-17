# python script to run end to end evaluations

from Pipeline import Pipeline
import pipeline_config
import csv
import os
# from jiwer import wer, cer
# from transformers import WhisperProcessor
# from evaluate import load

# -- static helper variables ---------------------------------------
# initialize processor depending on whisper model size 
# if pipeline_config.whisper_model_size == 'base-finetuned':
#     Processor = WhisperProcessor.from_pretrained("saahith/whisper-base.en-combined-v10")
# elif pipeline_config.whisper_model_size == 'base.en':
#     Processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
# elif pipeline_config.whisper_model_size == 'tiny-finetuned':
#     Processor = WhisperProcessor.from_pretrained("saahith/whisper-tiny.en-combined-v10")
# elif pipeline_config.whisper_model_size == 'tiny.en':
#     Processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
# wer_metric = load("wer")
# cer_metric = load("cer")


# --- helper methods -----------------
def get_ground_truth_transcript(recording):
    with open(f'Audio_Scenarios/2019_Test_Ground_Truth/{recording}-transcript.txt') as f:
        ground_truth = f.read()
    return ground_truth

def get_ground_truth_protocol(recording):
    with open(f'Audio_Scenarios/2019_Test_Ground_Truth/{recording}-protocol.txt') as f:
        ground_truth = f.read()
    return ground_truth

def get_wer_and_cer(recording, transcript):
    # tokenized_reference_text = Processor.tokenizer._normalize(get_ground_truth_transcript(recording))
    # tokenized_prediction_text = Processor.tokenizer._normalize(transcript)
    # wer = wer_metric.compute(references=[tokenized_reference_text], predictions=[tokenized_prediction_text])
    # cer = cer_metric.compute(references=[tokenized_reference_text], predictions=[tokenized_prediction_text])
    wer = 0
    cer = 0
    return wer, cer

def check_protocol_correct(recording, protocol):
    if protocol == -1: return -1
    ground_truth = get_ground_truth_protocol(recording)
    return int(protocol.lower() == ground_truth.lower())

# --- main ---------------------------
if __name__ == '__main__':
    '''
    this variable simply decides under which folder the evaluation runs are saved 
    change this variable, then to specifcy which device the protocol model runs on
    go to EMSAgent/default_sets.py and change the device variable on line 18 manually 
    to either 'cpu' (for CPU) or 'cuda' (for GPU)
    TODO: make device variable part of config file so you don't have to change it in two files
    '''
    device = 'cuda' # or cuda
    
    for recording in pipeline_config.recordings_to_test:
        # run one trial of the pipeline 
        for i in range(1,pipeline_config.num_trials_per_recording+1):
            # field names
            fields = ['time audio->transcript (s)', 'transcript', 'whisper confidence', 'WER', 'CER', 'time protocol input->output (s)', 'protocol prediction', 'protocol confidence', 'protocol correct? (1 = True, 0 = False, -1=None given)'] 

            # data rows of csv file for this run
            pipeline_config.curr_segment = []
            pipeline_config.rows_trial = []

            # run pipeline
            Pipeline(recording=recording)

            # calculate wer and cer after pipeline run is finished
            # check if protocol prediction is correct
            for row in pipeline_config.rows_trial:
                row[3], row[4] = get_wer_and_cer(recording, row[1])
                row[8] = check_protocol_correct(recording, row[6])
                
            # Write data to csv
            directory = f'evaluation_results/{device}/{pipeline_config.whisper_model_size}/{recording}-all-trials/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            print(f'Evaluation Results stored in {directory}')
            with open (f'{directory}{recording}-trial-{i}.csv', 'w+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(fields)
                writer.writerows(pipeline_config.rows_trial)
        # TODO
    #     # find average of all trials for the recording
    #     with open (f'evaluation_results/{device}/{recording}-average-{i}.csv', 'w') as csvFile:
    #         fields = ['time audio->transcript (s)', 'whisper confidence', 'WER', 'CER', 'time protocol input->output (s)', 'protocol prediction', 'protocol confidence', 'protocol accuracy'] 
    #         pass
    # # TODO
    # # find average of ALL RECORDINGS
    # with open (f'evaluation_results/{device}/all-recordings-average-{i}.csv', 'w') as csvFile:
    #     fields = ['time audio->transcript (s)', 'whisper confidence', 'WER', 'CER', 'time protocol input->output (s)', 'protocol prediction', 'protocol confidence', 'protocol accuracy'] 
    #     pass

    

    
