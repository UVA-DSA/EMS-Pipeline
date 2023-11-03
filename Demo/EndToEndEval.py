# python script to run end to end evaluations

from Pipeline import Pipeline
import pipeline_config
import os
import pandas as pd
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

# def get_ground_truth_one_hot_vector(recording):
#     ground_truth = get_ground_truth_protocol(recording)
#     one_hot_vector = [1 if label.lower() == ground_truth.lower() else 0 for label in ungroup_p_node]
#     return np.array(one_hot_vector)


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
    
    for trial in range(pipeline_config.num_trials_per_recording):
        for whisper_model in pipeline_config.whisper_model_sizes:
            for recording in pipeline_config.recordings_to_test:
                # set field names in trial data dictionary 
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
                if pipeline_config.action_recognition:
                    pipeline_config.vision_data['protocol'] = []
                    pipeline_config.vision_data['intervention recognition'] = []
                    pipeline_config.vision_data['intervention latency'] = []


                # run pipeline
                print("Running E2E Evaluation for Recording: ",recording)
                print("\n\n\n ********************** CONFIGURATION **************************** \n\n")

                print("speech-model:",whisper_model)
                print("protocol-model:",pipeline_config.protocol_model_type)
                print("protocol-device:",pipeline_config.protocol_model_device)
                print("isEnd-to-End:",pipeline_config.action_recognition)
                print("Trial:",trial)
                print("Recording:",recording)

                print("\n\n\n ******************************************************** \n\n")

                Pipeline(recording=recording, whisper_model=whisper_model, trial=trial)

                # Write data to csv
                directory = f'evaluation_results/{device}/{whisper_model}/'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                print(f'Evaluation Results stored in {directory}')

                df = pd.DataFrame(pipeline_config.trial_data)
                df.to_csv(f'{directory}T{trial}_SPEECH+PROTOCOL_{recording}.csv')

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

    

    
