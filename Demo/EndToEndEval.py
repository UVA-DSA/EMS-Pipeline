# python script to run end to end evaluations

from Pipeline import Pipeline
import pipeline_config
import csv


# --- helper methods -----------------

def get_ground_truth_transcript(recording):
    with open(f'Audio_Scenarios/2019_Test_Ground_Truth/{recording}-transcript.txt') as f:
        ground_truth = f.read()
    return ground_truth

def get_ground_truth_protocol(recording):
    with open(f'Audio_Scenarios/2019_Test_Ground_Truth/{recording}-protocol.txt') as f:
        ground_truth = f.read()
    return ground_truth

def calc_wer(recording, transcript):
    ground_truth = get_ground_truth_transcript(recording)
    ref_words = ground_truth.split()
    hyp_words = transcript.split()
	# Counting the number of substitutions, deletions, and insertions
    substitutions = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp)
    deletions = len(ref_words) - len(hyp_words)
    insertions = len(hyp_words) - len(ref_words)
	# Total number of words in the reference text
    total_words = len(ref_words)
	# Calculating the Word Error Rate (WER)
    wer = (substitutions + deletions + insertions) / total_words
    return wer
    
def calc_cer(recording, transcript):
    ground_truth = get_ground_truth_transcript(recording)
    ref_chars = list(ground_truth)
    hyp_chars = list(transcript)
	# Counting the number of substitutions, deletions, and insertions
    substitutions = sum(1 for ref, hyp in zip(ref_chars, hyp_chars) if ref != hyp)
    deletions = len(ref_chars) - len(hyp_chars)
    insertions = len(hyp_chars) - len(ref_chars)
	# Total number of chars in the reference text
    total_chars = len(ref_chars)
	# Calculating the Character Error Rate (CER)
    cer = (substitutions + deletions + insertions) / total_chars
    return cer

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
    device = 'cpu' # or cuda
        
    for recording in pipeline_config.recordings_to_test:
        # run one trial of the pipeline 
        for i in range(1,pipeline_config.num_trials_per_recording+1):
            # field names
            fields = ['time audio->transcript (s)', 'transcript', 'whisper confidence', 'WER', 'CER', 'time protocol input->output (s)', 'protocol prediction', 'protocol confidence', 'protocol correct? (1 = True, 0 = False, -1=None given)'] 

            # data rows of csv file for this run
            pipeline_config.curr_segment = []
            pipeline_config.rows_trial = []
            Pipeline(recording=recording)

            # calculate wer and cer after pipeline run is finished
            # check if protocol prediction is correct
            for row in pipeline_config.rows_trial:
                row[3] = calc_wer(recording, row[1])
                row[4] = calc_cer(recording, row[1])
                row[8] = check_protocol_correct(recording, row[6])
                
            # Write data to csv
            with open (f'evaluation_results/{device}/{recording}-all-trials/{recording}-trial-{i}.csv', 'w') as csvFile:
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

    

    
