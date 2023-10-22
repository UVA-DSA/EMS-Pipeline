# -- Pipeline option configuration -- 
datacollection = True
videostream = True
audiostream = True
smartwatchStream = True
conceptExtractionStream = True
protocolStream = True
interventionStream = True
transcriptStream = True


# ------------------- Experiment Control ------------------- #

endtoendspv = False #speech+protocol+vision



# ------------------- Speech to Text Control ------------------- #

speech_model = 'conformer' # 'conformer'
# speech_model = 'whisper' # 'conformer'

# -- Audio Recording Option -----
hardcoded = True
recording_name = '000_190105'
video_name = 'scenario_1'

# -- Whisper configuration ---
whisper_model_size = "base-finetuned"
whisper_model_size = "base.en"

# -- Whisper configuration ---
whisper_model_sizes = [
    # "base-wo-emsassist",
    # "base-wo-synth-v1",
    # "tiny-wo-synth-v1",
    # "base-finetuned-v6",
    # "base-finetuned-v5",
    # "base-finetuned-v4",
    # "base-finetuned-v3", #best
    # "base-finetuned-v2",
    # "tiny-finetuned-v5",
    # "tiny-finetuned-v4",
    # "tiny-finetuned-v3",
    # "tiny-finetuned-v2", #best
    # "tiny-finetuned",
    # "tiny.en", #baseline
    # "base.en", #baseline
    # "base-finetuned",
]





PATH_TO_WHISPER_CPP_FOLDER = "/home/cogems_nist/Desktop/CogEMS_NIST/whisper.cpp"
num_threads = 8
step = 2000
length = 30000
keep_ms = 200 #audio to keep from previous step in ms


# -- EMS Conformer configuration ------
conformer_model_type = 'all_14_model.tflite' 



# -- EMS Agent configuration ------
protocol_model_device = 'cuda' # 'cuda' or 'cpu'
protocol_model_type = 'DKEC-TinyClinicalBERT' #DKEC-TinyClinicalBERT, EMSAssist
protocol_model_type = 'EMSAssist' #DKEC-TinyClinicalBERT, EMSAssist

# -- EMS Vision configuration ------
vision_model_type = 'openai/clip-vit-base-patch32' 



# --- End to End evaluation testing configs --------
if not endtoendspv:
    recordings_to_test = [
        '000_190105',
        '001_190105',
        '002_190105',
        '003_190105',
        '004_190105',
        '005_190105',
        '006_190105',
        '007_190105',
        '008_190105',
        '009_190105',
        '010_190105',
        '011_190105'
    ]

else:
    recordings_to_test = [  # audio + video e2e recordings
        "scenario_1",
        "scenario_2",
        "scenario_3",
        "scenario_4",
        "scenario_5",
        "scenario_6",
        "scenario_7",
        "scenario_8",
    ]

# --- global variables used during end to end eval ----

data_save = True
curr_segment = []
vision_curr_segment = []
rows_trial = []
vision_rows_trial = []


num_trials = 3