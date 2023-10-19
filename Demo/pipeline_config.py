# -- Pipeline option configuration -- 
datacollection = True
videostream = True
audiostream = True
smartwatchStream = True
conceptExtractionStream = True
protocolStream = True
interventionStream = True
transcriptStream = True

# -- Audio Recording Option -----
hardcoded = True
recording_name = '000_190105'
video_name = 'scenario_1'

# -- Whisper configuration ---
whisper_model_size = "base-finetuned"
whisper_model_size = "base.en"

# -- Whisper configuration ---
whisper_model_sizes = [
    "tiny-finetuned-v2",
    "tiny-finetuned-q5",
    "tiny-q5",
    "tiny-finetuned",
    "tiny.en",
    "base.en",
    "base-q5",
    "base-finetuned",
    "base-finetuned-q5"
]

PATH_TO_WHISPER_CPP_FOLDER = "/home/cogems_nist/Desktop/CogEMS_NIST/whisper.cpp"
num_threads = 8
step = 3000
length = 30000
keep_ms = 500 #audio to keep from previous step in ms

# -- EMS Agent configuration ------
protocol_model_device = 'cuda' # 'cuda' or 'cpu'
protocol_model_type = 'DKEC-TinyClinicalBERT' #DKEC-TinyClinicalBERT, EMSAssist

# -- EMS Vision configuration ------
vision_model_type = 'openai/clip-vit-base-patch32' 


# --- End to End evaluation testing configs --------
recordings_to_test = [
    '000_190105',
    # '001_190105',
    # '002_190105',
    # '003_190105',
    # '004_190105',
    # '005_190105',
    # '006_190105',
    # '007_190105',
    # '008_190105',
    # '009_190105',
    # '010_190105',
    # '011_190105'
]

# --- global variables used during end to end eval ----
curr_segment = []
rows_trial = []