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
hardcoded = False
recording_name = '000_190105'

# -- Whisper configuration ---
whisper_model_size = "tiny-finetuned-q5"
#whisper_model_size = "tiny-q5"
#whisper_model_size = "tiny-finetuned"
#whisper_model_size = "tiny.en"
#whisper_model_size = "base.en"
#whisper_model_size = "base-q5"
#whisper_model_size = "base-finetuned"
#whisper_model_size = "base-finetuned-q5"
PATH_TO_WHISPER_CPP_FOLDER = "/home/cogems_nist/Desktop/CogEMS_NIST/whisper.cpp"
num_threads = 4
step = 4000
length = 8000
keep_ms = 200 #audio to keep from previous step in ms
# audio_ctx = 512 #audio to keep from previous step in ms

# --- End to End evaluation testing configs --------
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
num_trials_per_recording = 1

# --- global variables used during end to end eval ----
curr_segment = []
rows_trial = []
