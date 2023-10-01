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
recording_name = '000_190105.wav'

# -- Whisper configuration ---
model_size = "tiny"
PATH_TO_WHISPER_CPP_FOLDER = "/home/cogems_nist/Desktop/CogEMS_NIST/whisper.cpp"
num_threads = 4
step = 3000
length = 3000

# --- End to End eval configs --------
recordings_to_test = [
    '000_190105.wav',
    # '001_190105.wav',
    # '002_190105.wav',
    # '003_190105.wav',
    # '004_190105.wav',
    # '005_190105.wav',
    # '006_190105.wav',
    # '007_190105.wav',
    # '008_190105.wav',
    # '009_190105.wav',
    # '010_190105.wav',
    # '011_190105.wav',
    # 'CPR_transcript1.wav',
    # 'CPR_transcript2.wav',
    # 'CPR_transcript3.wav'
]
num_trials_per_recording = 2

# --- global variables used during end to end eval ----
curr_segment = []
rows_trial = []