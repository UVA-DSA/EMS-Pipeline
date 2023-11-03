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

speech_standalone = True

# -- Action Recognition Option -----
action_recognition= False
#clip_model = "CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-f16-q4.gguf"
clip_model = "CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-f16.gguf"
clip_threads = 1
protocol_fifo = '/tmp/protocolfifo'


# -- Whisper configuration ---
whisper_model_sizes = [
    #  "finetuned-base-final",
    #  "finetuned-tiny-final",
    #  "base-finetuned-wo-synth",
    #  "tiny-finetuned-wo-synth",
    #  "finetuned-tiny-v100",
    #  "finetuned-tiny-v101",
    #  "finetuned-tiny-v102",
    #  "finetuned-tiny-v103",
    #  "finetuned-tiny-v104",
    #  "finetuned-tiny-v105",
    #  "finetuned-tiny-v106",
    #  "finetuned-tiny-v107",
    #  "finetuned-tiny-v108",
    #  "finetuned-tiny-v109",
    #  "finetuned-tiny-v110",
     "finetuned-tiny-v111",
    #  "finetuned-tiny-v112",
    #  "finetuned-tiny-v113",
    #  "finetuned-base-v200",
    #  "finetuned-base-v201",
    #  "finetuned-base-v202",
    #  "finetuned-base-v203",
    #  "finetuned-base-v204",
    #  "finetuned-base-v205",
    #  "finetuned-base-v206",
    #  "finetuned-base-v207",
    #  "finetuned-base-v208",
    # "tiny-finetuned-v20",
    # "tiny-finetuned-v21",
    # "tiny-finetuned-v22",
    # "tiny-finetuned-v23",
    # "tiny-finetuned-v24",
    # "tiny-finetuned-v25",
    # "tiny-finetuned-v26",
    # "base-wo-emsassist",
    # "base-wo-synth-v1",
    # "tiny-wo-synth-v1",
    # "base-finetuned-v6",
    # "base-finetuned-v5",
    # "base-finetuned-v4",
    # "base-finetuned-v3", 
    # "base-finetuned-v2",
    # "tiny-finetuned-v5",
    # "tiny-finetuned-v4",
    # "tiny-finetuned-v3",
    # "tiny-finetuned-v2", 
    #"tiny-finetuned",
    #  "tiny.en", #baseline
    #  "base.en", #baseline
    # "base-finetuned" #best    
]

PATH_TO_WHISPER_CPP_FOLDER = "/home/cogems_nist/Desktop/CogEMS_NIST/whisper.cpp"
num_threads = 3
step = 4000
length = 8000
keep_ms = 200 #audio to keep from previous step in ms
# audio_ctx = 512 #audio to keep from previous step in ms
vth = 0.8 #voice activity
max_tokens = 10 #voice activity
is_streaming = False

# -- EMS Agent configuration ------
protocol_model_device = 'cuda' # 'cuda' or 'cpu'
# protocol_model_type = 'DKEC-TinyClinicalBERT' #DKEC-TinyClinicalBERT, EMSAssist
protocol_model_type = 'EMSAssist' #DKEC-TinyClinicalBERT, EMSAssist

num_trials_per_recording = 1

# --- End to End evaluation testing configs --------

if not action_recognition:
    recordings_to_test = [
        #  '000_190105',
        # #  '001_190105',
        '002_190105',
        '003_190105',
        '004_190105',
        '005_190105',
        '006_190105',
        '007_190105',
        '008_190105',
        '009_190105',
        '010_190105',
        '011_190105',
        # "scenario_1",
        # "scenario_2",
        # "scenario_3",
        # "scenario_4",
        # "scenario_5",
        # "scenario_6",
        # "scenario_7",
        # "scenario_8"
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
trial_data = dict()
