# -- Pipeline option configuration -- 
datacollection = True
videostream = True
audiostream = True
smartwatchStream = True
conceptExtractionStream = True
protocolStream = True
interventionStream = True
transcriptStream = True

# ------------------- Speech to Text Control ------------------- #
# speech_model = 'conformer' 
speech_model = 'whisper'

# -- EMS Conformer configuration ------
# conformer_model_type = 'all_14_model.tflite' # for tflite
# conformer_model_type = 'finetuned-conformer.tflite' # for finetuned tflite
conformer_model_type = 'h5'  # for base model

# -- EMS Agent configuration ------
protocol_model_device = 'cuda' # 'cuda' or 'cpu'
protocol_model_type = 'DKEC-TinyClinicalBERT' # CognitiveEMS
# # EMSAssist

# -- Audio Recording Option -----
hardcoded = False
recording_name = '000_190105'
video_name = 'scenario_1'

# -- Whisper configuration ---
whisper_model_size = "finetuned-base-v203"
# whisper_model_size = "finetuned-tiny-v111"

PATH_TO_WHISPER_CPP_FOLDER = "/home/cogems_nist/Desktop/CognitiveEMS/Pipeline/EMS_Whisper/whisper.cpp"
num_threads = 4
step = 4000
length = 12000
keep_ms = 100 #audio to keep from previous step in ms

# -- EMS Vision configuration ------
vision_model_type = 'openai/clip-vit-base-patch32' 

# ========EXPERIMENT CONFIGS ============================================================

evaluation = False
speech_standalone = False
protocol_standalone = False
action_standalone = False
endtoendspv = False #speech+protocol+vision

num_trials = 1
# -- Whisper configuration ---
whisper_model_sizes = [
    #  "finetuned-base-final",
    #  "finetuned-tiny-final",
     "base-finetuned-wo-synth", #f
     "tiny-finetuned-wo-synth", #f
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
     "finetuned-tiny-v111", #f
    #  "finetuned-tiny-v112",
    #  "finetuned-tiny-v113",
    #  "finetuned-base-v200",
    #  "finetuned-base-v201",
    #  "finetuned-base-v202",
     "finetuned-base-v203",
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
     "tiny.en", #baseline
     "base.en", #baseline
    # "base-finetuned" #best
]

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
        '011_190105',
        "scenario_1",
        "scenario_2",
        "scenario_3",
        "scenario_4",
        "scenario_5",
        "scenario_6",
        "scenario_7",
        "scenario_8",
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
        "scenario_8"
    ]

# --- global variables used during end to end eval ----
trial_data = dict()
vision_data = dict()
time_stamp = None
directory = None
image_directory = None
trial_num = None
curr_recording = None




# --- Network configurations ----

smartwatch_ip = "172.27.150.154"
smartwatch_port = 7889

smartglass_ip = ""
smartglass_port = 8889





# -- EMS Vision configuration ------
detr_version = "ems" #ems, or base
objectDetectionBoxesenabled = True

# --- socketio config ---
socketio_ipaddr  = 'http://localhost:5000'
feedback_topic = 'feedback'