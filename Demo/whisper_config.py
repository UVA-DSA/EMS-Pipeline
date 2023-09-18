model_name = "whisper-small.en"

use_vad = False # set this variable to false if you don't want to use voice activity detection
VAD_threshold = 0.93 # threshold for VAD to register sound as speech

PATH_TO_WHISPER_CPP_FOLDER = "/home/cogems_nist/Desktop/CogEMS_NIST/whisper.cpp"
# PATH_TO_WHISPER_CPP_FOLDER = "/Users/saahithjanapati/Desktop/whisper.cpp"
finalization_limit = -1 # finalize transcript after this many seconds (setting this to -1 will force it to finalize immediately)


# mode = "huggingface"
mode = "whispercpp"
AUDIO_SEGMENT_DURATION_LIMIT = 3 # length of segments processed by whisper