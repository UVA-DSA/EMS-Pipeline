model_name = "whisper-small.en"
VAD_threshold = 0.93 # threshold for VAD to register sound as speech
PATH_TO_WHISPER_CPP_FOLDER = "/home/cogems_nist/Desktop/CogEMS_NIST/whisper.cpp"
# mode = "huggingface"
mode = "whispercpp"
AUDIO_SEGMENT_DURATION_LIMIT = 3 # length of segments processed by whisper


"""
setting this to smaller number will let results be finalized quicker, 
but may also result in worse performance because of less 
context for whisper to work with
"""
