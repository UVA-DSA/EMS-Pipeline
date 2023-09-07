model_name = "whisper-small.en"
# VAD_threshold = 0.93 # threshold for VAD to register sound as speech
VAD_threshold = 0.87
PATH_TO_WHISPER_CPP_FOLDER = "/Users/saahithjanapati/Desktop/whisper.cpp"
mode = "huggingface"
AUDIO_SEGMENT_DURATION_LIMIT = 3 # length of segments processed by whisper


"""
setting this to smaller number will let results be finalized quicker, 
but may also result in worse performance because of less 
context for whisper to work with
"""
