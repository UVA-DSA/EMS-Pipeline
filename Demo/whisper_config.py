model_name = "whisper-small.en"
VAD_threshold = 0.93 # threshold for VAD to register sound as speech

"""
setting this to smaller number will let results be finalized quicker, 
but may also result in worse performance because of less 
context for whisper to work with
"""
AUDIO_SEGMENT_DURATION_LIMIT = 7 # length of segments processed by whisper
