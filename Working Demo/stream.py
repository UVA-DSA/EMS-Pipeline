import os
import io
from six.moves import queue
import pyaudio
import wave
import sys

RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

def FileStreamGenerator(wavefile):
    wf = wave.open(wavefile, 'rb')
    data = wf.readframes(CHUNK)
    while data != '':
        yield data
        data = wf.readframes(CHUNK)


# Google Cloud Speech API Recognition Thread for local audio file
def GoogleSpeechFile():

    # Set environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"

    from google.cloud import speech
    from google.cloud.speech import enums
    from google.cloud.speech import types
    client = speech.SpeechClient()

    # In practice, stream should be a generator yielding chunks of audio data.
    stream = FileStreamGenerator('File1.wav')
    requests = (types.StreamingRecognizeRequest(audio_content=chunk) for chunk in stream)

    config = types.RecognitionConfig(encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz = 16000, language_code = 'en-US')
    streaming_config = types.StreamingRecognitionConfig(config = config)

    # streaming_recognize returns a generator.
    responses = client.streaming_recognize(streaming_config, requests)

    for response in responses:
        # Once the transcription has settled, the first result will contain the
        # is_final result. The other results will be for subsequent portions of
        # the audio.
        for result in response.results:
            print('Finished: {}'.format(result.is_final))
            print('Stability: {}'.format(result.stability))
            alternatives = result.alternatives
            # The alternatives are ordered from most likely to least.
            for alternative in alternatives:
                print('Confidence: {}'.format(alternative.confidence))
                print(u'Transcript: {}'.format(alternative.transcript))

if __name__ == '__main__':
    GoogleSpeechFile()
