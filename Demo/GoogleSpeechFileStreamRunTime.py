import sys
import os
import time
from google.cloud import speech_v1 as speech
import numpy as np
import wave
import io
# assign directory
directory = './Audio_Scenarios/2019_Test/chunked_recordings/000_190105_maxSpeechDuration_7_min_silence_duration_70/'
 




#setting Google credential
os.environ['GOOGLE_APPLICATION_CREDENTIALS']= 'service-account.json'



# iterate over files in
# that directory
for filename in sorted(os.listdir(directory)):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):

        file_name = f

        if("CPR" in file_name):
            RATE = 44100
        else:
            RATE = 16000
        CHUNK = int(RATE)/10
        language_code = 'en-US'  # a BCP-47 language tag

        client = speech.SpeechClient()
        config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                                              enable_automatic_punctuation=True, model= "medical_dictation",
                                            sample_rate_hertz=RATE, language_code=language_code, profanity_filter=True)  # ,model='video')
        streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)



        with io.open(file_name, "rb") as audio_file:
            content = audio_file.read()
            # audio = speech.RecognitionAudio(content=content)

            stream = [content]

            start_t = time.time_ns()

            requests = (
                speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in stream)

            end_t = time.time_ns()

            print("Request Latency: {}".format((end_t-start_t)/1e6) )
                
            try:
                start_t = time.time_ns()

                # Sends the request to google to transcribe the audio
                # responses = client.recognize(request={"config": config, "audio": audio})
                responses = client.streaming_recognize(streaming_config, requests)

                end_t = time.time_ns()

                print("Response Latency: {}".format((end_t-start_t)/1e6) )

                start_t_final = time.time_ns()

                # Reads the response
                for response in responses:
                    start_t = time.time_ns()
                    
                    result = response.results[0]
                    if not result.is_final:
                        print("\n-------------- INTERIM RESULT -------------")
                        print("audio file name: ", file_name)
                        print("Interim Transcript: {}".format(result.alternatives[0]))
                        end_t = time.time_ns()

                        print("Interim Transcription Latency: {}".format((end_t-start_t)/1e6) )
                        print("-------------- END -------------\n")
                    else:
                        print("\n-------------- RESULTS -------------")
                        print("audio file name: ", file_name)
                        print("Final Transcript: {}".format(result.alternatives[0]))
                        end_t = time.time_ns()

                        print("Final Transcription Latency: {}".format((end_t-start_t_final)/1e6) )
                        print("-------------- END -------------\n")

            except Exception as e:
                print(e)