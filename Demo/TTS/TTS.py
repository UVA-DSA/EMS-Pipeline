import pyttsx

engine = pyttsx.init()

def TTS(text):
    engine.say(text)
    engine.runAndWait()

if __name__ == '__main__':

    #TTS("Hello, my name is Joe! I am a cognitive assistant for EMS. How may I help you?")
    #StoppableThread(target = TTS, args=(str(pr),)).start() # Text to speech
