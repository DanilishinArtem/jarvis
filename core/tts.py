import pyttsx3

class TTS:
    def __init__(self):
        self.engine = pyttsx3.init()

    def say(self, text: str):
        print("[JARVIS]: {}".format(text))
        self.engine.say(text)
        self.engine.runAndWait()