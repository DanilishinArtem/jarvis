import speech_recognition as sr

class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def listen(self) -> str:
        with self.microphone as source:
            print("Слушаю...")
            audio = self.recognizer.listen(source, phrase_time_limit=5)
        try:
            text = self.recognizer.recognize_google(audio, language="ru-RU")
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return ""