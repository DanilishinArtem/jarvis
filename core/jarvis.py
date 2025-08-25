import time
from .stt import SpeechRecognizer
from .tts import TTS
from .actions import ActionExecutor
from .logger import Logger

class Jarvis:
    def __init__(self):
        self.stt = SpeechRecognizer()
        self.tts = TTS()
        self.actions = ActionExecutor(self.tts)
        self.log = Logger()

    def run(self):
        self.tts.say("J.A.R.V.I.S. запущен и готов к работе.")
        while True:
            try:
                text = self.stt.listen()
                if not text:
                    continue
                self.log.write("Распознано: {}".format(text))
                handled = self.actions.execute(text.lower())
                if not handled:
                    self.tts.say("Я не понял ваш запрос.")
            except KeyboardInterrupt:
                self.tts.say("Останавливаюсь. До встречи!")
                break