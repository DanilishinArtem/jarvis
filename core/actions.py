import os
import time

class ActionExecutor:
    def __init__(self, tts):
        self.tts = tts

    def execute(self, command: str) -> bool:
        if "открой браузер" in command:
            os.system("start chrome")
            self.tts.say("Открываю браузер")
            return True
        
        elif "напомни через" in command:
            words = command.split()
            for i, w in enumerate(words):
                if w.isdigit():
                    sec = int(w)
                    self.tts.say("Напомню через {} секунд".format(sec))
                    time.sleep(sec)
                    self.tts.say("Напоминаю!")
                    return True
                
        elif "таймер" in command:
            self.tts.say("Таймер на 10 секунд")
            time.sleep(10)
            self.tts.say("Время вышло!")
            return True
        
        return False