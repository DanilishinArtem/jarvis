from datetime import datetime

class Logger:
    def __init__(self, file="jarvis.log"):
        self.file = file

    def write(self, message: str):
        with open(self.file, "a", encoding="utf-8") as f:
            f.write("[{}] {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message))