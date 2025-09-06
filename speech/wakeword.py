from nova.config import settings


class Wakeword:
    def __init__(self):
        self.key = settings.wakeword


    def detected(self, text: str) -> bool:
        return self.key in text.lower()