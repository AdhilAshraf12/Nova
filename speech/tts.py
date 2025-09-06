import subprocess
import soundfile as sf
import sounddevice as sd
from nova.config import settings

class TTS:
    def __init__(self):
        self.exe = settings.piper_exe
        self.voice = settings.piper_voice
        self.out = settings.piper_out

    def speak(self, text: str):
        # Generate wav via Piper
        cmd = [self.exe, "-m", self.voice, "-f", self.out, "-t", text]
        subprocess.run(cmd, check=True)
        # Play wav cross‑platform with sounddevice
        data, samplerate = sf.read(self.out, dtype='float32')
        sd.play(data, samplerate)
        sd.wait()