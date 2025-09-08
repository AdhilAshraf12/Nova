# speech/tts.py
import shutil
import subprocess
from pathlib import Path
import simpleaudio as sa
from nova.config import settings

def _exists(path: str) -> bool:
    p = Path(path)
    return bool(path and (p.exists() or shutil.which(path) is not None))

class TTS:
    def __init__(self):
        self.exe = settings.piper_exe
        self.voice = settings.piper_voice
        self.out = str(Path(settings.piper_out or "out.wav").resolve())
        self._have_piper = _exists(self.exe) and _exists(self.voice)

    def _run_piper(self, text: str):
        Path(self.out).parent.mkdir(parents=True, exist_ok=True)

        # Try long flags first
        cmd_long = [
            self.exe, "--model", self.voice,
            "--output_file", self.out,
            "--text", text
        ]
        # Then short flags (for older builds)
        cmd_short = [
            self.exe, "-m", self.voice,
            "-o", self.out,
            "-t", text
        ]

        try:
            subprocess.run(cmd_long, check=True)
        except Exception:
            subprocess.run(cmd_short, check=True)

    def _play_wav(self):
        wave = sa.WaveObject.from_wave_file(self.out)
        play = wave.play()
        play.wait_done()

    def speak(self, text: str):
        if not text:
            return
        if not self._have_piper:
            print("[TTS fallback]", text)
            return
        try:
            self._run_piper(text)
            if not Path(self.out).exists():
                raise FileNotFoundError(self.out)
            self._play_wav()
        except Exception as e:
            print("[TTS warn] Piper failed:", repr(e))
            print("[TTS fallback]", text)
