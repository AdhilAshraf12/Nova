from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    ollama_base: str = os.getenv("OLLAMA_BASE","http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL","llama3.2:3b")
    piper_exe: str = os.getenv("PIPER_EXE","piper")
    piper_voice: str = os.getenv("PIPER_VOICE","voices/en-us-libritts-high.onnx")
    piper_out: str = os.getenv("PIPER_OUT","out.wav")
    whisper_model: str = os.getenv("WHISPER_MODEL", "small")
    whisper_compute: str = os.getenv("WHISPER_COMPUTE_MODEL","int8")
    wakeword: str = os.getenv("WAKEWORD","nova").lower()
    memory_file: str = os.getenv("MEMORY_FILE","./data/memory.json")

settings = Settings()
