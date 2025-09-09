# speech/stt.py  (lazy Whisper init + RMS debug)
import queue
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from nova.config import settings

SAMPLE_RATE = 16000
RMS_THRESHOLD = 0.00025   # matches your ~0.00030–0.00034 noise floor
MIN_SPEECH_MS = 300
MAX_SILENCE_MS = 1200
PRE_SPEECH_MS = 200

class STT:
    def __init__(self):
        self.model = None  # lazy init to avoid early crashes
        self.audio_q = queue.Queue()
        self.pre_buffer = []
        self.pre_ms = 0

    def _ensure_model(self):
        if self.model is None:
            self.model = WhisperModel(
                settings.whisper_model,
                device="cpu",                      # avoid CUDA/cuDNN issues
                compute_type=settings.whisper_compute
            )

    def _audio_callback(self, indata, frames, time, status):
        if status:
            pass
        self.audio_q.put(indata.copy())

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(x))))

    def record_utterance(self):
        self._ensure_model()

        speech_ms = 0
        silence_ms = 0
        started = False
        collected = []

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            callback=self._audio_callback
        )
        with stream:
            while True:
                buf = self.audio_q.get()
                mono = buf[:, 0]
                chunk_ms = int(len(mono) * 1000 / SAMPLE_RATE)

                if not started:
                    self.pre_buffer.append(mono)
                    self.pre_ms += chunk_ms
                    while self.pre_ms > PRE_SPEECH_MS and len(self.pre_buffer) > 1:
                        drop = self.pre_buffer.pop(0)
                        self.pre_ms -= int(len(drop) * 1000 / SAMPLE_RATE)

                rms = self._rms(mono)
                # print(f"RMS: {rms:.5f}", flush=True)  # live mic level
                is_voice = rms >= RMS_THRESHOLD

                if is_voice:
                    silence_ms = 0
                    speech_ms += chunk_ms
                    if not started and speech_ms >= MIN_SPEECH_MS:
                        started = True
                        if self.pre_buffer:
                            collected.extend(self.pre_buffer)
                            self.pre_buffer = []
                            self.pre_ms = 0
                    if started:
                        collected.append(mono)
                else:
                    if started:
                        silence_ms += chunk_ms
                        collected.append(mono)
                        if silence_ms >= MAX_SILENCE_MS:
                            break
                    else:
                        speech_ms = 0

        if not collected:
            return ""
        audio = np.concatenate(collected, axis=0)
        segments, _ = self.model.transcribe(audio, language='en')
        text = ''.join(seg.text for seg in segments).strip()
        return text
