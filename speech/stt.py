import queue
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from nova.config import settings

SAMPLE_RATE = 16000
FRAME_MS = 20  # analysis frame
RMS_THRESHOLD = 0.012  # tweak if too sensitive/insensitive
MIN_SPEECH_MS = 300    # require this much speech before starting capture
MAX_SILENCE_MS = 800   # stop after this much silence

class STT:
    def __init__(self):
        self.model = WhisperModel(
            settings.whisper_model,
            compute_type=settings.whisper_compute
        )
        self.audio_q = queue.Queue()

    def _audio_callback(self, indata, frames, time, status):
        if status:
            pass
        # float32 mono 16k
        self.audio_q.put(indata.copy())

    def _rms(self, x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x))))

    def record_utterance(self):
        frame_samples = int(SAMPLE_RATE * FRAME_MS / 1000)
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
                # Pad/truncate to frame_samples
                if buf.shape[0] != frame_samples:
                    if buf.shape[0] > frame_samples:
                        buf = buf[:frame_samples]
                    else:
                        pad = np.zeros((frame_samples - buf.shape[0], 1), dtype=np.float32)
                        buf = np.vstack([buf, pad])

                rms = self._rms(buf[:, 0])
                is_voice = rms >= RMS_THRESHOLD

                if is_voice:
                    speech_ms += FRAME_MS
                    silence_ms = 0
                    if not started and speech_ms >= MIN_SPEECH_MS:
                        started = True
                    if started:
                        collected.append(buf[:, 0])
                else:
                    speech_ms = 0 if not started else speech_ms
                    if started:
                        silence_ms += FRAME_MS
                        collected.append(buf[:, 0])
                        if silence_ms >= MAX_SILENCE_MS:
                            break

        if not collected:
            return ""
        audio = np.concatenate(collected, axis=0)
        segments, _ = self.model.transcribe(audio, language='en')
        text = ''.join(seg.text for seg in segments).strip()
        return text
