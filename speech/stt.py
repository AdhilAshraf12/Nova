import os
import time
import queue
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from nova.config import settings

# Force CPU (avoid CUDA/cuDNN issues)
os.environ["CT2_FORCE_CPU"] = "1"

SAMPLE_RATE = 16000
FRAME_MS = 20
RMS_THRESHOLD = 0.006   # LOWERED so it starts more easily
MIN_SPEECH_MS = 200     # a little snappier
MAX_SILENCE_MS = 600
GET_TIMEOUT_SEC = 5.0   # queue get timeout so we don’t hang if mic stops
LISTEN_WINDOW_MS = 7000 # HARD CAP: if no speech starts in ~7s, we bail

DEBUG = False  # set True to print RMS values

class STT:
    def __init__(self, input_device_index=None):
        sd.default.samplerate = SAMPLE_RATE
        if input_device_index is not None:
            sd.default.device = (input_device_index, None)

        self.model = WhisperModel(
            settings.whisper_model,
            device="cpu",
            compute_type=settings.whisper_compute
        )
        self.audio_q = queue.Queue()

    def _audio_callback(self, indata, frames, time_info, status):
        if status and DEBUG:
            print("[STT] status:", status)
        self.audio_q.put(indata.copy())

    def _rms(self, x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x))))

    def record_utterance(self):
        start_time = time.time()
        frame_samples = int(SAMPLE_RATE * FRAME_MS / 1000)
        speech_ms = 0
        silence_ms = 0
        started = False
        collected = []

        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self._audio_callback
            )
        except Exception as e:
            print(f"[STT] Could not open microphone: {e}")
            return ""

        with stream:
            while True:
                # Hard listen window to avoid “stuck on Listening…”
                if (time.time() - start_time) * 1000 >= LISTEN_WINDOW_MS and not started:
                    if DEBUG:
                        print("[STT] listen window expired without voice")
                    return ""

                try:
                    buf = self.audio_q.get(timeout=GET_TIMEOUT_SEC)
                except queue.Empty:
                    print("[STT] No audio received from mic (timeout).")
                    return ""

                # Pad/truncate to a single analysis frame
                if buf.shape[0] != frame_samples:
                    if buf.shape[0] > frame_samples:
                        buf = buf[:frame_samples]
                    else:
                        pad = np.zeros((frame_samples - buf.shape[0], 1), dtype=np.float32)
                        buf = np.vstack([buf, pad])

                rms = self._rms(buf[:, 0])
                if DEBUG:
                    print(f"[STT] rms={rms:.4f} (thr={RMS_THRESHOLD})")

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
            # Nothing captured (likely threshold too high or you didn’t speak)
            return ""

        audio = np.concatenate(collected, axis=0)
        segments, _ = self.model.transcribe(audio, language="en")
        text = "".join(seg.text for seg in segments).strip()
        return text
