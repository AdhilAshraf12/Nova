from nova.config import settings
from nova.memory import Memory
from nova.tools import Tools
from nova.agent import Agent
from speech.tts import TTS
from speech.stt import STT
from speech.wakeword import Wakeword

print("Nova starting… (say 'nova' to wake)")

memory = Memory()
tools = Tools(memory)
agent = Agent(tools, memory)
stt = STT()  # optionally STT(input_device_index=YOUR_MIC_INDEX)
tts = TTS()
wake = Wakeword()

while True:
    print("Listening…")
    text = stt.record_utterance()
    print(f"You (raw): {text!r}")  # shows empty vs actual

    if not text:
        # Nothing captured this round; loop again
        continue

    if wake.detected(text):
        # If the user said "nova <command>" in one sentence, try to use the same text
        after = text.lower().split(wake.key, 1)[-1].strip()
        if after:
            cmd = after
        else:
            tts.speak("Hey, what's up?")
            cmd = stt.record_utterance()
            print(f"Cmd: {cmd!r}")

        if not cmd:
            continue

        reply = agent.ask(cmd)
        print("Nova:", reply)
        tts.speak(reply if isinstance(reply, str) else str(reply))
    else:
        # Not a wake phrase—ignore or do passive actions here
        pass
