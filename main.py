# main.py

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

stt = STT()
tts = TTS()
wake = Wakeword()

while True:
    print("Listening…")
    text = stt.record_utterance()
    if not text:
        continue
    print("You (raw):", repr(text))

    if wake.detected(text):
        tts.speak("Hey, what's up?")
        cmd = stt.record_utterance()
        if not cmd:
            continue
        print("Cmd:", cmd)
        reply = agent.ask(cmd)
        print("Nova:", reply)
        tts.speak(reply if isinstance(reply, str) else str(reply))
