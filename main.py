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
    print("You:", text)

    if wake.detected(text):
        tts.speak("Hey, what's up?")
        # Next utterance is the command
        cmd = stt.record_utterance()
        print("Cmd:", cmd)
        if not cmd:
            continue
        reply = agent.ask(cmd)
        print("Nova:", reply)
        # If reply was a tool result, you can format it differently
        tts.speak(reply if isinstance(reply, str) else str(reply))