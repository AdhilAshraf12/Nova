# main.py  (diagnostic prints + robust loop)

import sys
import traceback

from nova.memory import Memory
from nova.tools import Tools
from nova.agent import Agent
from speech.tts import TTS
from speech.stt import STT
from speech.wakeword import Wakeword

def main():
    print("Nova starting… (say 'nova' to wake)", flush=True)

    try:
        print("[init] memory", flush=True)
        memory = Memory()
        print("[init] tools", flush=True)
        tools = Tools(memory)
        print("[init] agent", flush=True)
        agent = Agent(tools, memory)
        print("[init] stt", flush=True)
        stt = STT()
        print("[init] tts", flush=True)
        tts = TTS()
        print("[init] wake", flush=True)
        wake = Wakeword()
        print("[ok] Initialized all components", flush=True)
    except Exception as e:
        print("[fatal] init failed:", repr(e), flush=True)
        traceback.print_exc()
        sys.exit(1)

    try:
        while True:
            print("Listening…", flush=True)
            text = stt.record_utterance()
            print("You (raw):", repr(text), flush=True)
            if not text:
                continue

            if wake.detected(text):
                try:
                    tts.speak("Hey, what's up?")
                except Exception as e:
                    print("[warn] TTS speak failed:", repr(e), flush=True)

                cmd = stt.record_utterance()
                print("Cmd:", repr(cmd), flush=True)
                if not cmd:
                    continue

                reply = agent.ask(cmd)
                print("Nova:", reply, flush=True)
                try:
                    tts.speak(reply if isinstance(reply, str) else str(reply))
                except Exception as e:
                    print("[warn] TTS speak failed:", repr(e), flush=True)
    except KeyboardInterrupt:
        print("\n[exit] KeyboardInterrupt", flush=True)
    except Exception as e:
        print("[fatal] runtime error:", repr(e), flush=True)
        traceback.print_exc()

if __name__ == "__main__":
    main()
