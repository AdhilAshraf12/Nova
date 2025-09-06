import json
import requests
from .config import settings


SYSTEM_PROMPT = open("data/prompts/system.txt", "r", encoding="utf-8").read()


class LLM:
    def __init__(self):
        self.base = settings.ollama_base
        self.model = settings.ollama_model


    def chat(self, messages):
        # Ollama /api/chat
        r = requests.post(
            f"{self.base}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"]
    
class Agent:
    def __init__(self, tools, memory):
        self.llm = LLM()
        self.tools = tools
        self.memory = memory
        self.system = {"role": "system", "content": SYSTEM_PROMPT}
        self.history = [self.system]


    def ask(self, user_text: str):
        self.history.append({"role": "user", "content": user_text})
        content = self.llm.chat(self.history)
        # Try tool-call convention
        try:
            maybe = json.loads(content)
            if isinstance(maybe, dict) and "tool" in maybe:
                res = self.tools.run(maybe["tool"], maybe.get("args", {}))
                tool_msg = f"Tool result: {res}"
                self.history.append({"role": "assistant", "content": tool_msg})
                return tool_msg
        except Exception:
            pass
        # Plain answer
        self.history.append({"role": "assistant", "content": content})
        return content