# nova/agent.py

import json
import requests
from .config import settings

SYSTEM_PROMPT = open("data/prompts/system.txt", "r", encoding="utf-8").read()

class LLMOffline(Exception):
    pass

class LLM:
    def __init__(self):
        self.base = settings.ollama_base.rstrip("/")
        self.model = settings.ollama_model

    def _health(self):
        # quick ping; if Ollama isn't running, this will fail fast
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=1.5)
            r.raise_for_status()
        except Exception as e:
            raise LLMOffline(f"Ollama not reachable at {self.base}.") from e

    def chat(self, messages):
        self._health()
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
        try:
            content = self.llm.chat(self.history)
        except LLMOffline as e:
            # Return a short, user-facing string instead of crashing
            content = "LLM offline. Start Ollama and pull a model."
        except requests.RequestException as e:
            content = f"LLM error: {e.__class__.__name__}"

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

        self.history.append({"role": "assistant", "content": content})
        return content
