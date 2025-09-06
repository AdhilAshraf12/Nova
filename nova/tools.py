import webbrowser
import subprocess
import platform
from typing import Any, Dict

class Tools:
    def __init__(self, memory):
        self.memory = memory

    def run(self, name: str, args: Dict[str, Any]):
        if name == "open_url":
            return self.open_url(args.get("url",""))
        if name == "remember":
            return self.remember(args.get("text",""))
        if name == "open_app":
            return self.open_app(args.get("path",""))
    
    def open_url(self, url:str):
        if not url:
            return {"ok": False, "error":"Missing url"}
        
    def remember(self, text:str):
        if not text:
            return {"ok": False, "error": "Missing text"}
        
        self.memory.add_fact(text)
        return {"ok": True, "saved": text}
    
    def open_app(self, path: str):
        if not path:
            return {"ok": False, "error": "Missing path"}
        try:
            if platform.system() == "Windows":
                subprocess.Popen([path], shell = True)

            else:
                subprocess.Popen([path])
            return {"ok": True, "launched": path}
        except Exception as e:
            return {"ok": False, "error": str(e)}
        