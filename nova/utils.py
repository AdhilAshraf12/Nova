import json 
from pathlib import Path

def load_json(path: str, default):
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(default, indent=2))
        return default
    return json.loads(p.read_text())

def save_json(path: str,data):
    p = Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2))