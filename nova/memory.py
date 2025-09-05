from .utils import load_json, save_json
from .config import settings

class Memory:
    def __init__(self):
        self.path = settings.memory_file
        self.state = load_json(self.path,{"facts": []})

    def add_fact(self,text: str):
        self.state["facts"].append(text)
        save_json(self.path,self.state)
    
    def search(self, query:str):
        #naive contains search
        return [f for f in self.state.get("facts",[]) if query.lower() in f.lower()]
    