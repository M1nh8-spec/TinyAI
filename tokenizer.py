"""Project-built Unicode character tokenizer; no external vocabulary or weights."""
import json

SPECIAL = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<USER>", "<ASSISTANT>", "<VI>", "<EN>", "<TRANSLATE>"]
class CharTokenizer:
    def __init__(self, stoi=None):
        self.stoi = stoi or {s:i for i,s in enumerate(SPECIAL)}
        self.itos = {i:s for s,i in self.stoi.items()}
    @property
    def pad_id(self): return self.stoi["<PAD>"]
    @property
    def eos_id(self): return self.stoi["<EOS>"]
    def fit(self, texts):
        chars = sorted(set("".join(texts)))
        for ch in chars:
            if ch not in self.stoi:
                i=len(self.stoi); self.stoi[ch]=i; self.itos[i]=ch
        return self
    def encode(self, text, add_bos=False, add_eos=False):
        ids=[]
        if add_bos: ids.append(self.stoi["<BOS>"])
        # Special markers are kept as atomic tokens; all other Unicode is character-level.
        i=0
        markers=sorted(SPECIAL, key=len, reverse=True)
        while i < len(text):
            marker=next((m for m in markers if text.startswith(m,i)), None)
            if marker:
                ids.append(self.stoi[marker]); i += len(marker)
            else:
                ids.append(self.stoi.get(text[i], self.stoi["<UNK>"])); i += 1
        if add_eos: ids.append(self.stoi["<EOS>"])
        return ids
    def decode(self, ids, skip_special=False):
        specials=set(SPECIAL)
        return "".join(self.itos.get(int(i), "") for i in ids if not(skip_special and self.itos.get(int(i)) in specials))
    def save(self,path):
        with open(path,"w",encoding="utf-8") as f: json.dump(self.stoi,f,ensure_ascii=False,indent=2)
    @classmethod
    def load(cls,path):
        with open(path,encoding="utf-8") as f: return cls(json.load(f))
