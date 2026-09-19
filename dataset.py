import json, random, torch
from torch.utils.data import Dataset

class ConversationDataset(Dataset):
    def __init__(self, records, tokenizer, context_length):
        self.items=[]; self.tok=tokenizer; self.n=context_length
        for r in records:
            text="<BOS> <"+r.get("language","en").upper()+"> "
            for turn in r["conversation"]: text += f"<{turn['role'].upper()}> {turn['text']} "
            ids=tokenizer.encode(text,add_eos=True)
            ids=ids[:context_length+1];
            if len(ids)>1: self.items.append(ids)
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        ids=self.items[i]; x=ids[:-1]; y=ids[1:]
        x=x+[self.tok.pad_id]*(self.n-len(x)); y=y+[-100]*(self.n-len(y))
        return torch.tensor(x),torch.tensor(y)

def load_records(path):
    with open(path,encoding="utf-8") as f:return json.load(f)
def split_records(records, fraction=.15, seed=1337):
    records=list(records); random.Random(seed).shuffle(records); cut=max(1,int(len(records)*fraction)); return records[cut:],records[:cut]
