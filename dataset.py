"""Conversation examples converted into overlapping language-model windows."""
import json
import random
import torch
from torch.utils.data import Dataset


class ConversationDataset(Dataset):
    def __init__(self, records, tokenizer, context_length, stride=None):
        self.items = []
        self.tok = tokenizer
        self.n = context_length
        stride = stride or max(1, context_length // 2)
        for record in records:
            text = "<BOS> <EN> "
            for turn in record["conversation"]:
                text += f"<{turn['role'].upper()}> {turn['text']} "
            ids = tokenizer.encode(text, add_eos=True)
            if len(ids) < 2:
                continue
            # Sliding windows create more useful examples than truncating each conversation.
            for start in range(0, max(1, len(ids) - 1), stride):
                window = ids[start:start + context_length + 1]
                if len(window) > 1:
                    self.items.append(window)
                if start + context_length + 1 >= len(ids):
                    break

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        ids = self.items[index]
        x, y = ids[:-1], ids[1:]
        x = x + [self.tok.pad_id] * (self.n - len(x))
        y = y + [-100] * (self.n - len(y))
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def load_records(path):
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return [r for r in records if r.get("language", "en").lower() == "en"]


def split_records(records, fraction=0.15, seed=1337):
    records = list(records)
    random.Random(seed).shuffle(records)
    # Keep a validation record when the dataset is very small.
    cut = max(1, int(len(records) * fraction))
    return records[cut:], records[:cut]
